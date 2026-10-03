"""Source-bound CACD-VS table/summary, separate from historical operating points."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.cacd_metrics_v2 import ALL_METRICS
from scripts.reevaluate_cacd_serial import KEYS, read_plan, verify
from scripts.render_fgnet_evidence import interval, number

LABELS = (
    "ROC-AUC",
    "Interpolated EER",
    "Discrete minimax EER",
    r"TAR@FAR=1\%",
    r"TAR@FAR=0.1\%",
    "LOFO fixed-grid accuracy",
)


def validate(payload):
    if (
        payload.get("execution_complete") is not True
        or payload.get("publication_ready") is not False
    ):
        raise ValueError("completed unresolved-publication result required")
    data = payload["inference"]
    if (
        data["metric_version"] != "empirical-roc-v2"
        or (data["n_pairs"], data["n_positive"], data["n_negative"], data["n_folds"])
        != (4000, 2000, 2000, 10)
        or set(data["models"]) != set(KEYS)
        or set(data["three_checkpoint_aggregate"]) != set(ALL_METRICS)
        or data["training_identity_independence"] != "unverified"
        or data["publication_ready"] is not False
    ):
        raise ValueError("full four-checkpoint pair-level CACD protocol required")
    boot = data["bootstrap"]
    if (
        boot["sampling_unit"] != "pair, stratified within each protocol fold and class"
        or boot["subject_metadata_available"] is not False
        or boot["shared_draws_across_models"] is not True
        or type(boot["requested"]) is not int
        or type(boot["valid"]) is not int
        or not 2 <= boot["valid"] <= boot["requested"]
        or boot["conditioning"]
        != "fixed checkpoints and protocol; not training-seed population or ensemble AUC"
        or data["accuracy"]["threshold_selection"]
        != "other protocol folds only; reselected per draw"
    ):
        raise ValueError("conditional shared pair bootstrap and fold-trained thresholds required")

    def scalar(v, *, signed=False):
        if (
            type(v) not in (int, float)
            or not math.isfinite(v)
            or not (-1 if signed else 0) <= v <= 1
        ):
            raise ValueError("finite bounded scalar metric required")

    def ci(values):
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError("two-ended interval required")
        for v in values:
            scalar(v, signed=True)
        if values[0] > values[1]:
            raise ValueError("ordered interval required")

    for metric in ALL_METRICS:
        frozen = data["models"]["frozen"][metric]["point"]
        values = []
        for role in KEYS:
            row = data["models"][role][metric]
            scalar(row["point"])
            scalar(row["delta_vs_frozen"], signed=True)
            ci(row["pair_ci95"])
            ci(row["paired_pair_delta_ci95"])
            if not math.isclose(
                row["delta_vs_frozen"], row["point"] - frozen, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("checkpoint delta inconsistent")
            if role != "frozen":
                values.append(row["point"])
        aggregate = data["three_checkpoint_aggregate"][metric]
        scalar(aggregate["mean"])
        scalar(aggregate["sd"])
        scalar(aggregate["mean_checkpoint_delta"], signed=True)
        ci(aggregate["pair_mean_ci95"])
        ci(aggregate["paired_pair_delta_ci95"])
        if (
            not math.isclose(aggregate["mean"], statistics.mean(values), rel_tol=0, abs_tol=1e-12)
            or not math.isclose(aggregate["sd"], statistics.stdev(values), rel_tol=0, abs_tol=1e-12)
            or not math.isclose(
                aggregate["mean_checkpoint_delta"],
                aggregate["mean"] - frozen,
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("fixed-checkpoint mean/SD/delta inconsistent")
    return data


def table(payload):
    data = validate(payload)
    rows = []
    for metric, label in zip(ALL_METRICS, LABELS, strict=True):
        aggregate = data["three_checkpoint_aggregate"][metric]
        rows.append(
            " & ".join(
                [
                    label,
                    number(data["models"]["frozen"][metric]["point"]),
                    number(aggregate["mean"]) + r"$\pm$" + number(aggregate["sd"]),
                    number(aggregate["mean_checkpoint_delta"], signed=True),
                    interval(aggregate["paired_pair_delta_ci95"]),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            "% BEGIN GENERATED CACD SERIAL ROC V2",
            r"\begin{table*}[!t]",
            r"\caption{Source-bound CACD-VS serial ROC-v2 evaluation: 4{,}000 official pairs, ten folds. "
            "Mean/SD describe three fixed legacy training checkpoints, not an ensemble or training-seed population. "
            "Paired mean-change intervals resample pairs jointly across checkpoints within each fold/class, "
            "conditional on these weights; reused subjects/photos are not accounted for without person metadata. "
            "Accuracy selects a fixed-grid threshold on the other nine folds, reselected per draw. "
            "ROC operating points are test-derived, not deployment-calibrated. Negative EER change is improvement. "
            "Historical CACD operating points are not pooled.}",
            r"\label{tab:cacd-serial-roc-v2}",
            r"\centering\footnotesize",
            r"\begin{tabular}{@{}lcccc@{}}",
            r"\toprule",
            r"Metric & Frozen & Tuned mean$\pm$SD & Tuned$-$frozen & Paired pair 95\% CI \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "% END GENERATED CACD SERIAL ROC V2",
            "",
        ]
    )


def summary(payload):
    data = validate(payload)

    def transition(metric):
        return (
            number(data["models"]["frozen"][metric]["point"])
            + r"$\to$"
            + number(data["three_checkpoint_aggregate"][metric]["mean"])
        )

    # This versioned summary requires the observed trade-off rather than inserting a generic claim.
    expected_signs = {
        "eer_interpolated": 1,
        "tar@far=0.01": -1,
        "accuracy_leave_one_fold_out_fixed_grid": -1,
    }
    for metric, sign in expected_signs.items():
        ci = data["three_checkpoint_aggregate"][metric]["paired_pair_delta_ci95"]
        if not (min(ci) > 0 if sign == 1 else max(ci) < 0):
            raise ValueError("operating-point narrative no longer supported by paired CI")
    for metric in ("roc_auc", "tar@far=0.001"):
        ci = data["three_checkpoint_aggregate"][metric]["paired_pair_delta_ci95"]
        if not ci[0] <= 0 <= ci[1]:
            raise ValueError("no-zero-exclusion narrative no longer supported")
    return (
        "% BEGIN GENERATED CACD SERIAL SUMMARY\n"
        "Source-bound CACD-VS ROC-AUC changes " + transition("roc_auc") + ". "
        "The three-checkpoint mean lowers 10-fold fixed-grid accuracy "
        + transition("accuracy_leave_one_fold_out_fixed_grid")
        + ", raises EER "
        + transition("eer_interpolated")
        + ", and lowers TAR@FAR1\\% "
        + transition("tar@far=0.01")
        + ". Their paired mean-change intervals exclude zero; AUC and TAR@FAR0.1\\% intervals include zero. "
        "These are conditional pair-level intervals, not subject or training-seed-population uncertainty; "
        "the supplement's CACD-VS serial ROC-v2 section reports all metrics.\n"
        "% END GENERATED CACD SERIAL SUMMARY\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("fresh presentation directory required")
    base = PROJECT_ROOT / "metrics/cacd_serial_20261003"
    read_plan(base)
    paths = []
    kinds = {
        "plan": "cacd-vs-serial-plan",
        "prepared": "cacd-vs-mapped-crops",
        **{k: "cacd-vs-serial-checkpoint" for k in KEYS},
        "summary": "cacd-vs-local-serial-4checkpoint-roc-v2",
    }
    for name, kind in kinds.items():
        manifest = base / f"{name}.manifest.json"
        native = verify(manifest, kind)
        if native["metrics"] != json.loads((base / f"{name}.json").read_text(encoding="utf-8")):
            raise ValueError("native/JSON phase metrics mismatch")
        paths.extend([manifest, base / f"{name}.json"])
        for record in native["inputs"] + native["outputs"]:
            path = Path(record["path"])
            paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
    inputs = list(
        dict.fromkeys(
            p.resolve()
            for p in [*paths, Path(__file__), PROJECT_ROOT / "scripts/render_fgnet_evidence.py"]
        )
    )
    before = [file_record(p) for p in inputs]
    payload = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    rendered, paragraph = table(payload), summary(payload)
    if before != [file_record(p) for p in inputs]:
        raise ValueError("presentation inputs changed")
    out.mkdir(parents=True)
    table_path, summary_path = out / "cacd_table.tex", out / "cacd_main_summary.tex"
    table_path.write_text(rendered, encoding="utf-8")
    summary_path.write_text(paragraph, encoding="utf-8")
    target = out / "presentation.manifest.json"
    write_experiment_manifest(
        target,
        experiment="cacd-serial-roc-v2-presentation",
        parameters={"fraction_precision": 4, "main_and_supplement": True},
        metrics={
            "pair_level_only": True,
            "legacy_results_pooled": False,
            "publication_ready": False,
        },
        inputs=inputs,
        outputs=[table_path, summary_path],
    )
    if before != json.loads(target.read_text(encoding="utf-8"))["inputs"]:
        target.unlink()
        raise ValueError("presentation inputs changed during write; marker withdrawn")
    print("Source-bound CACD table and operating-point summary generated; pair-level only")


if __name__ == "__main__":
    main()
