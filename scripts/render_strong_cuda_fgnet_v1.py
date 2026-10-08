"""Render bound per-checkpoint ROC evidence; never pool seeds or claim a mechanism."""

import argparse
import json
import math
import re
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs
from scripts.run_oriented_campaign import paths_from, verified

EXPERIMENT = "adaface-fixed8-cuda-full-fgnet-roc-v2"
CELL = re.compile(r"(random|lookalike)_(head|tail|full)_lr(1e-06|1e-05)_s(42|1|2)\Z")
METRICS = (
    ("roc_auc", "ROC-AUC"),
    ("eer_interpolated", "Interpolated EER"),
    ("eer_discrete_minimax", "Discrete minimax EER"),
    ("tar@far=0.01", r"TAR@FAR=1\%"),
    ("tar@far=0.001", r"TAR@FAR=0.1\%"),
)


def numeric(value, low, high):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError("finite in-range metric required")
    return value


def metric_checked(value):
    numeric(value["point"], 0, 1)
    numeric(value["delta_vs_frozen"], -1, 1)
    for field, low, high in (("ci95", 0, 1), ("delta_ci95", -1, 1)):
        bounds = value[field]
        if len(bounds) != 2 or numeric(bounds[0], low, high) > numeric(bounds[1], low, high):
            raise ValueError("ordered two-endpoint metric interval required")


def collect(natives):
    """Validate already hash-verified native dictionaries; does not verify files."""
    cells, frozen, protocols = {}, {}, {}
    for native in natives:
        p, metrics = native["parameters"], native["metrics"]
        if (native["experiment"] != EXPERIMENT or metrics.get("execution_complete") is not True
                or metrics.get("publication_ready") is not False
                or p.get("device") != "cuda" or p.get("protocol_seed") != 42
                or p.get("tolerance") != 2 or p.get("n_boot") != 2000
                or p.get("bootstrap_seed") != 0):
            raise ValueError("completed bounded shared-protocol CUDA evidence required")
        names = p["actual_cells"]
        if len(names) != 1 or not CELL.fullmatch(names[0]) or names[0] in cells:
            raise ValueError("one unique canonical evaluated checkpoint per binding required")
        name = names[0]
        cell = {}
        for stratum, pairs, subjects in (("overall", 5308, 82), ("large_gap_25plus", 430, 75)):
            part = metrics[stratum]
            if (part["metric_version"] != "empirical-roc-v2"
                    or part["n_pairs"] != pairs or part["n_subjects"] != subjects
                    or part["actual_tuned_checkpoint_count"] != 1
                    or set(part["models"]) != {"frozen", name}):
                raise ValueError("full shared FG-NET pair/subject coverage required")
            baseline = part["models"]["frozen"]
            values = part["models"][name]
            for key, _ in METRICS:
                metric_checked(baseline[key])
                metric_checked(values[key])
                if not math.isclose(values[key]["point"] - baseline[key]["point"],
                                    values[key]["delta_vs_frozen"], abs_tol=1e-12):
                    raise ValueError("paired point delta inconsistent with frozen reference")
            if stratum in frozen and frozen[stratum] != baseline:
                raise ValueError("shared frozen reference differs across bindings")
            frozen[stratum] = baseline
            protocols[stratum] = dict(n_pairs=pairs, n_subjects=subjects)
            cell[stratum] = {key: values[key] for key, _ in METRICS}
        cells[name] = cell
    if not cells:
        raise ValueError("at least one completed evaluated checkpoint required")
    return dict(cells=cells, frozen=frozen, protocols=protocols,
                evaluated_checkpoint_count=len(cells), expected_matrix_cells=36,
                mechanism_complete=False, publication_ready=False,
                limitation="per-checkpoint conditional subject intervals; no joint seed-mean inference, "
                           "training-seed population inference, mechanism or independence clearance")


def table(payload, stratum):
    """A long-format table keeps every operating point and CI explicitly labelled."""
    rows = []
    for name, cell in sorted(payload["cells"].items()):
        arm, scope, lr, seed = CELL.fullmatch(name).groups()
        label = f"{arm}/{scope}/{lr}/s{seed}"
        for key, title in METRICS:
            value = cell[stratum][key]
            lo, hi = value["delta_ci95"]
            rows.append(f"{label} & {title} & {payload['frozen'][stratum][key]['point']:.4f} "
                        f"& {value['point']:.4f} & {value['delta_vs_frozen']:+.4f} "
                        f"& $[{lo:+.4f},{hi:+.4f}]$ " + r"\\")
    count = payload["evaluated_checkpoint_count"]
    label = "Overall" if stratum == "overall" else "25+ years"
    return "\n".join((
        r"\begin{table*}[!t]",
        rf"\caption{{{label}: {count} evaluated fixed8 AdaFace checkpoints from the 36-cell matrix. "
        "Entries and paired subject intervals are per checkpoint, not a seed mean or an ensemble. "
        "No training-seed population inference, mechanism claim or identity-independence clearance. "
        "Empirical thresholds are reselected, not deployment-calibrated; no multiplicity correction.}",
        rf"\label{{tab:strong-fixed8-{stratum.replace('_', '-')}}}",
        r"\centering\scriptsize", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
        r"Cell & Metric & Frozen & Tuned & Delta & Paired delta 95\% CI \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}",
    )) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh presentation root required")
    natives = [verified(p, EXPERIMENT) for p in args.bindings]
    inputs = sorted(set([
        Path(__file__), *args.bindings,
        PROJECT_ROOT / "scripts/run_oriented_campaign.py",
        PROJECT_ROOT / "scripts/run_restricted_matched_campaign.py",
        PROJECT_ROOT / "scripts/evaluate_oriented_cuda_v1.py",
        PROJECT_ROOT / "src/age_gap/common/manifest.py",
        *(p for m in natives for p in paths_from(m)),
    ]))
    before = [file_record(p) for p in inputs]
    payload = collect(natives)
    rendered = {f"{s}_table.tex": table(payload, s) for s in payload["protocols"]}
    if before != [file_record(p) for p in inputs]:
        raise RuntimeError("presentation ancestry changed")
    args.out.mkdir(parents=True)
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    outputs = [summary]
    for name, content in rendered.items():
        path = args.out / name
        path.write_text(content, encoding="utf-8")
        outputs.append(path)
    target = args.out / "presentation.manifest.json"
    write_experiment_manifest(target, experiment="strong-fixed8-per-checkpoint-roc-presentation-v1",
                              parameters=dict(seed=None, aggregation="none; per-checkpoint only"),
                              metrics=payload, inputs=inputs, outputs=outputs)
    validate_written_inputs(target, before)
    print(f"rendered {len(payload['cells'])} bound checkpoints; no mechanism claim", flush=True)


if __name__ == "__main__":
    main()
