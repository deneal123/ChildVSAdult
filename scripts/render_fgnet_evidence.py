"""Render corrected comparator and fixed-gallery retrieval evidence from manifests."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest


def verified_result(path: Path, root: Path) -> dict:
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if file_record(path) not in manifest.get("outputs", []):
        raise ValueError("result checksum is not bound to manifest")
    for kind in ("inputs", "outputs"):
        for record in manifest.get(kind, []):
            source = Path(record["path"])
            if not source.is_absolute():
                source = root / source
            if file_record(source) != record:
                raise ValueError(f"{kind} checksum mismatch: {source.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: float, *, signed: bool = False) -> str:
    if not math.isfinite(value):
        raise ValueError("nonfinite metric")
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def interval(values: list[float]) -> str:
    if len(values) != 2 or values[0] > values[1]:
        raise ValueError("invalid interval")
    return "$[" + ",".join(number(v, signed=True) for v in values) + "]$"


def table(kind: str, caption: str, columns: str, header: str, rows: list[str]) -> str:
    return "\n".join([
        f"% BEGIN GENERATED FGNET {kind}", r"\begin{table*}[!t]",
        r"\caption{" + caption + "}", r"\label{tab:fgnet-" + kind.lower() + "}",
        r"\centering\footnotesize", r"\begin{tabular}{@{}" + columns + "@{}}",
        r"\toprule", header + r" \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table*}",
        f"% END GENERATED FGNET {kind}",
    ]) + "\n"


def render_comparators(payload: dict) -> str:
    if payload["protocol"] != "endpoint_age_matched":
        raise ValueError("legacy protocol is not comparable")
    rows = []
    for stratum, label in (("overall", "Overall"), ("large_gap_25plus", "25+")):
        frozen = payload["frozen_common_backbone"]["metrics"][stratum]
        rows.append(f"{label} & Frozen R50 & {number(frozen['roc_auc'])} & "
                    f"{number(frozen['eer'])} & {number(frozen['tar@far=0.01'])} & -- & -- " + r"\\")
        for key, name in (("mtlface-common-protocol", "MTLFace"), ("cacon-common-protocol", "CACon")):
            method = payload["comparators"][key]
            runs = method["runs"]
            if len(runs) != 3 or {r["seed"] for r in runs} != {1, 2, 42}:
                raise ValueError("three distinct seeds required")
            if any(not r["provenance"]["manifest_checkpoint_match"] for r in runs):
                raise ValueError("unbound checkpoint")
            sample = next(r for r in runs if r["seed"] == 42)
            paired = sample["paired_vs_frozen"][stratum]
            if any(r["paired_vs_frozen"][stratum]["n_pairs"] != paired["n_pairs"]
                   or r["paired_vs_frozen"][stratum]["n_subjects"] != paired["n_subjects"] for r in runs):
                raise ValueError("seed sample mismatch")
            aggregate = method["aggregate_across_seeds"][stratum]
            cells = []
            for metric in ("roc_auc", "eer", "tar@far=0.01"):
                values = aggregate[metric]
                if values["n_seeds"] != 3:
                    raise ValueError("incomplete aggregate")
                cells.append(number(values["mean"]) + r"$\pm$" + number(values["std"]))
            loso = sample["leave_one_subject_out_vs_frozen"][stratum]
            cells.extend([interval(paired["delta_ci95"]),
                          interval([loso["minimum_delta_auc"], loso["maximum_delta_auc"]])])
            rows.append(" & ".join([label, name, *cells]) + r" \\")
    return table("COMPARATORS", "Corrected endpoint-age-matched FG-NET, common ArcFace-R50/CASIA "
                 "backbone and eight-epoch requested budget. AUC/EER/TAR are means$\\pm$SD across "
                 "training seeds. Gain CI and leave-one-subject-out (LOSO) gain range are for seed 42 "
                 "versus frozen, not intervals of seed means. Resampling accounts for both pair endpoints; "
                 "all seeds reuse the same subjects. Low-FAR TAR is a descriptive ROC operating point, "
                 "not a dev-calibrated deployment threshold. Train--benchmark identity independence "
                 "remains unverified. These are controlled reimplementations, not leaderboard reproductions.",
                 "llccccc", "Stratum & Model & AUC & EER & TAR@FAR=1\\% & Gain 95\\% CI (s42) & LOSO range (s42)", rows)


def render_retrieval(payload: dict) -> str:
    if payload["primary_split"] != "test" or payload["training_identity_independence"] != "unverified":
        raise ValueError("unexpected retrieval scope; review caption before reuse")
    rows = []
    for stratum, label in (("overall", "Overall"), ("gap_25_plus", "25+")):
        reference = payload["results"]["tuned_seed42"]["test"]
        baseline = reference["frozen"][stratum]
        rows.append(" & ".join([label, "Frozen", str(int(baseline["n_queries"])),
                    *[number(baseline[k]) for k in ("recall@1", "recall@5", "recall@10", "mrr")], "--"]) + r" \\")
        for seed in (1, 2, 42):
            result = payload["results"][f"tuned_seed{seed}"]["test"]
            if result["frozen"][stratum] != baseline:
                raise ValueError("inconsistent frozen reference")
            tuned = result["tuned"][stratum]
            paired = result["paired_bootstrap"][stratum]
            if tuned["n_queries"] != baseline["n_queries"] or paired["n_queries"] != baseline["n_queries"]:
                raise ValueError("query count mismatch")
            rows.append(" & ".join([label, f"Tuned s{seed}", str(int(tuned["n_queries"])),
                        *[number(tuned[k]) for k in ("recall@1", "recall@5", "recall@10", "mrr")],
                        interval(paired["delta_ci95"]["recall@1"])]) + r" \\")
    return table("RETRIEVAL", "FaceNet fixed-gallery FG-NET retrieval. Per-seed results and paired "
                 "query-subject bootstrap intervals for Recall@1 gain; gallery is fixed, not resampled. "
                 "The 25+ Recall@1 gain intervals include zero. Identities are shared across gallery/query "
                 "as required for genuine retrieval, but images are disjoint. Dev/test query identities "
                 "were split before scoring. Legacy training manifests are absent and mined-train overlap "
                 "is unresolved: these are bounded small-gallery results, not evidence of independent "
                 "large-scale archival search.", "llccccc c", "Stratum & Model & Queries & Recall@1 & Recall@5 & Recall@10 & MRR & R@1 gain 95\\% CI", rows)


def render_errors(payload: dict) -> str:
    if payload["threshold_provenance"] != "development negatives only; frozen on test":
        raise ValueError("threshold is not dev-only")
    if payload["acceptance_convention"] != "strict: accepted iff cosine > threshold":
        raise ValueError("unknown acceptance convention")
    if payload["training_identity_independence"] != "unverified":
        raise ValueError("unexpected independence claim; review caption")
    rows = []
    names = (("frozen", "Frozen"), ("tuned_seed1", "Tuned s1"),
             ("tuned_seed2", "Tuned s2"), ("tuned_seed42", "Tuned s42"))
    for key, label in (("overall", "Overall"), ("gap_25_plus", "25+"),
                       ("child_lt13_to_adult_gt25", "Child-to-adult"),
                       ("blur_bin_0", "Blur: low sharpness"),
                       ("blur_bin_1", "Blur: middle"), ("blur_bin_2", "Blur: high sharpness")):
        for model, name in names:
            if key.startswith("blur_") and model not in ("frozen", "tuned_seed42"):
                continue
            op = payload["splits"][model]["test"]["operating_points"]["0.01"]
            if op["dev_fmr"] > .01 + 1e-12:
                raise ValueError("dev FMR exceeds calibration target")
            sample = op["strata"][key]
            if not sample["conditional_on_dev_threshold"] or sample["threshold_selection_variance_included"]:
                raise ValueError("unexpected interval estimand")
            if sample["n_pos"] <= 0 or sample["n_neg"] <= 0:
                raise ValueError("empty publication stratum")
            for metric, event, total in (("fmr", "false_accepts", "n_neg"),
                                         ("fnmr", "false_rejects", "n_pos")):
                if not math.isclose(sample[metric], sample[event] / sample[total], abs_tol=1e-12):
                    raise ValueError("rate/count mismatch")
                if sample[event] == 0 and sample[f"{metric}_ci95"] is not None:
                    raise ValueError("zero-event interval must be unavailable")
            ci = "--" if sample["fnmr_ci95"] is None else interval(sample["fnmr_ci95"])
            rows.append(" & ".join([label, name, f"{sample['false_accepts']}/{sample['n_neg']}",
                        f"{sample['false_rejects']}/{sample['n_pos']}", number(sample["fmr"]),
                        number(sample["fnmr"]), ci]) + r" \\")
    return table("ERRORS", "FG-NET test errors at separately dev-calibrated per-model "
                 "nominal FMR=1\\% thresholds, strict score$>$threshold. FA/FR are false accepts/rejects; "
                 "the observed test FMR need not equal the nominal target. FNMR intervals are "
                 "both-endpoint subject-bootstrap percentile intervals conditional on the fixed dev "
                 "threshold; calibration uncertainty is excluded. Blur cutpoints are dev-only, pair "
                 "sharpness is the lesser endpoint Laplacian variance. Zero-event empirical intervals "
                 "are not reliable risk bounds. Full exact-year, blur and per-seed results, including "
                 "the under-resolved nominal 0.1\\% point, are in the aggregate artifact; unlabelled "
                 "pose/scan/collage stay unknown. Training independence is unverified.", "llccccc",
                 "Stratum & Model & FA / negatives & FR / positives & FMR & FNMR & FNMR 95\\% CI", rows)


def cmc_curves(payload: dict) -> dict[str, list[float]]:
    render_retrieval(payload)  # apply the same scope/sample checks as the table
    curves = {"Frozen": payload["results"]["frozen_reference"]["cmc_test"]}
    for seed in (1, 2, 42):
        result = payload["results"][f"tuned_seed{seed}"]
        if result["cmc_test_frozen"] != curves["Frozen"]:
            raise ValueError("CMC frozen reference mismatch")
        curves[f"Tuned s{seed}"] = result["cmc_test_tuned"]
    length = len(curves["Frozen"])
    for name, curve in curves.items():
        if length < 10 or len(curve) != length:
            raise ValueError("inconsistent CMC rank coverage")
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in curve):
            raise ValueError("invalid CMC values")
        if any(a > b for a, b in zip(curve, curve[1:], strict=False)):
            raise ValueError("nonmonotone CMC")
        test = payload["results"]["tuned_seed42" if name == "Frozen" else f"tuned_seed{name[7:]}"]["test"]
        row = test["frozen" if name == "Frozen" else "tuned"]["overall"]
        if any(not math.isclose(curve[k - 1], row[f"recall@{k}"], abs_tol=1e-12) for k in (1, 5, 10)):
            raise ValueError("CMC and Recall@K differ")
    return curves


def plot_retrieval(payload: dict, source: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = cmc_curves(payload)
    output = PROJECT_ROOT / "latex/shared/figures/fig_fgnet_retrieval_cmc.pdf"
    fig, ax = plt.subplots(figsize=(4.7, 3.0), layout="constrained")
    for name, curve in curves.items():
        ax.plot(range(1, len(curve) + 1), curve, label=name, linewidth=1.4)
    ax.set(xlabel="Gallery rank K", ylabel="Recall@K (test queries)",
           xlim=(1, len(curves["Frozen"])), ylim=(0, 1.02))
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    write_experiment_manifest(
        source.parent / "retrieval_presentation.manifest.json", experiment="fgnet-fixed-gallery-cmc",
        parameters={"protocol_hash": payload["protocol_hash"], "scope": "fixed gallery; overlap unresolved"},
        metrics={"gallery_size": payload["source"]["n_identities"],
                 "reported_max_rank": len(curves["Frozen"]), "curves": curves},
        inputs=[source, source.with_suffix(".manifest.json"), Path(__file__)], outputs=[output],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("comparators", "retrieval", "errors"))
    parser.add_argument("--check", type=Path)
    parser.add_argument("--plot", action="store_true", help="render retrieval CMC with artifact manifest")
    args = parser.parse_args()
    filename = {"comparators": "metrics/comparator_fgnet_endpoint_age_matched.json",
                "retrieval": "metrics/fgnet_retrieval_20261002/fgnet_retrieval_study.json",
                "errors": "metrics/fgnet_error_breakdown/fgnet_error_breakdown.json"}[args.kind]
    payload = verified_result(PROJECT_ROOT / filename, PROJECT_ROOT)
    if args.plot:
        if args.kind != "retrieval":
            raise SystemExit("--plot is supported only for retrieval")
        plot_retrieval(payload, PROJECT_ROOT / filename)
    rendered = {"comparators": render_comparators, "retrieval": render_retrieval,
                "errors": render_errors}[args.kind](payload)
    if args.check:
        content = args.check.read_text(encoding="utf-8")
        begin, end = f"% BEGIN GENERATED FGNET {args.kind.upper()}", f"% END GENERATED FGNET {args.kind.upper()}"
        if content.count(begin) != 1 or content.count(end) != 1:
            raise SystemExit("missing/duplicate generated block")
        if content[content.index(begin):content.index(end) + len(end)] + "\n" != rendered:
            raise SystemExit("stale generated evidence table")
        print("Table matches checksum-verified artifacts; identity independence remains unresolved.")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
