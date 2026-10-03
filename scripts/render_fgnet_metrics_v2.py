"""Render global FG-NET ROC-v2 headline and operating-point evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.render_fgnet_evidence import interval, number, verified_result


def checked(payload):
    if payload["original_error_protocol_reused"] or payload["image_inference_performed"]:
        raise ValueError("global reconstructed cached protocol required")
    if payload["publication_ready"] or payload["cached_images"] != 650:
        raise ValueError("bounded cached-image coverage required")
    for key in ("overall", "large_gap_25plus"):
        part = payload[key]
        boot = part["bootstrap"]
        if (part["metric_version"] != "empirical-roc-v2" or not boot["shared_draws_across_models"]
                or not boot["thresholds_reselected"] or boot["valid"] < 2
                or part["n_positive"] != part["n_negative"]
                or part["training_identity_independence"] != "unverified"):
            raise ValueError("joint fixed-checkpoint person inference with explicit limits required")


def headline(payload):
    checked(payload)
    rows = []
    for key, name in (("large_gap_25plus", "25+ years (primary)"), ("overall", "Overall")):
        part = payload[key]
        value = part["three_checkpoint_aggregate"]["roc_auc"]
        rows.append(" & ".join((name, f"{part['n_pairs']} / {part['n_subjects']}",
            number(part["models"]["frozen"]["roc_auc"]["point"]),
            number(value["mean"]) + r" $\pm$ " + number(value["sd"]),
            "$" + number(value["mean_checkpoint_delta"], signed=True) + r"\pm" + number(value["sd"]) + "$",
            interval(value["delta_ci95"]))) + r" \\")
    return "\n".join(("% BEGIN GENERATED FGNET ENDPOINT TABLE", r"\begin{table*}[!t]",
        r"\caption{Source-bound FG-NET endpoint-age-matched evaluation (FaceNet), reconstructed from all 650 cached images. "
        "Means and SD cover three fixed training checkpoints. Gain CI uses shared person resamples of the mean checkpoint effect, "
        "not ensemble scores or the training-seed population. Legacy training provenance and train--benchmark identity independence remain unverified.}",
        r"\label{tab:endpoint}", r"\centering\footnotesize", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
        r"Stratum & Pairs / subjects & Frozen AUC & Tuned AUC & Gain & Paired mean gain 95\% CI \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}",
        "% END GENERATED FGNET ENDPOINT TABLE")) + "\n"


def operating(payload):
    checked(payload)
    labels = (("roc_auc", "ROC-AUC"), ("eer_interpolated", "Interpolated EER"),
        ("eer_discrete_minimax", "Discrete minimax EER"),
        ("tar@far=0.01", r"TAR@FAR=1\%"), ("tar@far=0.001", r"TAR@FAR=0.1\%"))
    rows = []
    for key, name in (("overall", "Overall"), ("large_gap_25plus", "25+ years")):
        part = payload[key]
        for metric, label in labels:
            values = part["three_checkpoint_aggregate"][metric]
            rows.append(" & ".join((name, label, number(part["models"]["frozen"][metric]["point"]),
                number(values["mean"]) + r"$\pm$" + number(values["sd"]),
                number(values["mean_checkpoint_delta"], signed=True), interval(values["delta_ci95"]))) + r" \\")
    return "\n".join(("% BEGIN GENERATED FGNET ROC V2", r"\begin{table*}[!t]",
        r"\caption{Global endpoint-age-matched FG-NET ROC-v2: overall 2654 positive/negative pairs and 82 subjects; "
        "25+ stratum 215 positive/negative pairs and 75 subjects. Shared person bootstrap across frozen and three tuned checkpoints; "
        "positive weight is one owner multiplicity, negative weight the product of both endpoints. "
        "Tuned entries are mean and SD of checkpoint metrics; paired delta CI is conditional on these fixed weights, not a training-seed population interval. "
        "TAR maximizes attainable empirical ROC at FAR no greater than target, accepting tied scores together. "
        "Thresholds are reselected per draw, not calibrated for deployment. Interpolated and discrete minimax EER are separate definitions; "
        "negative EER deltas favor tuning. Intervals are not multiplicity-corrected; training independence remains unverified.}",
        r"\label{tab:fgnet-roc-v2}", r"\centering\footnotesize", r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
        r"Stratum & Metric & Frozen & Tuned mean$\pm$SD & Mean delta & Paired delta 95\% CI \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}", "% END GENERATED FGNET ROC V2")) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=PROJECT_ROOT / "metrics/fgnet_metrics_v2_20261003/summary.json")
    args = parser.parse_args()
    paths = [args.result, args.result.with_suffix(".manifest.json"), Path(__file__), PROJECT_ROOT / "scripts/render_fgnet_evidence.py"]
    before = [file_record(path) for path in paths]
    payload = verified_result(args.result, PROJECT_ROOT)
    outputs = [args.result.parent / name for name in ("headline_table.tex", "operating_table.tex")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("fresh presentation outputs required")
    texts = [headline(payload), operating(payload)]
    if before != [file_record(path) for path in paths]:
        raise ValueError("render inputs changed")
    for path, content in zip(outputs, texts, strict=True):
        path.write_text(content, encoding="utf-8")
    target = args.result.parent / "presentation.manifest.json"
    write_experiment_manifest(target, experiment="fgnet-roc-v2-presentation", parameters={"seed": None},
        metrics={"publication_ready": False}, inputs=paths, outputs=outputs)
    if json.loads(target.read_text(encoding="utf-8"))["inputs"] != before:
        target.unlink()
        raise ValueError("render inputs changed while writing manifest")
    print("Generated source-bound headline and operating-point tables")


if __name__ == "__main__":
    main()
