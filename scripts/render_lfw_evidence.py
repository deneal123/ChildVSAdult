"""Generate a separate source-bound LFW table; never relabel historical results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.render_fgnet_evidence import interval, number, verified_result

METRICS = (
    ("accuracy_official_folds", "Official-fold accuracy"),
    ("roc_auc", "ROC-AUC"),
    ("eer", "Discrete ROC EER"),
    ("tar@far=0.01", r"TAR@FAR=1\%"),
    ("tar@far=0.001", r"TAR@FAR=0.1\%"),
)


def render(payload):
    if payload["protocol"] != "official_lfw_view2_contiguous_folds_source_bound":
        raise ValueError("source-bound official-fold protocol required")
    if payload["cache_full_transform_replay"] is not True or payload["seeds"] != [42, 1, 2]:
        raise ValueError("replayed cache and declared three checkpoints required")
    if (payload["n_pairs"], payload["n_positive"], payload["n_negative"]) != (6000, 3000, 3000):
        raise ValueError("full balanced View-2 protocol required")
    for model in payload["models"].values():
        if [row["fold"] for row in model["official_folds"]] != list(range(10)):
            raise ValueError("ten contiguous official folds required")
    bootstrap = payload["bootstrap"]
    if (bootstrap["sampling_unit"] != "person" or not bootstrap["accuracy_threshold_reselection"]
            or bootstrap["includes_training_seed_population_uncertainty"]):
        raise ValueError("conditional person bootstrap with accuracy reselection required")
    if min(bootstrap["n_valid_roc"], bootstrap["n_valid_accuracy"]) < 2:
        raise ValueError("insufficient valid resamples")
    rows = []
    for key, label in METRICS:
        aggregate = payload["seed_aggregate"][key]
        if aggregate["n_seeds"] != 3:
            raise ValueError("three checkpoint summaries required")
        rows.append(" & ".join([
            label, number(payload["models"]["frozen"]["metrics"][key]),
            number(aggregate["mean"]) + r"$\pm$" + number(aggregate["std"]),
            number(aggregate["delta_mean_vs_frozen"], signed=True),
            interval(aggregate["fixed_checkpoint_mean_gain_subject_ci95"]),
        ]) + r" \\")
    return "\n".join([
        "% BEGIN GENERATED LFW BOUND", r"\begin{table*}[!t]",
        r"\caption{Separate source-bound FaceNet LFW evaluation: "
        + f"{payload['n_pairs']:,}".replace(",", r"{,}") + " official View-2 pairs, "
        + f"{payload['n_subjects']:,}".replace(",", r"{,}")
        + " persons. Mean and SD describe three fixed legacy training checkpoints; gain CI resamples "
        "persons jointly for all checkpoints, conditional on these weights and folds, not a population of training seeds. "
        "Accuracy thresholds are selected on the other nine folds and reselected in each draw. "
        "ROC operating points are test-derived, not development-calibrated deployment thresholds; EER uses the discrete minimax ROC point. "
        "EER improvement has negative sign. Legacy interleaved-cache results are not pooled.}",
        r"\label{tab:lfw-bound}", r"\centering\footnotesize",
        r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
        r"Metric & Frozen & Tuned mean$\pm$SD & Tuned$-$frozen & Paired gain 95\% CI \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}",
        "% END GENERATED LFW BOUND",
    ]) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result", type=Path, default=PROJECT_ROOT / "metrics/lfw_bound_evaluation_20261002/lfw_bound_evaluation.json")
    args = p.parse_args()
    manifest = args.result.with_suffix(".manifest.json")
    inputs = [args.result, manifest, Path(__file__), PROJECT_ROOT / "scripts/render_fgnet_evidence.py"]
    before = [file_record(path) for path in inputs]
    table = render(verified_result(args.result, PROJECT_ROOT))
    if before != [file_record(path) for path in inputs]:
        raise ValueError("presentation inputs changed")
    target = args.result.parent / "lfw_table.tex"
    target.write_text(table, encoding="utf-8")
    presentation = write_experiment_manifest(args.result.parent / "lfw_presentation.manifest.json",
        experiment="lfw-source-bound-table-presentation", parameters={"table_label": "tab:lfw-bound"},
        inputs=inputs, outputs=[target], metrics={"legacy_results_pooled": False})
    if json.loads(presentation.read_text(encoding="utf-8"))["inputs"] != before:
        presentation.unlink()
        raise ValueError("inputs changed while writing presentation manifest")
    print(table)


if __name__ == "__main__":
    main()
