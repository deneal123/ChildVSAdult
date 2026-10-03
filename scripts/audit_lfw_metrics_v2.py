"""Source-bound saved-score LFW ROC-v2 audit; official-fold accuracy is unchanged."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.benchmark_metrics_v2 import KEYS, infer
from scripts.render_fgnet_evidence import verified_result

OLD_TO_NEW = {"roc_auc": "roc_auc", "eer": "eer_discrete_minimax",
              "tar@far=0.01": "tar@far=0.01", "tar@far=0.001": "tar@far=0.001"}


def compare(previous, updated):
    """Compare points, individual CIs, paired CIs and mean-effect CIs, not only AUC."""
    errors = []
    for old_metric, metric in OLD_TO_NEW.items():
        for key in KEYS:
            old, new = previous["models"][key], updated["models"][key][metric]
            errors.append(abs(old["metrics"][old_metric] - new["point"]))
            errors.extend(np.abs(np.asarray(old["subject_ci95"][old_metric]) - new["ci95"]).tolist())
            if key != "frozen":
                gain = old["paired_gain"][old_metric]
                errors.append(abs(gain["delta"] - new["delta_vs_frozen"]))
                errors.extend(np.abs(np.asarray(gain["subject_ci95"]) - new["delta_ci95"]).tolist())
        old, new = previous["seed_aggregate"][old_metric], updated["three_checkpoint_aggregate"][metric]
        errors.extend([abs(old["mean"] - new["mean"]), abs(old["std"] - new["sd"]),
                       abs(old["delta_mean_vs_frozen"] - new["mean_checkpoint_delta"])])
        errors.extend(np.abs(np.asarray(old["fixed_checkpoint_mean_gain_subject_ci95"]) - new["delta_ci95"]).tolist())
    return {"comparisons": len(errors), "max_absolute_difference": float(max(errors)),
            "matches_within_1e_minus_12": bool(max(errors) <= 1e-12),
            "metric_mapping": OLD_TO_NEW,
            "scope": "point and CI compatibility on actual saved scores and recorded subject resamples; not universal implementation equivalence"}


def aligned_endpoints(rows, labels, folds):
    if len(rows) != len(labels):
        raise ValueError("official person rows do not cover saved scores")
    if any(row["index"] != index or row["label"] != int(labels[index]) or row["fold"] != int(folds[index])
           for index, row in enumerate(rows)):
        raise ValueError("official person/label/fold ordering mismatch")
    return [row["subject_a"] for row in rows], [row["subject_b"] for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh audit output required")
    root = PROJECT_ROOT
    old_path = root / "metrics/lfw_bound_evaluation_20261002/lfw_bound_evaluation.json"
    previous = verified_result(old_path, root)
    old_manifest = old_path.with_suffix(".manifest.json")
    native = json.loads(old_manifest.read_text(encoding="utf-8"))
    scores_path = old_path.parent / "private/scores.npz"
    rows_path = root / "metrics/lfw_bound_cache_20261002/private/pair_rows.jsonl"
    if file_record(scores_path) not in native["outputs"] or file_record(rows_path) not in native["inputs"]:
        raise ValueError("scores/person protocol not bound to original evaluation")
    paths = [old_path, old_manifest, scores_path, rows_path, Path(__file__),
        root / "scripts/benchmark_metrics_v2.py", root / "scripts/verification_metrics_v2.py",
        root / "scripts/render_fgnet_evidence.py"]
    before = [file_record(p) for p in paths]
    with np.load(scores_path, allow_pickle=False) as data:
        if set(data.files) != {*KEYS, "labels", "folds"}:
            raise ValueError("four fixed checkpoints with original labels/folds required")
        scores = {key: data[key] for key in KEYS}
        labels, folds = data["labels"], data["folds"]
    if folds.dtype.kind not in "iu" or labels.dtype.kind not in "iu":
        raise ValueError("genuine integer protocol metadata required")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    a, b = aligned_endpoints(rows, labels, folds)
    bootstrap = previous["bootstrap"]
    updated = infer(scores, labels, a, b, n_boot=bootstrap["n_requested"], seed=bootstrap["seed"])
    if updated["bootstrap"]["valid"] != bootstrap["n_valid_roc"]:
        raise ValueError("subject draw eligibility changed")
    compatibility = compare(previous, updated)
    result = {"roc_v2": updated, "compatibility": compatibility,
        "official_fold_accuracy_recomputed": False,
        "accuracy_scope": "original official-fold results remain unchanged and separate; this audit covers ROC only",
        "image_inference_performed": False, "legacy_results_rewritten": False,
        "training_identity_independence": "unverified", "publication_ready": False}
    if before != [file_record(p) for p in paths]:
        raise ValueError("inputs changed during audit")
    args.out.mkdir(parents=True)
    output = args.out / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = output.with_suffix(".manifest.json")
    write_experiment_manifest(target, experiment="lfw-saved-score-roc-v2-compatibility",
        parameters={"seed": bootstrap["seed"], "n_boot": bootstrap["n_requested"], "image_inference": False},
        metrics=result, inputs=paths, outputs=[output])
    if json.loads(target.read_text(encoding="utf-8"))["inputs"] != before:
        target.unlink()
        raise ValueError("inputs changed while writing completed marker")
    print(json.dumps(compatibility, indent=2))


if __name__ == "__main__":
    main()
