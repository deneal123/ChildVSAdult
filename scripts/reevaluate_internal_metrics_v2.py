"""Source-bound ROC-v2 paired internal inference from saved private scores only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.internal_exact_age_sensitivity import select_blocks
from scripts.render_fgnet_evidence import verified_result
from scripts.verification_metrics_v2 import VERSION, empirical_metrics

METRICS = ("roc_auc", "eer_interpolated", "eer_discrete_minimax", "tar@far=0.01", "tar@far=0.001")


def vector(scores, labels, weights):
    result = empirical_metrics(scores, labels, weights=weights)
    return np.asarray([result["roc_auc"], result["eer_interpolated"], result["eer_discrete_minimax"],
                       result["operating_points"]["0.01"]["tar"], result["operating_points"]["0.001"]["tar"]])


def infer(frozen, tuned, exact, owners, impostors, *, n_boot=2000, seed=42):
    frozen, tuned = np.asarray(frozen, float), np.asarray(tuned, float)
    exact, owners, impostors = np.asarray(exact), np.asarray(owners), np.asarray(impostors)
    n = len(exact)
    if (not n or exact.dtype.kind != "b" or not exact.any() or frozen.shape != (2 * n,)
            or tuned.shape != frozen.shape or owners.shape != (n,) or impostors.shape != (n,)
            or (owners == impostors).any() or any(ids.dtype.kind not in "US" for ids in (owners, impostors))
            or any(not str(v).strip() for v in np.r_[owners, impostors])):
        raise ValueError("aligned complete matched scores, exact mask and distinct recorded persons required")
    if type(n_boot) is not int or n_boot < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive resample count and nonnegative seed required")
    labels = np.r_[np.ones(n, int), np.zeros(n, int)]
    names, masks = ("original_tolerance", "exact_age_subset"), (np.ones(n, bool), exact)
    def estimates(weights):
        rows = []
        for mask in masks:
            pair_mask = np.r_[mask, mask]
            doubled = np.r_[weights, weights][pair_mask]
            f = vector(frozen[pair_mask], labels[pair_mask], doubled)
            t = vector(tuned[pair_mask], labels[pair_mask], doubled)
            rows.append(np.stack([f, t, t - f], axis=1))
        return np.stack(rows)
    points = estimates(np.ones(n))
    subjects, inverse = np.unique(np.r_[owners, impostors], return_inverse=True)
    aa, bb = inverse[:n], inverse[n:]
    rng, draws = np.random.default_rng(seed), []
    for _ in range(n_boot):
        m = np.bincount(rng.integers(len(subjects), size=len(subjects)), minlength=len(subjects))
        weights = m[aa] * m[bb]
        if not (weights * exact).any():
            continue
        draws.append(estimates(weights))
    if not draws:
        raise ValueError("no valid shared subject draws")
    draws = np.asarray(draws)
    ci = np.percentile(draws, [2.5, 97.5], axis=0)
    differences = draws[:, 1, :, 2] - draws[:, 0, :, 2]
    change_ci = np.percentile(differences, [2.5, 97.5], axis=0)
    result = {"metric_version": VERSION, "subsets": {}, "exact_minus_original_model_delta": {}}
    for j, (name, mask) in enumerate(zip(names, masks, strict=True)):
        metrics = {}
        for k, metric in enumerate(METRICS):
            metrics[metric] = {tag: {"point": float(points[j, k, index]), "ci95": ci[:, j, k, index].tolist()}
                for index, tag in enumerate(("frozen", "tuned", "tuned_minus_frozen"))}
        pair_mask = np.r_[mask, mask]
        result["subsets"][name] = {"n_positive": int(mask.sum()), "n_negative": int(mask.sum()),
            "recorded_people": len(np.unique(np.r_[owners[mask], impostors[mask]])), "metrics": metrics,
            "operating_points": {tag: empirical_metrics(scores[pair_mask], labels[pair_mask])["operating_points"]
                                 for tag, scores in (("frozen", frozen), ("tuned", tuned))}}
    for k, metric in enumerate(METRICS):
        result["exact_minus_original_model_delta"][metric] = {"point": float(points[1, k, 2] - points[0, k, 2]),
                                                              "ci95": change_ci[:, k].tolist()}
    result["bootstrap"] = {"seed": seed, "requested": n_boot, "valid": len(draws), "recorded_people": len(subjects),
        "sampling": "joint recorded-person dyad-multiplicity product; positive/negative matched block same weight",
        "scope": "conditional fixed checkpoint, selected protocol and reselected test ROC thresholds; not training-seed uncertainty",
        "threshold_reselection": True, "multiple_comparisons_corrected": False}
    result["scope"] = "paired sensitivity of saved internal scores; subset selection changes composition; no new training or image processing"
    result["eer_scope"] = "interpolated ROC crossing and discrete minimax reported separately; do not pool with differently defined EER"
    result["deployment_calibration"] = False
    result["verified_identity_independence"] = False
    result["publication_ready"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "metrics/internal_exact_age_sensitivity_20261003")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    summary = args.source / "summary.json"
    previous = verified_result(summary, PROJECT_ROOT)
    manifest_path = summary.with_suffix(".manifest.json")
    native = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair_path = PROJECT_ROOT / "data/processed/experiments/pairs_internal_endpoint_age_matched.jsonl"
    people_path = PROJECT_ROOT / "data/processed/person_clusters.jsonl"
    score_path = args.source / "private/scores.npz"
    if file_record(score_path) not in native["outputs"] or any(file_record(p) not in native["inputs"] for p in (pair_path, people_path)):
        raise ValueError("saved score/pair/person source binding mismatch")
    groups = {}
    for row in read_jsonl(people_path):
        group, person = row["identity_group_id"], row["person_id"]
        if group in groups and groups[group] != person:
            raise ValueError("contradictory person mapping")
        groups[group] = person
    exact, owners, impostors = select_blocks(list(read_jsonl(pair_path)), groups)
    paths = [summary, manifest_path, pair_path, people_path, score_path, Path(__file__),
             PROJECT_ROOT / "scripts/verification_metrics_v2.py", PROJECT_ROOT / "scripts/internal_exact_age_sensitivity.py",
             PROJECT_ROOT / "scripts/audit_age_label_balance.py", PROJECT_ROOT / "scripts/render_fgnet_evidence.py",
             *sorted((PROJECT_ROOT / "src/age_gap").rglob("*.py")), *sorted((PROJECT_ROOT / "src/age_gap").rglob("*.toml"))]
    before = [file_record(p) for p in paths]
    with np.load(score_path, allow_pickle=False) as cache:
        if set(cache.files) != {"frozen", "tuned", "exact"} or not np.array_equal(cache["exact"], exact):
            raise ValueError("score cache exact mask does not match source pairs")
        result = infer(cache["frozen"], cache["tuned"], exact, owners, impostors)
    result["source_protocol"] = previous["protocol"]
    if before != [file_record(p) for p in paths]:
        raise ValueError("metric re-evaluation inputs changed")
    args.out.mkdir(parents=True)
    output = args.out / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = output.with_suffix(".manifest.json")
    write_experiment_manifest(target, experiment="internal-roc-v2-paired-inference", parameters={"seed": 42, "n_boot": 2000,
        "metric_version": VERSION, "threshold_reselection": "test empirical ROC, not calibrated deployment"},
        metrics=result, inputs=paths, outputs=[output])
    if json.loads(target.read_text(encoding="utf-8"))["inputs"] != before:
        target.unlink()
        raise ValueError("inputs changed while writing manifest; completed marker withdrawn")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
