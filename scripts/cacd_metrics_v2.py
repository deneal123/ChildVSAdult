"""Paired fold/class-stratified pair inference, not subject uncertainty, for CACD-VS."""
from __future__ import annotations

import numpy as np

from scripts.benchmark_metrics_v2 import KEYS, METRICS, metric_vector
from scripts.evaluate_lfw_bound import THRESHOLD_SHA256, official_fold_accuracy

ALL_METRICS = (*METRICS, "accuracy_leave_one_fold_out_fixed_grid")


def infer(scores, labels, folds, *, n_boot=2000, seed=0):
    y, f = np.asarray(labels), np.asarray(folds)
    if (set(scores) != set(KEYS) or y.ndim != 1 or f.shape != y.shape or y.dtype.kind not in "iu"
            or f.dtype.kind not in "iu" or set(np.unique(y)) != {0, 1}
            or not np.array_equal(np.unique(f), np.arange(len(np.unique(f)))) or len(np.unique(f)) < 2):
        raise ValueError("four checkpoints and genuine integer class/fold metadata required")
    if type(n_boot) is not int or n_boot < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive bootstrap budget and nonnegative seed required")
    vectors = [np.asarray(scores[k], float) for k in KEYS]
    if any(s.shape != y.shape or not np.isfinite(s).all() or np.any(np.abs(s) > 1) for s in vectors):
        raise ValueError("equal-length finite cosine scores in [-1,1] required")
    buckets = [np.flatnonzero((f == fold) & (y == label)) for fold in np.unique(f) for label in (0, 1)]
    if any(not len(bucket) for bucket in buckets):
        raise ValueError("each official fold must contain both classes")

    def values(s, yy, ff):
        return np.r_[metric_vector(s, yy), official_fold_accuracy(s, yy, ff)["accuracy"]]

    points = np.stack([values(s, y, f) for s in vectors])
    rng, draws = np.random.default_rng(seed), []
    for _ in range(n_boot):
        index = np.concatenate([bucket[rng.integers(len(bucket), size=len(bucket))] for bucket in buckets])
        draws.append(np.stack([values(s[index], y[index], f[index]) for s in vectors]))
    draws = np.asarray(draws)
    models = {}
    for j, key in enumerate(KEYS):
        ci = np.percentile(draws[:, j], [2.5, 97.5], axis=0).T
        delta_ci = np.percentile(draws[:, j] - draws[:, 0], [2.5, 97.5], axis=0).T
        models[key] = {metric: {"point": float(points[j, k]), "pair_ci95": ci[k].tolist(),
            "delta_vs_frozen": float(points[j, k] - points[0, k]), "paired_pair_delta_ci95": delta_ci[k].tolist()}
            for k, metric in enumerate(ALL_METRICS)}
    mean_draws = draws[:, 1:].mean(axis=1)
    delta_ci = np.percentile(mean_draws - draws[:, 0], [2.5, 97.5], axis=0).T
    mean_ci = np.percentile(mean_draws, [2.5, 97.5], axis=0).T
    mean = {metric: {"mean": float(points[1:, k].mean()), "sd": float(points[1:, k].std(ddof=1)),
        "pair_mean_ci95": mean_ci[k].tolist(), "mean_checkpoint_delta": float(points[1:, k].mean() - points[0, k]),
        "paired_pair_delta_ci95": delta_ci[k].tolist()} for k, metric in enumerate(ALL_METRICS)}
    return {"metric_version": "empirical-roc-v2", "n_pairs": len(y), "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()), "n_folds": len(np.unique(f)), "models": models,
        "three_checkpoint_aggregate": mean,
        "accuracy": {"threshold_grid": "fixed [-1,1],4001 points; lowest-threshold tie break",
            "threshold_grid_sha256": THRESHOLD_SHA256, "threshold_selection": "other protocol folds only; reselected per draw",
            "scope": "declared leave-one-fold-out fixed-grid estimand; not published algorithm reproduction",
            "legacy_grid_equivalence_claimed": False},
        "bootstrap": {"seed": seed, "requested": n_boot, "valid": len(draws),
            "sampling_unit": "pair, stratified within each protocol fold and class", "percentile_method": "linear",
            "shared_draws_across_models": True, "subject_metadata_available": False,
            "conditioning": "fixed checkpoints and protocol; not training-seed population or ensemble AUC"},
        "scope": "pair-level uncertainty does not account for reused subjects/photos; not deployment calibration; no multiplicity correction",
        "training_identity_independence": "unverified", "publication_ready": False}
