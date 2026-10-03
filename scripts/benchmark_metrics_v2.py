"""Joint external-person ROC-v2 inference for four fixed FaceNet checkpoints."""
from __future__ import annotations

import numpy as np

from scripts.verification_metrics_v2 import VERSION, empirical_metrics

KEYS = ("frozen", "tuned_seed42", "tuned_seed1", "tuned_seed2")
METRICS = ("roc_auc", "eer_interpolated", "eer_discrete_minimax", "tar@far=0.01", "tar@far=0.001")


def metric_vector(scores, labels, weights=None):
    row = empirical_metrics(scores, labels, weights=weights)
    return np.asarray([row["roc_auc"], row["eer_interpolated"], row["eer_discrete_minimax"],
        row["operating_points"]["0.01"]["tar"], row["operating_points"]["0.001"]["tar"]])


def infer(score_map, labels, subject_a, subject_b, *, n_boot=2000, seed=0):
    y, a, b = np.asarray(labels), np.asarray(subject_a), np.asarray(subject_b)
    if set(score_map) != set(KEYS) or y.ndim != 1 or a.shape != y.shape or b.shape != y.shape:
        raise ValueError("four fixed checkpoints and aligned person endpoints required")
    if y.dtype.kind not in "iu" or set(np.unique(y)) != {0, 1}:
        raise ValueError("integer binary labels required")
    if any(ids.dtype.kind not in "iuUS" or any(not str(v).strip() for v in ids) for ids in (a, b)):
        raise ValueError("nonmissing numeric/named benchmark persons required")
    if any(ids.dtype.kind in "iu" and (ids < 0).any() for ids in (a, b)):
        raise ValueError("nonnegative person IDs required")
    if (a[y == 1] != b[y == 1]).any() or (a[y == 0] == b[y == 0]).any():
        raise ValueError("person metadata contradicts labels")
    if type(n_boot) is not int or n_boot < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive resample budget and nonnegative seed required")
    scores = [np.asarray(score_map[key], float) for key in KEYS]
    points = np.stack([metric_vector(s, y) for s in scores])
    subjects, indices = np.unique(np.r_[a, b], return_inverse=True)
    aa, bb = indices[:len(y)], indices[len(y):]
    rng, draws = np.random.default_rng(seed), []
    for _ in range(n_boot):
        m = np.bincount(rng.integers(len(subjects), size=len(subjects)), minlength=len(subjects))
        w = m[aa] * np.where(y == 1, 1, m[bb])
        if min(w[y == 0].sum(), w[y == 1].sum()) <= 0:
            continue
        draws.append(np.stack([metric_vector(s, y, w) for s in scores]))
    if not draws:
        raise ValueError("no valid paired person resamples")
    draws = np.asarray(draws)
    models = {}
    for j, key in enumerate(KEYS):
        ci = np.percentile(draws[:, j], [2.5, 97.5], axis=0).T
        delta_ci = np.percentile(draws[:, j] - draws[:, 0], [2.5, 97.5], axis=0).T
        models[key] = {metric: {"point": float(points[j, k]), "ci95": ci[k].tolist(),
            "delta_vs_frozen": float(points[j, k] - points[0, k]), "delta_ci95": delta_ci[k].tolist()}
            for k, metric in enumerate(METRICS)}
    mean_draws = draws[:, 1:].mean(axis=1)
    mean_ci = np.percentile(mean_draws, [2.5, 97.5], axis=0).T
    delta_ci = np.percentile(mean_draws - draws[:, 0], [2.5, 97.5], axis=0).T
    aggregate = {metric: {"mean": float(points[1:, k].mean()), "sd": float(points[1:, k].std(ddof=1)),
        "mean_ci95": mean_ci[k].tolist(), "mean_checkpoint_delta": float(points[1:, k].mean() - points[0, k]),
        "delta_ci95": delta_ci[k].tolist()} for k, metric in enumerate(METRICS)}
    return {"metric_version": VERSION, "n_pairs": len(y), "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()), "n_subjects": len(subjects), "models": models,
        "three_checkpoint_aggregate": aggregate,
        "bootstrap": {"seed": seed, "requested": n_boot, "valid": len(draws),
            "positive_weight": "one shared person multiplicity", "negative_weight": "both-endpoint multiplicity product",
            "shared_draws_across_models": True, "thresholds_reselected": True,
            "conditioning": "fixed checkpoints and benchmark protocol; not training-seed population or ensemble-score AUC"},
        "scope": "test-derived ROC with separately named EER definitions; not deployment calibration; intervals not multiplicity-corrected",
        "training_identity_independence": "unverified", "publication_ready": False}
