"""Descriptive cosine separation; no Gaussian assumption, probe or threshold fitting."""

import numpy as np

VERSION = "fixed-pair-subject-separability-v1"


def separation(scores, labels, weights=None):
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights, float)
    if (scores.ndim != 1 or scores.shape != labels.shape or weights.shape != labels.shape
            or not np.isfinite(scores).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or set(np.unique(labels)) != {0, 1}):
        raise ValueError("finite aligned scores, binary labels and nonnegative weights required")
    means, variances = [], []
    for value in (0, 1):
        mask = labels == value
        if weights[mask].sum() == 0:
            return None
        mean = np.average(scores[mask], weights=weights[mask])
        means.append(float(mean))
        variances.append(float(np.average((scores[mask] - mean) ** 2, weights=weights[mask])))
    pooled = np.sqrt(np.mean(variances))
    return dict(negative_mean=means[0], positive_mean=means[1],
                negative_variance=variances[0], positive_variance=variances[1],
                standardized_mean_difference=float((means[1] - means[0]) / pooled)
                if pooled > 0 else None)


def infer(score_map, labels, subject_a, subject_b, *, resamples=2000, seed=0):
    """Shared subject draws; positives weighted once, negatives by endpoint product."""
    y, a, b = np.asarray(labels), np.asarray(subject_a), np.asarray(subject_b)
    if y.ndim != 1 or y.dtype.kind not in "iu" or set(np.unique(y)) != {0, 1}:
        raise ValueError("integer binary labels required")
    if a.shape != y.shape or b.shape != y.shape or "frozen" not in score_map:
        raise ValueError("aligned endpoint people and frozen comparator required")
    for ids in (a, b):
        if ids.dtype.kind not in "iuUS" or any(
            str(v).strip().lower() in {"", "none", "nan", "unknown"} for v in ids
        ) or (ids.dtype.kind in "iu" and np.any(ids < 0)):
            raise ValueError("known person IDs required")
    a, b = a.astype(str), b.astype(str)
    if np.any(a[y == 1] != b[y == 1]) or np.any(a[y == 0] == b[y == 0]):
        raise ValueError("person endpoints contradict labels")
    if type(resamples) is not int or resamples < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive resamples and nonnegative seed required")
    keys = ["frozen", *sorted(k for k in score_map if k != "frozen")]
    points = [separation(score_map[k], y) for k in keys]
    if any(p["standardized_mean_difference"] is None for p in points):
        return dict(status="unavailable: zero pooled score variance", version=VERSION)
    people, inverse = np.unique(np.r_[a, b], return_inverse=True)
    aa, bb = inverse[:len(y)], inverse[len(y):]
    rng, draws = np.random.default_rng(seed), []
    for _ in range(resamples):
        counts = np.bincount(rng.integers(len(people), size=len(people)), minlength=len(people))
        weights = counts[aa] * np.where(y == 1, 1, counts[bb])
        values = [separation(score_map[k], y, weights) for k in keys]
        if any(v is None or v["standardized_mean_difference"] is None for v in values):
            continue
        draws.append([v["standardized_mean_difference"] for v in values])
    if not draws:
        return dict(status="unavailable: no nondegenerate subject draws", version=VERSION)
    draws = np.asarray(draws)
    models = {}
    for index, key in enumerate(keys):
        models[key] = dict(**points[index],
                           ci95=np.quantile(draws[:, index], [.025, .975]).tolist(),
                           delta_vs_frozen=points[index]["standardized_mean_difference"]
                           - points[0]["standardized_mean_difference"],
                           delta_ci95=np.quantile(draws[:, index] - draws[:, 0], [.025, .975]).tolist())
    return dict(status="ok", version=VERSION, models=models, n_pairs=len(y),
                n_persons=len(people), requested_draws=resamples, valid_draws=len(draws), seed=seed,
                positive_weight="owner multiplicity once",
                negative_weight="endpoint multiplicity product",
                limitation="fixed image-pair cosine separation, not Gaussian d-prime or recognition accuracy; conditional fixed checkpoints/protocol, no training-seed population inference")
