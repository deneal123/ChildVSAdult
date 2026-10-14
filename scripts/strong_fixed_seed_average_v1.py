"""Fixed-seed ROC-v2 averaging for exactly three tuned strong checkpoints.

Estimand: the arithmetic mean of the three *individual* checkpoint ROC-v2
metrics, followed by a paired difference to the common frozen metric measured on
the same frozen score array, labels and recorded subject endpoints. It is
explicitly **not** the metric of averaged scores (no ensemble).

Every quantity is conditional on the three fixed tuned checkpoints keyed by the
integer training seeds ``{42, 1, 2}`` and on the recorded test subjects. This
module makes no training-seed population, full-36-cell, mechanism, multiplicity,
human-independence or deployment-calibration claim; ``publication_ready`` is
``False``. It is pure numerical code: no CLI, no native runner, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from scripts.benchmark_metrics_v2 import METRICS, metric_vector
from scripts.verification_metrics_v2 import VERSION

__all__ = ["FIXED_SEEDS", "FROZEN_KEY", "SCOPE", "infer"]

FROZEN_KEY = "frozen"
FIXED_SEEDS = (42, 1, 2)
_ORDERED_KEYS = (FROZEN_KEY, *FIXED_SEEDS)
_MISSING_TOKENS = frozenset({"", "none", "nan", "unknown", "null", "na"})

SCOPE = (
    "conditional on exactly three fixed tuned checkpoints (integer training seeds 42, 1, 2) "
    "and on the recorded test subjects with their caller-supplied frozen/tuned score arrays, "
    "binary labels and endpoints; the aggregate estimand is the arithmetic mean of the three "
    "individual checkpoint ROC-v2 metrics, not the metric of averaged scores (no ensemble); "
    "the sample standard deviation across the three checkpoint point estimates is descriptive "
    "and is not a training-seed population interval; no training-seed population inference, "
    "no full36-cell or mechanism claim, no multiplicity correction, no human-independence "
    "claim, no calibrated deployment claim; intervals are paired subject-bootstrap percentile "
    "intervals conditional on this fixed test set"
)


def _prepare_subjects(subject_ids, shape):
    ids = np.asarray(subject_ids)
    if ids.ndim != 1 or ids.shape != shape:
        raise ValueError("aligned recorded subject endpoints required")
    if ids.dtype.kind not in "iuUS":
        raise ValueError("known integer or string subject IDs required")
    if ids.dtype.kind in "iu" and (ids < 0).any():
        raise ValueError("nonnegative subject IDs required")
    strings = ids.astype(str)
    if any(value.strip().lower() in _MISSING_TOKENS for value in strings):
        raise ValueError("known nonmissing subject IDs required")
    return strings


def _prepare_labels(labels, subject_a, subject_b):
    y = np.asarray(labels)
    if y.ndim != 1 or y.dtype.kind not in "iu":
        raise ValueError("one-dimensional genuine integer binary labels required")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("both binary classes required")
    a = _prepare_subjects(subject_a, y.shape)
    b = _prepare_subjects(subject_b, y.shape)
    if (a[y == 1] != b[y == 1]).any() or (a[y == 0] == b[y == 0]).any():
        raise ValueError("recorded subject endpoints contradict the labels")
    if len(np.unique(np.concatenate([a, b]))) < 2:
        raise ValueError("at least two recorded subjects required")
    return y, a, b


def _prepare_matrix(scores, n_pairs):
    if not isinstance(scores, Mapping) or set(scores) != set(_ORDERED_KEYS):
        raise ValueError("exactly frozen and integer training seeds 42, 1 and 2 required")
    if any(not ((type(key) is str and key == FROZEN_KEY)
                or (type(key) is int and key in FIXED_SEEDS)) for key in scores):
        raise ValueError("exact builtin integer training seeds required; no bool/float keys")
    rows = []
    for key in _ORDERED_KEYS:
        raw = np.asarray(scores[key])
        if raw.ndim != 1 or raw.shape[0] != n_pairs:
            raise ValueError("aligned one-dimensional score vectors required")
        if raw.dtype.kind not in "iuf":
            raise ValueError("real numeric score vectors required")
        row = raw.astype(float)
        if not np.isfinite(row).all():
            raise ValueError("finite real score vectors required")
        rows.append(row)
    return np.stack(rows)


def _point_matrix(matrix, y, weights=None):
    return np.stack([metric_vector(matrix[i], y, weights) for i in range(matrix.shape[0])])


def _interval(values):
    return np.percentile(values, [2.5, 97.5], axis=0).T.tolist()


def infer(scores, labels, subject_a, subject_b, *, n_boot=2000, seed=0):
    """ROC-v2 three-fixed-checkpoint mean inference on caller-supplied pairs.

    ``scores`` maps ``"frozen"`` and the integer training seeds ``42, 1, 2`` to
    same-length real similarity vectors. ``labels`` are binary genuine(1)/
    impostor(0) integers; ``subject_a``/``subject_b`` are the recorded endpoints
    aligned to the same rows. No subject-only join is performed.
    """
    if type(n_boot) is not int or n_boot < 1:
        raise ValueError("positive integer bootstrap count required")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer bootstrap seed required")
    y, a, b = _prepare_labels(labels, subject_a, subject_b)
    n_pairs = len(y)
    matrix = _prepare_matrix(scores, n_pairs)

    def points_at(mask=None, weights=None):
        if mask is None:
            return _point_matrix(matrix, y, weights)
        subset = matrix[:, mask]
        sub_weights = None if weights is None else weights[mask]
        return _point_matrix(subset, y[mask], sub_weights)

    points = points_at()

    people, inverse = np.unique(np.concatenate([a, b]), return_inverse=True)
    aa, bb = inverse[:n_pairs], inverse[n_pairs:]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        multiplicity = np.bincount(
            rng.integers(len(people), size=len(people)), minlength=len(people)
        )
        weights = multiplicity[aa] * np.where(y == 1, 1.0, multiplicity[bb])
        if weights[y == 0].sum() <= 0 or weights[y == 1].sum() <= 0:
            continue
        draws.append(points_at(weights=weights))
    if not draws:
        raise ValueError("no valid subject resamples")
    draws = np.stack(draws)

    frozen_draws = draws[:, 0]
    mean_draws = draws[:, 1:].mean(axis=1)
    # Average paired differences before reduction to retain exact zero when
    # each tuned metric equals its frozen reference, including rounding cases.
    mean_delta_draws = (draws[:, 1:] - frozen_draws[:, None, :]).mean(axis=1)

    per_seed = {}
    for position, training_seed in enumerate(FIXED_SEEDS, start=1):
        per_seed[str(training_seed)] = {
            name: {
                "point": float(points[position, k]),
                "ci95": _interval(draws[:, position, k]),
                "delta_vs_frozen": float(points[position, k] - points[0, k]),
                "delta_ci95": _interval(draws[:, position, k] - frozen_draws[:, k]),
            }
            for k, name in enumerate(METRICS)
        }

    frozen = {
        name: {"point": float(points[0, k]), "ci95": _interval(frozen_draws[:, k])}
        for k, name in enumerate(METRICS)
    }

    loo_rows = []
    loo_unavailable = 0
    for person in people:
        mask = (a != person) & (b != person)
        if set(np.unique(y[mask])) == {0, 1}:
            subset = points_at(mask=mask)
            loo_rows.append((subset[1:] - subset[0]).mean(axis=0))
        else:
            loo_unavailable += 1
    loo = np.stack(loo_rows) if loo_rows else np.empty((0, len(METRICS)))

    aggregate = {}
    for k, name in enumerate(METRICS):
        tuned = points[1:, k]
        aggregate[name] = {
            "mean": float(tuned.mean()),
            "sd": 0.0 if np.all(tuned == tuned[0]) else float(tuned.std(ddof=1)),
            # Mean of the individual paired deltas (exactly zero for identical checkpoints).
            "mean_checkpoint_delta": float(np.mean(tuned - points[0, k])),
            "mean_ci95": _interval(mean_draws[:, k]),
            "delta_ci95": _interval(mean_delta_draws[:, k]),
            "loo_delta_min": float(loo[:, k].min()) if len(loo) else None,
            "loo_delta_max": float(loo[:, k].max()) if len(loo) else None,
        }

    return {
        "metric_version": VERSION,
        "fixed_seed_list": list(FIXED_SEEDS),
        "n_pairs": n_pairs,
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_subjects": len(people),
        "estimand": "arithmetic mean of the three individual fixed-checkpoint metrics",
        "sd_scope": "descriptive sample standard deviation (ddof=1) across the three checkpoints",
        "frozen": frozen,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "loo_valid": len(loo_rows),
        "loo_unavailable": loo_unavailable,
        "bootstrap": {
            "seed": seed,
            "requested": n_boot,
            "valid": len(draws),
            "invalid": n_boot - len(draws),
            "positive_weight": "one shared owner multiplicity",
            "negative_weight": "both-endpoint multiplicity product",
            "shared_weights_across_models": True,
            "thresholds_reselected_per_draw": True,
            "averaging": (
                "mean of the three per-checkpoint metrics recomputed inside each draw, then "
                "paired difference to the frozen metric of the same draw; precomputed endpoints "
                "are never averaged"
            ),
            "conditioning": "fixed three checkpoints and recorded test subjects",
        },
        "scope": SCOPE,
        "training_identity_independence": "unverified",
        "publication_ready": False,
    }

