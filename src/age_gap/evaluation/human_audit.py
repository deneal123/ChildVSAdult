"""Metrics for the preregistered, two-annotator supervision audit.

The module intentionally has no UI or dataset dependencies.  Annotation tools
write one JSONL row per blinded task and this module turns the two independent
files into agreement and gold-vs-automatic estimates with bootstrap intervals.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Unweighted Cohen's kappa for two equally ordered categorical vectors."""
    if len(a) != len(b) or not a:
        raise ValueError("kappa requires two non-empty vectors of equal length")
    observed = float(np.mean(np.asarray(a, dtype=object) == np.asarray(b, dtype=object)))
    labels = set(a) | set(b)
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[x] / len(a)) * (cb[x] / len(b)) for x in labels)
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)


def binary_metrics(truth: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float]:
    """Precision, recall and F1 for an automatic binary decision."""
    if len(truth) != len(predicted) or not truth:
        raise ValueError("binary metrics require non-empty vectors of equal length")
    tp = sum(t and p for t, p in zip(truth, predicted, strict=True))
    fp = sum(not t and p for t, p in zip(truth, predicted, strict=True))
    fn = sum(t and not p for t, p in zip(truth, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def percentile_ci(
    values: Sequence[float],
    statistic,
    *,
    seed: int = 20260922,
    iterations: int = 2000,
) -> tuple[float, float]:
    """Deterministic non-parametric 95% percentile bootstrap interval."""
    array = np.asarray(values)
    if array.shape[0] == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        indices = rng.integers(0, array.shape[0], array.shape[0])
        estimates[i] = float(statistic(array[indices]))
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)
