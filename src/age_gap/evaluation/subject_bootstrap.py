"""Paired ROC-AUC uncertainty when verification pairs reuse subjects.

The subject is the sampling unit. A positive pair receives the multiplicity of
its subject; a negative pair receives the product of its two subjects'
multiplicities. Frozen and tuned scores use the same resamples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class SubjectBootstrapAuc:
    frozen_auc: float
    tuned_auc: float
    delta_auc: float
    frozen_ci95: tuple[float, float]
    tuned_ci95: tuple[float, float]
    delta_ci95: tuple[float, float]
    n_subjects: int
    n_pairs: int
    n_valid_resamples: int
    n_requested_resamples: int


@dataclass(frozen=True)
class LeaveOneSubjectOutAuc:
    minimum_delta_auc: float
    maximum_delta_auc: float
    median_delta_auc: float
    n_evaluable_subjects: int
    n_subjects: int


def leave_one_subject_out_auc(
    frozen_scores: np.ndarray,
    tuned_scores: np.ndarray,
    labels: np.ndarray,
    subject_a: np.ndarray,
    subject_b: np.ndarray,
) -> LeaveOneSubjectOutAuc:
    """Summarize the paired gain after removing each subject and all incident pairs."""
    frozen = np.asarray(frozen_scores, dtype=float)
    tuned = np.asarray(tuned_scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    a = np.asarray(subject_a)
    b = np.asarray(subject_b)
    if frozen.ndim != 1 or any(x.shape != frozen.shape for x in (tuned, y, a, b)):
        raise ValueError("scores, labels, and both subject arrays must be equal-length vectors")
    if frozen.size == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("both positive and negative pairs are required")
    if not np.isfinite(frozen).all() or not np.isfinite(tuned).all():
        raise ValueError("scores must be finite")
    if np.any((y == 1) & (a != b)) or np.any((y == 0) & (a == b)):
        raise ValueError("subject endpoints contradict pair labels")
    subjects = np.unique(np.concatenate((a, b)))
    deltas = []
    for subject in subjects:
        keep = (a != subject) & (b != subject)
        if set(np.unique(y[keep])) != {0, 1}:
            continue
        deltas.append(
            float(roc_auc_score(y[keep], tuned[keep]) - roc_auc_score(y[keep], frozen[keep]))
        )
    if not deltas:
        raise ValueError("no evaluable leave-one-subject-out subsets")
    return LeaveOneSubjectOutAuc(
        minimum_delta_auc=float(np.min(deltas)),
        maximum_delta_auc=float(np.max(deltas)),
        median_delta_auc=float(np.median(deltas)),
        n_evaluable_subjects=len(deltas),
        n_subjects=len(subjects),
    )


def paired_subject_bootstrap_auc(
    frozen_scores: np.ndarray,
    tuned_scores: np.ndarray,
    labels: np.ndarray,
    subject_a: np.ndarray,
    subject_b: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> SubjectBootstrapAuc:
    """Resample subjects, retaining both endpoints' dependence in negative pairs.

    Positive pairs must have one shared subject and negative pairs two distinct
    subjects. Invalid replicate class balances are skipped and counted.
    """
    frozen = np.asarray(frozen_scores, dtype=float)
    tuned = np.asarray(tuned_scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    a = np.asarray(subject_a)
    b = np.asarray(subject_b)
    if frozen.ndim != 1 or any(x.shape != frozen.shape for x in (tuned, y, a, b)):
        raise ValueError("scores, labels, and both subject arrays must be equal-length vectors")
    if frozen.size == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("both positive and negative pairs are required")
    if not np.isfinite(frozen).all() or not np.isfinite(tuned).all():
        raise ValueError("scores must be finite")
    if np.any((y == 1) & (a != b)) or np.any((y == 0) & (a == b)):
        raise ValueError("subject endpoints contradict pair labels")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    subjects, inverse = np.unique(np.concatenate((a, b)), return_inverse=True)
    n_subjects = len(subjects)
    if n_subjects < 2:
        raise ValueError("at least two subjects are required")
    ia, ib = inverse[: len(y)], inverse[len(y) :]
    frozen_auc = float(roc_auc_score(y, frozen))
    tuned_auc = float(roc_auc_score(y, tuned))
    rng = np.random.default_rng(seed)
    samples: list[tuple[float, float]] = []
    for _ in range(n_boot):
        sampled = rng.integers(n_subjects, size=n_subjects)
        multiplicity = np.bincount(sampled, minlength=n_subjects)
        weights = multiplicity[ia] * np.where(y == 1, 1, multiplicity[ib])
        if not np.any(weights[y == 0]) or not np.any(weights[y == 1]):
            continue
        samples.append(
            (
                float(roc_auc_score(y, frozen, sample_weight=weights)),
                float(roc_auc_score(y, tuned, sample_weight=weights)),
            )
        )
    if not samples:
        raise ValueError("no valid subject-bootstrap resamples")
    boot = np.asarray(samples)
    delta = boot[:, 1] - boot[:, 0]

    def interval(values: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(values, [2.5, 97.5])
        return float(lo), float(hi)

    return SubjectBootstrapAuc(
        frozen_auc=frozen_auc,
        tuned_auc=tuned_auc,
        delta_auc=tuned_auc - frozen_auc,
        frozen_ci95=interval(boot[:, 0]),
        tuned_ci95=interval(boot[:, 1]),
        delta_ci95=interval(delta),
        n_subjects=n_subjects,
        n_pairs=len(y),
        n_valid_resamples=len(samples),
        n_requested_resamples=n_boot,
    )
