"""Subject-aware uncertainty must preserve paired model scores and both IDs."""

from __future__ import annotations

import numpy as np
import pytest

from age_gap.evaluation.subject_bootstrap import (
    leave_one_subject_out_auc,
    paired_subject_bootstrap_auc,
)


def test_paired_subject_bootstrap_is_deterministic_and_reports_delta() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    subject_a = np.array([0, 1, 2, 0, 0, 1, 1, 2])
    subject_b = np.array([0, 1, 2, 1, 2, 0, 2, 0])
    frozen = np.array([0.7, 0.45, 0.55, 0.6, 0.3, 0.4, 0.5, 0.35])
    tuned = np.array([0.8, 0.65, 0.7, 0.5, 0.2, 0.3, 0.4, 0.25])

    first = paired_subject_bootstrap_auc(
        frozen, tuned, labels, subject_a, subject_b, n_boot=200, seed=42
    )
    second = paired_subject_bootstrap_auc(
        frozen, tuned, labels, subject_a, subject_b, n_boot=200, seed=42
    )
    assert first == second
    assert first.n_subjects == 3
    assert first.n_pairs == 8
    assert first.n_valid_resamples <= first.n_requested_resamples == 200
    assert first.delta_auc == pytest.approx(first.tuned_auc - first.frozen_auc)
    assert first.delta_ci95[0] <= first.delta_ci95[1]


def test_rejects_inconsistent_subject_labels() -> None:
    with pytest.raises(ValueError, match="contradict"):
        paired_subject_bootstrap_auc(
            np.array([0.9, 0.1]),
            np.array([0.9, 0.1]),
            np.array([1, 0]),
            np.array([0, 0]),
            np.array([1, 1]),
        )


def test_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        paired_subject_bootstrap_auc(
            np.array([np.nan, 0.1]),
            np.array([0.9, 0.1]),
            np.array([1, 0]),
            np.array([0, 0]),
            np.array([0, 1]),
        )


def test_leave_one_subject_out_excludes_both_pair_endpoints() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    subject_a = np.array([0, 1, 2, 0, 0, 1, 1, 2])
    subject_b = np.array([0, 1, 2, 1, 2, 0, 2, 0])
    frozen = np.array([0.7, 0.45, 0.55, 0.6, 0.3, 0.4, 0.5, 0.35])
    tuned = np.array([0.8, 0.65, 0.7, 0.5, 0.2, 0.3, 0.4, 0.25])
    result = leave_one_subject_out_auc(frozen, tuned, labels, subject_a, subject_b)
    assert result.n_subjects == result.n_evaluable_subjects == 3
    assert result.minimum_delta_auc <= result.median_delta_auc <= result.maximum_delta_auc
