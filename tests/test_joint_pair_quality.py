import copy

import numpy as np
import pytest

from scripts.audit_joint_pair_quality import (
    diagnose_joint,
    joint_mmd,
    missing_pattern_tv,
    ordered_features,
    standardize,
)
from scripts.audit_repaired_nuisance import refit
from tests.test_repaired_nuisance import fixture


def dense_mmd(x, y, wx, wy, sigma):
    wx, wy = np.array(wx) / sum(wx), np.array(wy) / sum(wy)

    def kernel(a, b):
        return np.exp(-np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2) / (2 * sigma**2))

    return wx @ kernel(x, x) @ wx + wy @ kernel(y, y) @ wy - 2 * wx @ kernel(x, y) @ wy


def test_chunked_weighted_kernel_matches_dense_formula_for_multiple_views():
    x, y = np.array([[0, 1], [1, 2], [2, 0]]), np.array([[2, 1], [4, 3]])
    wx, wy = [[1, 2, 3], [0, 1, 0]], [[1, 2], [3, 0]]
    sigmas = [0.5, 1, 2]
    result, clipped = joint_mmd(x, y, wx, wy, sigmas=sigmas, chunk=1)
    expected = [[dense_mmd(x, y, a, b, s) for a, b in zip(wx, wy, strict=True)] for s in sigmas]
    np.testing.assert_allclose(result, expected, atol=1e-14)
    assert clipped == 0
    other, _ = joint_mmd(x, y, np.array(wx) * 7, np.array(wy) * 11, sigmas=sigmas, chunk=99)
    np.testing.assert_allclose(other, result, atol=1e-14)


def test_same_marginals_can_hide_joint_correlation_difference():
    x, y = np.array([[-1, -1], [1, 1]]), np.array([[-1, 1], [1, -1]])
    for column in (0, 1):
        np.testing.assert_array_equal(np.sort(x[:, column]), np.sort(y[:, column]))
    result, _ = joint_mmd(x, y, [[1, 1]], [[1, 1]], sigmas=[1])
    assert result[0, 0] > 0.7
    identical, _ = joint_mmd(x, x[::-1], [[1, 1]], [[1, 1]], sigmas=[1])
    assert identical[0, 0] == pytest.approx(0, abs=1e-14)


def test_splitting_mass_over_duplicate_observations_preserves_empirical_distance():
    x, y = np.array([[0], [1]]), np.array([[2]])
    expected, _ = joint_mmd(x, y, [[1, 1]], [[1]], sigmas=[1])
    duplicate, _ = joint_mmd(np.array([[0], [0], [1]]), y, [[0.5, 0.5, 1]], [[1]], sigmas=[1])
    np.testing.assert_allclose(expected, duplicate, atol=1e-14)


@pytest.mark.parametrize("weights", [[[0, 0]], [[-1, 2]], [[float("nan"), 1]], [[1]]])
def test_invalid_mass_fails_closed(weights):
    with pytest.raises(ValueError, match="weight"):
        joint_mmd(np.zeros((2, 1)), np.zeros((1, 1)), weights, [[1]], sigmas=[1])


def test_missing_pattern_is_joint_not_only_missing_fraction():
    x, y = [[0, 0], [1, 1]], [[0, 1], [1, 0]]
    assert missing_pattern_tv(x, y, [1, 1], [1, 1]) == 1
    with pytest.raises(ValueError, match="binary"):
        missing_pattern_tv([[2]], [[0]], [1], [1])


def test_quality_feature_order_is_explicit_and_ages_never_enter():
    row = dict(face_a="a", face_b="b", age_a=10, age_b=40)
    quality = {"a": dict(face_width=10), "b": dict(face_width=20)}
    result = ordered_features([row], quality)
    assert result.shape == (1, 36)
    np.testing.assert_array_equal(
        result, ordered_features([{**row, "age_a": 99, "age_b": 1}], quality)
    )
    swapped = ordered_features([{**row, "face_a": "b", "face_b": "a"}], quality)
    assert result[0, 2] == swapped[0, 11]
    assert result[0, 2] != swapped[0, 2]
    x, transform = standardize(np.concatenate([result, swapped]))
    assert np.isfinite(x).all() and transform["all_missing_columns"] == 16


def test_complete_joint_audit_uses_checked_ledger_and_never_closes_training_gate():
    args = fixture()
    _, diagnostics, ledger = refit(*args)
    before = copy.deepcopy((args, diagnostics, ledger))
    report = diagnose_joint(*args, diagnostics, ledger)
    assert (args, diagnostics, ledger) == before
    assert report["execution_complete"] and report["coupled_age_balance_verified"]
    assert not report["quality_balance_claimed"] and not report["training_weights_ready"]
    assert not report["publication_ready"]
    assert set(report["comparisons"]) == {
        "positive",
        "negative",
        "combined_paired_objective",
        "LOW_positive_vs_negative",
        "CROSS_positive_vs_negative",
    }
    assert set(report["comparisons"]["positive"]) == {
        "uniform",
        "quality_only",
        "younger_anchor",
        "older_anchor",
    }
    bad = copy.deepcopy(ledger)
    bad[0]["candidate_loss_weight"] += 0.1
    with pytest.raises(ValueError, match="ledger differs"):
        diagnose_joint(*args, diagnostics, bad)
