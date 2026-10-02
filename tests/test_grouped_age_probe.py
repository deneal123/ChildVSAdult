"""DRAFT tests for the corrected representation diagnostics (read-only audit artifact).

Covers, using synthetic CPU-only data:
  * identity-group-disjoint folds (no identity spans two folds);
  * explicit null when identity metadata is missing (no row-fold fallback);
  * explicit null on tiny samples;
  * training-mean intercept recovers an offset target;
  * a linearly predictable target is recovered with ~0 MAE;
  * per-fold mean/median constant baselines are present and correct;
  * linear CKA sanity (self = 1, permutation invariant).

Run:  python -m pytest draft/test_representation_diagnostics_fixed.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from age_gap.evaluation import representation_diagnostics as mod

identity_group_folds = mod.identity_group_folds
ridge_probe_with_baselines = mod.ridge_probe_with_baselines
ridge_predict = mod._ridge_predict
linear_cka = mod.linear_cka


# --------------------------------------------------------------------------------------
# Fold construction
# --------------------------------------------------------------------------------------
def test_identity_group_folds_are_disjoint() -> None:
    groups = [f"id{i // 4}" for i in range(80)]  # 20 identities x 4 rows
    gf = identity_group_folds(groups, 5, seed=42)
    assert gf.usable
    for i, g in enumerate(groups):
        others = [j for j in range(len(groups)) if j != i and groups[j] == g]
        assert all(gf.fold_of_row[j] == gf.fold_of_row[i] for j in others)
    # every fold holds whole identities and is non-empty
    assert min(gf.fold_sizes) > 0
    assert sum(gf.identity_counts) == gf.n_identities


def test_fold_assignment_is_seeded_and_not_prefix_based() -> None:
    groups = [f"id{i}" for i in range(30)]
    a = identity_group_folds(groups, 5, seed=42).fold_of_row
    b = identity_group_folds(groups, 5, seed=42).fold_of_row
    c = identity_group_folds(groups, 5, seed=7).fold_of_row
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # a pure prefix split would assign id0..id5 to fold 0; check that is not forced
    assert len(set(a[:6].tolist())) > 1


def test_missing_group_metadata_is_explicit_null() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 8))
    y = rng.normal(size=60)
    result = ridge_probe_with_baselines(x, y, [None] * 60, folds=5)
    assert result["status"].startswith("unavailable")
    assert np.isnan(result["mae"])
    assert result["n_folds_used"] == 0


def test_too_few_identities_is_explicit_null() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 8))
    y = rng.normal(size=40)
    groups = ["only"] * 40
    result = ridge_probe_with_baselines(x, y, groups, folds=5)
    assert result["status"] == "unavailable: fewer distinct identities than folds"
    assert np.isnan(result["mae"])


def test_tiny_sample_is_explicit_null() -> None:
    x = np.zeros((4, 3))
    y = np.zeros(4)
    result = ridge_probe_with_baselines(x, y, ["a", "a", "b", "b"], folds=5)
    assert result["status"].startswith("unavailable: too few rows")
    assert np.isnan(result["mae"])


# --------------------------------------------------------------------------------------
# Intercept and predictable targets
# --------------------------------------------------------------------------------------
def test_training_mean_intercept_on_constant_features() -> None:
    # With zero-variance features the ridge has nothing to fit, so it must fall back to the
    # training-fold mean. The original no-intercept dual ridge would predict ~0 instead.
    x_train = np.ones((10, 4))
    y_train = np.full(10, 35.0)
    x_test = np.ones((3, 4))
    prediction = ridge_predict(x_train, y_train, x_test, regularization=1.0)
    assert np.allclose(prediction, 35.0, atol=1e-6)


def test_offset_target_is_recovered_not_zero() -> None:
    rng = np.random.default_rng(3)
    rows, dims = 600, 6
    groups = [f"id{i // 6}" for i in range(rows)]  # 100 identities
    signal = rng.normal(size=(rows, dims))
    target = 50.0 + signal[:, 0] * 2.0  # large offset, small linear signal
    result = ridge_probe_with_baselines(signal, target, groups, folds=5, seed=1)
    assert result["status"] == "ok"
    # still beats the best constant; the constant baseline MAE is small because the target
    # is tightly concentrated around the 50.0 offset, while the ridge recovers the signal.
    assert result["mae"] < result["baseline_mean_mae"]
    assert result["skill_over_mean"] > 0.9
    assert 1.0 < result["baseline_mean_mae"] < 3.0


def test_known_predictable_age_gap() -> None:
    rng = np.random.default_rng(11)
    rows, dims = 500, 5
    groups = [f"g{i // 5}" for i in range(rows)]
    x = rng.normal(size=(rows, dims))
    y = 3.0 * x[:, 0] - 1.5 * x[:, 1] + 7.0
    result = ridge_probe_with_baselines(x, y, groups, folds=5, seed=2, regularization=1e-6)
    assert result["status"] == "ok"
    assert result["mae"] < 0.05
    assert result["skill_over_best_constant"] > 0.99


def test_per_fold_baselines_are_reported() -> None:
    rng = np.random.default_rng(5)
    rows = 200
    groups = [f"id{i // 4}" for i in range(rows)]
    x = rng.normal(size=(rows, 4))
    y = rng.normal(loc=20.0, scale=5.0, size=rows)
    result = ridge_probe_with_baselines(x, y, groups, folds=4, seed=0)
    assert result["status"] == "ok"
    assert result["n_folds_used"] == 4
    assert np.isfinite(result["baseline_mean_mae"])
    assert np.isfinite(result["baseline_median_mae"])
    assert len(result["fold_mae"]) == 4
    # on pure noise the ridge should not beat the best constant materially
    assert result["skill_over_best_constant"] < 0.5


# --------------------------------------------------------------------------------------
# CKA
# --------------------------------------------------------------------------------------
def test_cka_identity_and_row_permutation_invariance() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 16))
    y = rng.normal(size=(64, 16))
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-9)
    # CKA is invariant to a permutation applied to the rows of BOTH matrices, not to one.
    perm = rng.permutation(64)
    assert linear_cka(x[perm], y[perm]) == pytest.approx(linear_cka(x, y), abs=1e-9)


def test_cka_independent_is_much_below_dependent() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 32))
    y = rng.normal(size=(400, 32))
    # An orthogonal linear map preserves the centered kernel exactly, so CKA == 1.
    q, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    dependent = linear_cka(x, x @ q)
    independent = linear_cka(x, y)
    assert dependent == pytest.approx(1.0, abs=1e-6)
    # Finite-sample CKA between independent matrices is small but not exactly zero.
    assert independent < 0.25
    assert independent < dependent - 0.5


def test_cka_requires_matching_rows() -> None:
    rng = np.random.default_rng(2)
    assert np.isnan(linear_cka(rng.normal(size=(8, 4)), rng.normal(size=(7, 4))))
