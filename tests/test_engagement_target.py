"""Тесты защиты от утечек и корректности таргетов (ветка engagement).

Проверяем ровно то, из-за чего результат мог бы оказаться фикцией:
  * один человек не попадает одновременно в train и test;
  * PCA эмбеддингов фитится ТОЛЬКО на train;
  * перемешанный таргет обязан ломать сигнал;
  * модель охвата действительно снимает дисперсию.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from age_gap.engagement.model import (
    _design,
    assert_no_group_leakage,
    make_splits,
    ndcg_at_k,
    topk_lift,
)
from age_gap.engagement.target import REACH_EXOGENOUS, oof_reach_residual, shuffle_within_bucket

# ------------------------------------------------------------------ ranking metrics


def test_ndcg_perfect_and_reversed():
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert ndcg_at_k(y, y, k=5) == pytest.approx(1.0)
    assert ndcg_at_k(y, -y, k=5) < ndcg_at_k(y, y, k=5)


def test_ndcg_constant_truth_is_nan_or_zero():
    y = np.ones(10)
    val = ndcg_at_k(y, np.arange(10.0), k=5)
    assert np.isnan(val) or val == 0.0


def test_topk_lift_positive_for_good_ranking():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    assert topk_lift(y, y, k=50) > 1.0  # идеальное ранжирование
    assert abs(topk_lift(y, rng.normal(size=500), k=50)) < 0.5  # случайное ~0


# ------------------------------------------------------------------ group leakage


def _toy_groups(n=300, n_persons=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"person_id": rng.integers(0, n_persons, n).astype(str)})


def test_groupkfold_has_no_person_overlap():
    df = _toy_groups()
    splits = make_splits(df, n_splits=5)
    assert_no_group_leakage(df, splits)  # не должно бросить


def test_assert_detects_deliberate_leakage():
    df = _toy_groups()
    idx = np.arange(len(df))
    leaky = [(idx, idx)]  # train == test -> все person'ы пересекаются
    with pytest.raises(AssertionError, match="Утечка person_id"):
        assert_no_group_leakage(df, leaky)


# ------------------------------------------------------------------ PCA fitted on train only


def _toy_emb_df(n=120, d=6, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "n_photos": rng.integers(1, 4, n).astype(float),
        "caption_len": rng.integers(0, 200, n).astype(float),
    })
    for j in range(d):
        df[f"emb{j:03d}"] = rng.normal(size=n)
    return df


def test_pca_is_fitted_on_train_only():
    df = _toy_emb_df()
    cols = {"meta": ["n_photos", "caption_len"], "face": [], "domain": [], "reach": [],
            "emb": [c for c in df.columns if c.startswith("emb")]}
    tr = np.arange(0, 80)
    te = np.arange(80, 120)

    Xtr_a, Xte_a, _ = _design(df, ["meta", "emb"], cols, tr, te, n_pca=3)

    # Радикально портим ТЕСТОВЫЕ строки: если PCA подсматривает в test, train-матрица изменится.
    df2 = df.copy()
    df2.loc[te, [c for c in df.columns if c.startswith("emb")]] *= 1000.0
    Xtr_b, Xte_b, _ = _design(df2, ["meta", "emb"], cols, tr, te, n_pca=3)

    np.testing.assert_allclose(Xtr_a, Xtr_b, rtol=1e-9, atol=1e-9)
    assert not np.allclose(Xte_a, Xte_b), "test-проекция обязана измениться"


def test_design_emb_nan_imputed_by_train_median():
    df = _toy_emb_df(n=40, d=4)
    emb_cols = [c for c in df.columns if c.startswith("emb")]
    df.loc[35:, emb_cols] = np.nan  # NaN только в test
    cols = {"meta": ["n_photos"], "face": [], "domain": [], "reach": [], "emb": emb_cols}
    Xtr, Xte, names = _design(df, ["meta", "emb"], cols, np.arange(30), np.arange(30, 40), n_pca=2)
    assert np.isfinite(Xtr).all() and np.isfinite(Xte).all()
    assert names[-2:] == ["pca00", "pca01"]


# ------------------------------------------------------------------ reach residualisation


def _toy_reach_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in REACH_EXOGENOUS})
    df["person_id"] = rng.integers(0, 80, n).astype(str)
    df["owner_id"] = rng.integers(0, 2, n).astype(str)
    df["year"] = 2024
    df["month"] = rng.integers(1, 13, n)
    # таргет почти полностью определяется охватом
    df["e_raw"] = 3.0 * df["post_age_days"] - 2.0 * df["hour"] + rng.normal(scale=0.2, size=n)
    return df


def test_reach_residual_removes_variance_and_is_oof():
    df = _toy_reach_df()
    splits = make_splits(df, n_splits=4)
    resid, stats = oof_reach_residual(df, splits, variant="A")

    assert not np.isnan(resid).any()
    assert stats["reach_r2_oof"] > 0.7, "модель охвата обязана поймать сильный сигнал"
    assert np.var(resid) < np.var(df["e_raw"]), "остаток должен иметь меньшую дисперсию"


def test_reach_variant_b_requires_views_column():
    df = _toy_reach_df()
    df["log_views"] = np.random.default_rng(1).normal(size=len(df))
    splits = make_splits(df, n_splits=3)
    _, stats = oof_reach_residual(df, splits, variant="B")
    assert "log_views" in stats["reach_features"]

    with pytest.raises(ValueError):
        oof_reach_residual(df, splits, variant="C")


# ------------------------------------------------------------------ negative control


def test_shuffle_preserves_bucket_multiset_but_breaks_alignment():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "owner_id": rng.choice(["a", "b"], n),
        "year": rng.choice([2023, 2024], n),
        "month": rng.integers(1, 4, n),
    })
    y = rng.normal(size=n)
    y_sh = shuffle_within_bucket(df, y.copy(), seed=1)

    # внутри каждого бакета набор значений сохранён
    key = list(zip(df["owner_id"], df["year"], df["month"], strict=True))
    for k in set(key):
        m = np.array([kk == k for kk in key])
        np.testing.assert_allclose(np.sort(y[m]), np.sort(y_sh[m]))

    assert not np.allclose(y, y_sh), "перемешивание обязано менять порядок"
