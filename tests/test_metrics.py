"""Тесты метрик верификации (numpy)."""

from __future__ import annotations

import numpy as np

from age_gap.evaluation.metrics import (
    bootstrap_auc_ci,
    eer,
    roc_auc,
    tar_at_far,
    verification_metrics,
)


def test_roc_auc_perfect_separation():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert roc_auc(scores, labels) == 1.0


def test_roc_auc_inverted():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([1, 1, 0, 0])
    assert roc_auc(scores, labels) == 0.0


def test_roc_auc_ties_half():
    # Все скоры равны -> AUC = 0.5 (полная неопределённость).
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 1, 0, 0])
    assert roc_auc(scores, labels) == 0.5


def test_eer_perfect_separation():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert eer(scores, labels) == 0.0


def test_tar_at_far_perfect():
    scores = np.array([0.9, 0.85, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    # При идеальном разделении все позитивы принимаются даже при FAR=0.
    assert tar_at_far(scores, labels, far_target=0.001) == 1.0


def test_tar_at_far_threshold_allows_some_neg():
    # 10 негативов, 10 позитивов; позитивы выше негативов кроме одного перекрытия.
    neg = np.linspace(0.0, 0.5, 10)
    pos = np.linspace(0.4, 0.9, 10)
    scores = np.concatenate([pos, neg])
    labels = np.array([1] * 10 + [0] * 10)
    tar = tar_at_far(scores, labels, far_target=0.1)
    assert 0.0 <= tar <= 1.0


def test_verification_metrics_keys():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    m = verification_metrics(scores, labels)
    for key in ("roc_auc", "eer", "n_pos", "n_neg", "tar@far=0.01", "tar@far=0.001"):
        assert key in m


def test_empty_class_returns_nan():
    scores = np.array([0.9, 0.8])
    labels = np.array([1, 1])  # нет негативов
    assert np.isnan(roc_auc(scores, labels))
    assert np.isnan(eer(scores, labels))


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    pos = rng.normal(1.0, 1.0, 200)
    neg = rng.normal(-1.0, 1.0, 200)
    scores = np.concatenate([pos, neg])
    labels = np.array([1] * 200 + [0] * 200)
    point = roc_auc(scores, labels)
    lo, hi = bootstrap_auc_ci(scores, labels, n_boot=300, seed=0)
    assert lo < point < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_wider_for_small_sample():
    rng = np.random.default_rng(1)
    big = (
        np.concatenate([rng.normal(1, 1, 300), rng.normal(-1, 1, 300)]),
        np.array([1] * 300 + [0] * 300),
    )
    small = (
        np.concatenate([rng.normal(1, 1, 20), rng.normal(-1, 1, 20)]),
        np.array([1] * 20 + [0] * 20),
    )
    lo_b, hi_b = bootstrap_auc_ci(*big, n_boot=300, seed=0)
    lo_s, hi_s = bootstrap_auc_ci(*small, n_boot=300, seed=0)
    assert (hi_s - lo_s) > (hi_b - lo_b)


def test_bootstrap_ci_empty():
    lo, hi = bootstrap_auc_ci(np.array([]), np.array([]))
    assert np.isnan(lo) and np.isnan(hi)
