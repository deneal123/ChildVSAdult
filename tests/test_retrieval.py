"""Тесты retrieval-метрик (офлайн, синтетические эмбеддинги)."""

from __future__ import annotations

import numpy as np

from age_gap.evaluation.retrieval import retrieval_metrics


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_perfect_retrieval():
    # Две личности по 2 почти-одинаковых лица -> Rank-1 = 1.0.
    emb = np.stack(
        [_unit([1, 0, 0]), _unit([0.99, 0.01, 0]), _unit([0, 1, 0]), _unit([0.01, 0.99, 0])]
    )
    ids = np.array([0, 0, 1, 1])
    m = retrieval_metrics(emb, ids, ks=(1, 2))
    assert m["n_queries"] == 4
    assert m["rank1"] == 1.0
    assert m["recall@1"] == 1.0
    assert m["mrr"] == 1.0
    assert m["median_rank"] == 1.0


def test_singletons_excluded():
    # Личность 2 имеет одно лицо -> не считается как query (нет релевантного).
    emb = np.stack([_unit([1, 0, 0]), _unit([0.99, 0.01, 0]), _unit([0, 0, 1])])
    ids = np.array([0, 0, 2])
    m = retrieval_metrics(emb, ids, ks=(1,))
    assert m["n_queries"] == 2  # только два лица личности 0


def test_relevant_at_rank2_gives_half_mrr():
    # Для каждого query релевантный — на 2-й позиции (между ними дистрактор ближе).
    # a0 ближе к distractor d, чем к своему a1; проверяем recall@2 и mrr.
    emb = np.stack(
        [
            _unit([1, 0, 0]),
            _unit([0.6, 0.8, 0]),
            _unit([0.95, 0.05, 0]),
        ]  # a0, a1, d(другая личность)
    )
    ids = np.array([0, 0, 9])
    m = retrieval_metrics(emb, ids, ks=(1, 2))
    # a0: ближайший — d (rank1, нерелевантен), затем a1 (rank2). a1: ближайший a0 (rank1).
    assert m["recall@2"] == 1.0
    assert 0.0 < m["mrr"] < 1.0


def test_empty():
    assert retrieval_metrics(np.zeros((1, 3), np.float32), np.array([0]))["n_queries"] == 0.0
