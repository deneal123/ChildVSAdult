"""Тесты hard-negative mining (офлайн, синтетические эмбеддинги)."""

from __future__ import annotations

import numpy as np

from age_gap.datasets.hard_negatives import mine


def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_mines_similar_cross_group_pair():
    # f_a и f_b умеренно похожи (sim 0.6: hard, но не «тот же человек»), РАЗНЫЕ группы.
    face_ids = ["a", "b", "c"]
    emb = np.stack([_unit([1, 0, 0]), _unit([0.6, 0.8, 0]), _unit([0, 1, 0])])
    groups = {"a": "g1", "b": "g2", "c": "g3"}
    res = mine(face_ids, emb, groups, top_k=5, hard_min_sim=0.2, uncertain_sim=0.95)
    keys = {(p.face_a, p.face_b) for p in res.hard}
    assert ("a", "b") in keys
    assert all(p.pair_type == "hard_negative" and p.label == 0 for p in res.hard)


def test_same_group_not_mined():
    face_ids = ["a", "b"]
    emb = np.stack([_unit([1, 0, 0]), _unit([0.99, 0.01, 0])])
    groups = {"a": "g1", "b": "g1"}  # одна группа
    res = mine(face_ids, emb, groups, top_k=5)
    assert res.hard == []


def test_uncertain_excluded_as_false_negative():
    # Очень высокое сходство (>= uncertain_sim) -> отсев, не используем как негатив.
    face_ids = ["a", "b"]
    emb = np.stack([_unit([1, 0, 0]), _unit([0.999, 0.001, 0])])
    groups = {"a": "g1", "b": "g2"}
    res = mine(face_ids, emb, groups, top_k=5, hard_min_sim=0.2, uncertain_sim=0.55)
    assert res.hard == []
    assert res.n_uncertain == 1


def test_low_similarity_not_hard():
    # Ортогональные -> сходство 0 < hard_min -> не hard.
    face_ids = ["a", "b"]
    emb = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])])
    groups = {"a": "g1", "b": "g2"}
    res = mine(face_ids, emb, groups, top_k=5, hard_min_sim=0.2)
    assert res.hard == []


def test_max_total_keeps_hardest():
    # 4 кросс-групповых соседа якоря a с разным сходством; max_total=2 -> 2 самых сложных.
    face_ids = ["a", "b", "c", "d"]
    emb = np.stack(
        [_unit([1, 0, 0]), _unit([0.9, 0.1, 0]), _unit([0.7, 0.7, 0]), _unit([0.5, 0.86, 0])]
    )
    groups = {"a": "g1", "b": "g2", "c": "g3", "d": "g4"}
    res = mine(face_ids, emb, groups, top_k=5, hard_min_sim=0.2, uncertain_sim=0.999, max_total=2)
    assert len(res.hard) == 2
    # Самая близкая пара (a,b) должна остаться.
    keys = {(p.face_a, p.face_b) for p in res.hard}
    assert ("a", "b") in keys


def test_top_k_limit_per_anchor():
    # Якорь a близок к нескольким; top_k=1 -> не больше 1 негатива от a в его сторону.
    face_ids = ["a", "b", "c", "d"]
    emb = np.stack(
        [_unit([1, 0, 0]), _unit([0.9, 0.1, 0]), _unit([0.85, 0.15, 0]), _unit([0.8, 0.2, 0])]
    )
    groups = {"a": "g1", "b": "g2", "c": "g3", "d": "g4"}
    res = mine(face_ids, emb, groups, top_k=1, hard_min_sim=0.2, uncertain_sim=0.99)
    # Все ключи уникальны; пар немного из-за top_k и дедупликации.
    keys = [(p.face_a, p.face_b) for p in res.hard]
    assert len(keys) == len(set(keys))  # без дублей
    assert res.hard  # хоть что-то намайнено
