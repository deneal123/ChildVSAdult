"""Тесты near-duplicate дедупа (Part 3) — чистая логика на синтетических эмбеддингах."""

from __future__ import annotations

import numpy as np

from age_gap.common.schemas import IdentityGroup
from age_gap.datasets.dedup import find_redundant_faces


def _n(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=float)
    return a / np.linalg.norm(a)


def _group(faces: list[str]) -> IdentityGroup:
    return IdentityGroup(identity_group_id="g1", source_post_id="p1", faces=faces)


def test_dedup_marks_near_duplicate():
    emb = {"a": _n([1, 0, 0, 0]), "b": _n([1, 0.02, 0, 0]), "c": _n([0, 1, 0, 0])}
    red = find_redundant_faces([_group(["a", "b", "c"])], emb, threshold=0.97)
    assert red == {"g1": ["b"]}  # a≈b -> оставляем «a», «b» избыточен; «c» отличается


def test_dedup_all_distinct_empty():
    emb = {"a": _n([1, 0, 0]), "b": _n([0, 1, 0]), "c": _n([0, 0, 1])}
    assert find_redundant_faces([_group(["a", "b", "c"])], emb, threshold=0.97) == {}


def test_dedup_skips_singleton():
    assert find_redundant_faces([_group(["a"])], {"a": _n([1, 0])}, threshold=0.97) == {}


def test_dedup_triplet_all_duplicates():
    emb = {"a": _n([1, 0]), "b": _n([1, 0.01]), "c": _n([1, 0.02])}
    red = find_redundant_faces([_group(["a", "b", "c"])], emb, threshold=0.97)
    assert red == {"g1": ["b", "c"]}  # все дубликаты -> остаётся один «a»
