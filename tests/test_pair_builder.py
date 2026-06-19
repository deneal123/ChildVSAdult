"""Тесты генерации пар (TODO §6–7 / SKILL §12)."""

from __future__ import annotations

from age_gap.common.schemas import AgeLabel, IdentityGroup
from age_gap.datasets.pair_builder import build_negative_pairs, build_positive_pairs


def _group(gid: str, faces, ages=None, status="auto") -> IdentityGroup:
    age_labels = []
    if ages:
        age_labels = [AgeLabel(face_id=f, age=a) for f, a in ages.items()]
    return IdentityGroup(
        identity_group_id=gid,
        source_post_id=gid,
        faces=faces,
        age_labels=age_labels,
        status=status,
    )


def test_positive_combinations_count():
    g = _group("g1", ["a", "b", "c"], ages={"a": 5, "b": 15, "c": 30})
    pos = build_positive_pairs([g])
    assert len(pos) == 3  # C(3,2)
    assert all(p.label == 1 and p.pair_type == "positive_same_post" for p in pos)


def test_positive_age_gap():
    g = _group("g1", ["a", "b"], ages={"a": 7, "b": 31})
    (p,) = build_positive_pairs([g])
    assert p.age_gap == 24


def test_positive_age_gap_none_when_unknown():
    g = _group("g1", ["a", "b"], ages={"a": 7})  # возраст b неизвестен
    (p,) = build_positive_pairs([g])
    assert p.age_gap is None


def test_manual_review_group_has_no_positives():
    g = _group("g1", ["a", "b"], ages={"a": 5, "b": 30}, status="manual_review_required")
    assert build_positive_pairs([g]) == []


def test_single_face_group_has_no_positives():
    g = _group("g1", ["a"])
    assert build_positive_pairs([g]) == []


def test_negatives_are_cross_group():
    groups = [
        _group("g1", ["a1", "a2"], ages={"a1": 10, "a2": 20}),
        _group("g2", ["b1", "b2"], ages={"b1": 11, "b2": 40}),
    ]
    negs = build_negative_pairs(groups, n_per_positive=2, n_positives=2, seed=1)
    assert negs, "ожидались негативы"
    for p in negs:
        assert p.label == 0
        assert p.identity_group_a != p.identity_group_b


def test_negatives_deterministic():
    groups = [
        _group("g1", ["a1", "a2"], ages={"a1": 10, "a2": 20}),
        _group("g2", ["b1", "b2"], ages={"b1": 11, "b2": 40}),
    ]
    r1 = [p.pair_id for p in build_negative_pairs(groups, 2, 2, seed=7)]
    r2 = [p.pair_id for p in build_negative_pairs(groups, 2, 2, seed=7)]
    assert r1 == r2


def test_age_matched_negatives_same_bucket():
    # Бакеты: 6-12, 18-25, 26-35 (см. _AGE_BUCKETS). a1/b1 — дети; a2/b2 — взрослые.
    from age_gap.datasets.pair_builder import _age_bucket

    groups = [
        _group("g1", ["a1", "a2"], ages={"a1": 8, "a2": 30}),
        _group("g2", ["b1", "b2"], ages={"b1": 10, "b2": 33}),
    ]
    negs = build_negative_pairs(groups, 5, 10, seed=1, age_matched=True)
    assert negs, "ожидались age-matched негативы"
    for p in negs:
        assert p.identity_group_a != p.identity_group_b
        assert _age_bucket(p.age_a) == _age_bucket(p.age_b)  # один возрастной бакет
        assert p.pair_type == "negative_age_controlled"
