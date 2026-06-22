"""Тесты агрегации gender-метаданных (Part 1) — чистая логика, без моделей."""

from __future__ import annotations

from age_gap.common.schemas import IdentityGroup
from age_gap.datasets.gender_meta import aggregate_group


def test_aggregate_empty():
    assert aggregate_group(["a", "b"], {}) == (None, None, None)


def test_aggregate_unanimous():
    attrs = {"a": ("F", 20), "b": ("F", 30), "c": ("F", 40)}
    gender, age, cons = aggregate_group(["a", "b", "c"], attrs)
    assert gender == "F"
    assert age == 30  # медиана
    assert cons == 1.0


def test_aggregate_majority():
    attrs = {"a": ("M", 25), "b": ("M", 35), "c": ("F", 45), "d": ("M", 55)}
    gender, age, cons = aggregate_group(["a", "b", "c", "d"], attrs)
    assert gender == "M"
    assert cons == 0.75  # 3 из 4


def test_aggregate_skips_missing_faces():
    attrs = {"a": ("F", 22)}
    gender, age, cons = aggregate_group(["a", "missing"], attrs)
    assert gender == "F" and age == 22 and cons == 1.0


def test_identity_group_roundtrip_new_fields():
    g = IdentityGroup(
        identity_group_id="g1",
        source_post_id="p1",
        faces=["a"],
        apparent_gender="F",
        apparent_age=27,
        gender_consistency=0.8,
        identity_review="single",
    )
    g2 = IdentityGroup.from_dict(g.to_dict())
    assert g2.apparent_gender == "F"
    assert g2.apparent_age == 27
    assert g2.gender_consistency == 0.8
    assert g2.identity_review == "single"
