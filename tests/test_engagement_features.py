"""Guardrail-тесты сборки признаков (ветка engagement).

Эти тесты защищают ЭТИЧЕСКИЕ инварианты, а не «красоту кода»:
детские лица не должны попадать в признаки внешности, а несовершеннолетние
субъекты — в выборку вообще.
"""

from __future__ import annotations

import numpy as np

from age_gap.engagement.features import (
    ADULT_MIN_AGE,
    SUBJECT_MIN_AGE,
    _bbox_area,
    _text_features,
    minor_subject_ids,
    split_adult_child,
)


def _face(fid: str) -> dict:
    return {"face_id": fid, "is_usable": True, "det_score": 0.9, "face_quality_score": 0.8}


def test_child_faces_never_enter_appearance_features():
    faces = [_face("a"), _face("b"), _face("c")]
    ga = {"a": {"age_est": 7, "gender": 0}, "b": {"age_est": 30, "gender": 0}, "c": {"age_est": 17, "gender": 1}}
    adult, child = split_adult_child(faces, ga)

    adult_ids = {f["face_id"] for f, _ in adult}
    child_ids = {f["face_id"] for f, _ in child}
    assert adult_ids == {"b"}
    assert child_ids == {"a", "c"}
    # ключевой инвариант: ни одно лицо младше ADULT_MIN_AGE не попало во взрослые
    assert all(float(g["age_est"]) >= ADULT_MIN_AGE for _, g in adult)
    assert not (adult_ids & child_ids)


def test_face_without_age_estimate_is_dropped():
    faces = [_face("a"), _face("no_age")]
    adult, child = split_adult_child(faces, {"a": {"age_est": 25, "gender": 0}})
    assert [f["face_id"] for f, _ in adult] == ["a"]
    assert child == []


def test_minor_subjects_excluded_with_buffer():
    # буфер SUBJECT_MIN_AGE=20 против занижения оценщика возраста
    person_max_age = {"teen": 17.0, "edge": 19.9, "adult": 20.0, "older": 41.0}
    minors = minor_subject_ids(person_max_age)
    assert minors == {"teen", "edge"}
    assert SUBJECT_MIN_AGE > ADULT_MIN_AGE, "субъектный порог должен быть строже лицевого"


def test_adult_with_own_child_photo_is_kept():
    """Взрослый, выложивший своё детское фото, остаётся: субъект — взрослый."""
    person_max_age = {"p": 29.0}
    assert minor_subject_ids(person_max_age) == set()
    faces = [_face("then"), _face("now")]
    ga = {"then": {"age_est": 6, "gender": 0}, "now": {"age_est": 29, "gender": 0}}
    adult, child = split_adult_child(faces, ga)
    assert len(adult) == 1 and len(child) == 1  # детское фото учтено структурно, но не как внешность


def test_bbox_area():
    assert _bbox_area([0, 0, 10, 4]) == 40.0
    assert np.isnan(_bbox_area(None))
    assert np.isnan(_bbox_area([1, 2]))
    assert _bbox_area([10, 10, 0, 0]) == 0.0  # вырожденный bbox -> 0, не отрицательная площадь


def test_text_features():
    f = _text_features("Привет!! Как дела? #тогда #сейчас 🙂 МНЕ 25")
    assert f["n_exclam"] == 2
    assert f["n_questions"] == 1
    assert f["n_hashtags"] == 2
    assert f["has_emoji"] == 1 and f["emoji_count"] >= 1
    assert f["has_digit"] == 1
    assert 0.0 < f["upper_ratio"] <= 1.0

    empty = _text_features("")
    assert empty["caption_len"] == 0 and empty["upper_ratio"] == 0.0
