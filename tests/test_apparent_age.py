"""Тесты извлечения apparent-age из комментариев (офлайн)."""

from __future__ import annotations

from age_gap.datasets.apparent_age import collect_post_apparent_ages, extract_apparent_ages


def test_vyglyadit_na():
    assert extract_apparent_ages("выглядит на 20 максимум") == [20]


def test_dal_by():
    assert extract_apparent_ages("дал бы лет 25") == [25]
    assert extract_apparent_ages("дала бы 30") == [30]


def test_na_vid():
    assert extract_apparent_ages("на вид лет 35") == [35]


def test_ne_bolshe_and_max():
    assert 22 in extract_apparent_ages("не больше 22")
    assert 18 in extract_apparent_ages("максимум 18")


def test_no_age():
    assert extract_apparent_ages("красивое фото") == []
    assert extract_apparent_ages("") == []


def test_out_of_range_filtered():
    # 3 года — ниже MIN_AGE для apparent (воспринимаемый возраст реалистичен 5..90).
    assert extract_apparent_ages("на вид 3") == []


def test_collect_over_comments():
    comments = [{"text": "выглядит на 20"}, {"text": "дал бы 25"}, {"text": "ничего"}]
    ages = collect_post_apparent_ages(comments)
    assert sorted(ages) == [20, 25]
