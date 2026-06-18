"""Тесты regex-экстрактора возраста (SKILL §8.2 / TODO §8)."""

from __future__ import annotations

from age_gap.datasets.age_anchors import RegexAgeExtractor


def _ages(caption: str) -> list[int | None]:
    return [lbl.age for lbl in RegexAgeExtractor().extract(caption)]


def test_single_mne():
    assert _ages("мне 25") == [25]


def test_single_let():
    assert _ages("25 лет") == [25]


def test_na_foto_mne():
    assert _ages("на фото мне 7") == [7]


def test_tut_mne_bylo():
    assert _ages("тут мне было 14") == [14]


def test_left_right_mapping():
    labels = RegexAgeExtractor().extract("слева 5 лет, справа 30")
    refs = {lbl.photo_reference: lbl.age for lbl in labels}
    assert refs == {"left": 5, "right": 30}


def test_ordered_first_second():
    labels = RegexAgeExtractor().extract("в 12 и сейчас в 28")
    refs = {lbl.photo_reference: lbl.age for lbl in labels}
    assert refs == {"first": 12, "second": 28}


def test_ordered_without_seychas():
    labels = RegexAgeExtractor().extract("в 12 и в 28")
    refs = {lbl.photo_reference: lbl.age for lbl in labels}
    assert refs == {"first": 12, "second": 28}


def test_years_ago_is_not_age():
    # «10 лет назад» — это разрыв, а не возраст: не должен извлекаться.
    assert 10 not in _ages("10 лет назад и сейчас")


def test_empty_caption():
    assert _ages("") == []
    assert _ages("красивое фото природы") == []


def test_slash_sequence_three_photos():
    # Формат паблика «Запах минувших дней»: возрасты по числу фото, позиционно.
    labels = RegexAgeExtractor().extract("6/18/21", n_photos=3)
    assert [(lbl.age, lbl.photo_reference) for lbl in labels] == [
        (6, "position_0"),
        (18, "position_1"),
        (21, "position_2"),
    ]


def test_slash_sequence_two_photos():
    labels = RegexAgeExtractor().extract("5/26", n_photos=2)
    assert [lbl.age for lbl in labels] == [5, 26]
    assert all(lbl.confidence >= 0.8 for lbl in labels)


def test_slash_count_must_match_photos():
    # Несовпадение числа возрастов и фото -> не засчитываем (защита от дат и ошибок).
    assert RegexAgeExtractor().extract("6/18/21", n_photos=2) == []


def test_slash_rejects_date_like():
    # 18/06/2021: 2021 невалидный возраст -> не slash-последовательность возрастов.
    assert RegexAgeExtractor().extract("18/06/2021", n_photos=3) == []


def test_slash_needs_n_photos():
    # Без n_photos слэш-формат не активируется (нельзя проверить совпадение).
    assert RegexAgeExtractor().extract("5/26") == []
