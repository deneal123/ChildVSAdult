"""Тесты LLM-валидатора целостности групп (Part 2b) — парсинг офлайн, без сети."""

from __future__ import annotations

from age_gap.datasets.group_validator import build_user_message, parse_validation


def test_parse_valid_single():
    r = parse_validation('{"category": "single", "confidence": 0.9, "reason": "я в 5 и 25"}')
    assert r is not None
    assert r["category"] == "single"
    assert r["confidence"] == 0.9


def test_parse_multi_person():
    r = parse_validation('{"category": "multi_person", "confidence": 0.8, "reason": "я с сестрой"}')
    assert r is not None and r["category"] == "multi_person"


def test_parse_wrapped_in_text():
    r = parse_validation('Вот ответ: {"category": "collage", "confidence": 0.7} спасибо')
    assert r is not None and r["category"] == "collage"


def test_parse_invalid_category_is_none():
    assert parse_validation('{"category": "selfie", "confidence": 0.9}') is None


def test_parse_non_json_is_none():
    assert parse_validation("это не json") is None
    assert parse_validation("") is None


def test_parse_trailing_extra_brace():
    # GigaChat иногда добавляет лишнюю «}» в конце — балансировщик берёт первый полный объект.
    r = parse_validation('{"category": "single", "confidence": 1.0, "reason": "9 и 29 лет"}}')
    assert r is not None and r["category"] == "single"


def test_parse_clamps_confidence():
    r = parse_validation('{"category": "meme", "confidence": 5}')
    assert r is not None and r["confidence"] == 1.0


def test_build_user_message_includes_caption_and_counts():
    msg = build_user_message("я в детстве и сейчас", n_photos=2, n_faces=2)
    assert "я в детстве и сейчас" in msg
    assert "2" in msg
