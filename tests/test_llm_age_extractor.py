"""Тесты LLM-экстрактора возраста (офлайн: парсинг + комбинирование, без сети)."""

from __future__ import annotations

from age_gap.common.schemas import AgeLabel
from age_gap.datasets.llm_age_extractor import (
    CachedAgeExtractor,
    CombinedAgeExtractor,
    LLMAgeExtractor,
    parse_llm_ages,
)


def test_parse_plain_json_array():
    content = '[{"age": 7, "photo_reference": "left", "confidence": 0.9}, {"age": 30, "photo_reference": "right", "confidence": 0.8}]'
    labels = parse_llm_ages(content)
    assert [(lbl.age, lbl.photo_reference, lbl.source) for lbl in labels] == [
        (7, "left", "llm"),
        (30, "right", "llm"),
    ]


def test_parse_json_in_markdown_fence():
    content = 'Вот результат:\n```json\n[{"age": 16, "photo_reference": "unknown"}]\n```'
    labels = parse_llm_ages(content)
    assert len(labels) == 1 and labels[0].age == 16 and labels[0].photo_reference == "unknown"


def test_parse_filters_invalid_age_and_ref():
    content = '[{"age": 200}, {"age": 25, "photo_reference": "bogus"}, {"age": "x"}]'
    labels = parse_llm_ages(content)
    # 200 вне диапазона и "x" не int — отброшены; bogus ref -> unknown.
    assert len(labels) == 1
    assert labels[0].age == 25 and labels[0].photo_reference == "unknown"


def test_parse_empty_and_garbage():
    assert parse_llm_ages("") == []
    assert parse_llm_ages("нет тут json") == []
    assert parse_llm_ages("[]") == []


def test_parse_position_reference():
    content = '[{"age": 7, "photo_reference": "position_2", "confidence": 0.8}]'
    labels = parse_llm_ages(content)
    assert labels[0].photo_reference == "position_2" and labels[0].age == 7


class _FakeConnector:
    """Заглушка GigaChat: возвращает заранее заданный ответ chat()."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def chat(self, messages, llm_config=None):  # noqa: ANN001
        self.calls += 1
        return self.reply


def test_llm_extractor_with_fake_connector():
    conn = _FakeConnector('[{"age": 14, "photo_reference": "first", "confidence": 0.7}]')
    ext = LLMAgeExtractor(connector=conn)
    labels = ext.extract("сложная подпись про детство и зрелость")
    assert labels and labels[0].age == 14 and labels[0].source == "llm"
    assert conn.calls == 1


def test_llm_extractor_empty_caption_skips_network():
    conn = _FakeConnector("[]")
    assert LLMAgeExtractor(connector=conn).extract("   ") == []
    assert conn.calls == 0


def test_combined_prefers_regex_no_llm_call():
    conn = _FakeConnector('[{"age": 99}]')
    combined = CombinedAgeExtractor(llm_extractor=LLMAgeExtractor(connector=conn))
    labels = combined.extract("мне 25")  # regex справляется
    assert [lbl.age for lbl in labels] == [25]
    assert conn.calls == 0  # LLM не вызывался


def test_combined_falls_back_to_llm():
    conn = _FakeConnector('[{"age": 22, "photo_reference": "unknown", "confidence": 0.6}]')
    combined = CombinedAgeExtractor(llm_extractor=LLMAgeExtractor(connector=conn))
    # regex не извлечёт возраст из такой формулировки
    labels = combined.extract("тогда был совсем юным студентом")
    assert conn.calls == 1
    assert labels and labels[0].age == 22 and labels[0].source == "llm"


def test_combined_calls_llm_when_regex_unmappable_with_photos():
    # regex даёт один возраст уровня поста (unknown); фото 3 -> не привяжется -> зовём LLM.
    conn = _FakeConnector('[{"age": 25, "photo_reference": "position_0", "confidence": 0.7}]')
    combined = CombinedAgeExtractor(llm_extractor=LLMAgeExtractor(connector=conn))
    labels = combined.extract("мне 25", n_photos=3)
    assert conn.calls == 1
    assert labels[0].photo_reference == "position_0" and labels[0].source == "llm"


def test_combined_skips_llm_when_counts_match():
    # regex даёт 2 возраста, фото 2 -> позиционный фолбэк сработает -> LLM не нужен.
    conn = _FakeConnector("[]")
    combined = CombinedAgeExtractor(llm_extractor=LLMAgeExtractor(connector=conn))
    combined.extract("здесь 16 лет, а здесь 26 лет", n_photos=2)
    assert conn.calls == 0


def test_combined_returns_agelabel_type():
    combined = CombinedAgeExtractor(llm_extractor=LLMAgeExtractor(connector=_FakeConnector("[]")))
    labels = combined.extract("25 лет")
    assert all(isinstance(x, AgeLabel) for x in labels)


def test_cached_extractor_hit_returns_cached_labels():
    cache = {("6/18/21", 3): [AgeLabel(age=6, photo_reference="position_0", source="llm")]}
    ext = CachedAgeExtractor(cache)
    labels = ext.extract("6/18/21", 3)
    assert ext.hits == 1 and ext.misses == 0
    assert labels[0].age == 6 and labels[0].source == "llm"


def test_cached_extractor_miss_uses_fallback_regex():
    ext = CachedAgeExtractor({})  # пустой кэш -> fallback regex
    labels = ext.extract("мне 25")
    assert ext.misses == 1 and [lbl.age for lbl in labels] == [25]
