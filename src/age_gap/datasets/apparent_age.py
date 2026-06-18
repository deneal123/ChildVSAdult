"""Извлечение *воспринимаемого* возраста (apparent age) из комментариев (SKILL §8.4).

Это ВСПОМОГАТЕЛЬНЫЙ слабый сигнал, не основной признак идентичности. Комментарии вида
«выглядит на 20», «дал бы 25», «на вид лет 30», «максимум 18» дают оценку возраста на вид.
Разница между annotated age (подпись) и apparent age (комментарии) полезна для
age-disentanglement / apparent-age регуляризации (см. docs/deep-research.md).

Чистые regex-функции — тестируются офлайн.
"""

from __future__ import annotations

import re

MIN_AGE = 5
MAX_AGE = 90

# Паттерны воспринимаемого возраста (рус.).
_PATTERNS = [
    re.compile(r"выгляд\w+\s+на\s+(\d{1,2})"),  # выглядит/выглядишь на 20
    re.compile(r"да[лheё]\w*\s+бы\s+(?:лет\s+)?(\d{1,2})"),  # дал бы / дала бы (лет) 25
    re.compile(r"на\s+вид\s+(?:лет\s+)?(\d{1,2})"),  # на вид (лет) 30
    re.compile(r"максимум\s+(\d{1,2})"),  # максимум 18
    re.compile(r"не\s+больше\s+(\d{1,2})"),  # не больше 22
    re.compile(r"не\s+дашь?\s+(?:и\s+)?(\d{1,2})"),  # не дашь и 18
]


def extract_apparent_ages(text: str) -> list[int]:
    """Все воспринимаемые возрасты из одного комментария (валидные 5..90, без дублей)."""
    if not text:
        return []
    t = text.lower().replace("ё", "е")
    ages: list[int] = []
    for rx in _PATTERNS:
        for m in rx.finditer(t):
            age = int(m.group(1))
            if MIN_AGE <= age <= MAX_AGE and age not in ages:
                ages.append(age)
    return ages


def collect_post_apparent_ages(comments: list[dict]) -> list[int]:
    """Собрать apparent-age по всем комментариям поста (текст в поле ``text``)."""
    ages: list[int] = []
    for c in comments:
        ages.extend(extract_apparent_ages(str(c.get("text", ""))))
    return ages
