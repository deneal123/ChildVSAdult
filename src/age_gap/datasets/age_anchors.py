"""Извлечение возрастных якорей из подписей (SKILL §8.2 / TODO §8).

Сейчас реализован детерминированный regex-экстрактор для русских формулировок.
Интерфейс ``AgeExtractor`` (Protocol) оставляет место под LLM-экстрактор на Phase 4
(``LLMAgeExtractor`` — пока заглушка), не добавляя зависимости от LLM в MVP-1.

Приоритет (TODO §8):
    1) явная привязка слева/справа;
    2) упорядоченное перечисление («в 12 и сейчас в 28») -> first/second;
    3) одиночный возраст уровня поста (слабая привязка).
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from age_gap.common.schemas import AgeLabel

MIN_AGE = 1
MAX_AGE = 99

# Возраст с единицей измерения, но НЕ «N лет назад» (это разрыв, а не возраст).
_AGE_UNIT = r"(\d{1,2})\s*(?:лет|год(?:а|ик|иков)?)(?!\s*назад)"
# Возраст без единицы после «мне»/«в»: «мне 25», «в 7».
# Возраст без единицы после «мне»/«в», допускается «было»: «мне 25», «в 7», «мне было 14».
_AGE_BARE = r"(?:мне|в)\s*(?:было\s*)?(\d{1,2})\b"

# Явная привязка слева/справа: «слева 5(,| лет) справа 30».
_LEFT_AGE = re.compile(r"слев[ае]\D{0,15}?(\d{1,2})")
_RIGHT_AGE = re.compile(r"справ[ае]\D{0,15}?(\d{1,2})")

# Упорядоченное: «в 12 и (сейчас )?в 28», «в 12 и мне 28».
_ORDERED = re.compile(r"в\s*(\d{1,2})\b\D{0,25}?(?:и|,)\s*(?:сейчас\s*)?(?:в|мне)\s*(\d{1,2})\b")

# Слэш-перечисление возрастов по числу фото: «6/18/21», «5/26» (формат паблика «Запах
# минувших дней»). Вся подпись = последовательность 1-2-значных чисел через «/». Засчитываем
# только если число чисел совпадает с числом фото (отсекает даты: 18/06/2021 -> 2021 невалиден).
_SLASH_SEQ = re.compile(r"\s*(\d{1,2}(?:\s*/\s*\d{1,2})+)\s*")

_AGE_UNIT_RE = re.compile(_AGE_UNIT)
_AGE_BARE_RE = re.compile(_AGE_BARE)


def _valid(age: int) -> bool:
    return MIN_AGE <= age <= MAX_AGE


@runtime_checkable
class AgeExtractor(Protocol):
    def extract(self, caption: str, n_photos: int | None = None) -> list[AgeLabel]:
        """Извлечь возрасты. n_photos — число фото в посте (помогает LLM расставить позиции)."""
        ...


class RegexAgeExtractor:
    """Извлечение возраста на основе regex-паттернов (source="caption_regex")."""

    def extract(self, caption: str, n_photos: int | None = None) -> list[AgeLabel]:
        if not caption:
            return []
        text = caption.lower().replace("ё", "е")

        # Слэш-перечисление по числу фото — однозначная позиционная привязка, высший приоритет.
        labels = self._slash_sequence(text, n_photos)
        if labels:
            return labels

        labels = self._left_right(text)
        if labels:
            return labels

        labels = self._ordered(text)
        if labels:
            return labels

        return self._singles(text)

    def _slash_sequence(self, text: str, n_photos: int | None) -> list[AgeLabel]:
        if not n_photos or n_photos < 2:
            return []
        m = _SLASH_SEQ.fullmatch(text)
        if not m:
            return []
        ages = [int(x) for x in m.group(1).split("/")]
        # Засчитываем только при точном совпадении количества чисел и фото и валидности всех.
        if len(ages) != n_photos or not all(_valid(a) for a in ages):
            return []
        return [
            AgeLabel(
                age=a,
                photo_reference=f"position_{i}",
                source="caption_regex",
                confidence=0.85,
                mapping_confidence=0.85,
            )
            for i, a in enumerate(ages)
        ]

    def _left_right(self, text: str) -> list[AgeLabel]:
        lm = _LEFT_AGE.search(text)
        rm = _RIGHT_AGE.search(text)
        labels: list[AgeLabel] = []
        if lm and _valid(int(lm.group(1))):
            labels.append(self._label(int(lm.group(1)), "left", 0.9, 0.9))
        if rm and _valid(int(rm.group(1))):
            labels.append(self._label(int(rm.group(1)), "right", 0.9, 0.9))
        # Нужны обе стороны, иначе это не явная парная привязка.
        return labels if len(labels) == 2 else []

    def _ordered(self, text: str) -> list[AgeLabel]:
        m = _ORDERED.search(text)
        if not m:
            return []
        a, b = int(m.group(1)), int(m.group(2))
        if not (_valid(a) and _valid(b)):
            return []
        return [
            self._label(a, "first", 0.8, 0.7),
            self._label(b, "second", 0.8, 0.7),
        ]

    def _singles(self, text: str) -> list[AgeLabel]:
        # Собираем (позиция, возраст) по обоим паттернам и сортируем по позиции в тексте,
        # чтобы порядок возрастов соответствовал порядку упоминания (TODO §8, правило 2).
        found: list[tuple[int, int]] = []
        for rx in (_AGE_UNIT_RE, _AGE_BARE_RE):
            for m in rx.finditer(text):
                age = int(m.group(1))
                if _valid(age):
                    found.append((m.start(1), age))
        found.sort(key=lambda x: x[0])

        ages: list[int] = []
        for _, age in found:
            if age not in ages:
                ages.append(age)
        # Одиночный возраст уровня поста — слабая привязка к фото (photo_reference=unknown).
        # Порядок в списке сохраняет порядок упоминания: positional-маппинг делает
        # identity_groups, когда число возрастов совпадает с числом лиц.
        conf = 0.6 if len(ages) == 1 else 0.4
        return [self._label(a, "unknown", conf, 0.2) for a in ages]

    @staticmethod
    def _label(age: int, ref: str, conf: float, mapping_conf: float) -> AgeLabel:
        return AgeLabel(
            age=age,
            photo_reference=ref,
            source="caption_regex",
            confidence=conf,
            mapping_confidence=mapping_conf,
        )
