"""LLM-экстрактор возраста из подписей через GigaChat (TODO §12 Phase 4 / §8).

Дополняет regex для сложных формулировок («мне тогда было чуть за двадцать», «школьником
и уже отцом» и т.п.). Возвращает те же AgeLabel, что и regex, поэтому маппинг к лицам в
identity_groups (по photo_reference / позиции) работает без изменений.

Парсинг ответа вынесен в чистую функцию ``parse_llm_ages`` — её можно тестировать офлайн.
"""

from __future__ import annotations

import json
import re

from age_gap.common.logging import get_logger
from age_gap.common.schemas import AgeLabel
from age_gap.datasets.age_anchors import AgeExtractor, RegexAgeExtractor

log = get_logger(__name__)

MIN_AGE = 1
MAX_AGE = 99
_BASE_REFS = {"first", "second", "left", "right", "unknown"}
_POSITION_RE = re.compile(r"^position_\d+$")


def _normalize_ref(ref: object) -> str:
    """Допустимые ссылки: базовые слова + position_N; иначе unknown."""
    if isinstance(ref, str) and (ref in _BASE_REFS or _POSITION_RE.match(ref)):
        return ref
    return "unknown"


SYSTEM_PROMPT = (
    "Ты извлекаешь возраст человека из подписи к мультифото-посту (русский язык). "
    "Верни СТРОГО JSON-массив объектов без пояснений. Каждый объект: "
    '{"age": <целое 1..99>, "photo_reference": <см. ниже>, "confidence": <0..1>}. '
    "photo_reference — к какому фото относится возраст: "
    '"position_0" (первое фото), "position_1" (второе), "position_2" и т.д.; '
    'либо "unknown", если из подписи нельзя понять, к какому фото. '
    "Синонимы: «слева/сначала/в детстве» → раннее фото (меньший индекс), "
    "«справа/сейчас/теперь» → позднее фото (больший индекс). "
    "ВАЖНО: если подпись — это числа через слэш или дефис (например «6/18/21» при 3 фото, "
    "«5/26» или «5-26» при 2 фото), это ВОЗРАСТЫ человека по одному на каждое фото по порядку "
    "(position_0, position_1, ...), а НЕ дата. Так часто подписывают фото в разном возрасте. "
    "Извлекай ТОЛЬКО возраст человека на фото (не годы съёмки, не стаж/срок брака, "
    "не «N лет назад»). Если возраста нет — верни []. "
    "Число возрастов не обязано совпадать с числом фото."
)


def _build_user_message(caption: str, n_photos: int | None) -> str:
    if n_photos and n_photos > 0:
        return f"Фото в посте: {n_photos} (индексы position_0..position_{n_photos - 1}).\nПодпись: {caption}"
    return f"Подпись: {caption}"


# Достаём JSON-массив, даже если модель обернула его в ```json ... ``` или текст.
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_llm_ages(content: str) -> list[AgeLabel]:
    """Распарсить ответ модели в список AgeLabel. Невалидные элементы пропускаются."""
    if not content:
        return []
    match = _JSON_ARRAY_RE.search(content)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        log.warning("LLM вернул невалидный JSON: %r", content[:200])
        return []
    if not isinstance(items, list):
        return []

    labels: list[AgeLabel] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        age = item.get("age")
        if not isinstance(age, int) or not (MIN_AGE <= age <= MAX_AGE):
            continue
        ref = _normalize_ref(item.get("photo_reference", "unknown"))
        try:
            conf = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            conf = 0.6
        conf = min(1.0, max(0.0, conf))
        labels.append(
            AgeLabel(
                age=age,
                photo_reference=ref,
                source="llm",
                confidence=conf,
                mapping_confidence=0.6 if ref != "unknown" else 0.2,
            )
        )
    return labels


class LLMAgeExtractor:
    """Извлечение возраста через GigaChat (source="llm"). Совместим с AgeExtractor."""

    def __init__(self, connector: object | None = None) -> None:
        self._connector = connector  # для тестов можно передать заглушку

    def _get_connector(self) -> object:
        if self._connector is None:
            from age_gap.infrastructure.llm.client import get_client

            self._connector = get_client()
        return self._connector

    def extract(self, caption: str, n_photos: int | None = None) -> list[AgeLabel]:
        if not caption or not caption.strip():
            return []
        from age_gap.infrastructure.llm.base import run_coro_blocking

        connector = self._get_connector()
        messages = [
            ("system", SYSTEM_PROMPT),
            ("user", _build_user_message(caption.strip(), n_photos)),
        ]
        try:
            content = run_coro_blocking(connector.chat(messages))  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — сетевые/API сбои не должны валить пайплайн
            log.warning("LLM age extraction не удалась: %s", exc)
            return []
        return parse_llm_ages(content)


def _regex_sufficient(labels: list[AgeLabel], n_photos: int | None) -> bool:
    """Достаточен ли результат regex, чтобы НЕ звать LLM.

    Звать LLM стоит, только когда он реально поможет привязать возрасты к лицам:
    - regex пуст → LLM может извлечь из сложной подписи;
    - есть >=2 фото, все якоря unknown и число возрастов != числу фото → LLM расставит позиции.
    Иначе regex достаточно (есть явная ссылка; либо позиционный фолбэк сработает; либо
    привязка к парам всё равно невозможна при <2 фото / неизвестном числе фото).
    """
    if not labels:
        return False
    if any(lbl.photo_reference != "unknown" for lbl in labels):
        return True
    if n_photos is None or n_photos < 2:
        return True
    return len(labels) == n_photos  # сработает позиционный фолбэк


class CombinedAgeExtractor:
    """Regex как основной (дёшево, точно), LLM — для сложных/непривязываемых подписей.

    Если regex дал результат, который реально привяжется к лицам, — отдаём regex без сети.
    Иначе зовём LLM (он знает число фото и может расставить позиции). Это закрывает главную
    потерю — возрасты уровня поста (unknown), которые не маппятся на лица.
    """

    def __init__(
        self,
        regex_extractor: AgeExtractor | None = None,
        llm_extractor: AgeExtractor | None = None,
    ) -> None:
        self.regex: AgeExtractor = regex_extractor or RegexAgeExtractor()
        self.llm: AgeExtractor = llm_extractor or LLMAgeExtractor()

    def extract(self, caption: str, n_photos: int | None = None) -> list[AgeLabel]:
        labels = self.regex.extract(caption, n_photos)
        if _regex_sufficient(labels, n_photos):
            return labels
        llm_labels = self.llm.extract(caption, n_photos)
        # Если LLM ничего не дал — возвращаем хотя бы regex (возраст уровня поста полезен).
        return llm_labels or labels


class CachedAgeExtractor:
    """Отдаёт заранее посчитанные (LLM) метки из кэша; на промах — fallback (regex).

    Позволяет пересобрать identity_groups на LLM-возрастах БЕЗ повторных сетевых вызовов:
    дорогой LLM-проход делается один раз (scripts/audit_ages.py) и кэшируется, а build_groups
    читает готовые метки. Ключ кэша — (нормализованная подпись, число фото).
    """

    def __init__(
        self,
        cache: dict[tuple[str, int | None], list[AgeLabel]],
        fallback: AgeExtractor | None = None,
    ) -> None:
        self._cache = cache
        self._fallback: AgeExtractor = fallback or RegexAgeExtractor()
        self.hits = 0
        self.misses = 0

    def extract(self, caption: str, n_photos: int | None = None) -> list[AgeLabel]:
        key = ((caption or "").strip(), n_photos)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return self._fallback.extract(caption, n_photos)

    @classmethod
    def from_file(cls, path: str, fallback: AgeExtractor | None = None) -> CachedAgeExtractor:
        from age_gap.common.io import read_jsonl

        cache: dict[tuple[str, int | None], list[AgeLabel]] = {}
        for row in read_jsonl(path):
            key = ((row.get("caption") or "").strip(), row.get("n_photos"))
            cache[key] = [AgeLabel.from_dict(d) for d in row.get("labels", [])]
        return cls(cache, fallback=fallback)


def make_age_extractor(kind: str = "regex") -> AgeExtractor:
    """Фабрика экстрактора возраста по имени: regex | llm | combined."""
    kind = kind.lower()
    if kind == "regex":
        return RegexAgeExtractor()
    if kind == "llm":
        return LLMAgeExtractor()
    if kind == "combined":
        return CombinedAgeExtractor()
    raise ValueError(f"Неизвестный age-extractor: {kind!r} (regex|llm|combined)")
