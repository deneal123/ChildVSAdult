"""LLM-валидация целостности поста: один человек в разном возрасте vs шум (Part 2b).

Мультифото-пост — валидный «позитив», только если на всех фото ОДИН И ТОТ ЖЕ человек в разном
возрасте. Часть постов — мульти-человек («я с сестрой»), коллажи, мемы → шумные позитивы. LLM
(GigaChat) классифицирует пост по подписи; парсинг ответа вынесен в чистую ``parse_validation``
для офлайн-тестов. Async-проход и кэш — в ``scripts/validate_groups.py`` (как audit_ages).
"""

from __future__ import annotations

import json

from age_gap.common.logging import get_logger

log = get_logger(__name__)

CATEGORIES = ("single", "multi_person", "collage", "meme", "unknown")
# Категории, дающие ШУМНЫЕ позитивы (faces разных людей/не-портреты в одной группе).
NOISY = ("multi_person", "collage", "meme")

SYSTEM_PROMPT = (
    "Ты классифицируешь мультифото-пост из соцсети про возраст (русский язык). Определи по "
    "подписи, показывает ли пост ОДНОГО И ТОГО ЖЕ человека в разном возрасте или что-то другое. "
    "Верни СТРОГО JSON-объект без пояснений: "
    '{"category": <single|multi_person|collage|meme|unknown>, "confidence": <0..1>, '
    '"reason": "<кратко по-русски>"}. Категории: '
    '"single" — один человек на всех фото в разном возрасте («я в 5 и в 25», «тогда/сейчас», '
    "числа-возрасты через слэш «6/18/21»); "
    '"multi_person" — несколько РАЗНЫХ людей («я с сестрой», «мама и дочь», «на юбилее: 70, 40, 18»); '
    '"collage" — коллаж/монтаж/нарезка из многих фото; '
    '"meme" — мем/шутка/картинка из интернета, не личные портреты; '
    '"unknown" — по подписи невозможно определить. '
    "Опирайся на СМЫСЛ подписи, а не на число фото."
)


def build_user_message(caption: str, n_photos: int | None, n_faces: int | None = None) -> str:
    parts = []
    if n_photos:
        parts.append(f"Фото: {n_photos}")
    if n_faces is not None:
        parts.append(f"распознано лиц: {n_faces}")
    head = (", ".join(parts) + ".\n") if parts else ""
    return f"{head}Подпись: {caption}"


def _first_json_object(s: str) -> str | None:
    """Первый сбалансированный {...}-объект (игнорирует лишние хвостовые скобки, напр. «{...}}»)."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def parse_validation(content: str) -> dict | None:
    """Распарсить ответ LLM в {category, confidence, reason}; None если невалидно."""
    if not content:
        return None
    blob = _first_json_object(content)
    if blob is None:
        return None
    try:
        obj = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        log.warning("LLM вернул невалидный JSON: %r", content[:200])
        return None
    if not isinstance(obj, dict):
        return None
    cat = obj.get("category")
    if cat not in CATEGORIES:
        return None
    try:
        conf = min(1.0, max(0.0, float(obj.get("confidence", 0.6))))
    except (TypeError, ValueError):
        conf = 0.6
    reason = obj.get("reason")
    return {
        "category": cat,
        "confidence": conf,
        "reason": str(reason)[:200] if reason is not None else "",
    }
