"""Обезличивание для релиза (DATA_GOVERNANCE §4): хеш-id, страйп PII, только производное.

НЕ экспортирует сырые лица/подписи/VK-id. Идентификаторы → HMAC-SHA256 с СЕКРЕТНОЙ солью (соль не
публикуется → обратная связь с VK-профилем невозможна). Несовершеннолетние (apparent age < 18)
исключаются из релиза. Подписи/комментарии (PII) не экспортируются вовсе. См. docs/DATA_GOVERNANCE.md.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

from age_gap.common.schemas import IdentityGroup, Pair

MINOR_AGE = 18


def hash_id(value: str, salt: bytes, length: int = 16) -> str:
    """Псевдонимный стабильный id: HMAC-SHA256(salt, value), усечённый hex. Необратим без соли."""
    return hmac.new(salt, value.encode("utf-8"), hashlib.sha256).hexdigest()[:length]


def age_bucket(age: int | None) -> str:
    """Возрастной бакет (огрубление точного возраста для минимизации)."""
    if age is None:
        return "unknown"
    if age < 18:
        return "0-17"
    if age < 30:
        return "18-29"
    if age < 45:
        return "30-44"
    return "45+"


def anon_group(g: IdentityGroup, salt: bytes, minor_faces: set[str]) -> dict[str, Any] | None:
    """Обезличенная группа: хеш person-id + хеш лиц (без несовершеннолетних), бакет возраста.

    Возвращает None, если после исключения несовершеннолетних не осталось лиц.
    """
    faces = [f for f in g.faces if f not in minor_faces]
    if not faces:
        return None
    return {
        "person_id": hash_id(g.identity_group_id, salt),
        "faces": [hash_id(f, salt) for f in faces],
        "apparent_gender": g.apparent_gender,
        "apparent_age_bucket": age_bucket(g.apparent_age),
        "identity_review": g.identity_review,
    }


def anon_pair(p: Pair, salt: bytes, exclude_faces: set[str]) -> dict[str, Any] | None:
    """Обезличенная пара: хеш id лиц, метка, age_gap, split. None, если задействовано исключённое лицо."""
    if p.face_a in exclude_faces or p.face_b in exclude_faces:
        return None
    return {
        # pair_id восстанавливается из хешей (исходный pair_id содержит сырые face_id).
        "pair_id": f"{p.label}_{hash_id(p.face_a, salt, 12)}_{hash_id(p.face_b, salt, 12)}",
        "face_a": hash_id(p.face_a, salt),
        "face_b": hash_id(p.face_b, salt),
        "label": p.label,
        "age_gap": p.age_gap,
        "pair_type": p.pair_type,
        "split": p.split,
    }
