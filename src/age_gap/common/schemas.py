"""Схемы данных конвейера.

Соответствуют docs/SKILL.md §10 и docs/TODO.md §5. Каждый dataclass сериализуется в
обычный dict (для JSONL) через ``to_dict`` и восстанавливается через ``from_dict``.

Provenance-поля (SKILL §9.4) обязательны для каждого исходного поста: они фиксируют
источник, права/согласие и политику хранения. Без них элемент не должен попадать в датасет.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Provenance:
    """Происхождение и правовой статус элемента данных (SKILL §9.4)."""

    source: str = "vk_public"
    origin_url: str | None = None
    collection_date: str | None = None  # YYYY-MM-DD
    license_or_consent_status: str = "unknown"
    allowed_use: str = "research"
    retention_policy: str = "review_required"
    deletion_status: str = "active"
    review_status: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Provenance:
        return cls(**{k: d[k] for k in _field_names(cls) if k in d})


@dataclass
class Comment:
    comment_id: str
    text: str = ""
    likes_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Comment:
        return cls(**{k: d[k] for k in _field_names(cls) if k in d})


@dataclass
class Photo:
    photo_id: str
    url: str | None = None
    local_path: str | None = None
    order: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Photo:
        return cls(**{k: d[k] for k in _field_names(cls) if k in d})


@dataclass
class RawPost:
    """Сырой пост (TODO §5 «Raw post»)."""

    post_id: str
    source: str = "vk_public"
    url: str | None = None
    caption: str = ""
    date: str | None = None  # YYYY-MM-DD
    photos: list[Photo] = field(default_factory=list)
    comments: list[Comment] = field(default_factory=list)
    likes_count: int = 0
    reposts_count: int = 0
    views_count: int | None = None
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "source": self.source,
            "url": self.url,
            "caption": self.caption,
            "date": self.date,
            "photos": [p.to_dict() for p in self.photos],
            "comments": [c.to_dict() for c in self.comments],
            "likes_count": self.likes_count,
            "reposts_count": self.reposts_count,
            "views_count": self.views_count,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RawPost:
        return cls(
            post_id=d["post_id"],
            source=d.get("source", "vk_public"),
            url=d.get("url"),
            caption=d.get("caption", ""),
            date=d.get("date"),
            photos=[Photo.from_dict(p) for p in d.get("photos", [])],
            comments=[Comment.from_dict(c) for c in d.get("comments", [])],
            likes_count=d.get("likes_count", 0),
            reposts_count=d.get("reposts_count", 0),
            views_count=d.get("views_count"),
            provenance=Provenance.from_dict(d.get("provenance", {})),
        )


@dataclass
class FaceCrop:
    """Кроп лица (TODO §5 «Face crop» / SKILL §10 «Face»)."""

    face_id: str
    photo_id: str
    bbox: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    landmarks: list[list[float]] = field(default_factory=list)
    face_crop_path: str | None = None
    face_quality_score: float = 0.0
    det_score: float = 0.0
    num_faces_in_image: int = 0
    is_usable: bool = False
    reject_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FaceCrop:
        return cls(**{k: d[k] for k in _field_names(cls) if k in d})


@dataclass
class AgeLabel:
    """Возрастной якорь (SKILL §8.2 / §10 «Age label»)."""

    face_id: str | None = None
    age: int | None = None
    photo_reference: str = "unknown"  # left/right/first/second/unknown
    source: str = "caption_regex"  # caption_regex / manual / llm
    confidence: float = 0.0
    mapping_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgeLabel:
        return cls(**{k: d[k] for k in _field_names(cls) if k in d})


@dataclass
class IdentityGroup:
    """Группа личности (TODO §5 / SKILL §8.1).

    status: auto / mixed / manual_verified / manual_review_required
    """

    identity_group_id: str
    source_post_id: str
    faces: list[str] = field(default_factory=list)
    age_labels: list[AgeLabel] = field(default_factory=list)
    status: str = "auto"
    apparent_gender: str | None = None  # "F"/"M": majority genderage по лицам группы
    apparent_age: int | None = None  # медианный apparent-возраст лиц группы (genderage)
    gender_consistency: float | None = None  # доля лиц с majority-полом (1.0 = все согласны)
    identity_review: str | None = None  # LLM-валидация: single | multi_person | collage | meme | unknown

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_group_id": self.identity_group_id,
            "source_post_id": self.source_post_id,
            "faces": list(self.faces),
            "age_labels": [a.to_dict() for a in self.age_labels],
            "status": self.status,
            "apparent_gender": self.apparent_gender,
            "apparent_age": self.apparent_age,
            "gender_consistency": self.gender_consistency,
            "identity_review": self.identity_review,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IdentityGroup:
        return cls(
            identity_group_id=d["identity_group_id"],
            source_post_id=d["source_post_id"],
            faces=list(d.get("faces", [])),
            age_labels=[AgeLabel.from_dict(a) for a in d.get("age_labels", [])],
            status=d.get("status", "auto"),
            apparent_gender=d.get("apparent_gender"),
            apparent_age=d.get("apparent_age"),
            gender_consistency=d.get("gender_consistency"),
            identity_review=d.get("identity_review"),
        )


# Допустимые метки пар (SKILL §9.3).
PAIR_LABELS = ("negative", "hard_negative", "uncertain_negative", "manual_review_required")
POSITIVE_TYPES = ("positive_same_post",)
NEGATIVE_TYPES = ("negative_cross_group", "negative_age_controlled", "hard_negative")


@dataclass
class Pair:
    """Пара лиц (TODO §5 / SKILL §10 «Pair»).

    label: 1 — позитив (один человек), 0 — негатив. Для неопределённых негативов
    используется ``status`` (uncertain_negative / manual_review_required).
    """

    pair_id: str
    face_a: str
    face_b: str
    label: int  # 1 positive, 0 negative
    pair_type: str  # positive_same_post / negative_cross_group / ...
    identity_group_a: str | None = None
    identity_group_b: str | None = None
    age_a: int | None = None
    age_b: int | None = None
    age_gap: int | None = None
    hardness: str = "easy"  # easy / medium / hard
    status: str = "ok"  # ok / uncertain_negative / manual_review_required
    split: str | None = None  # train / val / test

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Pair:
        return cls(**{k: d[k] for k in _field_names(cls) if k in d})


def _field_names(cls: type) -> tuple[str, ...]:
    return tuple(f.name for f in cls.__dataclass_fields__.values())  # type: ignore[attr-defined]
