"""Построение групп личностей из постов (SKILL §8.1 / TODO §5).

Гипотеза: мультифото-пост = кандидат на одну личность в разном возрасте. Это не всегда
верно, поэтому неоднозначные посты помечаются ``manual_review_required`` и не идут в
автоматическую генерацию позитивов (SKILL §9.2).

Возрастные якоря из подписи (regex) сопоставляются с лицами по photo_reference, когда
это возможно (first/second/left/right -> порядок фото в посте).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from age_gap.common.io import data_path, read_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import AgeLabel, FaceCrop, IdentityGroup, RawPost
from age_gap.datasets.age_anchors import AgeExtractor, RegexAgeExtractor

log = get_logger(__name__)

# Слова-ссылки на фото -> порядковая позиция (0-индекс). position_N разбирается отдельно.
_REF_TO_POSITION = {
    "first": 0,
    "left": 0,
    "second": 1,
    "right": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
}


def _ref_position(ref: str) -> int | None:
    """Позиция фото (0-индекс) из photo_reference: first/left=0, second/right=1, position_N=N."""
    if ref in _REF_TO_POSITION:
        return _REF_TO_POSITION[ref]
    if ref.startswith("position_"):
        try:
            return int(ref.split("_", 1)[1])
        except ValueError:
            return None
    return None


def build_groups(
    posts_file: str | None = None,
    faces_file: str | None = None,
    groups_out: str | None = None,
    extractor: AgeExtractor | None = None,
    resume: bool = True,
) -> int:
    """Построить группы личностей; пишет инкрементально (JSONL append) и резюмируемо.

    Уже построенные группы (по source_post_id) пропускаются — при LLM-экстракторе это
    избавляет от повторных дорогих вызовов и переживает обрыв длинного прогона (>10 мин).
    Возвращает число НОВЫХ групп.
    """
    posts_file = posts_file or str(data_path("data_dir", "raw", "posts.jsonl"))
    faces_file = faces_file or str(data_path("data_dir", "interim", "faces.jsonl"))
    groups_out = groups_out or str(data_path("data_dir", "processed", "identity_groups.jsonl"))
    extractor = extractor or RegexAgeExtractor()

    # face_id -> FaceCrop (только usable).
    usable: dict[str, FaceCrop] = {}
    for row in read_jsonl(faces_file):
        fc = FaceCrop.from_dict(row)
        if fc.is_usable:
            usable[fc.face_id] = fc

    # usable-лица по фото; внутри фото — слева направо (коллаж «тогда/сейчас»: f0=раньше, f1=позже).
    faces_by_photo: dict[str, list[FaceCrop]] = {}
    for fc in usable.values():
        faces_by_photo.setdefault(fc.photo_id, []).append(fc)
    for group_faces in faces_by_photo.values():
        group_faces.sort(key=lambda fc: fc.bbox[0] if fc.bbox else 0.0)

    done: set[str] = set()
    if resume:
        done = {row.get("source_post_id", "") for row in read_jsonl(groups_out)}
        if done:
            log.info("Резюме build_groups: уже готово %d групп, продолжаем", len(done))

    Path(groups_out).parent.mkdir(parents=True, exist_ok=True)
    new = 0
    with open(groups_out, "a", encoding="utf-8") as f:
        for row in read_jsonl(posts_file):
            post = RawPost.from_dict(row)
            if post.post_id in done:
                continue
            # Порядковый номер фото в посте (0-индекс по ПОЛНОМУ списку фото, включая те,
            # чьё лицо отбраковано) — к нему привязывает позиции LLM/regex.
            photos_sorted = sorted(post.photos, key=lambda p: p.order)
            # Глобальный порядок лиц: по порядку фото, затем слева направо внутри фото (коллажи).
            post_faces = [fc for ph in photos_sorted for fc in faces_by_photo.get(ph.photo_id, [])]
            if not post_faces:
                continue

            age_labels = _map_ages(post.caption, post_faces, extractor)
            group = IdentityGroup(
                identity_group_id=post.post_id,
                source_post_id=post.post_id,
                faces=[fc.face_id for fc in post_faces],
                age_labels=age_labels,
                status=_group_status(post.caption, post_faces),
            )
            f.write(json.dumps(group.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            new += 1

    log.info("Построено новых групп: %d -> %s", new, groups_out)
    return new


def _map_ages(
    caption: str,
    post_faces: list[FaceCrop],
    extractor: AgeExtractor,
) -> list[AgeLabel]:
    """Сопоставить возрасты лицам по ГЛОБАЛЬНОЙ позиции (порядок фото + слева направо внутри кадра).

    post_faces уже в глобальном порядке. position_N / first / left / second / right -> индекс в
    post_faces. Для коллажа «тогда/сейчас»: left=раньше=post_faces[0], right=позже=post_faces[1].
    """
    n = len(post_faces)
    raw_labels = extractor.extract(caption, n)  # n = число лиц-слотов (для коллажа = число лиц кадра)
    if not raw_labels:
        return []

    # Позиционный fallback: все якоря «unknown», число возрастов == числу лиц (>=2) -> возраст i -> лицо i.
    all_unknown = all(lbl.photo_reference == "unknown" for lbl in raw_labels)
    if all_unknown and len(raw_labels) == n and n >= 2:
        return [
            AgeLabel(
                face_id=post_faces[i].face_id,
                age=lbl.age,
                photo_reference=f"position_{i}",
                source=lbl.source,
                confidence=lbl.confidence,
                mapping_confidence=0.5,  # порядковая эвристика, ниже явной привязки
            )
            for i, lbl in enumerate(raw_labels)
        ]

    # Явная привязка: first/left=0, second/right=1, position_N=N -> индекс в глобальном порядке лиц.
    mapped: list[AgeLabel] = []
    for label in raw_labels:
        pos = _ref_position(label.photo_reference)
        face = post_faces[pos] if (pos is not None and 0 <= pos < n) else None
        mapped.append(
            AgeLabel(
                face_id=face.face_id if face else None,
                age=label.age,
                photo_reference=label.photo_reference,
                source=label.source,
                confidence=label.confidence,
                mapping_confidence=label.mapping_confidence if face else 0.0,
            )
        )
    return mapped


# Признаки НЕСКОЛЬКИХ людей в подписи (RU+EN): коллаж с такой подписью — вероятно разные
# люди (пара/семья), а не один человек «тогда/сейчас» -> исключаем из автопозитивов.
_MULTI_PERSON = re.compile(
    r"\b(?:we|us|our|wife|husband|spouse|girlfriend|boyfriend|partner|parents?|mom|dad|"
    r"mother|father|brother|sister|siblings?|cousin|couple|twins?|friends?|family|kids|"
    r"children|son|daughter)\b|\band[ ]+(?:i|my)\b|&[ ]*i\b|"
    r"муж|жена|супруг|вдвоё?м|сем(?:ья|ьи|ей)|родител|брат|сестр|друзь|подруг|\bмы\b|\bнас\b",
    re.IGNORECASE,
)


def _group_status(caption: str, post_faces: list[FaceCrop]) -> str:
    """Авто-статус группы. Коллаж (несколько лиц в одном кадре) штатно идёт в позитивы (auto),
    КРОМЕ случаев, когда подпись указывает на нескольких людей (пара/семья) -> manual_review."""
    is_collage = any(fc.num_faces_in_image > 1 for fc in post_faces)
    if is_collage and caption and _MULTI_PERSON.search(caption):
        return "manual_review_required"
    return "auto"
