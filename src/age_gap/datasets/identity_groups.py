"""Построение групп личностей из постов (SKILL §8.1 / TODO §5).

Гипотеза: мультифото-пост = кандидат на одну личность в разном возрасте. Это не всегда
верно, поэтому неоднозначные посты помечаются ``manual_review_required`` и не идут в
автоматическую генерацию позитивов (SKILL §9.2).

Возрастные якоря из подписи (regex) сопоставляются с лицами по photo_reference, когда
это возможно (first/second/left/right -> порядок фото в посте).
"""

from __future__ import annotations

import json
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
            seq_by_photo = {ph.photo_id: i for i, ph in enumerate(photos_sorted)}
            total_photos = len(photos_sorted)
            post_faces = [
                usable[f"{ph.photo_id}_f0"] for ph in photos_sorted if f"{ph.photo_id}_f0" in usable
            ]
            if not post_faces:
                continue

            age_labels = _map_ages(post.caption, post_faces, seq_by_photo, total_photos, extractor)
            group = IdentityGroup(
                identity_group_id=post.post_id,
                source_post_id=post.post_id,
                faces=[fc.face_id for fc in post_faces],
                age_labels=age_labels,
                status=_group_status(post, post_faces),
            )
            f.write(json.dumps(group.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            new += 1

    log.info("Построено новых групп: %d -> %s", new, groups_out)
    return new


def _map_ages(
    caption: str,
    post_faces: list[FaceCrop],
    seq_by_photo: dict[str, int],
    total_photos: int,
    extractor: AgeExtractor,
) -> list[AgeLabel]:
    """Сопоставить возрасты лицам по позиции фото в посте (а не по индексу usable-лица).

    Позиция position_N от LLM/regex = порядковый номер фото в посте; привязываем к usable-лицу
    именно этого фото (если оно не отбраковано). Это исключает привязку к чужому лицу, когда
    часть фото поста отбракована.
    """
    # Порядковый номер фото -> usable-лицо этого фото; и лица в порядке фото.
    face_by_seq = {seq_by_photo.get(fc.photo_id, -1): fc for fc in post_faces}
    faces_in_order = sorted(post_faces, key=lambda fc: seq_by_photo.get(fc.photo_id, 0))

    # LLM получает ОБЩЕЕ число фото — индексы position_N совпадают с порядком фото в посте.
    raw_labels = extractor.extract(caption, total_photos)
    if not raw_labels:
        return []

    # Позиционный fallback (TODO §8, правило 2): все якоря «unknown», но число возрастов
    # совпадает с числом usable-лиц И с числом фото (нет отбраковки) — возраст i -> лицо i.
    all_unknown = all(lbl.photo_reference == "unknown" for lbl in raw_labels)
    if all_unknown and len(raw_labels) == len(faces_in_order) == total_photos and total_photos >= 2:
        return [
            AgeLabel(
                face_id=faces_in_order[i].face_id,
                age=lbl.age,
                photo_reference=f"position_{i}",
                source=lbl.source,
                confidence=lbl.confidence,
                mapping_confidence=0.5,  # порядковая эвристика, ниже явной привязки
            )
            for i, lbl in enumerate(raw_labels)
        ]

    # Явная привязка по позиции фото: first/left=0, second/right=1, position_N=N.
    mapped: list[AgeLabel] = []
    for label in raw_labels:
        pos = _ref_position(label.photo_reference)
        target = face_by_seq.get(pos) if pos is not None else None
        target_face = target.face_id if target else None
        mapped.append(
            AgeLabel(
                face_id=target_face,
                age=label.age,
                photo_reference=label.photo_reference,
                source=label.source,
                confidence=label.confidence,
                mapping_confidence=label.mapping_confidence if target_face else 0.0,
            )
        )
    return mapped


def _group_status(post: RawPost, post_faces: list[FaceCrop]) -> str:
    """Авто-статус группы. Неоднозначные посты -> manual_review_required (SKILL §9.2).

    Признак неоднозначности на этом этапе: фото поста, где детектор нашёл несколько лиц
    (коллаж/несколько людей), повышают риск неверного позитива.
    """
    # Если хотя бы у одного usable-лица в исходном фото было >1 лица — на ревью.
    if any(fc.num_faces_in_image > 1 for fc in post_faces):
        return "manual_review_required"
    if len(post_faces) < 2:
        # Группа из одного лица не даёт позитивных пар, но валидна как источник негативов.
        return "auto"
    return "auto"
