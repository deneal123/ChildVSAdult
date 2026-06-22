"""Агрегация apparent-пола/возраста (genderage) из per-face sidecar в группы и посты (Part 1).

Per-face genderage (``data/interim/face_genderage.jsonl``) сворачивается в person-level атрибуты
``IdentityGroup``: majority-пол + consistency (доля согласия) + медианный apparent-возраст. Также
пишется post-level сводка пола. Модуль только ЧИТАЕТ sidecar — без загрузки моделей (genderage
считается отдельно, ``compute_face_attributes``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from age_gap.common.io import data_path, read_jsonl, write_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import IdentityGroup

log = get_logger(__name__)

_GENDER = {0: "F", 1: "M"}


def load_face_attributes(path: str | None = None) -> dict[str, tuple[str, int]]:
    """{face_id: (gender_str, age_est)} из sidecar genderage."""
    path = path or str(data_path("data_dir", "interim", "face_genderage.jsonl"))
    out: dict[str, tuple[str, int]] = {}
    for r in read_jsonl(path):
        out[r["face_id"]] = (_GENDER.get(r["gender"], "?"), int(r["age_est"]))
    return out


def aggregate_group(
    faces: list[str], attrs: dict[str, tuple[str, int]]
) -> tuple[str | None, int | None, float | None]:
    """(majority-пол, медианный apparent-возраст, consistency) по лицам группы; None если нет данных."""
    genders: list[str] = []
    ages: list[int] = []
    for f in faces:
        if f in attrs:
            g, a = attrs[f]
            genders.append(g)
            ages.append(a)
    if not genders:
        return None, None, None
    cnt = Counter(genders)
    majority, maj_n = cnt.most_common(1)[0]
    return majority, int(median(ages)), round(maj_n / len(genders), 3)


def enrich_groups(
    groups_file: str | None = None, attrs: dict[str, tuple[str, int]] | None = None
) -> list[IdentityGroup]:
    """Проставить apparent_gender/age/consistency каждой группе и перезаписать файл (с бэкапом)."""
    groups_file = groups_file or str(data_path("data_dir", "processed", "identity_groups.jsonl"))
    attrs = attrs if attrs is not None else load_face_attributes()
    groups = [IdentityGroup.from_dict(r) for r in read_jsonl(groups_file)]
    n_set = 0
    for grp in groups:
        gender, age, cons = aggregate_group(grp.faces, attrs)
        grp.apparent_gender, grp.apparent_age, grp.gender_consistency = gender, age, cons
        if gender:
            n_set += 1
    backup = Path(groups_file).with_suffix(".jsonl.pre_gender_bak")
    Path(groups_file).replace(backup)
    write_jsonl(groups_file, (g.to_dict() for g in groups))
    log.info(
        "Gender агрегирован: %d/%d групп получили пол (бэкап -> %s)",
        n_set,
        len(groups),
        backup.name,
    )
    return groups


def write_post_gender(groups: list[IdentityGroup], out: str | None = None) -> int:
    """Post-level сводка: распределение пола персон поста + доминирующий пол."""
    out = out or str(data_path("data_dir", "processed", "post_gender.jsonl"))
    by_post: dict[str, list[str]] = defaultdict(list)
    for g in groups:
        if g.apparent_gender:
            by_post[g.source_post_id].append(g.apparent_gender)
    rows = [
        {
            "post_id": pid,
            "genders": dict(Counter(gs)),
            "dominant": Counter(gs).most_common(1)[0][0],
            "n_persons": len(gs),
        }
        for pid, gs in by_post.items()
    ]
    write_jsonl(out, rows)
    log.info("Post-level gender: %d постов -> %s", len(rows), out)
    return len(rows)
