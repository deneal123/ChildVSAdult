"""Пересборка ЧИСТОГО датасета (Part 2b + Part 3): prune групп перед пересборкой пар.

Убирает из групп (а) near-dup лица (`redundant_faces.jsonl`) — тривиальные позитивы и (б) целые
шумные группы (`identity_review` in NOISY: multi_person/collage/meme). Перезаписывает
``identity_groups.jsonl`` (с бэкапом), после чего стандартные ``build_pairs`` + ``split`` дают чистый
датасет без изменений в pair_builder/splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from age_gap.common.io import data_path, read_jsonl, write_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import IdentityGroup
from age_gap.datasets.group_validator import NOISY

log = get_logger(__name__)


def prune_groups(
    groups: list[IdentityGroup], redundant: set[str], drop_noisy: bool = True
) -> tuple[list[IdentityGroup], dict[str, int]]:
    """Убрать near-dup лица из групп и (опц.) выбросить шумные. Возвращает (clean, stats)."""
    clean: list[IdentityGroup] = []
    stats = {
        "groups_in": len(groups),
        "dropped_noisy": 0,
        "removed_faces": 0,
        "faces_in": sum(len(g.faces) for g in groups),
    }
    for g in groups:
        if drop_noisy and g.identity_review in NOISY:
            stats["dropped_noisy"] += 1
            continue
        kept = [f for f in g.faces if f not in redundant]
        stats["removed_faces"] += len(g.faces) - len(kept)
        g.faces = kept
        clean.append(g)
    stats["groups_out"] = len(clean)
    stats["faces_out"] = sum(len(g.faces) for g in clean)
    stats["multiface_out"] = sum(len(g.faces) >= 2 for g in clean)
    return clean, stats


def run(
    drop_noisy: bool = True,
    groups_file: str | None = None,
    redundant_file: str | None = None,
) -> dict[str, Any]:
    """Применить prune к identity_groups.jsonl (с бэкапом). Возвращает статистику."""
    groups_file = groups_file or str(data_path("data_dir", "processed", "identity_groups.jsonl"))
    redundant_file = redundant_file or str(
        data_path("data_dir", "processed", "redundant_faces.jsonl")
    )
    groups = [IdentityGroup.from_dict(r) for r in read_jsonl(groups_file)]
    redundant = {r["face_id"] for r in read_jsonl(redundant_file)}
    clean, stats = prune_groups(groups, redundant, drop_noisy=drop_noisy)
    backup = Path(groups_file).with_suffix(".jsonl.pre_prune_bak")
    Path(groups_file).replace(backup)
    write_jsonl(groups_file, (g.to_dict() for g in clean))
    log.info(
        "Prune: групп %d->%d (шумных выброшено %d), лиц %d->%d (near-dup убрано %d); бэкап -> %s",
        stats["groups_in"],
        stats["groups_out"],
        stats["dropped_noisy"],
        stats["faces_in"],
        stats["faces_out"],
        stats["removed_faces"],
        backup.name,
    )
    return stats


if __name__ == "__main__":
    run()
