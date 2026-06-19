"""Кластеризация групп (постов) в ЛИЧНОСТИ по эмбеддингам — для истинно leakage-safe сплита.

Проблема (см. docs/Observations.md): один человек встречается в РАЗНЫХ постах (и даже между
стенами), а наш сплит leakage-safe по ``identity_group_id`` = по посту → один человек может
попасть и в train, и в test, завышая ``our.*``.

Решение: сливаем две группы в одну личность, если между ними есть пара лиц с косинусом
≥ ``merge_threshold`` (надёжный «тот же человек», по ручной проверке ~0.85). Связные компоненты
групп = личности. Затем сплит делается по личности, а не по посту.

Порог высокий намеренно: ручная проверка показала, что при cos<0.85 это СМЕСЬ look-alike и
same-person — низкий порог сливал бы разных людей. Ложное слияние look-alike для leakage-safety
безвредно (два разных человека в одном сплите), а вот пропуск дубликата — вреден (утечка).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from age_gap.common.io import data_path, read_jsonl, write_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import IdentityGroup
from age_gap.models.embeddings import load_embeddings

log = get_logger(__name__)

MERGE_THRESHOLD = 0.85  # косинус, выше которого считаем лица одним человеком (по ручной проверке)


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # сжатие пути
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_groups(
    face_ids: list[str],
    embeddings: np.ndarray,  # (N, D) L2-нормированные
    group_by_face: dict[str, str],
    merge_threshold: float = MERGE_THRESHOLD,
    chunk: int = 2048,
) -> dict[str, str]:
    """Слить группы в личности по высокому косинусу между лицами. Возвращает group_id -> person_id.

    person_id = лексикографически минимальный group_id в компоненте (детерминированно).
    """
    groups_order: list[str] = list(dict.fromkeys(group_by_face[f] for f in face_ids))
    uf = _UnionFind(groups_order)
    g = [group_by_face[f] for f in face_ids]
    n = len(face_ids)

    for start in range(0, n, chunk):
        block = embeddings[start : start + chunk] @ embeddings.T  # (b, N)
        for bi in range(block.shape[0]):
            i = start + bi
            row = block[bi]
            row[i] = -1.0
            for j in np.nonzero(row >= merge_threshold)[0]:  # пары ≥ порога — редки
                if g[i] != g[int(j)]:
                    uf.union(g[i], g[int(j)])

    # person_id = минимальный group_id в компоненте.
    root_to_min: dict[str, str] = {}
    for gid in groups_order:
        r = uf.find(gid)
        if r not in root_to_min or gid < root_to_min[r]:
            root_to_min[r] = gid
    return {gid: root_to_min[uf.find(gid)] for gid in groups_order}


def run(
    groups_file: str | None = None,
    embeddings_file: str | None = None,
    out_file: str | None = None,
    merge_threshold: float = MERGE_THRESHOLD,
) -> dict[str, str]:
    """Построить карту group_id -> person_id и сохранить. Возвращает её.

    Read-only по отношению к pairs/splits — пишет только person_clusters.jsonl.
    """
    groups_file = groups_file or str(data_path("data_dir", "processed", "identity_groups.jsonl"))
    out_file = out_file or str(data_path("data_dir", "processed", "person_clusters.jsonl"))

    embeddings = load_embeddings(embeddings_file)
    if not embeddings:
        raise RuntimeError("Нет эмбеддингов: сначала запустите embed.py")

    group_by_face: dict[str, str] = {}
    for row in read_jsonl(groups_file):
        grp = IdentityGroup.from_dict(row)
        for fid in grp.faces:
            group_by_face[fid] = grp.identity_group_id

    face_ids = [f for f in embeddings if f in group_by_face]
    matrix = np.asarray([embeddings[f] for f in face_ids], dtype=np.float32)
    group_to_person = cluster_groups(face_ids, matrix, group_by_face, merge_threshold)

    n_groups = len(set(group_to_person))
    n_persons = len(set(group_to_person.values()))
    merged = n_groups - n_persons
    write_jsonl(
        Path(out_file),
        ({"identity_group_id": g, "person_id": p} for g, p in sorted(group_to_person.items())),
    )
    log.info(
        "Личности: групп=%d -> личностей=%d (слито %d, порог cos≥%.2f) -> %s",
        n_groups,
        n_persons,
        merged,
        merge_threshold,
        out_file,
    )
    return group_to_person


def merge_to_person_groups(
    group_to_person: dict[str, str],
    groups_file: str | None = None,
    out_file: str | None = None,
) -> int:
    """Схлопнуть группы одной личности в одну IdentityGroup (для leakage-safe сплита по человеку).

    Объединяет faces и age_labels всех постов личности; identity_group_id = person_id. После этого
    build_pairs/split работают по ЧЕЛОВЕКУ без изменений (бонус: cross-post longitudinal-позитивы).
    Перезаписывает groups_file (post-level бэкапится в .post_bak). Возвращает число личностей.
    """
    from collections import defaultdict

    groups_file = groups_file or str(data_path("data_dir", "processed", "identity_groups.jsonl"))
    out_file = out_file or groups_file

    members: dict[str, list[IdentityGroup]] = defaultdict(list)
    for row in read_jsonl(groups_file):
        g = IdentityGroup.from_dict(row)
        members[group_to_person.get(g.identity_group_id, g.identity_group_id)].append(g)

    merged: list[IdentityGroup] = []
    for pid, grps in members.items():
        faces: list[str] = []
        seen: set[str] = set()
        for g in grps:  # union faces без дублей, сохраняя порядок
            for f in g.faces:
                if f not in seen:
                    seen.add(f)
                    faces.append(f)
        age_labels = [a for g in grps for a in g.age_labels]
        statuses = {g.status for g in grps}
        if "manual_review_required" in statuses:
            status = "manual_review_required"
        elif len(grps) > 1 or statuses != {"auto"}:
            status = "mixed"
        else:
            status = "auto"
        merged.append(
            IdentityGroup(
                identity_group_id=pid,
                source_post_id=grps[0].source_post_id,
                faces=faces,
                age_labels=age_labels,
                status=status,
            )
        )

    if Path(out_file).exists():
        backup = Path(out_file).with_suffix(".jsonl.post_bak")
        Path(out_file).replace(backup)
        log.info("Post-level группы -> %s", backup)
    write_jsonl(Path(out_file), (g.to_dict() for g in merged))
    log.info("Person-level группы: %d -> %s", len(merged), out_file)
    return len(merged)
