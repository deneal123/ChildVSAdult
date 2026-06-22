"""Дедуп near-duplicate кадров внутри личности (Part 3): тривиальные позитивы (sim~1.0).

Внутри группы кадры с косинусом >= threshold (по умолчанию 0.97) — почти дубликаты (один кадр /
серия / репост). Позитив между такими кадрами тривиален (cos~1.0) и завышает our.overall. Находим
кластеры дубликатов (union-find внутри группы по ArcFace-эмбеддингам) и помечаем все, кроме одного
представителя, как избыточные. Артефакт ``redundant_faces.jsonl`` затем исключается из позитивов.
"""

from __future__ import annotations

import numpy as np

from age_gap.common.io import data_path, read_jsonl, write_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import IdentityGroup
from age_gap.datasets.person_clusters import _UnionFind

log = get_logger(__name__)


def find_redundant_faces(
    groups: list[IdentityGroup],
    embeddings: dict[str, np.ndarray],
    threshold: float = 0.97,
) -> dict[str, list[str]]:
    """{group_id: [избыточные face_ids]} — все, кроме представителя, в каждом near-dup кластере."""
    redundant: dict[str, list[str]] = {}
    for g in groups:
        faces = [f for f in g.faces if f in embeddings]
        if len(faces) < 2:
            continue
        uf = _UnionFind(faces)
        mat = np.stack([embeddings[f] for f in faces])  # ArcFace-эмбеддинги уже L2-нормированы
        sims = mat @ mat.T
        n = len(faces)
        for i in range(n):
            for j in range(i + 1, n):
                if sims[i, j] >= threshold:
                    uf.union(faces[i], faces[j])
        clusters: dict[str, list[str]] = {}
        for f in faces:
            clusters.setdefault(uf.find(f), []).append(f)
        red = [f for members in clusters.values() if len(members) > 1 for f in sorted(members)[1:]]
        if red:
            redundant[g.identity_group_id] = red
    return redundant


def run(
    threshold: float = 0.97,
    groups_file: str | None = None,
    emb_path: str | None = None,
    out: str | None = None,
) -> dict[str, list[str]]:
    """Найти near-dup лица во всех группах и записать плоский список избыточных face_ids."""
    from age_gap.models.embeddings import load_embeddings

    groups_file = groups_file or str(data_path("data_dir", "processed", "identity_groups.jsonl"))
    out = out or str(data_path("data_dir", "processed", "redundant_faces.jsonl"))
    groups = [IdentityGroup.from_dict(r) for r in read_jsonl(groups_file)]
    embeddings = load_embeddings(emb_path)
    redundant = find_redundant_faces(groups, embeddings, threshold)
    flat = sorted(f for fs in redundant.values() for f in fs)
    write_jsonl(out, ({"face_id": f} for f in flat))
    log.info(
        "Near-dup дедуп (thr=%.2f): групп с дублями=%d, избыточных лиц=%d -> %s",
        threshold,
        len(redundant),
        len(flat),
        out,
    )
    return redundant
