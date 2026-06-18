"""Retrieval-оценка кросс-возрастного поиска (TODO §11, требование paper-протокола).

Протокол: каждое лицо — query; галерея — остальные лица того же сплита; релевантны лица той
же личности (identity_group_id). Считаем Rank-1, Recall@K, MRR и медианный ранг первого
релевантного. Чистая функция ``retrieval_metrics`` тестируется офлайн.

Без утечки: оцениваем внутри одного сплита (по умолчанию test) — релевантные и дистракторы
берутся только из него.
"""

from __future__ import annotations

import numpy as np

from age_gap.common.io import data_path, read_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import IdentityGroup
from age_gap.models.embeddings import load_embeddings

log = get_logger(__name__)


def retrieval_metrics(
    embeddings: np.ndarray,  # (N, D), L2-нормированные
    identities: np.ndarray,  # (N,) целочисленные коды личностей
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Метрики ранжирования. Учитываются только query, у которых есть релевантный в галерее."""
    n = len(identities)
    if n < 2:
        return {"n_queries": 0.0}

    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)  # сам себя не извлекаем
    order = np.argsort(-sims, axis=1)  # (N, N) индексы галереи по убыванию сходства

    counts = np.bincount(identities)
    has_relevant = counts[identities] >= 2  # есть хотя бы один однофамилец в галерее

    first_rel_rank: list[int] = []
    recall_hits = {k: 0 for k in ks}
    rank1_hits = 0
    n_q = 0
    for i in range(n):
        if not has_relevant[i]:
            continue
        n_q += 1
        gallery_ids = identities[order[i]]
        rel_mask = gallery_ids == identities[i]
        first = int(np.argmax(rel_mask))  # позиция первого релевантного (0-индекс)
        first_rel_rank.append(first + 1)
        if first == 0:
            rank1_hits += 1
        for k in ks:
            if rel_mask[:k].any():
                recall_hits[k] += 1

    if n_q == 0:
        return {"n_queries": 0.0}

    ranks = np.asarray(first_rel_rank, dtype=float)
    out: dict[str, float] = {
        "n_queries": float(n_q),
        "rank1": rank1_hits / n_q,
        "mrr": float((1.0 / ranks).mean()),
        "median_rank": float(np.median(ranks)),
    }
    for k in ks:
        out[f"recall@{k}"] = recall_hits[k] / n_q
    return out


def _group_split() -> dict[str, str]:
    path = data_path("splits_dir", "group_splits.jsonl")
    return {r["identity_group_id"]: r["split"] for r in read_jsonl(path)}


def evaluate_retrieval(
    embeddings_file: str | None = None,
    groups_file: str | None = None,
    split: str | None = "test",
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Retrieval-метрики на лицах указанного сплита (по умолчанию test)."""
    groups_file = groups_file or str(data_path("data_dir", "processed", "identity_groups.jsonl"))
    embeddings = load_embeddings(embeddings_file)
    gsplit = _group_split()

    group_by_face: dict[str, str] = {}
    for row in read_jsonl(groups_file):
        g = IdentityGroup.from_dict(row)
        if split is not None and gsplit.get(g.identity_group_id) != split:
            continue
        for fid in g.faces:
            group_by_face[fid] = g.identity_group_id

    face_ids = [f for f in group_by_face if f in embeddings]
    if not face_ids:
        log.warning("Нет лиц с эмбеддингами в сплите %s", split)
        return {"n_queries": 0.0}

    codes = {gid: i for i, gid in enumerate(sorted({group_by_face[f] for f in face_ids}))}
    matrix = np.asarray([embeddings[f] for f in face_ids], dtype=np.float32)
    identities = np.asarray([codes[group_by_face[f]] for f in face_ids], dtype=np.int64)

    res = retrieval_metrics(matrix, identities, ks)
    log.info(
        "Retrieval(split=%s): queries=%d rank1=%.3f mrr=%.3f",
        split,
        int(res.get("n_queries", 0)),
        res.get("rank1", float("nan")),
        res.get("mrr", float("nan")),
    )
    return res
