"""Hard-negative mining по baseline-эмбеддингам (MVP-4, TODO §7 / SKILL §12).

Идея: визуально похожие люди ИЗ РАЗНЫХ групп — самые сложные негативы. Для каждого
якорного лица берём ближайших соседей в пространстве baseline-эмбеддингов из других групп.

False-negative защита (SKILL §9.3): если сходство ОЧЕНЬ высокое, пара может быть одним
человеком из разных постов — помечаем uncertain_negative (на ревью) и НЕ используем как
негатив. Чистая вычислительная логика вынесена в ``mine`` — тестируется офлайн.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from age_gap.common.io import data_path, read_jsonl, write_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import IdentityGroup, Pair
from age_gap.models.embeddings import load_embeddings

log = get_logger(__name__)

# Косинус ArcFace: для разных людей обычно < 0.3; очень высокое — подозрение на того же человека.
HARD_MIN_SIM = 0.20  # ниже — это просто лёгкий негатив, не «hard»
UNCERTAIN_SIM = 0.55  # выше — возможный тот же человек -> uncertain_negative (исключаем)


@dataclass
class MineResult:
    hard: list[Pair]
    n_uncertain: int  # сколько пар отсеяно как возможные false-negatives


def mine(
    face_ids: list[str],
    embeddings: np.ndarray,  # (N, D), L2-нормированные
    group_by_face: dict[str, str],
    top_k: int = 5,
    hard_min_sim: float = HARD_MIN_SIM,
    uncertain_sim: float = UNCERTAIN_SIM,
    max_total: int | None = None,
) -> MineResult:
    """Намайнить hard-негативы: для каждого лица — top_k ближайших из ДРУГИХ групп.

    max_total ограничивает итоговое число, оставляя САМЫЕ сложные (с наибольшим сходством) —
    защита от перекоса выборки, который иначе топит позитивы массой негативов.
    """
    if len(face_ids) < 2:
        return MineResult(hard=[], n_uncertain=0)

    sims = embeddings @ embeddings.T  # косинус (эмбеддинги нормированы)
    groups = [group_by_face.get(f, "") for f in face_ids]

    scored: list[tuple[float, tuple[str, str]]] = []  # (sim, key)
    seen: set[tuple[str, str]] = set()
    n_uncertain = 0

    for i, fa in enumerate(face_ids):
        order = np.argsort(-sims[i])  # соседи по убыванию сходства
        taken = 0
        for j in order:
            if j == i or groups[i] == groups[j]:
                continue
            s = float(sims[i, j])
            if s < hard_min_sim:
                break  # дальше только меньше
            key = (fa, face_ids[j]) if fa < face_ids[j] else (face_ids[j], fa)
            if key in seen:
                continue
            seen.add(key)
            if s >= uncertain_sim:
                n_uncertain += 1  # возможный тот же человек — не используем как негатив
                continue
            scored.append((s, key))
            taken += 1
            if taken >= top_k:
                break

    # Оставляем самые сложные (highest sim) при лимите.
    scored.sort(key=lambda x: -x[0])
    if max_total is not None and len(scored) > max_total:
        scored = scored[:max_total]

    hard = [
        Pair(
            pair_id=f"hardneg_{key[0]}__{key[1]}",
            face_a=key[0],
            face_b=key[1],
            label=0,
            pair_type="hard_negative",
            identity_group_a=group_by_face.get(key[0]),
            identity_group_b=group_by_face.get(key[1]),
            hardness="hard",
            status="ok",
        )
        for _, key in scored
    ]
    log.info("Hard negatives: %d (uncertain отсеяно: %d)", len(hard), n_uncertain)
    return MineResult(hard=hard, n_uncertain=n_uncertain)


def _pair_key(face_a: str, face_b: str) -> tuple[str, str]:
    return (face_a, face_b) if face_a < face_b else (face_b, face_a)


def run(
    groups_file: str | None = None,
    pairs_file: str | None = None,
    embeddings_file: str | None = None,
    pairs_out: str | None = None,
    top_k: int = 5,
    max_total: int | None = None,
) -> int:
    """Намайнить hard-негативы и дозаписать их в pairs.jsonl (без дублей). Возвращает их число.

    После запуска нужно пересобрать сплит (scripts/split.py), чтобы проставить split новым парам.
    """
    groups_file = groups_file or str(data_path("data_dir", "processed", "identity_groups.jsonl"))
    pairs_file = pairs_file or str(data_path("data_dir", "processed", "pairs.jsonl"))
    pairs_out = pairs_out or pairs_file

    embeddings = load_embeddings(embeddings_file)
    if not embeddings:
        raise RuntimeError("Нет baseline-эмбеддингов: сначала запустите embed.py")

    group_by_face: dict[str, str] = {}
    for row in read_jsonl(groups_file):
        g = IdentityGroup.from_dict(row)
        for fid in g.faces:
            group_by_face[fid] = g.identity_group_id

    # Майним только по лицам, у которых есть и эмбеддинг, и группа.
    face_ids = [f for f in embeddings if f in group_by_face]
    matrix = np.asarray([embeddings[f] for f in face_ids], dtype=np.float32)
    result = mine(face_ids, matrix, group_by_face, top_k=top_k, max_total=max_total)

    existing = [Pair.from_dict(r) for r in read_jsonl(pairs_file)]
    seen = {_pair_key(p.face_a, p.face_b) for p in existing}
    fresh = [p for p in result.hard if _pair_key(p.face_a, p.face_b) not in seen]

    combined = existing + fresh
    write_jsonl(Path(pairs_out), (p.to_dict() for p in combined))
    log.info(
        "Добавлено hard-негативов: %d (было пар %d -> стало %d); uncertain отсеяно %d",
        len(fresh),
        len(existing),
        len(combined),
        result.n_uncertain,
    )
    return len(fresh)
