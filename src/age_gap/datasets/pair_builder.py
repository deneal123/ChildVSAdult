"""Генерация позитивных и негативных пар (TODO §6–7 / SKILL §12).

Позитивы — все C(n,2) внутри одной группы личности (один человек в разном возрасте).
Негативы — между разными группами; при совпадении возрастного бакета помечаются как
age-controlled (сложнее). Hard-negative mining на эмбеддингах и false-negative контроль
через baseline-сходство — хук на MVP-4 (см. комментарии), т.к. эмбеддинги появляются
в MVP-2.

Группы со статусом ``manual_review_required`` не дают автоматических позитивов (SKILL §9.2).
"""

from __future__ import annotations

import itertools
import random

from age_gap.common.io import data_path, read_jsonl, write_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import IdentityGroup, Pair

log = get_logger(__name__)

# Группы, которые можно использовать для автоматических позитивов.
_POSITIVE_OK_STATUS = {"auto", "mixed", "manual_verified"}

# Возрастные бакеты для age-controlled негативов (TODO §11 breakdown).
_AGE_BUCKETS = [(0, 5), (6, 12), (13, 17), (18, 25), (26, 35), (36, 50), (51, 200)]


def _age_bucket(age: int | None) -> int | None:
    if age is None:
        return None
    for i, (lo, hi) in enumerate(_AGE_BUCKETS):
        if lo <= age <= hi:
            return i
    return None


def _ages_by_face(group: IdentityGroup) -> dict[str, int | None]:
    ages: dict[str, int | None] = {fid: None for fid in group.faces}
    for label in group.age_labels:
        if label.face_id and label.age is not None:
            ages[label.face_id] = label.age
    return ages


def build_positive_pairs(groups: list[IdentityGroup]) -> list[Pair]:
    pairs: list[Pair] = []
    for g in groups:
        if g.status not in _POSITIVE_OK_STATUS or len(g.faces) < 2:
            continue
        ages = _ages_by_face(g)
        for fa, fb in itertools.combinations(sorted(g.faces), 2):
            age_a, age_b = ages.get(fa), ages.get(fb)
            gap = abs(age_a - age_b) if age_a is not None and age_b is not None else None
            pairs.append(
                Pair(
                    pair_id=f"pos_{fa}__{fb}",
                    face_a=fa,
                    face_b=fb,
                    label=1,
                    pair_type="positive_same_post",
                    identity_group_a=g.identity_group_id,
                    identity_group_b=g.identity_group_id,
                    age_a=age_a,
                    age_b=age_b,
                    age_gap=gap,
                    hardness="easy",
                    status="ok",
                )
            )
    return pairs


def build_negative_pairs(
    groups: list[IdentityGroup],
    n_per_positive: int,
    n_positives: int,
    seed: int = 42,
    split: str | None = None,
) -> list[Pair]:
    """Кросс-групповые негативы со случайной выборкой (детерминированной по seed).

    Совпадение возрастного бакета -> negative_age_controlled (hardness=medium).
    Иначе negative_cross_group (hardness=easy).
    ``split`` — если задан, негативы строятся только среди переданных групп (предполагается,
    что все они одного сплита) и каждая пара штампуется этим сплитом (балансировка по сплитам).
    """
    rng = random.Random(seed)
    # Плоский список (face_id, group_id, age) по всем группам.
    flat: list[tuple[str, str, int | None]] = []
    for g in groups:
        ages = _ages_by_face(g)
        for fid in g.faces:
            flat.append((fid, g.identity_group_id, ages.get(fid)))

    target = max(0, n_per_positive * n_positives)
    pairs: list[Pair] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = target * 50 + 1000

    if len(flat) < 2:
        return pairs

    while len(pairs) < target and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(flat, 2)
        if a[1] == b[1]:  # одна группа — не негатив
            continue
        ordered = sorted((a[0], b[0]))
        key: tuple[str, str] = (ordered[0], ordered[1])
        if key in seen:
            continue
        seen.add(key)

        same_bucket = _age_bucket(a[2]) is not None and _age_bucket(a[2]) == _age_bucket(b[2])
        pair_type = "negative_age_controlled" if same_bucket else "negative_cross_group"
        hardness = "medium" if same_bucket else "easy"
        gap = abs(a[2] - b[2]) if a[2] is not None and b[2] is not None else None

        # NOTE(MVP-4): false-negative контроль — если baseline-сходство пары очень высокое,
        # помечать status="uncertain_negative"/"manual_review_required" и не использовать
        # как негатив до ревью (SKILL §9.3). Требует эмбеддингов из MVP-2.
        pairs.append(
            Pair(
                pair_id=f"neg_{key[0]}__{key[1]}",
                face_a=key[0],
                face_b=key[1],
                label=0,
                pair_type=pair_type,
                identity_group_a=a[1],
                identity_group_b=b[1],
                age_a=a[2],
                age_b=b[2],
                age_gap=gap,
                hardness=hardness,
                status="ok",
                split=split,
            )
        )

    if len(pairs) < target:
        log.warning("Сгенерировано %d негативов из %d целевых (мало групп/лиц)", len(pairs), target)
    return pairs


# NOTE(MVP-4): mine_hard_negatives(...) — ближайшие соседи в пространстве baseline-эмбеддингов
# из других групп как hard negatives. Реализуется после MVP-2 (есть эмбеддинги).


def build_pairs(
    groups_file: str | None = None,
    pairs_out: str | None = None,
    n_negatives_per_positive: int = 1,
    seed: int = 42,
) -> list[Pair]:
    groups_file = groups_file or str(data_path("data_dir", "processed", "identity_groups.jsonl"))
    pairs_out = pairs_out or str(data_path("data_dir", "processed", "pairs.jsonl"))

    groups = [IdentityGroup.from_dict(r) for r in read_jsonl(groups_file)]
    positives = build_positive_pairs(groups)
    negatives = build_negative_pairs(
        groups, n_per_positive=n_negatives_per_positive, n_positives=len(positives), seed=seed
    )
    all_pairs = positives + negatives

    n = write_jsonl(pairs_out, (p.to_dict() for p in all_pairs))
    log.info("Пар: %d (pos=%d, neg=%d) -> %s", n, len(positives), len(negatives), pairs_out)
    return all_pairs
