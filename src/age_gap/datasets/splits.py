"""Leakage-safe сплит по identity_group_id (SKILL §9.1 / TODO §11).

Критическое правило: ни одна identity_group_id не должна попасть одновременно в train и
test. Поэтому сплитятся ГРУППЫ, а не пары. Пара получает split только если обе её группы
в одном сплите; кросс-сплитовые негативы исключаются (split=None), чтобы не было утечки.
"""

from __future__ import annotations

import random
from collections import Counter

from age_gap.common.io import data_path, read_jsonl, write_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import IdentityGroup, Pair
from age_gap.datasets.pair_builder import build_negative_pairs

log = get_logger(__name__)

SPLITS = ("train", "val", "test")


def assign_group_splits(
    group_ids: list[str],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, str]:
    """Детерминированно назначить каждой группе один из train/val/test."""
    ids = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    # Остаток — в test, чтобы покрыть округление.
    assignment: dict[str, str] = {}
    for i, gid in enumerate(ids):
        if i < n_train:
            assignment[gid] = "train"
        elif i < n_train + n_val:
            assignment[gid] = "val"
        else:
            assignment[gid] = "test"
    return assignment


def split_pairs(pairs: list[Pair], group_split: dict[str, str]) -> tuple[list[Pair], int]:
    """Проставить split парам. Возвращает (пары, число исключённых кросс-сплитовых)."""
    dropped = 0
    for p in pairs:
        sa = group_split.get(p.identity_group_a or "")
        sb = group_split.get(p.identity_group_b or "")
        if sa is not None and sa == sb:
            p.split = sa
        else:
            p.split = None
            dropped += 1
    return pairs, dropped


def run(
    groups_file: str | None = None,
    pairs_file: str | None = None,
    pairs_out: str | None = None,
    split_map_out: str | None = None,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    neg_per_pos: float = 1.0,
) -> dict[str, str]:
    """Leakage-safe сплит групп + БАЛАНСИРОВАННЫЕ негативы ВНУТРИ каждого сплита.

    Позитивы (внутри группы) штампуются сплитом своей группы. Простые негативы генерируются
    заново ПОсплитово (оба лица из групп одного сплита) с балансом neg≈neg_per_pos·pos в каждом
    сплите — иначе при многих группах почти все глобальные негативы кросс-сплитовые и
    исключаются, оставляя val/test почти без негативов. Hard-негативы (mined) сохраняются, если
    их группы в одном сплите.
    """
    groups_file = groups_file or str(data_path("data_dir", "processed", "identity_groups.jsonl"))
    pairs_file = pairs_file or str(data_path("data_dir", "processed", "pairs.jsonl"))
    pairs_out = pairs_out or str(data_path("data_dir", "processed", "pairs.jsonl"))
    split_map_out = split_map_out or str(data_path("splits_dir", "group_splits.jsonl"))

    groups = [IdentityGroup.from_dict(r) for r in read_jsonl(groups_file)]
    group_split = assign_group_splits([g.identity_group_id for g in groups], ratios, seed)
    groups_by_split: dict[str, list[IdentityGroup]] = {s: [] for s in SPLITS}
    for g in groups:
        s = group_split.get(g.identity_group_id)
        if s is not None:
            groups_by_split[s].append(g)

    pairs = [Pair.from_dict(r) for r in read_jsonl(pairs_file)]

    # Позитивы (внутри группы → обе группы в одном сплите): штампуем сплитом.
    positives = [p for p in pairs if p.label == 1]
    for p in positives:
        p.split = group_split.get(p.identity_group_a or "")
    positives = [p for p in positives if p.split is not None]

    # Сохраняем уже намайненные hard-негативы, чьи группы в одном сплите.
    kept_hard: list[Pair] = []
    for p in pairs:
        if p.label == 0 and "hard" in (p.pair_type or ""):
            sa = group_split.get(p.identity_group_a or "")
            sb = group_split.get(p.identity_group_b or "")
            if sa is not None and sa == sb:
                p.split = sa
                kept_hard.append(p)

    pos_per = Counter(p.split for p in positives)
    hard_per = Counter(p.split for p in kept_hard)

    # Простые негативы — заново, балансированно, внутри каждого сплита.
    simple_neg: list[Pair] = []
    for i, s in enumerate(SPLITS):
        target = int(round(neg_per_pos * pos_per.get(s, 0))) - hard_per.get(s, 0)
        if target <= 0 or not groups_by_split[s]:
            continue
        simple_neg.extend(
            build_negative_pairs(
                groups_by_split[s], n_per_positive=1, n_positives=target, seed=seed + i + 1, split=s
            )
        )

    out_pairs = positives + kept_hard + simple_neg
    write_jsonl(pairs_out, (p.to_dict() for p in out_pairs))
    write_jsonl(
        split_map_out,
        ({"identity_group_id": gid, "split": s} for gid, s in sorted(group_split.items())),
    )

    by_split = {
        s: {
            "pos": sum(p.split == s and p.label == 1 for p in out_pairs),
            "neg": sum(p.split == s and p.label == 0 for p in out_pairs),
        }
        for s in SPLITS
    }
    log.info(
        "Сплит готов: групп=%d, пар=%d, баланс по сплитам=%s",
        len(group_split),
        len(out_pairs),
        by_split,
    )
    return group_split
