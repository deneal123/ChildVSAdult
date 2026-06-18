"""Тесты leakage-safe сплита (SKILL §9.1 / TODO §11)."""

from __future__ import annotations

from age_gap.common.schemas import AgeLabel, IdentityGroup, Pair
from age_gap.datasets.age_anchors import RegexAgeExtractor  # noqa: F401 (smoke import)
from age_gap.datasets.pair_builder import build_negative_pairs, build_positive_pairs
from age_gap.datasets.splits import assign_group_splits, split_pairs


def _make_groups(n: int) -> list[IdentityGroup]:
    groups = []
    for i in range(n):
        faces = [f"g{i}_a", f"g{i}_b"]
        groups.append(
            IdentityGroup(
                identity_group_id=f"g{i}",
                source_post_id=f"g{i}",
                faces=faces,
                age_labels=[
                    AgeLabel(face_id=faces[0], age=10 + i),
                    AgeLabel(face_id=faces[1], age=30 + i),
                ],
                status="auto",
            )
        )
    return groups


def test_each_group_single_split():
    gs = assign_group_splits([f"g{i}" for i in range(20)], seed=3)
    # dict по определению сопоставляет группе один сплит.
    assert set(gs.values()) <= {"train", "val", "test"}
    assert len(gs) == 20


def test_no_identity_leakage_across_splits():
    groups = _make_groups(20)
    group_split = assign_group_splits([g.identity_group_id for g in groups], seed=3)

    pairs: list[Pair] = build_positive_pairs(groups)
    pairs += build_negative_pairs(groups, n_per_positive=3, n_positives=len(pairs), seed=3)
    pairs, dropped = split_pairs(pairs, group_split)

    # Для каждого сплита собрать все группы, фигурирующие в его парах.
    groups_in_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for p in pairs:
        if p.split is None:
            continue
        groups_in_split[p.split].add(p.identity_group_a or "")
        groups_in_split[p.split].add(p.identity_group_b or "")

    # Ни одна группа не должна встречаться в двух разных сплитах.
    train, val, test = groups_in_split["train"], groups_in_split["val"], groups_in_split["test"]
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_cross_split_negatives_excluded():
    groups = _make_groups(10)
    group_split = assign_group_splits([g.identity_group_id for g in groups], seed=1)
    negs = build_negative_pairs(groups, n_per_positive=5, n_positives=10, seed=1)
    negs, dropped = split_pairs(negs, group_split)
    # Пары с проставленным split должны иметь обе группы в одном сплите.
    for p in negs:
        if p.split is not None:
            assert group_split[p.identity_group_a] == group_split[p.identity_group_b] == p.split
