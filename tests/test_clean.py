"""Тесты prune чистого датасета (Part 2b+3) — чистая логика, без файлов."""

from __future__ import annotations

from age_gap.common.schemas import IdentityGroup
from age_gap.datasets.clean import prune_groups


def _g(gid: str, faces: list[str], review: str | None = None) -> IdentityGroup:
    return IdentityGroup(
        identity_group_id=gid, source_post_id="p", faces=faces, identity_review=review
    )


def test_prune_drops_noisy_groups():
    groups = [_g("a", ["1", "2"], "multi_person"), _g("b", ["3", "4"], "single")]
    clean, stats = prune_groups(groups, redundant=set())
    assert [g.identity_group_id for g in clean] == ["b"]
    assert stats["dropped_noisy"] == 1


def test_prune_removes_redundant_faces():
    groups = [_g("a", ["1", "2", "3"], "single")]
    clean, stats = prune_groups(groups, redundant={"2"})
    assert clean[0].faces == ["1", "3"]
    assert stats["removed_faces"] == 1


def test_prune_keeps_singletons_for_negatives():
    groups = [_g("a", ["1"], "single")]
    clean, _ = prune_groups(groups, redundant=set())
    assert len(clean) == 1  # одиночка не выбрасывается (полезен как негатив)


def test_prune_can_disable_noisy_drop():
    groups = [_g("a", ["1", "2"], "collage")]
    clean, stats = prune_groups(groups, redundant=set(), drop_noisy=False)
    assert len(clean) == 1 and stats["dropped_noisy"] == 0


def test_prune_stats_counts():
    groups = [_g("a", ["1", "2"], "meme"), _g("b", ["3", "4", "5"], "single")]
    _, stats = prune_groups(groups, redundant={"4"})
    assert stats["groups_in"] == 2 and stats["groups_out"] == 1
    assert stats["faces_in"] == 5 and stats["faces_out"] == 2  # b: 3 faces - 1 redundant
