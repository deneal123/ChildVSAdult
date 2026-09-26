from __future__ import annotations

import numpy as np

from scripts.audit_missed_identity_merges import audit_candidates, split_membership


def test_split_membership_uses_both_pair_endpoints() -> None:
    pairs = [
        {"identity_group_a": "g1", "identity_group_b": "g2", "split": "train"},
        {"identity_group_a": "g3", "identity_group_b": "g4", "split": "val"},
    ]
    assert split_membership(pairs) == {
        "g1": {"train"},
        "g2": {"train"},
        "g3": {"val"},
        "g4": {"val"},
    }


def test_cross_split_candidate_is_found_without_returning_vectors() -> None:
    groups = [
        {"identity_group_id": "g-train", "faces": ["f1"]},
        {"identity_group_id": "g-val", "faces": ["f2"]},
        {"identity_group_id": "g-train-other", "faces": ["f3"]},
    ]
    people = {"g-train": "p1", "g-val": "p2", "g-train-other": "p3"}
    splits = {"g-train": {"train"}, "g-val": {"val"}, "g-train-other": {"train"}}
    vectors = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
    candidates, summary = audit_candidates(
        groups, people, splits, ["f1", "f2", "f3"], vectors, threshold=0.9, top_k=2, block_size=1
    )
    assert len(candidates) == 1
    assert {candidates[0]["group_id_a"], candidates[0]["group_id_b"]} == {"g-train", "g-val"}
    assert "centroid" not in candidates[0] and "embedding" not in candidates[0]
    assert summary["candidate_pairs_above_centroid_cosine_threshold"] == 1
    assert summary["face_level_recall_guaranteed"] is False
    assert summary["groups_with_embedding_centroid"] == 3


def test_conflicted_groups_are_reported_and_excluded_from_candidate_search() -> None:
    groups = [
        {"identity_group_id": "conflicted", "faces": ["f1"]},
        {"identity_group_id": "other", "faces": ["f2"]},
    ]
    candidates, summary = audit_candidates(
        groups,
        {"conflicted": "p1", "other": "p2"},
        {"conflicted": {"train", "test"}, "other": {"val"}},
        ["f1", "f2"],
        np.asarray([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32),
        threshold=0.9,
    )
    assert summary["groups_with_split_conflict"] == 1
    assert summary["groups_with_embedding_centroid"] == 1
    assert candidates == []
