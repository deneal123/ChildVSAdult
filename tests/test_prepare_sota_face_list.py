from pathlib import Path

import pytest

from scripts.prepare_sota_face_list import select


def inputs(labels=None):
    groups = [dict(identity_group_id="g", faces=["a", "b"], age_labels=labels or [])]
    splits = [dict(identity_group_id="g", split="train")]
    clusters = [dict(identity_group_id="g", person_id="p")]
    return groups, splits, clusters


def crop(face):
    return Path(face + ".jpg")


def test_conflict_is_masked_and_orphan_does_not_override_retained_age():
    groups, splits, clusters = inputs(
        [
            dict(face_id="a", age=0),
            dict(face_id="a", age=1),
            dict(face_id="b", age=10),
            dict(face_id="outside", age=60),
        ]
    )
    rows, _, counts, inventory = select(groups, splits, clusters, crop)
    assert rows[0]["age"] is None and rows[0]["age_status"] == "conflict_masked"
    assert rows[0]["age_candidates"] == [0, 1]
    assert rows[1]["age"] == 10
    assert counts["retained_age_conflict_masked"] == 1
    assert counts["train_unmapped_or_orphan_age_labels"] == 1
    assert not inventory["input_contract_clean"]


def test_zero_is_retained_and_missing_is_explicit():
    rows, _, counts, _ = select(*inputs([dict(face_id="a", age=0)]), crop)
    assert rows[0]["age"] == 0 and rows[1]["age"] is None
    assert counts["retained_age_missing"] == 1


def test_missing_crop_is_counted_and_group_minimum_rechecked():
    groups, splits, clusters = inputs()
    groups[0]["faces"].append("c")
    rows, decisions, counts, _ = select(
        groups, splits, clusters, lambda f: None if f == "b" else crop(f)
    )
    assert [r["face_id"] for r in rows] == ["a", "c"]
    assert counts["missing_crops"] == 1 and decisions[0]["reason"] == "missing_crop"
    with pytest.raises(ValueError, match="no eligible"):
        select(*inputs(), lambda f: None if f == "b" else crop(f))


def test_missing_person_never_uses_group_id_fallback():
    groups, splits, clusters = inputs()
    groups.append(dict(identity_group_id="unknown", faces=["x", "y"], age_labels=[]))
    splits.append(dict(identity_group_id="unknown", split="train"))
    rows, decisions, counts, _ = select(groups, splits, clusters, crop)
    assert all(row["person_id"] == "p" for row in rows)
    assert counts["missing_recorded_person_groups"] == 1
    assert decisions[0]["reason"] == "missing_recorded_person"


def test_split_overlap_is_fatal_not_repaired_by_filtering():
    groups, splits, clusters = inputs()
    groups.append(dict(identity_group_id="h", faces=["x", "y"], age_labels=[]))
    splits.append(dict(identity_group_id="h", split="test"))
    clusters.append(dict(identity_group_id="h", person_id="p"))
    with pytest.raises(ValueError, match="ambiguous"):
        select(groups, splits, clusters, crop)


def test_order_invariance_of_face_list_and_age_candidates():
    groups, splits, clusters = inputs([dict(face_id="a", age=1), dict(face_id="a", age=0)])
    expected = select(groups, splits, clusters, crop)[0]
    groups[0]["faces"].reverse()
    groups[0]["age_labels"].reverse()
    assert select(groups, splits, clusters, crop)[0] == expected


def test_heldout_groups_never_enter_list():
    groups, splits, clusters = inputs()
    groups.append(dict(identity_group_id="h", faces=["x", "y"], age_labels=[]))
    splits.append(dict(identity_group_id="h", split="test"))
    clusters.append(dict(identity_group_id="h", person_id="q"))
    rows, _, counts, _ = select(groups, splits, clusters, crop)
    assert len(rows) == 2 and counts["nontrain_groups"] == 1
