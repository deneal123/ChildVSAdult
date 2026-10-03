import hashlib
from copy import deepcopy

import numpy as np
import pytest

from scripts.replay_curation_snapshot import (
    EMBEDDINGS,
    PRE,
    compare_replay,
    validate_embeddings,
    verify_reference_inputs,
)


def fixture():
    pre = [{"identity_group_id": "a", "faces": ["x", "y"], "identity_review": "single"},
           {"identity_group_id": "b", "faces": ["z"], "identity_review": "multi_person"}]
    final = deepcopy(pre[:1])
    final[0]["faces"] = ["x"]
    return pre, final, {"a": ["y"]}


def test_face_and_metadata_replay_with_noise_removed():
    result = compare_replay(*fixture())
    assert result["ordered_faces_replayed"] and result["full_group_records_replayed"]
    assert result["dedup_removed_from_retained_groups"] == 1
    assert result["noisy_groups_dropped"] == 1


def test_metadata_change_does_not_hide_behind_face_match():
    pre, final, redundant = fixture()
    final[0]["identity_review"] = "unknown"
    result = compare_replay(pre, final, redundant)
    assert result["ordered_faces_replayed"]
    assert not result["full_group_records_replayed"]
    assert result["metadata_mismatched_groups"] == 1


def test_order_difference_not_counted_as_set_difference():
    pre, _, _ = fixture()
    final = deepcopy(pre[:1])
    final[0]["faces"].reverse()
    result = compare_replay(pre, final, {})
    assert result["face_set_mismatched_groups"] == 0
    assert result["face_order_only_mismatches"] == 1
    assert not result["ordered_faces_replayed"]


def test_missing_or_extra_group_prevents_success():
    pre, _, red = fixture()
    result = compare_replay(pre, [{"identity_group_id": "extra", "faces": []}], red)
    assert result["missing_group_count"] == result["extra_group_count"] == 1
    assert not result["full_group_records_replayed"]


@pytest.mark.parametrize("case", ["group", "face", "red_group", "red_face"])
def test_invalid_rows_rejected(case):
    pre, final, red = fixture()
    if case == "group":
        pre.append(pre[0])
    elif case == "face":
        final[0]["faces"] *= 2
    elif case == "red_group":
        red["unknown"] = []
    else:
        red["a"] = ["unknown"]
    with pytest.raises(ValueError):
        compare_replay(pre, final, red)


def test_identical_duplicate_vectors_are_explicitly_canonicalized():
    embeddings = validate_embeddings(np.array(["a", "a"], dtype=object), np.array([[1., 0.], [1., 0.]]))
    assert len(embeddings) == 1


def test_conflicting_duplicate_vectors_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        validate_embeddings(np.array(["a", "a"], dtype=object), np.array([[1., 0.], [0., 1.]]))


@pytest.mark.parametrize("matrix", [np.array([[2., 0.]]), np.array([[np.nan, 0.]]),
                                    np.array([[1, 0]]), np.array([1., 0.])])
def test_invalid_embeddings_rejected(matrix):
    with pytest.raises(ValueError):
        validate_embeddings(np.array(["a"], dtype=object), matrix)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "size", "sha"])
def test_reference_binding_rejects_malformed_or_changed_inputs(tmp_path, fault):
    records = []
    for relative in (PRE, EMBEDDINGS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"controlled fixture")
        records.append({"path": relative, "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    verify_reference_inputs(tmp_path, {"inputs": records})
    if fault == "missing":
        records.pop()
    elif fault == "duplicate":
        records.append(records[0])
    elif fault == "size":
        records[0]["bytes"] += 1
    else:
        records[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify_reference_inputs(tmp_path, {"inputs": records})
