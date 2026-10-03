import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.replay_curation_snapshot_global import compare_global_replay


def test_global_policy_removes_shared_face_from_other_group():
    pre = [{"identity_group_id": "a", "faces": ["x", "y"]},
           {"identity_group_id": "b", "faces": ["y"]}]
    final = deepcopy(pre)
    final[0]["faces"], final[1]["faces"] = ["x"], []
    result = compare_global_replay(pre, final, {"a": ["y"]})
    assert result["full_group_records_replayed"]
    assert result["detected_redundant_face_memberships"] == 1
    assert result["globally_removed_pre_face_memberships"] == 2
    assert result["pre_cross_group_shared_face_ids"] == 1
    assert result["final_empty_groups"] == 1


def test_group_local_prune_is_not_global_policy():
    pre = [{"identity_group_id": "a", "faces": ["x", "y"]},
           {"identity_group_id": "b", "faces": ["y"]}]
    final = deepcopy(pre)
    final[0]["faces"] = ["x"]
    result = compare_global_replay(pre, final, {"a": ["y"]})
    assert not result["ordered_faces_replayed"]
    assert result["face_set_mismatched_groups"] == 1


@pytest.mark.parametrize("red", [{"unknown": []}, {"a": ["unknown"]}])
def test_bad_original_red_map_cannot_be_silently_expanded(red):
    with pytest.raises(ValueError):
        compare_global_replay([{"identity_group_id": "a", "faces": ["x"]}], [], red)


def test_record_equality_does_not_imply_unique_ownership_or_person_purity():
    rows = [{"identity_group_id": "a", "faces": ["x"]},
            {"identity_group_id": "b", "faces": ["x"]}]
    result = compare_global_replay(rows, deepcopy(rows), {})
    assert result["full_group_records_replayed"]
    assert result["final_cross_group_shared_face_ids"] == 1
    assert "original dedup file/provenance not recovered" in result["scope"]


def test_real_global_replay_and_manuscript_are_checksum_bound():
    root = Path(__file__).resolve().parents[1]
    directory = root / "metrics/curation_replay_global_20261003_v2"
    payload = (directory / "summary.json").read_bytes()
    result = json.loads(payload)
    manifest = json.loads((directory / "summary.manifest.json").read_text(encoding="utf-8"))
    assert manifest["metrics"] == result
    assert manifest["outputs"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    for record in manifest["inputs"]:
        path = (root / record["path"]).resolve()
        assert path.is_relative_to(root)
        assert path.stat().st_size == record["bytes"]
        with path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == record["sha256"]
    assert result["full_group_records_replayed"]
    assert result["face_set_mismatched_groups"] == result["metadata_mismatched_groups"] == 0
    assert result["dedup_removed_from_retained_groups"] == 8190
    assert result["actual_retained_unique_faces"] == 40586
    assert result["final_cross_group_shared_face_ids"] == 0
    assert result["final_empty_groups"] == 1
    main = (root / "latex/papers/journal-1-tbiom/en/main.tex").read_text(encoding="utf-8")
    assert "Global-face-ID shadow replay at 0.97" in main
    assert "original dedup artifact is unrecovered" in main
