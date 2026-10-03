import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.audit_constituent_integrity import audit


def fixture():
    post = [{"identity_group_id": "g1", "source_post_id": "p1", "faces": ["a"]},
            {"identity_group_id": "g2", "source_post_id": "p2", "faces": ["b"]},
            {"identity_group_id": "g3", "source_post_id": "p3", "faces": ["c"]}]
    mapping = [{"identity_group_id": "g1", "person_id": "q1"},
               {"identity_group_id": "g2", "person_id": "q1"}]
    pre = [{"identity_group_id": "q1", "source_post_id": "p1", "faces": ["a", "b"], "identity_review": "single"},
           {"identity_group_id": "g3", "source_post_id": "p3", "faces": ["c"], "identity_review": "unknown"}]
    cache = [{"post_id": "p1", "category": "single"}, {"post_id": "p2", "category": "multi_person"},
             {"post_id": "p3", "category": "unknown"}]
    pairs = [{"pair_id": "pos", "label": 1, "split": "train", "identity_group_a": "q1", "identity_group_b": "q1"},
             {"pair_id": "neg", "label": 0, "split": "val", "identity_group_a": "g3", "identity_group_b": "q1"},
             {"pair_id": "unaffected", "label": 1, "split": "test", "identity_group_a": "g3", "identity_group_b": "g3"}]
    return post, mapping, pre, deepcopy(pre), cache, pairs


def test_secondary_noisy_caption_detected_and_both_pair_sides_excluded():
    result, candidates = audit(*fixture())
    assert result["retained_groups_with_noisy_constituent"] == 1
    assert result["retained_unique_faces_in_affected_groups"] == 2
    assert result["pair_counts_after_any_noisy_group_exclusion"] == {
        "test_positive": 1, "train_positive": 0, "val_negative": 0,
    }
    assert candidates[0]["representative_category"] == "single"
    assert candidates[0]["noisy_constituent_post_ids"] == ["p2"]


def test_missing_and_unknown_not_treated_as_single():
    post, mapping, pre, final, cache, pairs = fixture()
    pre[1]["identity_review"] = final[1]["identity_review"] = None
    result, _ = audit(post, mapping, pre, final, cache[:2], pairs)
    assert result["constituent_coverage"]["missing_constituent_posts"] == 1
    assert result["constituent_coverage"]["unknown_constituent_posts"] == 0
    assert result["all_constituents_cached_single_retained_groups"] == 0


def test_representative_noisy_group_already_removed_not_additional_candidate():
    post, mapping, pre, final, cache, pairs = fixture()
    pre[0]["identity_review"] = cache[0]["category"] = "multi_person"
    result, candidates = audit(post, mapping, pre, final[1:], cache, pairs[2:])
    assert result["any_constituent_noisy_pre_groups"] == 1
    assert result["retained_groups_with_noisy_constituent"] == 0
    assert candidates == []


@pytest.mark.parametrize("case", ["duplicate_cache", "bad_category", "rep_label", "face_order", "rep_source", "unknown_map", "bad_pair", "duplicate_pair"])
def test_inconsistent_inputs_rejected(case):
    post, mapping, pre, final, cache, pairs = fixture()
    if case == "duplicate_cache":
        cache.append(cache[0])
    elif case == "bad_category":
        cache[0]["category"] = "invalid"
    elif case == "rep_label":
        pre[0]["identity_review"] = "unknown"
    elif case == "face_order":
        pre[0]["faces"].reverse()
    elif case == "rep_source":
        pre[0]["source_post_id"] = "p2"
    elif case == "unknown_map":
        mapping.append({"identity_group_id": "absent", "person_id": "q1"})
    elif case == "bad_pair":
        pairs[0]["identity_group_a"] = "absent"
    else:
        pairs.append(pairs[0])
    with pytest.raises(ValueError):
        audit(post, mapping, pre, final, cache, pairs)


def test_no_caption_or_raw_response_copied_and_inputs_unchanged():
    values = fixture()
    values[4][0]["caption"] = "PRIVATE_CAPTION"
    values[4][0]["response"] = "PRIVATE_RESPONSE"
    original = deepcopy(values)
    result = audit(*values)
    assert values == original
    assert "PRIVATE_CAPTION" not in json.dumps(result)
    assert "PRIVATE_RESPONSE" not in json.dumps(result)


def test_real_audit_bound_and_no_retrained_or_human_gold_claim():
    root = Path(__file__).resolve().parents[1]
    directory = root / "metrics/constituent_integrity_20261003"
    manifest = json.loads((directory / "summary.manifest.json").read_text(encoding="utf-8"))
    for record in manifest["inputs"] + manifest["outputs"]:
        path = (root / record["path"]).resolve()
        assert path.is_relative_to(root)
        assert path.stat().st_size == record["bytes"]
        with path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == record["sha256"]
    result = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert manifest["metrics"] == result
    assert result["retained_groups_with_noisy_constituent"] == 17
    assert result["retained_unique_faces_in_affected_groups"] == 54
    assert result["original_pair_counts"]["test_positive"] == result["pair_counts_after_any_noisy_group_exclusion"]["test_positive"]
    assert "no true-identity purity or retrained effect" in result["scope"]
    private = directory / "private/candidates.jsonl"
    assert len(private.read_text(encoding="utf-8").splitlines()) == 17
