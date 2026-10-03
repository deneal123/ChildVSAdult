import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.audit_curation_lineage import summarize

ROOT = Path(__file__).resolve().parents[1]


def _rows():
    post = [{"identity_group_id": "g1", "faces": ["a", "b"]},
            {"identity_group_id": "g2", "faces": ["c"]}]
    mapping = [{"identity_group_id": "g1", "person_id": "p1"}]
    pre = [{"identity_group_id": "p1", "faces": ["a", "b"], "gender_consistency": 0.5},
           {"identity_group_id": "g2", "faces": ["c"]}]
    final = deepcopy(pre)
    final[0]["faces"] = ["a"]
    return post, mapping, pre, final, [{"face_id": "b"}]


def test_mapping_fallback_and_face_yields_are_distinct():
    result = summarize(*_rows())
    assert result["mapped_person_clusters"] == 1
    assert result["unmapped_fallback_groups"] == 1
    assert result["pre_prune_groups"] == 2
    assert result["retained_unique_faces"] == 2
    assert result["dedup_faces_removed_from_retained_groups"] == 1
    assert result["current_redundant_file_replays_final_faces"]


def test_changed_redundant_file_reports_unresolved_replay_not_success():
    post, mapping, pre, final, _ = _rows()
    result = summarize(post, mapping, pre, final, [])
    assert not result["current_redundant_file_replays_final_faces"]
    assert result["dedup_face_set_mismatched_groups"] == 1


def test_face_order_difference_is_not_a_set_difference():
    post, mapping, pre, _, _ = _rows()
    final = deepcopy(pre)
    final[0]["faces"].reverse()
    result = summarize(post, mapping, pre, final, [])
    assert result["dedup_face_set_mismatched_groups"] == 0
    assert result["dedup_face_order_only_mismatches"] == 1
    assert not result["current_redundant_file_replays_final_faces"]


def test_noise_pruning_is_separate_from_dedup():
    post, mapping, pre, final, red = _rows()
    pre[1]["identity_review"] = "multi_person"
    result = summarize(post, mapping, pre, final[:1], red)
    assert result["noisy_groups_dropped"] == 1
    assert result["faces_in_dropped_noisy_groups"] == 1
    assert result["dedup_faces_removed_from_retained_groups"] == 1


@pytest.mark.parametrize("mutation", ["duplicate", "unknown_mapping", "missing_group", "new_face"])
def test_inconsistent_lineage_rejected(mutation):
    post, mapping, pre, final, red = _rows()
    if mutation == "duplicate":
        post.append(post[0])
    elif mutation == "unknown_mapping":
        mapping.append({"identity_group_id": "unknown", "person_id": "p2"})
    elif mutation == "missing_group":
        final.pop()
    else:
        final[0]["faces"].append("new")
    with pytest.raises(ValueError):
        summarize(post, mapping, pre, final, red)


def test_manuscript_counts_follow_bound_lineage_not_conflated_units():
    directory = ROOT / "metrics/curation_lineage_20261003"
    result = directory / "summary.json"
    payload = result.read_bytes()
    metrics = json.loads(payload)
    manifest = json.loads((directory / "summary.manifest.json").read_text(encoding="utf-8"))
    assert manifest["metrics"] == metrics
    assert manifest["outputs"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["outputs"][0]["bytes"] == len(payload)
    for record in manifest["inputs"]:
        path = (ROOT / record["path"]).resolve()
        assert path.is_relative_to(ROOT)
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    main = (ROOT / "latex/papers/journal-1-tbiom/en/main.tex").read_text(encoding="utf-8")
    persons = main.split("\\label{sec:data-persons}", 1)[1].split("\\subsection", 1)[0]
    for key in ("mapped_post_groups", "mapped_person_clusters", "pre_prune_groups", "post_groups"):
        assert f"{metrics[key]:,}".replace(",", "{,}") in persons
    assert "one unmapped two-face group" in persons
    curation = main.split("\\label{sec:data-curation}", 1)[1].split("\\begin{table}", 1)[0]
    assert "does not separate multiple people" in curation
    assert "original dedup artifact is unrecovered" in curation
    assert not metrics["current_redundant_file_replays_final_faces"]
    supplement = (ROOT / "latex/papers/journal-1-tbiom/en/supplement.tex").read_text(encoding="utf-8")
    assert "flagged as probable multi-person" not in supplement
    assert "$21{,}847$ components" in supplement
    assert "singleton fallback gives $21{,}848$ pre-prune groups" in supplement
    assert "not a verified end-to-end replay" in supplement
