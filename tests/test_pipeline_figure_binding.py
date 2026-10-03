"""Presentation binding must not silently accept changed or inconsistent counts."""

import hashlib
import json

import pytest

from scripts.make_pipeline_figure import load_bound_funnel, pipeline_boxes


def _fixture(tmp_path, change=None):
    funnel = dict(posts=10, photos=20, face_records=18, usable_faces=12,
                  rejected_face_records=6, curated_faces=9, identity_groups=4,
                  person_clusters=5, positive_pairs=7, negative_pairs=7, final_pairs=14)
    if change:
        funnel.update(change)
    data = {"funnel": funnel}
    payload = json.dumps(data).encode()
    path = tmp_path / "data_funnel.json"
    path.write_bytes(payload)
    manifest = {"metrics": data, "outputs": [{"path": "metrics/data_funnel.json",
                "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_verified_counts_and_stage_order(tmp_path):
    path, _ = _fixture(tmp_path)
    boxes = pipeline_boxes(load_bound_funnel(path))
    assert "person merging" in boxes[2][0]
    assert "5 mapped clusters" in boxes[2][1]
    assert "Integrity prune" in boxes[3][0]
    assert "9 faces" in boxes[3][1]
    assert "4 retained groups" in boxes[3][1]
    assert "Recorded-person" in boxes[4][0]
    assert not any("audit pack" in detail for _, detail in boxes)


@pytest.mark.parametrize("change", [
    {"posts": True}, {"photos": -1}, {"person_clusters": 5.0},
    {"face_records": 19}, {"final_pairs": 15}, {"curated_faces": 13},
])
def test_inconsistent_or_invalid_counts_rejected(tmp_path, change):
    path, _ = _fixture(tmp_path, change)
    with pytest.raises(ValueError):
        load_bound_funnel(path)


def test_changed_aggregate_rejected(tmp_path):
    path, _ = _fixture(tmp_path)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="binding"):
        load_bound_funnel(path)


def test_manifest_metrics_mismatch_rejected(tmp_path):
    path, manifest = _fixture(tmp_path)
    manifest["metrics"]["funnel"]["posts"] = 11
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics differ"):
        load_bound_funnel(path)


def test_ambiguous_output_binding_rejected(tmp_path):
    path, manifest = _fixture(tmp_path)
    manifest["outputs"] *= 2
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="binding"):
        load_bound_funnel(path)
