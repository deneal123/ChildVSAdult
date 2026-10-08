import json

import pytest

import scripts.prepare_human_audit_handoff_v1 as module


@pytest.fixture
def source(tmp_path, monkeypatch):
    base = tmp_path / "source"
    (base / "images").mkdir(parents=True)
    relative = "images/" + "a" * 24 + ".png"
    (base / relative).write_bytes(b"synthetic-image-only")
    row = dict(task_id="synthetic-only", images=[relative], response=None)
    for name in ("annotator_a", "annotator_b"):
        (base / f"{name}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    for name in ("audit_key.jsonl", "adjudicated_gold.template.jsonl", "images.manifest.json",
                 "age_baseline_predictions.jsonl", "age_baseline_predictions.manifest.json"):
        (base / name).write_text("{}\n", encoding="utf-8")
    (base / "README.md").write_text("synthetic instructions", encoding="utf-8")
    monkeypatch.setattr(module, "validate", lambda *args: {"synthetic_fixture": True})
    return base


def test_reviewers_never_receive_keys_predictions_or_each_others_tasks(source, tmp_path):
    out = tmp_path / "handoff"
    module.prepare(source, out)
    for name in ("annotator_a", "annotator_b"):
        files = {p.relative_to(out / name).as_posix() for p in (out / name).rglob("*") if p.is_file()}
        assert files == {f"{name}.jsonl", "README.md", "images/" + "a" * 24 + ".png"}
        assert (out / name / f"{name}.jsonl").read_bytes() == (source / f"{name}.jsonl").read_bytes()
    manifest = json.loads((out / "coordinator.manifest.json").read_text(encoding="utf-8"))
    assert manifest["metrics"]["task_counts"] == dict(annotator_a=1, annotator_b=1)
    assert manifest["metrics"]["human_annotation_complete"] is False
    assert manifest["metrics"]["external_transfer_performed"] is False


def test_existing_output_is_preserved(source, tmp_path):
    out = tmp_path / "handoff"
    out.mkdir()
    marker = out / "marker"
    marker.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="fresh output"):
        module.prepare(source, out)
    assert marker.read_bytes() == b"preserve"


def test_path_traversal_refused_before_output(source, tmp_path):
    row = dict(task_id="synthetic", images=["../secret.png"], response=None)
    (source / "annotator_a.jsonl").write_text(json.dumps(row), encoding="utf-8")
    out = tmp_path / "handoff"
    with pytest.raises(ValueError, match="unsafe source"):
        module.prepare(source, out)
    assert not out.exists()


def test_validator_failure_does_not_create_package(source, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("invalid source pack")
    monkeypatch.setattr(module, "validate", fail)
    out = tmp_path / "handoff"
    with pytest.raises(RuntimeError, match="invalid source"):
        module.prepare(source, out)
    assert not out.exists()
