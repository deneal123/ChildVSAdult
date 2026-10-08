"""Versioned full-index coverage, native byte checks and fail-closed publication."""

import json

import pytest

from age_gap.common.manifest import file_record
from scripts import build_publication_artifact_index as original
from scripts import build_publication_artifact_index_v2 as index


def fixture_manifest(root, relative, *, input_path=None, record=None):
    path = root / "metrics" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    input_path = input_path or root / "scripts" / "fixture.py"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        input_path.write_text("fixture", encoding="utf-8")
    current = file_record(input_path)
    current["path"] = input_path.relative_to(root).as_posix()
    path.write_text(json.dumps(dict(experiment="synthetic", inputs=[record or current], outputs=[],
                                    parameters={"private_path": "metrics/study/private/scores.npz"})),
                    encoding="utf-8")
    return path


def test_extended_full_inclusion_preserves_original_and_isolated_globals(tmp_path):
    before = original.INCLUDE, original.PROJECT_ROOT, original.OUTPUT
    isolated = index.isolated_legacy(tmp_path, tmp_path / "new.json")
    assert set(original.INCLUDE) <= set(isolated.INCLUDE)
    assert set(index.EXTRA_INCLUDE) <= set(isolated.INCLUDE)
    assert set(index.CURRENT_REQUIRED) <= set(isolated.INCLUDE)
    assert before == (original.INCLUDE, original.PROJECT_ROOT, original.OUTPUT)
    assert isolated is not original


def test_full_current_coverage_and_publication_not_claimed(tmp_path):
    for relative in index.CURRENT_REQUIRED:
        fixture_manifest(tmp_path, relative)
    canonical = tmp_path / "metrics/publication_artifact_index.json"
    canonical.write_text("untouched canonical index", encoding="utf-8")
    out = tmp_path / "out"
    metrics = index.run(tmp_path, out)
    result = json.loads((out / "publication_artifact_index.json").read_text(encoding="utf-8"))
    assert metrics["execution_complete"] is True
    assert metrics["publication_ready"] is False
    assert metrics["artifact_index_complete"] is False
    assert {entry["manifest"] for entry in result["entries"]} == {
        "metrics/" + relative for relative in index.CURRENT_REQUIRED}
    assert not (set(index.CURRENT_REQUIRED) & set(result["missing_expected_manifests"]))
    assert "strong-backbone summary missing" in result["incomplete_experiments"]
    assert result["index_audit_version"] == 2
    assert "scores.npz" not in json.dumps(result)
    assert canonical.read_text(encoding="utf-8") == "untouched canonical index"
    manifest = json.loads((out / "summary.manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment"] == "publication-artifact-index-audit-v2"
    assert len(manifest["outputs"]) == 1
    assert manifest["parameters"]["canonical_index_updated"] is False


def test_required_current_presentations_missing_are_not_silently_skipped(tmp_path):
    (tmp_path / "metrics").mkdir()
    index.run(tmp_path, tmp_path / "out")
    result = json.loads((tmp_path / "out/publication_artifact_index.json").read_text(encoding="utf-8"))
    assert set(index.CURRENT_REQUIRED) <= set(result["missing_expected_manifests"])
    assert result["complete"] is False


def test_integrity_checks_bytes_and_reuses_snapshot_but_guard_rehashes(tmp_path):
    isolated = index.isolated_legacy(tmp_path, tmp_path / "unused.json")
    audit = index.IntegrityAudit(isolated)
    path = tmp_path / "code.py"
    path.write_text("before", encoding="utf-8")
    current = file_record(path)
    current["path"] = "code.py"
    assert audit.check([current])["valid"] is True
    assert audit.check([{**current, "bytes": current["bytes"] + 1}])["valid"] is False
    assert len(audit.snapshots) == 1
    path.write_text("after!", encoding="utf-8")  # Same length, different real bytes.
    with pytest.raises(RuntimeError, match="ancestry changed"):
        audit.guard()


@pytest.mark.parametrize("field,value", [("bytes", True), ("bytes", -1), ("sha256", "fake"),
                                        ("path", ""), ("path", 12)])
def test_malformed_native_records_rejected(tmp_path, field, value):
    isolated = index.isolated_legacy(tmp_path, tmp_path / "unused.json")
    audit = index.IntegrityAudit(isolated)
    record = dict(path="fake.py", bytes=0, sha256="a" * 64)
    record[field] = value
    with pytest.raises(ValueError, match="file record"):
        audit.check([record])


def test_private_missing_hash_or_name_not_exposed(tmp_path):
    isolated = index.isolated_legacy(tmp_path, tmp_path / "unused.json")
    result = index.IntegrityAudit(isolated).check([
        dict(path="metrics/study/private/Person_Name.jpg", bytes=1, sha256="a" * 64)])
    assert result["private_missing_count"] == 1
    assert result["valid"] is False
    assert "Person_Name" not in json.dumps(result)


def test_mutation_during_publication_removes_only_own_completion_marker(tmp_path, monkeypatch):
    bound = fixture_manifest(tmp_path, index.CURRENT_REQUIRED[0])
    original_writer = index.write_experiment_manifest

    def mutate(*args, **kwargs):
        result = original_writer(*args, **kwargs)
        bound.write_text("changed", encoding="utf-8")
        return result

    monkeypatch.setattr(index, "write_experiment_manifest", mutate)
    with pytest.raises(RuntimeError, match="ancestry changed"):
        index.run(tmp_path, tmp_path / "out")
    assert not (tmp_path / "out/summary.manifest.json").exists()
    assert (tmp_path / "out/publication_artifact_index.json").exists()
    assert bound.read_text(encoding="utf-8") == "changed"


def test_disappearance_during_publication_removes_completion_marker(tmp_path, monkeypatch):
    bound = fixture_manifest(tmp_path, index.CURRENT_REQUIRED[0])
    writer = index.write_experiment_manifest

    def remove(*args, **kwargs):
        result = writer(*args, **kwargs)
        bound.unlink()
        return result

    monkeypatch.setattr(index, "write_experiment_manifest", remove)
    with pytest.raises(FileNotFoundError):
        index.run(tmp_path, tmp_path / "out")
    assert not (tmp_path / "out/summary.manifest.json").exists()


def test_existing_output_refused(tmp_path):
    with pytest.raises(FileExistsError):
        index.run(tmp_path, tmp_path)
