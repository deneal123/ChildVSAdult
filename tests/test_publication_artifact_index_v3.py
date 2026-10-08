"""Synthetic tests for the bounded current-v3 publication index audit.

Every fixture is generated in a tiny temporary directory and every hash is over
synthetic bytes. No real faces, weights, captions or row paths are read, so a
green suite is a physical-snapshot audit check only: it is NOT evidence of
coordinator acceptance, native36 matrix completion, ethics/identity/disclosure
clearance or a real upload.
"""

import json
from pathlib import Path

import pytest

from age_gap.common.manifest import file_record
from scripts import build_publication_artifact_index as original
from scripts import build_publication_artifact_index_v2 as v2
from scripts import build_publication_artifact_index_v3 as index

HISTORICAL_V2_REQUIRED = tuple(v2.CURRENT_REQUIRED)
NEW_V2_V3_PATTERNS = (
    "common_mechanism_presentation*_v2_*/presentation.manifest.json",
    "age_probe_absolute_presentation*_v3_*/presentation.manifest.json",
    "common_pair_protocol*_v3_*/summary.manifest.json",
    "external_layout*_v2_*/summary.manifest.json",
    "scaling_layout*_v2_*/summary.manifest.json",
)


def fixture_manifest(root, relative, *, records=None, parameters=None):
    path = root / "metrics" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    source = root / "scripts" / "fixture.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        source.write_text("synthetic fixture", encoding="utf-8")
    record = file_record(source)
    record["path"] = source.relative_to(root).as_posix()
    path.write_text(json.dumps(dict(
        experiment="synthetic", inputs=records if records is not None else [record], outputs=[],
        parameters=parameters if parameters is not None else {})), encoding="utf-8")
    return path


def test_current_required_is_exactly_the_current_evidence_and_not_historical():
    assert set(index.CURRENT_REQUIRED) == {
        "common_mechanism_presentation9_v2_20261008/presentation.manifest.json",
        "age_probe_absolute_presentation14_v3_full_20261008/presentation.manifest.json",
        "strong_fixed8_roc_presentation9_v1_20261008/presentation.manifest.json",
        "common_pair_protocol7_v2_20261008/summary.manifest.json",
        "external_layout_v2_20261008/summary.manifest.json",
        "scaling_layout_v2_20261008/summary.manifest.json"}
    assert len(index.CURRENT_REQUIRED) == 6
    assert not (set(index.CURRENT_REQUIRED) & set(HISTORICAL_V2_REQUIRED))


def test_extended_inclusion_covers_new_v2_v3_patterns_without_dropping_v1():
    assert set(NEW_V2_V3_PATTERNS) <= set(index.EXTRA_INCLUDE)
    isolated = index.isolated_v2()
    legacy = isolated.isolated_legacy(Path.cwd(), Path.cwd() / "unused.json")
    assert set(original.INCLUDE) <= set(legacy.INCLUDE)
    assert set(index.EXTRA_INCLUDE) <= set(legacy.INCLUDE)
    assert set(index.CURRENT_REQUIRED) <= set(legacy.INCLUDE)


def test_isolated_reuse_does_not_mutate_canonical_globals(tmp_path):
    original_before = original.INCLUDE, original.PROJECT_ROOT, original.OUTPUT
    v2_before = v2.EXTRA_INCLUDE, v2.CURRENT_REQUIRED
    index.isolated_v2()
    for relative in index.CURRENT_REQUIRED:
        fixture_manifest(tmp_path, relative)
    index.run(tmp_path, tmp_path / "out")
    assert original_before == (original.INCLUDE, original.PROJECT_ROOT, original.OUTPUT)
    assert v2_before == (v2.EXTRA_INCLUDE, v2.CURRENT_REQUIRED)
    assert index.CURRENT_REQUIRED != v2.CURRENT_REQUIRED


def test_missing_current_evidence_is_flagged_not_silently_skipped(tmp_path):
    (tmp_path / "metrics").mkdir()
    metrics = index.run(tmp_path, tmp_path / "out")
    result = json.loads((tmp_path / "out/publication_artifact_index.json").read_text(encoding="utf-8"))
    assert set(index.CURRENT_REQUIRED) <= set(result["missing_expected_manifests"])
    assert result["complete"] is False
    assert result["index_audit_version"] == 3
    assert metrics["execution_complete"] is True
    assert metrics["publication_ready"] is False
    assert metrics["physical_snapshot_complete"] is False
    assert set(result["current_required_manifests"]) == {
        "metrics/" + name for name in index.CURRENT_REQUIRED}


def test_current_v2_v3_discovery_and_publication_not_claimed(tmp_path):
    for relative in index.CURRENT_REQUIRED:
        fixture_manifest(tmp_path, relative)
    fixture_manifest(tmp_path, "common_pair_protocol7_v3_20261009/summary.manifest.json")
    fixture_manifest(tmp_path, "common_mechanism_presentation10_v2_20261009/presentation.manifest.json")
    canonical = tmp_path / "metrics/publication_artifact_index.json"
    canonical.write_text("untouched canonical index", encoding="utf-8")
    metrics = index.run(tmp_path, tmp_path / "out")
    result = json.loads((tmp_path / "out/publication_artifact_index.json").read_text(encoding="utf-8"))
    discovered = {entry["manifest"] for entry in result["entries"]}
    assert "metrics/common_pair_protocol7_v3_20261009/summary.manifest.json" in discovered
    assert "metrics/common_mechanism_presentation10_v2_20261009/presentation.manifest.json" in discovered
    assert not (set(index.CURRENT_REQUIRED) & set(result["missing_expected_manifests"]))
    assert metrics["publication_ready"] is False
    assert result["publication_ready"] is False
    assert "scores.npz" not in json.dumps(result)
    assert canonical.read_text(encoding="utf-8") == "untouched canonical index"
    manifest = json.loads((tmp_path / "out/summary.manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment"] == "publication-artifact-index-audit-v3"
    assert manifest["parameters"]["canonical_index_updated"] is False
    assert manifest["metrics"]["publication_ready"] is False


def test_integrity_checks_bytes_and_same_size_checksum_mutation(tmp_path):
    isolated = index.isolated_v2()
    legacy = isolated.isolated_legacy(tmp_path, tmp_path / "unused.json")
    audit = isolated.IntegrityAudit(legacy)
    path = tmp_path / "code.py"
    path.write_text("before", encoding="utf-8")
    record = file_record(path)
    record["path"] = "code.py"
    assert audit.check([record])["valid"] is True
    assert audit.check([{**record, "bytes": record["bytes"] + 1}])["valid"] is False
    path.write_text("after!", encoding="utf-8")  # Same size, different real bytes.
    with pytest.raises(RuntimeError, match="ancestry changed"):
        audit.guard()


@pytest.mark.parametrize("field,value", [("bytes", True), ("bytes", -1), ("sha256", "fake"),
                                        ("path", ""), ("path", 12)])
def test_malformed_native_records_rejected(tmp_path, field, value):
    isolated = index.isolated_v2()
    legacy = isolated.isolated_legacy(tmp_path, tmp_path / "unused.json")
    audit = isolated.IntegrityAudit(legacy)
    record = dict(path="fake.py", bytes=0, sha256="a" * 64)
    record[field] = value
    with pytest.raises(ValueError, match="file record"):
        audit.check([record])


def test_private_paths_and_row_ids_not_exposed(tmp_path):
    fixture_manifest(
        tmp_path, index.CURRENT_REQUIRED[0],
        records=[dict(path="metrics/study/private/Person_Name.jpg", bytes=1, sha256="a" * 64)],
        parameters={"private_path": "metrics/study/private/scores.npz"})
    index.run(tmp_path, tmp_path / "out")
    result = json.loads((tmp_path / "out/publication_artifact_index.json").read_text(encoding="utf-8"))
    text = json.dumps(result)
    assert "Person_Name" not in text
    assert "scores.npz" not in text
    entry = result["entries"][0]
    assert entry["private_input_count"] == 1
    assert entry["input_integrity"]["private_missing_count"] == 1


def test_mutation_during_publication_removes_only_own_completion_marker(tmp_path, monkeypatch):
    import age_gap.common.manifest as manifest_module

    bound = fixture_manifest(tmp_path, index.CURRENT_REQUIRED[0])
    writer = manifest_module.write_experiment_manifest

    def mutate(*args, **kwargs):
        result = writer(*args, **kwargs)
        bound.write_text("changed", encoding="utf-8")
        return result

    monkeypatch.setattr(manifest_module, "write_experiment_manifest", mutate)
    with pytest.raises(RuntimeError, match="ancestry changed"):
        index.run(tmp_path, tmp_path / "out")
    assert not (tmp_path / "out/summary.manifest.json").exists()
    assert (tmp_path / "out/publication_artifact_index.json").exists()
    assert bound.read_text(encoding="utf-8") == "changed"


def test_disappearance_during_publication_removes_completion_marker(tmp_path, monkeypatch):
    import age_gap.common.manifest as manifest_module

    bound = fixture_manifest(tmp_path, index.CURRENT_REQUIRED[0])
    writer = manifest_module.write_experiment_manifest

    def remove(*args, **kwargs):
        result = writer(*args, **kwargs)
        bound.unlink()
        return result

    monkeypatch.setattr(manifest_module, "write_experiment_manifest", remove)
    with pytest.raises(FileNotFoundError):
        index.run(tmp_path, tmp_path / "out")
    assert not (tmp_path / "out/summary.manifest.json").exists()


def test_existing_output_refused(tmp_path):
    with pytest.raises(FileExistsError):
        index.run(tmp_path, tmp_path)


def test_completion_manifest_binds_canonical_producing_sources(tmp_path):
    fixture_manifest(tmp_path, index.CURRENT_REQUIRED[0])
    index.run(tmp_path, tmp_path / "out")
    manifest = json.loads((tmp_path / "out/summary.manifest.json").read_text(encoding="utf-8"))
    paths = {record["path"].replace("\\", "/") for record in manifest["inputs"]}
    assert {"scripts/build_publication_artifact_index_v3.py",
            "scripts/build_publication_artifact_index_v2.py",
            "scripts/build_publication_artifact_index.py",
            "scripts/export_public_evidence.py",
            "src/age_gap/common/manifest.py",
            "src/age_gap/common/io.py"} <= paths
    assert not any("pi-workers" in path for path in paths)
