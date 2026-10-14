"""Synthetic tests for the canonical v6 publication index audit.

Every fixture uses tiny pytest temporary synthetic bytes. No real faces, weights,
benchmark data or native index is read/run. A green suite is a physical-snapshot
contract check only: it is NOT evidence of
acceptance of the newest presentations, native36 matrix completion,
ethics/identity/disclosure clearance or a real upload.

The canonical producer is imported normally; no scratch import dependency.
"""

import json
from pathlib import Path

import pytest

import age_gap.common.manifest as manifest_module
from age_gap.common.manifest import file_record
from scripts import build_publication_artifact_index as original
from scripts import build_publication_artifact_index_v2 as v2
from scripts import build_publication_artifact_index_v3 as v3
from scripts import build_publication_artifact_index_v6 as v6

V3_REQUIRED = tuple(v3.CURRENT_REQUIRED)
NEWEST_PATTERNS = (
    "common_mechanism_presentation11_v2_*/presentation.manifest.json",
    "age_probe_absolute_presentation16_v3_full_*/presentation.manifest.json",
    "strong_fixed8_roc_presentation11_v1_*/presentation.manifest.json",
)


@pytest.fixture()
def synthetic_root(tmp_path):
    """Pytest-owned synthetic tree; no manual deletion or native-data reads."""
    return tmp_path


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


def read_index(out):
    return json.loads((out / "publication_artifact_index.json").read_text(encoding="utf-8"))


def test_current_required_is_exactly_the_new_v6_evidence():
    assert set(v6.CURRENT_REQUIRED) == {
        "common_mechanism_presentation11_v2_e339_20261009/presentation.manifest.json",
        "age_probe_absolute_presentation16_v3_full_e339_20261009/presentation.manifest.json",
        "strong_fixed8_roc_presentation11_v1_e339_20261009/presentation.manifest.json",
        "common_pair_protocol7_v2_20261008/summary.manifest.json",
        "external_layout_v2_20261008/summary.manifest.json",
        "scaling_layout_v2_20261008/summary.manifest.json",
        "strong_native_trajectories10_v3_e326_20261009/summary.manifest.json",
        "strong_history_presentation10_v1_layout2_e337_20261009/history.manifest.json",
        "strong_fixed_seed_mean4_v1_e350_20261009/summary.manifest.json",
        "fixed_checkpoint_means_presentation_v1_e362_20261009/presentation.manifest.json",
        "checkpoint_validation_tailhigh_s2_v6_e358_20261009/summary.manifest.json",
        "fixed_validation_presentation_v1_e368_20261009/presentation.manifest.json",
        "oriented_cuda_noise_perm1_v2_e381_20261009/training-bound.manifest.json",
        "oriented_cuda_noise_perm2_v2_e388_20261009/training-bound.manifest.json",
        "noise_perm12_cuda_fgnet_v2_e391_20261009/summary.manifest.json",
        "noise_permutations_presentation_v1_e397_20261009/presentation.manifest.json",
        "adaface_fixed8_cuda_lookalike_head_low_s42_e393_20261009/summary.manifest.json",
        "adaface_lookalike_head_low_s42_fgnet_e407_20261009/summary.manifest.json",
        "common_mechanism_lookalike_head_low_s42_e429_20261009/summary.manifest.json",
        "submission_visual_e434_20261009/summary.manifest.json",
    }
    assert len(v6.CURRENT_REQUIRED) == 20
    # Exactly the three inherited pair/layout literals overlap with v3; the three
    # newest mechanism/age/roc literals are new in v6.
    assert set(v6.CURRENT_REQUIRED) & set(V3_REQUIRED) == {
        "common_pair_protocol7_v2_20261008/summary.manifest.json",
        "external_layout_v2_20261008/summary.manifest.json",
        "scaling_layout_v2_20261008/summary.manifest.json"}


def test_extended_inclusion_covers_newest_patterns_without_dropping_v1():
    # v6 keeps every v3 glob and adds the newest literals via CURRENT_REQUIRED,
    # which isolated_legacy merges into the effective INCLUDE.
    assert set(v3.EXTRA_INCLUDE) <= set(v6.EXTRA_INCLUDE)
    v3_iso = v6.isolated_v3()
    v2_iso = v3_iso.isolated_v2()
    legacy = v2_iso.isolated_legacy(Path.cwd(), Path.cwd() / "unused.json")
    assert set(original.INCLUDE) <= set(legacy.INCLUDE)
    assert set(v6.EXTRA_INCLUDE) <= set(legacy.INCLUDE)
    assert set(v6.CURRENT_REQUIRED) <= set(legacy.INCLUDE)
    for newest in v6.CURRENT_REQUIRED[:3]:
        assert newest in legacy.INCLUDE
    assert set(NEWEST_PATTERNS).isdisjoint(v6.EXTRA_INCLUDE)


def test_exact_current_set_flags_missing_newest_gate_despite_historical_discovery(synthetic_root):
    missing_newest = "common_mechanism_presentation11_v2_e339_20261009/presentation.manifest.json"
    for relative in v6.CURRENT_REQUIRED:
        if relative != missing_newest:
            fixture_manifest(synthetic_root, relative)
    # A historical v2 presentation and a newer non-required pair gate are both discovered.
    fixture_manifest(synthetic_root,
                     "common_mechanism_presentation9_v2_20261008/presentation.manifest.json")
    fixture_manifest(synthetic_root, "common_pair_protocol7_v3_20261009/summary.manifest.json")

    metrics = v6.run(synthetic_root, synthetic_root / "out")
    result = read_index(synthetic_root / "out")
    discovered = {entry["manifest"] for entry in result["entries"]}
    assert "metrics/common_mechanism_presentation9_v2_20261008/presentation.manifest.json" in discovered
    assert "metrics/common_pair_protocol7_v3_20261009/summary.manifest.json" in discovered
    assert missing_newest in result["missing_expected_manifests"]
    assert result["complete"] is False
    assert result["index_audit_version"] == 6
    assert metrics["execution_complete"] is True
    assert metrics["publication_ready"] is False
    assert metrics["physical_snapshot_complete"] is False
    assert set(result["current_required_manifests"]) == {
        "metrics/" + name for name in v6.CURRENT_REQUIRED}
    # The non-required newer gate is discovery-only, never promoted to required.
    assert "metrics/common_pair_protocol7_v3_20261009/summary.manifest.json" not in result[
        "missing_expected_manifests"]


def test_strict_semantic_incomplete_is_preserved(synthetic_root):
    for relative in v6.CURRENT_REQUIRED:
        fixture_manifest(synthetic_root, relative)
    campaign = synthetic_root / "metrics/strong_backbone_fixed8_bn_frozen_v1_20260101"
    campaign.mkdir(parents=True)
    (campaign / "summary.json").write_text(json.dumps(
        {"campaign": {"complete": False, "completed_runs": 5, "expected_runs": 36}}), encoding="utf-8")

    metrics = v6.run(synthetic_root, synthetic_root / "out")
    result = read_index(synthetic_root / "out")
    assert any("incomplete" in item for item in result["incomplete_experiments"])
    assert result["complete"] is False
    assert metrics["incomplete_count"] >= 1
    assert metrics["publication_ready"] is False


def test_isolated_reuse_does_not_mutate_canonical_globals(synthetic_root):
    original_before = (original.INCLUDE, original.PROJECT_ROOT, original.OUTPUT)
    v2_before = (v2.EXTRA_INCLUDE, v2.CURRENT_REQUIRED)
    v3_before = (v3.EXTRA_INCLUDE, v3.CURRENT_REQUIRED)
    for relative in v6.CURRENT_REQUIRED:
        fixture_manifest(synthetic_root, relative)

    v6.run(synthetic_root, synthetic_root / "out")

    assert original_before == (original.INCLUDE, original.PROJECT_ROOT, original.OUTPUT)
    assert v2_before == (v2.EXTRA_INCLUDE, v2.CURRENT_REQUIRED)
    assert v3_before == (v3.EXTRA_INCLUDE, v3.CURRENT_REQUIRED)
    assert v6.CURRENT_REQUIRED != v3.CURRENT_REQUIRED


def test_no_canonical_index_or_global_mutation(synthetic_root):
    for relative in v6.CURRENT_REQUIRED:
        fixture_manifest(synthetic_root, relative)
    canonical = synthetic_root / "metrics/publication_artifact_index.json"
    canonical.write_text("untouched canonical index", encoding="utf-8")

    v6.run(synthetic_root, synthetic_root / "out")

    assert canonical.read_text(encoding="utf-8") == "untouched canonical index"
    assert (synthetic_root / "out/publication_artifact_index.json").exists()


def test_completion_binds_v6_v3_v2_v1_and_shared_sources(synthetic_root):
    fixture_manifest(synthetic_root, v6.CURRENT_REQUIRED[0])
    v6.run(synthetic_root, synthetic_root / "out")
    manifest = json.loads(
        (synthetic_root / "out/summary.manifest.json").read_text(encoding="utf-8"))
    paths = {record["path"].replace("\\", "/") for record in manifest["inputs"]}
    assert any(path.endswith("build_publication_artifact_index_v6.py") for path in paths)
    assert any(path.endswith("scripts/build_publication_artifact_index_v3.py") for path in paths)
    assert any(path.endswith("scripts/build_publication_artifact_index_v2.py") for path in paths)
    assert any(path.endswith("scripts/build_publication_artifact_index.py") for path in paths)
    assert "scripts/export_public_evidence.py" in paths
    assert "src/age_gap/common/manifest.py" in paths
    assert "src/age_gap/common/io.py" in paths
    assert "scripts/build_publication_artifact_index_v6.py" in paths
    assert "scripts/build_publication_artifact_index_v4.py" in paths
    assert "scripts/build_publication_artifact_index_v5.py" in paths
    assert not any("pi-workers" in path for path in paths)


def test_private_paths_and_row_ids_not_exposed(synthetic_root):
    fixture_manifest(
        synthetic_root, v6.CURRENT_REQUIRED[0],
        records=[dict(path="metrics/study/private/Person_Name.jpg", bytes=1, sha256="a" * 64)],
        parameters={"private_path": "metrics/study/private/scores.npz"})
    v6.run(synthetic_root, synthetic_root / "out")
    result = read_index(synthetic_root / "out")
    text = json.dumps(result)
    assert "Person_Name" not in text
    assert "scores.npz" not in text
    entry = result["entries"][0]
    assert entry["private_input_count"] == 1
    assert entry["input_integrity"]["private_missing_count"] == 1
    # The controlled record's real digest never reaches the sanitized payload.
    assert "a" * 64 not in text


def test_manifest_mutation_during_publication_removes_only_own_completion_marker(
        synthetic_root, monkeypatch):
    bound = fixture_manifest(synthetic_root, v6.CURRENT_REQUIRED[0])
    writer = manifest_module.write_experiment_manifest

    def mutate(*args, **kwargs):
        result = writer(*args, **kwargs)
        bound.write_text("changed", encoding="utf-8")
        return result

    monkeypatch.setattr(manifest_module, "write_experiment_manifest", mutate)
    with pytest.raises(RuntimeError, match="ancestry changed"):
        v6.run(synthetic_root, synthetic_root / "out")
    assert not (synthetic_root / "out/summary.manifest.json").exists()
    assert (synthetic_root / "out/publication_artifact_index.json").exists()
    assert bound.read_text(encoding="utf-8") == "changed"


def test_source_mutation_during_publication_fails_closed_without_marker(
        synthetic_root, monkeypatch):
    fixture_manifest(synthetic_root, v6.CURRENT_REQUIRED[0])
    extra = synthetic_root / "scripts" / "extra_source.py"
    extra.write_text("before", encoding="utf-8")
    real_sources = v6.producing_sources
    monkeypatch.setattr(v6, "producing_sources",
                        lambda a, b, c: (*real_sources(a, b, c), extra.resolve()))
    writer = manifest_module.write_experiment_manifest

    def mutate(*args, **kwargs):
        result = writer(*args, **kwargs)
        extra.write_text("after!", encoding="utf-8")  # Same size, real bytes differ.
        return result

    monkeypatch.setattr(manifest_module, "write_experiment_manifest", mutate)
    with pytest.raises(RuntimeError, match="ancestry changed"):
        v6.run(synthetic_root, synthetic_root / "out")
    assert not (synthetic_root / "out/summary.manifest.json").exists()


def test_source_disappearance_during_publication_removes_completion_marker(
        synthetic_root, monkeypatch):
    fixture_manifest(synthetic_root, v6.CURRENT_REQUIRED[0])
    extra = synthetic_root / "scripts" / "extra_source.py"
    extra.write_text("bound", encoding="utf-8")
    real_sources = v6.producing_sources
    monkeypatch.setattr(v6, "producing_sources",
                        lambda a, b, c: (*real_sources(a, b, c), extra.resolve()))
    writer = manifest_module.write_experiment_manifest

    def remove(*args, **kwargs):
        result = writer(*args, **kwargs)
        extra.unlink()
        return result

    monkeypatch.setattr(manifest_module, "write_experiment_manifest", remove)
    with pytest.raises(FileNotFoundError):
        v6.run(synthetic_root, synthetic_root / "out")
    assert not (synthetic_root / "out/summary.manifest.json").exists()


def test_manifest_selection_change_during_publication_is_fail_closed(
        synthetic_root, monkeypatch):
    bound = fixture_manifest(synthetic_root, v6.CURRENT_REQUIRED[0])
    writer = manifest_module.write_experiment_manifest

    def remove(*args, **kwargs):
        result = writer(*args, **kwargs)
        bound.unlink()
        return result

    monkeypatch.setattr(manifest_module, "write_experiment_manifest", remove)
    with pytest.raises(RuntimeError, match="selection changed"):
        v6.run(synthetic_root, synthetic_root / "out")
    assert not (synthetic_root / "out/summary.manifest.json").exists()


def test_existing_output_refused(synthetic_root):
    with pytest.raises(FileExistsError):
        v6.run(synthetic_root, synthetic_root)


def test_manifest_selection_change_before_writing_is_fail_closed(
        synthetic_root, monkeypatch):
    fixture_manifest(synthetic_root, v6.CURRENT_REQUIRED[0])
    phantom = (synthetic_root / "metrics" / "phantom" / "summary.manifest.json").resolve()
    real_isolated_v3 = v6.isolated_v3

    class Proxy:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def isolated_v2(self):
            inner = self._inner.isolated_v2()
            real = inner.manifest_paths
            calls = {"n": 0}

            def wrapped(legacy):
                calls["n"] += 1
                paths = real(legacy)
                return set(paths) | {phantom} if calls["n"] > 1 else paths

            inner.manifest_paths = wrapped
            return inner

    monkeypatch.setattr(v6, "isolated_v3", lambda: Proxy(real_isolated_v3()))
    with pytest.raises(RuntimeError, match="selection changed"):
        v6.run(synthetic_root, synthetic_root / "out")
    # The pre-write guard fires before the sanitized index and marker exist.
    assert not (synthetic_root / "out/publication_artifact_index.json").exists()
    assert not (synthetic_root / "out/summary.manifest.json").exists()


def test_publication_ready_is_always_false(synthetic_root):
    metrics = v6.run(synthetic_root, synthetic_root / "out")
    result = read_index(synthetic_root / "out")
    manifest = json.loads(
        (synthetic_root / "out/summary.manifest.json").read_text(encoding="utf-8"))
    assert metrics["publication_ready"] is False
    assert result["publication_ready"] is False
    assert manifest["metrics"]["publication_ready"] is False
    assert manifest["experiment"] == "publication-artifact-index-audit-v6"
    assert manifest["parameters"]["canonical_index_updated"] is False


@pytest.mark.parametrize("change", ["appear", "disappear", "empty_wildcard_directory"])
def test_semantic_gate_selection_change_after_writing_removes_marker(
        synthetic_root, monkeypatch, change):
    fixture_manifest(synthetic_root, v6.CURRENT_REQUIRED[0])
    gate = synthetic_root / "metrics/matched_agegap_arms/summary.json"
    gate.parent.mkdir(parents=True)
    if change == "disappear":
        gate.write_text('{"complete": false}', encoding="utf-8")
    writer = manifest_module.write_experiment_manifest

    def mutate(*args, **kwargs):
        result = writer(*args, **kwargs)
        if change == "disappear":
            gate.unlink()
        elif change == "appear":
            gate.write_text('{"complete": false}', encoding="utf-8")
        else:
            (synthetic_root / "metrics/lfw_bound_evaluation_new").mkdir()
        return result

    monkeypatch.setattr(manifest_module, "write_experiment_manifest", mutate)
    with pytest.raises(RuntimeError, match="semantic gate selection changed"):
        v6.run(synthetic_root, synthetic_root / "out")
    assert not (synthetic_root / "out/summary.manifest.json").exists()


@pytest.mark.parametrize("directory", ["lfw_bound_evaluation_new",
                                        "strong_backbone_fixed8_bn_frozen_new"])
def test_new_empty_semantic_directory_before_writing_is_refused(
        synthetic_root, monkeypatch, directory):
    fixture_manifest(synthetic_root, v6.CURRENT_REQUIRED[0])
    real = v6.isolated_v3

    def load():
        parent = real()
        load_v2 = parent.isolated_v2

        def isolated_v2():
            inner = load_v2()
            load_legacy = inner.isolated_legacy

            def isolated_legacy(*args):
                legacy = load_legacy(*args)
                original_main = legacy.main

                def mutate():
                    original_main()
                    (synthetic_root / "metrics" / directory).mkdir()

                legacy.main = mutate
                return legacy

            inner.isolated_legacy = isolated_legacy
            return inner

        parent.isolated_v2 = isolated_v2
        return parent

    monkeypatch.setattr(v6, "isolated_v3", load)
    with pytest.raises(RuntimeError, match="semantic gate selection changed"):
        v6.run(synthetic_root, synthetic_root / "out")
    assert not (synthetic_root / "out/publication_artifact_index.json").exists()
    assert not (synthetic_root / "out/summary.manifest.json").exists()


def test_semantic_gate_same_size_mutation_after_writing_is_refused(
        synthetic_root, monkeypatch):
    fixture_manifest(synthetic_root, v6.CURRENT_REQUIRED[0])
    gate = synthetic_root / "metrics/matched_agegap_arms/summary.json"
    gate.parent.mkdir(parents=True)
    gate.write_text('{"complete": false, "note": "a"}', encoding="utf-8")
    writer = manifest_module.write_experiment_manifest

    def mutate(*args, **kwargs):
        result = writer(*args, **kwargs)
        gate.write_text('{"complete": false, "note": "b"}', encoding="utf-8")
        return result

    monkeypatch.setattr(manifest_module, "write_experiment_manifest", mutate)
    with pytest.raises(RuntimeError, match="ancestry changed"):
        v6.run(synthetic_root, synthetic_root / "out")
    assert not (synthetic_root / "out/summary.manifest.json").exists()


def test_written_index_mutation_after_manifest_is_refused(synthetic_root, monkeypatch):
    fixture_manifest(synthetic_root, v6.CURRENT_REQUIRED[0])
    writer = manifest_module.write_experiment_manifest

    def mutate(*args, **kwargs):
        result = writer(*args, **kwargs)
        output = synthetic_root / "out/publication_artifact_index.json"
        output.write_bytes(b"X" * output.stat().st_size)
        return result

    monkeypatch.setattr(manifest_module, "write_experiment_manifest", mutate)
    with pytest.raises(RuntimeError, match="output snapshot differs"):
        v6.run(synthetic_root, synthetic_root / "out")
    assert not (synthetic_root / "out/summary.manifest.json").exists()



@pytest.mark.parametrize("missing", [
    "strong_native_trajectories10_v3_e326_20261009/summary.manifest.json",
    "strong_history_presentation10_v1_layout2_e337_20261009/history.manifest.json",
])
def test_native_history_literals_remain_missing_despite_other_history(synthetic_root, missing):
    for relative in v6.CURRENT_REQUIRED:
        if relative != missing:
            fixture_manifest(synthetic_root, relative)
    fixture_manifest(synthetic_root, "strong_native_trajectories5_v3_old/summary.manifest.json")
    fixture_manifest(synthetic_root, "strong_history_presentation10_v1_old/history.manifest.json")
    v6.run(synthetic_root, synthetic_root / "out")
    result = read_index(synthetic_root / "out")
    assert missing in result["missing_expected_manifests"]
    entries = {entry["manifest"] for entry in result["entries"]}
    assert "metrics/strong_native_trajectories5_v3_old/summary.manifest.json" in entries
    assert "metrics/strong_history_presentation10_v1_old/history.manifest.json" in entries
    assert result["publication_ready"] is False



def test_v6_does_not_mutate_v5_frozen_contract(synthetic_root):
    from scripts import build_publication_artifact_index_v5 as old

    before = (old.EXTRA_INCLUDE, old.CURRENT_REQUIRED)
    v6.run(synthetic_root, synthetic_root / "out")
    assert before == (old.EXTRA_INCLUDE, old.CURRENT_REQUIRED)
    assert len(old.CURRENT_REQUIRED) == 8


def test_only_current_visual_receipt_is_selected(synthetic_root):
    fixture_manifest(synthetic_root, "submission_visual_e401_20261009/summary.manifest.json")
    fixture_manifest(synthetic_root, "submission_visual_e434_20261009/summary.manifest.json")
    v6.run(synthetic_root, synthetic_root / "out")
    entries = {e["manifest"] for e in read_index(synthetic_root / "out")["entries"]}
    assert "metrics/submission_visual_e434_20261009/summary.manifest.json" in entries
    assert "metrics/submission_visual_e401_20261009/summary.manifest.json" not in entries


def test_prospective_weak_discovery_requires_completion_marker(synthetic_root):
    incomplete = synthetic_root / "metrics/facenet_fixed8_validation_s42_e410_20261009/private"
    incomplete.mkdir(parents=True)
    finished = "facenet_fixed8_validation_s1_e411_20261009/summary.manifest.json"
    fixture_manifest(synthetic_root, finished)
    v6.run(synthetic_root, synthetic_root / "out")
    result = read_index(synthetic_root / "out")
    entries = {entry["manifest"] for entry in result["entries"]}
    assert f"metrics/{finished}" in entries
    assert "metrics/facenet_fixed8_validation_s42_e410_20261009/summary.manifest.json" not in entries
    assert result["publication_ready"] is False
    assert "historical v2/v3/v4/v5" in result["limitation"]


def test_builder_cli_failure_redacts_error_and_preserves_partial(tmp_path, monkeypatch, capsys):
    out = tmp_path / "fresh-output"

    def fail(root, target):
        target.mkdir()
        (target / "partial.txt").write_text("synthetic partial", encoding="utf-8")
        raise RuntimeError("private-person-identifier")

    monkeypatch.setattr(v6, "run", fail)
    monkeypatch.setattr("sys.argv", ["builder", "--out", str(out)])
    with pytest.raises(SystemExit) as error:
        v6.main()
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out) == {"accepted": False, "reason": "RuntimeError"}
    assert (out / "partial.txt").read_text(encoding="utf-8") == "synthetic partial"
    assert not (out / "summary.manifest.json").exists()
