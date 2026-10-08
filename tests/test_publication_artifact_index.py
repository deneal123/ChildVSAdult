from __future__ import annotations

import json

from scripts import build_publication_artifact_index as index


def test_serial_cacd_native_and_presentation_coverage_is_explicit():
    assert "cacd_serial_20261003/*.manifest.json" in index.INCLUDE
    assert "cacd_serial_presentation_20261003/presentation.manifest.json" in index.INCLUDE
    assert not any("submission_snapshot" in path or "visual_review" in path for path in index.INCLUDE)


def test_fixed8_cuda_native_diagnostics_and_presentation_coverage_is_explicit():
    assert {
        "adaface_fixed8_cuda_*/summary.manifest.json",
        "facenet_fixed8_validation*_v2_*/summary.manifest.json",
        "strong_queue_remaining*_v2_*/*/train/summary.manifest.json",
        "strong_queue_remaining*_v2_*/*/fgnet/summary.manifest.json",
        "strong_mechanism_*_v1_*/summary.manifest.json",
        "common_mechanism_*_v1_*/summary.manifest.json",
        "strong_native_trajector*_v1_*/summary.manifest.json",
        "native_person_age_baselines*_v2_*/summary.manifest.json",
        "weak_native_cache_keys_v2_*/summary.manifest.json",
        "strong_fixed8_roc_presentation*_v1_*/presentation.manifest.json",
    } <= set(index.INCLUDE)
    assert not any("queue.json" in path for path in index.INCLUDE)


def test_nested_queue_native_manifests_discovered_without_ledger_completion(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics"
    queue = metrics / "strong_queue_remaining30_v2_test"
    manifests = []
    for stage in ("train", "fgnet"):
        path = queue / "random_head_lr1e-05_s42" / stage / "summary.manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"experiment": "synthetic", "inputs": [], "outputs": [],
                                    "parameters": {}}), encoding="utf-8")
        manifests.append(path.relative_to(tmp_path).as_posix())
    (queue / "queue.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    monkeypatch.setattr(index, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(index, "METRICS", metrics)
    monkeypatch.setattr(index, "OUTPUT", metrics / "index.json")
    monkeypatch.setattr(index, "INCLUDE", (
        "strong_queue_remaining*_v2_*/*/train/summary.manifest.json",
        "strong_queue_remaining*_v2_*/*/fgnet/summary.manifest.json",
    ))
    index.main()
    result = json.loads(index.OUTPUT.read_text(encoding="utf-8"))
    assert {row["manifest"] for row in result["entries"]} == set(manifests)
    assert result["complete"] is False
    assert "strong-backbone summary missing" in result["incomplete_experiments"]


def test_nested_literal_manifest_is_not_falsely_reported_missing(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics"
    nested = metrics / "nested"
    nested.mkdir(parents=True)
    (nested / "test.manifest.json").write_text(json.dumps({
        "experiment": "synthetic", "inputs": [], "outputs": [], "parameters": {},
    }), encoding="utf-8")
    monkeypatch.setattr(index, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(index, "METRICS", metrics)
    monkeypatch.setattr(index, "OUTPUT", metrics / "index.json")
    monkeypatch.setattr(index, "INCLUDE", ("nested/test.manifest.json",))
    index.main()
    result = json.loads(index.OUTPUT.read_text(encoding="utf-8"))
    assert result["entry_count"] == 1
    assert result["missing_expected_manifests"] == []
    assert result["complete"] is False  # no strong-backbone campaign evidence


def test_public_index_omits_private_embeddings_protocol_rows_and_checksums():
    records = [{"path": p, "sha256": "private-digest"} for p in (
        "metrics/study/private/private_embeddings_frozen.npz",
        "metrics/study/private/private_pair_scores.jsonl",
        "metrics/study/private/protocol_plan.json",
        "data/interim/faces/private.jpg",
    )]
    records.append({"path": "metrics/study/summary.json", "sha256": "public-digest"})
    public, hidden = index._public_records(records)
    assert hidden == 4
    assert public == [{"path": "metrics/study/summary.json", "sha256": "public-digest"}]


def test_private_windows_separator_path_is_hidden():
    public, hidden = index._public_records([{"path": r"metrics\study\private\template.npz"}])
    assert public == [] and hidden == 1


def test_constituent_candidates_and_join_digests_stay_private():
    private = {"path": "metrics/constituent_integrity_20261003/private/candidates.jsonl", "sha256": "private-join-digest"}
    aggregate = {"path": "metrics/constituent_integrity_20261003/summary.json", "sha256": "aggregate-digest"}
    public, hidden = index._public_records([private, aggregate])
    assert public == [aggregate]
    assert hidden == 1
    assert "private-join-digest" not in json.dumps(public)


def test_named_benchmark_images_and_array_digests_are_not_public():
    records = [{"path": path, "sha256": "biometric-digest"} for path in (
        "C:/benchmark/Person_Name/Person_Name_0001.jpg",
        "data/external/lfw_aligned.npz", "metrics/probe/features.npy",
    )]
    public, hidden = index._public_records(records)
    assert public == [] and hidden == 3


def test_integrity_errors_do_not_reveal_missing_benchmark_person_names(tmp_path, monkeypatch):
    monkeypatch.setattr(index, "PROJECT_ROOT", tmp_path)
    missing = {"path": "raw/Person_Name/Person_Name_0001.jpg", "sha256": "private"}
    result = index._record_integrity([missing])
    assert result["valid"] is False
    assert result["private_missing_count"] == 1
    assert result["missing"] == []
    assert "Person_Name" not in json.dumps(result)


def test_private_checksum_error_still_invalidates_index(tmp_path, monkeypatch):
    monkeypatch.setattr(index, "PROJECT_ROOT", tmp_path)
    photo = tmp_path / "face.jpg"
    photo.write_bytes(b"changed")
    result = index._record_integrity([{"path": "face.jpg", "sha256": "stale"}])
    assert result["valid"] is False
    assert result["private_checksum_mismatch_count"] == 1
    assert result["checksum_mismatch"] == []


def test_running_lfw_evaluation_is_explicitly_incomplete(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics"
    (metrics / "lfw_bound_evaluation_synthetic" / "private").mkdir(parents=True)
    monkeypatch.setattr(index, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(index, "METRICS", metrics)
    monkeypatch.setattr(index, "OUTPUT", metrics / "index.json")
    monkeypatch.setattr(index, "INCLUDE", ("lfw_bound_evaluation_*/lfw_bound_evaluation.manifest.json",))
    index.main()
    result = json.loads(index.OUTPUT.read_text(encoding="utf-8"))
    assert "lfw_bound_evaluation_synthetic: completed result or manifest missing" in result["incomplete_experiments"]
    assert result["entry_count"] == 0 and result["complete"] is False


def test_roc_v2_publication_manifests_are_explicitly_required():
    assert {
        "internal_metrics_v2_20261003/summary.manifest.json",
        "internal_metrics_v2_20261003/presentation.manifest.json",
        "fgnet_metrics_v2_20261003/summary.manifest.json",
        "fgnet_metrics_v2_20261003/presentation.manifest.json",
        "lfw_metrics_v2_20261003/summary.manifest.json",
    } <= set(index.INCLUDE)
