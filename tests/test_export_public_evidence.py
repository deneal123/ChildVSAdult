"""Focused synthetic tests for export_public_evidence (no canonical bundle created)."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts import export_public_evidence as builder

RESULT = "metrics/synth_result.json"
MANIFEST = "metrics/synth_result.manifest.json"


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read(archive, member):
    with zipfile.ZipFile(archive) as zf:
        return zf.read(member).decode("utf-8")


def _json(archive, member):
    return json.loads(_read(archive, member))


def make_root(tmp_path, result_payload=None, inputs=None, declared_sha=None, command=None,
              outputs=None):
    root = tmp_path / "root"
    result = root / RESULT
    _write(result, {"auc": 0.9} if result_payload is None else result_payload)
    sha = builder.sha256_file(result) if declared_sha is None else declared_sha
    manifest = {
        "schema_version": 1, "experiment": "synthetic", "inputs": inputs or [], "command": command,
        "outputs": outputs if outputs is not None else [
            {"path": RESULT, "bytes": result.stat().st_size, "sha256": sha}],
    }
    _write(root / MANIFEST, manifest)
    return root


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(builder, "EXPORTS", (RESULT,))


def test_rejects_checksum_mismatch(tmp_path, synthetic):
    root = make_root(tmp_path, declared_sha="0" * 64)
    with pytest.raises(builder.ChecksumError):
        builder.verify_manifest(RESULT, root)


def test_rejects_missing_direct_input(tmp_path, synthetic):
    """A missing declared input must abort, not merely be counted."""
    root = make_root(tmp_path, inputs=[{"path": "data/interim/faces/absent.jpg",
                                        "bytes": 10, "sha256": "a" * 64}])
    with pytest.raises(builder.ChecksumError):
        builder.verify_manifest(RESULT, root)


def test_rejects_malformed_manifest_record(tmp_path, synthetic):
    root = make_root(tmp_path, outputs=[{"path": RESULT, "bytes": 1, "sha256": "nothex"}])
    with pytest.raises(builder.ChecksumError):
        builder.verify_manifest(RESULT, root)


def test_rejects_unsafe_row_identifier_payload(tmp_path, synthetic):
    root = make_root(tmp_path, result_payload={"note": "row -134190297_456240108 leaked"})
    with pytest.raises(builder.UnsafeContentError):
        builder.build_export(root, tmp_path / "out")


def test_rejects_unsafe_per_image_hash_key(tmp_path, synthetic):
    root = make_root(tmp_path, result_payload={"image_sha256": "a" * 64})
    with pytest.raises(builder.UnsafeContentError):
        builder.build_export(root, tmp_path / "out")


def _real_input(root, rel, data=b"payload"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _rewrite_manifest(root, inputs):
    _write(root / MANIFEST, {
        "schema_version": 1, "experiment": "synthetic", "inputs": inputs,
        "outputs": [{"path": RESULT, "bytes": (root / RESULT).stat().st_size,
                     "sha256": builder.sha256_file(root / RESULT)}]})


def test_external_paths_and_private_records_are_sanitized(tmp_path, synthetic):
    root = make_root(tmp_path)
    ext = tmp_path / "ext" / "weights.pt"
    ext.parent.mkdir(parents=True, exist_ok=True)
    ext.write_bytes(b"w")
    inputs = [
        {"path": str(ext), "bytes": 1, "sha256": hashlib.sha256(b"w").hexdigest()},
        _real_input(root, "metrics/private/private_rows.jsonl"),
        _real_input(root, "data/interim/faces/crop_1.jpg", b"xy"),
    ]
    _rewrite_manifest(root, inputs)
    archive = builder.build_export(root, tmp_path / "out")
    text = _read(archive, "manifests/synth_result.json.sanitized.json")
    sanitized = json.loads(text)
    assert str(ext) not in text and "faces/crop_1.jpg" not in text
    assert sanitized["public_inputs"] == []
    assert sanitized["hidden_input_records"] == 3


def test_embedded_absolute_path_in_command_is_redacted(tmp_path, synthetic):
    """Machine paths inside command/tool strings are redacted, then re-scanned."""
    root = make_root(tmp_path, command="python train.py --weights C:\\Users\\someone\\.cache\\w.pt")
    archive = builder.build_export(root, tmp_path / "out")
    text = _read(archive, "manifests/synth_result.json.sanitized.json")
    assert "someone" not in text
    assert json.loads(text)["command"] == "python train.py --weights <external-artifact>/w.pt"


def test_certificate_has_no_raw_paths_or_private_ids(tmp_path, synthetic):
    root = make_root(tmp_path, result_payload={"checkpoint_sha256": "d" * 64, "auc": 0.9})
    _rewrite_manifest(root, [_real_input(root, "metrics/private/private_rows.jsonl"),
                             _real_input(root, "data/interim/faces/crop_1.jpg", b"xy")])
    archive = builder.build_export(root, tmp_path / "out")
    text = _read(archive, "CERTIFICATE.json")
    for needle in ("/private/", "private_rows", "faces/crop_1", "134190297"):
        assert needle not in text
    assert "checkpoint_sha256" in _read(archive, "results/synth_result.json")  # science kept
    assert json.loads(text)["entries"][0]["verification"]["inputs"]["valid"] is True


def test_result_and_manifest_checksums_recorded_separately(tmp_path, synthetic):
    root = make_root(tmp_path)
    archive = builder.build_export(root, tmp_path / "out")
    cert = _json(archive, "CERTIFICATE.json")
    entry = cert["entries"][0]
    assert entry["original_sha256"] == builder.sha256_file(root / RESULT)
    with zipfile.ZipFile(archive) as zf:
        assert entry["exported_sha256"] == hashlib.sha256(zf.read(entry["member"])).hexdigest()
    assert entry["verification"]["outputs"]["valid"] is True


def test_unresolved_gates_remain_explicit(tmp_path, synthetic):
    root = make_root(tmp_path, result_payload={
        "training_identity_independence": "unverified",
        "disclosures": ["not a publication-readiness claim"],
    })
    archive = builder.build_export(root, tmp_path / "out")
    cert = _json(archive, "CERTIFICATE.json")
    assert cert["unresolved"]["gates"]["training_identity_independence"] == "unverified"
    assert cert["unresolved"]["asserts_training_independence"] is False
    assert cert["unresolved"]["asserts_publication_ready"] is False
    assert cert["unresolved"]["asserts_experiment_complete"] is False
    assert "not_complete" not in cert  # no completeness assertion is fabricated


def test_preexisting_outdir_file_is_not_archived(tmp_path, synthetic):
    root = make_root(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "stray_secret.json").write_text("{\"leak\": 1}", encoding="utf-8")
    archive = builder.build_export(root, out)
    with zipfile.ZipFile(archive) as zf:
        assert not any("stray_secret" in n for n in zf.namelist())
        assert set(zf.namelist()) == {
            "results/synth_result.json",
            "manifests/synth_result.json.sanitized.json",
            "CERTIFICATE.json", "README.txt",
        }


def test_refuses_preexisting_zip_without_explicit_overwrite(tmp_path, synthetic):
    root = make_root(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / builder.ZIP_NAME).write_bytes(b"stale")
    with pytest.raises(builder.ArchiveError):
        builder.build_export(root, out)
    assert builder.build_export(root, out, overwrite=True).is_file()


def _expected_for(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as zf:
        return {n: hashlib.sha256(zf.read(n)).hexdigest() for n in zf.namelist()}


def test_archive_membership_allowlist_and_verification(tmp_path, synthetic):
    root = make_root(tmp_path)
    archive = builder.build_export(root, tmp_path / "out")
    assert builder.verify_archive(archive, _expected_for(archive)) == {
        "results/synth_result.json", "manifests/synth_result.json.sanitized.json",
        "CERTIFICATE.json", "README.txt"}
    rogue = tmp_path / "rogue.zip"
    with zipfile.ZipFile(rogue, "w") as zf:
        zf.writestr("metrics/raw_manifest.json", "{}")
    with pytest.raises(builder.ArchiveError):
        builder.verify_archive(rogue, {"metrics/raw_manifest.json": "x"})


def test_rejects_rogue_result_member(tmp_path, synthetic):
    root = make_root(tmp_path)
    archive = builder.build_export(root, tmp_path / "out")
    expected = _expected_for(archive)
    rogue = tmp_path / "rogue_result.zip"
    with zipfile.ZipFile(rogue, "w") as zf:
        zf.writestr("results/fake.json", "{}")
    with pytest.raises(builder.ArchiveError):
        builder.verify_archive(rogue, expected)
    # A genuine archive with one extra, unlisted result member must also be rejected.
    expected["results/fake.json"] = hashlib.sha256(b"{}").hexdigest()
    with pytest.raises(builder.ArchiveError):
        builder.verify_archive(archive, expected)


def test_rejects_duplicate_and_traversal_members(tmp_path, synthetic):
    root = make_root(tmp_path)
    archive = builder.build_export(root, tmp_path / "out")
    expected = _expected_for(archive)
    dup = tmp_path / "dup.zip"
    with zipfile.ZipFile(dup, "w") as zf:
        zf.writestr("results/synth_result.json", "{}")
        with pytest.warns(UserWarning, match="Duplicate name"):
            zf.writestr("results/synth_result.json", "{}")
    with pytest.raises(builder.ArchiveError):
        builder.verify_archive(dup, expected)
    trav = tmp_path / "trav.zip"
    with zipfile.ZipFile(trav, "w") as zf:
        zf.writestr("../escape.json", "{}")
    with pytest.raises(builder.ArchiveError):
        builder.verify_archive(trav, {"../escape.json": "x"})


def test_sanitize_value_redacts_row_ids_and_paths(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    value, hits = builder.sanitize_value(
        {"row": "vk_12345_67890", "ext": "D:\\data\\secret.bin", "auc": 0.91}, root)
    assert value == {"row": "<redacted-row-id>", "ext": "<external-artifact>/secret.bin",
                     "auc": 0.91}
    assert hits == 2


@pytest.mark.parametrize("payload", [
    {"subject_ids": [1, 2, 3]},
    {"embeddings": [[0.1, 0.2]]},
    {"caption": "a raw private post"},
    {"note": "client_secret=never-export"},
    {"note": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz1234"},
])
def test_refuses_identifying_payloads_and_credentials(tmp_path, synthetic, payload):
    root = make_root(tmp_path, result_payload=payload)
    with pytest.raises(builder.UnsafeContentError):
        builder.build_export(root, tmp_path / "out")


def test_sanitized_projection_omits_unreviewed_json_digests(tmp_path, synthetic):
    root = make_root(tmp_path)
    record = _real_input(root, "metrics/biometric-cache.json")
    _rewrite_manifest(root, [record])
    archive = builder.build_export(root, tmp_path / "out")
    text = _read(archive, "manifests/synth_result.json.sanitized.json")
    assert record["path"] not in text and record["sha256"] not in text
    assert _json(archive, "manifests/synth_result.json.sanitized.json")["hidden_input_records"] == 1


def test_member_tampering_is_detected(tmp_path, synthetic):
    root = make_root(tmp_path)
    archive = builder.build_export(root, tmp_path / "out")
    expected = _expected_for(archive)
    bad = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(bad, "w") as target:
        for name in source.namelist():
            target.writestr(name, b"changed" if name.startswith("manifests/") else source.read(name))
    with pytest.raises(builder.ArchiveError, match="hash mismatch"):
        builder.verify_archive(bad, expected)


def test_private_path_key_is_rejected_after_sanitation():
    with pytest.raises(builder.UnsafeContentError, match="private artifact path"):
        builder.scan_unsafe({"metrics/private/cache.json": 1})
