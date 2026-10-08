import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from age_gap.common.manifest import file_record
from scripts import export_cacd_evidence as export

ROOT = Path(__file__).resolve().parents[1]


def raw():
    return json.loads((ROOT / export.DIRECTORY / "summary.json").read_text(encoding="utf-8"))


def fixture(root):
    hidden = root / "private" / "Private_Person_0001.npz"
    hidden.parent.mkdir(parents=True)
    hidden.write_bytes(b"synthetic private data")
    folder = root / export.DIRECTORY
    folder.mkdir(parents=True)
    source = folder / "summary.json"
    data = raw()
    data["plan"]["scope"] = "password=not-exported C:/machine/private"
    data["inference"]["scope"] = "password=not-exported"
    data["inference"]["accuracy"]["scope"] = "private discarded prose"
    source.write_bytes(export.encode(data))
    mp = source.with_suffix(".manifest.json")
    mp.write_bytes(export.encode({"experiment": export.EXPERIMENT, "metrics": data,
        "inputs": [file_record(hidden)], "outputs": [file_record(source)],
        "command": ["password=not-exported"]}))
    presentation = root / export.PRESENTATION
    presentation.mkdir(parents=True)
    outputs = []
    for name, renderer in export.TABLES.items():
        path = presentation / name
        path.write_bytes(renderer(data).encode())
        outputs.append(file_record(path))
    pp = presentation / "presentation.manifest.json"
    pp.write_bytes(export.encode({"experiment": export.PRESENTATION_EXPERIMENT,
        "metrics": {"pair_level_only": True, "legacy_results_pooled": False, "publication_ready": False},
        "inputs": [file_record(source), file_record(mp)], "outputs": outputs}))
    return hidden, source, mp, pp


def test_safe_seven_member_export_and_distinct_hashes(tmp_path):
    root = tmp_path / "project"
    _, source, _, _ = fixture(root)
    path = export.build_export(root, tmp_path / "export")
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    assert set(members) == export.MEMBERS and len(members) == 7
    for marker in (b"password=", b"Private_Person", b"discarded prose", str(tmp_path).encode(), b"checkpoint_records"):
        assert all(marker not in value for value in members.values())
    certificate = json.loads(members["CERTIFICATE.json"])
    assert certificate["original_aggregate_sha256"] == export.sha256_file(source)
    assert certificate["exported_aggregate_sha256"] == export.digest(members["results/cacd.json"])
    assert certificate["original_aggregate_sha256"] != certificate["exported_aggregate_sha256"]
    for field in ("publication_ready", "full_reproduction", "privacy_certified", "ethics_clearance_attested",
                  "disclosure_review_completed", "private_records_listed"):
        assert certificate[field] is False
    public = json.loads(members["results/cacd.json"])
    assert public["inference"]["bootstrap"]["subject_metadata_available"] is False
    assert public["checkpoint_training_provenance"] == public["original_alignment_replay"] == "unverified"
    export.verify_archive(path, {name: export.digest(value) for name, value in members.items()})
    assert source in export.check_evidence(root, path)
    native = json.loads((path.parent / "export.manifest.json").read_text())
    assert native["outputs"] == [file_record(path)]
    assert native["metrics"]["member_count"] == 7


def test_crlf_generated_blocks_preserve_original_bytes(tmp_path):
    root = tmp_path / "project"
    _, _, _, pp = fixture(root)
    manifest = json.loads(pp.read_text())
    for name in export.TABLES:
        path = pp.parent / name
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    manifest["outputs"] = [file_record(pp.parent / name) for name in export.TABLES]
    pp.write_bytes(export.encode(manifest))
    path = export.build_export(root, tmp_path / "export")
    with zipfile.ZipFile(path) as archive:
        for name in export.TABLES:
            assert archive.read(f"tables/{name}") == (pp.parent / name).read_bytes()
            assert b"\r\n" in archive.read(f"tables/{name}")
    export.check_evidence(root, path)


@pytest.mark.parametrize("problem", ["caption", "model_field", "vector", "ci", "bool_count", "subject",
    "incomplete", "ready", "grid", "grid_hash", "percentile", "plan", "normalization", "bool_preprocess", "delta"])
def test_projection_rejects_unreviewed_fields_and_claims(problem):
    data = deepcopy(raw())
    inf = data["inference"]
    if problem == "caption":
        data["captions"] = ["private"]
    elif problem == "model_field":
        inf["models"]["frozen"]["roc_auc"]["person_id"] = "private"
    elif problem == "vector":
        inf["models"]["frozen"]["roc_auc"]["point"] = [.2] * 512
    elif problem == "ci":
        inf["models"]["frozen"]["roc_auc"]["pair_ci95"] = [-.2, .9]
    elif problem == "bool_count":
        inf["bootstrap"]["seed"] = False
    elif problem == "subject":
        inf["bootstrap"]["subject_metadata_available"] = True
    elif problem == "incomplete":
        data["execution_complete"] = False
    elif problem == "ready":
        data["publication_ready"] = True
    elif problem == "grid":
        inf["accuracy"]["legacy_grid_equivalence_claimed"] = True
    elif problem == "grid_hash":
        inf["accuracy"]["threshold_grid_sha256"] = "0" * 64
    elif problem == "percentile":
        inf["bootstrap"]["percentile_method"] = "nearest"
    elif problem == "plan":
        data["plan"]["execution_complete"] = True
    elif problem == "normalization":
        data["plan"]["preprocessing"]["normalization"] = "different"
    elif problem == "bool_preprocess":
        data["plan"]["preprocessing"]["shared_between_frozen_and_tuned"] = 1
    else:
        inf["three_checkpoint_aggregate"]["roc_auc"]["mean_checkpoint_delta"] = .5
    with pytest.raises(ValueError):
        export.project(data)


@pytest.mark.parametrize("problem", ["missing_private", "unbound", "table", "link", "metrics", "gate"])
def test_source_or_presentation_mismatch_prevents_archive(tmp_path, problem):
    root = tmp_path / "project"
    hidden, _, mp, pp = fixture(root)
    if problem == "missing_private":
        hidden.unlink()
    elif problem == "table":
        (pp.parent / "cacd_table.tex").write_bytes(b"wrong table")
    else:
        target = mp if problem in ("unbound", "metrics") else pp
        manifest = json.loads(target.read_text())
        if problem == "unbound":
            manifest["outputs"] = [file_record(hidden)]
        elif problem == "link":
            manifest["inputs"] = [file_record(hidden)]
        elif problem == "metrics":
            manifest["metrics"] = {}
        else:
            manifest["metrics"]["pair_level_only"] = False
        target.write_bytes(export.encode(manifest))
    out = tmp_path / "export"
    with pytest.raises(ValueError):
        export.build_export(root, out)
    assert not (out / export.ZIP_NAME).exists()


def test_nonempty_destination_is_preserved(tmp_path):
    out = tmp_path / "export"
    out.mkdir()
    sentinel = out / "keep"
    sentinel.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        export.build_export(tmp_path, out)
    assert sentinel.read_bytes() == b"keep"


@pytest.mark.parametrize("attack", ["duplicate", "extra", "comment", "wrong_hash"])
def test_archive_membership_metadata_and_hash_attacks(tmp_path, attack):
    root = tmp_path / "project"
    fixture(root)
    path = export.build_export(root, tmp_path / "export")
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    expected = {name: export.digest(value) for name, value in members.items()}
    attacked = tmp_path / "attacked.zip"
    with zipfile.ZipFile(attacked, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
        if attack == "duplicate":
            archive.writestr("README.txt", members["README.txt"])
        elif attack == "extra":
            archive.writestr("private/rows.json", b"{}")
        elif attack == "comment":
            archive.comment = b"unreviewed"
        else:
            expected["README.txt"] = "0" * 64
    with pytest.raises(ValueError):
        export.verify_archive(attacked, expected)


def test_late_mutation_withdraws_temporary_zip(tmp_path, monkeypatch):
    root = tmp_path / "project"
    _, source, _, _ = fixture(root)
    original = export.verify_archive

    def late(path, expected):
        original(path, expected)
        source.write_bytes(b"{}")

    monkeypatch.setattr(export, "verify_archive", late)
    out = tmp_path / "export"
    with pytest.raises(ValueError):
        export.build_export(root, out)
    assert not list(out.glob("*.zip*"))


@pytest.mark.parametrize("problem", ["projection", "table", "gate", "sanitized", "source", "readme"])
def test_rehashed_certificate_does_not_replace_native_evidence(tmp_path, problem):
    root = tmp_path / "project"
    _, source, _, _ = fixture(root)
    path = export.build_export(root, tmp_path / "export")
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    certificate = json.loads(members["CERTIFICATE.json"])
    if problem == "projection":
        result = json.loads(members["results/cacd.json"])
        result["alignment_fallback_fraction"] = .1
        members["results/cacd.json"] = export.encode(result)
        certificate["exported_aggregate_sha256"] = export.digest(members["results/cacd.json"])
    elif problem == "table":
        members["tables/cacd_table.tex"] = b"wrong table"
    elif problem == "gate":
        certificate["privacy_certified"] = True
    elif problem == "sanitized":
        result = json.loads(members["manifests/evaluation.sanitized.json"])
        result["locally_verified_input_records"] = 123456
        members["manifests/evaluation.sanitized.json"] = export.encode(result)
    elif problem == "readme":
        members["README.txt"] = b"unreviewed publication claim"
    else:
        source.write_bytes(b"{}")
    certificate["members_excluding_certificate"] = {name: export.digest(value) for name, value in members.items() if name != "CERTIFICATE.json"}
    members["CERTIFICATE.json"] = export.encode(certificate)
    attacked = tmp_path / "attacked.zip"
    with zipfile.ZipFile(attacked, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    with pytest.raises(ValueError):
        export.check_evidence(root, attacked)


def test_receipt_failure_withdraws_generated_archive(tmp_path, monkeypatch):
    root = tmp_path / "project"
    fixture(root)

    def fail(*args, **kwargs):
        raise OSError("receipt failed")

    monkeypatch.setattr(export, "write_experiment_manifest", fail)
    out = tmp_path / "export"
    with pytest.raises(OSError, match="receipt failed"):
        export.build_export(root, out)
    assert not list(out.glob("*.zip*"))
    assert not (out / "export.manifest.json").exists()
