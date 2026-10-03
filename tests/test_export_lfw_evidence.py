import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from age_gap.common.manifest import file_record
from scripts import export_lfw_evidence as export
from scripts.render_lfw_evidence import render

ROOT = Path(__file__).resolve().parents[1]


def raw_result():
    return json.loads((ROOT / export.DIRECTORY / "lfw_bound_evaluation.json").read_text(encoding="utf-8"))


def make_sources(tmp_path):
    root = tmp_path / "project"
    directory = root / export.DIRECTORY
    directory.mkdir(parents=True)
    private = root / "private"
    private.mkdir()
    hidden = private / "Named_Person_0001.jpg"
    hidden.write_bytes(b"synthetic private image fixture")
    raw = raw_result()
    for row in raw["provenance"].values():
        row["checkpoint_name"] = "private-checkpoint-name"
        row["checkpoint_sha256"] = "private-checkpoint-digest"
    source, table = directory / "lfw_bound_evaluation.json", directory / "lfw_table.tex"
    source.write_bytes(export.encode(raw))
    table.write_text(render(raw), encoding="utf-8")
    evaluation = directory / "lfw_bound_evaluation.manifest.json"
    evaluation.write_bytes(export.encode({"experiment": "lfw-bound-3seed-subject-evaluation",
        "inputs": [file_record(hidden)], "outputs": [file_record(source)],
        "command": ["private-command password=do-not-export"]}))
    presentation = directory / "lfw_presentation.manifest.json"
    presentation.write_bytes(export.encode({"experiment": "lfw-source-bound-table-presentation",
        "inputs": [file_record(source), file_record(evaluation)], "outputs": [file_record(table)]}))
    return root, directory, hidden


def test_whitelist_export_omits_private_records_and_binds_original_vs_exported_bytes(tmp_path):
    root, directory, _ = make_sources(tmp_path)
    archive_path = export.build_export(root, tmp_path / "export")
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == export.MEMBERS
        content = {name: archive.read(name) for name in archive.namelist()}
    exported = json.loads(content["results/lfw.json"])
    certificate = json.loads(content["CERTIFICATE.json"])
    assert exported["publication_ready"] is False
    assert exported["training_identity_independence"] == "unverified"
    assert exported["checkpoint_training_provenance"].startswith("legacy")
    assert "provenance" not in exported
    assert exported["checkpoint_records_omitted"] == 4
    for marker in (b"Named_Person", b"private-checkpoint", b"do-not-export", str(tmp_path).encode()):
        assert all(marker not in data for data in content.values())
    assert certificate["original_result_sha256"] == export.sha256_file(directory / "lfw_bound_evaluation.json")
    assert certificate["exported_result_sha256"] == export.digest(content["results/lfw.json"])
    assert certificate["original_result_sha256"] != certificate["exported_result_sha256"]
    export.verify_archive(archive_path, {name: export.digest(data) for name, data in content.items()})


@pytest.mark.parametrize("problem", ["unknown", "caption", "metric_vector", "ci_embedding", "fold_vector", "ready", "independence", "bootstrap", "version", "inference"])
def test_unreviewed_schema_and_sensitive_or_vector_fields_are_refused(problem):
    raw = deepcopy(raw_result())
    if problem == "unknown":
        raw["user_notes"] = "unreviewed text"
    elif problem == "caption":
        raw["models"]["tuned_seed1"]["captions"] = ["private caption"]
    elif problem == "metric_vector":
        raw["models"]["tuned_seed1"]["metrics"]["roc_auc"] = [.1] * 512
    elif problem == "ci_embedding":
        raw["models"]["tuned_seed1"]["subject_ci95"]["roc_auc"] = [.1] * 512
    elif problem == "fold_vector":
        raw["models"]["tuned_seed1"]["official_folds"][0]["threshold"] = [.1] * 512
    elif problem == "ready":
        raw["publication_ready"] = True
    elif problem == "independence":
        raw["training_identity_independence"] = "verified"
    elif problem == "bootstrap":
        raw["bootstrap"]["positive_weight"] = "unreviewed prose"
    elif problem == "version":
        raw["package_versions"]["torch"] = "C:/private/file"
    else:
        raw["seed_aggregate"]["roc_auc"]["fixed_checkpoint_mean_gain_subject_ci95"] = [-.1, .1]
    with pytest.raises(ValueError):
        export.project_result(raw)


@pytest.mark.parametrize("problem", ["missing_input", "changed_input", "size", "changed_table", "missing_link"])
def test_missing_tampered_or_unbound_originals_refuse_export(tmp_path, problem):
    root, directory, hidden = make_sources(tmp_path)
    if problem == "missing_input":
        hidden.unlink()
    elif problem == "changed_input":
        hidden.write_bytes(b"changed")
    elif problem == "changed_table":
        (directory / "lfw_table.tex").write_text("changed", encoding="utf-8")
    else:
        path = directory / "lfw_bound_evaluation.manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if problem == "size":
            manifest["inputs"][0]["bytes"] += 1
        else:
            manifest["outputs"] = [file_record(hidden)]
        path.write_bytes(export.encode(manifest))
    with pytest.raises(ValueError):
        export.build_export(root, tmp_path / "export")
    assert not (tmp_path / "export" / export.ZIP_NAME).exists()


def test_nonempty_output_is_preserved_and_not_overwritten(tmp_path):
    root, _, _ = make_sources(tmp_path)
    out = tmp_path / "export"
    out.mkdir()
    previous = out / export.ZIP_NAME
    previous.write_bytes(b"previous archive")
    with pytest.raises(FileExistsError):
        export.build_export(root, out)
    assert previous.read_bytes() == b"previous archive"


def test_late_private_input_change_refuses_completed_zip(tmp_path, monkeypatch):
    root, _, hidden = make_sources(tmp_path)
    original = export.verify_archive

    def mutate_after_zip(*args):
        original(*args)
        hidden.write_bytes(b"late change")

    monkeypatch.setattr(export, "verify_archive", mutate_after_zip)
    with pytest.raises(ValueError):
        export.build_export(root, tmp_path / "export")
    assert not list((tmp_path / "export").iterdir())


@pytest.mark.parametrize("problem", ["extra", "missing", "duplicate", "traversal", "tamper", "bad_map"])
def test_archive_verifier_rejects_membership_and_hash_attacks(tmp_path, problem):
    root, _, _ = make_sources(tmp_path)
    path = export.build_export(root, tmp_path / "export")
    with zipfile.ZipFile(path) as archive:
        data = {name: archive.read(name) for name in archive.namelist()}
    expected = {name: export.digest(value) for name, value in data.items()}
    entries = list(data.items())
    if problem == "extra":
        entries.append(("private.npz", b"extra"))
    elif problem == "missing":
        entries.pop()
    elif problem == "duplicate":
        entries.append(entries[0])
    elif problem == "traversal":
        entries[0] = ("../results/lfw.json", entries[0][1])
    elif problem == "tamper":
        entries[0] = (entries[0][0], b"{}")
    else:
        expected.pop("README.txt")
    malicious = tmp_path / "malicious.zip"
    with zipfile.ZipFile(malicious, "w") as archive:
        for name, value in entries:
            archive.writestr(name, value)
    with pytest.raises(ValueError):
        export.verify_archive(malicious, expected)
