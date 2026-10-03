import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from age_gap.common.manifest import file_record
from scripts import export_roc_v2_evidence as export
from scripts.audit_submission_snapshot import check_roc_evidence

ROOT = Path(__file__).resolve().parents[1]
REVIEWED_MEMBERS = {
    "results/fgnet.json", "results/internal.json", "results/lfw.json",
    "manifests/fgnet.sanitized.json", "manifests/internal.sanitized.json", "manifests/lfw.sanitized.json",
    "manifests/fgnet.presentation.sanitized.json", "manifests/internal.presentation.sanitized.json",
    "tables/headline_table.tex", "tables/operating_table.tex", "tables/internal_table.tex",
    "README.txt", "CERTIFICATE.json",
}


def raw(name):
    directory, _ = export.EXPORTS[name]
    return json.loads((ROOT / "metrics" / directory / "summary.json").read_text(encoding="utf-8"))


def fixture(root):
    hidden = root / "private" / "Named_Person_0001.npz"
    hidden.parent.mkdir(parents=True)
    hidden.write_bytes(b"synthetic private data, never exported")
    for name, (directory, experiment) in export.EXPORTS.items():
        folder = root / "metrics" / directory
        folder.mkdir(parents=True)
        data = raw(name)
        data["scope" if name != "lfw" else "accuracy_scope"] = "private dropped prose"
        source, mp = folder / "summary.json", folder / "summary.manifest.json"
        source.write_bytes(export.encode(data))
        mp.write_bytes(export.encode({"experiment": experiment, "metrics": data,
            "inputs": [file_record(hidden)], "outputs": [file_record(source)],
            "command": ["private-command password=never-export"]}))
        if name in export.PRESENTATIONS:
            pe, tables = export.PRESENTATIONS[name]
            outputs = []
            for filename, renderer in tables.items():
                table = folder / filename
                table.write_text(renderer(data), encoding="utf-8")
                outputs.append(file_record(table))
            (folder / "presentation.manifest.json").write_bytes(export.encode({"experiment": pe,
                "inputs": [file_record(source), file_record(mp)], "outputs": outputs}))
    return hidden


def test_export_binds_generated_tables_and_omits_private_sources(tmp_path):
    root = tmp_path / "project"
    fixture(root)
    path = export.build_export(root, tmp_path / "export")
    with zipfile.ZipFile(path) as archive:
        contents = {n: archive.read(n) for n in archive.namelist()}
    assert set(contents) == export.MEMBERS == REVIEWED_MEMBERS and len(contents) == 13
    for marker in (b"Named_Person", b"never-export", b"private dropped prose", str(tmp_path).encode()):
        assert all(marker not in data for data in contents.values())
    certificate = json.loads(contents["CERTIFICATE.json"])
    for key in ("publication_ready", "full_reproduction", "privacy_certified", "ethics_clearance_attested", "disclosure_review_completed"):
        assert certificate[key] is False
    assert certificate["training_identity_independence"] == "unverified"
    for name, (directory, _) in export.EXPORTS.items():
        binding = certificate["aggregate_bindings"][name]
        assert binding["original_aggregate_sha256"] == export.sha256_file(root / "metrics" / directory / "summary.json")
        assert binding["exported_aggregate_sha256"] == export.digest(contents[f"results/{name}.json"])
        assert binding["exported_aggregate_sha256"] != binding["original_aggregate_sha256"]
    export.verify_archive(path, {n: export.digest(b) for n, b in contents.items()})


@pytest.mark.parametrize("problem", ["caption", "unknown_model", "metric_vector", "ci_vector", "count_vector", "ready", "bootstrap", "threshold_vector", "lfw_gate"])
def test_unknown_fields_vectors_and_unreviewed_claims_refused(problem):
    name = "fgnet"
    data = deepcopy(raw(name))
    if problem == "caption":
        data["captions"] = ["private"]
    elif problem == "unknown_model":
        data["overall"]["models"]["Named_Person"] = {}
    elif problem == "metric_vector":
        data["overall"]["models"]["frozen"]["roc_auc"]["point"] = [.1] * 512
    elif problem == "ci_vector":
        data["overall"]["models"]["frozen"]["roc_auc"]["ci95"] = [.1] * 512
    elif problem == "count_vector":
        data["cached_subjects"] = [82]
    elif problem == "ready":
        data["publication_ready"] = True
    elif problem == "bootstrap":
        data["overall"]["bootstrap"]["positive_weight"] = "private caption"
    elif problem == "threshold_vector":
        name, data = "internal", deepcopy(raw("internal"))
        data["subsets"]["original_tolerance"]["operating_points"]["frozen"]["0.01"]["similarity_threshold"] = [.1] * 512
    else:
        name, data = "lfw", deepcopy(raw("lfw"))
        data["compatibility"]["matches_within_1e_minus_12"] = False
    with pytest.raises(ValueError):
        export.project(name, data)


@pytest.mark.parametrize("problem", ["missing_input", "changed_input", "changed_table", "no_link", "metrics_mismatch"])
def test_missing_tampered_or_unbound_inputs_refuse_archive(tmp_path, problem):
    root = tmp_path / "project"
    hidden = fixture(root)
    folder = root / "metrics" / export.EXPORTS["fgnet"][0]
    if problem == "missing_input":
        hidden.unlink()
    elif problem == "changed_input":
        hidden.write_bytes(b"changed")
    elif problem == "changed_table":
        (folder / "headline_table.tex").write_text("changed", encoding="utf-8")
    else:
        mp = folder / ("presentation.manifest.json" if problem == "no_link" else "summary.manifest.json")
        manifest = json.loads(mp.read_text())
        if problem == "no_link":
            manifest["inputs"] = [file_record(hidden)]
        else:
            manifest["metrics"] = {}
        mp.write_bytes(export.encode(manifest))
    with pytest.raises(ValueError):
        export.build_export(root, tmp_path / "export")
    assert not (tmp_path / "export" / export.ZIP_NAME).exists()


def test_late_change_and_nonempty_output_preserved(tmp_path, monkeypatch):
    root = tmp_path / "project"
    hidden = fixture(root)
    original = export.verify_archive

    def late(*args):
        original(*args)
        hidden.write_bytes(b"late change")

    monkeypatch.setattr(export, "verify_archive", late)
    with pytest.raises(ValueError):
        export.build_export(root, tmp_path / "export")
    assert not list((tmp_path / "export").iterdir())
    previous = tmp_path / "export" / "previous.zip"
    previous.write_bytes(b"user-owned")
    with pytest.raises(FileExistsError):
        export.build_export(root, tmp_path / "export")
    assert previous.read_bytes() == b"user-owned"


@pytest.mark.parametrize("problem", ["extra", "missing", "duplicate", "traversal", "tamper", "bad_map", "comment", "oversize"])
def test_archive_membership_and_integrity_attacks_refused(tmp_path, problem):
    root = tmp_path / "project"
    fixture(root)
    path = export.build_export(root, tmp_path / "export")
    with zipfile.ZipFile(path) as archive:
        entries = [(n, archive.read(n)) for n in archive.namelist()]
    expected = {n: export.digest(b) for n, b in entries}
    if problem == "extra":
        entries.append(("private.npz", b"private"))
    elif problem == "missing":
        entries.pop()
    elif problem == "duplicate":
        entries.append(entries[0])
    elif problem == "traversal":
        entries[0] = ("../private.json", entries[0][1])
    elif problem == "tamper":
        entries[0] = (entries[0][0], b"{}")
    elif problem == "bad_map":
        expected.pop("README.txt")
    elif problem == "oversize":
        entries[0] = (entries[0][0], b"x" * (export.MAX_MEMBER_BYTES + 1))
        expected[entries[0][0]] = export.digest(entries[0][1])
    target = tmp_path / "malicious.zip"
    with zipfile.ZipFile(target, "w") as archive:
        for n, b in entries:
            archive.writestr(n, b)
        if problem == "comment":
            archive.comment = b"private hidden prose"
    with pytest.raises(ValueError):
        export.verify_archive(target, expected)


def test_changed_source_before_single_read_refused(tmp_path, monkeypatch):
    root = tmp_path / "project"
    fixture(root)
    source = root / "metrics" / export.EXPORTS["fgnet"][0] / "summary.json"
    original = Path.read_bytes

    def changed_read(path):
        result = original(path)
        return result + b" " if path == source else result

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    with pytest.raises(ValueError, match="changed before parsing"):
        export.build_export(root, tmp_path / "export")


@pytest.mark.parametrize("problem", ["none", "projection", "table", "gate", "source"])
def test_submission_audit_checks_native_evidence_not_only_zip_checksums(tmp_path, problem):
    root = tmp_path / "project"
    fixture(root)
    path = export.build_export(root, tmp_path / "export")
    upload = tmp_path / "upload"
    upload.mkdir()
    destination = upload / "roc-v2-evidence.zip"
    if problem == "none":
        destination.write_bytes(path.read_bytes())
        inputs = check_roc_evidence(root, upload)
        assert root / "metrics" / export.EXPORTS["fgnet"][0] / "summary.json" in inputs
        return
    with zipfile.ZipFile(path) as archive:
        contents = {n: archive.read(n) for n in archive.namelist()}
    certificate = json.loads(contents["CERTIFICATE.json"])
    if problem == "projection":
        data = json.loads(contents["results/fgnet.json"])
        data["scope"] = "unreviewed changed scope"
        contents["results/fgnet.json"] = export.encode(data)
        certificate["aggregate_bindings"]["fgnet"]["exported_aggregate_sha256"] = export.digest(contents["results/fgnet.json"])
    elif problem == "table":
        contents["tables/headline_table.tex"] = b"changed table"
    elif problem == "gate":
        certificate["privacy_certified"] = True
    else:
        (root / "metrics" / export.EXPORTS["fgnet"][0] / "summary.json").write_bytes(b"{}")
    certificate["members_excluding_certificate"] = {n: export.digest(b) for n, b in contents.items() if n != "CERTIFICATE.json"}
    contents["CERTIFICATE.json"] = export.encode(certificate)
    with zipfile.ZipFile(destination, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError):
        check_roc_evidence(root, upload)
