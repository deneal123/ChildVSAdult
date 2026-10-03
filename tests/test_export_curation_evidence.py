import copy
import json
import zipfile
from pathlib import Path

import pytest

from age_gap.common.manifest import file_record
from scripts import export_curation_evidence as export

ROOT = Path(__file__).resolve().parents[1]


def results():
    return {name: json.loads((ROOT / "metrics" / directory / "summary.json").read_text(encoding="utf-8"))
            for name, (directory, _) in export.EXPORTS.items()}


def fixture_root(tmp_path):
    root = tmp_path / "project"
    for name, (directory, experiment) in export.EXPORTS.items():
        target = root / "metrics" / directory
        target.mkdir(parents=True)
        summary = target / "summary.json"
        raw = results()[name]
        summary.write_bytes(export.encode(raw))
        private = target / "private" / "private_canary.json"
        private.parent.mkdir()
        private.write_text('{"caption":"PRIVATE_CAPTION_CANARY", "person_id":"-12345678_98765432"}', encoding="utf-8")
        manifest = {"experiment": experiment, "metrics": raw,
                    "command": ["C:/private/machine/tool.py", "PRIVATE_COMMAND_CANARY"],
                    "parameters": {"private_parameter": "PRIVATE_PARAMETER_CANARY"},
                    "inputs": [file_record(private)], "outputs": [file_record(summary)]}
        (target / "summary.manifest.json").write_bytes(export.encode(manifest))
    return root


def test_exact_recursive_schema_and_semantics_are_accepted():
    raw = results()
    before = copy.deepcopy(raw)
    for name, value in raw.items():
        assert export.project_result(name, value) == value
    export.validate_consistency(raw)
    assert raw == before
    assert raw["lineage"]["current_redundant_file_replays_final_faces"] is False
    assert "not recovered" in raw["reconstruction"]["scope"]
    assert "no true-identity purity" in raw["integrity"]["scope"]


@pytest.mark.parametrize("kind", ["extra", "nested_extra", "count_vector", "count_bool", "count_string",
                                  "negative", "fraction_vector", "nan", "inf", "over_one", "prose", "gate"])
def test_refuses_unknown_fields_vectors_wrong_types_and_unreviewed_claims(kind):
    raw = results()["integrity"]
    if kind == "extra":
        raw["caption"] = "private"
    elif kind == "nested_extra":
        raw["constituent_coverage"]["person_id"] = "-12345678_98765432"
    elif kind == "count_vector":
        raw["positive_groups"] = [13142]
    elif kind == "count_bool":
        raw["positive_groups"] = True
    elif kind == "count_string":
        raw["positive_groups"] = "13142"
    elif kind == "negative":
        raw["positive_groups"] = -1
    elif kind == "fraction_vector":
        raw["positive_group_single_fraction_all"] = [0.977]
    elif kind in ("nan", "inf", "over_one"):
        raw["positive_group_single_fraction_all"] = {"nan": float("nan"), "inf": float("inf"), "over_one": 1.1}[kind]
    elif kind == "prose":
        raw["scope"] = "All labels human verified"
    else:
        raw["publication_ready"] = True
    with pytest.raises(ValueError):
        export.project_result("integrity", raw)


@pytest.mark.parametrize("change", ["snapshot", "denominator", "readme"])
def test_inconsistent_denominators_snapshots_or_readme_findings_refused(change):
    raw = results()
    if change == "snapshot":
        raw["lineage"]["retained_groups"] += 1
    elif change == "denominator":
        raw["integrity"]["positive_group_single_fraction_cached"] = 0.99
    else:
        raw["integrity"]["retained_groups_with_noisy_constituent"] += 1
    with pytest.raises(ValueError):
        export.validate_consistency(raw)


def test_export_exact_membership_checksums_and_count_only_manifests(tmp_path):
    root = fixture_root(tmp_path)
    paths = sorted(root.rglob("*json"))
    before = [file_record(path) for path in paths]
    target = export.build_export(root, tmp_path / "export")
    with zipfile.ZipFile(target) as archive:
        assert len(archive.namelist()) == 8
        assert set(archive.namelist()) == export.MEMBERS
        certificate = json.loads(archive.read("CERTIFICATE.json"))
        assert certificate["publication_ready"] is False
        assert certificate["human_adjudication_completed"] is False
        assert certificate["disclosure_review_completed"] is False
        assert certificate["full_reproduction"] is False
        assert b"not a privacy certification" in archive.read("README.txt")
        for name in export.EXPORTS:
            projection = json.loads(archive.read(f"manifests/{name}.sanitized.json"))
            assert projection["locally_verified_input_records"] == 1
            assert projection["record_paths_and_digests_omitted"] is True
            assert not {"inputs", "outputs", "parameters", "command"} & projection.keys()
            exported = archive.read(f"results/{name}.json")
            assert certificate["aggregate_bindings"][name]["exported_aggregate_sha256"] == export.digest(exported)
        for name, digest in certificate["members_excluding_certificate"].items():
            assert export.digest(archive.read(name)) == digest
        all_bytes = b"".join(archive.read(name) for name in archive.namelist())
        for canary in (b"PRIVATE_CAPTION_CANARY", b"PRIVATE_COMMAND_CANARY", b"PRIVATE_PARAMETER_CANARY",
                       b"-12345678_98765432", str(root).encode(), b"private_canary.json"):
            assert canary not in all_bytes
    assert before == [file_record(path) for path in paths]
    with pytest.raises(FileExistsError):
        export.build_export(root, tmp_path / "export")


@pytest.mark.parametrize("kind", ["input", "result", "binding", "metrics"])
def test_export_rejects_bad_input_result_or_manifest_binding_before_writing(tmp_path, kind):
    root = fixture_root(tmp_path)
    directory = root / "metrics" / export.EXPORTS["lineage"][0]
    manifest_path = directory / "summary.manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if kind == "input":
        (directory / "private/private_canary.json").write_bytes(b"tampered")
    elif kind == "result":
        (directory / "summary.json").write_bytes(b"tampered")
    elif kind == "binding":
        manifest["outputs"] = manifest["inputs"]
    else:
        manifest["metrics"]["retained_groups"] += 1
    manifest_path.write_bytes(export.encode(manifest))
    with pytest.raises(ValueError):
        export.build_export(root, tmp_path / "export")
    assert not (tmp_path / "export" / export.ZIP_NAME).exists()


@pytest.mark.parametrize("extra", ["../private.json", "results\\private.json", "private/candidates.jsonl", "CERTIFICATE.json"])
def test_archive_refuses_traversal_private_members_and_duplicates(tmp_path, extra):
    archive_path = tmp_path / "invalid.zip"
    expected = {name: export.digest(b"{}") for name in export.MEMBERS}
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in export.MEMBERS:
            archive.writestr(name, b"{}")
        archive.writestr(extra, b"{}")
    with pytest.raises(ValueError, match="membership"):
        export.verify_archive(archive_path, expected)


def test_archive_refuses_tampering(tmp_path):
    archive_path = tmp_path / "invalid.zip"
    expected = {name: export.digest(b"{}") for name in export.MEMBERS}
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in export.MEMBERS:
            archive.writestr(name, b"{\"tampered\":true}" if name == "results/lineage.json" else b"{}")
    with pytest.raises(ValueError, match="checksum"):
        export.verify_archive(archive_path, expected)
