import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from latex import build_submission
from latex.build_submission import _check_source_privacy, _latex_reference_errors

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("log", [
    "LaTeX Warning: Citation `paper' on page 4 undefined on input line 26\n0.",
    "LaTeX Warning: Citation `paper' on page 4\nundefined on input line 260.",
    "LaTeX Warning: Reference `table' on page 1 undefined on input line 10.",
    "LaTeX Warning: Label `table' multiply defined.",
    "LaTeX Warning: There were undefined references.",
])
def test_submission_gate_rejects_undefined_citations_and_references(log):
    assert _latex_reference_errors(log) > 0


def test_reference_gate_accepts_success_and_nonreference_warning():
    assert _latex_reference_errors("Output written on main.pdf (10 pages).") == 0
    assert _latex_reference_errors("LaTeX Warning: Text page 5 contains only floats.") == 0


def test_standalone_supplement_provides_its_own_bibliography():
    source = (ROOT / "latex/papers/journal-1-tbiom/en/supplement.tex").read_text(encoding="utf-8")
    assert r"\bibliographystyle{IEEEtran}" in source
    assert r"\bibliography{../../../shared/refs}" in source


def test_submission_privacy_check_accepts_aggregate_text(tmp_path) -> None:
    source = tmp_path / "main.tex"
    source.write_text(
        r"Pairs:\\ evidence at https://github.com/example/project; aggregate count: 31,865.",
        encoding="utf-8",
    )

    _check_source_privacy([source])


@pytest.mark.parametrize(
    "text",
    [
        r"private face -134190297_457601102",
        "client_secret=do-not-package-this-value",
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz1234",
        r"C:\Users\local-user\.cache\weights.pt",
        "/home/local-user/.cache/weights.pt",
    ],
)
def test_submission_privacy_check_rejects_identifiers_and_secrets(tmp_path, text) -> None:
    source = tmp_path / "main.tex"
    source.write_text(text, encoding="utf-8")

    with pytest.raises(RuntimeError):
        _check_source_privacy([source])


def test_tbiom_editorial_materials_disclose_prior_submission_and_minors() -> None:
    submission = ROOT / "latex/papers/journal-1-tbiom/submission"
    cover = (submission / "cover_letter.tex").read_text(encoding="utf-8")
    portal = (submission / "portal_answers.md").read_text(encoding="utf-8")

    for text in (cover, portal):
        assert "TBIOM-2026-07-0222" in text
        assert "another IEEE journal" not in text
        assert "minors are excluded" not in text
    assert "includes images depicting minors" in cover
    assert "Images depicting minors were included" in portal


def test_tbiom_public_release_does_not_promise_row_level_biometric_protocol() -> None:
    paper = ROOT / "latex/papers/journal-1-tbiom/en/main.tex"
    submission = ROOT / "latex/papers/journal-1-tbiom/submission"
    texts = [
        paper.read_text(encoding="utf-8"),
        (submission / "cover_letter.tex").read_text(encoding="utf-8"),
        (submission / "portal_answers.md").read_text(encoding="utf-8"),
    ]

    for text in texts:
        lowered = text.lower()
        assert "de-identified pair protocol" not in lowered
        assert "salted identifier hashes" not in lowered


def _synthetic_submission(tmp_path):
    sub = tmp_path / "submission"
    man = sub / "manuscript"
    (man / "figures").mkdir(parents=True)
    for name in ("main.tex", "main.bbl", "refs.bib", "IEEEtran.cls"):
        (man / name).write_text("aggregate only", encoding="utf-8")
    (man / "main.pdf").write_bytes(b"synthetic PDF")
    return sub


def test_tbiom_packages_evidence_separately_and_binds_its_hash(tmp_path, monkeypatch):
    sub = _synthetic_submission(tmp_path)
    root = tmp_path / "root"
    latex = root / "latex"
    latex.mkdir(parents=True)
    monkeypatch.setattr(build_submission, "LATEX", latex)
    monkeypatch.setattr(build_submission, "_check_nonblank_pages", lambda *args: None)

    def fake_export(command, cwd):
        if command[1] == "-m":
            assert command[2] in {"scripts.export_lfw_evidence", "scripts.export_curation_evidence", "scripts.export_roc_v2_evidence", "scripts.export_cacd_evidence"}
            filename = {"scripts.export_lfw_evidence": "lfw_evidence_bundle.zip",
                        "scripts.export_curation_evidence": "curation_evidence_bundle.zip",
                        "scripts.export_roc_v2_evidence": "roc_v2_evidence_bundle.zip",
                        "scripts.export_cacd_evidence": "cacd_evidence_bundle.zip"}[command[2]]
        else:
            assert "export_public_evidence.py" in command[1]
            filename = "public_evidence_bundle.zip"
        assert cwd == root
        staging = Path(command[command.index("--out") + 1])
        assert staging.is_relative_to(root / ".work")
        staging.mkdir(parents=True)
        with zipfile.ZipFile(staging / filename, "w") as archive:
            archive.writestr("CERTIFICATE.json", "{}")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(build_submission, "_run", fake_export)
    build_submission._package_uploads(sub, {"main": 1}, include_experiment_index=True)
    evidence = sub / "upload/publication-evidence.zip"
    manifest = json.loads((sub / "upload/artifact-manifest.json").read_text(encoding="utf-8"))
    record = next(item for item in manifest["artifacts"] if item["file"] == evidence.name)
    assert record["sha256"] == build_submission._sha256(evidence)
    lfw = sub / "upload/lfw-evidence.zip"
    lfw_record = next(item for item in manifest["artifacts"] if item["file"] == lfw.name)
    assert lfw_record["sha256"] == build_submission._sha256(lfw)
    curation = sub / "upload/curation-evidence.zip"
    curation_record = next(item for item in manifest["artifacts"] if item["file"] == curation.name)
    assert curation_record["sha256"] == build_submission._sha256(curation)
    roc = sub / "upload/roc-v2-evidence.zip"
    roc_record = next(item for item in manifest["artifacts"] if item["file"] == roc.name)
    assert roc_record["sha256"] == build_submission._sha256(roc)
    cacd = sub / "upload/cacd-evidence.zip"
    cacd_record = next(item for item in manifest["artifacts"] if item["file"] == cacd.name)
    assert cacd_record["sha256"] == build_submission._sha256(cacd)
    with zipfile.ZipFile(sub / "upload/manuscript-source.zip") as archive:
        assert "publication-evidence.zip" not in archive.namelist()
        assert "lfw-evidence.zip" not in archive.namelist()
        assert "curation-evidence.zip" not in archive.namelist()
        assert "roc-v2-evidence.zip" not in archive.namelist()
        assert "cacd-evidence.zip" not in archive.namelist()


def test_evidence_export_failure_preserves_previous_upload(tmp_path, monkeypatch):
    sub = _synthetic_submission(tmp_path)
    (sub / "upload").mkdir()
    old = sub / "upload/main.pdf"
    old.write_bytes(b"previous valid upload")
    monkeypatch.setattr(build_submission, "_run", lambda command, cwd:
                        subprocess.CompletedProcess(command, 1, "", "checksum failure"))
    with pytest.raises(RuntimeError, match="checksum failure"):
        build_submission._package_uploads(sub, {}, include_experiment_index=True)
    assert old.read_bytes() == b"previous valid upload"


def test_non_tbiom_package_does_not_export_tbiom_evidence(tmp_path, monkeypatch):
    sub = _synthetic_submission(tmp_path)
    monkeypatch.setattr(build_submission, "_check_nonblank_pages", lambda *args: None)
    monkeypatch.setattr(build_submission, "_run", lambda *args: pytest.fail("unexpected export"))
    build_submission._package_uploads(sub, {"main": 1})
    assert not (sub / "upload/publication-evidence.zip").exists()
    assert not (sub / "upload/lfw-evidence.zip").exists()
    assert not (sub / "upload/curation-evidence.zip").exists()
    assert not (sub / "upload/roc-v2-evidence.zip").exists()
    assert not (sub / "upload/cacd-evidence.zip").exists()


def test_lfw_export_failure_preserves_previous_uploads_after_first_export_succeeds(tmp_path, monkeypatch):
    sub = _synthetic_submission(tmp_path)
    (sub / "upload").mkdir()
    originals = {name: b"previous valid bytes" for name in ("main.pdf", "publication-evidence.zip", "lfw-evidence.zip")}
    for name, data in originals.items():
        (sub / "upload" / name).write_bytes(data)

    def exports(command, cwd):
        if command[1] == "-m":
            return subprocess.CompletedProcess(command, 1, "", "LFW checksum failure")
        staging = Path(command[command.index("--out") + 1])
        staging.mkdir(parents=True)
        (staging / "public_evidence_bundle.zip").write_bytes(b"new first archive")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(build_submission, "_run", exports)
    with pytest.raises(RuntimeError, match="LFW checksum failure"):
        build_submission._package_uploads(sub, {}, include_experiment_index=True)
    for name, data in originals.items():
        assert (sub / "upload" / name).read_bytes() == data


def test_nonblank_page_check_resolves_relative_pdf_before_changing_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_submission.shutil, "which", lambda command: "pdftotext")

    def fake_extract(command, cwd):
        assert Path(command[-2]) == tmp_path / "upload/main.pdf"
        assert cwd == tmp_path / "upload"
        return subprocess.CompletedProcess(command, 0, "page text", "")

    monkeypatch.setattr(build_submission, "_run", fake_extract)
    build_submission._check_nonblank_pages(Path("upload/main.pdf"), 1)


def test_curation_export_failure_preserves_all_previous_uploads(tmp_path, monkeypatch):
    sub = _synthetic_submission(tmp_path)
    (sub / "upload").mkdir()
    originals = {name: b"previous valid bytes" for name in (
        "main.pdf", "publication-evidence.zip", "lfw-evidence.zip", "curation-evidence.zip")}
    for name, data in originals.items():
        (sub / "upload" / name).write_bytes(data)

    def exports(command, cwd):
        if "scripts.export_curation_evidence" in command:
            return subprocess.CompletedProcess(command, 1, "", "curation checksum failure")
        staging = Path(command[command.index("--out") + 1])
        staging.mkdir(parents=True)
        name = "lfw_evidence_bundle.zip" if "scripts.export_lfw_evidence" in command else "public_evidence_bundle.zip"
        (staging / name).write_bytes(b"new intermediate archive")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(build_submission, "_run", exports)
    with pytest.raises(RuntimeError, match="curation checksum failure"):
        build_submission._package_uploads(sub, {}, include_experiment_index=True)
    for name, data in originals.items():
        assert (sub / "upload" / name).read_bytes() == data


def test_cacd_export_failure_preserves_all_previous_uploads(tmp_path, monkeypatch):
    sub = _synthetic_submission(tmp_path)
    (sub / "upload").mkdir()
    originals = {name: b"previous valid bytes" for name in (
        "main.pdf", "publication-evidence.zip", "lfw-evidence.zip", "curation-evidence.zip",
        "roc-v2-evidence.zip", "cacd-evidence.zip")}
    for name, data in originals.items():
        (sub / "upload" / name).write_bytes(data)

    def exports(command, cwd):
        if "scripts.export_cacd_evidence" in command:
            return subprocess.CompletedProcess(command, 1, "", "CACD checksum failure")
        staging = Path(command[command.index("--out") + 1])
        staging.mkdir(parents=True)
        filename = {"scripts.export_lfw_evidence": "lfw_evidence_bundle.zip",
            "scripts.export_curation_evidence": "curation_evidence_bundle.zip",
            "scripts.export_roc_v2_evidence": "roc_v2_evidence_bundle.zip"}.get(command[2], "public_evidence_bundle.zip")
        (staging / filename).write_bytes(b"new intermediate archive")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(build_submission, "_run", exports)
    with pytest.raises(RuntimeError, match="CACD checksum failure"):
        build_submission._package_uploads(sub, {}, include_experiment_index=True)
    for name, data in originals.items():
        assert (sub / "upload" / name).read_bytes() == data


def test_isolated_build_leaves_canonical_submission_untouched(tmp_path, monkeypatch):
    latex = tmp_path / "latex"
    paper = latex / "papers/journal-1-tnnls"
    (paper / "en").mkdir(parents=True)
    (paper / "en/main.tex").write_text("aggregate main", encoding="utf-8")
    (paper / "submission").mkdir()
    marker = paper / "submission/main.pdf"
    marker.write_bytes(b"active canonical build")
    strip = paper / "submission/strip_comments.py"
    strip.write_text("# fixture", encoding="utf-8")
    (latex / "shared/vendor/ieee-tnnls").mkdir(parents=True)
    (latex / "shared/refs.bib").write_text("refs", encoding="utf-8")
    (latex / "shared/vendor/ieee-tnnls/IEEEtran.cls").write_text("class", encoding="utf-8")
    monkeypatch.setattr(build_submission, "LATEX", latex)
    destination = tmp_path / ".work/isolated-submission"
    calls = []

    def run(command, cwd):
        calls.append((command, cwd))
        if command[0] == "pdflatex":
            (cwd / "main.log").write_text("Output written on main.pdf (1 pages).", encoding="utf-8")
            (cwd / "main.pdf").write_bytes(b"new isolated PDF")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(build_submission, "_run", run)
    packaged = []
    monkeypatch.setattr(build_submission, "_package_uploads", lambda sub, counts, **kwargs:
                        packaged.append((sub, counts, kwargs)))
    assert build_submission.build("journal-1-tnnls", submission_dir=destination)
    assert marker.read_bytes() == b"active canonical build"
    assert packaged == [(destination.resolve(), {"main": 1}, {"include_experiment_index": False})]
    assert calls[0][0][1] == str(strip)
    assert calls[0][1] == destination.resolve()
    assert (destination / "manuscript/main.tex").read_text() == "aggregate main"


def test_roc_export_failure_preserves_all_previous_uploads(tmp_path, monkeypatch):
    sub = _synthetic_submission(tmp_path)
    (sub / "upload").mkdir()
    originals = {name: b"previous valid bytes" for name in (
        "main.pdf", "publication-evidence.zip", "lfw-evidence.zip", "curation-evidence.zip", "roc-v2-evidence.zip")}
    for name, data in originals.items():
        (sub / "upload" / name).write_bytes(data)

    def exports(command, cwd):
        if "scripts.export_roc_v2_evidence" in command:
            return subprocess.CompletedProcess(command, 1, "", "ROC-v2 checksum failure")
        staging = Path(command[command.index("--out") + 1])
        staging.mkdir(parents=True)
        name = ("lfw_evidence_bundle.zip" if "scripts.export_lfw_evidence" in command else
                "curation_evidence_bundle.zip" if "scripts.export_curation_evidence" in command else "public_evidence_bundle.zip")
        (staging / name).write_bytes(b"intermediate evidence")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(build_submission, "_run", exports)
    with pytest.raises(RuntimeError, match="ROC-v2 checksum failure"):
        build_submission._package_uploads(sub, {}, include_experiment_index=True)
    for name, data in originals.items():
        assert (sub / "upload" / name).read_bytes() == data
