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
        assert "export_public_evidence.py" in command[1]
        assert cwd == root
        staging = Path(command[command.index("--out") + 1])
        assert staging.is_relative_to(root / ".work")
        staging.mkdir(parents=True)
        with zipfile.ZipFile(staging / "public_evidence_bundle.zip", "w") as archive:
            archive.writestr("CERTIFICATE.json", "{}")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(build_submission, "_run", fake_export)
    build_submission._package_uploads(sub, {"main": 1}, include_experiment_index=True)
    evidence = sub / "upload/publication-evidence.zip"
    manifest = json.loads((sub / "upload/artifact-manifest.json").read_text(encoding="utf-8"))
    record = next(item for item in manifest["artifacts"] if item["file"] == evidence.name)
    assert record["sha256"] == build_submission._sha256(evidence)
    with zipfile.ZipFile(sub / "upload/manuscript-source.zip") as archive:
        assert "publication-evidence.zip" not in archive.namelist()


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


def test_nonblank_page_check_resolves_relative_pdf_before_changing_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_submission.shutil, "which", lambda command: "pdftotext")

    def fake_extract(command, cwd):
        assert Path(command[-2]) == tmp_path / "upload/main.pdf"
        assert cwd == tmp_path / "upload"
        return subprocess.CompletedProcess(command, 0, "page text", "")

    monkeypatch.setattr(build_submission, "_run", fake_extract)
    build_submission._check_nonblank_pages(Path("upload/main.pdf"), 1)
