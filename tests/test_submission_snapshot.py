import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts import audit_submission_snapshot as audit

ROOT = Path(__file__).resolve().parents[1]


def fixture_tree(tmp_path):
    root, sub = tmp_path / "root", tmp_path / "submission"
    paper = root / "latex/papers/journal-1-tbiom"
    (paper / "en").mkdir(parents=True)
    (paper / "submission").mkdir()
    shutil.copy2(ROOT / "latex/papers/journal-1-tbiom/submission/strip_comments.py", paper / "submission/strip_comments.py")
    (root / "latex").joinpath("build_submission.py").write_text("# fixture", encoding="utf-8")
    shared = root / "latex/shared"
    (shared / "vendor/ieee-tnnls").mkdir(parents=True)
    (shared / "figures").mkdir()
    (shared / "refs.bib").write_bytes(b"bibliography")
    (shared / "vendor/ieee-tnnls/IEEEtran.cls").write_bytes(b"class")
    (shared / "figures/figure.pdf").write_bytes(b"figure PDF fixture")
    (root / "metrics").mkdir()
    index = {"entry_count": 0, "entries": [], "complete": False, "missing_expected_manifests": []}
    (root / "metrics/publication_artifact_index.json").write_bytes(json.dumps(index).encode())
    for stem, folder in (("main", "manuscript"), ("supplement", "supplement")):
        tex = "% author block\n\n\\graphicspath{{../../../shared/figures/}}\n\\includegraphics{figure.pdf}\nvalue \\% not comment % comment\n"
        canonical = paper / "en" / f"{stem}.tex"
        canonical.write_text(tex, encoding="utf-8")
        target = sub / folder
        (target / "figures").mkdir(parents=True)
        (target / f"{stem}.tex").write_bytes(audit.stripped_source(canonical, paper / "submission/strip_comments.py").encode())
        (target / f"{stem}.log").write_text("Output written on PDF", encoding="utf-8")
        (target / f"{stem}.pdf").write_bytes(b"PDF fixture")
        shutil.copy2(shared / "refs.bib", target / "refs.bib")
        shutil.copy2(shared / "vendor/ieee-tnnls/IEEEtran.cls", target / "IEEEtran.cls")
        shutil.copy2(shared / "figures/figure.pdf", target / "figures/figure.pdf")
    (sub / "manuscript/main.bbl").write_bytes(b"bibliography output")
    upload = sub / "upload"
    upload.mkdir()
    for name in audit.UPLOAD_MEMBERS - {"artifact-manifest.json"}:
        if name.endswith(".pdf"):
            folder = "manuscript" if name == "main.pdf" else "supplement"
            shutil.copy2(sub / folder / name, upload / name)
        else:
            (upload / name).write_bytes(b"ZIP fixture")
    with zipfile.ZipFile(upload / "manuscript-source.zip", "w") as archive:
        for name in ("main.tex", "main.bbl", "refs.bib", "IEEEtran.cls"):
            archive.write(sub / "manuscript" / name, name)
        archive.write(sub / "manuscript/figures/figure.pdf", "figures/figure.pdf")
        archive.write(root / "metrics/publication_artifact_index.json", "artifacts/publication_artifact_index.json")
    rows = [{"file": name, "bytes": (upload / name).stat().st_size,
             "sha256": audit.sha256_file(upload / name),
             "pages": {"main.pdf": 10, "supplement.pdf": 2}.get(name)}
            for name in sorted(audit.UPLOAD_MEMBERS - {"artifact-manifest.json"})]
    (upload / "artifact-manifest.json").write_text(json.dumps({"artifacts": rows}), encoding="utf-8")
    return root, sub


def test_master_transform_and_source_zip_exact_bytes(tmp_path):
    root, sub = fixture_tree(tmp_path)
    paths, index = audit.check_sources(root, sub)
    assert index["complete"] is False
    assert root / "metrics/publication_artifact_index.json" in paths
    text = (sub / "manuscript/main.tex").read_text()
    assert "author block" not in text
    assert "\\% not comment" in text and "% comment" not in text
    assert "\\graphicspath{{./figures/}}" in text
    assert not (root / "latex/papers/journal-1-tbiom/submission/__pycache__").exists()


@pytest.mark.parametrize("change", ["master", "refs", "figure", "extra_figure", "log", "upload_pdf", "empty_bbl", "index", "crlf"])
def test_source_mismatch_failures(tmp_path, change):
    root, sub = fixture_tree(tmp_path)
    if change == "master":
        (root / "latex/papers/journal-1-tbiom/en/main.tex").write_text("changed", encoding="utf-8")
    elif change == "refs":
        (sub / "supplement/refs.bib").write_bytes(b"changed")
    elif change == "figure":
        (sub / "manuscript/figures/figure.pdf").write_bytes(b"changed")
    elif change == "extra_figure":
        (sub / "manuscript/figures/extra.pdf").write_bytes(b"extra")
    elif change == "log":
        (sub / "manuscript/main.log").write_text("LaTeX Warning: There were undefined references.", encoding="utf-8")
    elif change == "upload_pdf":
        (sub / "upload/main.pdf").write_bytes(b"changed")
    elif change == "empty_bbl":
        (sub / "manuscript/main.bbl").write_bytes(b"\n ")
    elif change == "index":
        (root / "metrics/publication_artifact_index.json").write_bytes(b"changed")
    else:
        path = sub / "manuscript/main.tex"
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError):
        audit.check_sources(root, sub)


@pytest.mark.parametrize("extra", ["../private.txt", "main.tex", "supplement.tex"])
def test_source_archive_extra_or_duplicate_refused(tmp_path, extra):
    root, sub = fixture_tree(tmp_path)
    with zipfile.ZipFile(sub / "upload/manuscript-source.zip", "a") as archive:
        archive.writestr(extra, b"extra")
    with pytest.raises(ValueError, match="membership"):
        audit.check_sources(root, sub)


def test_optional_and_optionless_literal_figures():
    assert audit.literal_figures(r"\includegraphics{a.pdf}\includegraphics[width=.5]{b.pdf}") == {"a.pdf", "b.pdf"}


@pytest.mark.parametrize("tex", [r"\includegraphics*{a.pdf}", r"\includegraphics{../a.pdf}",
                                 r"\includegraphics{a.png}", r"\includegraphics{\macro}"])
def test_unhandled_figure_syntax_refused(tex):
    with pytest.raises(ValueError):
        audit.literal_figures(tex)


def test_upload_inventory_and_real_extractor_count(tmp_path):
    _, sub = fixture_tree(tmp_path)
    assert audit.check_upload(sub / "upload", lambda path: 10 if path.stem == "main" else 2) == {"main": 10, "supplement": 2}


@pytest.mark.parametrize("change", ["extra", "tamper", "duplicate", "pages", "limit"])
def test_upload_failures(tmp_path, change):
    _, sub = fixture_tree(tmp_path)
    upload = sub / "upload"
    manifest_path = upload / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "extra":
        (upload / "private.txt").write_bytes(b"private")
    elif change == "tamper":
        (upload / "main.pdf").write_bytes(b"changed")
    elif change == "duplicate":
        manifest["artifacts"].append(manifest["artifacts"][0])
    elif change in ("pages", "limit"):
        next(row for row in manifest["artifacts"] if row["file"] == "main.pdf")["pages"] = 11
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        audit.check_upload(upload, lambda path: 11 if change == "limit" and path.stem == "main" else (10 if path.stem == "main" else 2))


@pytest.mark.parametrize("case", ["ok", "blank", "count", "missing", "timeout"])
def test_pdf_extraction_bounded_and_fail_closed(monkeypatch, case):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 30
        if case == "missing":
            raise FileNotFoundError("missing tool")
        if case == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        output = "Pages: 2\n" if command[0] == "pdfinfo" else {"ok": "page1\fpage2\f", "blank": "page1\f\f", "count": "page1\f"}[case]
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(audit.subprocess, "run", run)
    if case == "ok":
        assert audit.pdf_pages(Path("fixture.pdf")) == 2
        assert len(calls) == 2
    else:
        with pytest.raises((ValueError, FileNotFoundError, subprocess.TimeoutExpired)):
            audit.pdf_pages(Path("fixture.pdf"))
