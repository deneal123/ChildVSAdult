import subprocess

import pytest

from latex import build_submission as builder


def test_compile_only_rejects_canonical_destination():
    with pytest.raises(ValueError, match="isolated"):
        builder.build("journal-1-tbiom", compile_only=True)


def test_compile_only_never_refreshes_index_or_packages(tmp_path, monkeypatch):
    latex = tmp_path / "latex"
    paper = latex / "papers/journal-1-tbiom"
    (paper / "en").mkdir(parents=True)
    (paper / "en/main.tex").write_text("aggregate", encoding="utf-8")
    (paper / "submission").mkdir()
    original = paper / "submission/main.pdf"
    original.write_bytes(b"canonical")
    (latex / "shared").mkdir()
    (latex / "shared/refs.bib").write_text("references", encoding="utf-8")
    index = tmp_path / "metrics/publication_artifact_index.json"
    index.parent.mkdir()
    index.write_bytes(b"unchanged index")
    monkeypatch.setattr(builder, "LATEX", latex)
    calls = []

    def run(command, cwd):
        calls.append(command)
        assert command[0] == "pdflatex"
        (cwd / "main.log").write_text("Output written on main.pdf (1 pages).", encoding="utf-8")
        (cwd / "main.pdf").write_bytes(b"isolated PDF")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(builder, "_run", run)

    def no_package(*args, **kwargs):
        pytest.fail("compile-only must not package uploads")

    monkeypatch.setattr(builder, "_package_uploads", no_package)
    out = tmp_path / "preview"
    assert builder.build("journal-1-tbiom", submission_dir=out, compile_only=True)
    assert len(calls) == 3
    assert index.read_bytes() == b"unchanged index"
    assert original.read_bytes() == b"canonical"
    assert not (out / "upload").exists()
