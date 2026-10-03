import pytest

from scripts.audit_compile_preview import preview


def fixture(tmp_path):
    paper = tmp_path / "latex/papers/journal-1-tbiom"
    (paper / "en").mkdir(parents=True)
    (paper / "submission").mkdir()
    (paper / "submission/strip_comments.py").write_text(
        "def strip_comment(line):\n    return line.split('%', 1)[0]\n", encoding="utf-8"
    )
    shared = tmp_path / "latex/shared"
    (shared / "vendor/ieee-tnnls").mkdir(parents=True)
    (shared / "refs.bib").write_text("references", encoding="utf-8")
    (shared / "vendor/ieee-tnnls/IEEEtran.cls").write_text("class", encoding="utf-8")
    out = tmp_path / "preview"
    for stem, directory in (("main", "manuscript"), ("supplement", "supplement")):
        (paper / f"en/{stem}.tex").write_text("aggregate\n", encoding="utf-8")
        stage = out / directory
        stage.mkdir(parents=True)
        (stage / f"{stem}.tex").write_text("aggregate\n", encoding="utf-8")
        (stage / f"{stem}.log").write_text("no reference errors", encoding="utf-8")
        (stage / f"{stem}.pdf").write_bytes(b"fixture PDF")
        (stage / "refs.bib").write_text("references", encoding="utf-8")
        (stage / "IEEEtran.cls").write_text("class", encoding="utf-8")
    return tmp_path, out


def test_preview_keeps_upload_visual_and_execution_gates_open(tmp_path):
    result, _, _ = preview(*fixture(tmp_path), extract=lambda _: 10)
    assert result["pages"] == {"main": 10, "supplement": 10}
    assert not result["compiler_execution_attested"] and not result["visual_review_completed"]
    assert not result["portal_proof_checked"] and not result["publication_ready"]


@pytest.mark.parametrize("failure", ["source", "refs", "undefined", "page_limit", "upload"])
def test_preview_fails_closed_on_inconsistent_source_or_publication_shape(tmp_path, failure):
    root, out = fixture(tmp_path)
    stage = out / "manuscript"
    count = 10
    if failure == "source":
        (stage / "main.tex").write_text("wrong source", encoding="utf-8")
    elif failure == "refs":
        (stage / "refs.bib").write_text("wrong references", encoding="utf-8")
    elif failure == "undefined":
        (stage / "main.log").write_text(
            "LaTeX Warning: There were undefined references.", encoding="utf-8"
        )
    elif failure == "page_limit":
        count = 11
    else:
        (out / "upload").mkdir()
    with pytest.raises(ValueError):
        preview(root, out, extract=lambda _: count)
