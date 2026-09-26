from pathlib import Path

import pytest

from latex.build_submission import _check_source_privacy

ROOT = Path(__file__).resolve().parents[1]


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
