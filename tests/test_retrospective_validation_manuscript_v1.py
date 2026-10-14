"""Keep native generated validation values and conditional scope in the article."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "latex/papers/journal-1-tbiom/en/supplement.tex"
BASE = ROOT / "metrics/fixed_validation_presentation_v1_e368_20261009"


def test_native_validation_projection_matches_article():
    text = SOURCE.read_text(encoding="utf-8")
    assert (BASE / "validation_table.tex").read_text(encoding="utf-8").strip() in text
    assert (BASE / "validation_statement.tex").read_text(encoding="utf-8").strip() in text


def test_validation_claim_is_retrospective_and_conditional():
    text = SOURCE.read_text(encoding="utf-8")
    section = text.split(r"\section{Retrospective Fixed-Checkpoint Validation Loss}")[1].split(r"\section{Global FG-NET")[0]
    for phrase in ("one completed", "without", "checkpoint reselection", "all pair weights",
                   "does not reconstruct", "does not identify", "not subject or training-seed population"):
        assert phrase in section
    for label in ("sec:retrospective-validation-loss", "tab:retrospective-validation-loss"):
        assert text.count(r"\label{" + label + "}") == 1
    assert not any(line.startswith("+\\") for line in text.splitlines())
