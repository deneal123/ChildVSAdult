"""The working description must match the unchanged aggregation rule, not confidence."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "latex/papers/journal-1-tbiom/en"


def test_gender_tie_missing_and_unknown_description_is_explicit():
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    supplement = (PAPER / "supplement.tex").read_text(encoding="utf-8")
    assert "Missing entries are skipped; ties retain the first encountered label" in main
    assert "not a confidence-based decision" in main
    for claim in ("first encountered", "order-dependent", "returns missing attributes",
                  r"Unexpected model labels map to \texttt{?}",
                  "Neither a tie nor a low consistency score automatically removes a group"):
        assert claim in supplement
    assert "consistency is diagnostic, not a pruning rule or identity evidence" in main
