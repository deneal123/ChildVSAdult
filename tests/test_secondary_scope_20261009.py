"""Secondary-study pointers must have actual source destinations, not promises."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "latex/papers/journal-1-tbiom/en"


def test_hard_negative_pointer_has_a_real_supplement_table_and_protocol():
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    supplement = (PAPER / "supplement.tex").read_text(encoding="utf-8")
    secondary = main.split(r"\subsection{Secondary Diagnostics}", 1)[1].split(r"\section{Discussion}", 1)[0]
    assert "supplement, hard-negative table and protocol" in secondary
    assert r"\paragraph{Historical hard-negative protocol.}" in supplement
    assert r"\label{tab:supp-hardneg}" in supplement
    assert "calibration" not in secondary
    assert "age-removal" not in secondary


def test_observed_objective_similarity_is_historical_and_overlap_is_named():
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    scope = main.split(r"\subsection{Scope of the Claims}", 1)[1].split(r"\section{Limitations}", 1)[0]
    assert "contrastive, ArcFace and CosFace objectives have similar historical point estimates" in scope
    assert "completed train--benchmark overlap adjudication" in scope
    assert "not demonstrated equivalence" in scope


def test_limitations_do_not_call_pending_human_supervision_audit_complete():
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    limits = main.split(r"\section{Limitations}", 1)[1].split(r"\section{Ethics", 1)[0]
    assert "The blinded human audit remains pending" in limits
    assert "persons---audited" not in limits
