"""Bound current prose to corrected operating points and nominal old analyses.

These regression checks are not validation of historical training provenance,
human identities, multiplicity-adjusted inference or publication readiness.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "latex/papers/journal-1-tbiom/en/main.tex"
SUPPLEMENT = ROOT / "latex/papers/journal-1-tbiom/en/supplement.tex"


def test_corrected_internal_tar_is_mixed_not_uniform_decline():
    result = json.loads((ROOT / "metrics/internal_metrics_v2_20261003/summary.json").read_text(
        encoding="utf-8"))["subsets"]["original_tolerance"]["metrics"]
    first = result["tar@far=0.01"]["tuned_minus_frozen"]
    second = result["tar@far=0.001"]["tuned_minus_frozen"]
    assert first["point"] > 0 > second["point"]
    for metric in (first, second):
        assert metric["ci95"][0] <= 0 <= metric["ci95"][1]
    main = MAIN.read_text(encoding="utf-8")
    abstract = main.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
    assert r"TAR rises at FAR1\% but falls at FAR0.1\%" in abstract
    assert "both change intervals including zero" in abstract
    assert "test gains AUC but loses low-FAR TAR" not in main
    assert r"positive at FAR1\%, negative at FAR0.1\%" in main
    assert "gains AUC while losing low-FAR TAR" not in main
    assert "Internal AUC rises, while TAR changes are mixed" in main


def test_apparent_stratum_intervals_are_nominal_and_not_holm_claim():
    main = MAIN.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    assert "a Holm correction is applied" not in main
    assert "without a documented Holm correction" in main
    assert "not multiplicity-adjusted or subject-level inference" in main
    assert "Native rerun/provenance is pending" in supplement
    assert "without documented multiplicity adjustment" in supplement
    caption = supplement.split(r"\label{fig:fairness}", 1)[0].rsplit(r"\caption{", 1)[1]
    assert "nominal" in caption and "pair-level" in caption
    assert "not multiplicity-adjusted or subject-level evidence" in caption
    assert "every gain is significant" not in main


def test_historical_pair_tests_do_not_certify_corrected_headline():
    main = MAIN.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    assert "historical pair-level DeLong/permutation outputs do not establish significance" in main
    assert "separate from corrected headline inference" in supplement
    assert "DeLong tests for correlated ROC curves" not in main


def test_cross_community_effects_remain_unverified_historical_points():
    main = MAIN.read_text(encoding="utf-8")
    paragraph = main.split(r"\subsection{Cross-Source Transfer}", 1)[1].split(
        r"\subsection{Cross-Platform Transfer", 1)[0]
    assert "$+$0.113" in paragraph and "$+$0.092" in paragraph
    assert "historical point estimates under the old random-impostor protocol" in paragraph
    assert "identity independence remain unverified" in paragraph
    assert "does not establish source equivalence" in paragraph
    assert "two independent sources" not in paragraph
