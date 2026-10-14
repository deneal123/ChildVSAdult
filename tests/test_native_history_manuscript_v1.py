"""The ten-history subset is traceable and stays descriptive in the supplement."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "latex/papers/journal-1-tbiom/en"
EVIDENCE = ROOT / "metrics/strong_history_presentation10_v1_layout2_e337_20261009"


def test_public_figure_is_exact_verified_native_copy():
    manifest = json.loads((EVIDENCE / "history.manifest.json").read_text(encoding="utf8"))
    assert manifest["metrics"]["execution_complete"] is True
    assert manifest["metrics"]["publication_ready"] is False
    assert manifest["metrics"]["membership"]["declared_cell_count"] == 10
    assert manifest["metrics"]["membership"]["matrix_cell_count"] == 36
    records = [r for r in manifest["outputs"] if r["path"].endswith("/history_figure.pdf")]
    assert len(records) == 1
    raw = (ROOT / "latex/shared/figures/strong_native_histories10_layout2_v1.pdf").read_bytes()
    assert len(raw) == records[0]["bytes"]
    assert hashlib.sha256(raw).hexdigest() == records[0]["sha256"]


def test_history_table_has_only_verified_fit_wrapper():
    source = (PAPER / "supplement.tex").read_text(encoding="utf8")
    actual = (
        source.split("% BEGIN GENERATED STRONG HISTORY V1", 1)[1]
        .split("% END GENERATED STRONG HISTORY V1", 1)[0]
        .strip()
    )
    expected = (EVIDENCE / "history_table.tex").read_text(encoding="utf8").strip()
    expected = expected.replace(
        "\\centering\\scriptsize\n\\begin{tabular}",
        "\\centering\\scriptsize\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}",
    )
    expected = expected.replace("\\end{tabular}", "\\end{tabular}%\n}")
    assert actual == expected


def test_histories_are_supplement_only_and_not_an_extra_matrix():
    supplement = (PAPER / "supplement.tex").read_text(encoding="utf8")
    main = (PAPER / "main.tex").read_text(encoding="utf8")
    assert "strong_native_histories10_layout2_v1.pdf" in supplement
    assert "strong_native_histories10_layout2_v1.pdf" not in main
    assert "frozen first ten independently accepted cells" in supplement
    assert "not the completed matrix" in supplement
    assert "Gradient norms aggregate different trainable tensor" in supplement
    assert "not directly comparable across scopes" in supplement
    assert "these trajectories alone do not decide ceiling, plasticity or forgetting" in supplement
