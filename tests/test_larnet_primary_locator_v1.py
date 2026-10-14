"""Preserve the checked PMLR locator and the pose-only scope of its analogy."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_larnet_matches_checked_pmlr_record_not_the_other_yang_paper():
    refs = (ROOT / "latex/shared/refs.bib").read_text(encoding="utf-8")
    entry = refs.split("@inproceedings{yang2021larnet,", 1)[1].split("\n}", 1)[0]
    assert "LARNet}: Lie Algebra Residual Network for Face Recognition" in entry
    assert "https://proceedings.mlr.press/v139/yang21d.html" in entry
    assert "yang21i" not in entry
    for field, value in (("volume", "139"), ("pages", "11738--11750"), ("year", "2021")):
        assert re.search(rf"\b{field}\s*=\s*\{{{value}\}}", entry)
    # The checked record displays no DOI; do not copy the distinct LARNeXt DOI.
    assert "10.1109/TPAMI.2023.3279378" not in entry


def test_larnet_and_larnext_remain_pose_analogies_not_cafr_baselines():
    main = (ROOT / "latex/papers/journal-1-tbiom/en/main.tex").read_text(encoding="utf-8")
    scope = main.split(r"\noindent\textbf{Scope and evaluation.}", 1)[1].split(
        r"\begin{table*}", 1)[0]
    assert r"\emph{pose}-induced" in scope
    assert "motivate our diagnostics only by analogy" in scope
