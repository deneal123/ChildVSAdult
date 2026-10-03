"""Guard cross-document targets independently of the current Roman numbers."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "latex/papers/journal-1-tbiom/en"


def roman(number):
    result = ""
    for value, symbol in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while number >= value:
            result += symbol
            number -= value
    return result


def main_targets():
    source = (PAPER / "main.tex").read_text(encoding="utf-8")
    sections, section, subsection = {}, 0, 0
    for match in re.finditer(r"\\(section|subsection)\{[^}]*\}\\label\{([^}]+)\}", source):
        kind, label = match.groups()
        if kind == "section":
            section += 1
            subsection = 0
            sections[label] = roman(section)
        else:
            subsection += 1
            sections[label] = roman(section) + "-" + chr(64 + subsection)
    tables = {label: roman(index) for index, label in enumerate(
        re.findall(r"\\label\{(tab:[^}]+)\}", source), start=1)}
    return sections, tables


def test_supplement_main_sections_point_to_their_actual_semantic_targets():
    source = (PAPER / "supplement.tex").read_text(encoding="utf-8")
    sections, _ = main_targets()
    targets = {
        "Pre-clean ablation:": "sec:res-main",
        "$+$real $0.848$": "sec:res-synth",
        "no objective-equivalence test": "sec:res-objective",
        "not an equivalence or label-noise guarantee": "sec:res-objective",
        "of the recorded gain at 10": "sec:res-scaling",
        "historical threshold audit": "sec:res-fairness",
        "stable over cosine": "sec:data-curation",
        "small, noisy sets": "sec:res-crossplatform",
        "null/negative strong-backbone gains": "sec:res-headroom",
        "formal institutional determination not yet obtained": "sec:ethics",
        "consent-based, human-in-the-loop settings": "sec:ethics",
        "if approval excludes a category": "sec:ethics",
    }
    for fragment, target in targets.items():
        rows = [line for line in source.splitlines() if fragment in line]
        assert len(rows) == 1, fragment
        assert re.findall(r"\\mainsec\{([^}]+)\}", rows[0]) == [sections[target]], target


def test_supplement_main_tables_are_qualified_and_match_main_numbering():
    source = (PAPER / "supplement.tex").read_text(encoding="utf-8")
    _, tables = main_targets()
    targets = {
        "stable over cosine": "tab:dedupsens",
        "Age extraction (LLM vs.": "tab:age",
        "Group integrity (LLM classifier)": "tab:integrity",
        "Near-duplicate inflation (within-person dedup)": "tab:dedupsens",
    }
    for fragment, target in targets.items():
        rows = [line for line in source.splitlines() if fragment in line]
        assert len(rows) == 1, fragment
        assert re.findall(r"\\maintab\{([^}]+)\}", rows[0]) == [tables[target]], target


def test_no_unqualified_literal_supplement_cross_document_references():
    source = (PAPER / "supplement.tex").read_text(encoding="utf-8")
    assert "all other section, table and figure references are local" in source
    assert not re.search(r"(?:Table|Sec\.)~[IVX]+(?:-[A-Z])?\b", source)
    assert r"\newcommand{\mainsec}[1]{main Sec.~#1}" in source
    assert r"\newcommand{\maintab}[1]{main Table~#1}" in source


def test_main_supplement_targets_use_descriptions_not_stale_literal_counters():
    source = (PAPER / "main.tex").read_text(encoding="utf-8")
    supplement = (PAPER / "supplement.tex").read_text(encoding="utf-8")
    assert not re.search(r"Supplementary Table~S-[IVX]+", source)
    for description, label in (("scaling table and figure", "tab:supp-scaling"),
                               ("apparent-strata table and figure", "tab:supp-strata"),
                               ("hard-negative table", "tab:supp-hardneg")):
        assert f"supplement, {description}" in source
        assert rf"\label{{{label}}}" in supplement
