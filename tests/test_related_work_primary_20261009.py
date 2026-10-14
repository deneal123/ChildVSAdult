"""Working-master positioning guards; not primary-literature or PDF acceptance."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "latex/papers/journal-1-tbiom/en/main.tex").read_text(encoding="utf-8")
BIB = (ROOT / "latex/shared/refs.bib").read_text(encoding="utf-8")


@pytest.mark.parametrize("key", ["hou2021mtmim", "hossain2026longitudinal", "wu2026goldilocks", "luevano2025identity"])
def test_verified_reference_has_one_entry_and_is_cited(key):
    assert BIB.count("{" + key + ",") == 1
    assert key in MAIN


def test_hou_publisher_pagination_and_registered_doi():
    entry = BIB.split("@inproceedings{hou2021mtmim,", 1)[1].split("\n}", 1)[0]
    assert "3672--3681" in entry
    assert "10.1109/ICCV48922.2021.00367" in entry
    assert "not a new disentanglement method" in MAIN


def test_synthesis_is_not_promoted_to_our_controlled_sota_result():
    assert "does not share our backbone or benchmark" in MAIN
    assert "FRAN controls do not cover all synthesis methods" in MAIN
    assert "10.1109/WACVW68408.2026.00107" in BIB
    assert "10.1109/IJCB65343.2025.11411093" in BIB


def test_benchmark_ceiling_is_not_our_mechanism_proof():
    assert "benchmark ceilings do not establish our fine-tuning failure mechanism" in MAIN
    assert "weak backbone for clean attribution" not in MAIN
    assert "controlled fine-tuning comparison" in MAIN
