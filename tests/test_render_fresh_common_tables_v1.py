"""Synthetic dictionary formatting, not native acceptance or PDF proof."""

import pytest

from scripts import render_fresh_common_tables_v1 as renderer
from tests.test_fresh_common_presentation_contract_v1 import fixture


def test_literal_three_seed_rows_and_fresh_labels():
    payload, tables = renderer.render(fixture())
    weak = tables["fresh_weak_table.tex"]
    strong = tables["fresh_strong_table.tex"]
    assert weak.count("fresh tuned, seed ") == 3
    assert strong.count("random, head, LR ") == 1
    assert "tab:fresh-common-weak" in weak
    assert "legacy weak headline" in weak
    assert "not training-seed population" in weak
    assert "without refitting within draws" in weak
    assert "Lower age MAE means easier age decoding" in weak
    assert payload["publication_ready"] is False
    assert "not the full 36-cell matrix" in strong


def test_unknown_family_refused():
    payload, _ = renderer.render(fixture())
    with pytest.raises(ValueError):
        renderer.family_table(payload, "ensemble")


def test_bad_native_never_formats():
    native = fixture()
    native["experiment"] = "common-image-index-mechanism-v1"
    with pytest.raises(ValueError):
        renderer.render(native)
