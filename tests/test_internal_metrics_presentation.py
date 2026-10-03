import json
from pathlib import Path

import pytest

from scripts.reevaluate_internal_metrics_v2 import infer
from scripts.render_internal_metrics_v2 import render


def payload():
    return infer([.9, .4, .3, .2, .5, .6], [.9, .9, .8, .2, .3, .4],
                 [True, False, True], ["a", "a", "b"], ["b", "c", "c"], n_boot=20)


def test_named_definitions_and_limits():
    text = render(payload())
    assert "Interpolated EER" in text and "Discrete minimax EER" in text
    assert r"TAR@FAR=0.1\%" in text
    assert r"\label{tab:internal-roc-v2}" in text
    assert "not calibrated for deployment" in text
    assert "one fixed seed-42 checkpoint" in text


@pytest.mark.parametrize("field,value", [("metric_version", "legacy"), ("deployment_calibration", True)])
def test_wrong_scope_rejected(field, value):
    data = payload()
    data[field] = value
    with pytest.raises(ValueError):
        render(data)


def test_real_table_is_verbatim_in_supplement():
    root = Path(__file__).resolve().parents[1]
    folder = root / "metrics/internal_metrics_v2_20261003"
    data = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    expected = render(data)
    assert (folder / "internal_table.tex").read_text(encoding="utf-8") == expected
    supplement = (root / "latex/papers/journal-1-tbiom/en/supplement.tex").read_text(encoding="utf-8")
    assert expected in supplement
    for subset in data["subsets"].values():
        for metric in ("tar@far=0.01", "tar@far=0.001", "eer_interpolated", "eer_discrete_minimax"):
            lo, hi = subset["metrics"][metric]["tuned_minus_frozen"]["ci95"]
            assert lo <= 0 <= hi
