import copy
import json
from pathlib import Path

import pytest

from scripts.render_fgnet_metrics_v2 import headline, operating

ROOT = Path(__file__).resolve().parents[1]


def payload():
    return json.loads((ROOT / "metrics/fgnet_metrics_v2_20261003/summary.json").read_text(encoding="utf-8"))


def test_named_metrics_and_conditioning():
    data = payload()
    text = headline(data) + operating(data)
    assert "not ensemble scores or the training-seed population" in text
    assert "Interpolated EER" in text and "Discrete minimax EER" in text
    assert r"TAR@FAR=0.1\%" in text
    assert "not calibrated for deployment" in text
    assert "$[+0.0074,+0.0916]$" in text
    assert "0.8491" in text and "0.0017" in text


@pytest.mark.parametrize("field,value", [("original_error_protocol_reused", True),
    ("image_inference_performed", True), ("publication_ready", True), ("cached_images", 648)])
def test_substitute_or_overclaim_rejected(field, value):
    data = payload()
    data[field] = value
    with pytest.raises(ValueError):
        headline(data)


def test_unpaired_inference_rejected():
    data = copy.deepcopy(payload())
    data["large_gap_25plus"]["bootstrap"]["shared_draws_across_models"] = False
    with pytest.raises(ValueError):
        operating(data)


def test_generated_tables_verbatim_and_low_far_limits():
    data = payload()
    folder = ROOT / "metrics/fgnet_metrics_v2_20261003"
    main = (ROOT / "latex/papers/journal-1-tbiom/en/main.tex").read_text(encoding="utf-8")
    supplement = (ROOT / "latex/papers/journal-1-tbiom/en/supplement.tex").read_text(encoding="utf-8")
    assert headline(data) == (folder / "headline_table.tex").read_text(encoding="utf-8")
    assert operating(data) == (folder / "operating_table.tex").read_text(encoding="utf-8")
    assert headline(data) in main
    assert operating(data) in supplement
    for metric in ("eer_interpolated", "eer_discrete_minimax", "tar@far=0.01", "tar@far=0.001"):
        lo, hi = data["large_gap_25plus"]["three_checkpoint_aggregate"][metric]["delta_ci95"]
        assert lo <= 0 <= hi
    assert "both TAR-change intervals include zero" in main
