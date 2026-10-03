import copy
import json
from pathlib import Path

import pytest

from scripts.render_cacd_metrics_v2 import LABELS, summary, table, validate

ROOT = Path(__file__).resolve().parents[1]


def payload():
    return json.loads(
        (ROOT / "metrics/cacd_serial_20261003/summary.json").read_text(encoding="utf-8")
    )


def test_all_six_metrics_are_generated_from_actual_source_bound_result():
    result = payload()
    rendered = table(result)
    assert all(label in rendered for label in LABELS)
    assert "0.9790" in rendered and "0.9707" in rendered
    assert "0.9625" in rendered and "0.9493" in rendered
    assert "conditional on these weights" in rendered
    assert "reused subjects/photos are not accounted for" in rendered
    assert "Historical CACD operating points are not pooled" in rendered
    paragraph = summary(result)
    assert "0.0220" in paragraph and "0.0313" in paragraph
    assert "AUC and TAR@FAR0.1" in paragraph and "intervals include zero" in paragraph
    assert "conditional pair-level" in paragraph


@pytest.mark.parametrize(
    "mutation", ["incomplete", "subject_ci", "ensemble", "delta", "mean", "sd", "interval", "nan"]
)
def test_semantic_or_numeric_corruption_fails_closed(mutation):
    result = payload()
    data = result["inference"]
    row = data["three_checkpoint_aggregate"]["roc_auc"]
    if mutation == "incomplete":
        result["execution_complete"] = False
    elif mutation == "subject_ci":
        data["bootstrap"]["subject_metadata_available"] = True
    elif mutation == "ensemble":
        data["bootstrap"]["conditioning"] = "ensemble confidence interval"
    elif mutation == "delta":
        row["mean_checkpoint_delta"] += 0.01
    elif mutation == "mean":
        row["mean"] -= 0.01
    elif mutation == "sd":
        row["sd"] += 0.01
    elif mutation == "interval":
        row["pair_mean_ci95"] = [0.99, 0.98]
    else:
        row["mean"] = float("nan")
    with pytest.raises(ValueError):
        validate(result)


def test_operating_point_narrative_cannot_survive_changed_interval_evidence():
    result = payload()
    result["inference"]["three_checkpoint_aggregate"]["eer_interpolated"][
        "paired_pair_delta_ci95"
    ] = [-0.01, 0.01]
    with pytest.raises(ValueError, match="narrative"):
        summary(result)


def test_rendering_is_nonmutating_and_does_not_export_paths_or_scores():
    result = payload()
    before = copy.deepcopy(result)
    output = table(result) + summary(result)
    assert result == before
    assert "models/" not in output and "C:\\" not in output and "scores.npz" not in output


def test_master_contains_exact_generated_main_and_supplement_blocks():
    result = payload()
    base = ROOT / "latex/papers/journal-1-tbiom/en"
    assert summary(result).strip() in (base / "main.tex").read_text(encoding="utf-8")
    assert table(result).strip() in (base / "supplement.tex").read_text(encoding="utf-8")
