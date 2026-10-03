import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.render_lfw_evidence import render

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "metrics/lfw_bound_evaluation_20261002/lfw_bound_evaluation.json"


def payload():
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_actual_supplement_block_equals_generated_aggregate_table():
    text = (ROOT / "latex/papers/journal-1-tbiom/en/supplement.tex").read_text(encoding="utf-8")
    table = render(payload())
    assert table.strip() in text
    assert table == (RESULT.parent / "lfw_table.tex").read_text(encoding="utf-8")
    assert "population of training seeds" in table
    assert "Legacy interleaved-cache results are not pooled" in table
    assert "Discrete ROC EER" in table
    assert "[-0.0037,+0.0091]" in table


@pytest.mark.parametrize("problem", ["legacy", "replay", "count", "folds", "calibration", "population", "valid_draws", "mean_seeds", "nan", "interval"])
def test_invalid_or_mislabelled_evidence_is_rejected(problem):
    result = deepcopy(payload())
    if problem == "legacy":
        result["protocol"] = "legacy"
    elif problem == "replay":
        result["cache_full_transform_replay"] = False
    elif problem == "count":
        result["n_pairs"] -= 1
    elif problem == "folds":
        result["models"]["frozen"]["official_folds"][0]["fold"] = 9
    elif problem == "calibration":
        result["bootstrap"]["accuracy_threshold_reselection"] = False
    elif problem == "population":
        result["bootstrap"]["includes_training_seed_population_uncertainty"] = True
    elif problem == "valid_draws":
        result["bootstrap"]["n_valid_roc"] = 0
    elif problem == "mean_seeds":
        result["seed_aggregate"]["roc_auc"]["n_seeds"] = 1
    elif problem == "nan":
        result["seed_aggregate"]["roc_auc"]["mean"] = float("nan")
    else:
        result["seed_aggregate"]["roc_auc"]["fixed_checkpoint_mean_gain_subject_ci95"] = [1, -1]
    with pytest.raises(ValueError):
        render(result)


def test_per_checkpoint_or_legacy_accuracy_does_not_replace_mean_column():
    result = payload()
    result["models"]["tuned_seed42"]["metrics"]["accuracy_official_folds"] = .1234
    table = render(result)
    assert "0.1234" not in table
    assert "0.9673" in table
