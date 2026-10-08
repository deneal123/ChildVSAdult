import numpy as np
import pytest

from scripts.strong_subject_inference_v1 import infer


def fixture():
    y = np.array([1, 1, 1, 0, 0, 0])
    a = np.array(["a", "b", "c", "a", "b", "c"])
    b = np.array(["a", "b", "c", "b", "c", "a"])
    return (
        dict(
            frozen=np.array([0.8, 0.5, 0.6, 0.7, 0.2, 0.4]),
            seed42=np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.4]),
        ),
        y,
        a,
        b,
    )


def test_single_checkpoint_not_fabricated_three_seeds():
    data = infer(*fixture(), n_boot=50)
    assert data["actual_tuned_checkpoint_count"] == 1
    assert set(data["models"]) == {"frozen", "seed42"}
    assert "three_checkpoint_aggregate" not in data
    assert data["loo_valid"] == 3
    assert data["models"]["seed42"]["roc_auc"]["delta_vs_frozen"] > 0


def test_same_scores_have_zero_paired_delta():
    scores, y, a, b = fixture()
    scores["seed42"] = scores["frozen"].copy()
    data = infer(scores, y, a, b, n_boot=50)
    for metric in data["models"]["seed42"].values():
        assert metric["delta_ci95"] == [0, 0]
        assert metric["loo_delta_min"] == metric["loo_delta_max"] == 0


@pytest.mark.parametrize("change", ["unknown", "contradiction", "nonfinite", "no_frozen"])
def test_invalid_metadata_or_scores_rejected(change):
    scores, y, a, b = fixture()
    if change == "unknown":
        a = a.astype("U7")
        a[0] = "unknown"
    elif change == "contradiction":
        b[0] = "b"
    elif change == "nonfinite":
        scores["seed42"][0] = np.nan
    else:
        scores.pop("frozen")
    with pytest.raises(ValueError):
        infer(scores, y, a, b, n_boot=20)


def test_repeated_subject_endpoints_deterministic():
    assert infer(*fixture(), n_boot=40, seed=3) == infer(*fixture(), n_boot=40, seed=3)
