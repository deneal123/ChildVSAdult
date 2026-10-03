import pytest

from scripts.internal_exact_age_sensitivity import compare_subsets
from scripts.reevaluate_internal_metrics_v2 import infer


def sample():
    return ([.9, .4, .3, .2, .5, .6], [.9, .9, .8, .2, .3, .4],
            [True, False, True], ["a", "a", "b"], ["b", "c", "c"])


def test_source_bound_auc_draws_agree_with_previous_paired_sensitivity():
    args = sample()
    result, previous = infer(*args, n_boot=40), compare_subsets(*args, n_boot=40)
    for name in ("original_tolerance", "exact_age_subset"):
        metric = result["subsets"][name]["metrics"]["roc_auc"]
        assert metric["tuned_minus_frozen"]["point"] == pytest.approx(previous[name]["gain_auc"])
        assert metric["tuned_minus_frozen"]["ci95"] == pytest.approx(previous[name]["ci95_frozen_tuned_gain"][2])
    assert result["bootstrap"]["threshold_reselection"]
    assert not result["deployment_calibration"]


def test_identical_models_have_zero_gain_for_every_metric():
    frozen, _, mask, owner, impostor = sample()
    result = infer(frozen, frozen, mask, owner, impostor, n_boot=40)
    for subset in result["subsets"].values():
        for metric in subset["metrics"].values():
            assert metric["tuned_minus_frozen"] == {"point": 0, "ci95": [0, 0]}
    for metric in result["exact_minus_original_model_delta"].values():
        assert metric == {"point": 0, "ci95": [0, 0]}


def test_whole_exact_mask_produces_zero_difference_in_model_delta():
    frozen, tuned, _, owner, impostor = sample()
    result = infer(frozen, tuned, [True, True, True], owner, impostor, n_boot=40)
    for metric in result["exact_minus_original_model_delta"].values():
        assert metric == {"point": 0, "ci95": [0, 0]}


@pytest.mark.parametrize("mask", [[1, 0, 1], [False, False, False]])
def test_invalid_mask(mask):
    frozen, tuned, _, owner, impostor = sample()
    with pytest.raises(ValueError):
        infer(frozen, tuned, mask, owner, impostor, n_boot=1)


def test_empty_person_id_rejected():
    frozen, tuned, mask, owner, impostor = sample()
    owner[0] = ""
    with pytest.raises(ValueError):
        infer(frozen, tuned, mask, owner, impostor, n_boot=1)


def test_small_far_ties_have_no_attainable_genuine_accepts():
    scores = [.5] * 6
    result = infer(scores, scores, [True, False, True], ["a", "a", "b"], ["b", "c", "c"], n_boot=40)
    metrics = result["subsets"]["original_tolerance"]["metrics"]
    assert metrics["eer_interpolated"]["frozen"] == {"point": .5, "ci95": [.5, .5]}
    assert metrics["eer_discrete_minimax"]["frozen"] == {"point": 1, "ci95": [1, 1]}
    assert metrics["tar@far=0.01"]["frozen"] == {"point": 0, "ci95": [0, 0]}
