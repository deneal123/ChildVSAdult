import numpy as np
import pytest

from scripts.internal_exact_age_sensitivity import compare_subsets, select_blocks


def rows():
    common = {"split": "test", "age_a": 1, "identity_group_a": "a"}
    return [dict(common, pair_id="p", label=1, face_a="child", face_b="adult", age_b=30,
                 age_gap=29, identity_group_b="a"),
            dict(common, pair_id="n", label=0, face_a="child", face_b="other", age_b=30,
                 age_gap=29, identity_group_b="b")]


def test_exact_selection_uses_no_scores():
    mask, a, b = select_blocks(rows(), {"a": "person1", "b": "person2"})
    assert mask.tolist() == [True]
    assert a.tolist() == ["person1"]
    assert b.tolist() == ["person2"]


def test_zero_exact_selection_is_not_completed():
    sample = rows()
    sample[1].update(age_b=31, age_gap=30)
    with pytest.raises(ValueError, match="no exact"):
        select_blocks(sample, {"a": "person1", "b": "person2"})


@pytest.mark.parametrize("mapping", [{}, {"a": "same", "b": "same"}, {"a": "", "b": "person2"}])
def test_unverified_or_contradictory_person_mapping(mapping):
    with pytest.raises(ValueError):
        select_blocks(rows(), mapping)


@pytest.mark.parametrize("field,value", [("split", "train"), ("age_gap", 28), ("face_a", "wrong")])
def test_bad_pair_binding(field, value):
    sample = rows()
    sample[1][field] = value
    with pytest.raises(ValueError):
        select_blocks(sample, {"a": "person1", "b": "person2"})


def test_tied_models_have_zero_gain_ci_in_both_subsets():
    scores = [.7, .8, .9, .2, .3, .4]
    result = compare_subsets(scores, scores, [True, False, True], ["a", "a", "b"], ["b", "c", "c"], n_boot=100)
    for name in ("original_tolerance", "exact_age_subset"):
        assert result[name]["gain_auc"] == 0
        assert result[name]["ci95_frozen_tuned_gain"][2] == [0, 0]
    assert result["exact_minus_original_gain"] == {"point": 0, "ci95": [0, 0]}
    assert result["bootstrap"]["valid"] <= 100


def test_joint_subset_comparison_reproducible_and_not_ensemble():
    args = ([.9, .4, .3, .2, .5, .6], [.9, .9, .8, .2, .3, .4],
            [True, False, True], ["a", "a", "b"], ["b", "c", "c"])
    assert compare_subsets(*args, n_boot=50) == compare_subsets(*args, n_boot=50)
    result = compare_subsets(*args, n_boot=50)
    assert result["exact_minus_original_gain"]["point"] == pytest.approx(
        result["exact_age_subset"]["gain_auc"] - result["original_tolerance"]["gain_auc"])


def test_tied_negative_scores_are_not_split_to_hit_low_far():
    result = compare_subsets([.8, .8, .8, .8], [.8, .8, .8, .8], [True, True],
                             ["a", "b"], ["b", "a"], n_boot=50)
    assert result["original_tolerance"]["empirical_tar_at_far"]["frozen"] == {"0.01": 0, "0.001": 0}


@pytest.mark.parametrize("exact", [[1, 0], [False, False]])
def test_bad_exact_masks(exact):
    with pytest.raises(ValueError):
        compare_subsets([.8, .8, .2, .2], [.8, .8, .2, .2], exact, ["a", "b"], ["b", "a"])


def test_nonfinite_scores_rejected():
    with pytest.raises(ValueError):
        compare_subsets([np.nan, .8, .2, .2], [.8, .8, .2, .2], [True, True], ["a", "b"], ["b", "a"])
