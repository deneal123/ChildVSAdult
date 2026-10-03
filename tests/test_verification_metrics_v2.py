import numpy as np
import pytest

from scripts.verification_metrics_v2 import eer_from_roc, empirical_metrics


def brute_tar(scores, labels, target, weights=None):
    s, y = np.asarray(scores), np.asarray(labels)
    w = np.ones(len(s)) if weights is None else np.asarray(weights)
    best = 0
    for threshold in [np.inf, *np.unique(s)]:
        accepted = s >= threshold
        far = w[(y == 0) & accepted].sum() / w[y == 0].sum()
        tar = w[(y == 1) & accepted].sum() / w[y == 1].sum()
        if far <= target:
            best = max(best, tar)
    return best


@pytest.mark.parametrize("target", [0, .001, .01, .25, .5, 1])
def test_exact_threshold_enumeration_with_ties_and_weights(target):
    s, y, w = [.9, .8, .8, .7, .7, .4], [1, 0, 1, 0, 1, 0], [1, 2, 3, 0, 4, 5]
    point = empirical_metrics(s, y, weights=w, far_targets=[target])["operating_points"][f"{target:g}"]
    assert point["tar"] == pytest.approx(brute_tar(s, y, target, w))
    assert point["far_achieved"] <= target


def test_maximizes_tar_between_last_accepted_and_first_rejected_impostor():
    scores = np.r_[np.full(200, .75), [.9, .8, .7], np.zeros(197)]
    labels = np.r_[np.ones(200, int), np.zeros(200, int)]
    point = empirical_metrics(scores, labels)["operating_points"]["0.01"]
    assert point["tar"] == 1
    assert point["far_achieved"] == .01


def test_tied_impostors_cannot_be_split_to_attain_far():
    scores = np.r_[np.full(200, .95), [.9, .9, .9], np.zeros(197)]
    labels = np.r_[np.ones(200, int), np.zeros(200, int)]
    point = empirical_metrics(scores, labels)["operating_points"]["0.01"]
    assert point["tar"] == 1
    assert point["far_achieved"] == 0


def test_all_ties_separate_interpolated_and_discrete_eer():
    result = empirical_metrics([.5, .5], [0, 1], far_targets=[0, .5, 1])
    assert result["roc_auc"] == .5
    assert result["eer_interpolated"] == .5
    assert result["eer_discrete_minimax"] == 1
    assert result["operating_points"]["0.5"]["tar"] == 0
    assert result["operating_points"]["0.5"]["reject_all"]


def test_perfect_and_reversed_eer():
    assert empirical_metrics([.9, .1], [1, 0])["eer_interpolated"] == 0
    assert empirical_metrics([.1, .9], [1, 0])["eer_interpolated"] == 1


@pytest.mark.parametrize("scores,labels,weights", [([], [], None), ([1, 2], [True, False], None),
    ([1, 2], [0., 1.], None), ([np.inf, 1], [0, 1], None), ([1, 2], [0, 0], None),
    ([1, 2], [0, 1], [-1, 1]), ([1, 2], [0, 1], [0, 1]), ([1, 2], [0, 1], [np.nan, 1])])
def test_invalid_inputs(scores, labels, weights):
    with pytest.raises(ValueError):
        empirical_metrics(scores, labels, weights=weights)


@pytest.mark.parametrize("targets", [[True], [-.1], [1.1], [np.nan], [], [.01, .01]])
def test_bad_targets(targets):
    with pytest.raises(ValueError):
        empirical_metrics([.1, .9], [0, 1], far_targets=targets)


def test_bad_roc_rejected():
    with pytest.raises(ValueError):
        eer_from_roc([0, .5, 1], [0, .8, .7])


def test_random_discrete_inputs_match_brute_threshold_enumeration():
    rng = np.random.default_rng(42)
    for _ in range(20):
        s = rng.integers(0, 4, size=20).astype(float)
        y = np.r_[np.zeros(10, int), np.ones(10, int)]
        for target in (0, .1, .3):
            point = empirical_metrics(s, y, far_targets=[target])["operating_points"][f"{target:g}"]
            assert point["tar"] == pytest.approx(brute_tar(s, y, target))
