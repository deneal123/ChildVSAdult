import numpy as np
import pytest

from scripts.benchmark_metrics_v2 import KEYS
from scripts.cacd_metrics_v2 import ALL_METRICS, infer
from scripts.reevaluate_cacd_metrics_v2 import canonical, checkpoint_records


def test_exact_canonical_order_and_integer_labels():
    y = (np.arange(4000) % 400 < 200).astype(int)
    f = np.arange(4000) // 400
    canonical(y, f)
    for yy, ff in ((y.astype(bool), f), (y[::-1], f), (y, np.arange(4000) % 10)):
        with pytest.raises(ValueError):
            canonical(yy, ff)


def test_shared_pair_draws_zero_delta_and_explicit_scope():
    scores = {key: [.9, .2, .8, .1] for key in KEYS}
    result = infer(scores, [1, 0, 1, 0], [0, 0, 1, 1], n_boot=10)
    for metric in ALL_METRICS:
        assert result["three_checkpoint_aggregate"][metric]["mean_checkpoint_delta"] == 0
        assert result["three_checkpoint_aggregate"][metric]["paired_pair_delta_ci95"] == [0, 0]
    assert result["bootstrap"]["subject_metadata_available"] is False
    assert "pair" in result["bootstrap"]["sampling_unit"]
    assert result["accuracy"]["legacy_grid_equivalence_claimed"] is False


@pytest.mark.parametrize("labels,folds", [([1., 0., 1., 0.], [0, 0, 1, 1]),
    ([1, 0, 1, 0], [0, 0, 2, 2]), ([1, 1, 0, 0], [0, 0, 1, 1])])
def test_invalid_or_single_class_fold_rejected(labels, folds):
    with pytest.raises(ValueError):
        infer({key: [.9, .2, .8, .1] for key in KEYS}, labels, folds, n_boot=2)


@pytest.mark.parametrize("bad_scores", [[.9, .2], [.9, np.nan, .8, .1], [1.2, .2, .8, .1]])
def test_invalid_score_vectors_rejected(bad_scores):
    scores = {key: [.9, .2, .8, .1] for key in KEYS}
    scores["tuned_seed1"] = bad_scores
    with pytest.raises(ValueError):
        infer(scores, [1, 0, 1, 0], [0, 0, 1, 1], n_boot=2)


def test_all_tied_scores_keep_distinct_eer_definitions_and_zero_low_far_tar():
    result = infer({key: [.5] * 4 for key in KEYS}, [1, 0, 1, 0], [0, 0, 1, 1], n_boot=2)
    frozen = result["models"]["frozen"]
    assert frozen["roc_auc"]["point"] == .5
    assert frozen["eer_interpolated"]["point"] == .5
    assert frozen["eer_discrete_minimax"]["point"] == 1
    assert frozen["tar@far=0.01"]["point"] == 0
    assert frozen["tar@far=0.001"]["point"] == 0
    assert frozen["accuracy_leave_one_fold_out_fixed_grid"]["point"] == .5


def test_checkpoint_roles_distinctness_and_missing_files(tmp_path):
    weights = {}
    for index, key in enumerate(KEYS):
        weights[key] = tmp_path / f"{key}.pt"
        weights[key].write_bytes(bytes([index]))
    assert set(checkpoint_records(weights)) == set(KEYS)
    with pytest.raises(ValueError):
        checkpoint_records({"frozen": weights["frozen"]})
    duplicate = {**weights, "tuned_seed2": weights["tuned_seed1"]}
    with pytest.raises(ValueError, match="distinct"):
        checkpoint_records(duplicate)
    weights["tuned_seed2"].unlink()
    with pytest.raises(FileNotFoundError):
        checkpoint_records(weights)
