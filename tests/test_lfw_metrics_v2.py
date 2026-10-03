import copy

import numpy as np
import pytest

from scripts.audit_lfw_metrics_v2 import OLD_TO_NEW, aligned_endpoints, compare
from scripts.benchmark_metrics_v2 import KEYS, infer
from scripts.evaluate_lfw_bound import roc_points
from scripts.verification_metrics_v2 import empirical_metrics


def test_alignment_rejects_person_rows_in_different_order():
    rows = [{"index": 0, "label": 1, "fold": 0, "subject_a": "a", "subject_b": "a"}]
    assert aligned_endpoints(rows, [1], [0]) == (["a"], ["a"])
    rows[0]["index"] = 1
    with pytest.raises(ValueError):
        aligned_endpoints(rows, [1], [0])
    with pytest.raises(ValueError):
        aligned_endpoints([], [1], [0])


def old_format(updated):
    old = {"models": {}, "seed_aggregate": {}}
    for key in KEYS:
        old["models"][key] = {"metrics": {}, "subject_ci95": {}, "paired_gain": {}}
        for old_metric, metric in OLD_TO_NEW.items():
            row = updated["models"][key][metric]
            old["models"][key]["metrics"][old_metric] = row["point"]
            old["models"][key]["subject_ci95"][old_metric] = row["ci95"]
            old["models"][key]["paired_gain"][old_metric] = {
                "delta": row["delta_vs_frozen"], "subject_ci95": row["delta_ci95"]}
    for old_metric, metric in OLD_TO_NEW.items():
        row = updated["three_checkpoint_aggregate"][metric]
        old["seed_aggregate"][old_metric] = {"mean": row["mean"], "std": row["sd"],
            "delta_mean_vs_frozen": row["mean_checkpoint_delta"],
            "fixed_checkpoint_mean_gain_subject_ci95": row["delta_ci95"]}
    return old


def test_compatibility_checks_paired_ci_not_only_points():
    scores = {key: np.array([.8, .4, .5, .1]) for key in KEYS}
    updated = infer(scores, [1, 1, 0, 0], ["a", "b", "a", "b"], ["a", "b", "b", "c"], n_boot=10)
    previous = old_format(updated)
    assert compare(previous, updated)["matches_within_1e_minus_12"]
    changed = copy.deepcopy(previous)
    changed["models"]["tuned_seed42"]["paired_gain"]["tar@far=0.001"]["subject_ci95"][0] += .02
    assert not compare(changed, updated)["matches_within_1e_minus_12"]


def test_legacy_eer_maps_to_minimax_not_interpolated():
    old = roc_points([.9, .9], [1, 0])
    new = empirical_metrics([.9, .9], [1, 0])
    assert old["eer"] == new["eer_discrete_minimax"] == 1
    assert new["eer_interpolated"] == .5


def test_far_epsilon_precludes_universal_equivalence_claim():
    scores, labels = [.9, .9, .1], [1, 0, 0]
    weights = [1, .0100000000005, .9899999999995]
    old = roc_points(scores, labels, weights)
    new = empirical_metrics(scores, labels, weights=weights)
    assert old["tar@far=0.01"] == 1
    assert new["operating_points"]["0.01"]["tar"] == 0
