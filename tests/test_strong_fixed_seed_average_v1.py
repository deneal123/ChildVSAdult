"""Synthetic, CPU-only tests for strong_fixed_seed_average_v1.

No native data, checkpoint, weight, cache, GPU, network or long experiment is
touched. Every fixture is generated in-memory, and the canonical ROC-v2 metric
helper (``scripts.benchmark_metrics_v2.metric_vector``) is imported read-only.

Run from the repository root:
    .venv/Scripts/python.exe -m pytest \
        tests/test_strong_fixed_seed_average_v1.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.benchmark_metrics_v2 import METRICS, metric_vector
from scripts.strong_fixed_seed_average_v1 import FIXED_SEEDS, infer

N_BOOT = 200


def fixture():
    """Six pairs over three subjects with an exact frozen/tuned score array."""
    y = np.array([1, 1, 1, 0, 0, 0])
    a = np.array(["a", "b", "c", "a", "b", "c"])
    b = np.array(["a", "b", "c", "b", "c", "a"])
    scores = {
        "frozen": np.array([0.8, 0.5, 0.6, 0.7, 0.2, 0.4]),
        42: np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.4]),
        1: np.array([0.85, 0.75, 0.65, 0.62, 0.25, 0.45]),
        2: np.array([0.7, 0.6, 0.55, 0.66, 0.35, 0.5]),
    }
    return scores, y, a, b


def point_metric(scores, y, key):
    return metric_vector(np.asarray(scores[key], float), y)


def manual_draw_metrics(matrix, y, weights):
    """Independent tiny oracle: canonical metric_vector per checkpoint."""
    return np.stack([metric_vector(matrix[i], y, weights) for i in range(matrix.shape[0])])


def test_identical_checkpoints_reduce_to_single_with_zero_delta():
    scores, y, a, b = fixture()
    for seed in FIXED_SEEDS:
        scores[seed] = scores["frozen"].copy()
    result = infer(scores, y, a, b, n_boot=N_BOOT)
    for name in METRICS:
        aggregate = result["aggregate"][name]
        assert aggregate["mean_checkpoint_delta"] == 0
        assert aggregate["delta_ci95"] == pytest.approx([0, 0], abs=1e-12)
        assert aggregate["sd"] == pytest.approx(0, abs=1e-12)
        assert aggregate["loo_delta_min"] == pytest.approx(0, abs=1e-12)
        assert aggregate["loo_delta_max"] == pytest.approx(0, abs=1e-12)
    for seed in FIXED_SEEDS:
        for metric in result["per_seed"][str(seed)].values():
            assert metric["delta_vs_frozen"] == 0
            assert metric["delta_ci95"] == pytest.approx([0, 0], abs=1e-12)


def test_score_ensemble_differs_from_metric_mean_of_individuals():
    scores, y, a, b = fixture()
    matrix = np.stack([np.asarray(scores[key], float) for key in ("frozen", *FIXED_SEEDS)])
    result = infer(scores, y, a, b, n_boot=N_BOOT)
    for k, name in enumerate(METRICS):
        individual_mean = np.mean([point_metric(scores, y, key)[k] for key in FIXED_SEEDS])
        ensemble_metric = metric_vector(matrix[1:].mean(axis=0), y)[k]
        assert result["aggregate"][name]["mean"] == pytest.approx(individual_mean)
        assert result["aggregate"][name]["mean"] != ensemble_metric


def test_metric_mean_and_ddof1_sd_formulas():
    scores, y, a, b = fixture()
    result = infer(scores, y, a, b, n_boot=N_BOOT)
    tuned = np.stack([point_metric(scores, y, key) for key in FIXED_SEEDS])
    for k, name in enumerate(METRICS):
        assert result["aggregate"][name]["mean"] == pytest.approx(tuned[:, k].mean())
        assert result["aggregate"][name]["sd"] == pytest.approx(tuned[:, k].std(ddof=1))
        frozen_point = point_metric(scores, y, "frozen")[k]
        assert result["aggregate"][name]["mean_checkpoint_delta"] == pytest.approx(
            tuned[:, k].mean() - frozen_point
        )
        for position, seed in enumerate(FIXED_SEEDS):
            assert result["per_seed"][str(seed)][name]["point"] == pytest.approx(tuned[position, k])
            assert result["per_seed"][str(seed)][name]["delta_vs_frozen"] == pytest.approx(
                tuned[position, k] - frozen_point
            )


def test_paired_shared_bootstrap_replays_manual_oracle():
    scores, y, a, b = fixture()
    matrix = np.stack([np.asarray(scores[key], float) for key in ("frozen", *FIXED_SEEDS)])
    n_boot, seed = 137, 0
    result = infer(scores, y, a, b, n_boot=n_boot, seed=seed)

    people, inverse = np.unique(np.concatenate([a.astype(str), b.astype(str)]), return_inverse=True)
    aa, bb = inverse[: len(y)], inverse[len(y) :]
    rng = np.random.default_rng(seed)
    mean_draws, frozen_draws = [], []
    for _ in range(n_boot):
        multiplicity = np.bincount(rng.integers(len(people), size=len(people)), minlength=len(people))
        weights = multiplicity[aa] * np.where(y == 1, 1.0, multiplicity[bb])
        if weights[y == 0].sum() <= 0 or weights[y == 1].sum() <= 0:
            continue
        draw = manual_draw_metrics(matrix, y, weights)
        mean_draws.append(draw[1:].mean(axis=0))
        frozen_draws.append(draw[0])
    mean_draws, frozen_draws = np.stack(mean_draws), np.stack(frozen_draws)

    assert result["bootstrap"]["valid"] == len(mean_draws)
    for k, name in enumerate(METRICS):
        aggregate = result["aggregate"][name]
        assert aggregate["mean_ci95"] == pytest.approx(
            np.percentile(mean_draws[:, k], [2.5, 97.5]).tolist()
        )
        # The delta CI averages three per-draw metrics and then subtracts frozen;
        # it is not the mean of precomputed interval endpoints.
        manual_delta = np.percentile(mean_draws[:, k] - frozen_draws[:, k], [2.5, 97.5]).tolist()
        assert aggregate["delta_ci95"] == pytest.approx(manual_delta)


def test_loo_removes_all_incident_pairs_and_counts():
    y = np.array([1, 1, 0, 0])
    a = np.array(["a", "b", "a", "b"])
    b = np.array(["a", "b", "c", "c"])
    scores = {
        "frozen": np.array([0.8, 0.5, 0.7, 0.2]),
        42: np.array([0.9, 0.8, 0.6, 0.3]),
        1: np.array([0.85, 0.75, 0.62, 0.25]),
        2: np.array([0.7, 0.6, 0.66, 0.35]),
    }
    result = infer(scores, y, a, b, n_boot=50)
    # Removing subject a or b leaves one positive and one negative pair; removing c
    # leaves only positives, so that subject is unavailable.
    assert result["loo_valid"] == 2
    assert result["loo_unavailable"] == 1
    for name in METRICS:
        assert result["aggregate"][name]["loo_delta_min"] is not None
        assert (
            result["aggregate"][name]["loo_delta_min"]
            <= result["aggregate"][name]["loo_delta_max"]
        )


@pytest.mark.parametrize(
    "change",
    ["label_mismatch", "genuine_same", "unknown_token", "nonfinite", "wrong_shape", "string_scores"],
)
def test_invalid_scores_metadata_and_shapes_rejected(change):
    scores, y, a, b = fixture()
    if change == "label_mismatch":
        b = b.copy()
        b[0] = "c"  # genuine endpoints no longer match
    elif change == "genuine_same":
        a = a.copy()
        b = b.copy()
        b[3] = a[3]  # impostor endpoints now share a subject
    elif change == "unknown_token":
        a = a.astype("U7")
        a[0] = "unknown"
    elif change == "nonfinite":
        scores[42] = scores[42].copy()
        scores[42][0] = np.nan
    elif change == "wrong_shape":
        scores[1] = scores[1][:-1]
    elif change == "string_scores":
        scores[2] = np.array(["a", "b", "c", "d", "e", "f"])
    with pytest.raises(ValueError):
        infer(scores, y, a, b, n_boot=20)


@pytest.mark.parametrize(
    "scores",
    [
        {42: [0.1], 1: [0.2], 2: [0.3]},
        {"frozen": [0.1], 1: [0.2], 2: [0.3], "43": [0.4]},
        {"frozen": [0.1], 42: [0.2], 1: [0.3]},
        {"frozen": [0.1], 42: [0.2], 1: [0.3], 2: [0.4], 7: [0.5]},
    ],
)
def test_seed_mapping_must_be_exact(scores):
    y = np.array([1, 0])
    a = np.array(["a", "b"])
    b = np.array(["a", "c"])
    with pytest.raises(ValueError):
        infer(scores, y, a, b, n_boot=10)


@pytest.mark.parametrize("bad", [True, np.True_, "2"])
def test_bool_and_string_seeds_rejected(bad):
    scores, y, a, b = fixture()
    del scores[2]
    scores[bad] = np.array([0.7, 0.6, 0.55, 0.66, 0.35, 0.5])
    with pytest.raises(ValueError):
        infer(scores, y, a, b, n_boot=10)


@pytest.mark.parametrize("kwargs", [{"n_boot": True}, {"n_boot": 0}, {"n_boot": 10, "seed": True}, {"seed": -1}])
def test_bool_and_negative_counts_rejected(kwargs):
    scores, y, a, b = fixture()
    with pytest.raises(ValueError):
        infer(scores, y, a, b, **kwargs)


def test_single_class_and_single_subject_rejected():
    scores = {
        "frozen": [0.8, 0.5],
        42: [0.9, 0.8],
        1: [0.85, 0.75],
        2: [0.7, 0.6],
    }
    with pytest.raises(ValueError):
        infer(scores, [1, 1], ["a", "b"], ["a", "b"], n_boot=10)
    with pytest.raises(ValueError):
        infer(scores, [1, 0], ["a", "a"], ["a", "a"], n_boot=10)


def test_deterministic_and_order_independent():
    scores, y, a, b = fixture()
    first = infer(scores, y, a, b, n_boot=80, seed=3)
    second = infer(scores, y, a, b, n_boot=80, seed=3)
    assert first == second

    reordered = {2: scores[2], "frozen": scores["frozen"], 1: scores[1], 42: scores[42]}
    third = infer(reordered, y, a, b, n_boot=80, seed=3)
    assert third["aggregate"] == first["aggregate"]
    assert third["per_seed"] == first["per_seed"]
    assert third["bootstrap"] == first["bootstrap"]


def test_machine_readable_scope_and_metadata():
    scores, y, a, b = fixture()
    result = infer(scores, y, a, b, n_boot=30)
    assert result["fixed_seed_list"] == [42, 1, 2]
    assert result["metric_version"] == "empirical-roc-v2"
    assert result["n_pairs"] == len(y)
    assert result["n_subjects"] == 3
    assert result["publication_ready"] is False
    assert result["training_identity_independence"] == "unverified"
    assert "not a training-seed population" in result["scope"]
    assert "no ensemble" in result["scope"]
    assert result["bootstrap"]["shared_weights_across_models"] is True
    assert result["bootstrap"]["requested"] == 30
    assert result["bootstrap"]["valid"] >= 1
    assert result["bootstrap"]["valid"] + result["bootstrap"]["invalid"] == 30


@pytest.mark.parametrize("bad", [1.0, np.int64(1), np.bool_(True)])
def test_numeric_equal_but_not_builtin_integer_seed_is_rejected(bad):
    scores, y, a, b = fixture()
    old = scores.pop(1)
    scores[bad] = old
    with pytest.raises(ValueError):
        infer(scores, y, a, b, n_boot=10)


def test_all_result_keys_survive_json_roundtrip():
    result = infer(*fixture(), n_boot=20)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert set(result["per_seed"]) == {"42", "1", "2"}


def test_identical_metric_draws_have_exact_zero_not_roundoff_ci():
    scores, y, a, b = fixture()
    for seed in FIXED_SEEDS:
        scores[seed] = scores["frozen"].copy()
    result = infer(scores, y, a, b, n_boot=137)
    for row in result["aggregate"].values():
        assert row["mean_checkpoint_delta"] == 0.0
        assert row["delta_ci95"] == [0.0, 0.0]
        assert row["loo_delta_min"] == row["loo_delta_max"] == 0.0
        assert row["sd"] == 0.0

