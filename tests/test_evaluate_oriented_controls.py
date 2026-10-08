import json

import numpy as np
import pytest

from scripts.benchmark_metrics_v2 import metric_vector
from scripts.evaluate_oriented_controls import paired_infer, readiness


def completed_campaign(tmp_path, monkeypatch, defect=None):
    from age_gap.common.manifest import file_record
    from scripts import evaluate_oriented_controls as scorer
    from scripts.run_oriented_campaign import PROTOCOL

    monkeypatch.setattr(scorer, "PROJECT_ROOT", tmp_path)
    models, campaign = tmp_path / "models", tmp_path / "campaign"
    models.mkdir()
    campaign.mkdir()
    arms = tmp_path / "metrics/oriented_exposure_matching_20261003/private"
    arms.mkdir(parents=True)
    pairs = {}
    for arm in ("low", "cross"):
        pairs[arm] = arms / f"{arm}_candidate_arm.jsonl"
        pairs[arm].write_text("{}\n", encoding="utf-8")
    common = [file_record(p) for p in pairs.values()]
    outputs = []
    for arm in ("low", "cross"):
        for seed in (42, 1, 2):
            checkpoint = models / f"{arm}_s{seed}.pt"
            checkpoint.write_bytes(b"synthetic; never load this model")
            train = checkpoint.with_suffix(".manifest.json")
            actual = dict(
                experiment="pair-contrastive-backbone-finetune",
                parameters=PROTOCOL | {"seed": seed, "pairs_file": str(pairs[arm])},
                inputs=[file_record(pairs[arm])],
                outputs=[file_record(checkpoint)],
            )
            first = arm == "low" and seed == 42
            if first and defect == "actual_checkpoint":
                actual["outputs"] = [file_record(pairs[arm])]
            train.write_text(json.dumps(actual), encoding="utf-8")
            cell = campaign / f"{arm}_s{seed}.manifest.json"
            bound = dict(
                experiment="oriented-source-crop-bound-training-cell",
                parameters=PROTOCOL | {"seed": seed, "arm": arm},
                metrics={"training_complete": True},
                inputs=common,
                outputs=[file_record(checkpoint), file_record(train)],
            )
            if first and defect == "cell_manifest":
                bound["outputs"] = [file_record(checkpoint)]
            if first and defect == "cell_inputs":
                bound["inputs"] = [file_record(pairs[arm])]
            if first and defect == "cell_protocol":
                bound["parameters"]["batchnorm_policy"] = "adapt_all"
            cell.write_text(json.dumps(bound), encoding="utf-8")
            outputs.extend(file_record(p) for p in (checkpoint, train, cell))
    parent = dict(
        experiment="oriented-uniform-serial-training",
        parameters=PROTOCOL.copy(),
        metrics={"training_complete": True, "completed_cells": 6},
        inputs=common,
        outputs=outputs,
    )
    if defect == "parent_protocol":
        parent["parameters"]["batchnorm_policy"] = "adapt_all"
    (campaign / "training-bound.manifest.json").write_text(json.dumps(parent), encoding="utf-8")
    return campaign, models


def test_completed_campaign_links_six_cells(tmp_path, monkeypatch):
    campaign, models = completed_campaign(tmp_path, monkeypatch)
    native, checkpoints = readiness(campaign, models)
    assert native["metrics"]["completed_cells"] == len(checkpoints) == 6


@pytest.mark.parametrize(
    "defect", ["actual_checkpoint", "cell_manifest", "cell_inputs", "cell_protocol", "parent_protocol"]
)
def test_checksum_valid_but_disconnected_campaign_refused(tmp_path, monkeypatch, defect):
    campaign, models = completed_campaign(tmp_path, monkeypatch, defect)
    with pytest.raises(ValueError):
        readiness(campaign, models)


def example():
    y = np.array([1, 1, 0, 0])
    a, b = np.array([1, 2, 1, 2]), np.array([1, 2, 2, 1])
    high, low = np.array([0.9, 0.8, 0.1, 0.2]), np.array([0.1, 0.2, 0.9, 0.8])
    return np.stack([high, low, high]), np.stack([high] * 3), y, a, b


def test_mean_checkpoint_not_score_ensemble():
    low, cross, y, a, b = example()
    result = paired_infer(low, cross, y, a, b, n_boot=30)
    auc = result["metrics"]["roc_auc"]
    assert auc["low"]["mean"] == pytest.approx(2 / 3)
    assert metric_vector(low.mean(axis=0), y)[0] == 1.0
    assert auc["cross_minus_low"]["mean"] == pytest.approx(1 / 3)
    assert result["bootstrap"]["shared_draws_across_all_models"]
    assert not result["causal_source_evidence"]


def test_frozen_absolute_and_pair_deltas():
    low, cross, y, a, b = example()
    result = paired_infer(low, cross, y, a, b, frozen=cross[0], n_boot=30)
    row = result["metrics"]["roc_auc"]
    assert row["frozen"]["point"] == 1.0
    assert row["low"]["mean_delta_vs_frozen"] == pytest.approx(-1 / 3)
    assert row["cross"]["delta_vs_frozen_ci95"] == [0.0, 0.0]
    assert set(result["metrics"]) == {
        "roc_auc",
        "eer_interpolated",
        "eer_discrete_minimax",
        "tar@far=0.01",
        "tar@far=0.001",
    }


def test_identical_arms_zero_delta_and_seed_reproducibility():
    low, _, y, a, b = example()
    first = paired_infer(low, low, y, a, b, n_boot=30, seed=2)
    assert first == paired_infer(low, low, y, a, b, n_boot=30, seed=2)
    for row in first["metrics"].values():
        assert row["cross_minus_low"]["mean_checkpoint_ci95"] == [0.0, 0.0]
    assert first["loo_valid"] == 0
    assert first["loo_unavailable"] == 2


def test_loo_removes_both_subject_endpoints():
    y = np.array([1, 1, 1, 0, 0, 0])
    a, b = np.array([1, 2, 3, 1, 2, 3]), np.array([1, 2, 3, 2, 3, 1])
    scores = np.tile([0.9, 0.8, 0.7, 0.1, 0.2, 0.3], (3, 1))
    result = paired_infer(scores, scores, y, a, b, n_boot=20)
    assert result["loo_valid"] == 3
    assert result["metrics"]["roc_auc"]["cross_minus_low"]["loo_min"] == 0.0


@pytest.mark.parametrize(
    "change",
    [
        "boolean_labels",
        "unknown_subject",
        "contradictory_subject",
        "nonfinite",
        "wrong_shape",
        "missing_frozen",
    ],
)
def test_reject_malformed_data(change):
    low, cross, y, a, b = example()
    kwargs = {}
    if change == "boolean_labels":
        y = y.astype(bool)
    elif change == "unknown_subject":
        a = np.array(["unknown", "2", "1", "2"])
    elif change == "contradictory_subject":
        b[0] = 2
    elif change == "nonfinite":
        low[0, 0] = np.inf
    elif change == "wrong_shape":
        low = low[:2]
    else:
        kwargs["frozen"] = np.zeros(3)
    with pytest.raises(ValueError):
        paired_infer(low, cross, y, a, b, n_boot=2, **kwargs)


def test_incomplete_campaign_refused_before_scoring(tmp_path):
    with pytest.raises(FileNotFoundError):
        readiness(tmp_path / "campaign", tmp_path / "models")
