"""Synthetic presentation checks, not actual experiment/hash verification."""

import copy

import pytest

from scripts import render_strong_cuda_fgnet_v1 as render


def native(name="random_tail_lr1e-06_s2"):
    value = dict(point=.8, ci95=[.7, .9], delta_vs_frozen=0., delta_ci95=[0., 0.])
    frozen = {key: copy.deepcopy(value) for key, _ in render.METRICS}
    tuned = copy.deepcopy(frozen)
    for v in tuned.values():
        v.update(point=.82, delta_vs_frozen=.02, delta_ci95=[-.01, .05])
    return dict(experiment=render.EXPERIMENT,
                parameters=dict(device="cuda", protocol_seed=42, tolerance=2,
                                n_boot=2000, bootstrap_seed=0, actual_cells=[name]),
                metrics=dict(execution_complete=True, publication_ready=False,
                             overall=dict(metric_version="empirical-roc-v2", n_pairs=5308,
                                          n_subjects=82, actual_tuned_checkpoint_count=1,
                                          models={"frozen": frozen, name: tuned}),
                             large_gap_25plus=dict(metric_version="empirical-roc-v2", n_pairs=430,
                                                  n_subjects=75, actual_tuned_checkpoint_count=1,
                                                  models={"frozen": frozen, name: tuned})))


def test_per_checkpoint_output_retains_limits_and_operating_points():
    payload = render.collect([native()])
    assert payload["evaluated_checkpoint_count"] == 1
    assert payload["expected_matrix_cells"] == 36
    assert payload["mechanism_complete"] is False
    assert payload["publication_ready"] is False
    text = render.table(payload, "large_gap_25plus")
    assert "TAR@FAR=0.1" in text and "Interpolated EER" in text
    assert "not a seed mean" in text and "No training-seed population inference" in text
    assert "[-0.0100,+0.0500]" in text
    assert "25+ years" in text


@pytest.mark.parametrize("change", ["incomplete", "different_protocol", "missing_coverage",
                                   "nonfinite", "backwards_ci", "invalid_delta", "extra_model"])
def test_rejects_incompatible_or_malformed_evidence(change):
    data = native()
    part = data["metrics"]["overall"]
    value = part["models"]["random_tail_lr1e-06_s2"]["roc_auc"]
    if change == "incomplete":
        data["metrics"]["execution_complete"] = False
    elif change == "different_protocol":
        data["parameters"]["tolerance"] = 3
    elif change == "missing_coverage":
        part["n_subjects"] = 81
    elif change == "nonfinite":
        value["point"] = float("nan")
    elif change == "backwards_ci":
        value["delta_ci95"] = [.05, -.01]
    elif change == "invalid_delta":
        value["delta_vs_frozen"] = .3
    else:
        part["models"]["not_bound"] = part["models"]["frozen"]
    with pytest.raises(ValueError):
        render.collect([data])


def test_duplicate_cell_and_empty_collection_fail_closed():
    with pytest.raises(ValueError, match="unique"):
        render.collect([native(), native()])
    with pytest.raises(ValueError, match="at least one"):
        render.collect([])


def test_different_shared_frozen_reference_rejected():
    second = native("random_tail_lr1e-06_s1")
    second["metrics"]["overall"]["models"]["frozen"]["roc_auc"]["ci95"] = [.6, .9]
    with pytest.raises(ValueError, match="frozen reference"):
        render.collect([native(), second])
