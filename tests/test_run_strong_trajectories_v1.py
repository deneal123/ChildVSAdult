"""Synthetic runner-contract tests; not native training or manifest verification."""

import sys

import pytest

from scripts import run_strong_trajectories_v1 as runner


def native():
    return dict(parameters=dict(device="cuda", checkpoint_selection="last_epoch",
                                selected_epoch=8, batchnorm_policy="frozen_all",
                                negative="random", trainable_scope="head",
                                learning_rate=1e-6, seed=42),
                metrics=dict(training_complete=True, history=[
                    dict(epoch=i, training_loss=.2, validation_auc=.9,
                         mean_gradient_norm=.3, epoch_seconds=1.) for i in range(1, 9)]))


def setup(monkeypatch, tmp_path, values):
    bindings = []
    for i, _value in enumerate(values):
        path = tmp_path / f"binding{i}.json"
        path.write_text("{}", encoding="utf-8")
        bindings.append(path)
    mapping = dict(zip(bindings, values, strict=True))
    monkeypatch.setattr(runner, "verified", lambda path, experiment: mapping[path])
    monkeypatch.setattr(runner, "paths_from", lambda manifest: [])
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["runner", "--bindings", *map(str, bindings), "--out", str(out)])
    return out


@pytest.mark.parametrize("key,value", [("device", "cpu"), ("selected_epoch", 7),
                         ("checkpoint_selection", "best_validation"),
                         ("batchnorm_policy", "adapt_all")])
def test_rejects_incompatible_contract_without_output(monkeypatch, tmp_path, key, value):
    data = native()
    data["parameters"][key] = value
    out = setup(monkeypatch, tmp_path, [data])
    with pytest.raises(ValueError, match="contract"):
        runner.main()
    assert not out.exists()


def test_rejects_incomplete_training(monkeypatch, tmp_path):
    data = native()
    data["metrics"]["training_complete"] = False
    out = setup(monkeypatch, tmp_path, [data])
    with pytest.raises(ValueError, match="contract"):
        runner.main()
    assert not out.exists()


def test_rejects_duplicate_cell_without_output(monkeypatch, tmp_path):
    out = setup(monkeypatch, tmp_path, [native(), native()])
    with pytest.raises(ValueError, match="duplicate"):
        runner.main()
    assert not out.exists()
