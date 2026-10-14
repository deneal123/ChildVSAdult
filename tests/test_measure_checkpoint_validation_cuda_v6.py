"""Synthetic component-wrapper compatibility; no neural/CUDA/native evidence."""

import copy
from types import SimpleNamespace

import pytest

from age_gap.common.manifest import file_record
from scripts import measure_checkpoint_validation_cuda_v5 as v5
from scripts import measure_checkpoint_validation_cuda_v6 as v6


def fixture(tmp_path):
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"synthetic-not-weights")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_bytes(b"synthetic-not-pairs")
    params = dict(backbone="adaface_ir101", epochs_requested=8, epochs_executed=8,
                  selected_epoch=8, checkpoint_selection="last_epoch", batchnorm_policy="frozen_all",
                  trainable_scope="tail", learning_rate=1e-5, batch_size=16, margin=0.3,
                  gap_weight=0.0, crops_dir="faces", seed=2, pairs_file=str(pairs))
    history = [dict(epoch=i, training_loss=0.2, validation_auc=0.8,
                    mean_gradient_norm=1.0, epoch_seconds=1.0) for i in range(1, 9)]
    component = dict(parameters=copy.deepcopy(params), inputs=[file_record(pairs)],
                     outputs=[file_record(checkpoint)], metrics=dict(history=copy.deepcopy(history)))
    parent = dict(params, device="cuda", threads=1, negative="random", torch_version="synthetic",
                  cuda_version="synthetic", gpu_name="synthetic", cudnn_version=123)
    return component, checkpoint, parent, history


def test_known_seven_wrapper_fields_are_not_training_parameter_disagreement(tmp_path):
    component, checkpoint, parent, history = fixture(tmp_path)
    assert set(parent) - set(component["parameters"]) == v6.NATIVE_WRAPPER_FIELDS
    v6.verify_component_binding_v6(component, checkpoint, parent, parent_history=history)
    with pytest.raises(ValueError, match="keys differ"):
        v5.verify_component_binding_v5(component, checkpoint, parent, parent_history=history)


@pytest.mark.parametrize("change", ["unknown-extra", "missing-shared", "bool-shared",
                                    "different-history", "same-valued-float-epoch"])
def test_no_training_parameter_or_history_guard_is_weakened(tmp_path, change):
    component, checkpoint, parent, history = fixture(tmp_path)
    if change == "unknown-extra":
        parent["hidden_optimizer_override"] = "forbidden"
    elif change == "missing-shared":
        component["parameters"]["declared_extra_training_setting"] = 1
    elif change == "bool-shared":
        component["parameters"]["gap_weight"] = False
    elif change == "different-history":
        component["metrics"]["history"][0]["training_loss"] = 0.3
    else:
        component["metrics"]["history"][0]["epoch"] = 1.0
    with pytest.raises(ValueError):
        v6.verify_component_binding_v6(component, checkpoint, parent, parent_history=history)


def test_isolation_and_source_closure_preserve_canonical_v5():
    old = (v5.EXPERIMENT, v5.__file__, v5.MANDATORY_SEEDS, v5.verify_component_binding_v5)
    module = v6.isolated_v5()
    assert module is not v5 and module.EXPERIMENT == v6.EXPERIMENT
    assert module.__file__ == v6.__file__
    assert "scripts.measure_checkpoint_validation_cuda_v5" in module.MANDATORY_SEEDS
    assert "scripts.measure_checkpoint_validation_cuda_v6" in module.MANDATORY_SEEDS
    assert module.verify_component_binding_v5 is v6.verify_component_binding_v6
    assert (v5.EXPERIMENT, v5.__file__, v5.MANDATORY_SEEDS, v5.verify_component_binding_v5) == old


def test_dry_cli_performs_no_cuda_or_native_preflight(tmp_path, monkeypatch, capsys):
    def forbidden():
        raise AssertionError("dry-run must not create an isolated measurement")
    monkeypatch.setattr(v6, "isolated_v5", forbidden)
    out = tmp_path / "never-created"
    v6.main(["--train", str(tmp_path / "missing"), "--out", str(out)])
    assert "no CUDA probe" in capsys.readouterr().out
    assert not out.exists()


def test_run_uses_only_owned_isolated_module(monkeypatch):
    sentinel = object()
    args = SimpleNamespace(train="unused", out="unused")
    module = SimpleNamespace(run=lambda supplied: sentinel if supplied is args else None)
    monkeypatch.setattr(v6, "isolated_v5", lambda: module)
    assert v6.run(args) is sentinel
