import copy

import pytest
import torch

from scripts.qualify_facenet_trunk_cache_v1 import compare_checkpoints


def checkpoint():
    return dict(
        selected_epoch=10,
        checkpoint_selection="last_epoch",
        state_dict={"net.weight": torch.ones(2), "net.count": torch.tensor(0)},
        history=[
            dict(
                epoch=i,
                training_loss=0.2,
                validation_auc=0.8,
                mean_gradient_norm=0.1,
                epoch_seconds=200,
            )
            for i in range(1, 11)
        ],
    )


def test_equal_and_timing_ignored():
    a = checkpoint()
    b = copy.deepcopy(a)
    b["history"][0]["epoch_seconds"] = 1
    assert compare_checkpoints(a, b)["reference_cell_equivalence_passed"]


def test_changed_state_or_history_fails():
    a = checkpoint()
    b = copy.deepcopy(a)
    b["state_dict"]["net.weight"].add_(0.01)
    assert not compare_checkpoints(a, b)["reference_cell_equivalence_passed"]
    b = copy.deepcopy(a)
    b["history"][5]["validation_auc"] = 0.81
    assert not compare_checkpoints(a, b)["reference_cell_equivalence_passed"]


def test_partial_rejected():
    a = checkpoint()
    b = copy.deepcopy(a)
    b["history"].pop()
    with pytest.raises(ValueError, match="sequential"):
        compare_checkpoints(a, b)
