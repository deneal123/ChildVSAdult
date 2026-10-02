import pytest
import torch

from age_gap.training.finetune import _apply_batchnorm_policy


def test_frozen_statistics_preserve_affine_gradients_and_dropout_mode():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Dropout(0.1))
    model.train()
    before = model[0].running_mean.clone()
    _apply_batchnorm_policy(model, "frozen_all")
    assert not model[0].training and model[1].training
    model(torch.randn(8, 3) + 20).sum().backward()
    assert torch.equal(before, model[0].running_mean)
    assert model[0].weight.grad is not None
    assert model[0].num_batches_tracked.item() == 0


def test_adaptation_remains_backward_compatible():
    model = torch.nn.BatchNorm1d(3).train()
    _apply_batchnorm_policy(model, "adapt_all")
    model(torch.randn(8, 3) + 20)
    assert model.num_batches_tracked.item() == 1


def test_invalid_policy_fails_closed():
    with pytest.raises(ValueError):
        _apply_batchnorm_policy(torch.nn.Linear(1, 1), "unknown")
