import copy

import pytest
import torch

from age_gap.models.facenet import FaceNetBackbone
from age_gap.training.finetune import _apply_batchnorm_policy, _set_trainable
from scripts.probe_facenet_head_cache_v1 import (
    cached_forward,
    capture_features,
    compare,
    probe,
    require_contract,
)


@pytest.fixture(scope="module")
def base():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = FaceNetBackbone(pretrained=None)
    _set_trainable(model, "head")
    model.train()
    _apply_batchnorm_policy(model, "frozen_all")
    return model


@pytest.fixture(scope="module")
def images():
    return torch.randn(2, 3, 160, 160, generator=torch.Generator().manual_seed(22))


def test_capture_preserves_rng_and_every_flag(base, images):
    before = torch.get_rng_state().clone()
    flags = [m.training for m in base.modules()]
    features = capture_features(base, images)
    assert features.shape == (2, 1792, 1, 1)
    assert not features.requires_grad
    assert torch.equal(before, torch.get_rng_state())
    assert flags == [m.training for m in base.modules()]


def test_native_eval_equals_cached_path(base, images):
    model = copy.deepcopy(base).eval()
    features = capture_features(model, images)
    with torch.no_grad():
        native = model(images)
        cached = cached_forward(model, features)
    assert compare(native, cached)["bitwise_equal"]


def test_live_head_not_frozen_embedding_adapter(base, images):
    model = copy.deepcopy(base).eval()
    features = capture_features(model, images)
    with torch.no_grad():
        old = cached_forward(model, features)
        model.net.last_bn.bias.add_(0.1)
        updated = cached_forward(model, features)
        native = model(images)
    assert not torch.equal(old, updated)
    assert compare(native, updated)["bitwise_equal"]


def test_continuing_dropout_gradients_adam_and_rebatching(base, images):
    batch = (images, images.flip(0), torch.tensor([1.0, 0.0]), torch.ones(2))
    rows = probe(base, batch, seed=42, sizes=(2, 1, 2))
    assert len(rows) == 3
    assert all(row["within_tolerance"] for row in rows)
    assert all("adam:last_linear.weight:exp_avg_sq" in row["comparisons"] for row in rows)


def test_adapting_bn_rejected(base):
    model = copy.deepcopy(base)
    model.net.last_bn.train()
    with pytest.raises(ValueError, match="BatchNorm"):
        require_contract(model)


def test_tail_or_full_scope_rejected(base):
    model = copy.deepcopy(base)
    _set_trainable(model, "tail")
    with pytest.raises(ValueError, match="three native head"):
        require_contract(model)


def test_other_architecture_rejected():
    with pytest.raises(ValueError, match="exact FaceNet"):
        require_contract(torch.nn.Linear(2, 2))


@pytest.mark.parametrize(
    "features",
    [
        torch.zeros(2, 512),
        torch.zeros(0, 1792, 1, 1),
        torch.zeros(1, 1792, 1, 1, requires_grad=True),
        torch.full((1, 1792, 1, 1), float("nan")),
        torch.zeros(1, 1792, 1, 1, dtype=torch.float64),
    ],
)
def test_invalid_cached_features_rejected(base, features):
    with pytest.raises(ValueError, match="pre-dropout"):
        cached_forward(base, features)


def test_nonfinite_comparison_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        compare(torch.ones(1), torch.tensor([float("nan")]))


def test_invalid_batch_schedule_rejected(base, images):
    with pytest.raises(ValueError, match="batch sizes"):
        probe(base, (images, images, torch.ones(2), torch.ones(2)), seed=1, sizes=(3,))
