import pytest
import torch

from age_gap.models.facenet import FaceNetBackbone
from age_gap.training.finetune import _apply_batchnorm_policy, _set_trainable
from scripts.facenet_cached_head_v1 import CachedFaceNetHead
from scripts.probe_facenet_head_cache_v1 import cached_forward


@pytest.fixture
def models():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    native = FaceNetBackbone(pretrained=None)
    wrapper = CachedFaceNetHead(native)
    _set_trainable(wrapper, "head")
    wrapper.train()
    _apply_batchnorm_policy(wrapper, "frozen_all")
    return native, wrapper


def test_native_checkpoint_keys_and_parameter_order(models):
    native, wrapper = models
    assert list(wrapper.state_dict()) == list(native.state_dict())
    assert all(key.startswith("net.") for key in wrapper.state_dict())
    assert [id(p) for p in wrapper.parameters()] == [id(p) for p in native.parameters()]
    state = {key: value.clone() for key, value in native.state_dict().items()}
    wrapper.load_state_dict(state, strict=True)


def test_train_eval_and_continuing_dropout_rng(models):
    native, wrapper = models
    features = torch.ones(2, 1792, 1, 1)
    rng = torch.get_rng_state().clone()
    output = wrapper(features)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    reference = cached_forward(native, features)
    assert torch.equal(output, reference)
    assert torch.equal(after, torch.get_rng_state())
    assert not torch.equal(rng, after)
    wrapper.eval()
    assert not native.training and not native.net.dropout.training
    rng = torch.get_rng_state().clone()
    wrapper(features)
    assert torch.equal(rng, torch.get_rng_state())


def test_bn_policy_required_after_train(models):
    _, wrapper = models
    wrapper.train()
    with pytest.raises(ValueError, match="BatchNorm"):
        wrapper(torch.ones(2, 1792, 1, 1))
