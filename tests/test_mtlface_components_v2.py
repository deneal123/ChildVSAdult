import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scripts.mtlface_components_v2 import MapAgeHead, SpatialAgeIdentitySplit, age_group, age_target
from scripts.mtlface_recognition_v2 import SpatialRecognitionAdapter
from tests.test_mtlface_recognition_v2 import TinyBackbone


def test_split_matches_independent_pool_and_attention_arithmetic():
    split = SpatialAgeIdentitySplit(3, 2).eval()
    maps = torch.randn(4, 3, 4, 5, requires_grad=True)
    pooled = []
    for pooling in (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d):
        pooled.append(torch.cat([pooling(n)(maps).reshape(4, -1) for n in (1, 2, 3)], 1))
    channel = split.channel((pooled[0] + pooled[1]).reshape(4, 42, 1, 1))
    spatial_input = torch.stack((maps.amax(dim=1), maps.mean(dim=1)), 1)
    expected_age = maps * (channel + split.spatial(spatial_input)) / 2
    identity, age = split(maps)
    torch.testing.assert_close(age, expected_age)
    torch.testing.assert_close(identity + age, maps)
    identity.square().mean().backward()
    assert maps.grad.abs().sum() > 0
    assert split.channel[0].weight.grad.abs().sum() > 0


def test_map_head_group_from_year_logits_and_distinct_bn_momentum():
    head = MapAgeHead(3, 2, 8).eval()
    split = SpatialAgeIdentitySplit(3, 2)
    assert head.age_output_layer[0].momentum == 0.1
    assert split.channel[3].momentum == split.spatial[1].momentum == 0.01
    maps = torch.randn(4, 3, 2, 2)
    years, groups = head(maps)
    layers = head.age_output_layer
    bn = layers[0]
    normalized = F.batch_norm(maps, bn.running_mean, bn.running_var, bn.weight, bn.bias)
    expected = F.linear(
        F.relu(F.linear(normalized.flatten(1), layers[2].weight, layers[2].bias)),
        layers[4].weight,
        layers[4].bias,
    )
    torch.testing.assert_close(years, expected)
    torch.testing.assert_close(
        groups, F.linear(years, head.group_output_layer.weight, head.group_output_layer.bias)
    )
    groups.sum().backward()
    assert layers[4].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("value", range(102))
def test_group_matches_strict_boundary_formula(value):
    expected = sum(value > boundary for boundary in (10, 20, 30, 40, 50, 60))
    assert age_group(value) == expected
    assert (
        torch.bucketize(torch.tensor(value), torch.tensor([10, 20, 30, 40, 50, 60])).item()
        == expected
    )


def test_only_none_is_missing_age():
    assert age_target(None) == -1 and age_target(0) == 0
    with pytest.raises(ValueError):
        age_group(None)
    for value in (True, 1.5, "10"):
        with pytest.raises(TypeError):
            age_target(value)
    with pytest.raises(ValueError):
        age_target(-1)


@pytest.mark.parametrize("shape", [(4, 3, 2), (4, 2, 2, 2), (4, 3, 0, 2)])
def test_bad_map_shape_refused(shape):
    for module in (SpatialAgeIdentitySplit(3, 2), MapAgeHead(3, 2, 8)):
        with pytest.raises(ValueError):
            module(torch.empty(shape))


def test_head_refuses_wrong_spatial_size():
    with pytest.raises(ValueError):
        MapAgeHead(3, 2, 8)(torch.randn(4, 3, 2, 3))


def test_real_components_integrate_with_adapter_and_frozen_bn_backward():
    model = SpatialRecognitionAdapter(
        TinyBackbone(), SpatialAgeIdentitySplit(3, 2), MapAgeHead(3, 2, 8), MapAgeHead(3, 2, 8)
    ).train()
    model.set_batchnorm_policy()
    before = {n: b.clone() for n, b in model.named_buffers()}

    def identity_head(embeddings, _):
        return embeddings[:, :3]

    losses = model.losses(
        torch.randn(4, 3, 2, 2),
        torch.tensor([0, 1, 2, 0]),
        torch.tensor([0, 10, 61, -1]),
        identity_head,
    )
    losses["total"].backward()
    for module in (model.separation, model.age_head, model.age_adversary):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name])


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_component_dimensions(bad):
    with pytest.raises(ValueError):
        MapAgeHead(channels=bad)
    with pytest.raises(ValueError):
        SpatialAgeIdentitySplit(channels=bad)
