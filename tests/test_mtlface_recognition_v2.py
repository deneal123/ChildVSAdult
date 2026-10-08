import pytest
import torch
from torch import nn

from scripts.mtlface_recognition_v2 import SpatialRecognitionAdapter, _ReverseGradient, age_loss


class TinyBackbone(nn.Module):
    input_size = 2

    def __init__(self):
        super().__init__()
        self.net = nn.Module()
        for name in (
            "conv1",
            "bn1",
            "prelu",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "bn2",
            "dropout",
        ):
            setattr(self.net, name, nn.Identity())
        self.net.fc = nn.Linear(12, 5)
        self.net.features = nn.BatchNorm1d(5)

    def preprocess(self, value):
        return value


class Split(nn.Module):
    def forward(self, value):
        return value * 0.75, value * 0.25


class TinyAge(nn.Module):
    def __init__(self):
        super().__init__()
        self.year = nn.Linear(12, 101)
        self.group = nn.Linear(101, 7)

    def forward(self, maps):
        logits = self.year(maps.flatten(1))
        return logits, self.group(logits)


def adapter():
    return SpatialRecognitionAdapter(TinyBackbone(), Split(), TinyAge(), TinyAge())


def test_adapter_separates_maps_before_existing_output_layer():
    model = adapter().eval()
    images = torch.randn(3, 3, 2, 2)
    embedding, identity, age = model.components(images)
    assert identity.shape == age.shape == images.shape
    torch.testing.assert_close(identity + age, images)
    expected = model.backbone.net.features(model.backbone.net.fc(identity.flatten(1)))
    torch.testing.assert_close(embedding, nn.functional.normalize(expected, dim=1))


def test_gradient_reversal_and_age_zero_supervision():
    value = torch.tensor([2.0], requires_grad=True)
    _ReverseGradient.apply(value).sum().backward()
    assert value.grad.item() == -1
    maps = torch.randn(3, 3, 2, 2, requires_grad=True)
    loss = age_loss(TinyAge(), maps, torch.tensor([0, -1, 10]))
    loss.backward()
    assert maps.grad[0].abs().sum() > 0
    assert maps.grad[1].abs().sum() == 0


def test_missing_ages_keep_zero_gradient_connection():
    maps = torch.randn(3, 3, 2, 2, requires_grad=True)
    loss = age_loss(TinyAge(), maps, torch.full((3,), -1))
    assert loss.item() == 0
    loss.backward()
    assert maps.grad is not None and maps.grad.abs().sum() == 0


def test_frozen_bn_is_explicit_and_does_not_freeze_affine_parameters():
    model = adapter().train()
    bn = model.backbone.net.features
    model.set_batchnorm_policy()
    assert not bn.training and bn.weight.requires_grad
    before = bn.running_mean.clone()
    model(torch.randn(3, 3, 2, 2))
    torch.testing.assert_close(before, bn.running_mean)
    with pytest.raises(ValueError):
        model.set_batchnorm_policy("adapt_all")


@pytest.mark.parametrize(
    "ages",
    [
        torch.tensor([True] * 3),
        torch.tensor([-2, 0, 1]),
        torch.tensor([1.0] * 3),
        torch.tensor([0, 1]),
    ],
)
def test_malformed_age_metadata_refused(ages):
    with pytest.raises(ValueError):
        age_loss(TinyAge(), torch.randn(3, 3, 2, 2), ages)


def test_aifr_losses_use_exact_declared_weights():
    model = adapter().eval()

    def identity_head(embeddings, _):
        return embeddings[:, :3]

    losses = model.losses(
        torch.randn(3, 3, 2, 2), torch.tensor([0, 1, 2]), torch.tensor([0, 10, -1]), identity_head
    )
    torch.testing.assert_close(
        losses["total"],
        losses["identity"] + 0.001 * losses["age"] + 0.002 * losses["adversarial_age"],
    )


def test_wrong_backbone_contract_refused():
    with pytest.raises(ValueError):
        SpatialRecognitionAdapter(nn.Linear(3, 3), Split(), TinyAge(), TinyAge())
