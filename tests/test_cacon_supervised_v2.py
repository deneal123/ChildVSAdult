import pytest
import torch
from torch import nn

from scripts.cacon_supervised_v2 import FinalLinearStage


def fixture():
    backbone = nn.Sequential(nn.Linear(4, 5), nn.BatchNorm1d(5))
    for parameter in backbone.parameters():
        parameter.grad = torch.ones_like(parameter)
    stage = FinalLinearStage(backbone, 5, 3)
    optimizer = torch.optim.SGD(stage.classifier.parameters(), lr=0.01)
    return stage, optimizer, torch.randn(4, 4), torch.tensor([0, 1, 2, 0])


def test_persistent_freeze_clears_stale_gradients_and_preserves_backbone_representation():
    stage, optimizer, images, labels = fixture()
    before = {key: value.clone() for key, value in stage.backbone.state_dict().items()}
    features = stage.verification_features(images, representation="backbone", normalize=False)
    logits = stage.verification_features(images, representation="logits", normalize=False)
    for _ in range(3):
        stage.train()
        result = stage.supervised_step(images, labels, optimizer)
        assert result["gradient_norm"] > 0
        assert not stage.backbone.training
    for key, value in stage.backbone.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert all(p.grad is None and not p.requires_grad for p in stage.backbone.parameters())
    torch.testing.assert_close(
        features, stage.verification_features(images, representation="backbone", normalize=False)
    )
    assert not torch.equal(
        logits, stage.verification_features(images, representation="logits", normalize=False)
    )


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([0, 1, 2, 0], dtype=torch.int32),
        torch.tensor([True] * 4),
        torch.tensor([-1, 0, 1, 2]),
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([0.0] * 4),
        torch.tensor([[0]] * 4),
        torch.tensor([], dtype=torch.int64),
    ],
)
def test_invalid_labels_refused(labels):
    stage, optimizer, images, _ = fixture()
    with pytest.raises(ValueError):
        stage.supervised_step(images, labels, optimizer)


def test_duplicate_optimizer_parameters_refused_before_update():
    stage, optimizer, images, labels = fixture()
    optimizer.param_groups[0]["params"].append(stage.classifier.weight)
    before = stage.classifier.weight.detach().clone()
    with pytest.raises(ValueError, match="exactly once"):
        stage.supervised_step(images, labels, optimizer)
    torch.testing.assert_close(before, stage.classifier.weight)


def test_optimizer_contamination_refused():
    stage, optimizer, images, labels = fixture()
    optimizer.param_groups[0]["params"].append(stage.backbone[0].weight)
    with pytest.raises(ValueError, match="exactly once"):
        stage.supervised_step(images, labels, optimizer)


def test_nonfinite_gradient_does_not_update_classifier():
    stage, optimizer, images, labels = fixture()
    before = {key: value.clone() for key, value in stage.classifier.state_dict().items()}
    handle = stage.classifier.weight.register_hook(lambda g: torch.full_like(g, float("inf")))
    try:
        with pytest.raises(FloatingPointError, match="gradient"):
            stage.supervised_step(images, labels, optimizer)
    finally:
        handle.remove()
    for key, value in stage.classifier.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_nonfinite_input_refused_before_forward():
    stage, optimizer, images, labels = fixture()
    images[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        stage.supervised_step(images, labels, optimizer)


def test_nonfinite_classifier_loss_refused_before_step(monkeypatch):
    stage, optimizer, images, labels = fixture()
    before = stage.classifier.weight.detach().clone()
    monkeypatch.setattr(stage, "forward", lambda _: torch.full((4, 3), float("inf")))
    with pytest.raises(FloatingPointError, match="loss"):
        stage.supervised_step(images, labels, optimizer)
    torch.testing.assert_close(stage.classifier.weight, before)


def test_wrong_embedding_shape_refused():
    stage = FinalLinearStage(nn.Linear(4, 6), 5, 3)
    with pytest.raises(ValueError, match="embedding shape"):
        stage(torch.randn(4, 4))


def test_explicit_verification_normalization_and_unknown_representation_refusal():
    stage, _, images, _ = fixture()
    for representation in ("backbone", "logits"):
        output = stage.verification_features(images, representation=representation, normalize=True)
        torch.testing.assert_close(output.norm(dim=1), torch.ones(4))
    with pytest.raises(ValueError):
        stage.verification_features(images, representation="projection", normalize=True)
    assert not hasattr(stage, "projection")
