from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from scripts.mtlface_components_v2 import MapAgeHead, SpatialAgeIdentitySplit
from scripts.mtlface_recognition_v2 import SpatialRecognitionAdapter
from scripts.mtlface_training_v2 import (
    CosFaceHead,
    FaceRecord,
    RecognitionDataset,
    recognition_step,
    recognition_weights_snapshot,
)
from tests.test_mtlface_recognition_v2 import TinyBackbone


def test_cosface_matches_direct_normalized_dot_product_and_target_margin():
    head = CosFaceHead(5, 3).double()
    embeddings = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 2, 1, 0])
    unit = embeddings / embeddings.norm(dim=1, keepdim=True)
    weights = head.weight / head.weight.norm(dim=1, keepdim=True)
    expected = unit @ weights.T
    for row, label in enumerate(labels):
        expected[row, label] -= 0.35
    expected *= 64
    logits = head(embeddings, labels)
    torch.testing.assert_close(logits, expected)
    F.cross_entropy(logits, labels).backward()
    assert embeddings.grad.abs().sum() > 0 and head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([-1, 0]),
        torch.tensor([0, 3]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([True, False]),
        torch.tensor([0]),
        torch.tensor([[0], [1]]),
    ],
)
def test_cosface_refuses_malformed_labels(labels):
    with pytest.raises(ValueError):
        CosFaceHead(5, 3)(torch.randn(2, 5), labels)


@pytest.mark.parametrize(
    "options",
    [
        {"scale": 0},
        {"scale": float("inf")},
        {"margin": -0.1},
        {"margin": float("nan")},
        {"dimension": 0},
        {"classes": True},
    ],
)
def test_cosface_refuses_malformed_policy(options):
    arguments = dict(dimension=5, classes=3) | options
    with pytest.raises(ValueError):
        CosFaceHead(**arguments)


def prep(image):
    return image.transpose(2, 0, 1).astype(np.float32) / 255


def test_dataset_preserves_zero_and_none_in_collated_labels(monkeypatch):
    monkeypatch.setattr(
        "scripts.mtlface_training_v2.cv2.imread", lambda _: np.ones((2, 2, 3), dtype=np.uint8)
    )
    records = [
        FaceRecord(Path("a"), 0, 0),
        FaceRecord(Path("b"), 1, None),
        FaceRecord(Path("c"), 0, 61),
    ]
    dataset = RecognitionDataset(records, prep)
    images, identities, ages = next(iter(DataLoader(dataset, batch_size=3, shuffle=False)))
    assert dataset.n_classes == 2
    assert ages.tolist() == [0, -1, 61]
    assert identities.tolist() == [0, 1, 0] and identities.dtype == torch.int64
    assert images.shape == (3, 3, 2, 2)


def test_missing_image_not_silently_skipped(monkeypatch):
    monkeypatch.setattr("scripts.mtlface_training_v2.cv2.imread", lambda _: None)
    dataset = RecognitionDataset([FaceRecord(Path("missing"), 0, 0)], prep)
    with pytest.raises(RuntimeError):
        dataset[0]


@pytest.mark.parametrize(
    "records",
    [
        [],
        [FaceRecord(Path("a"), 1, 0)],
        [FaceRecord(Path("a"), True, 0)],
        [FaceRecord(Path("a"), 0, -1)],
    ],
)
def test_dataset_rejects_invalid_metadata(records):
    with pytest.raises(ValueError):
        RecognitionDataset(records, prep)


def test_cosface_and_dataset_batch_integrate_with_spatial_adapter(monkeypatch):
    monkeypatch.setattr(
        "scripts.mtlface_training_v2.cv2.imread", lambda _: np.full((2, 2, 3), 100, dtype=np.uint8)
    )
    records = [FaceRecord(Path(str(i)), i % 2, [0, None, 10, 61][i]) for i in range(4)]
    dataset = RecognitionDataset(records, prep)
    images, identities, ages = next(iter(DataLoader(dataset, batch_size=4)))
    model = SpatialRecognitionAdapter(
        TinyBackbone(), SpatialAgeIdentitySplit(3, 2), MapAgeHead(3, 2, 8), MapAgeHead(3, 2, 8)
    ).train()
    model.set_batchnorm_policy()
    head = CosFaceHead(5, dataset.n_classes)
    losses = model.losses(images, identities, ages, head)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert head.weight.grad.abs().sum() > 0


def joint_model_and_batch():
    model = SpatialRecognitionAdapter(
        TinyBackbone(), SpatialAgeIdentitySplit(3, 2), MapAgeHead(3, 2, 8), MapAgeHead(3, 2, 8)
    )
    head = CosFaceHead(5, 2)
    batch = (torch.randn(4, 3, 2, 2), torch.tensor([0, 1, 0, 1]), torch.tensor([0, 10, 61, -1]))
    return model, head, batch


def test_joint_optimizer_step_updates_modules_and_keeps_frozen_bn_buffers():
    model, head, batch = joint_model_and_batch()
    model.backbone.net.fc.weight.requires_grad_(False)
    parameters = [p for module in (model, head) for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=1e-5)
    before = recognition_weights_snapshot(model, head)
    for _ in range(2):
        result = recognition_step(model, head, optimizer, batch)
        assert all(np.isfinite(value) for value in result.values())
        assert result["gradient_norm"] > 0
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(buffer, before["model"][name])
    torch.testing.assert_close(
        model.backbone.net.fc.weight, before["model"]["backbone.net.fc.weight"]
    )
    for name, module in (
        ("separation", model.separation),
        ("age_head", model.age_head),
        ("age_adversary", model.age_adversary),
    ):
        assert any(
            not torch.equal(p, before["model"][name + "." + key])
            for key, p in module.named_parameters()
        )
    assert not torch.equal(head.weight, before["identity_head"]["weight"])


def test_snapshot_does_not_alias_live_cpu_weights_or_buffers():
    model, head, _ = joint_model_and_batch()
    snapshot = recognition_weights_snapshot(model, head)
    saved = snapshot["identity_head"]["weight"].clone()
    with torch.no_grad():
        head.weight.add_(1)
        model.backbone.net.features.running_mean.add_(2)
    torch.testing.assert_close(snapshot["identity_head"]["weight"], saved)
    assert snapshot["model"]["backbone.net.features.running_mean"].eq(0).all()


def test_optimizer_missing_classifier_refused_before_step():
    model, head, batch = joint_model_and_batch()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    before = head.weight.detach().clone()
    with pytest.raises(ValueError, match="every trainable"):
        recognition_step(model, head, optimizer, batch)
    torch.testing.assert_close(head.weight, before)


def test_nonfinite_input_refused_before_weight_update():
    model, head, batch = joint_model_and_batch()
    optimizer = torch.optim.SGD([*model.parameters(), *head.parameters()], lr=1e-5)
    batch[0][0, 0, 0, 0] = float("nan")
    before = head.weight.detach().clone()
    with pytest.raises(ValueError, match="finite"):
        recognition_step(model, head, optimizer, batch)
    torch.testing.assert_close(head.weight, before)


def test_nonfinite_gradient_refuses_optimizer_step():
    model, head, batch = joint_model_and_batch()
    optimizer = torch.optim.SGD([*model.parameters(), *head.parameters()], lr=1e-5)
    before = recognition_weights_snapshot(model, head)
    handle = head.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    try:
        with pytest.raises(FloatingPointError, match="gradient"):
            recognition_step(model, head, optimizer, batch)
    finally:
        handle.remove()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before["model"][key])
    torch.testing.assert_close(head.weight, before["identity_head"]["weight"])


def test_optimizer_stale_after_scope_change_refused():
    model, head, batch = joint_model_and_batch()
    optimizer = torch.optim.SGD([*model.parameters(), *head.parameters()], lr=1e-5)
    model.backbone.net.fc.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="every trainable"):
        recognition_step(model, head, optimizer, batch)
