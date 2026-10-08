"""Spatial recognition-stage adapter; not a complete joint MTLFace reproduction.

The existing iResNet weights/head remain the common backbone adaptation. Spatial
AFD and map-level age heads must be supplied explicitly, never a vector gate.
Reference: Hzzone/MTLFace commit03ad579, backbone/aifr.py and models/fr.py.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


def age_loss(head, maps, ages):
    if (
        ages.ndim != 1
        or len(ages) != len(maps)
        or ages.dtype not in (torch.int32, torch.int64)
        or (ages < -1).any()
    ):
        raise ValueError("aligned integer ages with only -1 missing required")
    valid = ages >= 0
    if not valid.any():
        return maps.sum() * 0
    years, groups = head(maps[valid])
    if years.shape != (int(valid.sum()), 101) or groups.shape != (int(valid.sum()), 7):
        raise ValueError("101-year and seven-group logits required")
    values = ages[valid]
    expectation = years.softmax(dim=1) @ torch.arange(101, device=years.device, dtype=years.dtype)
    # bucketize/right=False implements strict age > threshold (10 remains group0).
    targets = torch.bucketize(values, values.new_tensor([10, 20, 30, 40, 50, 60]), right=False)
    return F.mse_loss(expectation, values.to(years.dtype)) + F.cross_entropy(groups, targets)


class SpatialRecognitionAdapter(nn.Module):
    def __init__(self, backbone, separation, age_head, age_adversary):
        super().__init__()
        required = (
            "conv1",
            "bn1",
            "prelu",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "bn2",
            "dropout",
            "fc",
            "features",
        )
        if not hasattr(backbone, "net") or any(not hasattr(backbone.net, n) for n in required):
            raise ValueError("common iResNet map/output-layer contract required")
        self.backbone = backbone
        self.separation = separation
        self.age_head = age_head
        self.age_adversary = age_adversary
        self.input_size = backbone.input_size
        self.preprocess = backbone.preprocess

    def components(self, images):
        net = self.backbone.net
        maps = net.prelu(net.bn1(net.conv1(images)))
        for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
            maps = layer(maps)
        identity, age = self.separation(maps)
        if identity.shape != maps.shape or age.shape != maps.shape or maps.ndim != 4:
            raise ValueError("spatial age and identity maps required before pooling")
        embedding = net.bn2(identity)
        embedding = net.dropout(torch.flatten(embedding, 1))
        embedding = net.features(net.fc(embedding))
        return F.normalize(embedding, dim=1), identity, age

    def forward(self, images):
        embedding, _, _ = self.components(images)
        return embedding

    def losses(self, images, identities, ages, identity_head):
        embedding, identity_map, age_map = self.components(images)
        identity = F.cross_entropy(identity_head(embedding, identities), identities)
        estimation = age_loss(self.age_head, age_map, ages)
        adversarial = age_loss(self.age_adversary, _ReverseGradient.apply(identity_map), ages)
        return dict(
            identity=identity,
            age=estimation,
            adversarial_age=adversarial,
            total=identity + 0.001 * estimation + 0.002 * adversarial,
        )

    def set_batchnorm_policy(self, policy="frozen_all"):
        if policy != "frozen_all":
            raise ValueError("this confirmatory adapter requires explicit frozen_all BN policy")
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        # Affine parameter trainability is set separately by the campaign scope.
