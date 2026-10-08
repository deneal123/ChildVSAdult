"""Declared CACon stage-2 linear-probe adaptation, not full-method parity.

Primary: arXiv:2312.11195v2, section2.1 describes label-based final-linear
fine-tuning; section3 reports SGD and batch512. Frozen encoder, new classifier,
cross-entropy, projection removal and verification representation are common
protocol choices, not recovered author details. Updating this classifier does
not update the frozen backbone's embeddings.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from scripts.mtlface_components_v2 import _positive_dimensions


class FinalLinearStage(nn.Module):
    def __init__(self, backbone, dimension, classes, *, normalize_input=False):
        super().__init__()
        _positive_dimensions(dimension, classes)
        if not isinstance(normalize_input, bool):
            raise ValueError("explicit boolean normalization policy required")
        self.backbone = backbone
        self.dimension, self.classes = dimension, classes
        self.normalize_input = normalize_input
        self.classifier = nn.Linear(dimension, classes)
        self._freeze_backbone()

    def _freeze_backbone(self):
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        # Stage1 may leave gradients on the transferred feature extractor.
        for parameter in self.backbone.parameters():
            parameter.grad = None

    def train(self, mode=True):
        super().train(mode)
        self._freeze_backbone()
        return self

    def extract_features(self, images):
        self._freeze_backbone()
        with torch.no_grad():
            features = self.backbone(images)
            if not isinstance(features, torch.Tensor) or features.shape != (
                len(images),
                self.dimension,
            ):
                raise ValueError("declared batch/embedding shape required")
            if not torch.isfinite(features).all():
                raise FloatingPointError("nonfinite backbone features")
            if self.normalize_input:
                features = F.normalize(features, dim=1)
        return features.detach()

    def forward(self, images):
        return self.classifier(self.extract_features(images))

    def verification_features(self, images, *, representation, normalize):
        """Require a declared downstream representation; never silently pick it."""
        if representation not in ("backbone", "logits") or not isinstance(normalize, bool):
            raise ValueError("explicit backbone/logits and boolean normalization required")
        with torch.no_grad():
            features = (
                self.extract_features(images) if representation == "backbone" else self(images)
            )
            if not torch.isfinite(features).all():
                raise FloatingPointError("nonfinite verification representation")
            return F.normalize(features, dim=1) if normalize else features

    def supervised_step(self, images, labels, optimizer):
        expected = {id(p) for p in self.classifier.parameters() if p.requires_grad}
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if set(actual) != expected or len(actual) != len(expected) or not expected:
            raise ValueError("optimizer must own every trainable classifier parameter exactly once")
        if (
            not isinstance(labels, torch.Tensor)
            or labels.dtype != torch.int64
            or labels.ndim != 1
            or len(labels) != len(images)
            or not len(labels)
            or labels.device != images.device
            or (labels < 0).any()
            or (labels >= self.classes).any()
        ):
            raise ValueError("aligned nonempty int64 labels in class range required")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("finite floating-point input required")
        self.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self(images), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite supervised loss; step refused")
        loss.backward()
        parameters = list(self.classifier.parameters())
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("nonfinite classifier gradient; step refused")
        norm = math.sqrt(
            sum(
                p.grad.detach().double().square().sum().item()
                for p in parameters
                if p.grad is not None
            )
        )
        optimizer.step()
        return dict(loss=loss.detach().item(), gradient_norm=norm, samples=len(labels))
