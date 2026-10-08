"""Recognition-stage input/head wiring, not a complete MTLFace training loop.

CosFace algorithm reference: Hzzone/MTLFace@03ad57942e19ee28733b03a441765481f6606460,
head/cosface.py (s=64, m=0.35). The image list must come from a separately
validated, source-bound split; integer labels here do not prove true-person
independence. No existing source-bound training module is modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from scripts.mtlface_components_v2 import _positive_dimensions, age_target


class CosFaceHead(nn.Module):
    def __init__(self, dimension, classes, scale=64.0, margin=0.35):
        super().__init__()
        _positive_dimensions(dimension, classes)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("finite positive scale required")
        if not math.isfinite(margin) or not 0 <= margin <= 1:
            raise ValueError("finite margin between zero and one required")
        self.scale, self.margin = float(scale), float(margin)
        self.dimension, self.classes = dimension, classes
        self.weight = nn.Parameter(torch.empty(classes, dimension))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, identities):
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError("declared embedding dimension required")
        if (
            identities.ndim != 1
            or identities.shape[0] != embeddings.shape[0]
            or identities.dtype != torch.int64
            or identities.device != embeddings.device
            or (identities < 0).any()
            or (identities >= self.classes).any()
        ):
            raise ValueError("aligned int64 identity labels within class range required")
        cosine = F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))
        target = F.one_hot(identities, num_classes=self.classes).to(cosine.dtype)
        return self.scale * (cosine - self.margin * target)


@dataclass(frozen=True)
class FaceRecord:
    path: Path
    identity: int
    age: int | None


class RecognitionDataset(Dataset):
    """Explicit records; no hidden split selection or silently omitted images."""

    def __init__(self, records, preprocess):
        self.records = tuple(records)
        if not self.records:
            raise ValueError("nonempty source-bound records required")
        labels = set()
        for record in self.records:
            if (
                isinstance(record.identity, bool)
                or not isinstance(record.identity, int)
                or record.identity < 0
            ):
                raise ValueError("nonnegative integer identity labels required")
            age_target(record.age)
            labels.add(record.identity)
        if labels != set(range(len(labels))):
            raise ValueError("contiguous identity labels starting at zero required")
        self.n_classes = len(labels)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(record.path))
        if image is None:
            raise RuntimeError(f"cannot read {record.path}")
        tensor = torch.from_numpy(np.asarray(self.preprocess(image)))
        if (
            tensor.ndim != 3
            or tensor.shape[0] != 3
            or not tensor.is_floating_point()
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("preprocessing must return finite three-channel CHW floats")
        return tensor, record.identity, age_target(record.age)


def recognition_step(model, identity_head, optimizer, batch):
    """One joint step with caller-declared optimizer/scope, no hidden schedule.

    Every trainable parameter must occur exactly once in the optimizer. All BN
    buffers are frozen after train(); affine trainability remains caller-owned.
    This function neither selects checkpoints nor defines a campaign protocol.
    """
    expected = {
        id(parameter)
        for module in (model, identity_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if not expected or set(actual) != expected or len(actual) != len(expected):
        raise ValueError("optimizer must own every trainable parameter exactly once")
    images, identities, ages = batch
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError("finite floating point images required")
    model.train()
    identity_head.train()
    model.set_batchnorm_policy("frozen_all")
    optimizer.zero_grad(set_to_none=True)
    losses = model.losses(images, identities, ages, identity_head)
    if not torch.isfinite(losses["total"]):
        raise FloatingPointError("nonfinite loss; optimizer step refused")
    losses["total"].backward()
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("nonfinite gradient; optimizer step refused")
    gradient_squared = sum(
        p.grad.detach().double().square().sum().item() for p in parameters if p.grad is not None
    )
    optimizer.step()
    return {name: value.detach().item() for name, value in losses.items()} | {
        "gradient_norm": math.sqrt(gradient_squared)
    }


def recognition_weights_snapshot(model, identity_head):
    """Independent CPU weight/buffer snapshot; not a resumable training state."""
    return {
        name: {key: tensor.detach().cpu().clone() for key, tensor in module.state_dict().items()}
        for name, module in (("model", model), ("identity_head", identity_head))
    }
