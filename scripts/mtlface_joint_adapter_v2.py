"""Common-backbone FAS shortcuts; not a complete joint training system.

Reference contract: Hzzone/MTLFace@03ad579, backbone/aifr.py::AIResNet.
Recognition normalization/BN policy stays the declared common adaptation.
No existing experiment-bound adapter or backbone source is modified.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from scripts.mtlface_recognition_v2 import SpatialRecognitionAdapter


class CommonJointAdapter(SpatialRecognitionAdapter):
    """Five encoder stages plus spatial identity/age maps for the FAS decoder."""

    def __init__(
        self,
        backbone,
        separation,
        age_head,
        age_adversary,
        *,
        stage_channels=(64, 64, 128, 256, 512),
    ):
        super().__init__(backbone, separation, age_head, age_adversary)
        if (
            len(stage_channels) != 5
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in stage_channels)
            or isinstance(self.input_size, bool)
            or not isinstance(self.input_size, int)
            or self.input_size < 16
            or self.input_size % 16
        ):
            raise ValueError("five positive stage channels and resolution divisible by16 required")
        self.stage_channels = tuple(stage_channels)
        self.full_joint_fas = False

    def shortcuts(self, images):
        if (
            images.ndim != 4
            or images.shape[0] < 1
            or images.shape[1:] != (3, self.input_size, self.input_size)
            or not images.is_floating_point()
            or not torch.isfinite(images).all()
        ):
            raise ValueError("finite floating BCHW images at declared resolution required")
        net = self.backbone.net
        value = net.prelu(net.bn1(net.conv1(images)))
        stages = [value]
        for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
            value = layer(value)
            stages.append(value)
        for index, (stage, channels) in enumerate(zip(stages, self.stage_channels, strict=True)):
            size = self.input_size // (2**index)
            if stage.shape != (len(images), channels, size, size):
                raise ValueError("encoder stage channel/spatial contract mismatch")
            if not torch.isfinite(stage).all():
                raise FloatingPointError("nonfinite encoder shortcut")
        identity, age = self.separation(stages[-1])
        if identity.shape != value.shape or age.shape != value.shape:
            raise ValueError("spatial age/identity contract mismatch")
        if not torch.isfinite(identity).all() or not torch.isfinite(age).all():
            raise FloatingPointError("nonfinite age/identity maps")
        return (*stages, identity, age)

    def components(self, images):
        *_, identity, age = self.shortcuts(images)
        net = self.backbone.net
        embedding = net.bn2(identity)
        embedding = net.dropout(torch.flatten(embedding, 1))
        embedding = net.features(net.fc(embedding))
        return F.normalize(embedding, dim=1), identity, age

    def forward(self, images, *, return_age=False, return_shortcuts=False):
        if not isinstance(return_age, bool) or not isinstance(return_shortcuts, bool):
            raise ValueError("boolean return-mode flags required")
        if return_age and return_shortcuts:
            raise ValueError("choose one return mode")
        if return_shortcuts:
            return self.shortcuts(images)
        outputs = self.components(images)
        return outputs if return_age else outputs[0]
