"""Behavioral reimplementation of the MTLFace **Face Aging (FAS)** generator and
PatchDiscriminator networks.

Scope and provenance
--------------------
This module is a *behavioral reimplementation* of the symbols

  * ``TaskRouter``, ``ResidualBlock``, ``Upsample``, ``AgingModule``,
    ``PatchDiscriminator``                     -- from ``common/networks.py``
  * ``get_norm_layer``, ``group2feature``      -- from ``common/ops.py``

taken from the pinned upstream repository

    Hzzone/MTLFace
    commit 03ad57942e19ee28733b03a441765481f6606460
    https://github.com/Hzzone/MTLFace/blob/03ad57942e19ee28733b03a441765481f6606460/common/networks.py
    https://github.com/Hzzone/MTLFace/blob/03ad57942e19ee28733b03a441765481f6606460/common/ops.py

Behaviour is reproduced from the **code** at that commit, *not* from the paper.
Where the code and the paper disagree (routing sigma, unit count, upsampler
normalization, output activation, discriminator norm), the pinned code wins and
the disagreement is recorded in :data:`REFERENCE_BEHAVIOUR`.

This is **architecture-only** work.  It performs no training and makes no parity
or reproduction claim.  See :data:`CLAIM_FLAGS` and :func:`assert_architecture_only`.

Fail-closed validation
----------------------
All public entry points validate their conditioning tensor (``int64``, correct
device, aligned batch, values in ``[0, age_group)``), backbone shortcut channel /
spatial contract, BCHW shape and finiteness before computing.  Invalid input
raises rather than silently broadcasting or truncating.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

PINNED_REPOSITORY = "Hzzone/MTLFace"
PINNED_COMMIT = "03ad57942e19ee28733b03a441765481f6606460"

# sha256 of the exact raw files this module was reimplemented from.
PINNED_FILE_SHA256 = {
    "common/networks.py": "688866a75a426eab6ad3a5b8c4167075d094f6d07912f87f5875a27f278a1dfe",
    "common/ops.py": "b90586116049a10a2281f991152f718af3a0625b28b09110a1ecc46ed59aafc5",
}

# Behavioural facts taken from the pinned code.  ``paper`` records the paper's
# value/choice where it differs, so that no reader can mistake this module for a
# paper-default implementation.
REFERENCE_BEHAVIOUR: dict[str, Any] = {
    "source": "pinned code, not paper",
    "taskrouter.sigma": 0.1,
    "taskrouter.unit_count": 128,
    "taskrouter.conv_dim_formula": "int((age_group - (age_group - 1) * sigma) * unit_count)",
    "taskrouter.conv_dim_age_group_7": 819,
    "taskrouter.paper_conv_dim_note": "paper reports overlap sigma=1/8; substituting it into this code formula gives800 instead of819",
    "agingmodule.repeat_num": 4,
    "agingmodule.upsample_norm": "BatchNorm2d + PReLU (paper describes InstanceNorm2d + ReLU)",
    "agingmodule.output": "input_img + learned_residual, no final tanh",
    "agingmodule.x_id_channels": 512,
    "patchdiscriminator.norm_layer": "sn",
    "patchdiscriminator.repeat_num": 4,
    "patchdiscriminator.first_conv_spectral_norm": False,
    "patchdiscriminator.output_conv_spectral_norm": False,
    "downloaded_weights": "none",
}

# Deliberate, declared hardenings versus the literal pinned text.  These do not
# change any numeric behaviour on valid input; they only convert silent failure
# into a raised error.
DECLARED_HARDENINGS = (
    "get_norm_layer raises NotImplementedError for an unknown norm name; the "
    "pinned code returns the NotImplementedError class object instead of raising.",
    "TaskRouter validates condition dtype/device/range/batch instead of relying "
    "on index_select broadcasting.",
    "AgingModule.forward validates the backbone shortcut contract before use.",
    "Root requires nonempty floating BCHW router inputs and matching shortcut dtype/device.",
    "Root rejects invalid config types and silent discriminator config overrides.",
    "One-hot conditioning retains the condition device; differential evidence is CPU-only.",
)

# Scope flags.  Any consumer that wants to assert non-architecture claims must go
# through :func:`assert_architecture_only`, which fails for every unsupported key.
CLAIM_FLAGS: dict[str, bool] = {
    "architecture_only": True,
    "full_method": False,
    "training_reproduced": False,
    "training_parity": False,
    "scientific_evaluation_complete": False,
    "publication_ready": False,
}

# Keys that this module can never support truthfully.
_UNSUPPORTED_CLAIMS = (
    "full_method",
    "training_reproduced",
    "training_parity",
    "scientific_evaluation_complete",
    "publication_ready",
)


def claim_flags() -> dict[str, bool]:
    """Return a copy of the explicit claim flags for this architecture-only port."""
    return dict(CLAIM_FLAGS)


def assert_architecture_only(flags: dict[str, Any] | None = None) -> None:
    """Fail closed if any unsupported (training/parity/… ) claim is truthy."""
    flags = dict(CLAIM_FLAGS) if flags is None else dict(flags)
    if not flags.get("architecture_only", False):
        raise ValueError("this port must be marked architecture_only=True")
    bad = [k for k in _UNSUPPORTED_CLAIMS if flags.get(k)]
    if bad:
        raise ValueError(f"architecture-only port must not claim: {sorted(bad)}")


def provenance() -> dict[str, Any]:
    """Machine-readable provenance record for the result manifest."""
    return {
        "repository": PINNED_REPOSITORY,
        "commit": PINNED_COMMIT,
        "files": dict(PINNED_FILE_SHA256),
        "behaviour": dict(REFERENCE_BEHAVIOUR),
        "declared_hardenings": list(DECLARED_HARDENINGS),
        "claims": claim_flags(),
    }


# --------------------------------------------------------------------------- #
# Reference configuration
# --------------------------------------------------------------------------- #
# Pinned architecture constants that are *not* parameterized (they are fixed by
# the recognition backbone and by ``common/networks.py``).
PINNED_AGE_GROUP = 7
PINNED_BACKBONE_CHANNELS = dict(image=3, x_1=64, x_2=64, x_3=128, x_4=256, x_5=512)
PINNED_ID_CHANNELS = 512
PINNED_UPSAMPLE_CHANNELS = {  # (x_channels, in_channels, out_channels) per pinned code
    "up_1": (512, 512 + 256, 256),
    "up_2": (256, 256 + 128, 128),
    "up_3": (128, 128 + 64, 64),
    "up_4": (64, 64 + 3, 32),
}
PINNED_PATCHD_CONV_DIM = 64
PINNED_PATCHD_REPEAT_NUM = 4
PINNED_PATCHD_NORM_LAYER = "sn"


@dataclass(frozen=True)
class FasReferenceConfig:
    """Explicit architecture configuration.

    The default (``pinned``) config reproduces the pinned code exactly.  Any
    deviation must set ``test_only=True``; an unexplained deviation is rejected,
    so paper hyper-parameters can never be silently mixed with code state.
    """

    age_group: int = PINNED_AGE_GROUP
    sigma: float = 0.1
    unit_count: int = 128
    resblock_repeat: int = 4
    patchd_conv_dim: int = PINNED_PATCHD_CONV_DIM
    patchd_repeat_num: int = PINNED_PATCHD_REPEAT_NUM
    patchd_norm_layer: str = PINNED_PATCHD_NORM_LAYER
    output_activation: str = "none"
    test_only: bool = False
    note: str = "pinned MTLFace commit " + PINNED_COMMIT

    def __post_init__(self) -> None:
        for name in (
            "age_group",
            "unit_count",
            "resblock_repeat",
            "patchd_conv_dim",
            "patchd_repeat_num",
        ):
            _positive_int(getattr(self, name), name)
        _sigma(self.sigma)
        if not isinstance(self.test_only, bool):
            raise ValueError("test_only must be boolean")
        if self.age_group < 1:
            raise ValueError("age_group must be >= 1")
        if not 0.0 <= self.sigma < 1.0:
            raise ValueError("sigma must be in [0, 1)")
        if self.unit_count < 1:
            raise ValueError("unit_count must be >= 1")
        if self.resblock_repeat < 1:
            raise ValueError("resblock_repeat must be >= 1")
        if self.patchd_repeat_num < 1:
            raise ValueError("patchd_repeat_num must be >= 1")
        if self.output_activation != "none":
            raise ValueError(
                "pinned AgingModule returns input_img + residual with no activation; "
                "output_activation must be 'none'"
            )
        if not self.test_only:
            self._assert_is_pinned()

    def _assert_is_pinned(self) -> None:
        expected = dict(
            age_group=PINNED_AGE_GROUP,
            sigma=0.1,
            unit_count=128,
            resblock_repeat=4,
            patchd_conv_dim=PINNED_PATCHD_CONV_DIM,
            patchd_repeat_num=PINNED_PATCHD_REPEAT_NUM,
            patchd_norm_layer=PINNED_PATCHD_NORM_LAYER,
        )
        actual = dict(
            age_group=self.age_group,
            sigma=self.sigma,
            unit_count=self.unit_count,
            resblock_repeat=self.resblock_repeat,
            patchd_conv_dim=self.patchd_conv_dim,
            patchd_repeat_num=self.patchd_repeat_num,
            patchd_norm_layer=self.patchd_norm_layer,
        )
        if actual != expected:
            differing = {k: (actual[k], expected[k]) for k in expected if actual[k] != expected[k]}
            raise ValueError(
                "non-pinned configuration requires test_only=True; differing fields "
                f"(actual, pinned): {differing}"
            )

    @property
    def conv_dim(self) -> int:
        """Routing width, ``int((age_group - (age_group - 1) * sigma) * unit_count)``."""
        return int((self.age_group - (self.age_group - 1) * self.sigma) * self.unit_count)

    @classmethod
    def pinned(cls, age_group: int = PINNED_AGE_GROUP) -> FasReferenceConfig:
        return cls(age_group=age_group)

    @classmethod
    def tiny_test_only(
        cls,
        *,
        age_group: int = 4,
        sigma: float = 0.25,
        unit_count: int = 4,
        resblock_repeat: int = 2,
        patchd_conv_dim: int = 4,
        patchd_repeat_num: int = 2,
        note: str = "explicit test-only tiny config",
    ) -> FasReferenceConfig:
        """Explicit test-only dimensions.  Never a silent default."""
        return cls(
            age_group=age_group,
            sigma=sigma,
            unit_count=unit_count,
            resblock_repeat=resblock_repeat,
            patchd_conv_dim=patchd_conv_dim,
            patchd_repeat_num=patchd_repeat_num,
            test_only=True,
            note=note,
        )


# --------------------------------------------------------------------------- #
# Validation helpers (fail closed)
# --------------------------------------------------------------------------- #
def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"positive integer {name} required")


def _sigma(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value < 1
    ):
        raise ValueError("finite sigma in[0,1) required")


def validate_condition(
    condition: torch.Tensor,
    *,
    age_group: int,
    batch_size: int | None = None,
    device: torch.device | None = None,
    name: str = "condition",
) -> torch.Tensor:
    """Validate an age-group conditioning tensor.

    Requires ``int64``, 1-D, on the same device as the operand, batch-aligned and
    with all values in ``[0, age_group)``.
    """
    if not torch.is_tensor(condition):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(condition)!r}")
    if condition.dtype != torch.int64:
        raise TypeError(f"{name} must be int64, got {condition.dtype}")
    if condition.dim() != 1:
        raise ValueError(f"{name} must be 1-D (B,), got shape {tuple(condition.shape)}")
    if batch_size is not None and condition.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch {condition.shape[0]} does not match operand batch {batch_size}"
        )
    if device is not None and condition.device != device:
        raise ValueError(f"{name} device {condition.device} does not match operand device {device}")
    if condition.numel() > 0:
        lo = int(condition.min())
        hi = int(condition.max())
        if lo < 0 or hi >= age_group:
            raise ValueError(f"{name} values must be in [0, {age_group}); got [{lo}, {hi}]")
    return condition


def validate_bchw_finite(
    tensor: torch.Tensor,
    *,
    name: str,
    channels: int | None = None,
    spatial: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Validate a finite, 4-D BCHW tensor with optional channel/spatial contract."""
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")
    if tensor.dim() != 4:
        raise ValueError(f"{name} must be BCHW (4-D), got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point() or min(tensor.shape) < 1:
        raise ValueError(f"{name} must be nonempty floating BCHW")
    if channels is not None and tensor.shape[1] != channels:
        raise ValueError(f"{name} channels {tensor.shape[1]} != required {channels}")
    if spatial is not None and tuple(tensor.shape[2:]) != tuple(spatial):
        raise ValueError(f"{name} spatial {tuple(tensor.shape[2:])} != required {tuple(spatial)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def validate_shortcut_contract(
    *,
    input_img: torch.Tensor,
    x_1: torch.Tensor,
    x_2: torch.Tensor,
    x_3: torch.Tensor,
    x_4: torch.Tensor,
    x_5: torch.Tensor,
    x_id: torch.Tensor,
    x_age: torch.Tensor,
) -> int:
    """Validate the pinned backbone shortcut channel/spatial contract.

    Expected for an ``S x S`` input (backbone names given in the task brief):

    ============  ======================  ==============================
    tensor        channels                spatial
    ============  ======================  ==============================
    input_img     3                       S x S
    x_1 (initial) 64                      S x S
    x_2 (layer1)  64                      S/2 x S/2
    x_3 (layer2)  128                     S/4 x S/4
    x_4 (layer3)  256                     S/8 x S/8
    x_5 (layer4)  512                     S/16 x S/16
    x_id          512                     S/16 x S/16
    x_age         512                     S/16 x S/16
    ============  ======================  ==============================

    Returns the input spatial size ``S`` (must be a positive multiple of 16).
    """
    validate_bchw_finite(input_img, name="input_img", channels=3)
    size = int(input_img.shape[2])
    if int(input_img.shape[3]) != size:
        raise ValueError("input_img must be square")
    if size % 16 != 0:
        raise ValueError(f"input_img spatial size {size} must be a positive multiple of 16")
    contract = [
        ("x_1", x_1, PINNED_BACKBONE_CHANNELS["x_1"], (size, size)),
        ("x_2", x_2, PINNED_BACKBONE_CHANNELS["x_2"], (size // 2, size // 2)),
        ("x_3", x_3, PINNED_BACKBONE_CHANNELS["x_3"], (size // 4, size // 4)),
        ("x_4", x_4, PINNED_BACKBONE_CHANNELS["x_4"], (size // 8, size // 8)),
        ("x_5", x_5, PINNED_BACKBONE_CHANNELS["x_5"], (size // 16, size // 16)),
        ("x_id", x_id, PINNED_ID_CHANNELS, (size // 16, size // 16)),
        ("x_age", x_age, PINNED_ID_CHANNELS, (size // 16, size // 16)),
    ]
    batch = input_img.shape[0]
    for name, tensor, channels, spatial in contract:
        validate_bchw_finite(tensor, name=name, channels=channels, spatial=spatial)
        if tensor.shape[0] != batch:
            raise ValueError(f"{name} batch {tensor.shape[0]} != input_img batch {batch}")
        if tensor.dtype != input_img.dtype or tensor.device != input_img.device:
            raise ValueError(f"{name} dtype/device must match input_img")
    return size


# --------------------------------------------------------------------------- #
# common/ops.py symbols
# --------------------------------------------------------------------------- #
def group2onehot(groups: torch.Tensor, age_group: int) -> torch.Tensor:
    """Pinned ``group2onehot``: ``torch.eye(age_group)[groups.squeeze()]``.

    A scalar/one-element group yields ``(1, age_group)`` (the pinned
    ``unsqueeze(0)`` branch); a batch yields ``(B, age_group)``.
    """
    if not torch.is_tensor(groups):
        raise TypeError("groups must be a torch.Tensor")
    if groups.dtype != torch.int64:
        raise TypeError(f"groups must be int64, got {groups.dtype}")
    if groups.dim() == 0:
        groups = groups.reshape(1)
    validate_condition(groups, age_group=age_group)
    return F.one_hot(groups, num_classes=age_group).to(torch.float32)


def group2feature(group: torch.Tensor, age_group: int, feature_size: int) -> torch.Tensor:
    """Pinned ``group2feature``: one-hot broadcast to ``(B, age_group, f, f)``."""
    if feature_size < 1:
        raise ValueError("feature_size must be >= 1")
    validate_condition(group, age_group=age_group, name="group")
    onehot = group2onehot(group, age_group)
    return onehot.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, feature_size, feature_size)


def get_norm_layer(norm_layer: str, module: nn.Module, **kwargs: Any) -> nn.Module:
    """Pinned ``get_norm_layer`` dispatcher.

    ``'none'`` returns the module, ``'bn'``/``'in'`` wrap it in a
    ``Sequential(module, norm)``, and ``'sn'`` returns the module wrapped in
    ``torch.nn.utils.spectral_norm``.
    """
    if norm_layer == "none":
        return module
    if norm_layer == "bn":
        return nn.Sequential(module, nn.BatchNorm2d(module.out_channels, **kwargs))
    if norm_layer == "in":
        return nn.Sequential(module, nn.InstanceNorm2d(module.out_channels, **kwargs))
    if norm_layer == "sn":
        return nn.utils.spectral_norm(module, **kwargs)
    # Declared hardening: the pinned text returns the class object without
    # raising, which defers the failure; we fail closed immediately.
    raise NotImplementedError(f"unsupported norm_layer {norm_layer!r}")


# --------------------------------------------------------------------------- #
# common/networks.py symbols
# --------------------------------------------------------------------------- #
class TaskRouter(nn.Module):
    """Pinned ``TaskRouter``: per-sample routing mask over ``conv_dim`` units.

    ``conv_dim = int((age_group - (age_group - 1) * sigma) * unit_count)``.  Group
    ``i`` activates units ``[start_i, start_i + unit_count)`` with
    ``start_{i+1} = int(start_i + (1 - sigma) * unit_count)``, so adjacent groups
    overlap.  For age_group=7, sigma=0.1, unit_count=128 this gives conv_dim=819
    while the last unit (index 818) is never activated.
    """

    def __init__(self, unit_count: int, age_group: int, sigma: float) -> None:
        super().__init__()
        _positive_int(unit_count, "unit_count")
        _positive_int(age_group, "age_group")
        _sigma(sigma)
        if unit_count < 1:
            raise ValueError("unit_count must be >= 1")
        if age_group < 1:
            raise ValueError("age_group must be >= 1")
        if not 0.0 <= sigma < 1.0:
            raise ValueError("sigma must be in [0, 1)")
        self.unit_count = int(unit_count)
        self.age_group = int(age_group)
        self.sigma = float(sigma)

        conv_dim = int((age_group - (age_group - 1) * sigma) * unit_count)
        self.conv_dim = conv_dim
        mapping = torch.zeros((age_group, conv_dim))
        start = 0
        for i in range(age_group):
            mapping[i, start : start + unit_count] = 1
            start = int(start + (1 - sigma) * unit_count)
        self.register_buffer("_unit_mapping", mapping)

    def forward(self, inputs: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        validate_bchw_finite(inputs, name="router input", channels=self.conv_dim)
        if inputs.dim() < 2 or inputs.size(1) != self.conv_dim:
            raise ValueError(
                f"TaskRouter input channel {inputs.size(1) if inputs.dim() > 1 else '?'} "
                f"!= conv_dim {self.conv_dim}"
            )
        validate_condition(
            task_ids,
            age_group=self.age_group,
            batch_size=inputs.size(0),
            device=inputs.device,
            name="task_ids",
        )
        mask = torch.index_select(self._unit_mapping, 0, task_ids).unsqueeze(2).unsqueeze(3)
        return inputs * mask

    # -- introspection helpers (not part of the pinned surface) ------------- #
    def active_units(self, group: int) -> list[int]:
        """Indices activated for ``group`` (mask == 1)."""
        if not 0 <= group < self.age_group:
            raise ValueError("group out of range")
        return torch.nonzero(self._unit_mapping[group], as_tuple=False).flatten().tolist()

    def always_zero_units(self) -> list[int]:
        """Unit indices that no age group ever activates (pinned quirk)."""
        return torch.nonzero(self._unit_mapping.sum(0) == 0, as_tuple=False).flatten().tolist()


class ResidualBlock(nn.Module):
    """Pinned ``ResidualBlock``: two routed conv-BN stages with a PReLU residual."""

    def __init__(self, unit_count: int, age_group: int, sigma: float) -> None:
        super().__init__()
        conv_dim = int((age_group - (age_group - 1) * sigma) * unit_count)
        self.conv_dim = conv_dim
        self.conv1 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, 1, 1), nn.BatchNorm2d(conv_dim))
        self.router1 = TaskRouter(unit_count, age_group, sigma)
        self.conv2 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, 1, 1), nn.BatchNorm2d(conv_dim))
        self.router2 = TaskRouter(unit_count, age_group, sigma)
        self.relu1 = nn.PReLU(conv_dim)
        self.relu2 = nn.PReLU(conv_dim)

    def forward(self, inputs: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        x, task_ids = inputs[0], inputs[1]
        residual = x
        x = self.router1(self.conv1(x), task_ids)
        x = self.relu1(x)
        x = self.router2(self.conv2(x), task_ids)
        return {0: self.relu2(residual + x), 1: task_ids}


class Upsample(nn.Module):
    """Pinned ``Upsample``: BatchNorm2d + PReLU double-conv with a 1x1 shortcut."""

    def __init__(self, x_channels: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.x_channels = x_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.PReLU(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(x_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if x.size(2) < up.size(2):
            x = F.interpolate(input=x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.size(2) != up.size(2) or x.size(3) != up.size(3):
            raise ValueError(
                f"Upsample skip spatial {tuple(x.shape[2:])} != up spatial {tuple(up.shape[2:])} "
                "after optional x2 interpolation"
            )
        p = self.conv(torch.cat([x, up], dim=1))
        sc = self.shortcut(x)
        p = p + sc
        p2 = self.conv2(p)
        return p + p2


class AgingModule(nn.Module):
    """Pinned ``AgingModule`` generator.

    ``forward`` returns ``input_img + learned_residual`` with no output
    activation (no tanh).  Note the pinned code consumes ``x_2``, ``x_3``,
    ``x_4`` and ``x_id``; ``x_1``, ``x_5`` and ``x_age`` are part of the call
    signature/shortcut contract but do not affect the output.  This is preserved
    exactly and asserted by the test-suite.
    """

    def __init__(
        self,
        age_group: int = PINNED_AGE_GROUP,
        repeat_num: int | None = None,
        config: FasReferenceConfig | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = FasReferenceConfig.pinned(age_group)
        if repeat_num is not None:
            config = replace(config, resblock_repeat=repeat_num)
        if config.age_group != age_group:
            raise ValueError(
                f"age_group {age_group} does not match config age_group {config.age_group}"
            )
        if not config.test_only:
            config._assert_is_pinned()
        self.config = config
        self.age_group = config.age_group

        sigma = config.sigma
        unit_count = config.unit_count
        conv_dim = config.conv_dim
        self.conv_dim = conv_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(PINNED_ID_CHANNELS, conv_dim, 1, 1, 0),
            nn.BatchNorm2d(conv_dim),
            nn.PReLU(conv_dim),
        )
        self.router = TaskRouter(unit_count, config.age_group, sigma)
        layers = [
            ResidualBlock(unit_count, config.age_group, sigma)
            for _ in range(config.resblock_repeat)
        ]
        self.transform = nn.Sequential(*layers)
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_dim, PINNED_ID_CHANNELS, 1, 1, 0),
            nn.BatchNorm2d(PINNED_ID_CHANNELS),
            nn.PReLU(PINNED_ID_CHANNELS),
        )

        self.up_1 = Upsample(*PINNED_UPSAMPLE_CHANNELS["up_1"])
        self.up_2 = Upsample(*PINNED_UPSAMPLE_CHANNELS["up_2"])
        self.up_3 = Upsample(*PINNED_UPSAMPLE_CHANNELS["up_3"])
        self.up_4 = Upsample(*PINNED_UPSAMPLE_CHANNELS["up_4"])
        self.conv3 = nn.Conv2d(32, 3, 1, 1, 0)
        self.__init_weights()

    def forward(
        self,
        input_img: torch.Tensor,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        x_3: torch.Tensor,
        x_4: torch.Tensor,
        x_5: torch.Tensor,
        x_id: torch.Tensor,
        x_age: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        validate_condition(
            condition,
            age_group=self.age_group,
            batch_size=input_img.size(0),
            device=input_img.device,
            name="condition",
        )
        validate_shortcut_contract(
            input_img=input_img,
            x_1=x_1,
            x_2=x_2,
            x_3=x_3,
            x_4=x_4,
            x_5=x_5,
            x_id=x_id,
            x_age=x_age,
        )
        x_id = self.conv1(x_id)
        x_id = self.router(x_id, condition)
        inputs = {0: x_id, 1: condition}
        x = self.transform(inputs)[0]
        x = self.conv2(x)
        x = self.up_1(x, x_4)
        x = self.up_2(x, x_3)
        x = self.up_3(x, x_2)
        x = self.up_4(x, input_img)
        x = self.conv3(x)
        return input_img + x

    def residual(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Return the learned residual ``forward(...) - input_img`` (test helper)."""
        input_img = kwargs.get("input_img", args[0] if args else None)
        if input_img is None:
            raise TypeError("residual requires input_img")
        out = self.forward(*args, **kwargs)
        return out - input_img

    def __init_weights(self) -> None:
        for m in self.modules():
            if m.__class__.__name__.find("Conv") != -1:
                m.weight.data.normal_(0, 0.01)
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.data.fill_(0)


class PatchDiscriminator(nn.Module):
    """Pinned ``PatchDiscriminator`` with spectral normalization.

    The first conv and the final 1-channel output conv are **not** wrapped in
    spectral normalization; every intermediate conv is.  The first SN conv
    receives the concatenated age-group one-hot feature (``+ age_group`` input
    channels), which is how conditioning enters the discriminator.
    """

    def __init__(
        self,
        age_group: int | None = None,
        conv_dim: int | None = None,
        repeat_num: int | None = None,
        norm_layer: str | None = None,
        config: FasReferenceConfig | None = None,
    ) -> None:
        super().__init__()
        for value, name in (
            (age_group, "age_group"),
            (conv_dim, "conv_dim"),
            (repeat_num, "repeat_num"),
        ):
            if value is not None:
                _positive_int(value, name)
        config = config or FasReferenceConfig.pinned()
        for value, expected in (
            (age_group, config.age_group),
            (conv_dim, config.patchd_conv_dim),
            (repeat_num, config.patchd_repeat_num),
            (norm_layer, config.patchd_norm_layer),
        ):
            if value is not None and (isinstance(value, bool) or value != expected):
                raise ValueError("discriminator arguments must match explicit config")
        self.config = config
        age_group, conv_dim, repeat_num, norm_layer = (
            config.age_group,
            config.patchd_conv_dim,
            config.patchd_repeat_num,
            config.patchd_norm_layer,
        )
        self.age_group = int(age_group)
        self.conv_dim = int(conv_dim)
        self.repeat_num = int(repeat_num)
        self.norm_layer = norm_layer
        self.use_bias = True

        self.conv1 = nn.Conv2d(3, self.conv_dim, kernel_size=4, stride=2, padding=1)
        sequence: list[nn.Module] = []
        nf_mult = 1
        for n in range(1, self.repeat_num):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                get_norm_layer(
                    self.norm_layer,
                    nn.Conv2d(
                        self.conv_dim * nf_mult_prev + (self.age_group if n == 1 else 0),
                        self.conv_dim * nf_mult,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=self.use_bias,
                    ),
                ),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**self.repeat_num, 8)
        sequence += [
            get_norm_layer(
                self.norm_layer,
                nn.Conv2d(
                    self.conv_dim * nf_mult_prev,
                    self.conv_dim * nf_mult,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=self.use_bias,
                ),
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.conv_dim * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        validate_bchw_finite(inputs, name="discriminator inputs", channels=3)
        if (
            inputs.shape[2] != inputs.shape[3]
            or _expected_patchd_spatial(inputs.shape[2], self.repeat_num) < 1
        ):
            raise ValueError("square input large enough for PatchD required")
        validate_condition(
            condition,
            age_group=self.age_group,
            batch_size=inputs.size(0),
            device=inputs.device,
            name="condition",
        )
        x = F.leaky_relu(self.conv1(inputs), 0.2, inplace=True)
        condition_feature = group2feature(condition, self.age_group, x.size(2)).to(x)
        return self.main(torch.cat([x, condition_feature], dim=1))

    # -- introspection helper (not part of the pinned surface) -------------- #
    def spectral_norm_flags(self) -> list[bool]:
        """Per-``main``-child flag: is the child spectrally normalized?"""
        flags: list[bool] = []
        for child in self.main:
            if isinstance(child, nn.Conv2d):
                flags.append(hasattr(child, "weight_orig") or hasattr(child, "parametrizations"))
            else:
                flags.append(False)
        return flags


# --------------------------------------------------------------------------- #
# Bounded self-check / evidence entry point
# --------------------------------------------------------------------------- #
def _expected_patchd_spatial(size: int, repeat_num: int) -> int:
    """Mirror the pinned layer arithmetic to predict the PatchD output size."""
    s = (size - 4 + 2 * 1) // 2 + 1  # conv1
    for _ in range(1, repeat_num):
        s = (s - 4 + 2 * 1) // 2 + 1
    s = (s - 4 + 2 * 1) // 1 + 1
    s = (s - 4 + 2 * 1) // 1 + 1
    return s


def self_check(size: int = 112, seed: int = 0) -> dict[str, Any]:
    """Bounded, architecture-only forward pass at the pinned dimensions.

    Uses random weights (never real/downloaded weights) and a single thread.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    assert_architecture_only()
    config = FasReferenceConfig.pinned()
    generator = AgingModule(config=config).eval()
    discriminator = PatchDiscriminator(config=config).eval()

    if size % 16 != 0:
        raise ValueError("size must be a multiple of 16")
    half = size // 2
    inputs = dict(
        input_img=torch.randn(1, 3, size, size),
        x_1=torch.randn(1, 64, size, size),
        x_2=torch.randn(1, 64, half, half),
        x_3=torch.randn(1, 128, half // 2, half // 2),
        x_4=torch.randn(1, 256, half // 4, half // 4),
        x_5=torch.randn(1, 512, half // 8, half // 8),
        x_id=torch.randn(1, 512, half // 8, half // 8),
        x_age=torch.randn(1, 512, half // 8, half // 8),
        condition=torch.tensor([3], dtype=torch.int64),
    )
    with torch.no_grad():
        out = generator(**inputs)
        logits = discriminator(inputs["input_img"], inputs["condition"])
    return {
        "conv_dim": config.conv_dim,
        "generator_params": sum(p.numel() for p in generator.parameters()),
        "discriminator_params": sum(p.numel() for p in discriminator.parameters()),
        "output_shape": tuple(out.shape),
        "output_finite": bool(torch.isfinite(out).all()),
        "residual_nonzero": bool((out - inputs["input_img"]).abs().max() > 0),
        "logits_shape": tuple(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "expected_logits_spatial": _expected_patchd_spatial(size, config.patchd_repeat_num),
        "always_zero_units": generator.router.always_zero_units(),
        "claims": claim_flags(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Architecture-only self-check for the pinned MTLFace FAS networks."
    )
    parser.add_argument("--size", type=int, default=112, help="square input size (multiple of 16)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    report = self_check(size=args.size, seed=args.seed)
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
