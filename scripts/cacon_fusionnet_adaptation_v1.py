"""Declared ADAPTATION of the FusionNet (ICIP2018) network shape.

ADAPTATION ONLY - not a trained model, not a full reproduction and not a full MTFE.
No training loop or optimizer is implemented. Native/GPU execution is unvalidated here;
tiny synthetic CPU forward/backward checks are not scientific training evidence. The Gabor/AdaBoost patch
selector, a trained supervisor, authorised training data and any native training lineage are
absent.

FACTUAL (primary source; see PRIMARY_LOCATORS): a 96x96 gray-scale face plus five 24x24
patches fed sequentially (sec.3.1, sec.4.1); all Fig.2 blocks residual and B1-B5 contain
bottleneck layers; no feature reduction to B5; batch normalization before each convolution;
global average pooling then a single fully-connected layer (sec.3.2); Eq.5
x_i = B_i([x_{i-1}, s_i]) concatenates the previous block output with the i-th patch feature
map BEFORE that block's units; Eq.6 eliminates negatives, applies Softmax, normalizes to sum 1
and computes E(O) = sum_i p_i y_i; Fig.2 labels the main stream 32/64/96/128/160/192 at
spatial 24, so those widths are HARD architecture here and are not caller configuration.

DECLARED (never paper facts): stem kind, stem kernels, block depths, bottleneck widths;
BatchNorm->ReLU->conv placement (the source names no activation); bias=False following the
BN-before-conv statement; AdaptiveAvgPool2d(24) for the unpublished 96->24 mechanism; and the
output==concatenation-width policy. The 32-wide patch contribution is DERIVED_CONSTRAINT
arithmetic over the observed sequence under that policy, NOT uniquely dictated by the source.
The Fig.2 blue "7" (face) and "5" (patch) marks stay UNRESOLVED - candidate readings are
kernel size, stack depth or channel count, and a channel count contradicts the labelled
widths; they are never used as a width here.

Two hardening rules beyond the earlier port: (1) ``FusionNetAdaptationConfig`` is a FROZEN
dataclass whose ``validate`` is re-run before the network creates any module or draws any RNG;
(2) every public block checks the LIVE module dtype/device and every floating parameter and
buffer against the incoming tensors BEFORE any BatchNorm can run.

Primary source: H. Wang, X. Wei, V. Sanchez, C.-T. Li, "Fusion Network for Face-based Age
Estimation", ICIP 2018, pp. 2675-2679,
<https://wrap.warwick.ac.uk/120150/2/WRAP-fusion-network-face-based-age-estimation-Li-2018.pdf>
(sha256 1e60275a91317c27cecc5e3c7887ad0bfafaf5dd511c0a3719b7fd4d47e2d25b). Printed page =
physical PDF page - 1. Anchors: physical p3 = printed p2 (sec.3.1-3.2), physical p4 = printed
p3 (Fig.2 caption, Eq.4, Eq.5, Eq.6), physical p5 = printed p4 (sec.4.1: dlib 96x96
gray-scale, patches 24x24, PyTorch, SGD with momentum, batch 64, 200 epochs, lr 0.1 x0.1 every
50). Policy: experiments before a committee decision are permitted; the release/submission/
licence review gate is retained.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn
from torch.nn import functional as torch_functional

from scripts.cacon_mtfe_components_v1 import expectation_age_readout

__all__ = [
    "SOURCE_FACT", "DERIVED_CONSTRAINT", "DECLARED_CHOICE", "SOURCE_UNRESOLVED",
    "PRIMARY_LOCATORS", "FACTUAL_CONTRACT", "DECLARED_CHOICES", "UNRESOLVED_ITEMS",
    "STEM_KINDS", "OBSERVED_MAIN_STREAM_WIDTHS", "OBSERVED_SPATIAL", "PATCH_CONTRIBUTION",
    "FACE_CROP", "PATCH_CROP", "PATCH_INPUTS", "FACE_CHANNELS", "TRAINING_REFERENCE",
    "REVIEW_GATE", "FusionNetAdaptationConfig", "FusionNetBlock", "FusionNetAdaptation",
    "FusionNetInputs", "FusionNetOutputs", "crop_patches", "make_inputs",
    "expectation_age_readout",
]

SOURCE_FACT = "source_fact"
DERIVED_CONSTRAINT = "derived_constraint"
DECLARED_CHOICE = "declared_choice"
SOURCE_UNRESOLVED = "source_unresolved"

PRIMARY_LOCATORS: tuple[tuple[str, str, str], ...] = (
    ("paper", "ICIP 2018, pp. 2675-2679",
     "https://wrap.warwick.ac.uk/120150/2/WRAP-fusion-network-face-based-age-"
     "estimation-Li-2018.pdf"),
    ("pdf sha256", "saved source",
     "1e60275a91317c27cecc5e3c7887ad0bfafaf5dd511c0a3719b7fd4d47e2d25b"),
    ("physical p3 = printed p2, sec.3.1-3.2",
     "patch selection; all Fig.2 blocks residual; B1-B5 bottleneck layers; no feature "
     "reduction to B5; BN before each convolution; global average pooling + single FC",
     SOURCE_FACT),
    ("physical p4 = printed p3, Fig.2 caption",
     "five patches P1-P5 fed sequentially; final output from a single FC layer; main-stream "
     "width labels", SOURCE_FACT),
    ("physical p4 = printed p3, sec.3.2 Eq.5",
     "x_i = B_i([x_{i-1}, s_i]) for i in [1,5]; block-wise rather than layer-wise shortcuts",
     SOURCE_FACT),
    ("physical p4 = printed p3, sec.3.3 Eq.6",
     "negatives eliminated, Softmax, normalize to 1, E(O) = sum_i p_i y_i", SOURCE_FACT),
    ("physical p5 = printed p4, sec.4.1",
     "96x96 gray-scale crop, 24x24 patches, PyTorch, SGD with momentum, batch 64, 200 epochs, "
     "lr 0.1 dropping x0.1 every 50", SOURCE_FACT),
)

STEM_KINDS: tuple[str, ...] = ("plain_norm_conv", "residual_bottleneck")

FACTUAL_CONTRACT: tuple[tuple[str, str], ...] = (
    ("raw inputs", "96x96 gray-scale face plus five 24x24 gray-scale patches (sec.4.1)"),
    ("Eq.5", "each block consumes torch.cat([previous, patch_feature_map]) before its units"),
    ("normalization", "BatchNorm before each convolution in every branch (sec.3.2)"),
    ("residual bottleneck", "blocks are residual and contain bottleneck layers (sec.3.2)"),
    ("B5", "no feature reduction before global average pooling (sec.3.2)"),
    ("readout", "negative elimination, Softmax, normalize, single FC, Eq.6 expectation"),
    ("observed stream", "Fig.2 labels 32/64/96/128/160/192 at spatial 24, hard architecture"),
)

DECLARED_CHOICES: tuple[tuple[str, str], ...] = (
    ("patch geometry", "caller-supplied boxes; zero coordinates and bilinear resize are implementation choices"),
    ("stem kind", "whether the per-input stems are residual is unresolved in the source, so "
     f"the caller declares one of {STEM_KINDS}"),
    ("stem kernels", "the Fig.2 blue '7'/'5' marks are unresolved; kernels are caller-declared"),
    ("block depths", "bottleneck units per block are unpublished"),
    ("bottleneck widths", "1x1 reduction widths are unpublished"),
    ("activation", "BatchNorm->ReLU->conv placement; the source names no activation"),
    ("face reduction", "AdaptiveAvgPool2d(24) reaches the labelled spatial size; unpublished"),
    ("output width policy", "block output width equals its own concatenation width; the source "
     "states no output width, so this is declared, not a source fact"),
)

UNRESOLVED_ITEMS: tuple[tuple[str, str, str], ...] = (
    ("blue face-branch mark '7'", SOURCE_UNRESOLVED,
     "kernel size, stack depth or channel count; a channel count contradicts the labelled "
     "widths; never used as a width"),
    ("blue patch-branch mark '5'", SOURCE_UNRESOLVED,
     "kernel size, stack depth or channel count; never used as a width"),
    ("stem residual status", SOURCE_UNRESOLVED,
     "sec.3.2 says all Fig.2 blocks are residual while the stems are drawn as their own "
     "units; resolved only by the mandatory stem_kind declaration"),
    ("patch contribution width", DERIVED_CONSTRAINT,
     "the 32-wide patch contribution is arithmetic over the observed sequence under the "
     "declared output==concat policy, NOT a recovered fact"),
)

OBSERVED_MAIN_STREAM_WIDTHS: tuple[int, int, int, int, int, int] = (32, 64, 96, 128, 160, 192)
OBSERVED_SPATIAL = 24
PATCH_CONTRIBUTION = 32
FACE_CROP = 96
PATCH_CROP = 24
PATCH_INPUTS = 5
FACE_CHANNELS = 1
TRAINING_REFERENCE = (
    "sec.4.1 (physical p5 = printed p4): PyTorch, SGD with momentum, batch 64, 200 epochs, "
    "initial lr 0.1 dropping by 0.1 every 50 epochs. This module implements none of that."
)
REVIEW_GATE = (
    "Experiments before a committee decision are permitted. Retained gate: release, "
    "submission and licence review of the ICIP/WRAP source. No trained supervisor, no "
    "Gabor/AdaBoost selector, no authorised training data and no native training lineage "
    "exist here."
)


def _int(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int (bool is not an integer), got {type(value)}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    return _int(value, name=name, minimum=1)


def _nonnegative_int(value: object, *, name: str) -> int:
    return _int(value, name=name, minimum=0)


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number (bool refused), got {type(value)}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value}")
    return number


def _int_tuple(value: object, *, name: str, length: int) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of {length} ints, got {type(value)}")
    if len(value) != length:
        raise ValueError(f"{name} must have exactly {length} entries, got {len(value)}")
    return tuple(_positive_int(item, name=f"{name}[{i}]") for i, item in enumerate(value))


def _require_tensor(value: object, *, name: str, ndim: int) -> torch.Tensor:
    """Read-only raw-tensor check; idempotent, so it is safe on mutable input."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dims, got shape {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if any(size == 0 for size in value.shape):
        raise ValueError(f"{name} must be nonempty, got shape {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _require_matching(reference: torch.Tensor, other: torch.Tensor, *, name: str) -> None:
    if other.dtype != reference.dtype:
        raise TypeError(f"{name} dtype {other.dtype} must match the reference dtype "
                        f"{reference.dtype}")
    if other.device != reference.device:
        raise ValueError(f"{name} device {other.device} must match the reference device "
                         f"{reference.device}")


def _check_module_float_contract(module: nn.Module, reference: torch.Tensor, *,
                                 label: str) -> None:
    """Every floating parameter and buffer must match the reference dtype and device.

    Called BEFORE any BatchNorm can run, so a heterogeneous module fails closed with all
    running statistics untouched. Compares each entry to the live reference rather than to a
    cached value, which is what detects a module that is internally mixed.
    """
    for name, parameter in module.named_parameters():
        if not parameter.is_floating_point():
            raise TypeError(f"{label} parameter {name} must be floating point")
        _require_matching(reference, parameter, name=f"{label} parameter {name}")
    for name, buffer in module.named_buffers():
        if name.endswith("num_batches_tracked"):
            if buffer.dtype != torch.int64 or buffer.numel() != 1:
                raise TypeError(f"{label} buffer {name} must be an int64 scalar counter")
            if buffer.device != reference.device:
                raise ValueError(f"{label} buffer {name} must match the reference device")
        else:
            if not buffer.is_floating_point():
                raise TypeError(f"{label} buffer {name} must be floating point")
            _require_matching(reference, buffer, name=f"{label} buffer {name}")


class _BNConv(nn.Sequential):
    """BatchNorm -> ReLU -> conv (sec.3.2 order; the ReLU is a declared choice)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                      padding=kernel_size // 2, bias=False),
        )


class _BottleneckUnit(nn.Module):
    """1x1 reduce then 3x3 restore, residual add; both convolutions BN-first."""

    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int,
                 *, project_shortcut: bool):
        super().__init__()
        self.reduce = _BNConv(in_channels, bottleneck_channels, 1)
        self.expand = _BNConv(bottleneck_channels, out_channels, 3)
        self.shortcut = (_BNConv(in_channels, out_channels, 1) if project_shortcut
                         else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(self.reduce(x)) + self.shortcut(x)


class _Stem(nn.Module):
    """Per-input extraction from a raw image to the labelled 32-wide stem."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int, target_size: int,
                 *, pool: bool, stem_kind: str, bottleneck_channels: int | None):
        super().__init__()
        if stem_kind not in STEM_KINDS:
            raise ValueError(f"stem_kind must be one of {STEM_KINDS}, got {stem_kind!r}")
        self.norm_conv = _BNConv(in_channels, out_channels, kernel)
        self.residual = None
        if stem_kind == "residual_bottleneck":
            if bottleneck_channels is None:
                raise ValueError("stem_bottleneck_channels is required for a residual stem")
            self.residual = _BottleneckUnit(out_channels, bottleneck_channels, out_channels,
                                            project_shortcut=False)
        self.pool = nn.AdaptiveAvgPool2d(target_size) if pool else nn.Identity()
        self.out_channels = out_channels
        self.target_size = target_size
        self.stem_kind = stem_kind

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        x = self.norm_conv(raw)
        if self.residual is not None:
            x = self.residual(x)
        return self.pool(x)


class FusionNetBlock(nn.Module):
    """Residual bottleneck block consuming [x_{i-1}, s_i] (Eq.5).

    Public on purpose: a caller may build one block directly. The constructor checks its
    arguments (positive, bool-proof, and a bottleneck strictly narrower than the block width)
    and ``forward`` re-checks both inputs - channel count, spatial size, batch size, dtype,
    device and finiteness - and the live module dtype/device of every floating parameter and
    buffer BEFORE any BatchNorm runs.
    """

    def __init__(self, previous_channels: int, patch_channels: int,
                 bottleneck_channels: int, depth: int):
        super().__init__()
        self.previous_channels = _positive_int(previous_channels, name="previous_channels")
        self.patch_channels = _positive_int(patch_channels, name="patch_channels")
        self.concat_channels = self.previous_channels + self.patch_channels
        self.out_channels = self.concat_channels
        bottleneck = _positive_int(bottleneck_channels, name="bottleneck_channels")
        if bottleneck >= self.out_channels:
            raise ValueError(f"bottleneck_channels={bottleneck} must be strictly narrower than "
                             f"the block width {self.out_channels}; a bottleneck cannot be a "
                             "bottleneck otherwise")
        self.bottleneck_channels = bottleneck
        self.depth = _positive_int(depth, name="depth")
        self.units = nn.ModuleList([
            _BottleneckUnit(self.concat_channels if index == 0 else self.out_channels,
                            bottleneck, self.out_channels, project_shortcut=index == 0)
            for index in range(self.depth)
        ])

    def _check_inputs(self, previous: torch.Tensor, patch_feature_map: torch.Tensor) -> None:
        _require_tensor(previous, name="previous block output", ndim=4)
        _require_tensor(patch_feature_map, name="patch feature map", ndim=4)
        _require_matching(previous, patch_feature_map, name="patch feature map")
        if previous.shape[0] != patch_feature_map.shape[0]:
            raise ValueError("previous block output and patch feature map must share the batch "
                             f"dimension, got {previous.shape[0]} and "
                             f"{patch_feature_map.shape[0]}")
        if previous.shape[1] != self.previous_channels:
            raise ValueError(f"previous block output must have {self.previous_channels} "
                             f"channels, got {previous.shape[1]}")
        if patch_feature_map.shape[1] != self.patch_channels:
            raise ValueError(f"patch feature map must have {self.patch_channels} channels, "
                             f"got {patch_feature_map.shape[1]}")
        if previous.shape[-2:] != patch_feature_map.shape[-2:]:
            raise ValueError("Eq.5 concatenation needs one spatial size: previous block output "
                             f"is {tuple(previous.shape[-2:])} but the patch feature map is "
                             f"{tuple(patch_feature_map.shape[-2:])}; the observed stream stays "
                             f"{OBSERVED_SPATIAL}x{OBSERVED_SPATIAL}")

    def forward(self, previous: torch.Tensor, patch_feature_map: torch.Tensor) -> torch.Tensor:
        self._check_inputs(previous, patch_feature_map)
        _check_module_float_contract(self, previous, label="block")
        x = torch.cat([previous, patch_feature_map], dim=1)  # Eq.5 [x_{i-1}, s_i]
        self.last_concat_channels = x.shape[1]
        for unit in self.units:
            x = unit(x)
        if x.shape[1] != self.concat_channels:
            raise AssertionError("a block must not reduce features: B5 has no feature reduction")
        return x


@dataclasses.dataclass(frozen=True)
class FusionNetAdaptationConfig:
    """Mandatory declarations, enforced by a frozen dataclass with strict validation.

    ``__post_init__`` validates on construction, and ``FusionNetAdaptation`` calls
    ``validate`` again before it creates any module or draws any RNG, so a config corrupted
    after construction is rejected instead of silently building a wrong network. The observed
    widths and spatial size are NOT fields and are NOT configurable.
    """

    bottleneck_channels: tuple[int, int, int, int, int]
    block_depths: tuple[int, int, int, int, int]
    num_age_classes: int
    age_centers: tuple[float, ...]
    face_stem_kernel: int
    patch_stem_kernel: int
    stem_kind: str
    stem_bottleneck_channels: int | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Strict, idempotent revalidation of the LIVE field values.

        Re-normalizes each field through the same guards used at construction, so an in-place
        tamper is either corrected or rejected. ``object.__setattr__`` is the frozen-dataclass
        escape hatch used for that normalization.
        """

        def set_field(name: str, value: object) -> None:
            object.__setattr__(self, name, value)

        set_field("block_depths", _int_tuple(self.block_depths, name="block_depths",
                                             length=PATCH_INPUTS))
        set_field("bottleneck_channels",
                  _int_tuple(self.bottleneck_channels, name="bottleneck_channels",
                             length=PATCH_INPUTS))
        for index, (bottleneck, block_width) in enumerate(
                zip(self.bottleneck_channels, OBSERVED_MAIN_STREAM_WIDTHS[1:], strict=True)):
            if bottleneck >= block_width:
                raise ValueError(f"bottleneck_channels[{index}]={bottleneck} must be strictly "
                                 f"narrower than the labelled block width {block_width}")
        set_field("face_stem_kernel", _positive_int(self.face_stem_kernel,
                                                    name="face_stem_kernel"))
        set_field("patch_stem_kernel", _positive_int(self.patch_stem_kernel,
                                                     name="patch_stem_kernel"))
        for name in ("face_stem_kernel", "patch_stem_kernel"):
            if getattr(self, name) % 2 == 0:
                raise ValueError(f"{name} must be odd to preserve the spatial size with "
                                 "symmetric padding")
        if self.stem_kind not in STEM_KINDS:
            raise ValueError(f"stem_kind must be one of {STEM_KINDS}, got {self.stem_kind!r}; "
                             "the source does not settle whether the stems are residual")
        if self.stem_kind == "residual_bottleneck":
            if self.stem_bottleneck_channels is None:
                raise ValueError("stem_bottleneck_channels is required when stem_kind is "
                                 "'residual_bottleneck'; there is no default")
            stem_bottleneck = _positive_int(self.stem_bottleneck_channels,
                                            name="stem_bottleneck_channels")
            if stem_bottleneck >= OBSERVED_MAIN_STREAM_WIDTHS[0]:
                raise ValueError("stem_bottleneck_channels must be narrower than the labelled "
                                 f"stem width {OBSERVED_MAIN_STREAM_WIDTHS[0]}")
            set_field("stem_bottleneck_channels", stem_bottleneck)
        elif self.stem_bottleneck_channels is not None:
            raise ValueError("stem_bottleneck_channels must be None for a plain stem")
        num_age_classes = _positive_int(self.num_age_classes, name="num_age_classes")
        if num_age_classes < 2:
            raise ValueError("num_age_classes must be >= 2 to form a distribution")
        set_field("num_age_classes", num_age_classes)
        if not isinstance(self.age_centers, tuple):
            raise TypeError("age_centers must be a tuple with one center per age class")
        if len(self.age_centers) != num_age_classes:
            raise ValueError("age_centers must have one center per age class")
        centers = []
        for index, center in enumerate(self.age_centers):
            number = _finite_real(center, name=f"age_centers[{index}]")
            if number < 0:
                raise ValueError(f"age_centers[{index}] must be nonnegative")
            centers.append(number)
        set_field("age_centers", tuple(centers))

    def summary(self) -> dict[str, object]:
        return {
            "bottleneck_channels": self.bottleneck_channels,
            "block_depths": self.block_depths,
            "num_age_classes": self.num_age_classes,
            "age_centers": self.age_centers,
            "stem_kind": self.stem_kind,
            "stem_bottleneck_channels": self.stem_bottleneck_channels,
            "face_stem_kernel": self.face_stem_kernel,
            "patch_stem_kernel": self.patch_stem_kernel,
            "observed_main_stream_widths": OBSERVED_MAIN_STREAM_WIDTHS,
            "observed_spatial": OBSERVED_SPATIAL,
        }


class FusionNetInputs:
    """Raw 96x96 gray-scale faces and five raw 24x24 gray-scale patches."""

    __slots__ = ("face", "patches")

    def __init__(self, face: torch.Tensor, patches: tuple[torch.Tensor, ...]):
        self.face = face
        self.patches = patches
        self.validate()

    def validate(self) -> FusionNetInputs:
        _require_tensor(self.face, name="face", ndim=4)
        if self.face.shape[1] != FACE_CHANNELS:
            raise ValueError("sec.4.1 publishes gray-scale faces, so face must have 1 channel")
        if self.face.shape[-2:] != (FACE_CROP, FACE_CROP):
            raise ValueError(f"face must be {FACE_CROP}x{FACE_CROP}, got "
                             f"{tuple(self.face.shape[-2:])}")
        if not isinstance(self.patches, tuple):
            raise TypeError(f"patches must be a tuple of {PATCH_INPUTS} tensors, got "
                            f"{type(self.patches)}")
        if len(self.patches) != PATCH_INPUTS:
            raise ValueError(f"exactly {PATCH_INPUTS} patch inputs are required, got "
                             f"{len(self.patches)}")
        for index, patch in enumerate(self.patches):
            _require_tensor(patch, name=f"patches[{index}]", ndim=4)
            _require_matching(self.face, patch, name=f"patches[{index}]")
            if patch.shape[1] != FACE_CHANNELS:
                raise ValueError(f"patches[{index}] must be gray-scale")
            if patch.shape[-2:] != (PATCH_CROP, PATCH_CROP):
                raise ValueError(f"patches[{index}] must be {PATCH_CROP}x{PATCH_CROP}")
            if patch.shape[0] != self.face.shape[0]:
                raise ValueError("patches and face must share the batch dimension")
        return self


class FusionNetOutputs:
    """Named forward results."""

    __slots__ = ("feature_vector", "logits", "probabilities", "prediction")

    def __init__(self, feature_vector: torch.Tensor, logits: torch.Tensor,
                 probabilities: torch.Tensor, prediction: torch.Tensor):
        self.feature_vector = feature_vector
        self.logits = logits
        self.probabilities = probabilities
        self.prediction = prediction


def crop_patches(face: torch.Tensor,
                 boxes: tuple[tuple[int, int, int, int], ...]) -> tuple[torch.Tensor, ...]:
    """Crop the caller's boxes and resize to 24x24 (sec.4.1).

    Coordinates may be zero: ``(0, 0, h, w)`` is legal. Height and width must be positive and
    the box must be nonempty and inside the 96x96 crop. The boxes are caller data because the
    selected pixels are unpublished.
    """
    _require_tensor(face, name="face", ndim=4)
    if face.shape[1] != FACE_CHANNELS:
        raise ValueError("sec.4.1 publishes gray-scale faces, so face must have 1 channel")
    if face.shape[-2:] != (FACE_CROP, FACE_CROP):
        raise ValueError(f"face must be {FACE_CROP}x{FACE_CROP}")
    if not isinstance(boxes, tuple):
        raise TypeError("boxes must be a tuple of (top, left, height, width) tuples")
    if len(boxes) != PATCH_INPUTS:
        raise ValueError(f"exactly {PATCH_INPUTS} patch boxes are required, got {len(boxes)}")
    height, width = face.shape[-2:]
    patches = []
    for index, box in enumerate(boxes):
        if not isinstance(box, tuple) or len(box) != 4:
            raise TypeError(f"box {index} must be a (top, left, height, width) tuple")
        top = _nonnegative_int(box[0], name=f"box{index}[top]")
        left = _nonnegative_int(box[1], name=f"box{index}[left]")
        box_height = _positive_int(box[2], name=f"box{index}[height]")
        box_width = _positive_int(box[3], name=f"box{index}[width]")
        if top + box_height > height or left + box_width > width:
            raise ValueError(f"box {index} {(top, left, box_height, box_width)} falls outside "
                             f"the {height}x{width} face crop")
        region = face[:, :, top:top + box_height, left:left + box_width]
        patches.append(torch_functional.interpolate(region, size=(PATCH_CROP, PATCH_CROP),
                                                   mode="bilinear", align_corners=False))
    return tuple(patches)


def make_inputs(face: torch.Tensor,
                boxes: tuple[tuple[int, int, int, int], ...]) -> FusionNetInputs:
    """Crop raw patches from a raw face batch and wrap both as validated raw inputs."""
    return FusionNetInputs(face=face, patches=crop_patches(face, boxes))


class FusionNetAdaptation(nn.Module):
    """FusionNet shape adaptation: 96x96 gray face + five 24x24 patches.

    stems (32) -> five residual bottleneck blocks -> 64/96/128/160/192 at 24x24 -> global
    average pool -> single FC -> Eq.6 readout. Nothing sits between B5 and pooling.
    """

    def __init__(self, config: FusionNetAdaptationConfig):
        super().__init__()
        if not isinstance(config, FusionNetAdaptationConfig):
            raise TypeError("config must be a FusionNetAdaptationConfig")
        # Revalidate the live config BEFORE any module (and therefore any RNG draw) is created.
        config.validate()
        self.config = dataclasses.replace(config)
        self._construction_config_summary = self.config.summary()
        stem_width = OBSERVED_MAIN_STREAM_WIDTHS[0]
        self.face_stem = _Stem(FACE_CHANNELS, stem_width, config.face_stem_kernel,
                               OBSERVED_SPATIAL, pool=True, stem_kind=config.stem_kind,
                               bottleneck_channels=config.stem_bottleneck_channels)
        self.patch_stems = nn.ModuleList([
            _Stem(FACE_CHANNELS, PATCH_CONTRIBUTION, config.patch_stem_kernel, OBSERVED_SPATIAL,
                  pool=False, stem_kind=config.stem_kind,
                  bottleneck_channels=config.stem_bottleneck_channels)
            for _ in range(PATCH_INPUTS)
        ])
        blocks = []
        previous_channels = stem_width
        for index in range(PATCH_INPUTS):
            blocks.append(FusionNetBlock(previous_channels=previous_channels,
                                         patch_channels=PATCH_CONTRIBUTION,
                                         bottleneck_channels=config.bottleneck_channels[index],
                                         depth=config.block_depths[index]))
            previous_channels = blocks[-1].out_channels
        self.blocks = nn.ModuleList(blocks)
        if previous_channels != OBSERVED_MAIN_STREAM_WIDTHS[-1]:
            raise AssertionError(f"realized B5 width {previous_channels} must equal the "
                                 f"observed label {OBSERVED_MAIN_STREAM_WIDTHS[-1]}")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(previous_channels, config.num_age_classes)

    def realized_widths(self) -> tuple[int, ...]:
        return (self.face_stem.out_channels, *(block.out_channels for block in self.blocks))

    def _validate_entry(self, inputs: object) -> FusionNetInputs:
        """Fail closed on the CURRENT module and input state before any module runs."""
        if not isinstance(inputs, FusionNetInputs):
            raise TypeError("inputs must be a FusionNetInputs built from raw face and patches")
        self.config.validate()
        if self.config.summary() != self._construction_config_summary:
            raise ValueError("configuration changed after network construction")
        inputs.validate()
        _check_module_float_contract(self, inputs.face, label="network")
        for index, patch in enumerate(inputs.patches):
            _require_matching(inputs.face, patch, name=f"patches[{index}]")
        return inputs

    def features(self, inputs: FusionNetInputs) -> torch.Tensor:
        inputs = self._validate_entry(inputs)
        previous = self.face_stem(inputs.face)
        for index, block in enumerate(self.blocks):
            previous = block(previous, self.patch_stems[index](inputs.patches[index]))
        return torch.flatten(self.pool(previous), 1)

    def forward(self, inputs: FusionNetInputs) -> FusionNetOutputs:
        vector = self.features(inputs)
        logits = self.classifier(vector)
        probabilities = torch.softmax(torch.relu(logits), dim=1)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        prediction = expectation_age_readout(probabilities, age_centers=self.config.age_centers)
        return FusionNetOutputs(feature_vector=vector, logits=logits,
                                probabilities=probabilities, prediction=prediction)
