"""TIP2021 AOFS generator / discriminator architecture reconstruction (ADAPTATION).

Primary source (cached, read-only):
  https://wrap.warwick.ac.uk/id/eprint/153968/7/WRAP-Age-oriented-face-synthesis-
  conditional-discriminator-pool-adversarial-triplet-2021.pdf
  sha256 7449b5fb96aaf8dcfaa65c29dded1d27a554f776dc6adbcaa098837b3d5cefb6
  Table II  generator, physical page 8 (printed page 7)
  Table III discriminator, physical page 8 (printed page 7)
  Eq.(3) CDP selected feature adversarial loss, physical page 6
  Eq.(8) image adversarial loss, physical page 7
  Sec. IV-C network architecture / Sec. IV-E settings, physical pages 8-9

Every plan field is explicit; the named adaptation factory also has declared
defaults, which are implementation choices rather than recovered author settings.
The module refuses an inconsistent plan before any
forward, and `GeneratorPlan.choice_metadata()` reports for each field whether it
is a literal source row (`source:...`) or a caller ADAPTATION (`adaptation...`).

Source-faithful rows kept exactly:
  * stem k=7 s=1 p=1, encoder k=3 s=2 p=1, residual conv k=3 (s=2 in table),
    decoder two k=3 s=2 p=1 transpose convs, InstanceNorm + ReLU on hidden
    layers, final decoder InstanceNorm + Tanh.
  * Table III image D: four k=3 s=2 p=1 InstanceNorm+LeakyReLU, then k=3 s=1
    p=1 with no norm/activation. Feature D: widths 128/64/32/16/1.

Explicit ADAPTATIONS (never claimed as source):
  * The literal Table II arithmetic is not an image-to-image map: the k7/s1/p1
    stem yields 124 and two stride-2 residual convs per block x6 bottom out at
    1x1, which two stride-2 transposes cannot restore; identity skip cannot
    bridge stride 2. `assert_literal_table_inconsistent()` proves this.
  * residual conv stride and per-block first-conv stride are caller-declared;
    the second residual conv stride is FIXED at 1 so the plan trace equals the
    forward. `residual_conv.stride != 1` is rejected.
  * skip type (`identity`/`projection`/`none`), projection geometry, decoder
    `output_padding`, label injection, channel widths, norm kind, affine flag,
    leaky slope, residual activation order and final-decoder normalisation are
    all explicit caller choices.
  * residual activation order: the table lists a second ReLU, but its position
    relative to an unspecified skip is unknown; both orders are adaptations.
  * CDP forward is selected-head-only; unselected heads and buffers untouched.

Random weights; synthetic shape/routing tests only. NOT trained, no MTFE, no
author weights, no published-system parity. Losses are NOT implemented here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

AGE_REGIME_TIP2021 = "tip2021_4groups_le30_31-40_41-50_51plus"
AGE_REGIME_CACON_5YEAR = "cacon_five_year_bins"
KNOWN_AGE_REGIMES = (AGE_REGIME_TIP2021, AGE_REGIME_CACON_5YEAR)
NORM_KINDS = ("instance2d", "batch2d", "none")
RESIDUAL_ACTIVATION_ORDERS = ("pre_add_relu", "post_add_relu")
SKIP_KINDS = ("identity", "projection", "none")
LABEL_INJECTIONS = ("add_projected",)
FINAL_DECODER_NORMS = ("instance", "none")
FEATURE_NORMS = ("instance1d", "layernorm", "batch1d", "none")
TABLE_II_RESIDUAL_BLOCKS = 6
METADATA_FIELDS = (
    "stem", "encoder", "residual_conv", "decoder", "residual_strides",
    "output_padding", "skip", "label_injection", "num_age_categories",
    "age_regime", "norm_kind", "final_decoder_norm", "residual_activation_order",
    "affine", "channels", "image_size", "residual_blocks", "out_channels",
)


class PlanError(ValueError):
    """Caller-declared plan is inconsistent or under-specified."""


class LiteralTableContradiction(PlanError):
    """The literal Table II arithmetic cannot be a valid image-to-image map."""


def _strict_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise PlanError(f"{name} must be a strict int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise PlanError(f"{name} must be >= {minimum}")
    return value


def _finite_positive(name: str, value: object) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise PlanError(f"{name} must be a real number")
    result = float(value)
    if not (result == result and result not in (float("inf"), float("-inf"))):
        raise PlanError(f"{name} must be finite")
    if result <= 0.0:
        raise PlanError(f"{name} must be strictly positive")
    return result


@dataclass(frozen=True)
class ConvSpec:
    kernel: int
    stride: int
    padding: int

    def __post_init__(self) -> None:
        _strict_int("kernel", self.kernel, minimum=1)
        _strict_int("stride", self.stride, minimum=1)
        _strict_int("padding", self.padding, minimum=0)
        if self.padding > self.kernel // 2:
            raise PlanError("padding must be <= kernel // 2 for a sane declared plan")


def conv_out(size: int, spec: ConvSpec) -> int:
    _strict_int("size", size, minimum=1)
    return (size + 2 * spec.padding - spec.kernel) // spec.stride + 1


def transpose_out(size: int, spec: ConvSpec, output_padding: int) -> int:
    _strict_int("size", size, minimum=1)
    _strict_int("output_padding", output_padding, minimum=0)
    if output_padding >= spec.stride:
        raise PlanError("output_padding must be smaller than stride for dilation-one transposes")
    return (size - 1) * spec.stride - 2 * spec.padding + spec.kernel + output_padding


TABLE_II_STEM = ConvSpec(7, 1, 1)
TABLE_II_ENCODER = ConvSpec(3, 2, 1)
TABLE_II_RESIDUAL_CONV = ConvSpec(3, 2, 1)
TABLE_II_DECODER = (ConvSpec(3, 2, 1), ConvSpec(3, 2, 1))


def literal_table_trace(image_size: int = 128,
                        residual_blocks: int = TABLE_II_RESIDUAL_BLOCKS) -> tuple[int, ...]:
    """Resolution trace of the literal Table II rows (no skip, output_padding 0)."""
    _strict_int("image_size", image_size, minimum=1)
    _strict_int("residual_blocks", residual_blocks, minimum=1)
    shapes = [image_size, conv_out(image_size, TABLE_II_STEM)]
    shapes.append(conv_out(shapes[-1], TABLE_II_ENCODER))
    for _ in range(residual_blocks):
        shapes.append(conv_out(shapes[-1], TABLE_II_RESIDUAL_CONV))
        shapes.append(conv_out(shapes[-1], TABLE_II_RESIDUAL_CONV))
    for spec in TABLE_II_DECODER:
        shapes.append(transpose_out(shapes[-1], spec, 0))
    return tuple(shapes)


def assert_literal_table_inconsistent(image_size: int = 128,
                                      residual_blocks: int = TABLE_II_RESIDUAL_BLOCKS
                                      ) -> tuple[int, ...]:
    """Raise LiteralTableContradiction with the computed literal trace."""
    trace = literal_table_trace(image_size, residual_blocks)
    if trace[1] != image_size or trace[-1] != image_size:
        raise LiteralTableContradiction(
            f"literal Table II is not an image-to-image map for {image_size}: trace={trace}; "
            f"k7/s1/p1 stem already changes resolution ({trace[0]}->{trace[1]}); stride-2 "
            "residual convs collapse to 1x1 and two stride-2 transposes cannot restore it; "
            "identity skip cannot bridge stride 2")
    return trace


@dataclass(frozen=True)
class GeneratorPlan:
    """Fully caller-declared generator contract; every field is mandatory."""

    image_size: int
    num_age_categories: int
    age_regime: str
    stem_channels: int
    encoder_channels: int
    residual_channels: int
    decoder_channels: int
    out_channels: int
    residual_blocks: int
    residual_strides: tuple[int, ...]
    stem: ConvSpec
    encoder: ConvSpec
    residual_conv: ConvSpec
    decoder: tuple[ConvSpec, ...]
    output_padding: tuple[int, ...]
    skip: str
    label_injection: str
    norm_kind: str
    final_decoder_norm: str
    residual_activation_order: str
    affine: bool

    # -- validation ---------------------------------------------------------
    def _check_scalars(self) -> None:
        _strict_int("image_size", self.image_size, minimum=8)
        _strict_int("num_age_categories", self.num_age_categories, minimum=2)
        _strict_int("residual_blocks", self.residual_blocks, minimum=1)
        _strict_int("out_channels", self.out_channels, minimum=1)
        for name in ("stem_channels", "encoder_channels", "residual_channels",
                     "decoder_channels"):
            _strict_int(name, getattr(self, name), minimum=1)
        if self.out_channels != 3:
            raise PlanError("RGB output requires out_channels == 3")
        if not isinstance(self.age_regime, str) or not self.age_regime:
            raise PlanError("age_regime must be an explicit nonempty descriptor")
        if self.age_regime == AGE_REGIME_TIP2021 and self.num_age_categories != 4:
            raise PlanError("the named TIP four-group regime requires exactly four categories")
        for name, value in (("skip", self.skip), ("label_injection", self.label_injection),
                            ("norm_kind", self.norm_kind),
                            ("final_decoder_norm", self.final_decoder_norm),
                            ("residual_activation_order", self.residual_activation_order)):
            if not isinstance(value, str):
                raise PlanError(f"{name} must be a declared string choice")
        if self.skip not in SKIP_KINDS:
            raise PlanError(f"skip must be one of {SKIP_KINDS}")
        if self.label_injection not in LABEL_INJECTIONS:
            raise PlanError(f"label_injection must be one of {LABEL_INJECTIONS}")
        if self.norm_kind not in NORM_KINDS:
            raise PlanError(f"norm_kind must be one of {NORM_KINDS}")
        if self.final_decoder_norm not in FINAL_DECODER_NORMS:
            raise PlanError(f"final_decoder_norm must be one of {FINAL_DECODER_NORMS}")
        if self.residual_activation_order not in RESIDUAL_ACTIVATION_ORDERS:
            raise PlanError(
                f"residual_activation_order must be one of {RESIDUAL_ACTIVATION_ORDERS}")
        if not isinstance(self.affine, bool):
            raise PlanError("affine must be an explicit bool adaptation choice")

    def _check_specs(self) -> None:
        if not isinstance(self.stem, ConvSpec) or not isinstance(self.encoder, ConvSpec):
            raise PlanError("stem and encoder must be ConvSpec")
        if not isinstance(self.residual_conv, ConvSpec):
            raise PlanError("residual_conv must be ConvSpec")
        if self.residual_conv.stride != 1:
            raise PlanError(
                "residual_conv.stride must be 1: ResidualBlock fixes the second conv "
                "stride to 1, so a different value would desynchronise trace and forward")
        if any(not isinstance(spec, ConvSpec) for spec in self.decoder):
            raise PlanError("every decoder entry must be ConvSpec")
        if len(self.decoder) < 1 or len(self.decoder) != len(self.output_padding):
            raise PlanError("one output_padding per decoder transpose required")
        if len(self.residual_strides) != self.residual_blocks:
            raise PlanError("residual_strides must have exactly residual_blocks entries")
        for stride in self.residual_strides:
            _strict_int("residual stride", stride, minimum=1)
            if stride not in (1, 2):
                raise PlanError("declared residual first-conv stride must be 1 or 2")
        for op in self.output_padding:
            _strict_int("output_padding", op, minimum=0)

    def resolution_trace(self) -> tuple[int, ...]:
        """Exact forward trace: first residual conv uses residual_strides, second uses 1."""
        self._check_specs()
        shapes = [self.image_size, conv_out(self.image_size, self.stem)]
        shapes.append(conv_out(shapes[-1], self.encoder))
        for stride in self.residual_strides:
            first = ConvSpec(self.residual_conv.kernel, stride, self.residual_conv.padding)
            shapes.append(conv_out(shapes[-1], first))
            shapes.append(conv_out(shapes[-1], self.residual_conv))
        for spec, pad in zip(self.decoder, self.output_padding, strict=True):
            shapes.append(transpose_out(shapes[-1], spec, pad))
        return tuple(shapes)

    def _check_skip_alignment(self) -> None:
        size = conv_out(conv_out(self.image_size, self.stem), self.encoder)
        for stride in self.residual_strides:
            first = ConvSpec(self.residual_conv.kernel, stride, self.residual_conv.padding)
            main = conv_out(size, first)
            final = conv_out(main, self.residual_conv)
            if self.skip == "projection":
                projected = skip_projection_alignment(
                    size, self.residual_conv, stride)[1]
                if projected != final:
                    raise PlanError(
                        f"projection skip geometry mismatch at size {size}: main {final} "
                        f"vs projection {projected}; adjust kernel/padding/stride")
            elif self.skip == "identity" and (stride != 1
                                              or self.encoder_channels != self.residual_channels):
                raise PlanError("identity skip requires stride 1 and equal channels")
            elif self.skip == "identity" and final != size:
                raise PlanError("identity skip geometry mismatch after both residual convolutions")
            size = final

    def module_shape_expectations(self, batch: int) -> dict[str, tuple[int, int, int]]:
        """Expected (C,H,W) per conv module, to be checked against forward hooks."""
        _strict_int("batch", batch, minimum=1)
        self.validate()
        sizes = [self.image_size]
        expect: dict[str, tuple[int, int, int]] = {}
        sizes.append(conv_out(sizes[-1], self.stem))
        expect["stem"] = (self.stem_channels, sizes[-1], sizes[-1])
        sizes.append(conv_out(sizes[-1], self.encoder))
        expect["encoder"] = (self.encoder_channels, sizes[-1], sizes[-1])
        for index, stride in enumerate(self.residual_strides):
            first = ConvSpec(self.residual_conv.kernel, stride, self.residual_conv.padding)
            sizes.append(conv_out(sizes[-1], first))
            expect[f"residual.{index}.conv1"] = (self.residual_channels, sizes[-1], sizes[-1])
            if self.skip == "projection":
                expect[f"residual.{index}.proj"] = (self.residual_channels, sizes[-1], sizes[-1])
            sizes.append(conv_out(sizes[-1], self.residual_conv))
            expect[f"residual.{index}.conv2"] = (self.residual_channels, sizes[-1], sizes[-1])
        for index, (spec, pad) in enumerate(zip(self.decoder, self.output_padding, strict=True)):
            sizes.append(transpose_out(sizes[-1], spec, pad))
            channels = self.out_channels if index == len(self.decoder) - 1 else self.decoder_channels
            expect[f"decoder.{index}"] = (channels, sizes[-1], sizes[-1])
        if tuple(sizes) != self.resolution_trace():
            raise PlanError("internal: module expectations disagree with resolution_trace")
        return expect

    def choice_metadata(self) -> dict[str, str]:
        """Per-field provenance: literal source row vs caller ADAPTATION."""
        def mark(flag: bool, source: str) -> str:
            return source if flag else "adaptation:not_in_source"
        meta = {
            "stem": mark(self.stem == TABLE_II_STEM, "source:TableII p8 k7s1p1"),
            "encoder": mark(self.encoder == TABLE_II_ENCODER, "source:TableII p8 k3s2p1"),
            "residual_conv": mark(
                self.residual_conv == ConvSpec(3, 1, 1),
                "source:TableII p8 k3 plus adaptation:second-stride-fixed-1"),
            "decoder": mark(self.decoder == TABLE_II_DECODER,
                            "source:TableII p8 two k3s2p1 transposes"),
            "residual_strides": mark(
                self.residual_strides == (TABLE_II_RESIDUAL_CONV.stride,) * self.residual_blocks,
                "source:TableII p8 residual conv stride 2"),
            "output_padding": "adaptation:not_in_source",
            "skip": "adaptation:not_in_source",
            "label_injection": "adaptation:not_in_source",
            "num_age_categories": "caller_declared:no_source_value",
            "age_regime": "caller_declared:no_source_value",
            "norm_kind": mark(self.norm_kind == "instance2d",
                              "source:TableII p8 InstanceNorm"),
            "final_decoder_norm": mark(self.final_decoder_norm == "instance",
                                       "source:TableII p8 InstanceNorm+Tanh"),
            "residual_activation_order": "adaptation:skip-relative activation order not printed",
            "affine": "adaptation:affine flag not printed",
            "channels": "adaptation:widths not printed",
            "image_size": mark(self.image_size == 128, "source:SecIV 128x128 images"),
            "residual_blocks": mark(self.residual_blocks == 6, "source:TableII six residual blocks"),
            "out_channels": "adaptation:RGB output interface",
        }
        return meta

    def assert_metadata_complete(self) -> dict[str, str]:
        meta = self.choice_metadata()
        missing = [k for k in METADATA_FIELDS if k not in meta]
        if missing:
            raise PlanError(f"metadata missing fields: {missing}")
        return meta

    def validate(self) -> None:
        self._check_scalars()
        self._check_specs()
        self.assert_metadata_complete()
        trace = self.resolution_trace()
        assert_norm_sizes_above_one(trace, norm_kind=self.norm_kind,
                                    final_decoder_norm=self.final_decoder_norm)
        if trace[-1] != self.image_size:
            raise PlanError(
                f"declared plan does not restore resolution: trace={trace} expected "
                f"{self.image_size}; declare decoder/output_padding/channel plan explicitly")
        self._check_skip_alignment()


def literal_table_plan(*, stem_channels: int, encoder_channels: int,
                       residual_channels: int, decoder_channels: int,
                       num_age_categories: int, age_regime: str = AGE_REGIME_TIP2021) -> GeneratorPlan:
    """The literal Table II plan; validate() is expected to reject it."""
    return GeneratorPlan(
        image_size=128, num_age_categories=num_age_categories, age_regime=age_regime,
        stem_channels=stem_channels, encoder_channels=encoder_channels,
        residual_channels=residual_channels, decoder_channels=decoder_channels,
        out_channels=3, residual_blocks=TABLE_II_RESIDUAL_BLOCKS,
        residual_strides=(TABLE_II_RESIDUAL_CONV.stride,) * TABLE_II_RESIDUAL_BLOCKS,
        stem=TABLE_II_STEM, encoder=TABLE_II_ENCODER, residual_conv=TABLE_II_RESIDUAL_CONV,
        decoder=TABLE_II_DECODER, output_padding=(0, 0), skip="identity",
        label_injection="add_projected", norm_kind="instance2d",
        final_decoder_norm="instance", residual_activation_order="pre_add_relu", affine=True)


def skip_projection_alignment(size: int, residual_conv: ConvSpec,
                             stride: int) -> tuple[int, int]:
    """Return (main, projected) spatial sizes for the first residual conv.

    A 1x1/stride projection matches the main path only when these are equal.
    """
    _strict_int("stride", stride, minimum=1)
    main = conv_out(size, ConvSpec(residual_conv.kernel, stride, residual_conv.padding))
    projected = conv_out(size, ConvSpec(1, stride, 0))
    return main, projected


def assert_norm_sizes_above_one(trace: Sequence[int], *, norm_kind: str,
                               final_decoder_norm: str) -> None:
    """Reject a plan whose InstanceNorm would see a spatial size of 1 or less."""
    if any(size <= 1 for size in trace) and (
            norm_kind != "none" or final_decoder_norm == "instance"):
        raise PlanError(
            f"every intermediate spatial size feeding InstanceNorm must exceed 1: {trace}")


def _norm_2d(kind: str, channels: int, affine: bool) -> nn.Module:
    if kind == "instance2d":
        return nn.InstanceNorm2d(channels, affine=affine)
    if kind == "batch2d":
        return nn.BatchNorm2d(channels, affine=affine)
    return nn.Identity()


TABLE_II_STEM_SAME_PADDING = ConvSpec(7, 1, 3)


def declared_adaptation_plan(*, stem_channels: int, encoder_channels: int,
                             residual_channels: int, decoder_channels: int,
                             num_age_categories: int, age_regime: str,
                             image_size: int = 128, out_channels: int = 3,
                             residual_blocks: int = TABLE_II_RESIDUAL_BLOCKS,
                             downsample_blocks: int = 1, skip: str = "projection",
                             norm_kind: str = "instance2d",
                             final_decoder_norm: str = "instance",
                             residual_activation_order: str = "pre_add_relu",
                             affine: bool = True, label_injection: str = "add_projected",
                             decoder: tuple[ConvSpec, ...] = TABLE_II_DECODER,
                             output_padding: tuple[int, ...] = (1, 1),
                             stem: ConvSpec = TABLE_II_STEM_SAME_PADDING,
                             encoder: ConvSpec = TABLE_II_ENCODER) -> GeneratorPlan:
    """Named adaptation defaults, not recovered author hyperparameters."""
    _strict_int("downsample_blocks", downsample_blocks, minimum=0)
    if downsample_blocks > residual_blocks:
        raise PlanError("downsample_blocks out of range")
    strides = tuple(2 if i < downsample_blocks else 1 for i in range(residual_blocks))
    plan = GeneratorPlan(
        image_size=image_size, num_age_categories=num_age_categories, age_regime=age_regime,
        stem_channels=stem_channels, encoder_channels=encoder_channels,
        residual_channels=residual_channels, decoder_channels=decoder_channels,
        out_channels=out_channels, residual_blocks=residual_blocks,
        residual_strides=strides, stem=stem, encoder=encoder,
        residual_conv=ConvSpec(3, 1, 1), decoder=decoder, output_padding=output_padding,
        skip=skip, label_injection=label_injection, norm_kind=norm_kind,
        final_decoder_norm=final_decoder_norm,
        residual_activation_order=residual_activation_order, affine=affine)
    plan.validate()
    return plan


class ResidualBlock(nn.Module):
    """Residual block with explicitly declared, unverified skip-relative ordering."""

    def __init__(self, in_channels: int, out_channels: int, conv: ConvSpec,
                 stride: int, skip: str, norm_kind: str, affine: bool,
                 activation_order: str) -> None:
        super().__init__()
        if skip not in SKIP_KINDS or norm_kind not in NORM_KINDS:
            raise PlanError("explicit supported skip and normalization required")
        if not isinstance(conv, ConvSpec) or conv.stride != 1:
            raise PlanError("second residual convolution requires a stride-one ConvSpec")
        if activation_order not in RESIDUAL_ACTIVATION_ORDERS:
            raise PlanError(f"activation_order must be one of {RESIDUAL_ACTIVATION_ORDERS}")
        self.stride = _strict_int("stride", stride, minimum=1)
        self.skip = skip
        self.activation_order = activation_order
        first = ConvSpec(conv.kernel, self.stride, conv.padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, first.kernel, first.stride,
                               padding=first.padding, bias=False)
        self.norm1 = _norm_2d(norm_kind, out_channels, affine)
        self.conv2 = nn.Conv2d(out_channels, out_channels, conv.kernel, 1,
                               padding=conv.padding, bias=False)
        self.norm2 = _norm_2d(norm_kind, out_channels, affine)
        if skip == "identity":
            if in_channels != out_channels or self.stride != 1:
                raise PlanError("identity skip requires equal channels and stride 1")
            self.proj: nn.Module | None = None
        elif skip == "projection":
            self.proj = nn.Conv2d(in_channels, out_channels, 1, self.stride, bias=False)
        else:
            self.proj = None

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        return F.relu(value, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._activate(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.skip == "none":
            return self._activate(out)
        skip = x if self.proj is None else self.proj(x)
        if self.activation_order == "pre_add_relu":
            out = self._activate(out)
            return out + skip
        return self._activate(out + skip)


class AOFSGenerator(nn.Module):
    """Label-conditioned generator over a validated caller plan (ADAPTATION)."""

    def __init__(self, plan: GeneratorPlan) -> None:
        super().__init__()
        plan.validate()
        self.plan = plan
        self.stem = nn.Conv2d(3, plan.stem_channels, plan.stem.kernel, plan.stem.stride,
                              padding=plan.stem.padding, bias=False)
        self.stem_norm = _norm_2d(plan.norm_kind, plan.stem_channels, plan.affine)
        self.encoder = nn.Conv2d(plan.stem_channels, plan.encoder_channels,
                                 plan.encoder.kernel, plan.encoder.stride,
                                 padding=plan.encoder.padding, bias=False)
        self.encoder_norm = _norm_2d(plan.norm_kind, plan.encoder_channels, plan.affine)
        blocks, in_channels = [], plan.encoder_channels
        for stride in plan.residual_strides:
            blocks.append(ResidualBlock(in_channels, plan.residual_channels,
                                        plan.residual_conv, stride, plan.skip,
                                        plan.norm_kind, plan.affine,
                                        plan.residual_activation_order))
            in_channels = plan.residual_channels
        self.residual = nn.ModuleList(blocks)
        self.label = nn.Linear(plan.num_age_categories, plan.residual_channels)
        decoder, norms = [], []
        channels = plan.residual_channels
        for index, (spec, pad) in enumerate(zip(plan.decoder, plan.output_padding, strict=True)):
            last = index == len(plan.decoder) - 1
            out_channels = plan.out_channels if last else plan.decoder_channels
            decoder.append(nn.ConvTranspose2d(channels, out_channels, spec.kernel, spec.stride,
                                              padding=spec.padding, output_padding=pad, bias=False))
            if not last:
                norms.append(_norm_2d(plan.norm_kind, out_channels, plan.affine))
            elif plan.final_decoder_norm == "instance":
                norms.append(nn.InstanceNorm2d(out_channels, affine=plan.affine))
            else:
                norms.append(nn.Identity())
            channels = out_channels
        self.decoder = nn.ModuleList(decoder)
        self.decoder_norm = nn.ModuleList(norms)

    def _check_input(self, source: torch.Tensor, age_labels: torch.Tensor) -> None:
        if (not isinstance(source, torch.Tensor) or source.ndim != 4
                or source.shape[1] != 3 or not source.is_floating_point()
                or source.numel() == 0 or not torch.isfinite(source).all()):
            raise PlanError("source must be a finite floating nonempty (B,3,H,W) tensor")
        if source.shape[2] != self.plan.image_size or source.shape[3] != self.plan.image_size:
            raise PlanError(f"source must be {self.plan.image_size}x{self.plan.image_size}")
        if (not isinstance(age_labels, torch.Tensor) or age_labels.ndim != 1
                or age_labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
                or age_labels.shape[0] != source.shape[0] or age_labels.device != source.device):
            raise PlanError("one integer age label per source image required")
        if (age_labels < 0).any() or (age_labels >= self.plan.num_age_categories).any():
            raise PlanError("age label outside declared num_age_categories")

    def forward(self, source: torch.Tensor, age_labels: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (synthesized_image, encoder_feature, one_hot_label)."""
        self._check_input(source, age_labels)
        x = F.relu(self.stem_norm(self.stem(source)), inplace=False)
        encoder_feature = F.relu(self.encoder_norm(self.encoder(x)), inplace=False)
        out = encoder_feature
        for block in self.residual:
            out = block(out)
        one_hot = F.one_hot(age_labels.long(), self.plan.num_age_categories).to(out.dtype)
        out = out + self.label(one_hot).view(out.shape[0], -1, 1, 1)
        last = len(self.decoder) - 1
        for index, conv in enumerate(self.decoder):
            out = conv(out)
            if index < last:
                out = F.relu(self.decoder_norm[index](out), inplace=False)
            elif self.plan.final_decoder_norm == "instance":
                out = torch.tanh(self.decoder_norm[index](out))
            else:
                out = torch.tanh(out)
        return out, encoder_feature, one_hot


class ImageDiscriminator(nn.Module):
    """Table III image-level patch discriminator; channels/slope/affine declared."""

    def __init__(self, channels: Sequence[int], *, leaky_slope: float,
                 affine: bool) -> None:
        super().__init__()
        if len(channels) != 6 or channels[0] != 3 or channels[-1] != 1:
            raise PlanError("image discriminator channels must be (3,h1,h2,h3,h4,1)")
        for name, value in zip(("in", "h1", "h2", "h3", "h4", "out"), channels, strict=True):
            _strict_int(f"image discriminator {name} channels", value, minimum=1)
        if not isinstance(affine, bool):
            raise PlanError("affine must be an explicit bool adaptation choice")
        self.channels = tuple(channels)
        self.leaky_slope = _finite_positive("leaky_slope", leaky_slope)
        self.affine = affine
        body, norms = [], []
        for index in range(4):
            body.append(nn.Conv2d(channels[index], channels[index + 1], 3, 2, 1, bias=False))
            norms.append(nn.InstanceNorm2d(channels[index + 1], affine=affine))
        self.body = nn.ModuleList(body)
        self.body_norm = nn.ModuleList(norms)
        self.final = nn.Conv2d(channels[4], channels[5], 3, 1, 1, bias=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if (not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[1] != 3
                or not image.is_floating_point() or image.numel() == 0
                or not torch.isfinite(image).all()):
            raise PlanError("image discriminator expects a finite floating nonempty (B,3,H,W)")
        height, width = image.shape[2:]
        for _ in range(4):
            height, width = (height + 1) // 2, (width + 1) // 2
            if height * width <= 1:
                raise PlanError("image discriminator InstanceNorm needs spatial area greater than one")
        out = image
        for conv, norm in zip(self.body, self.body_norm, strict=True):
            out = F.leaky_relu(norm(conv(out)), self.leaky_slope, inplace=False)
        return self.final(out)


class FeatureDiscriminator(nn.Module):
    """Table III feature-level discriminator; in_dim and norm are caller-declared."""

    WIDTHS = (128, 64, 32, 16, 1)

    def __init__(self, in_dim: int, *, norm: str, leaky_slope: float,
                 affine: bool, widths: Sequence[int] = WIDTHS) -> None:
        super().__init__()
        _strict_int("in_dim", in_dim, minimum=1)
        if tuple(widths) != tuple(self.WIDTHS):
            raise PlanError(f"Table III feature widths are {self.WIDTHS}")
        if norm not in FEATURE_NORMS:
            raise PlanError(f"norm must be one of {FEATURE_NORMS}")
        if not isinstance(affine, bool):
            raise PlanError("affine must be an explicit bool adaptation choice")
        self.in_dim = in_dim
        self.norm_kind = norm
        self.affine = affine
        self.leaky_slope = _finite_positive("leaky_slope", leaky_slope)
        dims = (in_dim, *self.WIDTHS)
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(4)])
        if norm == "instance1d":
            # ADAPTATION: authors print "Instance" for FC layers but FC-norm semantics
            # are unresolved. InstanceNorm1d(1) over a (B,1,width) view is one named
            # choice; it is never selected implicitly.
            self.norms: nn.ModuleList = nn.ModuleList(
                [nn.InstanceNorm1d(1, affine=affine) for _ in range(4)])
        elif norm == "layernorm":
            self.norms = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(4)])
        elif norm == "batch1d":
            self.norms = nn.ModuleList(
                [nn.BatchNorm1d(dims[i + 1], affine=affine) for i in range(4)])
        else:
            self.norms = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.head = nn.Linear(dims[4], dims[5])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or not features.is_floating_point() or features.numel() == 0
                or not torch.isfinite(features).all()):
            raise PlanError("feature discriminator expects finite floating nonempty (B,in_dim)")
        if features.shape[1] != self.in_dim:
            raise PlanError(f"feature dim {features.shape[1]} != declared {self.in_dim}")
        out = features
        for linear, norm in zip(self.layers, self.norms, strict=True):
            out = linear(out)
            if isinstance(norm, nn.InstanceNorm1d):
                out = norm(out.unsqueeze(1)).squeeze(1)
            elif isinstance(norm, (nn.BatchNorm1d, nn.LayerNorm)):
                out = norm(out)
            out = F.leaky_relu(out, self.leaky_slope, inplace=False)
        return self.head(out)


def _validate_labels(target_labels: torch.Tensor, *, categories: int,
                     batch: int | None) -> None:
    if (not isinstance(target_labels, torch.Tensor) or target_labels.ndim != 1
            or target_labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
            or target_labels.numel() == 0):
        raise PlanError("target_labels must be a nonempty rank-1 integer tensor")
    if batch is not None and target_labels.shape[0] != batch:
        raise PlanError("one integer target label per feature required")
    if (target_labels < 0).any() or (target_labels >= categories).any():
        raise PlanError("target label outside declared CDP categories")


class ConditionalDiscriminatorPool(nn.Module):
    """CDP with selected-head-only forward (Eq.3). Unselected heads are not run."""

    def __init__(self, in_dim: int, num_categories: int, *, norm: str,
                 leaky_slope: float, affine: bool) -> None:
        super().__init__()
        _strict_int("in_dim", in_dim, minimum=1)
        _strict_int("num_categories", num_categories, minimum=1)
        self.num_categories = num_categories
        self.in_dim = in_dim
        self.norm_kind = norm
        self.affine = affine
        self.heads = nn.ModuleList([
            FeatureDiscriminator(in_dim, norm=norm, leaky_slope=leaky_slope, affine=affine)
            for _ in range(num_categories)])

    def _check_features(self, features: torch.Tensor) -> None:
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or not features.is_floating_point() or features.numel() == 0
                or not torch.isfinite(features).all()):
            raise PlanError("CDP expects finite floating nonempty (B,in_dim) features")
        if features.shape[1] != self.in_dim:
            raise PlanError(f"feature dim {features.shape[1]} != declared {self.in_dim}")

    def forward(self, features: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """Run only the heads selected by target_labels; return (B,)."""
        self._check_features(features)
        batch = features.shape[0]
        _validate_labels(target_labels, categories=self.num_categories, batch=batch)
        if target_labels.device != features.device:
            raise PlanError("target_labels must be on the same device as features")
        # Reject the entire call before any earlier selected head can update BN.
        for category, head in enumerate(self.heads):
            count = int((target_labels == category).sum().item())
            if head.norm_kind == "batch1d" and count == 1:
                raise PlanError(f"BatchNorm CDP head {category}: at least 2 rows required, no fallback")
        outputs: list[torch.Tensor | None] = [None] * batch
        for category, head in enumerate(self.heads):
            index = (target_labels == category).nonzero(as_tuple=True)[0]
            if index.numel() == 0:
                continue
            if head.norm_kind == "batch1d" and index.numel() < 2:
                raise PlanError(
                    f"BatchNorm CDP head {category} selected with {index.numel()} row(s); "
                    "at least 2 rows required, no fallback")
            selected = head(features.index_select(0, index))
            for row, position in enumerate(index.tolist()):
                outputs[position] = selected[row]
        if any(value is None for value in outputs):
            raise PlanError("internal: every target label must select a head")
        return torch.stack([value for value in outputs if value is not None]).squeeze(-1)

    def forward_all(self, features: torch.Tensor) -> torch.Tensor:
        """Inspection only: runs every head; not the CDP training path."""
        self._check_features(features)
        return torch.cat([head(features) for head in self.heads], dim=1)

    def selected_head_indices(self, target_labels: torch.Tensor) -> tuple[int, ...]:
        _validate_labels(target_labels, categories=self.num_categories, batch=None)
        return tuple(sorted({int(v) for v in target_labels.tolist()}))
