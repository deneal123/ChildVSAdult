"""Source-grounded MTFE feature-interface slice for the TIP2021 AOFS generator.

This implements only a geometric feature interface, NOT trained MTFE or the
full AOFS/CACon method. It implements only the largest executable slice that the
primary sources actually specify, and it refuses to invent the rest.

Primary sources (full URLs, cited by physical page as reviewed):
- TIP2021 AOFS, III-B Eq1 (physical p5) and Fig.3 (physical p6):
  https://wrap.warwick.ac.uk/id/eprint/153968/7/WRAP-Age-oriented-face-synthesis-conditional-discriminator-pool-adversarial-triplet-2021.pdf
  III-B states the decomposition is taken from [54] (=OE-CNN) and the backbone is
  ResNet-50, and that the [54] regression loss is replaced by an age regression
  model [56] (=FusionNet). Fig.3 states task-specific features are resized before
  the corresponding CDP / Adversarial Triplet head. The source does NOT publish the
  resize geometry, head shapes or the radial/direction routing; those are caller
  choices here.
- OE-CNN ECCV2018, section 2.1 Eq1 (physical p4): for feature x in R^n,
      xsphere = {r; theta_1..theta_n},  x = x_age * x_id,
      x_age = ||x||_2,  x_id = {x_1/||x||_2, ..., x_n/||x||_2}, ||x_id||_2 = 1.
  section 2.2 Eq2: Lage = (1/2M) sum_i ||f(nx_i) - z_i||_2^2, linear f(x)=k*x+b.
  section 2.2 Eq3: A-Softmax identity loss on x_id with scale s.
  https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/yitong_wang_Orthogonal_Deep_Features_ECCV_2018_paper.pdf
- FusionNet ICIP2018, sections 3.1-3.3 (physical p3-4) Eq5/Eq6:
  https://wrap.warwick.ac.uk/120150/2/WRAP-fusion-network-face-based-age-estimation-Li-2018.pdf
  Eq6 age expectation E(O) = sum_i p_i y_i after ReLU+softmax.

What this IS: the geometric radial/direction split of a caller-supplied feature
vector (TIP III-B Eq1 / OE-CNN Eq1), an explicitly-declared resize-head contract
(NOT recovered TIP architecture), explicit age/identity readout interfaces (OE-CNN
Eq2/Eq3, FusionNet Eq6), a frozen-backbone interface that preserves image gradients
on request, and explicit dimensions/scale checks for the two declared consumers
(feature-age CDP and the adversarial-triplet identity loss).

What this is NOT: a trained MTFE. Applying Eq1 to random or frozen features is a
geometric decomposition, not learned age/identity disentanglement. No optimizer
step, no training loop, no discriminator, no generator, no weights and no age/identity
supervisor are provided. The caller must supply architecture, pretrained weights,
training choices and real age/identity labels. This component is ALWAYS
'geometric_decomposition'; a trained-MTFE claim cannot be made from an asserted
boolean or digest string and would require independently verified native training
lineage outside this artifact.

Declared adaptations (NOT recovered author settings) are marked DECLARED wherever
they occur. No author default is invented: every numeric threshold (safe-division
minimum norm, resize targets/interpolation, A-Softmax scale, linear readout
coefficients) is a mandatory caller argument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PRIMARY_LOCATORS",
    "SafeDivPolicy",
    "ResizeHeadContract",
    "MTFEProvenance",
    "FrozenBackboneInterface",
    "MTFEFeatureInterface",
    "MTFEOutputs",
    "radial_direction_split",
    "unit_norm",
    "reconstruction_residual",
    "linear_age_readout",
    "expectation_age_readout",
    "direction_cosine_logits",
    "check_identity_dimension",
    "MISSING_TRAINED_MTFE_STAGES",
]

PRIMARY_LOCATORS: tuple[str, ...] = (
    "https://wrap.warwick.ac.uk/id/eprint/153968/7/WRAP-Age-oriented-face-synthesis-"
    "conditional-discriminator-pool-adversarial-triplet-2021.pdf",
    "https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/"
    "yitong_wang_Orthogonal_Deep_Features_ECCV_2018_paper.pdf",
    "https://wrap.warwick.ac.uk/120150/2/WRAP-fusion-network-face-based-age-"
    "estimation-Li-2018.pdf",
)

_REJECT = "reject"
_DECLARED_EPSILON = "declared_epsilon"
_SINGULAR_POLICIES = (_REJECT, _DECLARED_EPSILON)

# Exactly what this geometric component does NOT provide. A trained MTFE requires
# every stage below, none of which is implemented or assertable here.
MISSING_TRAINED_MTFE_STAGES: tuple[str, ...] = (
    "real age/identity-labelled training data with documented permissible governance",
    "ResNet-50 (or caller) MTFE architecture and its trained checkpoint provenance",
    "TIP-specific age supervisor [56]/FusionNet integration; not assumed OE-CNN Eq2",
    "source-faithful age/identity supervision and documented distinction from OE-CNN",
    "TIP-specific supervision/weight balance; OE-CNN and AOFS settings kept distinct",
    "MTFE optimizer/schedule/freeze policy from TIP IV-E; not AOFS margin conflation",
    "independently verified native training lineage (logs, checkpoints, seeds)",
)


def _finite_floating(name: str, value: torch.Tensor, ndim: int | None = None) -> None:
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.numel() == 0 or (ndim is not None and value.ndim != ndim)):
        raise ValueError(f"{name}: nonempty floating tensor with required rank")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name}: finite values required")


def _check_overflow(name: str, value: torch.Tensor) -> None:
    """Reject nonfinite results (overflow/underflow-to-inf) after computation."""
    if not torch.isfinite(value).all():
        raise ValueError(f"{name}: overflowed to nonfinite values")


def _require_same_dtype_device(name: str, reference: torch.Tensor,
                               *values: torch.Tensor) -> None:
    for value in values:
        if value.dtype != reference.dtype:
            raise ValueError(f"{name}: dtype must match {reference.dtype}")
        if value.device != reference.device:
            raise ValueError(f"{name}: device must match {reference.device}")


def _representable_positive(name: str, value, dtype: torch.dtype) -> float:
    """Require a positive scalar that is representable in `dtype`.

    This is an EXPLICIT declared representability policy: the value must be
    finite, > 0, and >= finfo(dtype).tiny, so a float16 policy cannot silently
    underflow to 0. It does not invent a source value.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name}: scalar required")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name}: finite real required")
    if value <= 0:
        raise ValueError(f"{name}: strictly positive required")
    if not dtype.is_floating_point:
        raise ValueError(f"{name}: floating dtype required")
    tiny = float(torch.finfo(dtype).tiny)
    if float(value) < tiny:
        raise ValueError(
            f"{name}: {value} is below finfo({dtype}).tiny={tiny}; "
            "not representable as a positive divisor in this dtype")
    if float(value) > float(torch.finfo(dtype).max):
        raise ValueError(f"{name}: divisor exceeds dtype finite range")
    return float(value)


def _positive_real(name: str, value) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name}: finite strictly positive real required")
    return float(value)


def _nonnegative_real(name: str, value) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name}: finite nonnegative real required")
    return float(value)


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name}: positive integer required")
    return value


@dataclass(frozen=True)
class SafeDivPolicy:
    """Policy for the division x_id = x / ||x||_2 in OE-CNN Eq1.

    The source assumes ||x||_2 > 0. 'reject' (faithful) raises on any norm not
    strictly above an explicit caller-supplied minimum. 'declared_epsilon' is an
    explicitly named ADAPTATION that clamps the denominator; it is never selected
    implicitly. The minimum is validated against the actual tensor dtype in
    radial_direction_split, because a value below finfo(dtype).tiny is not a
    representable positive divisor (e.g. 1e-8 underflows in float16).
    """

    singular_policy: Literal["reject", "declared_epsilon"]
    minimum_norm: float

    def __post_init__(self) -> None:
        if self.singular_policy not in _SINGULAR_POLICIES:
            raise ValueError(f"singular_policy must be one of {_SINGULAR_POLICIES}")
        _positive_real("minimum_norm", self.minimum_norm)

    def singular_mask(self, radial: torch.Tensor) -> torch.Tensor:
        return radial <= self.minimum_norm


def radial_direction_split(features: torch.Tensor, *, policy: SafeDivPolicy,
                           tolerance: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """OE-CNN Eq1 / TIP III-B Eq1: x -> (r = ||x||_2, u = x / ||x||_2).

    Returns radial (B,) and direction (B,R); direction is unit only on regular rows.
    Zero/near-zero norms are rejected or handled by an explicit declared policy.
    """
    _finite_floating("features", features, ndim=2)
    if not isinstance(policy, SafeDivPolicy):
        raise ValueError("explicit SafeDivPolicy required")
    _positive_real("tolerance", tolerance)
    minimum_norm = _representable_positive("minimum_norm", policy.minimum_norm,
                                           features.dtype)
    radial = torch.linalg.vector_norm(features, dim=-1)
    _check_overflow("radial", radial)
    singular = policy.singular_mask(radial)
    if singular.any():
        if policy.singular_policy == _REJECT:
            bad = int(singular.sum().item())
            raise ValueError(
                f"features: {bad} row(s) have ||x||_2 <= minimum_norm="
                f"{minimum_norm}; OE-CNN Eq1 division is undefined")
        denominator = radial.clamp_min(minimum_norm)
    else:
        denominator = radial
    direction = features / denominator[:, None]
    _check_overflow("direction", direction)
    # The unit-norm contract of Eq1 holds only where ||x||_2 > minimum_norm.
    # Under 'declared_epsilon' a clamped row is a finite zero/near-zero vector and
    # is explicitly NOT claimed to be unit; the caller sees singular_mask.
    # The tolerance is an explicit declared implementation policy, not a source value.
    regular = ~singular
    if regular.any():
        norm_error = (torch.linalg.vector_norm(
            direction[regular], dim=-1) - 1.0).abs().max()
        if float(norm_error.detach()) > tolerance:
            raise ValueError("direction: ||x_id||_2 != 1 beyond tolerance")
    return radial, direction


def unit_norm(direction: torch.Tensor, *, tolerance: float = 1e-4) -> torch.Tensor:
    """||direction||_2 per row; OE-CNN Eq1 requires this equal 1.

    The tolerance is an explicit declared implementation policy (not a source
    value), needed only for floating-point equality of the unit constraint.
    """
    _finite_floating("direction", direction, ndim=2)
    _positive_real("tolerance", tolerance)
    norms = torch.linalg.vector_norm(direction, dim=-1)
    _check_overflow("direction row norms", norms)
    if (norms - 1.0).abs().max() > tolerance:
        raise ValueError("direction rows must be unit L2 norm (OE-CNN Eq1)")
    return norms


def reconstruction_residual(features: torch.Tensor, radial: torch.Tensor,
                            direction: torch.Tensor) -> torch.Tensor:
    """x - r * x_id; Eq1 reconstruction contract, expected ~0."""
    _finite_floating("features", features, ndim=2)
    _finite_floating("radial", radial, ndim=1)
    _finite_floating("direction", direction, ndim=2)
    _require_same_dtype_device("radial", features, radial)
    _require_same_dtype_device("direction", features, direction)
    if radial.shape[0] != features.shape[0] or direction.shape != features.shape:
        raise ValueError("features (B,R), radial (B,), direction (B,R) required")
    if (radial < 0).any():
        raise ValueError("radial must be a nonnegative norm")
    residual = features - radial[:, None] * direction
    _check_overflow("reconstruction residual", residual)
    return residual


@dataclass(frozen=True)
class ResizeHeadContract:
    """TIP Fig.3 'we resize each set of task-specific features' contract.

    TIP does not publish the source/target spatial sizes, interpolation mode or
    align_corners, so all of them are mandatory caller arguments. Any use of this
    contract is flagged as a DECLARED adaptation, not a recovered author setting.
    """

    source_kind: Literal["direction_map", "radial_scalar"]
    target_height: int
    target_width: int
    mode: Literal["bilinear", "nearest"]
    align_corners: bool | None
    source_height: int | None = None
    source_width: int | None = None
    declared_adaptation: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if self.source_kind not in ("direction_map", "radial_scalar"):
            raise ValueError("source_kind must be direction_map or radial_scalar")
        if self.mode not in ("bilinear", "nearest"):
            raise ValueError("mode must be bilinear or nearest")
        _positive_int("target_height", self.target_height)
        _positive_int("target_width", self.target_width)
        if self.mode == "bilinear":
            if not isinstance(self.align_corners, bool):
                raise ValueError("bilinear requires explicit boolean align_corners")
        elif self.align_corners is not None:
            raise ValueError("nearest must not set align_corners")
        if self.source_kind == "direction_map":
            _positive_int("source_height", self.source_height)
            _positive_int("source_width", self.source_width)
        elif self.source_height is not None or self.source_width is not None:
            raise ValueError("radial_scalar has no spatial source shape")

    @property
    def source_cells(self) -> int | None:
        if self.source_kind == "radial_scalar":
            return None
        return self.source_height * self.source_width

    def _interpolate(self, maps: torch.Tensor) -> torch.Tensor:
        if maps.shape[-2:] == (self.target_height, self.target_width):
            return maps
        if self.mode == "bilinear":
            return F.interpolate(maps, size=(self.target_height, self.target_width),
                                 mode="bilinear", align_corners=self.align_corners)
        return F.interpolate(maps, size=(self.target_height, self.target_width),
                             mode="nearest")

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """direction_map: (B,R) -> (B,1,H,W); radial_scalar: (B,) -> (B,1,H,W).

        DECLARED caller adaptation: TIP Fig.3 does not publish this geometry, so
        any shape produced here is a caller choice, never recovered architecture.
        """
        _finite_floating("head input", tensor)
        if self.source_kind == "direction_map":
            _finite_floating("head input", tensor, ndim=2)
            if tensor.shape[1] != self.source_cells:
                raise ValueError(
                    "direction feature dim must equal source_height*source_width; "
                    "caller must supply the real backbone reshape")
            maps = tensor.reshape(tensor.shape[0], 1, self.source_height, self.source_width)
        else:
            if tensor.ndim != 1:
                raise ValueError("radial_scalar expects radial (B,)")
            maps = tensor.reshape(-1, 1, 1, 1).expand(
                -1, 1, self.target_height, self.target_width).contiguous()
        result = self._interpolate(maps)
        if result.shape[1:] != (1, self.target_height, self.target_width):
            raise ValueError("resize head produced unexpected shape")
        if result.shape[0] != tensor.shape[0] or result.dtype != tensor.dtype \
                or result.device != tensor.device:
            raise ValueError("resize head must preserve batch, dtype and device")
        _check_overflow("resize head output", result)
        return result


@dataclass(frozen=True)
class MTFEProvenance:
    """Provenance record that NEVER certifies a trained MTFE.

    A trained-MTFE claim cannot be established by this component from an asserted
    boolean or a digest string: mechanism is always 'geometric_decomposition'.
    Full trained-MTFE status requires independently verified native training
    lineage (real training data, optimizer schedule, checkpoints) established
    outside this artifact, so no claim field is accepted here.
    """

    source_locators: tuple[str, ...]
    backbone_id: str
    feature_dim: int
    declared_adaptations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_locators:
            raise ValueError("at least one primary source locator required")
        if any(not str(loc).startswith("https://") for loc in self.source_locators):
            raise ValueError("source locators must be full primary https URLs")
        if not isinstance(self.backbone_id, str) or not self.backbone_id:
            raise ValueError("backbone_id required")
        _positive_int("feature_dim", self.feature_dim)
        if any(not isinstance(a, str) or not a for a in self.declared_adaptations):
            raise ValueError("declared_adaptations must be nonempty strings")

    @property
    def mechanism(self) -> str:
        return "geometric_decomposition"

    @property
    def trained_status(self) -> str:
        return "provenance_unverified"


class FrozenBackboneInterface:
    """Frozen image->feature backbone used as the TIP III-B MTFE front end.

    Ownership: the CALLER retains ownership of `module`; this wrapper keeps a
    borrowed reference only (it does not copy weights and does not move devices).
    The caller must pre-freeze every parameter and put every submodule in eval
    mode; both are re-verified on EVERY forward, so a parameter unfrozen (or a
    submodule switched to train) after construction raises instead of silently
    updating BN running statistics or applying dropout.

    On request, the image gradient graph is preserved so a downstream
    synthesized-image loss can reach the input even though weights are frozen.
    """

    def __init__(self, module: nn.Module, *, feature_dim: int, input_rank: int,
                 preserve_input_grad: bool) -> None:
        if not isinstance(module, nn.Module):
            raise ValueError("module must be a torch.nn.Module")
        _positive_int("feature_dim", feature_dim)
        if type(preserve_input_grad) is not bool:
            raise ValueError("explicit boolean preserve_input_grad required")
        if type(input_rank) is not int or input_rank not in (2, 4):
            raise ValueError("input_rank must explicitly declare 2 (features) or 4 (BCHW images)")
        self.input_rank = input_rank
        self.owned_by_caller = True
        self.module = module
        self.feature_dim = feature_dim
        self.preserve_input_grad = preserve_input_grad
        self._assert_frozen_and_eval()

    def _assert_frozen_and_eval(self) -> None:
        for name, parameter in self.module.named_parameters():
            if parameter.requires_grad:
                raise ValueError(
                    f"backbone parameter {name!r} must be caller-frozen "
                    "(requires_grad=False) before use")
        for name, submodule in self.module.named_modules():
            if submodule.training:
                raise ValueError(
                    f"backbone submodule {name or '<root>'!r} must be in eval() "
                    "mode; train mode would update BN statistics/apply dropout")
        for name, buffer in self.module.named_buffers():
            if buffer.is_floating_point() and not torch.isfinite(buffer).all():
                raise ValueError(f"backbone buffer {name!r} must be finite")

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor) or not images.is_floating_point():
            raise ValueError("images: floating tensor required")
        if images.ndim != self.input_rank or images.numel() == 0:
            raise ValueError(f"images: rank-{self.input_rank} nonempty input required")
        _finite_floating("images", images, ndim=self.input_rank)
        self._assert_frozen_and_eval()
        if self.preserve_input_grad:
            if not images.requires_grad:
                raise ValueError("preserve_input_grad=True requires images.requires_grad")
            features = self.module(images)
            if not isinstance(features, torch.Tensor):
                raise ValueError("backbone must return a tensor")
            if not features.requires_grad:
                raise RuntimeError(
                    "frozen backbone cut the input graph; do not call under no_grad")
        else:
            with torch.no_grad():
                features = self.module(images)
        if not isinstance(features, torch.Tensor) or not features.is_floating_point():
            raise ValueError("backbone must return a floating tensor")
        if features.ndim != 2:
            raise ValueError("backbone output must be rank-2 (B, feature_dim)")
        if features.shape[0] != images.shape[0] or features.shape[1] != self.feature_dim:
            raise ValueError("backbone output must be (B, feature_dim)")
        _finite_floating("backbone features", features, ndim=2)
        return features


@dataclass
class MTFEOutputs:
    """Exact tensors handed to the declared consumers.

    radial_age (B,): age-specific scalar r; consumed by the feature-age CDP path
        and by the age readout. Scale is the raw feature norm.
    direction_identity (B,R): unit x_id (OE-CNN Eq1); consumed by the adversarial
        triplet loss (canonical euclidean needs matching R across banks).
    cdp_age_map (B,1,Hc,Wc): declared resize of the age-specific feature.
    atl_identity (B,R): identity feature for the triplet loss.
    atl_identity_map (B,1,Ha,Wa) or None: declared resize of the identity feature.
    identity_logits (B,C) or None: s * cos(theta) pre-softmax (OE-CNN Eq3 head input).
    """

    features: torch.Tensor
    radial_age: torch.Tensor
    direction_identity: torch.Tensor
    cdp_age_map: torch.Tensor
    atl_identity: torch.Tensor
    atl_identity_map: torch.Tensor | None = None
    identity_logits: torch.Tensor | None = None


class MTFEFeatureInterface:
    """Composes frozen backbone + Eq1 split + declared resize/consumer contract."""

    def __init__(self, backbone: FrozenBackboneInterface, *, policy: SafeDivPolicy,
                 provenance: MTFEProvenance, cdp_head: ResizeHeadContract,
                 atl_head: ResizeHeadContract | None = None,
                 identity_weight: torch.Tensor | None = None,
                 identity_scale: float | None = None) -> None:
        if not isinstance(backbone, FrozenBackboneInterface):
            raise ValueError("FrozenBackboneInterface required")
        if not isinstance(policy, SafeDivPolicy):
            raise ValueError("SafeDivPolicy required")
        if not isinstance(provenance, MTFEProvenance):
            raise ValueError("MTFEProvenance required")
        if provenance.feature_dim != backbone.feature_dim:
            raise ValueError("provenance.feature_dim must match backbone feature_dim")
        if not isinstance(cdp_head, ResizeHeadContract):
            raise ValueError("ResizeHeadContract required")
        if cdp_head.source_kind == "direction_map":
            raise ValueError(
                "cdp_head must take the age-specific radial scalar; the identity "
                "resize contract belongs to atl_head")
        if atl_head is not None and not isinstance(atl_head, ResizeHeadContract):
            raise ValueError("atl_head must be a ResizeHeadContract")
        if atl_head is not None and atl_head.source_kind != "direction_map":
            raise ValueError("atl_head must take the identity direction map")
        if (identity_weight is None) != (identity_scale is None):
            raise ValueError("identity_weight and identity_scale must be given together")
        if identity_weight is not None:
            _finite_floating("identity_weight", identity_weight, ndim=2)
            if identity_weight.shape[1] != backbone.feature_dim:
                raise ValueError("identity weight must be (C, feature_dim)")
            if identity_weight.requires_grad:
                raise ValueError("frozen MTFE proposal forbids trainable identity head")
            if _finite_real("identity_scale", identity_scale) <= 0:
                raise ValueError("identity_scale must be strictly positive")
        self.backbone = backbone
        self.policy = policy
        self.provenance = provenance
        self.cdp_head = cdp_head
        self.atl_head = atl_head
        self.identity_weight = identity_weight
        self.identity_scale = identity_scale

    def split(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return radial_direction_split(features, policy=self.policy)

    def forward(self, images: torch.Tensor) -> MTFEOutputs:
        features = self.backbone(images)
        radial, direction = self.split(features)
        cdp_age_map = self.cdp_head.apply(radial)
        atl_map = None if self.atl_head is None else self.atl_head.apply(direction)
        logits = None
        if self.identity_weight is not None:
            logits = direction_cosine_logits(direction, self.identity_weight,
                                             scale=self.identity_scale)
        return MTFEOutputs(
            features=features,
            radial_age=radial,
            direction_identity=direction,
            cdp_age_map=cdp_age_map,
            atl_identity=direction,
            atl_identity_map=atl_map,
            identity_logits=logits,
        )


def linear_age_readout(radial: torch.Tensor, *, slope, intercept) -> torch.Tensor:
    """OE-CNN Eq2 mapping f(nx) = k * nx + b (k, b caller-supplied, no default).

    Coefficients must be finite real scalars (bool is refused); the result must
    stay finite in the radial tensor's dtype or an overflow is raised.
    """
    _finite_floating("radial", radial, ndim=1)
    if (radial < 0).any():
        raise ValueError("radial must be a nonnegative norm")
    k = _finite_real("slope", slope)
    b = _finite_real("intercept", intercept)
    out = radial * k + b
    _check_overflow("linear age readout", out)
    return out


def _finite_real(name: str, value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name}: scalar required")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real) \
            or not math.isfinite(value):
        raise ValueError(f"{name}: finite real scalar required")
    return float(value)


def expectation_age_readout(probabilities: torch.Tensor, *,
                            age_centers: tuple[float, ...],
                            require_ordered_centers: bool = False) -> torch.Tensor:
    """FusionNet Eq6 E(O) = sum_i p_i y_i.

    DECLARED adaptation boundary: FusionNet builds p_i with ReLU + softmax; this
    function requires the caller to supply an already-normalized probability
    distribution and the real age centers. Age centers must be finite,
    nonnegative numbers (bool refused); if `require_ordered_centers` is set they
    must be strictly increasing. Output must stay finite in the input dtype.
    """
    _finite_floating("probabilities", probabilities, ndim=2)
    if type(require_ordered_centers) is not bool:
        raise ValueError("explicit boolean require_ordered_centers required")
    if not age_centers or len(age_centers) != probabilities.shape[1]:
        raise ValueError("one age center per probability column required")
    centers = torch.tensor([_finite_real("age center", c) for c in age_centers],
                           dtype=probabilities.dtype, device=probabilities.device)
    if not torch.isfinite(centers).all():
        raise ValueError("finite age centers required in the probability dtype")
    if (centers < 0).any():
        raise ValueError("age centers must be nonnegative")
    if require_ordered_centers and not bool((centers[1:] > centers[:-1]).all()):
        raise ValueError("age centers must be strictly increasing when ordered")
    if (probabilities < 0).any():
        raise ValueError("probabilities must be nonnegative")
    row_sums = probabilities.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
        raise ValueError("probabilities must be normalized per row")
    out = probabilities @ centers
    _check_overflow("expectation age readout", out)
    return out


def direction_cosine_logits(direction: torch.Tensor, weight: torch.Tensor, *,
                            scale, tolerance: float = 1e-4) -> torch.Tensor:
    """OE-CNN Eq3 angular head input s * cos(theta); cos from unit x_id and
    normalized nonzero class-weight rows.

    The integer margin m and the modified cos(m*theta) term of Eq3 belong to the
    identity loss, not to this feature interface; only the scale factor is fixed
    here and it is caller-supplied (no invented default). Each class-weight row is
    normalized to unit norm before the dot product, so a nonunit weight row does
    not silently rescale its cosine. Zero/overflowing weight norms and incompatible
    dtype/device are refused, and the output must be finite.
    """
    _positive_real("tolerance", tolerance)
    unit_norm(direction, tolerance=tolerance)
    _finite_floating("weight", weight, ndim=2)
    _require_same_dtype_device("weight", direction, weight)
    if weight.shape[1] != direction.shape[1]:
        raise ValueError("weight must be (C, feature_dim)")
    s = _finite_real("scale", scale)
    if s <= 0:
        raise ValueError("scale must be strictly positive")
    norms = torch.linalg.vector_norm(weight, dim=-1, keepdim=True)
    _check_overflow("weight row norms", norms)
    if (norms <= tolerance).any():
        raise ValueError(
            "weight: every class-weight row must have ||w||_2 > tolerance "
            "(zero/near-zero direction is undefined)")
    unit_weight = weight / norms
    _check_overflow("normalized weight", unit_weight)
    out = s * (direction @ unit_weight.T)
    _check_overflow("cosine logits", out)
    return out


def check_identity_dimension(direction: torch.Tensor, *, expected_dim: int) -> None:
    """Adversarial-triplet identity banks must share a feature dimension."""
    _finite_floating("direction", direction, ndim=2)
    _positive_int("expected_dim", expected_dim)
    if direction.shape[1] != expected_dim:
        raise ValueError(
            f"identity feature dim {direction.shape[1]} != declared {expected_dim}; "
            "the canonical euclidean consumer requires matching dimensions")


