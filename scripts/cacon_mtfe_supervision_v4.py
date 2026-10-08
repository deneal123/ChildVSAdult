"""MTFE supervision contract v4 (TIP2021 AOFS) - enforced-freeze, trainable proposal.

Canonical untrained supervision component: no weights are loaded, no real
face image is inferred, and no optimizer/optimizer.step() exists. Synthetic
forward/backward (autograd) tests are allowed; training is not performed.

Root-reviewed v4 port: independent of worker scratch paths and the unaccepted v3
implementation. Existing native-bound geometry/loss sources remain unchanged.
Age head/resize/sampling are declared adaptations, not full FusionNet/AOFS parity.
Corrections:

  1. ``MTFEConfig.check_stage_freeze_consistency`` is now invoked by
     ``MTFEConfig.__post_init__``, so an invalid stage/freeze combination cannot be
     constructed at all (previously it was a dead method and a FROZEN
     ``mtfe_pretraining`` instance could be built).
  2. The declared freeze policy is now ENFORCED across the attached backbone, the
     age head and both resize heads: their parameters are marked
     ``requires_grad=False`` and their modules are held in ``eval()`` (so BatchNorm
     statistics do not mutate) even if the caller calls ``module.train()``.
  3. Freezing never calls ``torch.no_grad()``: the forward graph stays
     autograd-tracked, so a frozen MTFE can still pass gradients to its INPUT,
     which is what lets it supervise a generator.
  4. ``AgeRegressionHead.input_dim`` must be 1 for BOTH radial layouts, because the
     radial component is always a scalar norm per sample. The v3 'column' path
     allowed ``input_dim == 2`` and only crashed later in the Linear layer.
  5. ``hidden_dims`` must be a tuple of positive, non-boolean integers (v3 accepted
     any value and crashed in ``nn.Linear``).
  6. ``reduction`` must be one of ``none``/``mean``/``sum`` (v3 silently treated an
     invalid value as ``mean``), ``target_age`` must be finite and nonnegative, and
     the multi-task objective validates scalar/dtype/device/finite inputs and the
     finiteness of its output.
  7. ``IdentityBankDeclaration`` requires all three of anchor/positive/negative to be
     nonempty AND pairwise distinct (v3 only compared anchor with positive).
  8. No provenance is implied that the MTFE is trained or pre-trained: the only
     provenance strings carried over are the v3 author facts and the explicit
     "unresolved" notices. There is no trained/pretrained flag anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import torch
from torch import nn

from scripts.cacon_mtfe_components_v1 import (
    SafeDivPolicy,
    expectation_age_readout,
    radial_direction_split,
    reconstruction_residual,
)
from scripts.cacon_tip2021_losses_v1 import prescribed_triplets

__all__ = [
    "MTFEConfig",
    "MTFEModule",
    "AgeRegressionHead",
    "LearnedResizeHead",
    "MTFEForward",
    "IdentityBankDeclaration",
    "TrainingStage",
    "FreezePolicy",
    "V4_CHANGES",
    "V4_NON_CLAIMS",
    "AGE_HEAD_INPUT_DIM",
    "radial_direction_split",
    "reconstruction_residual",
    "prescribed_triplets",
    "SafeDivPolicy",
]

AGE_HEAD_INPUT_DIM = 1

V4_CHANGES: tuple[str, ...] = (
    "MTFEConfig.__post_init__ calls check_stage_freeze_consistency (invalid "
    "stage/freeze cannot be constructed)",
    "declared freeze policy is enforced on backbone + heads + resize heads "
    "requires_grad=False and eval(), and survives a caller module.train()",
    "freezing keeps the autograd graph so gradients still reach the input",
    "AgeRegressionHead.input_dim must be 1 for both radial_layout values",
    "hidden_dims must be a tuple of positive non-boolean ints",
    "reduction must be exactly none/mean/sum",
    "target_age must be finite and nonnegative",
    "multi-task objective validates scalar/dtype/device/finite inputs and output",
    "IdentityBankDeclaration requires three nonempty, pairwise distinct banks",
)

V4_NON_CLAIMS: tuple[str, ...] = (
    "no optimizer, no optimizer.step(), no training loop, no weights loaded",
    "no real face image is read or inferred in this module",
    "no trained/pre-trained provenance is asserted for any MTFE component",
    "the declared age head and resize geometry are caller adaptations, not author "
    "settings",
)


class TrainingStage(str, Enum):
    MTFE_PRETRAINING = "mtfe_pretraining"
    AOFS_GENERATION = "aofs_generation"


class FreezePolicy(str, Enum):
    TRAINABLE = "trainable"
    FROZEN = "frozen"


_REDUCTIONS = ("none", "mean", "sum")


# --------------------------------------------------------------------------- #
# Validated caller declarations.                                              #
# --------------------------------------------------------------------------- #


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name}: integer required (bool is not an integer)")
    if value < 1:
        raise ValueError(f"{name}: must be >= 1")
    return value


def _finite_real(name: str, value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name}: scalar required")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: real scalar required")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name}: finite scalar required")
    return value


def _finite_floating(name: str, value: torch.Tensor, ndim: int | None = None) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name}: torch.Tensor required")
    if not value.is_floating_point():
        raise ValueError(f"{name}: floating point tensor required")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name}: rank-{ndim} tensor required")
    if value.numel() == 0:
        raise ValueError(f"{name}: nonempty tensor required")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name}: finite values required")


def _reduction(name: str, reduction) -> str:
    if reduction not in _REDUCTIONS:
        raise ValueError(
            f"{name}: reduction must be exactly one of {_REDUCTIONS}, got {reduction!r}")
    return reduction


@dataclass(frozen=True)
class IdentityBankDeclaration:
    """Declared identity supervision banks for the MTFE stage.

    TIP III-B states the Adversarial Triplet loss supervises identity features but
    publishes no pretraining banks, so all three descriptions are caller
    declarations. They must be nonempty and PAIRWISE distinct: an "anchor" identical
    to a "positive" (or a "negative") is not a triplet at all.
    """

    anchor: str
    positive: str
    negative: str
    declared_adaptation: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        cleaned: dict[str, str] = {}
        for name in ("anchor", "positive", "negative"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty declared description")
            cleaned[name] = value.strip().lower()
        pairs = (
            ("anchor", "positive"),
            ("anchor", "negative"),
            ("positive", "negative"),
        )
        for left, right in pairs:
            if cleaned[left] == cleaned[right]:
                raise ValueError(
                    f"{left} and {right} must be declared as different banks")


@dataclass(frozen=True)
class MTFEConfig:
    """Mandatory caller configuration; every field is required (no defaults)."""

    feature_dim: int
    num_identity_classes: int
    margin: float
    policy: SafeDivPolicy
    identity_sampling: IdentityBankDeclaration
    stage: TrainingStage
    freeze_policy: FreezePolicy
    task_weight_age: float
    task_weight_identity: float
    radial_layout: Literal["scalar", "column"]

    def __post_init__(self) -> None:
        _positive_int("feature_dim", self.feature_dim)
        _positive_int("num_identity_classes", self.num_identity_classes)
        if not isinstance(self.policy, SafeDivPolicy):
            raise ValueError("explicit canonical SafeDivPolicy required")
        if not isinstance(self.identity_sampling, IdentityBankDeclaration):
            raise ValueError("explicit IdentityBankDeclaration required")
        if not isinstance(self.stage, TrainingStage):
            raise ValueError("explicit TrainingStage required")
        if not isinstance(self.freeze_policy, FreezePolicy):
            raise ValueError("explicit FreezePolicy required; TIP does not state it")
        if self.radial_layout not in ("scalar", "column"):
            raise ValueError("radial_layout must be 'scalar' or 'column'")
        _finite_real("margin", self.margin)
        if self.margin < 0:
            raise ValueError("margin must be nonnegative")
        for name in ("task_weight_age", "task_weight_identity"):
            if _finite_real(name, getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be strictly positive")
        # v4: enforce the stage/freeze combination AT CONSTRUCTION, not on request.
        self.check_stage_freeze_consistency()

    def check_stage_freeze_consistency(self) -> str:
        """Return the declared-policy wording, or refuse an impossible combination."""
        if self.stage is TrainingStage.MTFE_PRETRAINING:
            if self.freeze_policy is FreezePolicy.FROZEN:
                raise ValueError(
                    "REFUSED: the MTFE owns its parameters during mtfe_pretraining, so "
                    "it cannot be declared FROZEN")
            return ("declared adaptation: stage=mtfe_pretraining, "
                    "freeze_policy=trainable (TIP does not state an AOFS-stage freeze "
                    "policy either, so it must be declared explicitly)")
        return ("declared adaptation: stage=aofs_generation, "
                f"freeze_policy={self.freeze_policy.value} (TIP only calls the MTFE "
                "'pre-trained'; whether it is frozen is unpublished)")


# --------------------------------------------------------------------------- #
# Trainable heads (declared adaptations with explicit shape/dtype contracts).  #
# --------------------------------------------------------------------------- #


class AgeRegressionHead(nn.Module):
    """Trainable age head standing in for the referenced age regression model [56].

    Input: the radial r as (B,) or (B, 1). The radial component is ALWAYS a scalar
    norm per sample, so ``input_dim`` must be 1 for both `radial_layout` values.
    Output: an age prediction (B,) using FusionNet's published readout order
    (eliminate negatives -> softmax -> normalize -> sum_i p_i y_i).

    `num_age_classes` j and `age_centers` y_i are mandatory: TIP publishes neither.
    """

    def __init__(self, *, input_dim: int, num_age_classes: int,
                 age_centers: tuple[float, ...],
                 hidden_dims: tuple[int, ...] = ()) -> None:
        super().__init__()
        self.input_dim = _positive_int("input_dim", input_dim)
        if self.input_dim != AGE_HEAD_INPUT_DIM:
            raise ValueError(
                f"input_dim must be {AGE_HEAD_INPUT_DIM}: the radial component is a "
                "scalar norm per sample, so both 'scalar' and 'column' layouts feed "
                "(B, 1)")
        self.num_age_classes = _positive_int("num_age_classes", num_age_classes)
        if self.num_age_classes < 2:
            raise ValueError("num_age_classes must be >= 2 for a softmax distribution")
        if not isinstance(hidden_dims, tuple):
            raise ValueError("hidden_dims must be a tuple of positive integers")
        checked_hidden: list[int] = []
        for index, width in enumerate(hidden_dims):
            checked_hidden.append(_positive_int(f"hidden_dims[{index}]", width))
        if len(age_centers) != self.num_age_classes:
            raise ValueError("one age center per age class required")
        centers = torch.tensor([_finite_real("age center", c) for c in age_centers],
                               dtype=torch.float64)
        if (centers < 0).any():
            raise ValueError("age centers must be nonnegative")
        if not bool((centers[1:] > centers[:-1]).all()):
            raise ValueError("age centers must be strictly increasing (ordered grid)")
        self.register_buffer("age_centers", centers)
        widths = [self.input_dim, *checked_hidden, self.num_age_classes]
        self.mlp = nn.Sequential(
            *[nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])

    def scores(self, radial: torch.Tensor) -> torch.Tensor:
        _finite_floating("radial", radial)
        if radial.ndim == 1:
            radial = radial.unsqueeze(1)
        elif radial.ndim != 2 or radial.shape[1] != self.input_dim:
            raise ValueError(f"radial must be (B,) or (B,{self.input_dim})")
        if (radial < 0).any():
            raise ValueError("radial is a norm and must be nonnegative")
        scores = self.mlp(radial)
        _finite_floating("age scores", scores, ndim=2)
        return scores

    def forward(self, radial: torch.Tensor) -> torch.Tensor:
        scores = self.scores(radial)
        probabilities = torch.softmax(torch.relu(scores), dim=1)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        return expectation_age_readout(
            probabilities,
            age_centers=tuple(float(c) for c in self.age_centers.tolist()),
        )

    def loss(self, predicted: torch.Tensor, target: torch.Tensor, *,
             kind: Literal["l1", "l2"],
             reduction: Literal["none", "mean", "sum"] = "mean") -> torch.Tensor:
        """Age supervision loss; TIP does not publish the family, so kind is required."""
        _finite_floating("predicted", predicted)
        _finite_floating("target", target)
        _reduction("age loss", reduction)
        if kind not in ("l1", "l2"):
            raise ValueError("kind must be explicitly 'l1' or 'l2'")
        if predicted.shape != target.shape:
            raise ValueError("predicted and target must share a shape")
        if predicted.dtype != target.dtype or predicted.device != target.device:
            raise ValueError("predicted and target must share dtype and device")
        if bool((target < 0).any()):
            raise ValueError("target_age must be nonnegative")
        difference = (predicted - target).abs()
        values = difference if kind == "l1" else difference.square()
        _finite_floating("age loss values", values)
        if reduction == "none":
            return values
        return values.sum() if reduction == "sum" else values.mean()


class LearnedResizeHead(nn.Module):
    """Trainable resize head mapping a task-specific feature to a consumer width.

    TIP Fig.3 (printed p5) resizes task-specific features before the CDP
    feature-level discriminator or the ATL but publishes no geometry, so
    `target_dim` (the consumer input width) is a mandatory caller argument and the
    projection is learned. `declared_adaptation` stays True by construction.
    """

    def __init__(self, *, in_dim: int, target_dim: int) -> None:
        super().__init__()
        self.in_dim = _positive_int("in_dim", in_dim)
        self.target_dim = _positive_int("target_dim", target_dim)
        self.projection = nn.Linear(self.in_dim, self.target_dim)
        self.declared_adaptation = True

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _finite_floating("resize head input", features, ndim=2)
        if features.shape[1] != self.in_dim:
            raise ValueError(f"resize head input width {features.shape[1]} != {self.in_dim}")
        _require_head_dtype_device(features, self.projection.weight,
                                   name="resize head input")
        out = self.projection(features)
        _finite_floating("resize head output", out, ndim=2)
        if out.shape[1] != self.target_dim:
            raise ValueError("resize head produced an unexpected width")
        return out


def _require_head_dtype_device(features: torch.Tensor, reference: torch.Tensor, *,
                               name: str) -> None:
    """The heads hold real parameters, so inputs must match their dtype/device."""
    if features.dtype != reference.dtype:
        raise ValueError(
            f"{name}: dtype {features.dtype} must match the module parameter dtype "
            f"{reference.dtype}; cast the module or the input explicitly")
    if features.device != reference.device:
        raise ValueError(
            f"{name}: device {features.device} must match the module parameter "
            f"device {reference.device}")


# --------------------------------------------------------------------------- #
# The trainable MTFE module with an ENFORCED freeze policy.                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MTFEForward:
    """Tensors handed to the declared consumers."""

    radial: torch.Tensor
    direction: torch.Tensor
    reconstruction_error: torch.Tensor
    age_prediction: torch.Tensor
    cdp_input: torch.Tensor
    identity_feature: torch.Tensor
    identity_input: torch.Tensor | None = None


class MTFEModule(nn.Module):
    """Trainable MTFE path: backbone features -> Eq.1 split -> age + identity heads.

    The backbone is caller-owned and is registered as a submodule so the declared
    freeze policy can cover it. Nothing here creates an optimizer, steps one, loads
    weights, or reads a dataset. Freezing marks parameters `requires_grad=False` and
    holds the modules in `eval()`; it never uses `torch.no_grad()`, so the graph
    still carries gradients to the input (a frozen MTFE can supervise a generator).

    The declared `freeze_policy` is physical, not advisory:

    * attributes are assigned BEFORE `apply_declared_freeze_policy()` so the policy
      covers the age head, both resize heads and the attached backbone;
    * `train()` re-applies `eval()` + `requires_grad=False` on those modules, so a
      caller's `module.train()` cannot re-enable backbone BatchNorm updates;
    * every forward re-asserts the policy, so external tampering is caught loudly.
    """

    def __init__(self, config: MTFEConfig, *, age_head: AgeRegressionHead,
                 resize_head: LearnedResizeHead, backbone: nn.Module | None = None,
                 identity_resize_head: LearnedResizeHead | None = None) -> None:
        super().__init__()
        if not isinstance(config, MTFEConfig):
            raise ValueError("explicit MTFEConfig required")
        if not isinstance(age_head, AgeRegressionHead):
            raise ValueError("explicit AgeRegressionHead required")
        if not isinstance(resize_head, LearnedResizeHead):
            raise ValueError("explicit LearnedResizeHead required")
        if age_head.input_dim != AGE_HEAD_INPUT_DIM:
            raise ValueError(
                f"age_head.input_dim must be {AGE_HEAD_INPUT_DIM} for both radial "
                "layouts (the radial component is always a scalar)")
        if resize_head.in_dim != 1:
            raise ValueError(
                "the age-specific resize head consumes the radial scalar (in_dim == 1); "
                "TIP does not publish a different geometry")
        if identity_resize_head is not None:
            if not isinstance(identity_resize_head, LearnedResizeHead):
                raise ValueError("identity_resize_head must be a LearnedResizeHead")
            if identity_resize_head.in_dim != config.feature_dim:
                raise ValueError(
                    "identity_resize_head.in_dim must equal config.feature_dim")
        if backbone is not None and not isinstance(backbone, nn.Module):
            raise ValueError("backbone must be an nn.Module or None")
        self.config = config
        self.age_head = age_head
        self.resize_head = resize_head
        self.identity_resize_head = identity_resize_head
        self.backbone = backbone
        self.margin = float(config.margin)
        # Apply the (already validated) declared policy immediately.
        self.apply_declared_freeze_policy()

    # -- freeze policy ------------------------------------------------------- #

    def _policy_modules(self) -> tuple[nn.Module, ...]:
        modules = [self.age_head, self.resize_head]
        if self.identity_resize_head is not None:
            modules.append(self.identity_resize_head)
        if self.backbone is not None:
            modules.append(self.backbone)
        return tuple(modules)

    def is_declared_frozen(self) -> bool:
        return self.config.freeze_policy is FreezePolicy.FROZEN

    def apply_declared_freeze_policy(self) -> MTFEModule:
        """Make the declared policy physically true (idempotent).

        FROZEN forces `requires_grad=False` and `eval()` on the age head, both resize
        heads and the attached backbone. TRAINABLE deliberately does NOT overwrite the
        caller's own `requires_grad` flags: the backbone is caller-owned, so this
        module only refuses to *re-enable* nothing and leaves the caller's choices in
        place (the standard `nn.Module.train()` propagation still applies).
        """
        if not self.is_declared_frozen():
            return self
        for module in self._policy_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            module.eval()
        return self

    def _assert_frozen_consistency(self) -> None:
        if not self.is_declared_frozen():
            return
        for module in self._policy_modules():
            if any(child.training for child in module.modules()):
                raise RuntimeError(
                    "declared FROZEN policy violated: a frozen submodule is in "
                    "training mode; BatchNorm statistics could mutate")
            for name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    raise RuntimeError(
                        f"declared FROZEN policy violated: {name} still requires grad")
            for name, buffer in module.named_buffers():
                if buffer.is_floating_point() and buffer.requires_grad:
                    raise RuntimeError(
                        f"declared FROZEN policy violated: buffer {name} requires grad")

    def train(self, mode: bool = True) -> MTFEModule:
        """`train()` cannot defeat a declared freeze (v3 left everything trainable)."""
        super().train(mode)
        if self.is_declared_frozen():
            for module in self._policy_modules():
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
                    parameter.grad = None
                module.eval()
        self._assert_frozen_consistency()
        return self

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters the declared policy actually makes trainable."""
        return tuple(p for p in self.parameters() if p.requires_grad)

    # -- feature extraction -------------------------------------------------- #

    def forward_features(self, features: torch.Tensor) -> MTFEForward:
        """The single decomposition path: raw features -> Eq.1 split -> heads."""
        self._assert_frozen_consistency()
        _finite_floating("features", features, ndim=2)
        _require_head_dtype_device(features, self.age_head.mlp[0].weight,
                                   name="raw features")
        if features.shape[1] != self.config.feature_dim:
            raise ValueError(
                f"raw feature width {features.shape[1]} != declared feature_dim "
                f"{self.config.feature_dim} (R is a caller argument)")
        radial, direction = radial_direction_split(features, policy=self.config.policy)
        residual = reconstruction_residual(features, radial, direction)
        error = torch.linalg.vector_norm(residual, dim=-1).max()
        # The radial component is always a scalar per sample, so the age head always
        # consumes (B, 1); `radial_layout` only describes the consumer layout.
        head_input = radial.unsqueeze(1)
        age_prediction = self.age_head(head_input)
        cdp_input = self.resize_head(radial.unsqueeze(1))
        identity_input = None
        if self.identity_resize_head is not None:
            identity_input = self.identity_resize_head(direction)
        return MTFEForward(
            radial=radial,
            direction=direction,
            reconstruction_error=error,
            age_prediction=age_prediction,
            cdp_input=cdp_input,
            identity_feature=direction,
            identity_input=identity_input,
        )

    def forward(self, images: torch.Tensor) -> MTFEForward:
        self._assert_frozen_consistency()  # Before backbone BN can mutate any buffer.
        if self.backbone is None:
            raise ValueError(
                "no backbone attached; call forward_features with caller raw features")
        _finite_floating("images", images)
        features = self.backbone(images)
        _finite_floating("backbone features", features, ndim=2)
        return self.forward_features(features)

    # -- supervision terms --------------------------------------------------- #

    def age_term(self, outputs: MTFEForward, target_age: torch.Tensor, *,
                 kind: Literal["l1", "l2"],
                 reduction: Literal["none", "mean", "sum"] = "mean") -> torch.Tensor:
        if not isinstance(outputs, MTFEForward):
            raise ValueError("explicit MTFEForward required")
        return self.age_head.loss(outputs.age_prediction, target_age, kind=kind,
                                  reduction=_reduction("age term", reduction))

    def identity_term(self, anchor: torch.Tensor, positive: torch.Tensor,
                      negative: torch.Tensor, *,
                      reduction: Literal["none", "mean", "sum"] = "sum") -> torch.Tensor:
        """Adversarial Triplet loss on identity features (TIP III-B printed p4).

        Delegates to the canonical loss module; the bank *sampling* remains the
        caller's declared adaptation (`IdentityBankDeclaration`).
        """
        return prescribed_triplets(anchor, positive, negative, self.margin,
                                   adversarial=True,
                                   reduction=_reduction("identity term", reduction))

    def mtfe_objective(self, age_loss: torch.Tensor, identity_loss: torch.Tensor
                       ) -> torch.Tensor:
        """Multi-task objective with declared weights (TIP: unpublished)."""
        _finite_floating("age_loss", age_loss)
        _finite_floating("identity_loss", identity_loss)
        for name, value in (("age_loss", age_loss), ("identity_loss", identity_loss)):
            if value.numel() != 1:
                raise ValueError(f"{name}: scalar loss required")
        if age_loss.dtype != identity_loss.dtype:
            raise ValueError("age_loss and identity_loss must share a dtype")
        if age_loss.device != identity_loss.device:
            raise ValueError("age_loss and identity_loss must share a device")
        objective = (self.config.task_weight_age * age_loss.reshape(())
                     + self.config.task_weight_identity * identity_loss.reshape(()))
        if not bool(torch.isfinite(objective).all()):
            raise ValueError("mtfe objective produced a non-finite value")
        return objective
