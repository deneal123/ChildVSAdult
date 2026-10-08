"""Synthetic unit tests for the geometric MTFE feature interface.

These tests exercise only the geometric decomposition (OE-CNN Eq1 / TIP III-B
Eq1), declared resize/readout contracts, safe-division policy and frozen-backbone
gradient routing on tiny PRESCRIBED tensors. They do NOT train anything, call
optimizer.step(), use real face images or establish any full-method parity. A
passing suite does not make random/frozen features a trained MTFE, and this
component is always reported as geometric/unverified.

Run:
    python -m pytest tests/test_cacon_mtfe_components_v1.py -q
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from scripts import cacon_mtfe_components_v1 as M

REJECT = M.SafeDivPolicy(singular_policy="reject", minimum_norm=1e-6)


def _heads(source_hw=(2, 2), target_hw=(4, 6)):
    """A declared direction-map resize head with caller-chosen geometry."""
    return M.ResizeHeadContract(source_kind="direction_map", target_height=target_hw[0],
                                target_width=target_hw[1], mode="bilinear",
                                align_corners=False, source_height=source_hw[0],
                                source_width=source_hw[1])


# --- Eq1 reconstruction, scaling, batch independence -----------------------

def test_eq1_reconstruction_and_unit_direction():
    x = torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, -2.0], [0.0, 3.0]])
    r, u = M.radial_direction_split(x, policy=REJECT)
    assert torch.allclose(r, torch.tensor([5.0, 1.0, 2.0, 3.0]))
    assert torch.allclose(torch.linalg.vector_norm(u, dim=-1), torch.ones(4), atol=1e-6)
    residual = M.reconstruction_residual(x, r, u)
    assert torch.allclose(residual, torch.zeros_like(x), atol=1e-6)
    # x_id is the raw direction; explicit values for row 0.
    assert torch.allclose(u[0], torch.tensor([0.6, 0.8]))


def test_eq1_is_scale_equivariant():
    x = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    for scale in (0.5, 1.0, 7.0):
        r, u = M.radial_direction_split(x * scale, policy=REJECT)
        assert torch.allclose(r, torch.tensor([5.0, torch.linalg.vector_norm(x[1])]) * scale)
        r0, u0 = M.radial_direction_split(x, policy=REJECT)
        assert torch.allclose(u, u0, atol=1e-6)


def test_eq1_batch_independent_decomposition():
    x = torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])
    r_all, u_all = M.radial_direction_split(x, policy=REJECT)
    for i in range(len(x)):
        r_one, u_one = M.radial_direction_split(x[i:i + 1], policy=REJECT)
        assert r_one.item() == pytest.approx(r_all[i].item())
        assert torch.allclose(u_one[0], u_all[i], atol=1e-7)


# --- Safe division: zero / near-zero ---------------------------------------

def test_reject_policy_raises_on_zero_and_near_zero():
    for bad in (torch.zeros(1, 3), torch.full((1, 3), 1e-9)):
        with pytest.raises(ValueError, match="undefined"):
            M.radial_direction_split(bad, policy=REJECT)


def test_declared_epsilon_policy_is_explicit_finite_and_differentiable():
    policy = M.SafeDivPolicy(singular_policy="declared_epsilon", minimum_norm=0.5)
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0]], requires_grad=True)
    r, u = M.radial_direction_split(x, policy=policy)
    assert torch.isfinite(u).all() and torch.isfinite(r).all()
    # Regular rows are unit; the clamped zero row is explicitly NOT claimed unit.
    assert torch.allclose(torch.linalg.vector_norm(u[1], dim=-1), torch.ones(()), atol=1e-6)
    assert torch.linalg.vector_norm(u[0], dim=-1).item() == pytest.approx(0.0, abs=1e-6)
    assert r[0].item() == pytest.approx(0.0)  # raw norm preserved
    u.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad[0].abs().sum() > 0  # nonzero gradient through the clamp


def test_policy_and_norm_arguments_rejected_when_missing():
    x = torch.ones(1, 2)
    with pytest.raises(ValueError):
        M.radial_direction_split(x, policy="reject")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        M.SafeDivPolicy(singular_policy="clamp", minimum_norm=1e-3)
    with pytest.raises(ValueError):
        M.SafeDivPolicy(singular_policy="reject", minimum_norm=0.0)


# --- Missing / singular / malformed inputs ---------------------------------

def test_split_refuses_nonfinite_and_wrong_rank():
    with pytest.raises(ValueError, match="finite"):
        M.radial_direction_split(torch.tensor([[1.0, float("nan")]]), policy=REJECT)
    with pytest.raises(ValueError, match="rank"):
        M.radial_direction_split(torch.ones(3), policy=REJECT)
    with pytest.raises(ValueError):
        M.radial_direction_split(torch.empty(0, 3), policy=REJECT)


def test_reconstruction_residual_shape_guards():
    x = torch.ones(2, 3)
    with pytest.raises(ValueError):
        M.reconstruction_residual(x, torch.ones(3), torch.ones(2, 3))
    with pytest.raises(ValueError):
        M.reconstruction_residual(x, torch.ones(2), torch.ones(2, 4))


# --- Frozen backbone interface + image gradients ---------------------------

class _LinearBackbone(nn.Module):
    def __init__(self, in_features=6, out_features=4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, images):
        return images @ self.weight.T + self.bias


class _BatchNormBackbone(nn.Module):
    """Backbone whose forward updates BN running stats if left in train mode."""

    def __init__(self, in_features=6, out_features=4):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features, affine=False)
        self.linear = nn.Linear(in_features, out_features)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, images):
        return self.linear(self.bn(images))


class _DropoutBackbone(nn.Module):
    def __init__(self, in_features=6, out_features=4, p=0.5):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.linear = nn.Linear(in_features, out_features)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, images):
        return self.linear(self.drop(images))


def _frozen_backbone(preserve, module=None):
    if module is None:
        module = _LinearBackbone()
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()  # caller is responsible for eval mode
    return M.FrozenBackboneInterface(module, feature_dim=4,
                                     input_rank=2, preserve_input_grad=preserve)


def test_frozen_backbone_rejects_trainable_parameters():
    module = _LinearBackbone()
    module.eval()
    with pytest.raises(ValueError, match="caller-frozen"):
        M.FrozenBackboneInterface(module, feature_dim=4, input_rank=2, preserve_input_grad=False)


def test_frozen_backbone_requires_eval_mode():
    module = _LinearBackbone()
    for p in module.parameters():
        p.requires_grad_(False)
    assert module.training is True
    with pytest.raises(ValueError, match="eval"):
        M.FrozenBackboneInterface(module, feature_dim=4, input_rank=2, preserve_input_grad=False)


def test_bn_running_buffers_unchanged_and_dropout_is_eval():
    bn = _frozen_backbone(False, module=_BatchNormBackbone())
    before = {k: v.clone() for k, v in bn.module.named_buffers()}
    images = torch.randn(8, 6)
    bn(images)
    for k, v in bn.module.named_buffers():
        assert torch.equal(v, before[k]), f"buffer {k} changed"
    dr = _frozen_backbone(False, module=_DropoutBackbone(p=0.9))
    out_a = dr(torch.ones(4, 6))
    out_b = dr(torch.ones(4, 6))
    assert torch.allclose(out_a, out_b)  # deterministic => eval, no dropout


def test_frozen_backbone_refuses_params_unfrozen_after_construction():
    backbone = _frozen_backbone(preserve=False)
    backbone.module.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="caller-frozen"):
        backbone(torch.randn(2, 6))


def test_frozen_backbone_refuses_train_mode_after_construction():
    backbone = _frozen_backbone(preserve=False, module=_BatchNormBackbone())
    backbone.module.train()
    with pytest.raises(ValueError, match="eval"):
        backbone(torch.randn(4, 6))


def test_frozen_backbone_refuses_bad_outputs_and_inputs():
    class _Tiny(nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind
            self.p = nn.Parameter(torch.zeros(1), requires_grad=False)

        def forward(self, images):
            if self.kind == "nontensor":
                return float(images.sum())
            if self.kind == "rank3":
                return images[:, None]
            if self.kind == "width":
                return images[:, :2]
            return images.sum().expand(images.shape[0], 4)

    for kind, match in (("nontensor", "floating tensor"), ("rank3", "rank-2"),
                        ("width", "feature_dim")):
        tiny = _Tiny(kind).eval()
        backbone = M.FrozenBackboneInterface(tiny, feature_dim=4,
                                             input_rank=2, preserve_input_grad=False)
        with pytest.raises(ValueError, match=match):
            backbone(torch.randn(3, 6))
    good = M.FrozenBackboneInterface(_Tiny("ok").eval(), feature_dim=4,
                                     input_rank=2, preserve_input_grad=False)
    with pytest.raises(ValueError, match="rank-2"):
        good(torch.randn(3))
    with pytest.raises(ValueError):
        good(torch.empty(0, 6))


def test_preserve_input_grad_requires_and_keeps_graph():
    backbone = _frozen_backbone(preserve=True)
    images = torch.randn(3, 6, requires_grad=True)
    features = backbone(images)
    assert features.requires_grad
    features.sum().backward()
    assert images.grad is not None and images.grad.abs().sum() > 0


def test_preserve_input_grad_refuses_detached_input():
    backbone = _frozen_backbone(preserve=True)
    with pytest.raises(ValueError, match="requires_grad"):
        backbone(torch.randn(2, 6))


def test_no_preserve_path_returns_detached_features():
    backbone = _frozen_backbone(preserve=False)
    features = backbone(torch.randn(2, 6))
    assert not features.requires_grad


def test_frozen_weights_are_not_updated_without_optimizer():
    backbone = _frozen_backbone(preserve=True)
    before = backbone.module.weight.detach().clone()
    images = torch.randn(2, 6, requires_grad=True)
    backbone(images).sum().backward()
    assert torch.equal(backbone.module.weight.detach(), before)


# --- Resize head contract ---------------------------------------------------

def test_resize_head_shape_and_declared_adaptation():
    head = _heads(source_hw=(2, 2), target_hw=(4, 6))
    assert head.declared_adaptation is True
    assert head.source_cells == 4
    direction = torch.randn(3, 4)
    out = head.apply(direction)
    assert out.shape == (3, 1, 4, 6)
    with pytest.raises(ValueError, match="source_height"):
        head.apply(torch.randn(3, 5))


def test_resize_head_requires_caller_geometry_and_interpolation():
    with pytest.raises(ValueError, match="align_corners"):
        M.ResizeHeadContract(source_kind="direction_map", target_height=4,
                             target_width=4, mode="bilinear", align_corners=None,
                             source_height=2, source_width=2)
    with pytest.raises(ValueError, match="nearest"):
        M.ResizeHeadContract(source_kind="direction_map", target_height=4,
                             target_width=4, mode="nearest", align_corners=False,
                             source_height=2, source_width=2)
    with pytest.raises(ValueError):
        M.ResizeHeadContract(source_kind="direction_map", target_height=0,
                             target_width=4, mode="bilinear", align_corners=False,
                             source_height=2, source_width=2)


def test_radial_scalar_head_expands_without_interpolation():
    head = M.ResizeHeadContract(source_kind="radial_scalar", target_height=3,
                                target_width=5, mode="nearest", align_corners=None)
    out = head.apply(torch.tensor([1.0, 2.0]))
    assert out.shape == (2, 1, 3, 5)
    assert torch.allclose(out[0], torch.ones(1, 3, 5))
    with pytest.raises(ValueError):
        head.apply(torch.ones(2, 4))


# --- Age / identity readouts -----------------------------------------------

def test_linear_age_readout_eq2():
    radial = torch.tensor([1.0, 2.0, 3.0])
    out = M.linear_age_readout(radial, slope=2.0, intercept=-1.0)
    assert torch.allclose(out, torch.tensor([1.0, 3.0, 5.0]))
    with pytest.raises(ValueError):
        M.linear_age_readout(radial, slope=float("inf"), intercept=0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        M.linear_age_readout(torch.tensor([-1.0, 2.0]), slope=1.0, intercept=0.0)
    with pytest.raises(ValueError):
        M.linear_age_readout(radial, slope=True, intercept=0.0)


def test_expectation_age_readout_eq6_and_normalization_guard():
    probs = torch.tensor([[0.25, 0.25, 0.5], [1.0, 0.0, 0.0]])
    centers = (10.0, 20.0, 30.0)
    out = M.expectation_age_readout(probs, age_centers=centers)
    assert torch.allclose(out, torch.tensor([22.5, 10.0]))
    with pytest.raises(ValueError, match="normalized"):
        M.expectation_age_readout(torch.tensor([[0.5, 0.5, 0.5]]), age_centers=centers)
    with pytest.raises(ValueError, match="age center"):
        M.expectation_age_readout(probs, age_centers=(10.0, 20.0))
    with pytest.raises(ValueError, match="nonnegative"):
        M.expectation_age_readout(probs, age_centers=(10.0, -1.0, 30.0))
    with pytest.raises(ValueError, match="increasing"):
        M.expectation_age_readout(probs, age_centers=(30.0, 20.0, 10.0),
                                  require_ordered_centers=True)
    M.expectation_age_readout(probs, age_centers=centers, require_ordered_centers=True)
    with pytest.raises(ValueError):
        M.expectation_age_readout(probs, age_centers=(10.0, 20.0, True))
    with pytest.raises(ValueError):
        M.expectation_age_readout(probs, age_centers=centers,
                                  require_ordered_centers=1)


def test_cosine_logits_eq3_head_and_scale_required():
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    direction = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    logits = M.direction_cosine_logits(direction, weight, scale=4.0)
    assert torch.allclose(logits, torch.tensor([[4.0, 0.0], [0.0, -4.0]]))
    with pytest.raises(ValueError):
        M.direction_cosine_logits(direction, weight, scale=0.0)
    with pytest.raises(ValueError):
        M.direction_cosine_logits(direction, torch.ones(2, 3), scale=4.0)


def test_cosine_logits_normalizes_nonunit_weight_rows():
    direction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # Row 0 scaled x3 and row 1 scaled x0.5: cosines must be identical to unit.
    weight = torch.tensor([[3.0, 0.0], [0.0, 0.5]])
    got = M.direction_cosine_logits(direction, weight, scale=2.0)
    assert torch.allclose(got, torch.tensor([[2.0, 0.0], [0.0, 2.0]]), atol=1e-6)
    # A zero weight row is an undefined direction and must be refused.
    with pytest.raises(ValueError, match="class-weight row"):
        M.direction_cosine_logits(direction, torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
                                  scale=2.0)
    # Overflowing weight norms must be refused, not silently normalized.
    big = torch.tensor([[3e38, 3e38], [0.0, 1.0]])
    with pytest.raises(ValueError, match="overflow"):
        M.direction_cosine_logits(direction, big, scale=2.0)
    # dtype/device compatibility is required.
    with pytest.raises(ValueError, match="dtype"):
        M.direction_cosine_logits(direction, weight.double(), scale=2.0)


# --- Numeric representability (float16 epsilon underflow) ------------------


def test_minimum_norm_below_dtype_tiny_is_refused_float16():
    x = torch.tensor([[1.0, 0.0]], dtype=torch.float16)
    # 1e-8 is below finfo(float16).tiny (~6.1e-5): an underflowing divisor.
    policy = M.SafeDivPolicy(singular_policy="declared_epsilon", minimum_norm=1e-8)
    with pytest.raises(ValueError, match="representable"):
        M.radial_direction_split(x, policy=policy)
    ok = M.SafeDivPolicy(singular_policy="declared_epsilon", minimum_norm=1e-3)
    r, u = M.radial_direction_split(x, policy=ok)
    assert torch.isfinite(u).all() and torch.isfinite(r).all()


def test_radial_direction_split_refuses_overflowing_feature():
    with pytest.raises(ValueError, match="overflow"):
        M.direction_cosine_logits(torch.tensor([[1.0, 0.0]]),
                                  torch.tensor([[1e30, 0.0]]), scale=1.0)
    # A huge feature norm times a unit direction stays finite; overflow in the
    # residual product (r * u) must be caught.
    with pytest.raises(ValueError, match="overflow"):
        M.reconstruction_residual(torch.tensor([[3e38, 0.0]]),
                                  torch.tensor([3e38]), torch.tensor([[2.0, 0.0]]))


# --- dtype/device and gradcheck --------------------------------------------


def test_reconstruction_and_readouts_require_matching_dtype():
    features = torch.ones(2, 3)
    radial = torch.ones(2)
    direction = torch.ones(2, 3)
    M.reconstruction_residual(features, radial, direction)  # ok
    with pytest.raises(ValueError, match="dtype"):
        M.reconstruction_residual(features, radial.double(), direction)
    with pytest.raises(ValueError, match="dtype"):
        M.reconstruction_residual(features, radial, direction.half())
    with pytest.raises(ValueError, match="nonnegative"):
        M.reconstruction_residual(features, -radial, direction)


def test_gradcheck_radial_direction_split():
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    policy = M.SafeDivPolicy(singular_policy="reject", minimum_norm=1e-6)

    def fn(t):
        r, u = M.radial_direction_split(t, policy=policy, tolerance=1e-8)
        return r, u

    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_cosine_logits():
    w = torch.randn(3, 4, dtype=torch.float64)
    d = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    unit = d / torch.linalg.vector_norm(d, dim=-1, keepdim=True)

    def fn(t):
        # gradcheck perturbs the unit input by ~eps, so the unit test tolerance
        # must exceed that perturbation; 1e-3 is broader than eps=1e-6.
        return M.direction_cosine_logits(t, w, scale=2.0, tolerance=1e-3)

    assert torch.autograd.gradcheck(fn, (unit,), eps=1e-6, atol=1e-4, rtol=1e-3)


# --- Provenance gate: geometric vs trained MTFE ----------------------------

def _provenance(**kw):
    base = dict(source_locators=M.PRIMARY_LOCATORS, backbone_id="synthetic-lin",
                feature_dim=4)
    base.update(kw)
    return M.MTFEProvenance(**base)


def test_provenance_never_certifies_trained_mtfe():
    prov = _provenance()
    assert prov.mechanism == "geometric_decomposition"
    assert prov.trained_status == "provenance_unverified"
    # No boolean/digest field can promote a geometric split to a trained MTFE.
    with pytest.raises(TypeError):
        M.MTFEProvenance(source_locators=M.PRIMARY_LOCATORS, backbone_id="x",
                         feature_dim=4, is_pretrained_mtfe=True)
    assert _provenance(backbone_id="pretrained-backbone").trained_status == "provenance_unverified"


def test_provenance_refuses_nonprimary_locators_and_bad_declared():
    with pytest.raises(ValueError, match="https"):
        _provenance(source_locators=("a blog post",))
    with pytest.raises(ValueError, match="https"):
        _provenance(source_locators=("http://insecure.example",))
    with pytest.raises(ValueError):
        _provenance(declared_adaptations=("",))


# --- End-to-end interface + consumer contracts -----------------------------

def _interface(preserve=True, identity_head=False):
    cdp = M.ResizeHeadContract(source_kind="radial_scalar", target_height=2,
                               target_width=2, mode="nearest", align_corners=None)
    atl = _heads(source_hw=(1, 4), target_hw=(2, 2))
    weight = torch.randn(5, 4) if identity_head else None
    return M.MTFEFeatureInterface(
        _frozen_backbone(preserve), policy=REJECT, provenance=_provenance(),
        cdp_head=cdp, atl_head=atl, identity_weight=weight,
        identity_scale=4.0 if identity_head else None)


def test_interface_outputs_have_declared_shapes_and_scales():
    interface = _interface(identity_head=True)
    images = torch.randn(3, 6, requires_grad=True)
    out = interface.forward(images)
    assert out.radial_age.shape == (3,)
    assert out.direction_identity.shape == (3, 4)
    assert out.atl_identity.shape == (3, 4)
    assert out.cdp_age_map.shape == (3, 1, 2, 2)
    assert out.atl_identity_map is not None and out.atl_identity_map.shape == (3, 1, 2, 2)
    assert out.identity_logits is not None and out.identity_logits.shape == (3, 5)
    assert torch.allclose(torch.linalg.vector_norm(out.direction_identity, dim=-1),
                          torch.ones(3), atol=1e-6)
    M.check_identity_dimension(out.atl_identity, expected_dim=4)


def test_interface_surfaces_singular_input_as_error():
    interface = _interface(preserve=False)
    images = torch.zeros(2, 6)
    backbone = interface.backbone
    with torch.no_grad():
        backbone.module.weight.zero_()
        backbone.module.bias.zero_()
    with pytest.raises(ValueError, match="undefined"):
        interface.forward(images)


def test_interface_preserves_image_grad_through_frozen_backbone():
    interface = _interface(preserve=True)
    images = torch.randn(2, 6, requires_grad=True)
    out = interface.forward(images)
    out.atl_identity.sum().backward()
    assert images.grad is not None and images.grad.abs().sum() > 0


def test_interface_dimension_mismatch_and_head_inputs_are_refused():
    cdp = M.ResizeHeadContract(source_kind="radial_scalar", target_height=2,
                               target_width=2, mode="nearest", align_corners=None)
    with pytest.raises(ValueError, match="feature_dim"):
        M.MTFEFeatureInterface(_frozen_backbone(False), policy=REJECT,
                               provenance=_provenance(feature_dim=8), cdp_head=cdp)
    with pytest.raises(ValueError, match="together"):
        M.MTFEFeatureInterface(_frozen_backbone(False), policy=REJECT,
                               provenance=_provenance(), cdp_head=cdp,
                               identity_weight=torch.randn(5, 4))
    with pytest.raises(ValueError, match="trainable identity head"):
        M.MTFEFeatureInterface(_frozen_backbone(False), policy=REJECT,
                               provenance=_provenance(), cdp_head=cdp,
                               identity_weight=torch.randn(5, 4, requires_grad=True),
                               identity_scale=4.0)
    with pytest.raises(ValueError, match="radial scalar"):
        M.MTFEFeatureInterface(_frozen_backbone(False), policy=REJECT,
                               provenance=_provenance(),
                               cdp_head=_heads(source_hw=(1, 4), target_hw=(2, 2)))


# --- Missing trained-MTFE stages are explicitly enumerated -----------------


def test_missing_trained_mtfe_stages_are_enumerated():
    stages = M.MISSING_TRAINED_MTFE_STAGES
    assert isinstance(stages, tuple) and len(stages) >= 6
    assert any("lineage" in s for s in stages)
    assert any("optimizer" in s for s in stages)
    assert any("data" in s for s in stages)


# --- Canonical consumer compatibility (shape/scale only) -------------------

def test_direction_is_accepted_by_canonical_euclidean_consumer():
    from scripts import cacon_tip2021_losses_v1 as L

    torch.manual_seed(0)
    x = torch.randn(4, 4)
    _, unit = M.radial_direction_split(x, policy=REJECT)
    distances = L.euclidean(unit, unit)
    assert distances.shape == (4, 4)
    assert torch.allclose(distances, distances.T, atol=1e-6)
    assert torch.allclose(torch.diagonal(distances), torch.zeros(4), atol=1e-6)
    # Unit directions bound pairwise distance by 2 (scale contract).
    assert float(distances.max()) <= 2.0 + 1e-5


def test_actual_bchw_images_have_input_gradient_through_frozen_eval_backbone():
    module = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4),
                           nn.AdaptiveAvgPool2d(1), nn.Flatten()).double().eval()
    module.requires_grad_(False)
    backbone = M.FrozenBackboneInterface(module, feature_dim=4, input_rank=4,
                                         preserve_input_grad=True)
    before = {name: value.clone() for name, value in module.state_dict().items()}
    images = torch.randn(2, 3, 4, 4, dtype=torch.float64, requires_grad=True)
    features = backbone(images)
    assert features.shape == (2, 4)
    features.square().sum().backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()
    assert images.grad.abs().sum() > 0
    assert all(p.grad is None for p in module.parameters())
    assert all(torch.equal(value, before[name]) for name, value in module.state_dict().items())
    with pytest.raises(ValueError, match="rank-4"):
        backbone(torch.randn(2, 4, dtype=torch.float64))


def test_explicit_backbone_input_rank_cannot_be_guessed():
    module = _LinearBackbone().eval().requires_grad_(False)
    for rank in (True, 1, 3, 5, "4"):
        with pytest.raises(ValueError, match="input_rank"):
            M.FrozenBackboneInterface(module, feature_dim=4, input_rank=rank,
                                      preserve_input_grad=False)


def test_divisor_above_dtype_range_is_refused_before_clamp():
    policy = M.SafeDivPolicy(singular_policy="declared_epsilon", minimum_norm=1e100)
    with pytest.raises(ValueError, match="range"):
        M.radial_direction_split(torch.ones(1, 2), policy=policy)


def test_invalid_atl_contract_is_cleanly_refused():
    cdp = M.ResizeHeadContract(source_kind="radial_scalar", target_height=2,
                               target_width=2, mode="nearest", align_corners=None)
    with pytest.raises(ValueError, match="atl_head"):
        M.MTFEFeatureInterface(_frozen_backbone(False), policy=REJECT,
                               provenance=_provenance(), cdp_head=cdp, atl_head="wrong")


