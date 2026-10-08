"""Synthetic CPU-only shape/routing tests for the AOFS architecture ADAPTATION.

Tiny random weights only: no GPU, no real faces, no training, no optimizer.step.
These tests show the declared graph is executable and gradients route as
specified, NOT that the method works. The MTFE front end is the canonical
geometric interface in `scripts/cacon_mtfe_components_v1.py` around a test-only
random frozen backbone; it is a decomposition, never a trained MTFE. Loss terms
come from `scripts/cacon_tip2021_losses_v1.py` and are not reimplemented.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from scripts import cacon_aofs_networks_v1 as net
from scripts import cacon_mtfe_components_v1 as mt
from scripts import cacon_tip2021_losses_v1 as loss

torch.set_num_threads(1)

IMAGE = 128
BATCH = 4
CATEGORIES = 5
REGIME = net.AGE_REGIME_CACON_5YEAR
FEATURE_DIM = 16
PLAN_KWARGS = dict(stem_channels=4, encoder_channels=8, residual_channels=8,
                   decoder_channels=4, num_age_categories=CATEGORIES, age_regime=REGIME)


def make_plan(**overrides):
    kwargs = dict(PLAN_KWARGS)
    kwargs.update(overrides)
    return net.declared_adaptation_plan(**kwargs)


def make_generator(plan=None):
    plan = make_plan() if plan is None else plan
    return plan, net.AOFSGenerator(plan)


def make_source(seed: int = 0) -> torch.Tensor:
    return torch.randn(BATCH, 3, IMAGE, IMAGE, generator=torch.Generator().manual_seed(seed))


def make_labels() -> torch.Tensor:
    return torch.tensor([0, 1, 2, 4], dtype=torch.long)


def make_pool(norm: str = "instance1d") -> net.ConditionalDiscriminatorPool:
    return net.ConditionalDiscriminatorPool(
        FEATURE_DIM, CATEGORIES, norm=norm, leaky_slope=0.2, affine=True)


def clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


# --- literal table contradiction -------------------------------------------


def test_literal_table_trace_is_not_image_to_image():
    trace = net.literal_table_trace()
    assert trace[0] == IMAGE and trace[1] == 124 and trace[-1] == 1
    with pytest.raises(net.LiteralTableContradiction):
        net.assert_literal_table_inconsistent()


def test_literal_table_plan_is_refused_before_forward():
    plan = net.literal_table_plan(stem_channels=4, encoder_channels=8,
                                  residual_channels=8, decoder_channels=4,
                                  num_age_categories=CATEGORIES)
    with pytest.raises(net.PlanError):
        net.AOFSGenerator(plan)


# --- mandatory declared plan ------------------------------------------------


def test_no_four_group_default_and_explicit_regime_required():
    with pytest.raises(TypeError):
        net.declared_adaptation_plan(stem_channels=4, encoder_channels=8,
                                     residual_channels=8, decoder_channels=4)
    import inspect

    fields = inspect.signature(net.GeneratorPlan).parameters
    assert fields["num_age_categories"].default is inspect.Parameter.empty
    assert fields["age_regime"].default is inspect.Parameter.empty
    assert REGIME != net.AGE_REGIME_TIP2021


def test_declared_plan_restores_resolution_and_metadata_complete():
    plan, generator = make_generator()
    trace = plan.resolution_trace()
    assert trace[0] == trace[-1] == IMAGE
    assert isinstance(generator, net.AOFSGenerator)
    meta = plan.assert_metadata_complete()
    assert set(net.METADATA_FIELDS) <= set(meta)
    assert meta["num_age_categories"].startswith("caller_declared")
    assert meta["final_decoder_norm"].startswith("source")
    assert meta["residual_activation_order"].startswith("adaptation")


def test_metadata_distinguishes_adaptation_choices():
    adapted = replace(make_plan(), final_decoder_norm="none",
                      residual_activation_order="post_add_relu", norm_kind="batch2d",
                      output_padding=(0, 1))
    meta = adapted.choice_metadata()
    assert meta["final_decoder_norm"].startswith("adaptation")
    assert meta["residual_activation_order"].startswith("adaptation")
    assert meta["norm_kind"].startswith("adaptation")
    assert meta["output_padding"].startswith("adaptation")


def test_final_decoder_instancenorm_matches_source_row():
    plan, generator = make_generator()
    assert plan.final_decoder_norm == "instance"
    assert isinstance(generator.decoder_norm[-1], nn.InstanceNorm2d)
    off = make_generator(make_plan(final_decoder_norm="none"))[1]
    assert not isinstance(off.decoder_norm[-1], nn.InstanceNorm2d)


def test_strict_int_and_finite_validation_reject_bool_and_float():
    for bad in (True, 3.0, "4"):
        with pytest.raises(net.PlanError):
            net.ConvSpec(kernel=bad, stride=1, padding=0)
    with pytest.raises(net.PlanError):
        make_plan(image_size=True)
    with pytest.raises(net.PlanError):
        net.ImageDiscriminator((3, True, 8, 16, 32, 1), leaky_slope=0.2, affine=True)


def test_output_padding_and_residual_stride_validity():
    with pytest.raises(net.PlanError):
        replace(make_plan(), output_padding=(0, 0)).validate()
    with pytest.raises(net.PlanError):
        replace(make_plan(), residual_conv=net.ConvSpec(3, 2, 1)).validate()
    with pytest.raises(net.PlanError):
        net.transpose_out(8, net.ConvSpec(3, 2, 1), output_padding=2)


def test_intermediate_size_one_feeding_norm_is_rejected():
    broken = replace(make_plan(), image_size=8, residual_blocks=2, residual_strides=(2, 2))
    with pytest.raises(net.PlanError, match="exceed 1"):
        broken.validate()
    full = replace(broken, norm_kind="none", final_decoder_norm="none")
    with pytest.raises(net.PlanError, match="restore resolution"):
        full.validate()


def test_projection_geometry_mismatch_is_rejected():
    plan = make_plan()
    assert net.skip_projection_alignment(64, plan.residual_conv, 2) == (32, 32)
    assert net.skip_projection_alignment(64, net.ConvSpec(3, 1, 0), 1) == (62, 64)
    broken = replace(plan, residual_conv=net.ConvSpec(3, 1, 0))
    with pytest.raises(net.PlanError, match="geometry mismatch"):
        broken._check_skip_alignment()


# --- generator forward ------------------------------------------------------


def test_generator_forward_shapes_labels_and_orientation():
    plan, generator = make_generator()
    image, encoder_feature, one_hot = generator(make_source(), make_labels())
    assert image.shape == (BATCH, 3, IMAGE, IMAGE)
    assert encoder_feature.shape == (BATCH, 8, IMAGE // 2, IMAGE // 2)
    assert one_hot.shape == (BATCH, CATEGORIES)
    assert torch.allclose(one_hot.sum(1), torch.ones(BATCH))
    assert torch.isfinite(image).all()
    other = make_generator(make_plan(residual_activation_order="post_add_relu"))[1]
    other.load_state_dict(generator.state_dict())
    main, _, _ = generator(make_source(), make_labels())
    alt, _, _ = other(make_source(), make_labels())
    assert not torch.allclose(main, alt, atol=1e-5)


def test_generator_rejects_wrong_resolution_labels_and_nonfinite():
    _, generator = make_generator()
    with pytest.raises(net.PlanError):
        generator(torch.randn(BATCH, 3, 64, 64), make_labels())
    with pytest.raises(net.PlanError):
        generator(make_source(), torch.tensor([0, 1, 2, 9]))
    with pytest.raises(net.PlanError):
        generator(make_source(), make_labels().float())
    nonfinite = make_source()
    nonfinite[0, 0, 0, 0] = float("nan")
    with pytest.raises(net.PlanError):
        generator(nonfinite, make_labels())


def test_label_conditioning_changes_output():
    _, generator = make_generator()
    source = make_source()
    a, _, _ = generator(source, torch.zeros(BATCH, dtype=torch.long))
    b, _, _ = generator(source, torch.full((BATCH,), CATEGORIES - 1, dtype=torch.long))
    assert not torch.allclose(a, b)


def test_hook_shapes_match_plan_including_skip_and_decoder():
    plan, generator = make_generator()
    expected = plan.module_shape_expectations(BATCH)
    seen: dict[str, tuple[int, int, int]] = {}
    handles = []
    for name, module in generator.named_modules():
        if name in expected:
            handles.append(module.register_forward_hook(
                lambda mod, inp, out, key=name: seen.__setitem__(key, tuple(out.shape[1:]))))
    try:
        generator(make_source(), make_labels())
    finally:
        for handle in handles:
            handle.remove()
    assert set(seen) == set(expected)
    assert seen == expected
    # Projected skip and both decoder outputs are part of the checks.
    assert any(name.endswith(".proj") for name in seen)
    assert "decoder.0" in seen and "decoder.1" in seen


# --- image discriminator ----------------------------------------------------


def test_image_discriminator_patch_output_and_guards():
    disc = net.ImageDiscriminator((3, 8, 16, 32, 64, 1), leaky_slope=0.2, affine=True)
    logits = disc(make_source())
    assert logits.shape == (BATCH, 1, 8, 8)
    with pytest.raises(net.PlanError):
        net.ImageDiscriminator((3, 8, 16, 32, 64, 2), leaky_slope=0.2, affine=True)
    with pytest.raises(TypeError):
        net.ImageDiscriminator((3, 8, 16, 32, 64, 1))


# --- feature discriminator / CDP -------------------------------------------


def test_feature_norm_is_a_mandatory_caller_choice():
    features = torch.randn(BATCH, FEATURE_DIM)
    for norm in net.FEATURE_NORMS:
        out = net.FeatureDiscriminator(FEATURE_DIM, norm=norm, leaky_slope=0.2, affine=True)
        assert out(features).shape == (BATCH, 1)
    with pytest.raises(TypeError):
        net.FeatureDiscriminator(FEATURE_DIM)
    with pytest.raises(net.PlanError):
        net.FeatureDiscriminator(FEATURE_DIM, norm="instance1d", leaky_slope=0.2,
                                 affine=True, widths=(8, 4, 2, 1))
    with pytest.raises(net.PlanError):
        net.FeatureDiscriminator(FEATURE_DIM, norm="instance1d", leaky_slope=0.2,
                                 affine=True)(torch.randn(BATCH, FEATURE_DIM + 1))


def test_instancenorm1d_view_choice_and_degeneracy():
    norm = nn.InstanceNorm1d(1, affine=True)
    with torch.no_grad():
        norm.weight.fill_(1.0)
        norm.bias.zero_()
    constant = torch.full((2, 1, 4), 5.0)
    assert torch.allclose(norm(constant), torch.zeros_like(constant))
    out = net.FeatureDiscriminator(FEATURE_DIM, norm="instance1d", leaky_slope=0.2,
                                   affine=True)(torch.randn(BATCH, FEATURE_DIM))
    assert out.shape == (BATCH, 1) and torch.isfinite(out).all()


def test_cdp_selected_only_forward_matches_full_selector_by_value():
    pool = make_pool()
    features = torch.randn(BATCH, FEATURE_DIM)
    labels = make_labels()
    selected = pool(features, labels)
    full = loss.selected_cdp_logits(pool.forward_all(features), labels)
    assert selected.shape == full.shape == (BATCH,)
    assert torch.allclose(selected, full)
    assert pool.selected_head_indices(labels) == (0, 1, 2, 4)
    assert pool.selected_head_indices(torch.tensor([3, 3], dtype=torch.long)) == (3,)


def test_cdp_label_validation_before_shape_access_and_no_fallback():
    pool = make_pool()
    with pytest.raises(net.PlanError):
        pool.selected_head_indices([0, 1])
    with pytest.raises(net.PlanError):
        pool.selected_head_indices(torch.tensor([], dtype=torch.long))
    with pytest.raises(net.PlanError):
        pool.selected_head_indices(torch.tensor([0, 1], dtype=torch.float32))
    with pytest.raises(net.PlanError):
        pool(torch.randn(BATCH - 1, FEATURE_DIM), make_labels())
    with pytest.raises(net.PlanError):
        pool(torch.full((BATCH, FEATURE_DIM), float("nan")), make_labels())
    batch_norm = make_pool("batch1d").train()
    with pytest.raises(net.PlanError, match="at least 2 rows"):
        batch_norm(torch.randn(3, FEATURE_DIM), torch.tensor([0, 1, 1]))


def test_cdp_gradients_reach_only_selected_heads_by_value():
    pool = make_pool()
    reference = copy.deepcopy(pool)
    features = torch.randn(BATCH, FEATURE_DIM)
    labels = make_labels()
    selected_features = features.clone().requires_grad_(True)
    full_features = features.clone().requires_grad_(True)
    selected_loss = loss.generator_loss(pool(selected_features, labels))
    full_loss = loss.generator_loss(
        loss.selected_cdp_logits(reference.forward_all(full_features), labels))
    selected_loss.backward()
    full_loss.backward()
    assert torch.allclose(selected_loss, full_loss)
    assert torch.allclose(selected_features.grad, full_features.grad, atol=1e-6)
    chosen = set(pool.selected_head_indices(labels))
    for category, head in enumerate(pool.heads):
        grads = [p.grad for p in head.parameters()]
        if category in chosen:
            assert any(g is not None and g.abs().sum() > 0 for g in grads)
        else:
            assert all(g is None for g in grads)
    # The full-selector reference exercise every head by value.


def test_cdp_unselected_batchnorm_buffers_untouched_with_clone_snapshots():
    pool = make_pool("batch1d").train()
    features = torch.randn(BATCH, FEATURE_DIM)
    labels = torch.tensor([0, 2, 0, 2], dtype=torch.long)
    before = [clone_state(head) for head in pool.heads]
    pool(features, labels)
    selected = set(pool.selected_head_indices(labels))
    for category, head in enumerate(pool.heads):
        after = head.state_dict()
        for key, value in before[category].items():
            if category in selected:
                if "running_" in key:
                    assert not torch.equal(value, after[key]), "selected BN buffer must update"
            else:
                assert torch.equal(value, after[key]), f"unselected head {category} changed {key}"


def test_cdp_forward_all_is_inspection_only_but_shapes_agree():
    pool = make_pool()
    features = torch.randn(BATCH, FEATURE_DIM)
    assert pool.forward_all(features).shape == (BATCH, CATEGORIES)


# --- canonical MTFE geometry interface (test-only random backbone) ---------


class _RandomFrozenBackbone(nn.Module):
    """Test-only random frozen image->feature map; NOT an MTFE and not trained."""

    def __init__(self, feature_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.linear = nn.Linear(3 * 4 * 4, feature_dim)
        self.requires_grad_(False)
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.linear(self.pool(images).flatten(1))


def make_mtfe() -> mt.MTFEFeatureInterface:
    backbone = mt.FrozenBackboneInterface(
        _RandomFrozenBackbone(), feature_dim=FEATURE_DIM, input_rank=4,
        preserve_input_grad=True)
    provenance = mt.MTFEProvenance(
        source_locators=mt.PRIMARY_LOCATORS, backbone_id="test-only-random-frozen",
        feature_dim=FEATURE_DIM,
        declared_adaptations=("synthetic shape/routing test stand-in",))
    cdp_head = mt.ResizeHeadContract(source_kind="radial_scalar", target_height=4,
                                     target_width=4, mode="bilinear", align_corners=True)
    atl_head = mt.ResizeHeadContract(source_kind="direction_map", source_height=4,
                                     source_width=4, target_height=4, target_width=4,
                                     mode="bilinear", align_corners=True)
    return mt.MTFEFeatureInterface(
        backbone, policy=mt.SafeDivPolicy(singular_policy="reject", minimum_norm=1e-6),
        provenance=provenance, cdp_head=cdp_head, atl_head=atl_head)


def test_mtfe_geometry_interface_preserves_input_grad_not_a_trained_mtfe():
    mtfe = make_mtfe()
    plan, generator = make_generator()
    image, _, _ = generator(make_source(), make_labels())
    assert image.requires_grad
    outputs = mtfe.forward(image)
    # CDP path uses the declared 1x4x4 age map flattened; a declared synthetic wiring.
    assert outputs.cdp_age_map.shape == (BATCH, 1, 4, 4)
    assert outputs.cdp_age_map.flatten(1).shape == (BATCH, FEATURE_DIM)
    assert outputs.atl_identity.shape == (BATCH, FEATURE_DIM)
    assert outputs.features.requires_grad and outputs.cdp_age_map.requires_grad
    assert mtfe.provenance.mechanism == "geometric_decomposition"
    assert mtfe.provenance.trained_status == "provenance_unverified"


# --- gradient routing -------------------------------------------------------


def test_gstep_with_frozen_image_d_and_feature_pool_and_mtfe_only_reaches_g():
    plan, generator = make_generator()
    image_d = net.ImageDiscriminator((3, 8, 16, 32, 64, 1), leaky_slope=0.2, affine=True)
    pool = make_pool()
    mtfe = make_mtfe()
    for module in (image_d, pool):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()
    assert all(not p.requires_grad for p in image_d.parameters())
    assert all(not p.requires_grad for p in pool.parameters())

    real = make_source(1).requires_grad_(True)
    fake, _, _ = generator(make_source(), make_labels())
    outputs = mtfe.forward(fake)
    feature_logits = pool(outputs.cdp_age_map.flatten(1), make_labels())
    feature_loss = loss.generator_loss(feature_logits)
    triplet_loss = _make_triplet_loss(mtfe, real, outputs)
    image_loss = loss.generator_loss(image_d(fake))
    overall = loss.overall_generator_loss(image_loss, feature_loss, triplet_loss["loss"],
                                          lambda_feature=1.0, lambda_at=0.001)
    overall.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in generator.parameters())
    for module in (image_d, pool, mtfe.backbone.module):
        assert all(p.grad is None for p in module.parameters())


def _make_triplet_loss(mtfe, real, fake_outputs):
    func = mtfe.forward
    anchor = func(real).atl_identity
    built = torch.Generator().manual_seed(2)
    banks_same = torch.randn(3, FEATURE_DIM, generator=built)
    banks_target = torch.randn(3, FEATURE_DIM, generator=built)
    # Per-anchor different-identity eligibility: drop one gallery column per anchor.
    keep = torch.ones(BATCH, 3, dtype=torch.bool)
    for row in range(BATCH):
        keep[row, row % 3] = False
    return loss.aofs_adversarial_triplet(
        anchor, fake_outputs.atl_identity, banks_same, banks_target, keep, keep,
        margin=0.2, detach_context=True)


def test_full_synthetic_generator_loss_eq3_eq7_eq8_eq9_has_owner_grads():
    _, generator = make_generator()
    image_d = net.ImageDiscriminator((3, 8, 16, 32, 64, 1), leaky_slope=0.2, affine=True)
    pool = make_pool()
    mtfe = make_mtfe()
    for module in (image_d, pool):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    real = make_source(1).requires_grad_(True)
    fake, _, _ = generator(make_source(), make_labels())
    outputs = mtfe.forward(fake)
    feature_loss = loss.generator_loss(pool(outputs.cdp_age_map.flatten(1), make_labels()))
    triplet = _make_triplet_loss(mtfe, real, outputs)
    image_loss = loss.generator_loss(image_d(fake))
    assert float(feature_loss.detach()) != 0.0 and float(triplet["loss"].detach()) != 0.0
    overall = loss.overall_generator_loss(image_loss, feature_loss, triplet["loss"],
                                          lambda_feature=1.0, lambda_at=0.001)
    expected = image_loss + 1.0 * feature_loss + 0.001 * triplet["loss"]
    assert torch.allclose(overall, expected)
    overall.backward()
    assert all(p.grad is not None for p in generator.parameters())
    assert all(p.grad is None for p in image_d.parameters())


def test_dstep_reaches_image_d_only_with_detached_fake():
    _, generator = make_generator()
    image_d = net.ImageDiscriminator((3, 8, 16, 32, 64, 1), leaky_slope=0.2, affine=True)
    real = make_source(1)
    fake, _, _ = generator(make_source(), make_labels())
    value = loss.discriminator_loss(image_d(real), image_d(fake.detach()))
    value.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in image_d.parameters())
    assert all(p.grad is None for p in generator.parameters())


def test_no_optimizer_steps_or_parameter_mutation():
    _, generator = make_generator()
    before = [p.detach().clone() for p in generator.parameters()]
    out, _, _ = generator(make_source(), make_labels())
    out.sum().backward()
    assert all(p.grad is not None for p in generator.parameters())
    assert all(torch.equal(b, p.detach())
               for b, p in zip(before, generator.parameters(), strict=True))


def test_cdp_late_singleton_refusal_changes_no_earlier_head_buffers():
    pool = make_pool("batch1d").train()
    before = clone_state(pool)
    with pytest.raises(net.PlanError, match="at least 2 rows"):
        pool(torch.randn(3, FEATURE_DIM), torch.tensor([0, 0, 1]))
    after = pool.state_dict()
    assert all(torch.equal(value, after[key]) for key, value in before.items())


def test_identity_skip_checks_geometry_after_both_convolutions():
    plan = replace(make_plan(), residual_strides=(1,) * 6, skip="identity",
                   residual_conv=net.ConvSpec(3, 1, 0))
    with pytest.raises(net.PlanError, match="identity skip geometry mismatch"):
        plan._check_skip_alignment()


def test_known_tip_four_group_regime_refuses_five_categories():
    with pytest.raises(net.PlanError, match="exactly four"):
        make_plan(age_regime=net.AGE_REGIME_TIP2021)


def test_residual_constructor_refuses_unsupported_skip_before_forward():
    with pytest.raises(net.PlanError, match="supported skip"):
        net.ResidualBlock(8, 8, net.ConvSpec(3, 1, 1), 1,
                          "unverified", "instance2d", True, "pre_add_relu")


def test_metadata_preserves_unknown_skip_order_and_named_block_count():
    plan = make_plan()
    metadata = plan.assert_metadata_complete()
    assert "not printed" in metadata["residual_activation_order"]
    assert metadata["residual_blocks"].startswith("source")
    assert replace(plan, residual_blocks=5).choice_metadata()["residual_blocks"].startswith("adaptation")


def test_image_discriminator_rejects_collapsed_instance_norm_before_forward():
    disc = net.ImageDiscriminator((3, 4, 4, 4, 4, 1), leaky_slope=0.2, affine=True)
    with pytest.raises(net.PlanError, match="spatial area"):
        disc(torch.randn(2, 3, 8, 8))


def test_stride_two_kernel_one_transpose_accepts_valid_output_padding():
    assert net.transpose_out(4, net.ConvSpec(1, 2, 0), 1) == 8


@pytest.mark.parametrize("blocks", [True, -1, 0])
def test_literal_trace_requires_positive_integer_block_count(blocks):
    with pytest.raises(net.PlanError):
        net.literal_table_trace(residual_blocks=blocks)
