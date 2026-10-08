"""Synthetic forward/backward tests for the v4 enforced-freeze MTFE proposal.

These tests pin the v4 corrections: construction-time stage/freeze refusal, a
freeze policy that is physically true (no parameter gradients, BatchNorm statistics
frozen even under a caller `train()`), gradients still reaching the INPUT, the
age-head input contract, and the added shape/dtype/finite guards. No optimizer, no
optimizer.step(), no weights, no real face images, no training.

Run:
    uv run python -m pytest \
      .work/pi-workers/20261008-517e6042/cacon-mtfe-component-v1/supervisor_v3/proposal_v4/tests/test_cacon_mtfe_supervision_v4.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts import cacon_mtfe_supervision_v4 as V4

MODULE_PATH = Path(V4.__file__)
REJECT = V4.SafeDivPolicy(singular_policy="reject", minimum_norm=1e-6)


def _banks():
    return V4.IdentityBankDeclaration(
        anchor="source face identity feature",
        positive="synthesized same-identity other-age feature",
        negative="different-identity feature")


def _config(**overrides):
    base = dict(
        feature_dim=8,
        num_identity_classes=4,
        margin=0.3,
        policy=REJECT,
        identity_sampling=_banks(),
        stage=V4.TrainingStage.MTFE_PRETRAINING,
        freeze_policy=V4.FreezePolicy.TRAINABLE,
        task_weight_age=1.0,
        task_weight_identity=1.0,
        radial_layout="scalar",
    )
    base.update(overrides)
    return V4.MTFEConfig(**base)


def _age_head(classes=4, dim=1):
    return V4.AgeRegressionHead(input_dim=dim, num_age_classes=classes,
                                age_centers=(20.0, 35.0, 45.0, 60.0)[:classes])


def _module(*, feature_dim=8, target_dim=16, freeze=None, stage=None, backbone=None,
            radial_layout="scalar"):
    config = _config(
        feature_dim=feature_dim,
        radial_layout=radial_layout,
        stage=stage or V4.TrainingStage.MTFE_PRETRAINING,
        freeze_policy=freeze or V4.FreezePolicy.TRAINABLE)
    return V4.MTFEModule(
        config,
        age_head=_age_head(),
        resize_head=V4.LearnedResizeHead(in_dim=1, target_dim=target_dim),
        identity_resize_head=V4.LearnedResizeHead(in_dim=feature_dim, target_dim=32),
        backbone=backbone,
    )


class _BNBackbone(nn.Module):
    """Tiny caller-owned backbone exposing BatchNorm (the AOFS-freeze hazard)."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(width)
        self.projection = nn.Linear(width, width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(images))


# --- construction-time stage/freeze refusal ----------------------------------

def test_frozen_mtfe_pretraining_config_is_refused_at_construction():
    with pytest.raises(ValueError, match="REFUSED"):
        _config(freeze_policy=V4.FreezePolicy.FROZEN)


def test_stage_freeze_check_is_wired_into_post_init():
    config = _config()
    assert "declared adaptation" in config.check_stage_freeze_consistency()
    aofs = _config(stage=V4.TrainingStage.AOFS_GENERATION,
                   freeze_policy=V4.FreezePolicy.FROZEN)
    assert "aofs_generation" in aofs.check_stage_freeze_consistency()
    assert "frozen" in aofs.check_stage_freeze_consistency()


def test_construction_refuses_missing_or_bad_declarations():
    with pytest.raises(ValueError, match="FreezePolicy"):
        _config(freeze_policy="frozen")
    with pytest.raises(ValueError, match="TrainingStage"):
        _config(stage="mtfe_pretraining")
    with pytest.raises(ValueError, match="radial_layout"):
        _config(radial_layout="tensor")
    with pytest.raises(ValueError, match="IdentityBankDeclaration"):
        _config(identity_sampling="anchor/positive/negative")
    with pytest.raises(ValueError, match="margin"):
        _config(margin=-1.0)
    with pytest.raises(ValueError, match="task_weight_age"):
        _config(task_weight_age=0.0)
    with pytest.raises(ValueError, match="SafeDivPolicy"):
        _config(policy="reject")


def test_identity_banks_must_be_nonempty_and_pairwise_distinct():
    for name in ("anchor", "positive", "negative"):
        kwargs = dict(anchor="a", positive="b", negative="c")
        kwargs[name] = "   "
        with pytest.raises(ValueError, match="nonempty"):
            V4.IdentityBankDeclaration(**kwargs)
    for left, right in (("anchor", "positive"), ("anchor", "negative"),
                        ("positive", "negative")):
        kwargs = dict(anchor="same", positive="same", negative="other")
        kwargs[left] = "Same"
        kwargs[right] = "same"
        with pytest.raises(ValueError, match="different banks"):
            V4.IdentityBankDeclaration(**kwargs)


# --- the age head contract ---------------------------------------------------

def test_age_head_input_dim_must_be_one_for_both_layouts():
    with pytest.raises(ValueError, match="input_dim must be 1"):
        _age_head(dim=2)
    for layout in ("scalar", "column"):
        module = _module(radial_layout=layout)
        module.forward_features(torch.randn(3, 8, dtype=torch.float32))


def test_age_head_hidden_dims_must_be_positive_non_bool_ints():
    with pytest.raises(ValueError, match="tuple"):
        _age_head_with(hidden_dims=[4])
    with pytest.raises(ValueError, match="hidden_dims"):
        _age_head_with(hidden_dims=(0,))
    with pytest.raises(ValueError, match="hidden_dims"):
        _age_head_with(hidden_dims=(4.0,))
    with pytest.raises(ValueError, match="hidden_dims"):
        _age_head_with(hidden_dims=(True,))
    head = _age_head_with(hidden_dims=(6, 4))
    assert [layer.out_features for layer in head.mlp] == [6, 4, 4]


def _age_head_with(*, hidden_dims, classes=4):
    return V4.AgeRegressionHead(input_dim=1, num_age_classes=classes,
                                age_centers=(20.0, 35.0, 45.0, 60.0)[:classes],
                                hidden_dims=hidden_dims)


def test_age_head_refuses_bad_centers_and_classes():
    with pytest.raises(ValueError, match=">= 2"):
        V4.AgeRegressionHead(input_dim=1, num_age_classes=1, age_centers=(10.0,))
    with pytest.raises(ValueError, match="one age center"):
        V4.AgeRegressionHead(input_dim=1, num_age_classes=3, age_centers=(10.0, 20.0))
    with pytest.raises(ValueError, match="nonnegative"):
        V4.AgeRegressionHead(input_dim=1, num_age_classes=2, age_centers=(-1.0, 20.0))
    with pytest.raises(ValueError, match="increasing"):
        V4.AgeRegressionHead(input_dim=1, num_age_classes=2, age_centers=(30.0, 20.0))


# --- reduction / target / objective guards -----------------------------------

def test_reduction_must_be_explicitly_valid():
    module = _module()
    outputs = module.forward_features(torch.randn(5, 8))
    target = torch.full((5,), 45.0)
    assert module.age_term(outputs, target, kind="l1", reduction="none").shape == (5,)
    for reduction in ("mean", "sum"):
        assert module.age_term(outputs, target, kind="l2", reduction=reduction).ndim == 0
    for bad in ("avg", "", None, "MEAN"):
        with pytest.raises(ValueError, match="reduction must be exactly"):
            module.age_term(outputs, target, kind="l1", reduction=bad)
    with pytest.raises(ValueError, match="kind"):
        module.age_term(outputs, target, kind="l3", reduction="mean")


def test_target_age_must_be_nonnegative_finite_and_aligned():
    module = _module()
    outputs = module.forward_features(torch.randn(4, 8))
    with pytest.raises(ValueError, match="nonnegative"):
        module.age_term(outputs, torch.full((4,), -1.0), kind="l1")
    with pytest.raises(ValueError, match="finite"):
        module.age_term(outputs, torch.tensor([1.0, 2.0, float("nan"), 4.0]), kind="l1")
    with pytest.raises(ValueError, match="share a shape"):
        module.age_term(outputs, torch.full((3,), 40.0), kind="l1")
    with pytest.raises(ValueError, match="dtype"):
        module.age_term(outputs, torch.full((4,), 40.0, dtype=torch.float64), kind="l1")


def test_multi_task_objective_validates_shape_dtype_device_and_finiteness():
    module = _module()
    age = torch.tensor(2.0)
    identity = torch.tensor(6.0)
    assert torch.isclose(module.mtfe_objective(age, identity), torch.tensor(8.0))
    with pytest.raises(ValueError, match="scalar"):
        module.mtfe_objective(torch.ones(2), identity)
    with pytest.raises(ValueError, match="dtype"):
        module.mtfe_objective(age, torch.tensor(6.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        module.mtfe_objective(torch.tensor(float("nan")), identity)
    # Finite inputs whose weighted sum overflows the dtype must be refused.
    with pytest.raises(ValueError, match="finite"):
        module.mtfe_objective(torch.tensor(3e38), torch.tensor(3e38))


def test_weights_scale_the_two_terms():
    module = _module()
    module.config = _config(task_weight_age=2.0, task_weight_identity=3.0)
    value = module.mtfe_objective(torch.tensor(1.0), torch.tensor(1.0))
    assert torch.isclose(value, torch.tensor(5.0))


# --- frozen policy is physically enforced ------------------------------------

def test_frozen_policy_leaves_no_trainable_parameters_and_keeps_eval():
    module = _module(freeze=V4.FreezePolicy.FROZEN,
                     stage=V4.TrainingStage.AOFS_GENERATION)
    # A caller asking for training mode cannot defeat the declared freeze.
    module.train()
    assert module.trainable_parameters() == ()
    for sub in (module.age_head, module.resize_head, module.identity_resize_head):
        assert not sub.training
    module.apply_declared_freeze_policy()
    assert module.trainable_parameters() == ()


def test_frozen_policy_gives_no_parameter_grads_but_keeps_input_gradient():
    # FusionNet's published ReLU-first readout can zero a batch whose age scores are
    # all negative; seed so the non-degeneracy of the INPUT gradient is informative.
    torch.manual_seed(0)
    module = _module(freeze=V4.FreezePolicy.FROZEN,
                     stage=V4.TrainingStage.AOFS_GENERATION)
    features = torch.randn(6, 8, requires_grad=True)
    outputs = module.forward_features(features)
    module.age_term(outputs, torch.full((6,), 45.0), kind="l1").backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is None, f"frozen parameter received grad: {name}"
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0, "a frozen MTFE must still supervise its input"
    assert outputs.age_prediction.requires_grad
    assert outputs.age_prediction.grad_fn is not None


def test_frozen_backbone_bn_stats_do_not_change_even_under_train():
    backbone = _BNBackbone()
    module = _module(freeze=V4.FreezePolicy.FROZEN,
                     stage=V4.TrainingStage.AOFS_GENERATION, backbone=backbone)
    module.train()
    images = torch.randn(4, 8, requires_grad=True)
    before = {name: buf.clone() for name, buf in backbone.named_buffers()}
    outputs = module(images)
    outputs.age_prediction.sum().backward()
    after = {name: buf for name, buf in backbone.named_buffers()}
    assert before, "the BN backbone must expose running buffers"
    for name, value in before.items():
        assert torch.equal(value, after[name]), f"frozen BN buffer mutated: {name}"
    assert not backbone.norm.training
    for parameter in backbone.parameters():
        assert parameter.grad is None
    assert images.grad is not None and images.grad.abs().sum() > 0


def test_trainable_policy_keeps_parameters_and_bn_stats_trainable():
    torch.manual_seed(0)
    backbone = _BNBackbone()
    module = _module(backbone=backbone)
    module.train()
    assert len(module.trainable_parameters()) == len(tuple(module.parameters()))
    images = torch.randn(4, 8, requires_grad=True)
    module(images).age_prediction.sum().backward()
    assert backbone.norm.running_mean.abs().sum() > 0, "trainable BN must update stats"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in module.age_head.parameters())


def test_frozen_consistency_is_rechecked_every_forward():
    module = _module(freeze=V4.FreezePolicy.FROZEN,
                     stage=V4.TrainingStage.AOFS_GENERATION)
    module.forward_features(torch.randn(3, 8))
    # Simulate external tampering: a frozen parameter must not silently become live.
    next(module.age_head.parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError, match="FROZEN policy violated"):
        module.forward_features(torch.randn(3, 8))
    # A caller train() also restores and re-validates the declared policy.
    module.train()
    module.forward_features(torch.randn(3, 8))


# --- still no training machinery ---------------------------------------------

def test_module_contains_no_optimizer_or_training_calls():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            assert name != "step", "optimizer step in module"
            assert name != "backward", "backward call in module"
            assert name not in ("Adam", "SGD", "AdamW"), f"optimizer in module: {name}"
    assert not any(name.startswith("torch.optim") for name in imported), imported


def test_provenance_strings_do_not_claim_trained_or_pretrained_weights():
    text = MODULE_PATH.read_text(encoding="utf-8").lower()
    for banned in ("trained_weights", "pretrained_weights", "load_state_dict",
                   "torch.load", "checkpoint"):
        assert banned not in text, f"unattested model provenance: {banned}"
    assert "no trained/pre-trained provenance is asserted" in " ".join(V4.V4_NON_CLAIMS)


def test_nested_bn_tampering_refused_before_backbone_buffer_mutation():
    backbone = _BNBackbone()
    module = _module(freeze=V4.FreezePolicy.FROZEN,
                     stage=V4.TrainingStage.AOFS_GENERATION, backbone=backbone)
    backbone.norm.train()  # Parent still reports eval; descendant must also be checked.
    before = {name: value.clone() for name, value in backbone.named_buffers()}
    with pytest.raises(RuntimeError, match="FROZEN policy violated"):
        module(torch.randn(4, 8))
    for name, value in backbone.named_buffers():
        assert torch.equal(value, before[name]), name


def test_freezing_clears_stale_backbone_parameter_gradients():
    backbone = _BNBackbone()
    for parameter in backbone.parameters():
        parameter.grad = torch.ones_like(parameter)
    module = _module(freeze=V4.FreezePolicy.FROZEN,
                     stage=V4.TrainingStage.AOFS_GENERATION, backbone=backbone)
    assert all(parameter.grad is None for parameter in module.parameters())


def test_declared_adaptation_cannot_be_relabelled_author_fact():
    with pytest.raises(TypeError):
        V4.IdentityBankDeclaration("a", "b", "c", declared_adaptation=False)


def test_age_loss_overflow_refused_before_objective_composition():
    head = _age_head()
    with pytest.raises(ValueError, match="finite"):
        head.loss(torch.tensor([3e38]), torch.tensor([0.0]), kind="l2")


def test_canonical_component_has_no_worker_tree_dependency():
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "_V3" not in text and "_v3_module" not in text
    assert "proposal/scripts" not in text and ".work/" not in text


def test_frozen_running_buffer_autograd_tampering_refused():
    backbone = _BNBackbone()
    module = _module(freeze=V4.FreezePolicy.FROZEN,
                     stage=V4.TrainingStage.AOFS_GENERATION, backbone=backbone)
    backbone.norm.running_mean.requires_grad_(True)
    with pytest.raises(RuntimeError, match="buffer.*requires grad"):
        module(torch.randn(4, 8))
