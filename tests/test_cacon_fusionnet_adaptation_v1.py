"""Focused synthetic tests for the lean canonical-ready FusionNet shape adaptation.

The module and its canonical readout helper use normal package imports.
No scratch directories or ancestor-depth assumptions are used.

No weights, optimizer, optimizer step, training loop, real face image or GPU. Tiny CPU
forward/backward only. Run from the repository root:

    python -m pytest tests/test_cacon_fusionnet_adaptation_v1.py -q
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest
import torch
import torch.nn.functional as torch_functional
from torch import nn

import scripts.cacon_mtfe_components_v1 as canonical
from scripts import cacon_fusionnet_adaptation_v1 as PORT

MODULE_PATH = Path(PORT.__file__)

BOXES = ((0, 0, 12, 12), (20, 30, 10, 14), (40, 12, 16, 16), (60, 60, 12, 12), (10, 70, 18, 10))


def _config(**overrides):
    base = dict(bottleneck_channels=(8, 8, 8, 8, 8), block_depths=(2, 1, 1, 1, 1),
                num_age_classes=3, age_centers=(20.0, 40.0, 60.0), face_stem_kernel=7,
                patch_stem_kernel=5, stem_kind="plain_norm_conv")
    base.update(overrides)
    return PORT.FusionNetAdaptationConfig(**base)


def _net(**overrides):
    return PORT.FusionNetAdaptation(_config(**overrides))


def _inputs(batch=2):
    face = torch.randn(batch, 1, PORT.FACE_CROP, PORT.FACE_CROP)
    return PORT.make_inputs(face, BOXES)


def _bn_snapshot(net):
    snapshot = {}
    for name, module in net.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            snapshot[name] = (int(module.num_batches_tracked),
                              module.running_mean.detach().clone(),
                              module.running_var.detach().clone())
    assert snapshot, "the network must contain BatchNorm layers for these tests to mean anything"
    return snapshot


def _bn_same(left, right):
    return left.keys() == right.keys() and all(
        left[name][0] == right[name][0]
        and torch.equal(left[name][1], right[name][1])
        and torch.equal(left[name][2], right[name][2])
        for name in left)


def _count_bn_calls(net):
    counter = {"calls": 0}
    handles = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            handles.append(module.register_forward_hook(
                lambda *_: counter.__setitem__("calls", counter["calls"] + 1)))
    return counter, handles


# --- isolated import contract --------------------------------------------------

def test_canonical_helper_is_the_real_package_function():
    assert PORT.expectation_age_readout is canonical.expectation_age_readout
    assert PORT.expectation_age_readout.__module__ == "scripts.cacon_mtfe_components_v1"


def test_module_uses_a_plain_package_import_and_no_scratch_paths():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert "scripts.cacon_mtfe_components_v1" in imported
    assert imported <= {"__future__", "dataclasses", "math", "torch", "torch.nn",
                        "torch.nn.functional", "scripts.cacon_mtfe_components_v1"}, imported
    for banned in ("sys.path", "importlib", "parents[", "../raw", ".work", "os.environ"):
        assert banned not in source, f"scratch/ancestor lookup leaked into the module: {banned}"


# --- provenance tags: factual vs derived vs declared vs unresolved -------------

def test_contract_tags_separate_facts_from_choices_and_unknowns():
    assert PORT.SOURCE_FACT == "source_fact"
    assert PORT.DERIVED_CONSTRAINT == "derived_constraint"
    assert PORT.DECLARED_CHOICE == "declared_choice"
    assert PORT.SOURCE_UNRESOLVED == "source_unresolved"
    for locator, _substance, tag in PORT.PRIMARY_LOCATORS[2:]:
        assert tag == PORT.SOURCE_FACT, locator
    assert "wrap.warwick.ac.uk" in PORT.PRIMARY_LOCATORS[0][2]
    assert PORT.PRIMARY_LOCATORS[1][2] == (
        "1e60275a91317c27cecc5e3c7887ad0bfafaf5dd511c0a3719b7fd4d47e2d25b")
    for name, tag, _detail in PORT.UNRESOLVED_ITEMS:
        assert tag in (PORT.SOURCE_UNRESOLVED, PORT.DERIVED_CONSTRAINT), name
    tags = {tag for _name, tag, _detail in PORT.UNRESOLVED_ITEMS}
    assert PORT.SOURCE_UNRESOLVED in tags
    assert PORT.DERIVED_CONSTRAINT in tags
    declared = dict(PORT.DECLARED_CHOICES)
    for key in ("stem kind", "stem kernels", "block depths", "bottleneck widths", "activation",
                "output width policy"):
        assert key in declared, key
    assert "32" in dict((n, d) for n, _t, d in PORT.UNRESOLVED_ITEMS)["patch contribution width"]


def test_unresolved_blue_marks_and_stem_status_are_recorded_never_widths():
    unresolved = dict((name, (tag, detail)) for name, tag, detail in PORT.UNRESOLVED_ITEMS)
    assert unresolved["blue face-branch mark '7'"][0] == PORT.SOURCE_UNRESOLVED
    assert unresolved["blue patch-branch mark '5'"][0] == PORT.SOURCE_UNRESOLVED
    assert "never used as a width" in unresolved["blue face-branch mark '7'"][1]
    assert unresolved["stem residual status"][0] == PORT.SOURCE_UNRESOLVED
    assert 5 not in PORT.OBSERVED_MAIN_STREAM_WIDTHS + (PORT.PATCH_CONTRIBUTION,)
    assert 7 not in PORT.OBSERVED_MAIN_STREAM_WIDTHS + (PORT.PATCH_CONTRIBUTION,)


# --- observed stream is hard architecture --------------------------------------

def test_observed_widths_and_spatial_are_hard_and_not_configurable():
    assert PORT.OBSERVED_MAIN_STREAM_WIDTHS == (32, 64, 96, 128, 160, 192)
    assert PORT.OBSERVED_SPATIAL == 24
    assert PORT.PATCH_CONTRIBUTION == 32
    net = _net()
    assert net.realized_widths() == PORT.OBSERVED_MAIN_STREAM_WIDTHS
    assert net.classifier.in_features == 192
    assert net.face_stem.target_size == 24
    assert net.features(_inputs()).shape == (2, 192)
    fields = {field.name for field in dataclasses.fields(PORT.FusionNetAdaptationConfig)}
    for banned in ("block_channels", "block_spatial_sizes", "face_stem_channels",
                   "patch_stem_channels", "b5_feature_reduction", "main_stream_widths"):
        assert banned not in fields, banned


def test_b5_no_reduction_is_structural_and_nothing_sits_before_pooling():
    net = _net()
    for index, block in enumerate(net.blocks):
        assert block.out_channels == block.concat_channels
        assert block.concat_channels == PORT.OBSERVED_MAIN_STREAM_WIDTHS[index + 1]
    assert not hasattr(net, "b5_feature_reduction")
    assert len([m for m in net.modules() if isinstance(m, nn.Linear)]) == 1
    assert list(net._modules) == ["face_stem", "patch_stems", "blocks", "pool", "classifier"]


# --- Eq.5 concatenation before the block, and BN-before-conv everywhere --------

def test_block_consumes_the_eq5_concatenation_before_its_units():
    net = _net().eval()
    inputs = _inputs()
    captured = {}

    def hook(_module, args):
        captured["input"] = args[0].detach().clone()

    handle = net.blocks[0].units[0].reduce.register_forward_pre_hook(hook)
    with torch.no_grad():
        previous = net.face_stem(inputs.face)
        patch_map = net.patch_stems[0](inputs.patches[0])
        first_block = net.blocks[0](previous, patch_map)
    handle.remove()
    assert torch.allclose(captured["input"], torch.cat([previous, patch_map], dim=1))
    assert captured["input"].shape[1] == 32 + PORT.PATCH_CONTRIBUTION
    assert first_block.shape[1] == 64
    assert not any("refresh" in name for name in dict(net.named_modules()))


def test_every_convolution_is_batchnorm_then_relu_then_conv_without_bias():
    net = _net()
    wrappers = [m for m in net.modules() if isinstance(m, PORT._BNConv)]
    convs = [m for m in net.modules() if isinstance(m, nn.Conv2d)]
    assert wrappers and len(wrappers) == len(convs)
    for wrapper in wrappers:
        assert isinstance(wrapper[0], nn.BatchNorm2d)
        assert isinstance(wrapper[1], nn.ReLU)
        assert isinstance(wrapper[2], nn.Conv2d)
        assert wrapper[2].bias is None


def test_patch_stream_i_cannot_influence_an_earlier_block():
    net = _net()
    net.train()
    inputs = _inputs()
    for patch in inputs.patches:
        patch.requires_grad_(True)
    first = net.blocks[0](net.face_stem(inputs.face),
                          net.patch_stems[0](inputs.patches[0]))
    first.sum().backward()
    assert inputs.patches[4].grad is None, "P5 must not reach B1"


# --- raw crop geometry ---------------------------------------------------------

def test_crop_patches_allows_zero_coordinates_and_rejects_bad_boxes():
    face = torch.randn(1, 1, PORT.FACE_CROP, PORT.FACE_CROP)
    patches = PORT.crop_patches(face, ((0, 0, 24, 24),) + BOXES[1:])
    assert torch.allclose(patches[0], face[:, :, 0:24, 0:24])
    assert patches[0].shape == (1, 1, PORT.PATCH_CROP, PORT.PATCH_CROP)
    expected = torch_functional.interpolate(face[:, :, 20:30, 30:44], size=(24, 24),
                                            mode="bilinear", align_corners=False)
    assert torch.allclose(patches[1], expected)
    with pytest.raises(ValueError, match=">= 1"):
        PORT.crop_patches(face, ((0, 0, 0, 24),) + BOXES[1:])
    with pytest.raises(ValueError, match="must be >= 0"):
        PORT.crop_patches(face, ((-1, 0, 12, 12),) + BOXES[1:])
    with pytest.raises(TypeError, match="bool is not an integer"):
        PORT.crop_patches(face, ((True, 0, 12, 12),) + BOXES[1:])
    with pytest.raises(ValueError, match="falls outside"):
        PORT.crop_patches(face, ((90, 10, 12, 12),) + BOXES[1:])
    with pytest.raises(ValueError, match="exactly 5 patch boxes"):
        PORT.crop_patches(face, BOXES[:4])


# --- entry guards: mutable input, mixed dtype, BN untouched --------------------

def test_mutated_raw_input_fails_before_any_batchnorm_runs():
    net = _net()
    net.train()
    inputs = _inputs(batch=1)
    before = _bn_snapshot(net)
    counter, handles = _count_bn_calls(net)
    inputs.face[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        net(inputs)
    inputs.face[0, 0, 0, 0] = 0.0
    inputs.patches[4][0, 0, 3, 3] = float("inf")
    with pytest.raises(ValueError, match=r"patches\[4\] must contain only finite values"):
        net.forward(inputs)
    for handle in handles:
        handle.remove()
    assert counter["calls"] == 0, "BatchNorm ran on a mutated raw input"
    after = _bn_snapshot(net)
    assert _bn_same(before, after)
    for name, (_counter, mean, var) in after.items():
        assert torch.isfinite(mean).all() and torch.isfinite(var).all(), name


def test_live_dtype_and_shape_mutation_fail_before_any_batchnorm_runs():
    net = _net()
    inputs = _inputs()
    counter, handles = _count_bn_calls(net)
    inputs.face.data = inputs.face.data.double()
    for patch in inputs.patches:
        patch.data = patch.data.double()
    with pytest.raises(TypeError, match="must match the reference dtype"):
        net(inputs)
    inputs.face.data = inputs.face.data.float().reshape(2, 1, 48, 192)
    with pytest.raises(ValueError, match="96x96"):
        net(inputs)
    for handle in handles:
        handle.remove()
    assert counter["calls"] == 0
    assert _bn_snapshot(net)["face_stem.norm_conv.0"][0] == 0


def test_heterogeneous_module_dtype_is_rejected_with_all_buffers_unchanged():
    net = _net()
    net.train()
    target = next(m for m in net.modules() if isinstance(m, nn.BatchNorm2d))
    target.running_mean.data = target.running_mean.data.double()
    before = _bn_snapshot(net)
    assert before[list(before)[0]][1].dtype == torch.float64, "the corruption must be real"
    counter, handles = _count_bn_calls(net)
    with pytest.raises(TypeError, match="buffer"):
        net(_inputs())
    second = _net()
    second.train()
    with torch.no_grad():
        second.face_stem.norm_conv[2].weight.data = (
            second.face_stem.norm_conv[2].weight.data.double())
    second_before = _bn_snapshot(second)
    counter2, handles2 = _count_bn_calls(second)
    with pytest.raises(TypeError, match="parameter"):
        second(_inputs())
    for handle in [*handles, *handles2]:
        handle.remove()
    assert counter["calls"] == 0 and counter2["calls"] == 0
    assert _bn_same(before, _bn_snapshot(net))
    assert _bn_same(second_before, _bn_snapshot(second))


def test_inputs_refuse_geometry_nonfinite_and_plain_tensors():
    face = torch.randn(2, 1, PORT.FACE_CROP, PORT.FACE_CROP)
    good = PORT.crop_patches(face, BOXES)
    with pytest.raises(ValueError, match="gray-scale"):
        PORT.FusionNetInputs(face=torch.randn(2, 3, 96, 96), patches=good)
    with pytest.raises(ValueError, match="96x96"):
        PORT.FusionNetInputs(face=torch.randn(2, 1, 64, 64), patches=good)
    with pytest.raises(TypeError, match="tuple of 5 tensors"):
        PORT.FusionNetInputs(face=face, patches=list(good))
    with pytest.raises(ValueError, match="exactly 5 patch inputs"):
        PORT.FusionNetInputs(face=face, patches=good[:4])
    with pytest.raises(ValueError, match="nonempty"):
        PORT.FusionNetInputs(face=torch.randn(0, 1, 96, 96),
                             patches=tuple(torch.randn(0, 1, 24, 24) for _ in range(5)))
    with pytest.raises(ValueError, match="finite"):
        PORT.FusionNetInputs(face=torch.full((2, 1, 96, 96), float("nan")), patches=good)
    with pytest.raises(TypeError, match="FusionNetInputs"):
        _net()(torch.randn(2, 1, 96, 96))


# --- config is a frozen dataclass revalidated before modules/RNG ----------------

def test_config_is_a_frozen_dataclass_and_sets_are_refused():
    config = _config()
    assert dataclasses.is_dataclass(config)
    assert config.__dataclass_params__.frozen is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.stem_kind = "residual_bottleneck"
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.age_centers = (1.0, 2.0, 3.0)


def test_config_tamper_is_revalidated_before_any_module_or_rng_use():
    config = _config()
    object.__setattr__(config, "age_centers", (20.0, 40.0))  # structurally invalid tamper
    torch.manual_seed(1234)
    rng_before = torch.get_rng_state()
    with pytest.raises(ValueError, match="one center per age class"):
        PORT.FusionNetAdaptation(config)
    assert torch.equal(rng_before, torch.get_rng_state()), "config was accepted far enough to " \
        "draw RNG before the revalidation rejected it"

    second = _config()
    object.__setattr__(second, "bottleneck_channels", (99, 8, 8, 8, 8))
    with pytest.raises(ValueError, match="labelled block width"):
        PORT.FusionNetAdaptation(second)

    third = _config()
    object.__setattr__(third, "bottleneck_channels", (16, 16, 16, 16, 16))
    net = PORT.FusionNetAdaptation(third)  # a valid tamper is rebuilt from the live values
    assert net.blocks[0].bottleneck_channels == 16
    assert third.bottleneck_channels == (16, 16, 16, 16, 16)
    assert net.realized_widths() == PORT.OBSERVED_MAIN_STREAM_WIDTHS


def test_frozen_config_summary_reports_exact_age_centers_for_eq6_provenance():
    summary = _config(age_centers=(18.0, 36.5, 61.0)).summary()
    assert summary["age_centers"] == (18.0, 36.5, 61.0)
    assert summary["observed_main_stream_widths"] == PORT.OBSERVED_MAIN_STREAM_WIDTHS
    assert summary["observed_spatial"] == PORT.OBSERVED_SPATIAL
    assert summary["stem_kind"] == "plain_norm_conv"


# --- direct public block: constructor and forward guards before BN -------------

def test_direct_block_constructor_is_positive_and_boolproof():
    block = PORT.FusionNetBlock(previous_channels=32, patch_channels=32,
                                bottleneck_channels=8, depth=1)
    assert block.concat_channels == 64 and block.out_channels == 64
    with pytest.raises(ValueError, match="strictly narrower"):
        PORT.FusionNetBlock(previous_channels=32, patch_channels=32,
                            bottleneck_channels=64, depth=1)
    with pytest.raises(TypeError, match="bool is not an integer"):
        PORT.FusionNetBlock(previous_channels=True, patch_channels=32,
                            bottleneck_channels=8, depth=1)
    with pytest.raises(TypeError, match="bool is not an integer"):
        PORT.FusionNetBlock(previous_channels=32, patch_channels=32,
                            bottleneck_channels=8, depth=False)
    with pytest.raises(ValueError, match=">= 1"):
        PORT.FusionNetBlock(previous_channels=32, patch_channels=0, bottleneck_channels=8,
                            depth=1)


def test_direct_block_checks_channel_spatial_batch_dtype_and_finiteness_before_bn():
    block = PORT.FusionNetBlock(previous_channels=32, patch_channels=32,
                                bottleneck_channels=8, depth=1)
    counter, handles = _count_bn_calls(block)
    previous = torch.randn(2, 32, 24, 24)
    with pytest.raises(ValueError, match="must have 32 channels"):
        block(torch.randn(2, 16, 24, 24), torch.randn(2, 32, 24, 24))
    with pytest.raises(ValueError, match="must have 32 channels"):
        block(previous, torch.randn(2, 16, 24, 24))
    with pytest.raises(ValueError, match="one spatial size"):
        block(previous, torch.randn(2, 32, 12, 12))
    with pytest.raises(ValueError, match="batch dimension"):
        block(previous, torch.randn(3, 32, 24, 24))
    with pytest.raises(TypeError, match="must match the reference dtype"):
        block(previous, torch.randn(2, 32, 24, 24, dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        block(previous.clone().fill_(float("nan")), torch.randn(2, 32, 24, 24))
    with pytest.raises(ValueError, match="finite"):
        block(previous, torch.full((2, 32, 24, 24), float("inf")))
    for handle in handles:
        handle.remove()
    assert counter["calls"] == 0, "BatchNorm ran while rejecting malformed block inputs"
    assert _bn_snapshot(block)["units.0.reduce.0"][0] == 0


def test_direct_block_rejects_module_input_dtype_and_device_mismatch_before_bn():
    block = PORT.FusionNetBlock(previous_channels=32, patch_channels=32,
                                bottleneck_channels=8, depth=1)
    block.train()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.data = parameter.data.double()
        for buffer in block.buffers():
            if buffer.is_floating_point():
                buffer.data = buffer.data.double()
    before = _bn_snapshot(block)
    counter, handles = _count_bn_calls(block)
    with pytest.raises(TypeError, match="parameter"):
        block(torch.randn(2, 32, 24, 24), torch.randn(2, 32, 24, 24))
    for handle in handles:
        handle.remove()
    assert counter["calls"] == 0, "BatchNorm ran on a module/input dtype mismatch"
    assert _bn_same(before, _bn_snapshot(block))
    assert torch.isfinite(_bn_snapshot(block)["units.0.reduce.0"][1]).all()


def test_direct_block_rejects_a_mixed_buffer_and_leaves_every_buffer_unchanged():
    block = PORT.FusionNetBlock(previous_channels=32, patch_channels=32,
                                bottleneck_channels=8, depth=1)
    block.train()
    target = next(m for m in block.modules() if isinstance(m, nn.BatchNorm2d))
    target.running_var.data = target.running_var.data.double()
    before = _bn_snapshot(block)
    assert any(snapshot[2].dtype == torch.float64 for snapshot in before.values())
    counter, handles = _count_bn_calls(block)
    with pytest.raises(TypeError, match="buffer"):
        block(torch.randn(2, 32, 24, 24), torch.randn(2, 32, 24, 24))
    for handle in handles:
        handle.remove()
    assert counter["calls"] == 0
    assert _bn_same(before, _bn_snapshot(block))


# --- declarations: stems, depths, bottlenecks, kernels, classes ----------------

def test_mandatory_declarations_have_no_defaults_and_both_stem_kinds_work():
    for name in ("stem_kind", "bottleneck_channels", "block_depths", "num_age_classes",
                 "age_centers", "face_stem_kernel", "patch_stem_kernel"):
        assert inspect.signature(PORT.FusionNetAdaptationConfig).parameters[name].default \
            is inspect.Parameter.empty, name
    block_parameters = inspect.signature(PORT.FusionNetBlock.__init__).parameters
    for name in ("previous_channels", "patch_channels", "bottleneck_channels", "depth"):
        assert block_parameters[name].default is inspect.Parameter.empty, name
    for stem_kind in PORT.STEM_KINDS:
        overrides = {"stem_kind": stem_kind}
        if stem_kind == "residual_bottleneck":
            overrides["stem_bottleneck_channels"] = 8
        else:
            with pytest.raises(ValueError, match="must be None for a plain stem"):
                _config(stem_kind=stem_kind, stem_bottleneck_channels=8)
        net = _net(**overrides)
        assert net.face_stem.stem_kind == stem_kind
        assert (net.face_stem.residual is not None) == (stem_kind == "residual_bottleneck")
        assert net.realized_widths() == PORT.OBSERVED_MAIN_STREAM_WIDTHS
        inputs = _inputs(batch=1)
        inputs.face.requires_grad_(True)
        net(inputs).logits.sum().backward()
        assert inputs.face.grad is not None and inputs.face.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="there is no default"):
        _config(stem_kind="residual_bottleneck")
    with pytest.raises(ValueError, match="stem_kind must be one of"):
        _config(stem_kind="residual")


def test_config_rejects_bool_nonfinite_wrong_arity_even_kernels_and_wide_bottlenecks():
    with pytest.raises(TypeError, match="bool is not an integer"):
        _config(block_depths=(True, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="exactly 5 entries"):
        _config(block_depths=(1, 1, 1, 1))
    with pytest.raises(ValueError, match="finite"):
        _config(age_centers=(float("nan"), 40.0, 60.0))
    with pytest.raises(TypeError, match="real number"):
        _config(age_centers=(True, 40.0, 60.0))
    with pytest.raises(ValueError, match="nonnegative"):
        _config(age_centers=(-5.0, 40.0, 60.0))
    with pytest.raises(ValueError, match="one center per age class"):
        _config(num_age_classes=3, age_centers=(20.0, 40.0))
    with pytest.raises(ValueError, match="num_age_classes"):
        _config(num_age_classes=1, age_centers=(30.0,))
    with pytest.raises(ValueError, match="must be odd"):
        _config(patch_stem_kernel=4)
    with pytest.raises(ValueError, match="labelled block width"):
        _config(bottleneck_channels=(64, 8, 8, 8, 8))
    with pytest.raises(TypeError, match="FusionNetAdaptationConfig"):
        PORT.FusionNetAdaptation(object())


# --- gradients and Eq.6 readout ------------------------------------------------

def test_gradients_reach_the_raw_face_every_patch_and_the_classifier():
    net = _net()
    net.train()
    inputs = _inputs()
    inputs.face.requires_grad_(True)
    for patch in inputs.patches:
        patch.requires_grad_(True)
    net(inputs).logits.sum().backward()
    assert inputs.face.grad is not None and inputs.face.grad.abs().sum() > 0
    for index, patch in enumerate(inputs.patches):
        assert patch.grad is not None and patch.grad.abs().sum() > 0, f"patch P{index + 1}"
    assert net.classifier.weight.grad.abs().sum() > 0
    for stem in net.patch_stems:
        assert stem.norm_conv[2].weight.grad.abs().sum() > 0
    for block in net.blocks:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())


def test_readout_uses_the_canonical_eq6_helper_and_handles_all_negative_scores():
    net = _net()
    out = net(_inputs())
    assert torch.allclose(out.prediction,
                          canonical.expectation_age_readout(out.probabilities,
                                                            age_centers=(20.0, 40.0, 60.0)))
    assert torch.allclose(out.probabilities.sum(dim=1), torch.ones(2))
    with torch.no_grad():
        net.classifier.bias.fill_(-50.0)
        net.classifier.weight.zero_()
    uniform = net(_inputs())
    assert torch.allclose(uniform.probabilities, torch.full((2, 3), 1 / 3))
    assert torch.allclose(uniform.prediction, torch.full((2,), 40.0))


# --- no training machinery in an adaptation proposal ---------------------------

def test_module_contains_no_training_weights_or_optimizer_calls():
    source = MODULE_PATH.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            assert name not in ("step", "backward"), f"training call in module: {name}"
            assert name not in ("Adam", "SGD", "AdamW"), f"optimizer in module: {name}"
    lowered = source.lower()
    for banned in ("load_state_dict", "torch.load", "checkpoint", "pretrained",
                   "no training is permitted", "training is banned"):
        assert banned not in lowered, banned
    assert "Experiments before a committee decision are permitted" in PORT.REVIEW_GATE
    assert "release, submission and licence review" in PORT.REVIEW_GATE
    assert "no native training lineage" in PORT.REVIEW_GATE
    assert "implements none of that" in PORT.TRAINING_REFERENCE


def test_network_detaches_the_callers_config_for_stable_readout_provenance():
    config = _config()
    net = PORT.FusionNetAdaptation(config)
    assert net.config is not config
    object.__setattr__(config, "age_centers", (1.0, 2.0, 3.0))
    assert net.config.age_centers == (20.0, 40.0, 60.0)


@pytest.mark.parametrize("centers", [(1.0, 2.0, 3.0), (20.0, 40.0)])
def test_post_construction_config_corruption_fails_before_batchnorm(centers):
    net = _net()
    inputs = _inputs()
    before = _bn_snapshot(net)
    counter, handles = _count_bn_calls(net)
    object.__setattr__(net.config, "age_centers", centers)
    with pytest.raises(ValueError):
        net(inputs)
    for handle in handles:
        handle.remove()
    assert counter["calls"] == 0
    assert _bn_same(before, _bn_snapshot(net))


@pytest.mark.parametrize("target", ["parameter", "running_mean", "counter"])
def test_integral_weight_or_running_stat_and_float_counter_are_refused_before_bn(target):
    block = PORT.FusionNetBlock(previous_channels=32, patch_channels=32,
                               bottleneck_channels=8, depth=1)
    norm = next(module for module in block.modules() if isinstance(module, nn.BatchNorm2d))
    if target == "parameter":
        norm.weight = nn.Parameter(norm.weight.detach().to(torch.int64), requires_grad=False)
    elif target == "running_mean":
        norm.running_mean = norm.running_mean.to(torch.int64)
    else:
        norm.num_batches_tracked = norm.num_batches_tracked.float()
    before = _bn_snapshot(block)
    counter, handles = _count_bn_calls(block)
    with pytest.raises(TypeError):
        block(torch.randn(2, 32, 24, 24), torch.randn(2, 32, 24, 24))
    for handle in handles:
        handle.remove()
    assert counter["calls"] == 0
    assert _bn_same(before, _bn_snapshot(block))


def test_patch_geometry_is_declared_not_claimed_as_a_paper_fact():
    assert "patch geometry" not in dict(PORT.FACTUAL_CONTRACT)
    assert "patch geometry" in dict(PORT.DECLARED_CHOICES)
