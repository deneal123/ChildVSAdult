"""Focused synthetic tests for the architecture-only MTLFace FAS networks port.

Covers the FULL FAS generator + discriminator architecture reimplemented in
``mtlface_fas_networks_v2.py`` from the pinned Hzzone/MTLFace commit
``03ad57942e19ee28733b03a441765481f6606460``.

All tests are bounded, synthetic and CPU-only (1 thread), use random weights and
never download or load real weights.  Pinned full-dimension forward passes are
batch-size 1 (cheap: ~0.4 s); everything else uses the explicit test-only tiny
config or single modules.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from scripts import mtlface_fas_networks_v2 as fas

RAW = Path(os.environ["MTLFACE_REFERENCE_DIR"]) if os.environ.get("MTLFACE_REFERENCE_DIR") else None


@pytest.fixture(autouse=True)
def _bounded_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    yield


# --------------------------------------------------------------------------- #
# synthetic tensors
# --------------------------------------------------------------------------- #
def backbone_tensors(size: int = 112, batch: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    half = size // 2

    def r(*shape):
        return torch.randn(*shape, generator=g)

    return dict(
        input_img=r(batch, 3, size, size),
        x_1=r(batch, 64, size, size),
        x_2=r(batch, 64, half, half),
        x_3=r(batch, 128, half // 2, half // 2),
        x_4=r(batch, 256, half // 4, half // 4),
        x_5=r(batch, 512, half // 8, half // 8),
        x_id=r(batch, 512, half // 8, half // 8),
        x_age=r(batch, 512, half // 8, half // 8),
    )


# --------------------------------------------------------------------------- #
# provenance / claims
# --------------------------------------------------------------------------- #
def test_provenance_is_pinned_and_hashes_recorded():
    prov = fas.provenance()
    assert prov["commit"] == "03ad57942e19ee28733b03a441765481f6606460"
    assert prov["repository"] == "Hzzone/MTLFace"
    assert prov["files"]["common/networks.py"].startswith("688866a7")
    assert prov["files"]["common/ops.py"].startswith("b9058611")
    # the raw files we fetched must hash to the recorded values
    import hashlib

    for rel, digest in fas.PINNED_FILE_SHA256.items() if RAW else []:
        raw = RAW / rel.replace("/", "_")
        assert raw.exists(), f"missing raw source {raw}"
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == digest


def test_architecture_only_claims_are_explicit_and_enforced():
    flags = fas.claim_flags()
    assert flags["architecture_only"] is True
    for key in (
        "full_method",
        "training_reproduced",
        "training_parity",
        "scientific_evaluation_complete",
        "publication_ready",
    ):
        assert flags[key] is False
    fas.assert_architecture_only(flags)  # must not raise
    with pytest.raises(ValueError):
        fas.assert_architecture_only({**flags, "full_method": True})
    with pytest.raises(ValueError):
        fas.assert_architecture_only({**flags, "architecture_only": False})


def test_reference_behaviour_records_code_not_paper():
    ref = fas.REFERENCE_BEHAVIOUR
    assert ref["taskrouter.sigma"] == 0.1
    assert ref["taskrouter.unit_count"] == 128
    assert ref["taskrouter.conv_dim_age_group_7"] == 819
    assert ref["agingmodule.repeat_num"] == 4
    assert "BatchNorm2d" in ref["agingmodule.upsample_norm"]
    assert "no final tanh" in ref["agingmodule.output"]
    assert ref["patchdiscriminator.norm_layer"] == "sn"
    assert ref["patchdiscriminator.first_conv_spectral_norm"] is False
    assert fas.DECLARED_HARDENINGS  # non-empty


# --------------------------------------------------------------------------- #
# TaskRouter: conv_dim, exact masks, overlapping routes
# --------------------------------------------------------------------------- #
def test_taskrouter_conv_dim_pinned_is_819_not_paper_width():
    router = fas.TaskRouter(128, 7, 0.1)
    assert router.conv_dim == 819
    assert router._unit_mapping.shape == (7, 819)
    # last unit is never activated by any group (pinned quirk)
    assert router.always_zero_units() == [818]


def test_taskrouter_exact_masks_and_overlap():
    router = fas.TaskRouter(128, 7, 0.1)
    expected_starts = [0, 115, 230, 345, 460, 575, 690]
    for group, start in enumerate(expected_starts):
        active = router.active_units(group)
        assert active == list(range(start, start + 128)), group
    # adjacent groups overlap by 13 units
    for g in range(6):
        a, b = set(router.active_units(g)), set(router.active_units(g + 1))
        assert len(a & b) == 13
        assert len(a) == len(b) == 128
    # non-adjacent groups do not overlap
    assert not (set(router.active_units(0)) & set(router.active_units(2)))
    # a mask row sums to exactly unit_count
    assert int(router._unit_mapping[3].sum()) == 128


def test_taskrouter_zeroes_inactive_channels_per_sample():
    router = fas.TaskRouter(4, 3, 0.25)  # conv_dim = int((3-2*0.25)*4) = 10
    assert router.conv_dim == 10
    x = torch.ones(2, 10, 2, 2)
    ids = torch.tensor([0, 2], dtype=torch.int64)
    out = router(x, ids)
    for row, group in enumerate((0, 2)):
        active = set(router.active_units(group))
        for c in range(10):
            expected = 1.0 if c in active else 0.0
            assert torch.allclose(out[row, c], torch.full((2, 2), expected))


def test_taskrouter_conditioning_validation_fails_closed():
    router = fas.TaskRouter(4, 3, 0.25)
    x = torch.zeros(2, 10, 1, 1)
    with pytest.raises(TypeError):  # wrong dtype
        router(x, torch.tensor([0, 1], dtype=torch.int32))
    with pytest.raises(ValueError):  # wrong rank
        router(x, torch.tensor([[0], [1]], dtype=torch.int64))
    with pytest.raises(ValueError):  # out of range high
        router(x, torch.tensor([0, 3], dtype=torch.int64))
    with pytest.raises(ValueError):  # out of range low
        router(x, torch.tensor([-1, 1], dtype=torch.int64))
    with pytest.raises(ValueError):  # batch mismatch
        router(x, torch.tensor([0, 1, 2], dtype=torch.int64))
    with pytest.raises(ValueError):  # channel mismatch
        router(torch.zeros(2, 11, 1, 1), torch.tensor([0, 1], dtype=torch.int64))


# --------------------------------------------------------------------------- #
# ResidualBlock
# --------------------------------------------------------------------------- #
def test_residual_block_routing_and_residual_shape():
    block = fas.ResidualBlock(4, 3, 0.25).eval()
    x = torch.randn(2, block.conv_dim, 5, 5)
    ids = torch.tensor([0, 1], dtype=torch.int64)
    out = block({0: x, 1: ids})
    assert set(out.keys()) == {0, 1}
    assert out[0].shape == x.shape
    assert torch.isfinite(out[0]).all()
    assert torch.equal(out[1], ids)


def test_residual_block_gradients_reach_routers_and_convs():
    block = fas.ResidualBlock(4, 3, 0.25)
    x = torch.randn(1, block.conv_dim, 3, 3, requires_grad=True)
    ids = torch.tensor([2], dtype=torch.int64)
    block({0: x, 1: ids})[0].sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = {n: p.grad for n, p in block.named_parameters()}
    assert all(g is not None for g in grads.values())
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert grads["conv1.0.weight"].abs().sum() > 0


# --------------------------------------------------------------------------- #
# get_norm_layer / group2feature (common/ops.py)
# --------------------------------------------------------------------------- #
def test_get_norm_layer_variants_and_fail_closed():
    conv = torch.nn.Conv2d(3, 5, 3, padding=1)
    assert fas.get_norm_layer("none", conv) is conv
    bn = fas.get_norm_layer("bn", torch.nn.Conv2d(3, 5, 3, padding=1))
    assert isinstance(bn, torch.nn.Sequential) and isinstance(bn[1], torch.nn.BatchNorm2d)
    inn = fas.get_norm_layer("in", torch.nn.Conv2d(3, 5, 3, padding=1))
    assert isinstance(inn[1], torch.nn.InstanceNorm2d)
    sn = fas.get_norm_layer("sn", torch.nn.Conv2d(3, 5, 3, padding=1))
    assert hasattr(sn, "weight_orig") or hasattr(sn, "parametrizations")
    with pytest.raises(NotImplementedError):  # declared hardening
        fas.get_norm_layer("bogus", conv)


def test_group2feature_shape_values_and_validation():
    feat = fas.group2feature(torch.tensor([1, 3], dtype=torch.int64), 4, 5)
    assert feat.shape == (2, 4, 5, 5)
    assert torch.allclose(feat[0].sum(), torch.tensor(25.0))
    assert torch.allclose(feat[1, 3], torch.ones(5, 5))
    assert torch.allclose(feat[1, 1], torch.zeros(5, 5))
    assert int(feat[0, 1, 0, 0]) == 1
    with pytest.raises(ValueError):
        fas.group2feature(torch.tensor([4], dtype=torch.int64), 4, 5)
    with pytest.raises(TypeError):
        fas.group2feature(torch.tensor([1.0]), 4, 5)


def test_group2onehot_scalar_branch():
    one = fas.group2onehot(torch.tensor([2], dtype=torch.int64), 4)
    assert one.shape == (1, 4)
    assert int(one[0, 2]) == 1


# --------------------------------------------------------------------------- #
# shortcut contract + no mutation
# --------------------------------------------------------------------------- #
def test_shortcut_contract_exact_pinned_dims():
    t = backbone_tensors(112)
    size = fas.validate_shortcut_contract(**t)
    assert size == 112
    # the exact channel/spatial contract required by the task brief
    assert t["x_1"].shape[1:] == (64, 112, 112)
    assert t["x_2"].shape[1:] == (64, 56, 56)
    assert t["x_3"].shape[1:] == (128, 28, 28)
    assert t["x_4"].shape[1:] == (256, 14, 14)
    assert t["x_5"].shape[1:] == (512, 7, 7)
    assert t["x_id"].shape[1:] == (512, 7, 7)
    assert t["x_age"].shape[1:] == (512, 7, 7)


def test_shortcut_contract_rejects_violations():
    base = backbone_tensors(112)
    bad = dict(base)
    bad["x_3"] = torch.randn(1, 64, 28, 28)  # wrong channels
    with pytest.raises(ValueError):
        fas.validate_shortcut_contract(**bad)
    bad = dict(base)
    bad["x_2"] = torch.randn(1, 64, 64, 64)  # wrong spatial
    with pytest.raises(ValueError):
        fas.validate_shortcut_contract(**bad)
    bad = dict(base)
    bad["x_id"] = torch.randn(2, 512, 7, 7)  # batch mismatch
    with pytest.raises(ValueError):
        fas.validate_shortcut_contract(**bad)
    bad = dict(base)
    bad["input_img"] = torch.full((1, 3, 112, 112), float("nan"))
    with pytest.raises(ValueError):
        fas.validate_shortcut_contract(**bad)


def test_generator_does_not_mutate_inputs():
    gen = fas.AgingModule().eval()
    t = backbone_tensors(112)
    keep = {k: v.clone() for k, v in t.items()}
    with torch.no_grad():
        gen(**t, condition=torch.tensor([3], dtype=torch.int64))
    for k, v in t.items():
        assert torch.equal(v, keep[k]), f"{k} was mutated"


# --------------------------------------------------------------------------- #
# full pinned generator: residual, no tanh, gradients, conditioning
# --------------------------------------------------------------------------- #
def test_generator_full_pinned_forward_residual_and_no_tanh():
    gen = fas.AgingModule().eval()
    assert gen.conv_dim == 819
    assert len(gen.transform) == 4
    t = backbone_tensors(112)
    condition = torch.tensor([3], dtype=torch.int64)
    with torch.no_grad():
        out = gen(**t, condition=condition)
    assert out.shape == (1, 3, 112, 112)
    assert torch.isfinite(out).all()
    residual = out - t["input_img"]
    assert residual.abs().max() > 0  # learned residual is applied
    # no final tanh: scale conv3 up and confirm the output can leave [-1, 1]
    with torch.no_grad():
        gen.conv3.weight.mul_(1000.0)
        blown = gen(**t, condition=condition)
    assert (blown - t["input_img"]).abs().max() > 1.0
    assert blown.abs().max() > 1.0


def test_generator_upsampler_is_bn_prelu_not_in_relu():
    gen = fas.AgingModule()
    for name in ("up_1", "up_2", "up_3", "up_4"):
        ups = getattr(gen, name)
        norm_types = {type(m) for m in ups.conv.modules()} | {type(m) for m in ups.conv2.modules()}
        assert torch.nn.BatchNorm2d in norm_types
        assert torch.nn.InstanceNorm2d not in norm_types
        assert torch.nn.PReLU in norm_types
        assert torch.nn.ReLU not in norm_types


def test_generator_gradients_flow_to_all_parameters():
    gen = fas.AgingModule().eval()
    t = backbone_tensors(112)
    for v in t.values():
        v.requires_grad_(True)
    out = gen(**t, condition=torch.tensor([3], dtype=torch.int64))
    out.sum().backward()
    assert t["x_id"].grad is not None and torch.isfinite(t["x_id"].grad).all()
    assert t["input_img"].grad is not None and t["input_img"].grad.abs().sum() > 0
    for name, p in gen.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
    assert gen.conv1[0].weight.grad.abs().sum() > 0
    assert gen.conv3.weight.grad.abs().sum() > 0
    # router1 in the first residual block must receive gradient
    assert gen.transform[0].router1._unit_mapping.grad is None  # buffer, not a param


def test_generator_conditioning_changes_output():
    gen = fas.AgingModule().eval()
    t = backbone_tensors(112)
    with torch.no_grad():
        out0 = gen(**t, condition=torch.tensor([0], dtype=torch.int64))
        out6 = gen(**t, condition=torch.tensor([6], dtype=torch.int64))
    assert out0.shape == out6.shape
    assert not torch.allclose(out0, out6), "different age groups must route differently"
    # same condition is deterministic in eval mode
    with torch.no_grad():
        again = gen(**t, condition=torch.tensor([0], dtype=torch.int64))
    assert torch.equal(out0, again)


def test_generator_pinned_ignores_x1_x5_xage():
    # Pinned code signature takes x_1/x_5/x_age but never uses them; preserve it.
    gen = fas.AgingModule().eval()
    t = backbone_tensors(112)
    condition = torch.tensor([2], dtype=torch.int64)
    with torch.no_grad():
        base = gen(**t, condition=condition)
        alt = dict(t)
        alt["x_1"] = torch.randn_like(t["x_1"])
        alt["x_5"] = torch.randn_like(t["x_5"])
        alt["x_age"] = torch.randn_like(t["x_age"])
        changed = gen(**alt, condition=condition)
    assert torch.equal(base, changed)


def test_generator_conditioning_validation_fails_closed():
    gen = fas.AgingModule().eval()
    t = backbone_tensors(112)
    with torch.no_grad():
        with pytest.raises(TypeError):
            gen(**t, condition=torch.tensor([3], dtype=torch.int32))
        with pytest.raises(ValueError):
            gen(**t, condition=torch.tensor([7], dtype=torch.int64))
        with pytest.raises(ValueError):
            gen(**t, condition=torch.tensor([-1], dtype=torch.int64))
        with pytest.raises(ValueError):
            gen(**t, condition=torch.tensor([0, 1], dtype=torch.int64))


# --------------------------------------------------------------------------- #
# PatchDiscriminator
# --------------------------------------------------------------------------- #
def test_patchd_output_shape_and_conditioning_width():
    disc = fas.PatchDiscriminator().eval()
    assert disc.conv_dim == 64
    assert disc.repeat_num == 4
    first_sn_conv = disc.main[0]
    inner = first_sn_conv[0] if isinstance(first_sn_conv, torch.nn.Sequential) else first_sn_conv
    assert inner.in_channels == 64 * 1 + 7  # conv_dim + age_group on the n==1 stage
    x = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        out = disc(x, torch.tensor([3], dtype=torch.int64))
    assert out.shape == (1, 1, 5, 5)
    assert torch.isfinite(out).all()


def test_patchd_spectral_norm_placement_matches_author():
    disc = fas.PatchDiscriminator()
    flags = disc.spectral_norm_flags()
    # first conv (conv1) is separate and not SN
    assert not (hasattr(disc.conv1, "weight_orig") or hasattr(disc.conv1, "parametrizations"))
    # every intermediate conv in main is SN; the final output conv is not
    sn_children = [i for i, c in enumerate(disc.main) if isinstance(c, torch.nn.Conv2d)]
    assert len(sn_children) >= 2
    last = sn_children[-1]
    assert flags[last] is False, "final 1-channel output conv must NOT be spectral-normalized"
    for i in sn_children[:-1]:
        assert flags[i] is True, f"main[{i}] should be spectral-normalized"


def test_patchd_input_gradient_and_output_shape():
    disc = fas.PatchDiscriminator()
    x = torch.randn(1, 3, 112, 112, requires_grad=True)
    out = disc(x, torch.tensor([3], dtype=torch.int64))
    assert out.shape == (1, 1, 5, 5)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    for name, p in disc.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_patchd_conditioning_validation_fails_closed():
    disc = fas.PatchDiscriminator().eval()
    x = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        with pytest.raises(TypeError):
            disc(x, torch.tensor([3], dtype=torch.int32))
        with pytest.raises(ValueError):
            disc(x, torch.tensor([7], dtype=torch.int64))
        with pytest.raises(ValueError):
            disc(x, torch.tensor([0, 1], dtype=torch.int64))
        with pytest.raises(ValueError):
            disc(torch.randn(1, 4, 112, 112), torch.tensor([3], dtype=torch.int64))


# --------------------------------------------------------------------------- #
# explicit config / no silent defaults
# --------------------------------------------------------------------------- #
def test_non_pinned_config_requires_explicit_test_only():
    with pytest.raises(ValueError):
        fas.FasReferenceConfig(age_group=4)
    with pytest.raises(ValueError):
        fas.FasReferenceConfig(sigma=0.125)  # paper-style sigma silently rejected
    tiny = fas.FasReferenceConfig.tiny_test_only()
    assert tiny.test_only is True
    assert tiny.conv_dim == int((4 - 3 * 0.25) * 4) == 13
    gen = fas.AgingModule(age_group=4, config=tiny)
    assert gen.conv_dim == 13
    assert len(gen.transform) == 2


def test_tiny_config_generator_and_discriminator_run():
    tiny = fas.FasReferenceConfig.tiny_test_only()
    gen = fas.AgingModule(age_group=4, config=tiny).eval()
    disc = fas.PatchDiscriminator(config=tiny).eval()
    t = backbone_tensors(16, seed=1)
    cond = torch.tensor([2], dtype=torch.int64)
    with torch.no_grad():
        out = gen(**t, condition=cond)
        logits = disc(t["input_img"], cond)
    assert out.shape == (1, 3, 16, 16)
    assert logits.dim() == 4 and logits.shape[0] == 1 and logits.shape[1] == 1
    assert torch.isfinite(out).all() and torch.isfinite(logits).all()


# --------------------------------------------------------------------------- #
# bounded self-check
# --------------------------------------------------------------------------- #
def test_self_check_reports_pinned_architecture():
    report = fas.self_check(size=112, seed=0)
    assert report["conv_dim"] == 819
    assert report["output_shape"] == (1, 3, 112, 112)
    assert report["logits_shape"] == (1, 1, 5, 5)
    assert report["output_finite"] and report["logits_finite"]
    assert report["residual_nonzero"] is True
    assert report["always_zero_units"] == [818]
    assert report["claims"]["architecture_only"] is True
    assert report["claims"]["training_parity"] is False


# --------------------------------------------------------------------------- #
# differential equivalence against the pinned raw sources
# --------------------------------------------------------------------------- #
def _load_pinned_raw():
    """Load the two pinned raw files into a synthetic ``common`` package.

    The pinned ``ops.py`` imports ``torch._six`` (removed in torch>=2.0), so a
    compatibility shim is installed only for the duration of the load.
    """
    if RAW is None:
        pytest.skip("set MTLFACE_REFERENCE_DIR to opt in to checked upstream execution")
    for rel, digest in fas.PINNED_FILE_SHA256.items():
        assert hashlib.sha256((RAW / rel.replace("/", "_")).read_bytes()).hexdigest() == digest
    saved = {
        k: sys.modules.get(k) for k in ("torch._six", "common", "common.ops", "common.networks")
    }
    if "torch._six" not in sys.modules:
        import collections.abc as cabc

        shim = types.ModuleType("torch._six")
        shim.container_abcs = cabc
        shim.string_classes = (str, bytes)
        sys.modules["torch._six"] = shim

    pkg = types.ModuleType("common")
    pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["common"] = pkg

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, RAW / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    try:
        ops = load("common.ops", "common_ops.py")
        pkg.ops = ops  # type: ignore[attr-defined]
        net = load("common.networks", "common_networks.py")
    except Exception:
        _restore(saved)
        raise
    return net, saved


def _restore(saved):
    for key, value in saved.items():
        if value is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value


def test_differential_against_pinned_raw_generator_and_discriminator():
    net, saved = _load_pinned_raw()
    try:
        torch.manual_seed(1234)
        ours_gen = fas.AgingModule().eval()
        torch.manual_seed(1234)
        ref_gen = net.AgingModule(age_group=7).eval()
        # exact weight transfer proves structural equivalence
        ours_gen.load_state_dict(ref_gen.state_dict())

        t = backbone_tensors(112, seed=7)
        cond = torch.tensor([5], dtype=torch.int64)
        with torch.no_grad():
            ours = ours_gen(**t, condition=cond)
            ref = ref_gen(
                *[t[k] for k in ("input_img", "x_1", "x_2", "x_3", "x_4", "x_5", "x_id", "x_age")],
                cond,
            )
        assert ours.shape == ref.shape
        assert torch.allclose(ours, ref, atol=1e-6, rtol=1e-5), (ours - ref).abs().max()

        torch.manual_seed(99)
        ours_d = fas.PatchDiscriminator().eval()
        torch.manual_seed(99)
        ref_d = net.PatchDiscriminator(7, norm_layer="sn", repeat_num=4).eval()
        ours_d.load_state_dict(ref_d.state_dict())
        x = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            lo = ours_d(x, cond)
            lr = ref_d(x, cond)
        assert lo.shape == lr.shape == (1, 1, 5, 5)
        assert torch.allclose(lo, lr, atol=1e-6, rtol=1e-5), (lo - lr).abs().max()
    finally:
        _restore(saved)


def test_differential_taskrouter_masks():
    net, saved = _load_pinned_raw()
    try:
        ref = net.TaskRouter(128, 7, 0.1)
        ours = fas.TaskRouter(128, 7, 0.1)
        # the pinned class exposes no conv_dim attribute; compare the mapping itself
        assert ref._unit_mapping.shape == ours._unit_mapping.shape == (7, 819)
        assert ours.conv_dim == 819
        assert torch.equal(ref._unit_mapping, ours._unit_mapping)
    finally:
        _restore(saved)


@pytest.mark.parametrize("shape", [(2, 10), (2, 10, 2), (0, 10, 2, 2)])
def test_router_invalid_rank_or_empty_refused(shape):
    router = fas.TaskRouter(4, 3, 0.25)
    with pytest.raises(ValueError):
        router(torch.ones(shape), torch.zeros(shape[0], dtype=torch.int64))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unit_count": True},
        {"unit_count": 4.5},
        {"test_only": "false"},
        {"patchd_conv_dim": 0},
        {"sigma": True},
    ],
)
def test_config_invalid_types_refused(kwargs):
    with pytest.raises(ValueError):
        fas.FasReferenceConfig(**kwargs)


def test_discriminator_cannot_silently_depart_from_pinned_config():
    with pytest.raises(ValueError):
        fas.PatchDiscriminator(conv_dim=4)
    with pytest.raises(ValueError):
        fas.PatchDiscriminator(repeat_num=3)
    with pytest.raises(ValueError):
        fas.PatchDiscriminator(config=fas.FasReferenceConfig.tiny_test_only(), conv_dim=64)


def test_shortcuts_mixed_dtype_and_integer_input_refused():
    tensors = backbone_tensors()
    tensors["x_1"] = tensors["x_1"].double()
    with pytest.raises(ValueError, match="dtype/device"):
        fas.validate_shortcut_contract(**tensors)
    tensors = backbone_tensors()
    tensors["input_img"] = tensors["input_img"].long()
    with pytest.raises(ValueError, match="floating"):
        fas.validate_shortcut_contract(**tensors)


def test_full_pinned_networks_connect_to_actual_joint_adapter_and_fas_step():
    from scripts.mtlface_components_v2 import MapAgeHead, SpatialAgeIdentitySplit
    from scripts.mtlface_fas_training_v2 import fas_step
    from scripts.mtlface_joint_adapter_v2 import CommonJointAdapter

    backbone = torch.nn.Module()
    backbone.input_size = 112
    backbone.preprocess = lambda image: image
    net = backbone.net = torch.nn.Module()
    net.conv1 = torch.nn.Conv2d(3, 64, 1)
    net.bn1 = torch.nn.BatchNorm2d(64)
    net.prelu = torch.nn.PReLU(64)
    for name, before, after in (
        ("layer1", 64, 64),
        ("layer2", 64, 128),
        ("layer3", 128, 256),
        ("layer4", 256, 512),
    ):
        setattr(net, name, torch.nn.Conv2d(before, after, 2, stride=2))
    net.bn2 = torch.nn.BatchNorm2d(512)
    net.dropout = torch.nn.Dropout()
    net.fc = torch.nn.Linear(512 * 7 * 7, 5)
    net.features = torch.nn.BatchNorm1d(5)
    recognizer = CommonJointAdapter(
        backbone, SpatialAgeIdentitySplit(512), MapAgeHead(512, 7), MapAgeHead(512, 7)
    )
    generator, discriminator = fas.AgingModule(), fas.PatchDiscriminator()
    g_opt = torch.optim.SGD(generator.parameters(), lr=1e-5)
    d_opt = torch.optim.SGD(discriminator.parameters(), lr=1e-5)
    before_g = generator.conv3.weight.detach().clone()
    before_d = discriminator.conv1.weight.detach().clone()
    before_r = net.conv1.weight.detach().clone()
    result = fas_step(
        recognizer,
        generator,
        discriminator,
        g_opt,
        d_opt,
        torch.randn(1, 3, 112, 112),
        torch.randn(1, 3, 112, 112),
        torch.tensor([3]),
        generator_bn_policy="frozen",
    )
    assert result["generator_updated"] and result["discriminator_updated"]
    assert not torch.equal(generator.conv3.weight, before_g)
    assert not torch.equal(discriminator.conv1.weight, before_d)
    assert torch.equal(net.conv1.weight, before_r)
    assert result["generator_gradient_norm"] > 0
