import copy

import pytest
import torch
from torch import nn

from scripts.mtlface_joint_epoch_v3 import JointEpochFailure, run_joint_epoch
from tests.test_mtlface_target_stream_v3 import fixture as target_fixture
from tests.test_smoke_mtlface_joint_v3 import Discriminator, Generator, Recognizer


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(2)
    model, head, gen, disc = Recognizer(), nn.Linear(3, 8), Generator(), Discriminator()
    fr_opt = torch.optim.SGD([*model.parameters(), *head.parameters()], lr=0.001)
    g_opt = torch.optim.SGD(gen.parameters(), lr=0.001)
    d_opt = torch.optim.SGD(disc.parameters(), lr=0.001)
    stream = target_fixture(tmp_path)
    kwargs = dict(
        batch_size=3, source_rng=torch.Generator().manual_seed(43), device="cpu",
        generator_bn_policy="frozen", encoder_probe=model.stem,
    )
    return (model, head, gen, disc, fr_opt, g_opt, d_opt, stream), kwargs


def test_full_epoch_actual_steps_and_entry_exposures(setup):
    args, kwargs = setup
    before = [copy.deepcopy(m.state_dict()) for m in args[:4]]
    reports = []
    result = run_joint_epoch(*args, **kwargs, progress=lambda ledger: reports.append(copy.deepcopy(ledger)))
    ledger = result["ledger"]
    assert ledger["epoch_complete"] and not ledger["training_complete"]
    assert ledger["completed_batches"] == 3
    assert ledger["source_unique_rows"] == ledger["source_decoded_views"] == 8
    assert ledger["target_sampled_draws"] == ledger["target_decoded_views"] == 8
    assert ledger["optimizer_steps"] == dict(fr=3, generator=3, discriminator=3)
    assert ledger["optimizer_step_attempts"] == ledger["optimizer_steps"]
    assert ledger["fr_updated_state_finite"] is True
    assert ledger["fr_updated_state_checks"] == 3
    assert ledger["entry_forward_images"] == dict(
        encoder_fr=8, encoder_fas=16, generator_fas=8, discriminator_fas=24,
    )
    assert [r["completed_source_samples"] for r in reports] == [3, 6, 8]
    for module, prior in zip(args[:4], before, strict=True):
        assert any(not torch.equal(value, prior[name]) for name, value in module.state_dict().items())
    assert not args[0].stem._forward_hooks
    assert not args[2]._forward_hooks and not args[3]._forward_hooks
    assert all(not opt._optimizer_step_post_hooks for opt in args[4:7])
    assert all(not opt._optimizer_step_pre_hooks for opt in args[4:7])


def test_rng_and_target_ledger_continue_across_epochs(setup):
    args, kwargs = setup
    first = run_joint_epoch(*args, **kwargs)["ledger"]
    second = run_joint_epoch(*args, **kwargs)["ledger"]
    assert second["target_stream_before"] == first["target_stream_after"]
    assert sum(second["target_stream_after"]["sampled_row_draws_by_group"]) == 16
    assert second["target_decoded_views"] == 8
    assert second["source_rng_initial_sha256"] == first["source_rng_current_sha256"]


def test_generator_failure_preserves_observed_partial_steps(setup, monkeypatch):
    args, kwargs = setup

    def fail():
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(args[5], "step", fail)
    with pytest.raises(JointEpochFailure, match="injected") as info:
        run_joint_epoch(*args, **kwargs)
    ledger = info.value.ledger
    assert not ledger["epoch_complete"]
    assert ledger["completed_batches"] == 0
    assert ledger["optimizer_steps"] == dict(fr=1, discriminator=1, generator=0)
    assert ledger["optimizer_step_attempts"] == dict(fr=1, discriminator=1, generator=1)
    assert ledger["source_decoded_views"] == ledger["target_decoded_views"] == 3
    assert ledger["entry_forward_images"] == dict(
        encoder_fr=3, encoder_fas=6, generator_fas=3, discriminator_fas=9,
    )
    assert not args[0].stem._forward_hooks
    assert all(not opt._optimizer_step_post_hooks for opt in args[4:7])


def test_changed_source_crop_fails_before_training(setup):
    args, kwargs = setup
    for row in args[-1].records:
        row.path.write_bytes(b"changed")
    with pytest.raises(JointEpochFailure, match="native binding") as info:
        run_joint_epoch(*args, **kwargs)
    assert info.value.ledger["optimizer_steps"] == dict(fr=0, generator=0, discriminator=0)
    assert info.value.ledger["source_decoded_views"] == 0


@pytest.mark.parametrize("key,value", [("batch_size", 0), ("batch_size", True), ("generator_bn_policy", "implicit"), ("gan_weight", float("nan")), ("progress", True)])
def test_invalid_contract_refused_without_mutation(setup, key, value):
    args, kwargs = setup
    kwargs[key] = value
    before = copy.deepcopy(args[0].state_dict())
    with pytest.raises((ValueError, TypeError)):
        run_joint_epoch(*args, **kwargs)
    assert all(torch.equal(v, before[k]) for k, v in args[0].state_dict().items())
    assert args[-1].ledger()["batch_partition"] == []


def test_shared_rng_refused(setup):
    args, kwargs = setup
    kwargs["source_rng"] = args[-1].generator
    with pytest.raises(ValueError, match="distinct"):
        run_joint_epoch(*args, **kwargs)


def test_wrong_optimizer_refused_before_fr_update(setup):
    args, kwargs = setup
    args = list(args)
    args[5] = torch.optim.SGD(args[0].parameters(), lr=0.001)
    with pytest.raises(ValueError, match="own"):
        run_joint_epoch(*args, **kwargs)
    assert args[-1].ledger()["batch_partition"] == []


def test_progress_failure_is_partial_even_after_valid_batch(setup):
    args, kwargs = setup

    def fail(ledger):
        raise RuntimeError("progress storage failed")

    with pytest.raises(JointEpochFailure, match="storage") as info:
        run_joint_epoch(*args, **kwargs, progress=fail)
    assert info.value.ledger["completed_batches"] == 1
    assert not info.value.ledger["epoch_complete"]


def test_unobserved_encoder_probe_refuses_false_coverage(setup):
    args, kwargs = setup
    # Belongs to model, but called for age loss only, not encoder entry.
    kwargs["encoder_probe"] = args[0].age_head.group
    with pytest.raises(JointEpochFailure, match="exposure mismatch"):
        run_joint_epoch(*args, **kwargs)


def test_progress_cannot_mutate_internal_ledger(setup):
    args, kwargs = setup

    def mutate(snapshot):
        snapshot["optimizer_steps"]["fr"] = -500
        snapshot["entry_forward_images"]["encoder_fr"] = -500

    result = run_joint_epoch(*args, **kwargs, progress=mutate)
    assert result["ledger"]["optimizer_steps"]["fr"] == 3
    assert result["ledger"]["entry_forward_images"]["encoder_fr"] == 8


def test_target_decode_failure_does_not_first_update_fr(setup, monkeypatch):
    args, kwargs = setup

    def fail(count):
        args[-1].draw_indices(count)
        raise RuntimeError("target decode failed")

    monkeypatch.setattr(args[-1], "draw_images", fail)
    with pytest.raises(JointEpochFailure, match="target decode") as info:
        run_joint_epoch(*args, **kwargs)
    assert info.value.ledger["optimizer_steps"] == dict(fr=0, generator=0, discriminator=0)
    assert info.value.ledger["source_decoded_views"] == 3
    assert info.value.ledger["target_sampled_draws"] == 3
    assert info.value.ledger["target_decoded_views"] == 0


def test_fr_optimizer_failure_does_not_credit_step(setup, monkeypatch):
    args, kwargs = setup

    def fail():
        raise RuntimeError("FR optimizer failed")

    monkeypatch.setattr(args[4], "step", fail)
    with pytest.raises(JointEpochFailure, match="FR optimizer") as info:
        run_joint_epoch(*args, **kwargs)
    assert info.value.ledger["optimizer_steps"] == dict(fr=0, generator=0, discriminator=0)
    assert info.value.ledger["entry_forward_images"]["encoder_fr"] == 3
    assert info.value.ledger["target_decoded_views"] == 3


def test_returned_fr_step_with_nonfinite_state_refused_before_fas(setup, monkeypatch):
    args, kwargs = setup
    original = args[4].step

    def corrupt():
        original()
        with torch.no_grad():
            args[0].stem.weight.fill_(float("nan"))

    monkeypatch.setattr(args[4], "step", corrupt)
    with pytest.raises(JointEpochFailure, match="nonfinite FR updated state") as info:
        run_joint_epoch(*args, **kwargs)
    ledger = info.value.ledger
    assert ledger["optimizer_steps"] == dict(fr=1, generator=0, discriminator=0)
    assert ledger["fr_updated_state_finite"] is False
    assert not ledger["epoch_complete"]
