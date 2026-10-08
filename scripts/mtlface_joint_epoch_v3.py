"""Full source epoch, FR then FAS, with observed steps/exposures.

This is the declared common-backbone adaptation, not paper/budget parity.
Probe counters describe completed entry-module forwards, not FLOPs or successful
whole-model evaluations. Failures preserve a partial ledger and never rollback.
"""

from __future__ import annotations

import copy
import hashlib
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.mtlface_fas_training_v2 import fas_step
from scripts.mtlface_target_stream_v3 import TargetAgeStream
from scripts.mtlface_training_v2 import recognition_step


class JointEpochFailure(RuntimeError):
    def __init__(self, message, ledger):
        super().__init__(message)
        self.ledger = copy.deepcopy(ledger)


def _optimizer_owns(modules, optimizer):
    expected = {id(p) for m in modules for p in m.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if not expected or set(actual) != expected or len(actual) != len(expected):
        raise ValueError("optimizer must own each trainable parameter exactly once")


def _finite_modules(modules):
    for module in modules:
        for value in module.state_dict().values():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise FloatingPointError("nonfinite FR updated state")


class _ObservedOptimizer:
    """Delegate unchanged operations; count attempts even for custom step methods.

    No optimizer method/hook is replaced, and the caller retains the original
    optimizer for serialization. A thrown step may have partially mutated state.
    """

    def __init__(self, optimizer, role, ledger):
        self.optimizer, self.role, self.ledger = optimizer, role, ledger

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def step(self, *args, **kwargs):
        self.ledger["optimizer_step_attempts"][self.role] += 1
        result = self.optimizer.step(*args, **kwargs)
        self.ledger["optimizer_steps"][self.role] += 1
        return result


class _SourceViews(Dataset):
    def __init__(self, stream, ledger):
        self.stream = stream
        self.ledger = ledger
        self.seen = set()

    def __len__(self):
        return len(self.stream.dataset)

    def __getitem__(self, index):
        self.stream._guard()
        row = self.stream.records[index]
        self.stream._verify(row.path.resolve())
        item = self.stream.dataset[index]
        self.stream._verify(row.path.resolve())
        self.stream._guard()
        if item[1] != row.identity or item[2] != (-1 if row.age is None else row.age):
            raise ValueError("source decoded metadata changed")
        self.seen.add(index)
        self.ledger["source_decoded_views"] += 1
        self.ledger["source_unique_rows"] = len(self.seen)
        return item


def run_joint_epoch(
    model,
    head,
    generator,
    discriminator,
    fr_optimizer,
    g_optimizer,
    d_optimizer,
    target_stream,
    *,
    batch_size,
    source_rng,
    device,
    generator_bn_policy,
    encoder_probe=None,
    gan_weight=75.0,
    identity_weight=0.002,
    age_weight=10.0,
    progress=None,
):
    """One source pass, no drop/early selection; caller retains both RNGs.

    Targets are decoded before FR, so invalid target data do not first mutate FR.
    Valid batches perform FR then D then G. All optimizers/probes are validated
    before mutation. A failed epoch is partial, even after the last optimizer step.
    """
    if not isinstance(target_stream, TargetAgeStream):
        raise TypeError("native-bound TargetAgeStream required")
    target_stream._guard()
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("positive integer batch size required")
    if (
        not isinstance(source_rng, torch.Generator)
        or source_rng.device.type != "cpu"
        or source_rng is target_stream.generator
    ):
        raise ValueError("distinct caller-owned CPU source and target RNGs required")
    if generator_bn_policy not in ("adapt", "frozen"):
        raise ValueError("explicit generator BN policy required")
    weights = (gan_weight, identity_weight, age_weight)
    if any(
        isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
        for w in weights
    ) or not any(weights):
        raise ValueError("finite nonnegative loss weights, at least one positive, required")
    if progress is not None and not callable(progress):
        raise TypeError("callable progress callback required")
    _optimizer_owns((model, head), fr_optimizer)
    _optimizer_owns((generator,), g_optimizer)
    _optimizer_owns((discriminator,), d_optimizer)
    owners = [
        {id(p) for m in modules for p in (*m.parameters(), *m.buffers())}
        for modules in ((model, head), (generator,), (discriminator,))
    ]
    if any(owners[i] & owners[j] for i in range(3) for j in range(i)):
        raise ValueError("FR/G/D state ownership must be disjoint")
    if encoder_probe is None:
        encoder_probe = model.backbone.net.conv1
    if not isinstance(encoder_probe, nn.Module) or not any(
        encoder_probe is child for child in model.modules()
    ):
        raise ValueError("encoder entry probe must belong to recognizer")
    initial_target = target_stream.ledger()
    initial_source_rng = hashlib.sha256(source_rng.get_state().numpy().tobytes()).hexdigest()
    ledger = dict(
        completed_batches=0,
        completed_source_samples=0,
        last_batch_size=0,
        last_batch_summary=None,
        source_decoded_views=0,
        source_unique_rows=0,
        optimizer_steps=dict(fr=0, generator=0, discriminator=0),
        optimizer_step_attempts=dict(fr=0, generator=0, discriminator=0),
        optimizer_step_units="returned step calls; exceptions may contain uncounted partial mutations",
        fr_updated_state_finite=None,
        fr_updated_state_checks=0,
        entry_forward_images=dict(encoder_fr=0, encoder_fas=0, generator_fas=0, discriminator_fas=0),
        epoch_complete=False,
        training_complete=False,
        scientific_evaluation_complete=False,
    )
    phase = "idle"
    handles = []
    totals = dict(recognition={}, fas={})

    def forward_hook(role):
        def hook(module, args, output):
            if phase not in ("fr", "fas") or not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("unaccountable entry forward")
            key = role + "_" + phase
            if key not in ledger["entry_forward_images"]:
                raise RuntimeError("unexpected model role/phase")
            ledger["entry_forward_images"][key] += len(args[0])
        return hook

    def snapshot():
        current = target_stream.ledger()
        return copy.deepcopy(dict(
            **ledger,
            source_rng_initial_sha256=initial_source_rng,
            source_rng_current_sha256=hashlib.sha256(
                source_rng.get_state().numpy().tobytes()
            ).hexdigest(),
            target_stream_before=initial_target,
            target_stream_after=current,
            target_sampled_draws=sum(current["sampled_row_draws_by_group"])
            - sum(initial_target["sampled_row_draws_by_group"]),
            target_decoded_views=sum(current["successfully_decoded_views_by_group"])
            - sum(initial_target["successfully_decoded_views_by_group"]),
        ))

    try:
        for module, role in (
            (encoder_probe, "encoder"),
            (generator, "generator"),
            (discriminator, "discriminator"),
        ):
            handles.append(module.register_forward_hook(forward_hook(role)))
        observed_fr = _ObservedOptimizer(fr_optimizer, "fr", ledger)
        observed_g = _ObservedOptimizer(g_optimizer, "generator", ledger)
        observed_d = _ObservedOptimizer(d_optimizer, "discriminator", ledger)
        source = _SourceViews(target_stream, ledger)
        loader = DataLoader(
            source, batch_size=batch_size, shuffle=True, generator=source_rng,
            num_workers=0, drop_last=False,
        )
        for images, identities, ages in loader:
            size = len(images)
            targets, groups = target_stream.draw_images(size)
            batch = tuple(value.to(device) for value in (images, identities, ages))
            phase = "fr"
            ledger["fr_updated_state_finite"] = None
            fr = recognition_step(model, head, observed_fr, batch)
            try:
                _finite_modules((model, head))
            except FloatingPointError:
                ledger["fr_updated_state_finite"] = False
                raise
            ledger["fr_updated_state_finite"] = True
            ledger["fr_updated_state_checks"] += 1
            phase = "fas"
            fas = fas_step(
                model, generator, discriminator, observed_g, observed_d,
                batch[0], targets.to(device), groups.to(device),
                generator_bn_policy=generator_bn_policy,
                gan_weight=gan_weight, identity_weight=identity_weight, age_weight=age_weight,
            )
            phase = "idle"
            for stage, result in (("recognition", fr), ("fas", fas)):
                for name, value in result.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    if not math.isfinite(value):
                        raise FloatingPointError("nonfinite step summary")
                    totals[stage][name] = totals[stage].get(name, 0.0) + value * size
            ledger["completed_batches"] += 1
            ledger["completed_source_samples"] += size
            ledger["last_batch_size"] = size
            ledger["last_batch_summary"] = dict(recognition=fr, fas=fas)
            if progress is not None:
                progress(snapshot())
        samples = len(source)
        batches = math.ceil(samples / batch_size)
        if (
            ledger["source_unique_rows"] != samples
            or ledger["source_decoded_views"] != samples
            or ledger["completed_source_samples"] != samples
            or ledger["completed_batches"] != batches
            or ledger["optimizer_steps"] != dict(fr=batches, generator=batches, discriminator=batches)
            or ledger["optimizer_step_attempts"] != ledger["optimizer_steps"]
            or ledger["fr_updated_state_checks"] != batches
            or ledger["entry_forward_images"] != dict(
                encoder_fr=samples, encoder_fas=2 * samples,
                generator_fas=samples, discriminator_fas=3 * samples,
            )
            or snapshot()["target_decoded_views"] != samples
        ):
            raise RuntimeError("observed full-epoch coverage/step/exposure mismatch")
        ledger["epoch_complete"] = True
        return dict(
            ledger=snapshot(),
            sample_weighted_means={
                stage: {name: value / samples for name, value in values.items()}
                for stage, values in totals.items()
            },
        )
    except Exception as error:
        ledger["epoch_complete"] = False
        # Snapshot failure must not hide the original training failure.
        try:
            partial = snapshot()
        except Exception:
            partial = dict(**ledger, target_ledger_unavailable=True)
        raise JointEpochFailure(str(error), partial) from error
    finally:
        for handle in handles:
            handle.remove()
