"""Explicit alternating FAS steps; no full-system training/parity claim.

Loss reference: Hzzone/MTLFace@03ad579, models/fas.py::FAS.train.
Common adaptation: freeze recognizer parameters during FAS, but retain the
generated-image gradient through it; freeze D parameters during the G step.
Caller owns optimizers/LR/betas and an explicit generator BN policy.
Failed G steps can follow a completed D step: failures expose partial updates,
not a rollback or a completed training cell.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class FASStepFailure(RuntimeError):
    def __init__(self, message, *, discriminator_updated, generator_updated):
        super().__init__(message)
        self.discriminator_updated = discriminator_updated
        self.generator_updated = generator_updated


def _owned(module, optimizer):
    expected = {id(p) for p in module.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if not expected or set(actual) != expected or len(actual) != len(expected):
        raise ValueError("optimizer must own each trainable parameter exactly once")


@contextmanager
def _frozen(module, *, evaluation):
    parameters = [(p, p.requires_grad) for p in module.parameters()]
    modes = [(child, child.training) for child in module.modules()]
    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        if evaluation:
            module.eval()
        yield
    finally:
        for parameter, flag in parameters:
            parameter.requires_grad_(flag)
        for child, flag in modes:
            child.training = flag


def _finite(value, name):
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise FloatingPointError(f"nonfinite {name}")


def _patch_logits(value, batch_size):
    _finite(value, "discriminator logits")
    if value.ndim != 4 or value.shape[:2] != (batch_size, 1) or min(value.shape[2:]) < 1:
        raise ValueError("B1HW discriminator logits required")


def _finite_state(module):
    for value in module.state_dict().values():
        if value.is_floating_point():
            _finite(value, "updated model state")


def _backward(loss, optimizer):
    _finite(loss, "loss")
    if loss.ndim != 0:
        raise ValueError("scalar loss required")
    loss.backward()
    squared = 0.0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                _finite(parameter.grad, "gradient")
                squared += parameter.grad.detach().double().square().sum().item()
    norm = math.sqrt(squared)
    if not math.isfinite(norm):
        raise FloatingPointError("nonfinite gradient norm")
    return norm


def fas_step(
    recognizer,
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    source_images,
    target_images,
    target_groups,
    *,
    generator_bn_policy,
    gan_weight=75.0,
    identity_weight=0.002,
    age_weight=10.0,
):
    """One D-then-G update. Frozen recognition weights are NOT fine-tuned here.

    Generator BN can adapt or freeze, explicitly. D remains in train mode
    during the G pass, so spectral-normalization buffers can still change.
    An exception does not undo earlier parameter/buffer updates.
    """
    if generator_bn_policy not in ("adapt", "frozen"):
        raise ValueError("explicit adapt/frozen generator BN policy required")
    for weight in (gan_weight, identity_weight, age_weight):
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError("finite nonnegative loss weights required")
    if not any((gan_weight, identity_weight, age_weight)):
        raise ValueError("at least one generator loss must be enabled")
    for image in (source_images, target_images):
        if (
            image.ndim != 4
            or image.shape[0] < 1
            or image.shape[1] != 3
            or min(image.shape[2:]) < 1
            or not image.is_floating_point()
        ):
            raise ValueError("floating BCHW source/target images required")
        _finite(image, "input image")
    if (
        source_images.shape != target_images.shape
        or source_images.device != target_images.device
        or source_images.dtype != target_images.dtype
    ):
        raise ValueError("aligned source/target image contract required")
    if (
        target_groups.shape != (len(source_images),)
        or target_groups.dtype != torch.int64
        or target_groups.device != source_images.device
        or (target_groups < 0).any()
        or (target_groups >= 7).any()
    ):
        raise ValueError("aligned int64 target groups in0..6 required")
    if not hasattr(recognizer, "age_head"):
        raise ValueError("recognizer age head required")
    _owned(generator, g_optimizer)
    _owned(discriminator, d_optimizer)
    owners = [
        {id(p) for p in (*module.parameters(), *module.buffers())}
        for module in (recognizer, generator, discriminator)
    ]
    if any(owners[i] & owners[j] for i in range(3) for j in range(i)):
        raise ValueError("recognizer/generator/discriminator state must be disjoint")

    discriminator_updated = generator_updated = False
    generator.train()
    if generator_bn_policy == "frozen":
        for module in generator.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    discriminator.train()
    g_optimizer.zero_grad(set_to_none=True)
    d_optimizer.zero_grad(set_to_none=True)
    try:
        with _frozen(recognizer, evaluation=True):
            with torch.no_grad():
                shortcuts = recognizer(source_images, return_shortcuts=True)
            if not isinstance(shortcuts, tuple) or len(shortcuts) != 7:
                raise ValueError("five encoder stages and identity/age maps required")
            for value in shortcuts:
                _finite(value, "source shortcut")
            synthesized = generator(source_images, *shortcuts, condition=target_groups)
            if synthesized.shape != source_images.shape:
                raise ValueError("generated image shape mismatch")
            _finite(synthesized, "generated image")

            real = discriminator(target_images, target_groups)
            fake = discriminator(synthesized.detach(), target_groups)
            _patch_logits(real, len(source_images))
            _patch_logits(fake, len(source_images))
            if real.shape != fake.shape:
                raise ValueError("real/fake patch contract mismatch")
            d_loss = 0.5 * ((real - 1).square().mean() + fake.square().mean())
            d_norm = _backward(d_loss, d_optimizer)
            d_optimizer.step()
            discriminator_updated = True
            _finite_state(discriminator)

            with _frozen(discriminator, evaluation=False):
                _, identity_map, age_map = recognizer(synthesized, return_age=True)
                if identity_map.shape != shortcuts[-2].shape:
                    raise ValueError("synthesized identity-map shape mismatch")
                _finite(identity_map, "synthesized identity map")
                _finite(age_map, "synthesized age map")
                _, group_logits = recognizer.age_head(age_map)
                if group_logits.shape != (len(source_images), 7):
                    raise ValueError("seven-group age logits required")
                _finite(group_logits, "age logits")
                logits = discriminator(synthesized, target_groups)
                _patch_logits(logits, len(source_images))
                gan_loss = 0.5 * (logits - 1).square().mean()
                identity_loss = F.mse_loss(shortcuts[-2], identity_map)
                age_loss = F.cross_entropy(group_logits, target_groups)
                g_loss = (
                    gan_weight * gan_loss + identity_weight * identity_loss + age_weight * age_loss
                )
                g_norm = _backward(g_loss, g_optimizer)
                g_optimizer.step()
                generator_updated = True
                _finite_state(generator)
                _finite_state(discriminator)
    except Exception as error:
        g_optimizer.zero_grad(set_to_none=True)
        d_optimizer.zero_grad(set_to_none=True)
        raise FASStepFailure(
            str(error),
            discriminator_updated=discriminator_updated,
            generator_updated=generator_updated,
        ) from error
    return dict(
        discriminator_loss=d_loss.item(),
        generator_loss=g_loss.item(),
        adversarial_loss=gan_loss.item(),
        identity_loss=identity_loss.item(),
        age_loss=age_loss.item(),
        discriminator_gradient_norm=d_norm,
        generator_gradient_norm=g_norm,
        discriminator_updated=True,
        generator_updated=True,
        recognizer_updated=False,
        generator_bn_policy=generator_bn_policy,
        full_joint_fas=False,
    )
