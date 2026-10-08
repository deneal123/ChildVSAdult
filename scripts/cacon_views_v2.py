"""Declared CACon crop/color/blur composition; generator is a separate input.

CACon section2.2 specifies these augmentation families, not the numeric policy
below. Values/order are common-protocol choices. Images are uint8 BGR; callers
own a continuing NumPy RNG and must not reseed per item/epoch.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from scripts.cacon_crop_v2 import random_resized_crop


@dataclass(frozen=True)
class ViewPolicy:
    color_probability: float = 0.8
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    blur_probability: float = 0.5
    sigma_min: float = 0.1
    sigma_max: float = 2.0
    blur_kernel: int = 3

    def __post_init__(self):
        for name in (
            "color_probability",
            "brightness",
            "contrast",
            "saturation",
            "blur_probability",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("finite color/blur bounds between zero and one required")
        for name in ("hue", "sigma_min", "sigma_max"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("finite numeric hue/sigma policy required")
        if not 0 <= self.hue <= 0.5:
            raise ValueError("finite hue amplitude between zero and 0.5 required")
        if not 0 < self.sigma_min <= self.sigma_max or not math.isfinite(self.sigma_max):
            raise ValueError("ordered finite positive sigma bounds required")
        if type(self.blur_kernel) is not int or self.blur_kernel < 1 or self.blur_kernel % 2 != 1:
            raise ValueError("positive odd blur kernel required")


DEFAULT_VIEW_POLICY = ViewPolicy()


def _image(image):
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 1
    ):
        raise ValueError("nonempty three-channel uint8 BGR image required")


def augment_view(image, rng, *, policy=DEFAULT_VIEW_POLICY):
    _image(image)
    if not isinstance(policy, ViewPolicy):
        raise ValueError("explicit ViewPolicy required")
    cropped, crop = random_resized_crop(image, rng)
    out = cropped.copy()
    color = None
    if rng.random() < policy.color_probability:
        brightness = float(rng.uniform(1 - policy.brightness, 1 + policy.brightness))
        contrast = float(rng.uniform(1 - policy.contrast, 1 + policy.contrast))
        saturation = float(rng.uniform(1 - policy.saturation, 1 + policy.saturation))
        hue_degrees = float(rng.uniform(-policy.hue, policy.hue) * 360)
        pixels = out.astype(np.float32) / 255
        pixels = np.clip((pixels - pixels.mean()) * contrast + pixels.mean(), 0, 1)
        pixels = np.clip(pixels * brightness, 0, 1)
        hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_degrees) % 360
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 1)
        pixels = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # Clipping rather than convertScaleAbs: negative values must not reflect.
        out = np.clip(np.rint(pixels * 255), 0, 255).astype(np.uint8)
        color = dict(
            brightness=brightness, contrast=contrast, saturation=saturation, hue_degrees=hue_degrees
        )
    sigma = None
    if rng.random() < policy.blur_probability:
        sigma = float(rng.uniform(policy.sigma_min, policy.sigma_max))
        out = cv2.GaussianBlur(out, (policy.blur_kernel, policy.blur_kernel), sigma)
    return out, dict(
        crop=crop,
        color=color,
        blur_sigma=sigma,
        policy=asdict(policy),
        order="crop/resize, contrast, brightness, HSV saturation/hue, blur",
        scope="declared common-protocol augmentation; not author hyperparameters",
    )


def three_views(source, synthesized, rng, *, policy=DEFAULT_VIEW_POLICY):
    _image(source)
    _image(synthesized)
    if source.shape != synthesized.shape:
        raise ValueError("source and synthesized view dimensions must match")
    first, first_metadata = augment_view(source, rng, policy=policy)
    second, second_metadata = augment_view(source, rng, policy=policy)
    return (first, second, synthesized.copy()), dict(
        first=first_metadata,
        second=second_metadata,
        third_policy="provided generated image copied unchanged; declared choice",
        generator_implemented=False,
        generator_provenance_verified=False,
    )
