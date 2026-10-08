"""Explicit stochastic crop/resize component for a future CACon campaign.

CACon §2.2 requires crop/resize alongside color/blur. Numeric crop policy below
is a declared common-protocol choice, not recovered CACon author hyperparameters.
Caller owns a continuing RNG stream; do not reseed per item on every epoch.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def random_resized_crop(image, rng, *, scale=(0.08, 1.0), ratio=(0.75, 4 / 3), attempts=10):
    image = np.asarray(image)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or min(image.shape[:2]) < 1
    ):
        raise ValueError("nonempty three-channel uint8 image required")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("explicit continuing NumPy Generator required")
    if (
        len(scale) != 2
        or len(ratio) != 2
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (*scale, *ratio)
        )
        or not 0 < scale[0] <= scale[1] <= 1
        or not 0 < ratio[0] <= ratio[1]
        or type(attempts) is not int
        or attempts < 1
    ):
        raise ValueError("ordered positive finite crop bounds and attempt budget required")
    height, width = image.shape[:2]
    fallback = True
    for _ in range(attempts):
        area = height * width * rng.uniform(*scale)
        aspect = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        crop_width, crop_height = round(math.sqrt(area * aspect)), round(math.sqrt(area / aspect))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = int(rng.integers(height - crop_height + 1))
            left = int(rng.integers(width - crop_width + 1))
            fallback = False
            break
    if fallback:
        aspect = width / height
        if aspect < ratio[0]:
            crop_width, crop_height = width, min(height, max(1, round(width / ratio[0])))
        elif aspect > ratio[1]:
            crop_height, crop_width = height, min(width, max(1, round(height * ratio[1])))
        else:
            crop_width, crop_height = width, height
        top, left = (height - crop_height) // 2, (width - crop_width) // 2
    cropped = image[top : top + crop_height, left : left + crop_width]
    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    metadata = dict(
        top=top,
        left=left,
        height=crop_height,
        width=crop_width,
        fallback=fallback,
        scale=list(scale),
        ratio=list(ratio),
        attempts=attempts,
        interpolation="INTER_LINEAR",
        output_shape=list(image.shape),
        scope="declared crop component only; color/blur/generator/two-stage training not covered",
    )
    return resized, metadata
