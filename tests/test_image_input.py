from __future__ import annotations

import numpy as np
from PIL import Image

from prom_service.artifacts import ModelSpec
from prom_service.image import InferenceEngine


def test_inference_input_honours_artifact_channel_order(tmp_path):
    image = Image.new("RGB", (1, 1), (255, 0, 0))
    common = dict(
        path=tmp_path / "unused.onnx",
        sha256="0" * 64,
        input_name="input",
        output_name="output",
        image_size=32,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    rgb = InferenceEngine._input(image, ModelSpec(**common, color_order="rgb"))
    bgr = InferenceEngine._input(image, ModelSpec(**common, color_order="bgr"))
    assert np.allclose(rgb[0, :, 0, 0], [1.0, 0.0, 0.0])
    assert np.allclose(bgr[0, :, 0, 0], [0.0, 0.0, 1.0])
