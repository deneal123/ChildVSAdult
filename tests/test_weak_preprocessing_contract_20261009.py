"""Synthetic preprocessing/source contracts only; no model construction or forward."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from age_gap.models.facenet import FaceNetBackbone, preprocess_bgr, preprocess_rgb
from age_gap.training.finetune import _bb_prep

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("side", [112, 160, 217])
def test_bgr_resize_normalization_contract_without_model(side):
    image = np.full((side, side, 3), [13, 101, 229], dtype=np.uint8)
    actual = FaceNetBackbone.preprocess(None, image, True)
    expected = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (160, 160),
                          interpolation=cv2.INTER_AREA).astype(np.float32)
    expected = ((expected - 127.5) / 128.0).transpose(2, 0, 1)
    assert actual.shape == (3, 160, 160)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, preprocess_bgr(image))
    np.testing.assert_array_equal(actual, preprocess_rgb(image[:, :, ::-1]))


def test_training_wrapper_uses_same_bgr_preprocessing_as_external_call():
    image = np.arange(112 * 112 * 3, dtype=np.uint8).reshape(112, 112, 3)
    calls = []

    def preprocess(value, bgr):
        calls.append(bgr)
        return FaceNetBackbone.preprocess(None, value, bgr)

    wrapper = SimpleNamespace(preprocess=preprocess)
    np.testing.assert_array_equal(_bb_prep(wrapper)(image), wrapper.preprocess(image, True))
    assert calls == [True, True]


@pytest.mark.parametrize("version", [3, 4])
def test_external_and_retrospective_paths_delegate_to_backbone_bgr_contract(version):
    tree = ast.parse((ROOT / f"scripts/evaluate_facenet_fixed8_cuda_v{version}.py").read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("embed_pool", "measure_validation_loss"):
        calls = [node for node in ast.walk(functions[name]) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "preprocess"]
        assert len(calls) == 1
        assert ast.unparse(calls[0].func) == "model.preprocess"
        assert len(calls[0].args) == 2 and isinstance(calls[0].args[1], ast.Constant)
        assert calls[0].args[1].value is True
    assert "return lambda im: prep(im, True)" in inspect.getsource(_bb_prep)
