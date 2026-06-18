"""Тесты арх backbone (без сети/весов): сборка + форма эмбеддинга + препроцессинг."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from age_gap.models.adaface_ir import AdaFaceBackbone, Backbone
from age_gap.models.backbones import BACKBONES, make_backbone
from age_gap.models.iresnet import ArcFaceBackbone, IBasicBlock, IResNet


def _img() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (112, 112, 3), dtype=np.uint8)


def test_iresnet_forward_shape():
    net = IResNet(IBasicBlock, [3, 4, 14, 3]).eval()
    x = torch.zeros(1, 3, 112, 112)
    with torch.no_grad():
        assert net(x).shape == (1, 512)


def test_adaface_backbone_forward_shape():
    net = Backbone(num_layers=50, mode="ir").eval()
    x = torch.zeros(1, 3, 112, 112)
    with torch.no_grad():
        assert net(x).shape == (1, 512)


def test_arcface_preprocess_112_rgb():
    bb = ArcFaceBackbone("arcface_r50_casia", pretrained=False)
    chw = bb.preprocess(_img(), bgr=True)
    assert chw.shape == (3, 112, 112) and chw.dtype == np.float32


def test_adaface_preprocess_and_normalized_output():
    bb = AdaFaceBackbone("adaface_ir50", pretrained=False).eval()
    chw = bb.preprocess(_img(), bgr=True)
    assert chw.shape == (3, 112, 112)
    with torch.no_grad():
        z = bb(torch.from_numpy(chw).unsqueeze(0))
    assert z.shape == (1, 512)
    assert abs(float(z.norm()) - 1.0) < 1e-4  # L2-нормирован


def test_make_backbone_unknown_raises():
    with pytest.raises(ValueError, match="неизвестный backbone"):
        make_backbone("nope")


def test_registry_names_dispatch():
    # Все зарегистрированные имена должны диспетчеризоваться без ошибки построения (без весов).
    for name in BACKBONES:
        if name == "facenet":
            continue  # facenet тянет веса из сети — пропускаем в офлайн-тесте
        make_backbone(name, pretrained=False)
