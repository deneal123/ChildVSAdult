"""Слабый ОБУЧАЕМЫЙ face-бэкбон для честного эксперимента «помогают ли наши данные».

InceptionResnetV1 (facenet-pytorch), претрейн casia-webface — заметно слабее ArcFace r50,
поэтому есть запас по качеству на cross-age бенчмарке. В отличие от ONNX-моделей insightface,
веса обучаемы в PyTorch: можно заморозить → измерить, затем дообучить на НАШИХ парах.

Препроцессинг: вход сети — RGB 160x160, нормировка (x-127.5)/128. Наши кропы 112x112 (BGR,
ArcFace-выравнивание) ресайзятся в 160; изображения бенчмарка обрабатываются так же.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

INPUT_SIZE = 160


class FaceNetBackbone(nn.Module):
    """Обёртка InceptionResnetV1 с L2-нормированным выходом (косинус = скалярное произведение)."""

    # Единый интерфейс с ArcFaceBackbone: слои для разморозки при дообучении.
    trainable_scopes = {
        "head": ("last_linear", "last_bn"),
        "tail": ("block8", "last_linear", "last_bn"),
    }
    input_size = INPUT_SIZE

    def __init__(self, pretrained: str | None = "casia-webface") -> None:
        super().__init__()
        from facenet_pytorch import InceptionResnetV1

        self.net = InceptionResnetV1(pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1)

    def preprocess(self, img: np.ndarray, bgr: bool = True) -> np.ndarray:
        """uint8 HxWx3 -> CHW float32 160x160 RGB, нормировка facenet (x-127.5)/128."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if bgr else img
        return preprocess_rgb(rgb)


def preprocess_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """RGB-изображение (uint8) -> CHW float32 160x160, нормировка facenet (x-127.5)/128."""
    if img_rgb.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
        img_rgb = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    arr = (img_rgb.astype(np.float32) - 127.5) / 128.0
    return np.transpose(arr, (2, 0, 1))  # CHW


def preprocess_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """BGR-изображение (cv2.imread) -> CHW float32 RGB 160x160."""
    return preprocess_rgb(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def to_batch_bgr(images_bgr: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack([preprocess_bgr(im) for im in images_bgr]))


def to_batch_rgb(images_rgb: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack([preprocess_rgb(im) for im in images_rgb]))
