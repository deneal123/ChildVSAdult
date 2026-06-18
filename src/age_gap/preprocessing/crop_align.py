"""Выравнивание и сохранение кропа лица.

Используется insightface.utils.face_align.norm_crop — выравнивание по 5 ключевым точкам
к каноническому шаблону ArcFace (по умолчанию 112x112), что нужно для качественных
эмбеддингов на этапе baseline (MVP-2).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def align_face(image_bgr: np.ndarray, kps: list[list[float]], image_size: int = 112) -> np.ndarray:
    """Выровнять лицо по ключевым точкам. Возвращает BGR-кроп image_size x image_size."""
    from insightface.utils import face_align

    landmark = np.asarray(kps, dtype=np.float32)
    return face_align.norm_crop(image_bgr, landmark=landmark, image_size=image_size)


def save_crop(crop_bgr: np.ndarray, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), crop_bgr)
    return dest
