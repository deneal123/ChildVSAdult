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


def margin_crop(image_bgr: np.ndarray, bbox: list[float], margin: float = 0.4, size: int = 256) -> np.ndarray:
    """Свободный квадратный кроп лица с полями (для эстетики/красоты, НЕ identity-выравнивание).

    Центрируется на bbox, сторона = max(w,h)*(1+2*margin) — захватывает волосы/подбородок/контекст.
    Выход за кадр добивается репликацией края. Возвращает BGR size x size.
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin) / 2.0
    xa, ya, xb, yb = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    pad_l, pad_t, pad_r, pad_b = max(0, -xa), max(0, -ya), max(0, xb - w), max(0, yb - h)
    crop = image_bgr[max(0, ya):min(h, yb), max(0, xa):min(w, xb)]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=image_bgr.dtype)
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def save_crop(crop_bgr: np.ndarray, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), crop_bgr)
    return dest
