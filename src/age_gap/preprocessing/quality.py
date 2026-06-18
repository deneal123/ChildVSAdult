"""Скоринг качества лица и решение об usable/reject.

Чистые функции без зависимости от моделей — удобно тестировать. Пороги вынесены в
константы; решения и причины реджекта логируются на уровне pipeline (SKILL §11).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Пороги (можно вынести в settings при необходимости).
MIN_FACE_SIZE = 48.0  # минимальная сторона bbox, px
MIN_DET_SCORE = 0.5  # уверенность детектора
MIN_BLUR_VAR = 30.0  # дисперсия лапласиана: ниже — слишком размыто


@dataclass
class QualityResult:
    quality_score: float
    blur_var: float
    is_usable: bool
    reject_reason: str | None


def blur_variance(face_bgr: np.ndarray) -> float:
    """Дисперсия лапласиана как мера резкости (меньше = размытее)."""
    if face_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess(
    face_bgr: np.ndarray,
    face_size: float,
    det_score: float,
    num_faces_in_image: int,
) -> QualityResult:
    """Оценить качество и принять решение usable/reject.

    Порядок проверок задаёт приоритет причины реджекта.
    """
    blur = blur_variance(face_bgr)

    reject: str | None = None
    if num_faces_in_image == 0:
        reject = "no_face_detected"
    elif num_faces_in_image > 1:
        # Несколько лиц без сопоставления цели — на ручную проверку (SKILL §11).
        reject = "multiple_faces_no_target_mapping"
    elif det_score < MIN_DET_SCORE:
        reject = "low_detector_confidence"
    elif face_size < MIN_FACE_SIZE:
        reject = "face_too_small"
    elif blur < MIN_BLUR_VAR:
        reject = "too_blurry"

    # Композитный скор в [0, 1]: уверенность детектора, нормированный размер и резкость.
    size_term = min(1.0, face_size / 160.0)
    blur_term = min(1.0, blur / 200.0)
    quality_score = round(float(0.5 * det_score + 0.3 * size_term + 0.2 * blur_term), 4)

    return QualityResult(
        quality_score=quality_score,
        blur_var=round(blur, 2),
        is_usable=reject is None,
        reject_reason=reject,
    )
