"""Детекция лиц через InsightFace (RetinaFace из набора buffalo_l).

Модель загружается лениво при первом вызове (скачивается в кэш insightface). На CPU
используется ctx_id=-1. Возвращаются bbox, 5 ключевых точек, det_score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from age_gap.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class DetectedFace:
    bbox: list[float]  # [x1, y1, x2, y2]
    kps: list[list[float]]  # 5 x [x, y]
    det_score: float

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height


class FaceDetector:
    """Обёртка над insightface.app.FaceAnalysis."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: int = 640,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.det_size = det_size
        self.device = device  # None -> авто (settings/доступность)
        self._app: Any | None = None

    def _ensure_app(self) -> Any:
        if self._app is None:
            from insightface.app import FaceAnalysis  # импорт здесь, чтобы не тянуть в тестах

            from age_gap.common.device import onnx_ctx_id, onnx_providers

            providers = onnx_providers(self.device)
            ctx_id = onnx_ctx_id(self.device)
            log.info(
                "Загрузка модели InsightFace '%s' (ctx_id=%d, providers=%s)",
                self.model_name,
                ctx_id,
                providers,
            )
            app = FaceAnalysis(name=self.model_name, providers=providers)
            app.prepare(ctx_id=ctx_id, det_size=(self.det_size, self.det_size))
            self._app = app
        return self._app

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        app = self._ensure_app()
        faces = app.get(image_bgr)
        result: list[DetectedFace] = []
        for f in faces:
            bbox = [float(v) for v in f.bbox.tolist()]
            kps = [[float(x), float(y)] for x, y in np.asarray(f.kps).tolist()]
            result.append(DetectedFace(bbox=bbox, kps=kps, det_score=float(f.det_score)))
        return result
