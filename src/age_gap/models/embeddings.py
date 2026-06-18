"""Baseline-эмбеддинги лиц через замороженный ArcFace (insightface w600k_r50).

MVP-2, Stage 1 (TODO §10): без обучения — берём готовый face-энкодер и считаем
эмбеддинги выровненных кропов. Эмбеддинги L2-нормируются, так что косинусная близость
= скалярное произведение. Результат кэшируется в .npz (face_ids + матрица).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from age_gap.common.io import PROJECT_ROOT, data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.common.schemas import FaceCrop

log = get_logger(__name__)

DEFAULT_REC_MODEL = "buffalo_l/w600k_r50.onnx"


def _model_path(model: str) -> str:
    """Абсолютный путь к onnx-модели распознавания в кэше insightface."""
    root = os.path.expanduser("~/.insightface/models")
    return os.path.join(root, model)


class ArcFaceEmbedder:
    """Обёртка над recognition-моделью insightface (get_feat по выровненному кропу)."""

    def __init__(self, model: str = DEFAULT_REC_MODEL, device: str | None = None) -> None:
        self.model_file = _model_path(model)
        self.device = device  # None -> авто
        self._rec: Any = None

    def _ensure(self) -> Any:
        if self._rec is None:
            from insightface import model_zoo

            from age_gap.common.device import onnx_ctx_id, onnx_providers

            providers = onnx_providers(self.device)
            ctx_id = onnx_ctx_id(self.device)
            log.info(
                "Загрузка ArcFace-модели %s (ctx_id=%d, providers=%s)",
                self.model_file,
                ctx_id,
                providers,
            )
            rec = model_zoo.get_model(self.model_file, providers=providers)
            rec.prepare(ctx_id=ctx_id)
            self._rec = rec
        return self._rec

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """512-мерный L2-нормированный эмбеддинг выровненного кропа 112x112 (BGR)."""
        rec = self._ensure()
        feat = np.asarray(rec.get_feat(aligned_bgr)).reshape(-1)
        norm = float(np.linalg.norm(feat))
        return feat / norm if norm > 0 else feat


def compute_embeddings(
    faces_file: str | None = None,
    out_file: Path | None = None,
    model: str = DEFAULT_REC_MODEL,
) -> dict[str, np.ndarray]:
    """Посчитать эмбеддинги для всех usable-лиц и сохранить в .npz."""
    faces_file = faces_file or str(data_path("data_dir", "interim", "faces.jsonl"))
    out_file = out_file or data_path("embeddings_cache_dir", "baseline_arcface.npz")

    embedder = ArcFaceEmbedder(model=model)
    face_ids: list[str] = []
    vectors: list[np.ndarray] = []

    for row in read_jsonl(faces_file):
        fc = FaceCrop.from_dict(row)
        if not fc.is_usable or not fc.face_crop_path:
            continue
        crop_path = resolve_path(fc.face_crop_path)
        img = cv2.imread(str(crop_path))
        if img is None:
            log.warning("Не удалось прочитать кроп %s", crop_path)
            continue
        vectors.append(embedder.embed(img))
        face_ids.append(fc.face_id)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(vectors, dtype=np.float32) if vectors else np.zeros((0, 512), np.float32)
    np.savez(out_file, face_ids=np.asarray(face_ids, dtype=object), embeddings=matrix)
    log.info("Эмбеддинги: %d лиц -> %s", len(face_ids), out_file)
    return dict(zip(face_ids, vectors, strict=True))


def load_embeddings(path: Path | str | None = None) -> dict[str, np.ndarray]:
    path = Path(path) if path else data_path("embeddings_cache_dir", "baseline_arcface.npz")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    data = np.load(path, allow_pickle=True)
    ids = [str(x) for x in data["face_ids"]]
    mat = data["embeddings"]
    return {fid: mat[i] for i, fid in enumerate(ids)}
