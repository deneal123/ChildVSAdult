"""Внешний cross-age бенчмарк FG-NET (другой домен, чем VK — для чистой атрибуции).

FG-NET: 1002 фото, 82 субъекта, возраст 0–69, разрывы до ~45 лет. Имена файлов кодируют
субъекта и возраст: ``001A02.JPG`` = субъект 001, возраст 2. Лица детектируются и
выравниваются нашим InsightFace-пайплайном (как и VK-кропы) и кэшируются в .npz.

Пары: позитивы — внутри субъекта (с известным age_gap); негативы — между субъектами
(сбалансировано). Метрики — overall/large-gap ROC-AUC + 10-fold accuracy.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

from age_gap.common.io import data_path, resolve_path
from age_gap.common.logging import get_logger
from age_gap.evaluation.benchmark_external import accuracy_10fold, pair_scores
from age_gap.evaluation.metrics import roc_auc, tar_at_far

log = get_logger(__name__)

_NAME_RE = re.compile(r"(\d+)A(\d+)", re.IGNORECASE)  # 001A02 -> subject 001, age 02


def _ext_dir() -> Path:
    return resolve_path(str(data_path("data_dir", "external")))


def prepare_crops(zip_path: Path | str | None = None, cache: Path | str | None = None) -> Path:
    """Детектировать+выровнять лица FG-NET и закэшировать в .npz. Возвращает путь кэша."""
    zip_path = Path(zip_path) if zip_path else _ext_dir() / "FGNET.zip"
    cache = Path(cache) if cache else _ext_dir() / "fgnet_crops.npz"
    if cache.exists():
        log.info("FG-NET кэш уже есть: %s", cache)
        return cache

    from age_gap.preprocessing.crop_align import align_face
    from age_gap.preprocessing.detect import FaceDetector

    # Детекция на CPU: onnxruntime-gpu (cuDNN12) и torch cu132 (cuDNN13) конфликтуют в одном
    # процессе (WinError 127). facenet-эмбеддинги при этом считаются на GPU.
    detector = FaceDetector(device="cpu")
    crops: list[np.ndarray] = []
    subjects: list[int] = []
    ages: list[int] = []
    skipped = 0

    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        for name in names:
            m = _NAME_RE.search(Path(name).stem)
            if not m:
                continue
            buf = np.frombuffer(z.read(name), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                skipped += 1
                continue
            try:
                faces = detector.detect(img)
            except Exception as exc:  # noqa: BLE001
                log.warning("FG-NET %s: ошибка детекции %s", name, exc)
                skipped += 1
                continue
            if not faces:
                skipped += 1
                continue
            target = max(faces, key=lambda f: f.area)
            crop = align_face(img, target.kps)  # 112x112 BGR
            crops.append(crop)
            subjects.append(int(m.group(1)))
            ages.append(int(m.group(2)))

    arr = np.stack(crops).astype(np.uint8)
    np.savez(
        cache,
        crops=arr,
        subjects=np.asarray(subjects, np.int64),
        ages=np.asarray(ages, np.int64),
    )
    log.info("FG-NET: лиц %d (skipped %d) -> %s", len(crops), skipped, cache)
    return cache


def load_pairs(
    cache: Path | str | None = None,
    neg_per_pos: float = 1.0,
    seed: int = 42,
) -> tuple[list, list, np.ndarray, np.ndarray]:
    """Собрать cross-age пары. Возвращает (images_a, images_b, issame, age_gap[-1 для нег.])."""
    cache = Path(cache) if cache else _ext_dir() / "fgnet_crops.npz"
    data = np.load(cache, allow_pickle=False)
    crops, subjects, ages = data["crops"], data["subjects"], data["ages"]
    rng = np.random.default_rng(seed)

    by_subject: dict[int, list[int]] = {}
    for i, s in enumerate(subjects.tolist()):
        by_subject.setdefault(s, []).append(i)

    pos: list[tuple[int, int, int]] = []  # (i, j, age_gap)
    for idxs in by_subject.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                pos.append((ia, ib, int(abs(int(ages[ia]) - int(ages[ib])))))

    # Негативы: случайные пары разных субъектов, баланс ~ neg_per_pos * |pos|.
    n_neg = int(len(pos) * neg_per_pos)
    neg: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    n = len(subjects)
    while len(neg) < n_neg and attempts < n_neg * 50 + 1000:
        attempts += 1
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if subjects[i] == subjects[j]:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        neg.append((i, j, -1))

    a_idx = [p[0] for p in pos] + [q[0] for q in neg]
    b_idx = [p[1] for p in pos] + [q[1] for q in neg]
    issame = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    gaps = np.asarray([p[2] for p in pos] + [q[2] for q in neg], dtype=np.int64)

    images_a = [crops[i] for i in a_idx]
    images_b = [crops[i] for i in b_idx]
    log.info("FG-NET пар: pos=%d, neg=%d", len(pos), len(neg))
    return images_a, images_b, issame, gaps


def evaluate(
    backbone: torch.nn.Module, device: str, large_gap_threshold: int = 25
) -> dict[str, float]:
    images_a, images_b, issame, gaps = load_pairs()
    scores = pair_scores(backbone, images_a, images_b, device, rgb=False)
    mask = (issame == 0) | ((issame == 1) & (gaps >= large_gap_threshold))
    return {
        "n_pairs": float(len(issame)),
        "accuracy_10fold": accuracy_10fold(scores, issame),
        "roc_auc": roc_auc(scores, issame),
        "large_gap_auc": roc_auc(scores[mask], issame[mask]) if mask.any() else float("nan"),
        "tar@far=0.01": tar_at_far(scores, issame, 0.01),
    }
