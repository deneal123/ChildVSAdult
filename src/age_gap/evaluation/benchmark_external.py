"""Оценка face-бэкбона на ВНЕШНЕМ известном бенчмарке верификации.

Поддержаны:
- LFW через sklearn.fetch_lfw_pairs (канонический, автозагрузка) — протокол 10-fold accuracy;
- insightface .bin (agedb_30.bin / calfw.bin / lfw.bin) — если файл положен в data/external/
  (cross-age бенчмарки). Формат: pickle (список jpeg-байтов + список issame).

Метрики: 10-fold accuracy (LFW-протокол) + ROC-AUC + EER + TAR@FAR — всё на numpy.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

from age_gap.common.logging import get_logger
from age_gap.evaluation.metrics import eer, roc_auc, tar_at_far
from age_gap.models.facenet import preprocess_bgr, preprocess_rgb

log = get_logger(__name__)


def load_lfw(subset: str = "10_folds") -> tuple[list, list, np.ndarray]:
    """LFW-пары через sklearn. Возвращает (images_a_rgb, images_b_rgb, issame)."""
    from sklearn.datasets import fetch_lfw_pairs

    data = fetch_lfw_pairs(subset=subset, color=True, resize=1.0)
    pairs = data.pairs  # (N, 2, H, W, 3), float
    issame = data.target.astype(np.int64)  # 1 = same
    if pairs.max() <= 1.0 + 1e-6:
        pairs = pairs * 255.0
    pairs = pairs.astype(np.uint8)
    a = [pairs[i, 0] for i in range(pairs.shape[0])]
    b = [pairs[i, 1] for i in range(pairs.shape[0])]
    log.info("LFW(%s): пар=%d (pos=%d)", subset, len(a), int(issame.sum()))
    return a, b, issame


def load_bin(path: Path | str) -> tuple[list, list, np.ndarray]:
    """insightface .bin (agedb_30/calfw/lfw). Возвращает (images_a_bgr, images_b_bgr, issame)."""
    with open(path, "rb") as f:
        bins, issame_list = pickle.load(f, encoding="bytes")
    imgs = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in bins]
    a = imgs[0::2]
    b = imgs[1::2]
    issame = np.asarray([int(bool(x)) for x in issame_list], dtype=np.int64)
    log.info("BIN %s: пар=%d (pos=%d)", path, len(a), int(issame.sum()))
    return a, b, issame


def _embed(backbone: torch.nn.Module, batch: torch.Tensor, device: str) -> np.ndarray:
    backbone.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(batch), 128):
            chunk = batch[i : i + 128].to(device)
            out.append(backbone(chunk).cpu().numpy())
    return np.concatenate(out, axis=0)


def _preprocess_fn(backbone: torch.nn.Module):
    """Препроцессинг конкретного backbone (его размер/нормировка); fallback — facenet."""
    prep = getattr(backbone, "preprocess", None)
    if prep is not None:
        return prep

    def _fallback(img: np.ndarray, bgr: bool = True) -> np.ndarray:
        return preprocess_bgr(img) if bgr else preprocess_rgb(img)

    return _fallback


def pair_scores(
    backbone: torch.nn.Module,
    images_a: list[np.ndarray],
    images_b: list[np.ndarray],
    device: str,
    rgb: bool,
) -> np.ndarray:
    """Косинусная близость по парам (эмбеддинги L2-нормированы в бэкбоне).

    ``rgb`` — в каком порядке каналов поданы изображения (LFW = RGB, .bin/наши кропы = BGR);
    каждый backbone приводит их к своему входу через ``preprocess``.
    """
    prep = _preprocess_fn(backbone)
    ba = torch.from_numpy(np.stack([prep(im, bgr=not rgb) for im in images_a]))
    bb = torch.from_numpy(np.stack([prep(im, bgr=not rgb) for im in images_b]))
    ea, eb = _embed(backbone, ba, device), _embed(backbone, bb, device)
    return (ea * eb).sum(axis=1)


def accuracy_10fold(scores: np.ndarray, labels: np.ndarray, folds: int = 10) -> float:
    """LFW-протокол: порог подбирается на 9 фолдах, точность считается на 10-м; среднее."""
    n = len(scores)
    idx = np.arange(n)
    fold_id = idx % folds
    thresholds = np.linspace(scores.min(), scores.max(), 400)
    accs: list[float] = []
    for f in range(folds):
        train_m = fold_id != f
        test_m = fold_id == f
        best_t = thresholds[0]
        best_acc = 0.0
        for t in thresholds:
            pred = scores[train_m] >= t
            acc = float((pred == (labels[train_m] == 1)).mean())
            if acc > best_acc:
                best_acc, best_t = acc, t
        pred = scores[test_m] >= best_t
        accs.append(float((pred == (labels[test_m] == 1)).mean()))
    return float(np.mean(accs))


def evaluate(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "n_pairs": float(len(labels)),
        "accuracy_10fold": accuracy_10fold(scores, labels),
        "roc_auc": roc_auc(scores, labels),
        "eer": eer(scores, labels),
        "tar@far=0.01": tar_at_far(scores, labels, 0.01),
        "tar@far=0.001": tar_at_far(scores, labels, 0.001),
    }


def evaluate_lfw(
    backbone: torch.nn.Module, device: str, subset: str = "10_folds"
) -> dict[str, float]:
    a, b, issame = load_lfw(subset)
    scores = pair_scores(backbone, a, b, device, rgb=True)
    return evaluate(scores, issame)


def evaluate_bin(backbone: torch.nn.Module, device: str, path: Path | str) -> dict[str, float]:
    a, b, issame = load_bin(path)
    scores = pair_scores(backbone, a, b, device, rgb=False)
    return evaluate(scores, issame)
