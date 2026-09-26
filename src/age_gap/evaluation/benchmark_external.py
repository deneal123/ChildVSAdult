"""Оценка face-бэкбона на ВНЕШНЕМ известном бенчмарке верификации.

Поддержаны:
- LFW: выровненный кеш data/external/lfw_aligned.npz (см. scripts/build_lfw_aligned.py);
  fallback — sklearn.fetch_lfw_pairs, но БЕЗ выравнивания (ArcFace/AdaFace на нём рушатся);
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

from age_gap.common.io import data_path
from age_gap.common.logging import get_logger
from age_gap.evaluation.metrics import eer, roc_auc, tar_at_far
from age_gap.models.facenet import preprocess_bgr, preprocess_rgb

log = get_logger(__name__)


def load_lfw(subset: str = "10_folds") -> tuple[list, list, np.ndarray, bool]:
    """LFW-пары. Возвращает (images_a, images_b, issame, rgb).

    Приоритет — выровненный кеш data/external/lfw_aligned.npz (5-точечный norm_crop 112px, тот же
    пайплайн, что и для наших кропов; строится scripts/build_lfw_aligned.py). Он в BGR.

    Fallback — sklearn ``fetch_lfw_pairs`` (RGB), но это funneled-изображения БЕЗ выравнивания:
    ArcFace/AdaFace на них рушатся (r100: LFW 0.83 при AgeDB-30 0.98, что невозможно, т.к.
    AgeDB-30 сложнее). Пока кеш не построен, метрики LFW сравнивать между бэкбонами нельзя.
    """
    cached = Path(str(data_path("data_dir", "external", "lfw_aligned.npz")))
    if cached.exists():
        z = np.load(cached)
        mr = float(z["miss_rate"]) if "miss_rate" in z else float("nan")
        if mr > 0.05:                      # детектор промахнулся на >5% лиц -> кеш негоден
            raise RuntimeError(
                f"lfw_aligned.npz негоден: детектор промахнулся на {mr:.1%} лиц. "
                "Пересоберите: uv run python scripts/build_lfw_aligned.py")
        if np.isnan(mr):                   # кеш собран до появления поля miss_rate
            log.warning("lfw_aligned.npz без miss_rate (старый формат) — проверьте, что LFW-точность "
                        "сильного бэкбона ~0.99, иначе пересоберите кеш")
        # ВАЖНО: сначала материализуем массив, потом режем. z["a"][i] в цикле распаковывал бы
        # весь сжатый массив на КАЖДОЙ итерации (6000 раз -> OOM).
        za, zb = z["a"], z["b"]
        a = list(za)
        b = list(zb)
        issame = z["issame"].astype(np.int64)
        log.info("LFW(выровненный кеш): пар=%d (pos=%d)", len(a), int(issame.sum()))
        return a, b, issame, False
    log.warning("LFW БЕЗ ВЫРАВНИВАНИЯ (sklearn fallback) — числа занижены и несравнимы между "
                "бэкбонами; постройте scripts/build_lfw_aligned.py")
    from sklearn.datasets import fetch_lfw_pairs

    data = fetch_lfw_pairs(subset=subset, color=True, resize=1.0)
    pairs = data.pairs  # (N, 2, H, W, 3), float
    issame = data.target.astype(np.int64)  # 1 = same
    if pairs.max() <= 1.0 + 1e-6:
        pairs = pairs * 255.0
    pairs = pairs.astype(np.uint8)
    a = [pairs[i, 0] for i in range(pairs.shape[0])]
    b = [pairs[i, 1] for i in range(pairs.shape[0])]
    log.info("LFW(%s, sklearn/невыровненный): пар=%d (pos=%d)", subset, len(a), int(issame.sum()))
    return a, b, issame, True


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


def accuracy_10fold(
    scores: np.ndarray,
    labels: np.ndarray,
    folds: int = 10,
    fold_ids: np.ndarray | None = None,
) -> float:
    """LFW-протокол: порог подбирается на 9 фолдах, точность считается на 10-м; среднее."""
    n = len(scores)
    idx = np.arange(n)
    fold_id = idx % folds if fold_ids is None else np.asarray(fold_ids)
    if len(fold_id) != n:
        raise ValueError("fold_ids must have one value per pair")
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


def _accuracy_fold_values(
    scores: np.ndarray, labels: np.ndarray, fold_ids: np.ndarray
) -> list[float]:
    values: list[float] = []
    thresholds = np.linspace(scores.min(), scores.max(), 400)
    for fold in sorted(np.unique(fold_ids)):
        train = fold_ids != fold
        test = fold_ids == fold
        train_predictions = scores[train, None] >= thresholds[None, :]
        train_accuracy = (train_predictions == (labels[train, None] == 1)).mean(axis=0)
        best_threshold = thresholds[int(np.argmax(train_accuracy))]
        values.append(float(((scores[test] >= best_threshold) == (labels[test] == 1)).mean()))
    return values


def _bootstrap_cacd_intervals(
    scores: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Stratified pair bootstrap plus a fold bootstrap for protocol accuracy."""
    rng = np.random.default_rng(seed)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    auc_values: list[float] = []
    eer_values: list[float] = []
    tar_01_values: list[float] = []
    tar_001_values: list[float] = []
    for _ in range(n_boot):
        pos = positive[rng.integers(0, len(positive), len(positive))]
        neg = negative[rng.integers(0, len(negative), len(negative))]
        sample_scores = np.concatenate([pos, neg])
        sample_labels = np.concatenate(
            [np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)]
        )
        auc_values.append(roc_auc(sample_scores, sample_labels))
        eer_values.append(eer(sample_scores, sample_labels))
        tar_01_values.append(tar_at_far(sample_scores, sample_labels, 0.01))
        tar_001_values.append(tar_at_far(sample_scores, sample_labels, 0.001))

    fold_values = np.asarray(_accuracy_fold_values(scores, labels, fold_ids))
    accuracy_values = [
        float(fold_values[rng.integers(0, len(fold_values), len(fold_values))].mean())
        for _ in range(n_boot)
    ]

    def interval(values: list[float]) -> list[float]:
        return [float(value) for value in np.percentile(values, [2.5, 97.5])]

    return {
        "accuracy_10fold_ci95": interval(accuracy_values),
        "roc_auc_ci95": interval(auc_values),
        "eer_ci95": interval(eer_values),
        "tar@far=0.01_ci95": interval(tar_01_values),
        "tar@far=0.001_ci95": interval(tar_001_values),
    }


def evaluate(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "n_pairs": float(len(labels)),
        "accuracy_10fold": accuracy_10fold(scores, labels),
        "roc_auc": roc_auc(scores, labels),
        "eer": eer(scores, labels),
        "tar@far=0.01": tar_at_far(scores, labels, 0.01),
        "tar@far=0.001": tar_at_far(scores, labels, 0.001),
    }


def load_cacd_vs(path: Path | str) -> tuple[list, list, np.ndarray, np.ndarray]:
    """Load the aligned CACD-VS cache and its canonical identity-disjoint folds."""
    cache = np.load(path)
    miss_rate = float(cache["miss_rate"])
    if miss_rate > 0.05:
        raise RuntimeError(f"CACD-VS cache has excessive detector miss rate: {miss_rate:.1%}")
    return (
        list(cache["a"]),
        list(cache["b"]),
        cache["issame"].astype(np.int64),
        cache["fold_ids"].astype(np.int64),
    )


def evaluate_cacd_vs(
    backbone: torch.nn.Module, device: str, path: Path | str
) -> dict[str, float | list[float]]:
    """Canonical CACD-VS evaluation: supplied 10 folds, AUC, EER, and TAR@FAR."""
    first, second, labels, fold_ids = load_cacd_vs(path)
    scores = pair_scores(backbone, first, second, device, rgb=False)
    metrics = evaluate(scores, labels)
    metrics["accuracy_10fold"] = accuracy_10fold(scores, labels, fold_ids=fold_ids)
    metrics.update(_bootstrap_cacd_intervals(scores, labels, fold_ids))
    return metrics


def evaluate_lfw(
    backbone: torch.nn.Module, device: str, subset: str = "10_folds"
) -> dict[str, float]:
    a, b, issame, rgb = load_lfw(subset)
    scores = pair_scores(backbone, a, b, device, rgb=rgb)
    return evaluate(scores, issame)


def evaluate_bin(backbone: torch.nn.Module, device: str, path: Path | str) -> dict[str, float]:
    a, b, issame = load_bin(path)
    scores = pair_scores(backbone, a, b, device, rgb=False)
    return evaluate(scores, issame)
