"""Метрики верификации (SKILL §15 / TODO §11) на чистом numpy.

Вход: ``scores`` — близость пары (больше = вероятнее один человек), ``labels`` — 1 (позитив)
или 0 (негатив). Реализованы ROC-AUC (через ранги, Mann–Whitney U), EER и TAR@FAR.
"""

from __future__ import annotations

import numpy as np


def _rankdata_avg(a: np.ndarray) -> np.ndarray:
    """Средние ранги (как scipy.stats.rankdata, method='average') — для AUC с ничьими."""
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # ранги с 1
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC = P(score(pos) > score(neg)) с учётом ничьих."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_avg(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Equal Error Rate: точка, где FAR ≈ FRR."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    thresholds = np.unique(scores)
    best = 1.0
    for t in thresholds:
        far = float((neg >= t).mean())  # негативы, ошибочно принятые
        frr = float((pos < t).mean())  # позитивы, ошибочно отклонённые
        best = min(best, max(far, frr))
    return best


def tar_at_far(scores: np.ndarray, labels: np.ndarray, far_target: float) -> float:
    """TAR при заданном FAR: доля принятых позитивов при пороге, дающем FAR ≤ far_target."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    neg_sorted = np.sort(neg)[::-1]  # по убыванию
    allowed = int(np.floor(far_target * len(neg)))  # сколько негативов можно пропустить
    if allowed <= 0:
        # Порог строго выше максимального негатива.
        threshold = float(neg_sorted[0])
        return float((pos > threshold).mean())
    threshold = float(neg_sorted[allowed - 1])
    return float((pos >= threshold).mean())


def verification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    far_targets: tuple[float, ...] = (0.01, 0.001),
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    out: dict[str, float] = {
        "n_pairs": float(len(labels)),
        "n_pos": float((labels == 1).sum()),
        "n_neg": float((labels == 0).sum()),
        "roc_auc": roc_auc(scores, labels),
        "eer": eer(scores, labels),
    }
    for far in far_targets:
        out[f"tar@far={far:g}"] = tar_at_far(scores, labels, far)
    return out


def bootstrap_auc_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Перцентильный bootstrap-CI для ROC-AUC: ресэмплинг пар с возвратом.

    Возвращает (lo, hi) для уровня (1−alpha). Учитывает конечность тестовой выборки — ширина
    растёт на малых стратах (мало позитивов/негативов).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n = len(labels)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = roc_auc(scores[idx], labels[idx])
        if not np.isnan(a):
            aucs.append(a)
    if not aucs:
        return float("nan"), float("nan")
    lo = float(np.percentile(aucs, 100 * alpha / 2))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, hi
