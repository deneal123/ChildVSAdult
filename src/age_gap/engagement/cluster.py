"""Без обучения: кластеризация постов на «архетипы» и средняя симпатия по кластеру.

KMeans по контентным признакам (атрибуты лица + PCA эмбеддингов + мета).
k подбирается по silhouette. Для каждого кластера — размер и средняя симпатия
с bootstrap-CI, чтобы не выдавать шум за различия.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from age_gap.common.logging import get_logger
from age_gap.engagement.features import feature_columns

log = get_logger(__name__)


def _content_matrix(df: pd.DataFrame, n_pca: int = 16) -> tuple[np.ndarray, list[str]]:
    cols = feature_columns(df)
    plain = list(dict.fromkeys(cols["meta"] + cols["face"] + cols["domain"]))
    X = SimpleImputer(strategy="median").fit_transform(df[plain].to_numpy(dtype=float))
    X = StandardScaler().fit_transform(X)
    names = list(plain)
    if cols["emb"]:
        E = df[cols["emb"]].to_numpy(dtype=np.float32)
        E = np.where(np.isnan(E), np.nanmedian(E, axis=0), E)
        k = min(n_pca, E.shape[1])
        P = PCA(n_components=k, random_state=0).fit_transform(E)
        X = np.hstack([X, StandardScaler().fit_transform(P)])
        names += [f"pca{i:02d}" for i in range(k)]
    return X, names


def _boot_ci(v: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(v, size=len(v), replace=True))) for _ in range(n)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def cluster_archetypes(
    df: pd.DataFrame, y: np.ndarray, k_range: tuple[int, ...] = (3, 4, 5, 6, 7, 8),
    sample: int = 5000, seed: int = 0,
) -> dict[str, Any]:
    X, names = _content_matrix(df)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)

    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(X[idx])
        scores[k] = float(silhouette_score(X[idx], km.labels_))
    best_k = max(scores, key=scores.get)
    log.info("KMeans silhouette: %s -> k=%d", {k: round(v, 4) for k, v in scores.items()}, best_k)

    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(X)
    labels = km.labels_

    clusters = []
    for c in range(best_k):
        m = labels == c
        lo, hi = _boot_ci(y[m])
        prof = {
            col: float(np.nanmean(df.loc[m, col].to_numpy(dtype=float)))
            for col in ["share_female_adult", "age_est_median", "n_photos",
                        "has_child_photo", "age_gap_label", "face_quality_mean"]
            if col in df.columns
        }
        clusters.append({
            "cluster": int(c),
            "size": int(m.sum()),
            "mean_sympathy": float(np.mean(y[m])),
            "ci95": [lo, hi],
            "profile": prof,
        })
    clusters.sort(key=lambda d: -d["mean_sympathy"])

    return {
        "silhouette_by_k": scores,
        "best_k": int(best_k),
        "n_features": len(names),
        "clusters": clusters,
        "labels": labels.tolist(),
    }
