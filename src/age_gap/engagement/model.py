"""Модели, кросс-валидация и метрики для «симпатии сверх охвата».

Ключевые защиты от самообмана:
  * ``GroupKFold(person_id)`` — один человек НИКОГДА не попадает и в train, и в test.
    Без этого ArcFace-эмбеддинг (почти константа для человека) даёт фиктивно высокий R²:
    модель просто узнаёт человека, а не предсказывает реакцию аудитории.
  * PCA эмбеддингов обучается ТОЛЬКО на train-фолде.
  * Негативный контроль: перемешать таргет внутри (сообщество × год-месяц) →
    качество обязано схлопнуться к нулю.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from age_gap.common.logging import get_logger
from age_gap.engagement.features import feature_columns

log = get_logger(__name__)

N_SPLITS = 5
N_PCA = 32


# ------------------------------------------------------------------ metrics


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 50) -> float:
    """NDCG@k с релевантностью = min-max нормировка истинного таргета в [0,1].

    Таргет-остаток может быть отрицательным, поэтому линейно масштабируем — это
    сохраняет порядок и делает gain неотрицательным.
    """
    if len(y_true) < 2:
        return float("nan")
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    rel = (y_true - lo) / (hi - lo) if hi > lo else np.zeros_like(y_true)
    k = min(k, len(y_true))
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    top = np.argsort(-y_score)[:k]
    dcg = float(np.sum(rel[top] * disc))
    idcg = float(np.sum(np.sort(rel)[::-1][:k] * disc))
    return dcg / idcg if idcg > 0 else float("nan")


def topk_lift(y_true: np.ndarray, y_score: np.ndarray, k: int = 50) -> float:
    """На сколько стандартных отклонений средний таргет в предсказанном топ-k выше общего."""
    if len(y_true) < k or np.std(y_true) == 0:
        return float("nan")
    top = np.argsort(-y_score)[:k]
    return float((np.mean(y_true[top]) - np.mean(y_true)) / np.std(y_true))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = 50) -> dict[str, float]:
    rho = spearmanr(y_true, y_pred).statistic
    tau = kendalltau(y_true, y_pred).statistic
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(rho) if rho == rho else float("nan"),
        "kendall": float(tau) if tau == tau else float("nan"),
        f"ndcg@{k}": ndcg_at_k(y_true, y_pred, k),
        f"lift@{k}": topk_lift(y_true, y_pred, k),
    }


# ------------------------------------------------------------------ splits / models


def make_splits(df: pd.DataFrame, n_splits: int = N_SPLITS) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = df["person_id"].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(df, groups=groups))


def assert_no_group_leakage(df: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]) -> None:
    persons = df["person_id"].astype(str).to_numpy()
    for i, (tr, te) in enumerate(splits):
        overlap = set(persons[tr]) & set(persons[te])
        if overlap:
            raise AssertionError(f"Утечка person_id в фолде {i}: {len(overlap)} пересечений")


def _build_model(name: str) -> Any:
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, min_samples_leaf=30,
            l2_regularization=1.0, random_state=0,
        )
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    if name == "dummy":
        return DummyRegressor(strategy="median")
    raise ValueError(name)


# ------------------------------------------------------------------ evaluation


@dataclass
class EvalResult:
    model: str
    target: str
    blocks: list[str]
    n_features: int
    folds: list[dict[str, float]]
    oof_pred: np.ndarray
    oof_true: np.ndarray

    def aggregate(self) -> dict[str, Any]:
        keys = self.folds[0].keys()
        agg = {k: (float(np.mean([f[k] for f in self.folds])),
                   float(np.std([f[k] for f in self.folds]))) for k in keys}
        overall = regression_metrics(self.oof_true, self.oof_pred)
        return {
            "model": self.model,
            "target": self.target,
            "blocks": self.blocks,
            "n_features": self.n_features,
            "cv_mean_std": {k: {"mean": m, "std": s} for k, (m, s) in agg.items()},
            "oof_overall": overall,
        }


def _design(
    df: pd.DataFrame, blocks: list[str], cols: dict[str, list[str]],
    tr: np.ndarray, te: np.ndarray, n_pca: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Матрицы train/test. PCA эмбеддингов фитится ТОЛЬКО на train."""
    plain = [c for b in blocks if b != "emb" for c in cols[b]]
    plain = list(dict.fromkeys(plain))
    Xtr = df.iloc[tr][plain].to_numpy(dtype=float)
    Xte = df.iloc[te][plain].to_numpy(dtype=float)
    names = list(plain)

    if "emb" in blocks and cols["emb"]:
        E = df[cols["emb"]].to_numpy(dtype=np.float32)
        Etr, Ete = E[tr], E[te]
        med = np.nanmedian(Etr, axis=0)
        Etr = np.where(np.isnan(Etr), med, Etr)
        Ete = np.where(np.isnan(Ete), med, Ete)
        k = min(n_pca, Etr.shape[1], max(2, Etr.shape[0] - 1))
        pca = PCA(n_components=k, random_state=0).fit(Etr)  # <-- только train
        Xtr = np.hstack([Xtr, pca.transform(Etr)])
        Xte = np.hstack([Xte, pca.transform(Ete)])
        names += [f"pca{i:02d}" for i in range(k)]

    return Xtr, Xte, names


def evaluate(
    df: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    blocks: list[str],
    target_name: str,
    model_name: str = "hgb",
    n_pca: int = N_PCA,
) -> EvalResult:
    cols = feature_columns(df)
    oof = np.full(len(df), np.nan)
    folds: list[dict[str, float]] = []
    n_feat = 0

    for tr, te in splits:
        Xtr, Xte, names = _design(df, blocks, cols, tr, te, n_pca)
        n_feat = len(names)
        m = _build_model(model_name).fit(Xtr, y[tr])
        pred = m.predict(Xte)
        oof[te] = pred
        folds.append(regression_metrics(y[te], pred))

    return EvalResult(model_name, target_name, blocks, n_feat, folds, oof, y)


def permutation_importance_oof(
    df: pd.DataFrame, y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]],
    blocks: list[str], n_pca: int = N_PCA, n_repeats: int = 5, seed: int = 0, top: int = 25,
) -> list[dict[str, Any]]:
    """Permutation importance по Spearman на held-out фолде (первый фолд)."""
    cols = feature_columns(df)
    tr, te = splits[0]
    Xtr, Xte, names = _design(df, blocks, cols, tr, te, n_pca)
    m = _build_model("hgb").fit(Xtr, y[tr])
    base = spearmanr(y[te], m.predict(Xte)).statistic
    rng = np.random.default_rng(seed)

    drops: list[dict[str, Any]] = []
    for j, name in enumerate(names):
        losses = []
        for _ in range(n_repeats):
            Xp = Xte.copy()
            rng.shuffle(Xp[:, j])
            losses.append(base - spearmanr(y[te], m.predict(Xp)).statistic)
        drops.append({"feature": name, "spearman_drop": float(np.mean(losses)),
                      "std": float(np.std(losses))})
    drops.sort(key=lambda d: -d["spearman_drop"])
    return drops[:top]
