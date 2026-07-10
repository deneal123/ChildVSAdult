"""Построение таргетов: сырая вовлечённость и «симпатия сверх охвата».

Сырые лайки ≈ ОХВАТ (размер паблика, время, алгоритм ленты, возраст поста), а не
отношение аудитории к человеку. Поэтому:

1. ``E_raw`` — композит вовлечённости: первая главная компонента PCA от
   z(log1p likes), z(log1p comments), z(log1p reposts). Веса берутся из данных,
   а не назначаются руками. Знак фиксируем так, чтобы нагрузка лайков была > 0.

2. ``E_w`` — интерпретируемый sensitivity-вариант: log1p(L + 3C + 5R)
   (комментарий/репост «дороже» лайка).

3. ``y_symp`` — ОСТАТОК после модели охвата (partialling-out / FWL):
       y_symp = E_raw − ĝ(X_reach),  ĝ — строго out-of-fold (кросс-фиттинг).
   Два варианта контролей:
     * A (exogenous-only): сообщество, время, возраст поста, формат. БЕЗ views.
     * B (консервативный): дополнительно log1p(views).
   Views частично СЛЕДСТВИЕ вовлечённости (виральный пост показывают чаще), поэтому
   вариант B может «вычесть» часть самой симпатии. Истина между A и B — печатаем оба.

4. ``y_pct`` — модель-свободный таргет: перцентиль E_raw внутри бакета
   (сообщество × год-месяц). Проверка устойчивости выводов без всякой модели.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from age_gap.common.logging import get_logger

log = get_logger(__name__)

COUNT_COLS = ["likes", "comments", "reposts"]

REACH_EXOGENOUS = [
    "owner_code", "hour", "weekday", "month", "year", "post_age_days",
    "n_photos", "is_pinned", "marked_as_ads", "is_repost", "caption_len",
]


def _pca1(matrix: np.ndarray) -> tuple[np.ndarray, PCA, float]:
    """PCA1 со знаком, зафиксированным по первой колонке (лайки/лайки-на-показ > 0)."""
    z = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=1, random_state=0).fit(z)
    sign = 1.0 if pca.components_[0][0] >= 0 else -1.0
    return sign * pca.transform(z)[:, 0], pca, sign


def add_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Добавить таргеты. ГЛАВНЫЙ — e_rate (ставка на просмотр). Возвращает (df, info).

    Строки без просмотров отбрасываются: без знаменателя нельзя посчитать ставку, а
    покрытие views ~99–100%, так что потеря пренебрежимо мала и делает все таргеты
    сопоставимыми на одном наборе строк.
    """
    df = df.copy()
    v = pd.to_numeric(df["views"], errors="coerce")
    n0 = len(df)
    df = df[v.notna() & (v > 0)].reset_index(drop=True)
    v = pd.to_numeric(df["views"], errors="coerce")
    dropped = n0 - len(df)
    if dropped:
        log.info("Отброшено постов без просмотров (нужны для per-view ставки): %d", dropped)

    df["owner_code"] = pd.Categorical(df["owner_id"]).codes
    for c in COUNT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df[f"log_{c}"] = np.log1p(df[c])
    df["log_views"] = np.log1p(v)

    # E_raw — композит суммарной вовлечённости (БЕЗ просмотров), для сравнения.
    e_raw, pca_raw, sign_raw = _pca1(df[[f"log_{c}" for c in COUNT_COLS]].to_numpy())
    df["e_raw"] = e_raw

    # E_rate (ГЛАВНЫЙ) — композит сглаженных ЛОГ-СТАВОК на просмотр: log((count+1)/(views+1)).
    rate_mat = np.column_stack([np.log((df[c] + 1.0) / (v + 1.0)) for c in COUNT_COLS])
    e_rate, pca_rate, sign_rate = _pca1(rate_mat)
    df["e_rate"] = e_rate

    df["e_w"] = np.log1p(df["likes"] + 3.0 * df["comments"] + 5.0 * df["reposts"])

    bucket = [df["owner_id"].astype(str), df["year"].astype("Int64"), df["month"].astype("Int64")]
    df["y_pct"] = df.groupby(bucket, dropna=False)["e_rate"].rank(pct=True)

    logv = df["log_views"]
    sp = lambda a, b: float(pd.Series(a).corr(pd.Series(b), method="spearman"))  # noqa: E731
    info = {
        "n_dropped_no_views": int(dropped),
        "pca1_loadings": {c: float(sign_raw * w) for c, w in zip(COUNT_COLS, pca_raw.components_[0], strict=True)},
        "pca1_explained_variance_ratio": float(pca_raw.explained_variance_ratio_[0]),
        "rate_loadings": {c: float(sign_rate * w) for c, w in zip(COUNT_COLS, pca_rate.components_[0], strict=True)},
        "rate_explained_variance_ratio": float(pca_rate.explained_variance_ratio_[0]),
        # Диагностика: сырой композит тянется за просмотрами, ставка — нет.
        "spearman_e_raw_vs_log_views": sp(df["e_raw"], logv),
        "spearman_e_rate_vs_log_views": sp(df["e_rate"], logv),
        "spearman_e_raw_vs_e_rate": sp(df["e_raw"], df["e_rate"]),
        "n_buckets": int(df.groupby(bucket, dropna=False).ngroups),
    }
    log.info("Таргеты: rate_loadings=%s | E_raw~views=%.3f E_rate~views=%.3f",
             info["rate_loadings"], info["spearman_e_raw_vs_log_views"],
             info["spearman_e_rate_vs_log_views"])
    return df, info


def reach_columns(variant: str) -> list[str]:
    cols = list(REACH_EXOGENOUS)
    if variant == "B":
        cols.append("log_views")
    elif variant != "A":
        raise ValueError(f"variant must be 'A' or 'B', got {variant!r}")
    return cols


def _reach_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.08, max_depth=None, min_samples_leaf=40,
        l2_regularization=1.0, random_state=0,
    )


def oof_reach_residual(
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    variant: str,
    target: str = "e_raw",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Кросс-фиттинг: ŷ_reach предсказывается моделью, НЕ видевшей этот фолд.

    Возвращает (residual, stats). residual = y − ŷ_reach.
    """
    cols = reach_columns(variant)
    X = df[cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    oof = np.full(len(df), np.nan)

    for tr, te in splits:
        m = _reach_model().fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])

    if np.isnan(oof).any():
        raise RuntimeError("OOF-предсказания охвата неполны — проверьте разбиение")

    resid = y - oof
    stats = {
        "variant": variant,
        "reach_features": cols,
        "reach_r2_oof": float(r2_score(y, oof)),
        "var_total": float(np.var(y)),
        "var_residual": float(np.var(resid)),
        "share_variance_explained_by_reach": float(1.0 - np.var(resid) / np.var(y)),
    }
    log.info("Reach-модель (%s): OOF R²=%.4f", variant, stats["reach_r2_oof"])
    return resid, stats


def shuffle_within_bucket(df: pd.DataFrame, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Негативный контроль: перемешать таргет внутри (сообщество × год-месяц)."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    keys = list(zip(df["owner_id"].astype(str), df["year"], df["month"], strict=True))
    idx: dict[Any, list[int]] = {}
    for i, k in enumerate(keys):
        idx.setdefault(k, []).append(i)
    for positions in idx.values():
        vals = out[positions]
        rng.shuffle(vals)
        out[positions] = vals
    return out
