"""Шаг 3: какая ГОЛОВА лучше на concat-эмбеддинге? Только реальные метки, без симуляций.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_head_ablation.py

Гиперпараметры (pca=40, alpha=200) подбирались под старый 768-d beauty-эмбеддинг. Для concat
(1536-d) они почти наверняка не оптимальны. Плюс вкус может быть НЕЛИНЕЕН в эмбеддинге.

Сравниваем (все — residual поверх популяционного приора):
  ridge      — линейный (текущий), сетка по pca × alpha
  rbf        — kernel ridge с RBF (нелинейность)
  mlp        — небольшой MLP
  gbm        — градиентный бустинг на PCA-фичах

Метрики: OOF Spearman с твоими 2339 оценками (5-fold) + pairwise accuracy на 215 парах.
ВАЖНО: гиперпараметры выбираются ВНУТРИ train-фолда (вложенный подбор), иначе будет утечка
и фейковое улучшение — на эти грабли мы в этом раунде уже наступали.

Пишет metrics/recsys_head_ablation.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def _make(kind, **kw):
    if kind == "ridge":
        return Ridge(alpha=kw["alpha"])
    if kind == "rbf":
        return KernelRidge(kernel="rbf", alpha=kw["alpha"], gamma=kw.get("gamma", 1.0 / kw["pca"]))
    if kind == "mlp":
        return MLPRegressor(hidden_layer_sizes=(64,), alpha=kw["alpha"], max_iter=800,
                            early_stopping=True, random_state=0)
    if kind == "gbm":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                             min_samples_leaf=30, l2_regularization=1.0,
                                             random_state=0)
    raise ValueError(kind)


def _fit_predict(kind, E, y, prior, tr, te, pca_k, alpha):
    sc = StandardScaler().fit(E[tr])
    k = min(pca_k, len(tr) - 1, E.shape[1])
    pca = PCA(k, random_state=0, whiten=True).fit(sc.transform(E[tr]))
    Ztr, Zte = pca.transform(sc.transform(E[tr])), pca.transform(sc.transform(E[te]))
    a, b = np.polyfit(prior[tr], y[tr], 1)
    resid = y[tr] - (a * prior[tr] + b)
    m = _make(kind, alpha=alpha, pca=k).fit(Ztr, resid)
    return (a * prior[te] + b) + m.predict(Zte)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="reports/rating/pairs.jsonl")
    args = ap.parse_args()

    z = np.load(resolve_path("data_beauty", "cache", "recsys_features.npz"), allow_pickle=True)
    E, prior, y = z["E"], z["prior"], z["y"]
    Ep, prior_p, pair_ids = z["Ep"], z["prior_p"], list(z["pair_ids"])
    log.info("меток=%d эмбеддинг=%d-d", len(y), E.shape[1])

    prs = [r for r in read_jsonl(resolve_path(args.pairs)) if r.get("winner")]
    pid = {f: k for k, f in enumerate(pair_ids)}

    GRID = {
        "ridge": [{"pca": p, "alpha": a} for p in [40, 80, 160, 320] for a in [50, 200, 800, 3000]],
        "rbf": [{"pca": p, "alpha": a} for p in [40, 80, 160] for a in [0.3, 1.0, 3.0]],
        "mlp": [{"pca": p, "alpha": a} for p in [40, 80] for a in [1.0, 10.0]],
        "gbm": [{"pca": p, "alpha": 0.0} for p in [40, 80]],
    }

    results = {}
    outer = list(KFold(5, shuffle=True, random_state=0).split(E))
    for kind, grid in GRID.items():
        oof = np.full(len(y), np.nan)
        picked = []
        for tr, te in outer:
            # ВЛОЖЕННЫЙ подбор гиперпараметров ВНУТРИ train-фолда (никакой утечки)
            best, best_s = None, -np.inf
            inner = list(KFold(3, shuffle=True, random_state=1).split(tr))
            for g in grid:
                sc = []
                for itr, ite in inner:
                    p = _fit_predict(kind, E, y, prior, tr[itr], tr[ite], g["pca"], g["alpha"])
                    sc.append(spearmanr(y[tr[ite]], p).statistic)
                s = float(np.mean(sc))
                if s > best_s:
                    best_s, best = s, g
            picked.append(best)
            oof[te] = _fit_predict(kind, E, y, prior, tr, te, best["pca"], best["alpha"])
        taste = float(spearmanr(y, oof).statistic)

        # пары: голова на ВСЕХ метках с самым частым выбранным гиперпараметром
        g = max(picked, key=picked.count)
        allx = np.arange(len(y))
        s_p = _fit_predict(kind, np.vstack([E, Ep]), np.concatenate([y, np.zeros(len(Ep))]),
                           np.concatenate([prior, prior_p]), allx,
                           np.arange(len(y), len(y) + len(Ep)), g["pca"], g["alpha"])
        ok = [r for r in prs if np.isfinite(s_p[pid[r["a"]]]) and np.isfinite(s_p[pid[r["b"]]])
              and s_p[pid[r["a"]]] != s_p[pid[r["b"]]]]
        acc = float(np.mean([1.0 if (s_p[pid[r["a"]]] > s_p[pid[r["b"]]]) == (r["winner"] == r["a"])
                             else 0.0 for r in ok]))
        results[kind] = {"taste_oof_spearman": round(taste, 4), "pairs_accuracy": round(acc, 4),
                         "chosen": g}
        log.info("%-6s taste_OOF=%.4f  pairs=%.4f  (%s)", kind, taste, acc, g)

    results["_reference"] = {"ridge_pca40_alpha200_prev": 0.5032, "prior_only_pairs": 0.6884}
    dst = data_path("metrics_dir", "recsys_head_ablation.json")
    dst.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
