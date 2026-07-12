"""СПАМ-СВАЙПИНГ: отравляет ли он реципрокный рекомендер и чинит ли это калибровка?

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_sim5.py

Спамер лайкает почти всех -> у него P(B->A)≈1 -> сырой реципрокный скор P(A->B)*P(B->A) считает
его ИДЕАЛЬНЫМ матчем для каждого. Матчи растут, но они пустые: он не хотел именно тебя.

ЧАСТЬ 1 — отравление и фикс.
  raw        : ранжируем по P(A->B) * P(B->A)                       (сырая взаимность)
  calibrated : ранжируем по P(A->B) * pct_B(A)                      (перцентиль A ВНУТРИ предпочтений B)
  Метрики: matches/100 (сырые) и QUALITY matches/100 — обе стороны держат другого в своём топ-25%.
  Свипаем долю спамеров.

ЧАСТЬ 2 — детекция без ручных правил: байесовская голова юзера пытается предсказать ЕГО ЖЕ свайпы
  на отложенных. У честного вкус есть -> AUC>0.5. У спамера свайпы ~константа -> AUC≈0.5.
  Считаем AUC детектора (right-rate vs right-rate + предсказуемость).

Пишет metrics/recsys_sim5.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from age_gap.common.io import data_path, resolve_path
from age_gap.common.logging import get_logger
from age_gap.recsys import BayesianLinearHead

log = get_logger(__name__)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pct(v):
    return np.argsort(np.argsort(v)) / (len(v) - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pca", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--show", type=int, default=100)
    ap.add_argument("--taste-corr", type=float, default=0.80)
    args = ap.parse_args()

    z = np.load(resolve_path("data_beauty", "cache", "recsys_features.npz"), allow_pickle=True)
    E, prior, y = z["E"], z["prior"], z["y"]
    n = len(y)
    sc = StandardScaler().fit(E)
    pca = PCA(args.pca, random_state=0, whiten=True).fit(sc.transform(E))
    Z = pca.transform(sc.transform(E))
    sd_p = float(np.std(prior))
    w_norm = sd_p * np.sqrt(1.0 / args.taste_corr ** 2 - 1.0)

    results = {}
    # ---------------- ЧАСТЬ 1: отравление реципрокности ----------------
    for spam_frac in [0.0, 0.1, 0.3]:
        agg = {p: {"m": [], "q": []} for p in ["raw", "calibrated"]}
        for s in range(args.seeds):
            rng = np.random.default_rng(600 + s)
            W = rng.normal(size=(n, args.pca))
            W /= np.linalg.norm(W, axis=1, keepdims=True)
            W *= w_norm
            S = prior[None, :] + W @ Z.T                       # как B оценивает всех
            # переборчивость: честные — по своей привлекательности; спамеры — лайкают ~90%
            pct_p = _pct(prior)
            q = 0.45 + 0.45 * pct_p
            is_spam = rng.random(n) < spam_frac
            q[is_spam] = 0.10                                  # спамер отвергает лишь 10%
            thr = np.array([np.quantile(S[i], q[i]) for i in range(n)])
            sdS = S.std(1) + 1e-9

            a_idx = int(np.argmin(np.abs(prior - np.quantile(prior, 0.5))))   # A — средний
            zA, prior_A = Z[a_idx], float(prior[a_idx])
            score_B_of_A = prior_A + W @ zA
            p_b_likes_a = _sig(2.0 * (score_B_of_A - thr) / sdS)              # СЫРАЯ взаимность
            # КАЛИБРОВАННАЯ: перцентиль A внутри собственных предпочтений B (не зависит от порога B)
            pct_B_of_A = np.array([(S[i] < score_B_of_A[i]).mean() for i in range(n)])

            thrA = np.quantile(y, 0.77)
            p_a_likes_b = _sig(2.0 * (y - thrA) / (y.std() + 1e-9))
            pct_A_of_B = _pct(y)                                              # где B в топе у A

            cand = np.arange(n)
            cand = cand[cand != a_idx]
            for pol in ["raw", "calibrated"]:
                key = p_a_likes_b[cand] * (p_b_likes_a[cand] if pol == "raw" else pct_B_of_A[cand])
                sel = cand[np.argsort(-key)[: args.show]]
                agg[pol]["m"].append(float(np.mean(p_a_likes_b[sel] * p_b_likes_a[sel])))
                # QUALITY: обе стороны держат другого в своём топ-25%
                agg[pol]["q"].append(float(np.mean((pct_A_of_B[sel] > 0.75) & (pct_B_of_A[sel] > 0.75))))
        for pol in agg:
            results[f"spam{spam_frac:g}/{pol}"] = {
                "matches_per_100": round(float(np.mean(agg[pol]["m"])) * 100, 2),
                "QUALITY_matches_per_100": round(float(np.mean(agg[pol]["q"])) * 100, 2),
            }
            r = results[f"spam{spam_frac:g}/{pol}"]
            log.info("спам=%2.0f%% %-11s матчей/100: %5.1f | КАЧЕСТВЕННЫХ: %5.1f",
                     100 * spam_frac, pol, r["matches_per_100"], r["QUALITY_matches_per_100"])

    # ---------------- ЧАСТЬ 2: детекция спамера по предсказуемости ----------------
    rng = np.random.default_rng(7)
    n_users, n_swipes = 300, 120
    W = rng.normal(size=(n_users, args.pca))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    W *= w_norm
    spam = rng.random(n_users) < 0.3
    feats, labels = [], []
    for u in range(n_users):
        seen = rng.choice(n, size=n_swipes, replace=False)
        s_u = prior[seen] + Z[seen] @ W[u]
        thr_u = np.quantile(s_u, 0.10 if spam[u] else 0.77)
        p = _sig(2.0 * (s_u - thr_u) / (s_u.std() + 1e-9))
        sw = (rng.random(n_swipes) < p).astype(float)
        tr, te = np.arange(80), np.arange(80, n_swipes)
        head = BayesianLinearHead(args.pca, prior_precision=50.0)
        head.update_batch(Z[seen[tr]], sw[tr] - sw[tr].mean())
        pred = head.predict(Z[seen[te]])[0]
        try:
            auc = roc_auc_score(sw[te], pred) if 0 < sw[te].mean() < 1 else 0.5
        except ValueError:
            auc = 0.5
        feats.append([sw.mean(), auc])          # right-rate, предсказуемость
        labels.append(int(spam[u]))
    F, L = np.array(feats), np.array(labels)
    auc_rate = roc_auc_score(L, F[:, 0])                       # только right-rate
    auc_pred = roc_auc_score(L, -F[:, 1])                      # только предсказуемость (ниже = спам)
    auc_both = roc_auc_score(L, F[:, 0] - F[:, 1])             # вместе
    log.info("ДЕТЕКЦИЯ: AUC right-rate=%.3f | предсказуемость=%.3f | вместе=%.3f",
             auc_rate, auc_pred, auc_both)
    log.info("  честные: right=%.2f AUC=%.2f | спамеры: right=%.2f AUC=%.2f",
             F[L == 0, 0].mean(), F[L == 0, 1].mean(), F[L == 1, 0].mean(), F[L == 1, 1].mean())

    results["detection"] = {
        "auc_right_rate_only": round(float(auc_rate), 3),
        "auc_predictability_only": round(float(auc_pred), 3),
        "auc_combined": round(float(auc_both), 3),
        "genuine": {"right_rate": round(float(F[L == 0, 0].mean()), 3),
                    "swipe_predictability_auc": round(float(F[L == 0, 1].mean()), 3)},
        "spammer": {"right_rate": round(float(F[L == 1, 0].mean()), 3),
                    "swipe_predictability_auc": round(float(F[L == 1, 1].mean()), 3)},
    }

    out = {"note": "raw = P(A→B)*P(B→A); calibrated = P(A→B)*перцентиль A внутри предпочтений B",
           "quality_match": "обе стороны держат другого в своём топ-25%", "curves": results}
    dst = data_path("metrics_dir", "recsys_sim5.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["curves"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
