"""РЕЦИПРОКНОСТЬ: показывать не «кто нравится мне», а «кому я тоже нравлюсь».

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_sim4.py

Односторонний рекомендер максимизирует P(A->B) — «понравится ли B пользователю A». Но матч
требует взаимности: P(match) = P(A->B) * P(B->A). Самых привлекательных показывают всем, они
переборчивы -> отвечают редко -> матчей мало.

Популяция строится на РЕАЛЬНЫХ данных: 1619 лиц с популяционным приором красоты и эмбеддингами.
  * вкус каждого: w_i случайного направления, норма откалибрована так, чтобы ранжирование юзера
    коррелировало с популяционным приором на ~0.80 — ровно как ИЗМЕРЕНО у реального юзера;
  * переборчивость растёт с собственной привлекательностью (топовые лайкают ~10%, нижние ~55%);
  * фокусный юзер A имеет НАСТОЯЩИЙ вкус (твои 1619 оценок), его привлекательность свипаем.

Политики показа: random | one_sided (max P(A->B)) | reciprocal (max P(A->B)*P(B->A)).
Пишет metrics/recsys_sim4.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from age_gap.common.io import data_path, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pca", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--show", type=int, default=100, help="сколько профилей показываем A")
    ap.add_argument("--taste-corr", type=float, default=0.80,
                    help="корреляция вкуса юзера с популяционным приором (ИЗМЕРЕНО = 0.80)")
    args = ap.parse_args()

    z = np.load(resolve_path("data_beauty", "cache", "recsys_features.npz"), allow_pickle=True)
    E, prior, y = z["E"], z["prior"], z["y"]
    n = len(y)

    sc = StandardScaler().fit(E)
    pca = PCA(args.pca, random_state=0, whiten=True).fit(sc.transform(E))
    Z = pca.transform(sc.transform(E))                      # whitened -> var(w·z) = ||w||²
    sd_p = float(np.std(prior))
    # ||w|| подобрана так, чтобы corr(score_i, prior) ≈ taste_corr
    w_norm = sd_p * np.sqrt(1.0 / args.taste_corr ** 2 - 1.0)

    # переборчивость: чем привлекательнее сам, тем выше порог (лайкает меньшую долю)
    pct = np.argsort(np.argsort(prior)) / (n - 1)
    q = 0.45 + 0.45 * pct                                    # доля отвергаемых: 45%..90%

    results = {}
    for a_pct_name, a_pct in [("low", 0.15), ("mid", 0.50), ("high", 0.85)]:
        # привлекательность фокусного юзера A (его собственное лицо)
        prior_A = float(np.quantile(prior, a_pct))
        agg = {p: {"match": [], "p_a_likes": [], "p_b_likes": []} for p in ["random", "one_sided", "reciprocal"]}
        for s in range(args.seeds):
            rng = np.random.default_rng(500 + s)
            # вкусы кандидатов
            W = rng.normal(size=(n, args.pca))
            W /= np.linalg.norm(W, axis=1, keepdims=True)
            W *= w_norm
            # P(B -> A): как каждый кандидат B оценивает лицо юзера A
            #   score_B(A) = prior_A + w_B · z_A ; z_A неизвестен -> берём случайное лицо как «лицо A»
            a_idx = int(np.argmin(np.abs(prior - prior_A)))
            zA = Z[a_idx]
            score_B_of_A = prior_A + W @ zA
            # порог каждого B по его собственному распределению оценок
            S = prior[None, :] + W @ Z.T                     # [B, j] — как B оценивает всех
            thr = np.quantile(S, q[:, None], axis=1).diagonal() if False else \
                np.array([np.quantile(S[i], q[i]) for i in range(n)])
            sdS = S.std(1) + 1e-9
            p_b_likes_a = _sig(2.0 * (score_B_of_A - thr) / sdS)

            # P(A -> B): НАСТОЯЩИЙ вкус юзера (его оценки 1..5)
            thrA = np.quantile(y, 0.77)                       # A лайкает ~23% (как в его разметке)
            p_a_likes_b = _sig(2.0 * (y - thrA) / (y.std() + 1e-9))

            cand = np.arange(n)
            cand = cand[cand != a_idx]
            for pol in ["random", "one_sided", "reciprocal"]:
                if pol == "random":
                    sel = rng.choice(cand, size=args.show, replace=False)
                elif pol == "one_sided":
                    sel = cand[np.argsort(-p_a_likes_b[cand])[: args.show]]
                else:
                    sel = cand[np.argsort(-(p_a_likes_b[cand] * p_b_likes_a[cand]))[: args.show]]
                agg[pol]["match"].append(float(np.mean(p_a_likes_b[sel] * p_b_likes_a[sel])))
                agg[pol]["p_a_likes"].append(float(np.mean(p_a_likes_b[sel])))
                agg[pol]["p_b_likes"].append(float(np.mean(p_b_likes_a[sel])))

        for pol in agg:
            results[f"A_{a_pct_name}/{pol}"] = {
                "matches_per_100_shown": round(float(np.mean(agg[pol]["match"])) * 100, 2),
                "mean_P(A_likes_B)": round(float(np.mean(agg[pol]["p_a_likes"])), 3),
                "mean_P(B_likes_A)": round(float(np.mean(agg[pol]["p_b_likes"])), 3),
            }
        for pol in ["random", "one_sided", "reciprocal"]:
            r = results[f"A_{a_pct_name}/{pol}"]
            log.info("A=%-4s %-11s матчей/100: %5.2f | P(A→B)=%.3f  P(B→A)=%.3f",
                     a_pct_name, pol, r["matches_per_100_shown"], r["mean_P(A_likes_B)"], r["mean_P(B_likes_A)"])

    out = {"show": args.show, "seeds": args.seeds, "taste_corr_with_prior": args.taste_corr,
           "pickiness": "порог лайка растёт с собственной привлекательностью (45%..90% отвергается)",
           "note": "матч = P(A→B)*P(B→A); one_sided ранжирует только по P(A→B)",
           "curves": results}
    dst = data_path("metrics_dir", "recsys_sim4.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["curves"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
