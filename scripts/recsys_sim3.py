"""КАЧЕСТВО ВЫДАЧИ: становятся ли показываемые лица лучше по мере персонализации?

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_sim3.py

Прошлые симуляции мерили КАЧЕСТВО ОБУЧЕНИЯ (Spearman на отложенных лицах). Здесь — продуктовая
метрика: насколько хороши лица, которые юзер РЕАЛЬНО видит. Считаем по блокам показов:
  * средний ИСТИННЫЙ рейтинг показанных лиц (твоя оценка 1..5);
  * доля «свайпнул бы вправо» (y>=3) — прокси match-rate.

Политики: random | prior_only (популяционная, без обучения) | personalized (prior+голова, учится
онлайн) | thompson | oracle (argmax по истинному y — верхняя граница).

Два режима метки: alpha=1.0 (свайп чисто про лицо) и alpha=0.5 (грязный свайп — био/контекст).
Второй проверяет риск из sim2: персонализация может стать ХУЖЕ популяционной модели.

Пишет metrics/recsys_sim3.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.recsys import BayesianLinearHead

log = get_logger(__name__)


def _z(v):
    return (v - np.mean(v)) / (np.std(v) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", default="reports/rating/ratings.jsonl")
    ap.add_argument("--pca", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--prior-precision", type=float, default=300.0)
    args = ap.parse_args()

    z = np.load(resolve_path("data_beauty", "cache", "recsys_features.npz"), allow_pickle=True)
    E, prior, y = z["E"], z["prior"], z["y"]

    # не-лицевой фактор для грязной метки
    from age_gap import contrastive
    rows = [r for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")]
    hd = data_path("data_dir", "interim", "faces_hires")
    rows = [r for r in rows if (hd / f"{r['face_id']}.jpg").exists()]
    ft = contrastive.build_face_table("vk").drop_duplicates("face_id").set_index("face_id")
    eng = np.array([float(ft.loc[r["face_id"], "e_rate"]) if r["face_id"] in ft.index else np.nan
                    for r in rows])
    eng[~np.isfinite(eng)] = np.nanmedian(eng)

    sc = StandardScaler().fit(E)
    pca = PCA(args.pca, random_state=0, whiten=True).fit(sc.transform(E))
    Z = pca.transform(sc.transform(E))
    zy, ze = _z(y), _z(eng)

    nblocks = args.n // args.block
    results = {}
    for alpha in [1.0, 0.5]:
        for pol in ["random", "prior_only", "personalized", "thompson", "oracle"]:
            blocks_y = np.zeros((args.seeds, nblocks))
            blocks_r = np.zeros((args.seeds, nblocks))
            for s in range(args.seeds):
                rng = np.random.default_rng(400 + s)
                latent = alpha * zy + (1 - alpha) * ze
                pswipe = 1.0 / (1.0 + np.exp(-2.0 * (latent - np.quantile(latent, 0.77))))
                swipe = (rng.random(len(y)) < pswipe).astype(float)
                a, b = np.polyfit(prior, swipe, 1)
                pr = a * prior + b

                head = BayesianLinearHead(args.pca, prior_precision=args.prior_precision)
                remaining = list(range(len(y)))
                shown = []
                for _ in range(args.n):
                    if not remaining:
                        break
                    idx = np.array(remaining)
                    if pol == "random":
                        j = int(rng.choice(idx))
                    elif pol == "prior_only":
                        j = int(idx[np.argmax(pr[idx])])
                    elif pol == "oracle":
                        j = int(idx[np.argmax(y[idx])])
                    elif pol == "thompson":
                        j = int(idx[np.argmax(pr[idx] + head.thompson(Z[idx], rng))])
                    else:  # personalized greedy
                        j = int(idx[np.argmax(pr[idx] + head.predict(Z[idx])[0])])
                    shown.append(j)
                    if pol in ("personalized", "thompson"):
                        head.update(Z[j], swipe[j] - pr[j])
                    remaining.remove(j)
                sh = np.array(shown[: nblocks * args.block]).reshape(nblocks, args.block)
                blocks_y[s] = y[sh].mean(1)
                blocks_r[s] = (y[sh] >= 3).mean(1)
            key = f"alpha{alpha:g}/{pol}"
            results[key] = {
                "mean_true_rating_by_block": [round(float(v), 3) for v in blocks_y.mean(0)],
                "right_swipe_rate_by_block": [round(float(v), 3) for v in blocks_r.mean(0)],
            }
            log.info("%-26s rating: %s | right%%: %s", key,
                     " ".join(f"{v:.2f}" for v in blocks_y.mean(0)),
                     " ".join(f"{100*v:.0f}" for v in blocks_r.mean(0)))

    out = {"pool_mean_rating": round(float(y.mean()), 3),
           "pool_right_rate": round(float((y >= 3).mean()), 3),
           "block": args.block, "seeds": args.seeds,
           "note": "качество ВЫДАЧИ: средняя истинная оценка показанных лиц по блокам показов",
           "curves": results}
    dst = data_path("metrics_dir", "recsys_sim3.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
