"""Симуляция пер-юзерной головы на РЕАЛЬНЫХ данных: свайпы, active learning, смещение экспозиции.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_sim.py

«Юзер» — это ТЫ: твои 1619 оценок берутся как истинные предпочтения. Свайп симулируется из них
(right при оценке >=3, ~23% right-rate + шум) — то есть модель учится на 1 бите вместо 2.3.
Голова — байесовская линейная на остатке поверх популяционного приора (age_gap/recsys.py),
обновляется ОНЛАЙН на каждом свайпе.

Сравниваем политики показа:
  random      — случайный кандидат;
  uncertainty — active learning (где голова не уверена);
  thompson    — исследование + эксплуатация;
  exploit     — ЖАДНАЯ (только лучшие) -> демонстрирует смещение экспозиции.

Метрики на каждом чекпоинте: Spearman на отложенных лицах + accuracy на 215 парах then/now
(независимый домен). Пишет metrics/recsys_sim.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from age_gap import beauty
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.recsys import POLICIES, BayesianLinearHead

log = get_logger(__name__)


def _features(args, device):
    """Замороженные эмбеддинги + популяционный приор (кешируются)."""
    cache = resolve_path("data_beauty", "cache", "recsys_features.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        log.info("features cache HIT")
        return (z["E"], z["prior"], z["y"], z["Ep"], z["prior_p"], list(z["pair_ids"]))

    ck = torch.load(resolve_path(args.encoder), map_location=device, weights_only=False)
    bb = ck.get("backbone", "dinov2")
    m = beauty.BeautyRegressor(bb, unfreeze_top=int(ck["unfreeze_vision"])).to(device)
    m.load_state_dict(ck["state_dict"])
    mu, sd = float(ck["mu"]), float(ck["sd"])

    rows = [r for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")]
    hd = data_path("data_dir", "interim", "faces_hires")
    rows = [r for r in rows if (hd / f"{r['face_id']}.jpg").exists()]
    paths = [hd / f"{r['face_id']}.jpg" for r in rows]
    y = np.array([float(r["score"]) for r in rows])
    E = beauty.embed_paths(m, paths, device, bb, batch=64).astype(np.float64)
    prior = beauty.score_paths(m, paths, device, mu, sd, bb, batch=64).astype(np.float64)

    prs = [r for r in read_jsonl(resolve_path(args.pairs)) if r.get("winner")]
    ids = sorted({r["a"] for r in prs} | {r["b"] for r in prs})
    tn = resolve_path("data", "interim", "faces_hires")
    live = [p if (p := tn / f"{i}.jpg").exists() else None for i in ids]
    Ep = beauty.embed_paths(m, live, device, bb, batch=64).astype(np.float64)
    prior_p = beauty.score_paths(m, live, device, mu, sd, bb, batch=64).astype(np.float64)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, E=E, prior=prior, y=y, Ep=Ep, prior_p=prior_p, pair_ids=np.array(ids))
    return E, prior, y, Ep, prior_p, ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", default="reports/rating/ratings.jsonl")
    ap.add_argument("--pairs", default="reports/rating/pairs.jsonl")
    ap.add_argument("--encoder", default="data_beauty/weights/beauty_dinov2.pt")
    ap.add_argument("--pca", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--n-max", type=int, default=1000)
    ap.add_argument("--prior-precision", type=float, nargs="+", default=[5.0],
                    help="сила усадки к приору; больше = безопаснее холодный старт")
    args = ap.parse_args()

    device = beauty.pick_device()
    E, prior, y, Ep, prior_p, pair_ids = _features(args, device)
    prs = [r for r in read_jsonl(resolve_path(args.pairs)) if r.get("winner")]
    pid = {f: k for k, f in enumerate(pair_ids)}

    # отложенные лица для честной оценки вкуса
    rng0 = np.random.default_rng(0)
    perm = rng0.permutation(len(y))
    hold, pool = perm[:400], perm[400:]

    # whiten=True — КРИТИЧНО: без него компоненты имеют разные дисперсии, усадка действует
    # неравномерно и холодный старт проваливается. С whitening усадка = n/(n+λ) по всем осям.
    sc = StandardScaler().fit(E[pool])
    pca = PCA(args.pca, random_state=0, whiten=True).fit(sc.transform(E[pool]))
    Z = pca.transform(sc.transform(E)).astype(np.float64)
    Zp = pca.transform(sc.transform(Ep)).astype(np.float64)

    ckpts = [0, 25, 50, 100, 200, 400, 700, 1000]
    ckpts = [c for c in ckpts if c <= min(args.n_max, len(pool))]

    def pair_acc(score_map):
        ok = [r for r in prs if np.isfinite(score_map[pid[r["a"]]]) and np.isfinite(score_map[pid[r["b"]]])
              and score_map[pid[r["a"]]] != score_map[pid[r["b"]]]]
        return float(np.mean([1.0 if (score_map[pid[r["a"]]] > score_map[pid[r["b"]]]) == (r["winner"] == r["a"])
                              else 0.0 for r in ok]))

    results = {}
    for pp in args.prior_precision:
      for mode in ["swipe", "likert"]:
        for pol_name in (["random", "uncertainty", "thompson", "exploit"] if mode == "swipe" else ["random"]):
            curves = {c: {"taste": [], "pairs": []} for c in ckpts}
            for seed in range(args.seeds):
                rng = np.random.default_rng(100 + seed)
                # «истинная» реакция юзера
                if mode == "swipe":
                    truth = (y >= 3.0).astype(float)
                    flip = rng.random(len(y)) < 0.05          # шум свайпа
                    truth = np.where(flip, 1 - truth, truth)
                else:
                    truth = y.copy()
                # калибровка приора в шкалу отклика (на pool)
                a, b = np.polyfit(prior[pool], truth[pool], 1)
                pr_cal = a * prior + b
                pr_cal_p = a * prior_p + b

                head = BayesianLinearHead(args.pca, prior_precision=pp, noise_var=1.0)
                remaining = list(pool)
                pick = POLICIES[pol_name]
                for step in range(max(ckpts) + 1):
                    if step in curves:
                        s_hold = pr_cal[hold] + head.predict(Z[hold])[0]
                        s_pair = pr_cal_p + head.predict(Zp)[0]
                        curves[step]["taste"].append(float(spearmanr(y[hold], s_hold).statistic))
                        curves[step]["pairs"].append(pair_acc(s_pair))
                    if not remaining or step == max(ckpts):
                        break
                    cand = np.array(rng.choice(remaining, size=min(40, len(remaining)), replace=False))
                    j = pick(cand, rng, head=head, Z=Z, prior=pr_cal)
                    head.update(Z[j], truth[j] - pr_cal[j])   # учим на ОСТАТКЕ
                    remaining.remove(j)
            key = f"pp{pp:g}/{mode}/{pol_name}"
            results[key] = [{"n": c,
                             "taste_spearman": round(float(np.mean(curves[c]["taste"])), 4),
                             "pairs_acc": round(float(np.mean(curves[c]["pairs"])), 4)} for c in ckpts]
            log.info("%-28s %s", key, " ".join(
                f"n={r['n']}:{r['taste_spearman']:.3f}/{r['pairs_acc']:.3f}" for r in results[key]))

    out = {"pca": args.pca, "seeds": args.seeds, "n_holdout": len(hold), "n_pool": len(pool),
           "swipe_rule": "right if rating>=3 (+5% noise) -> ~1 bit per interaction",
           "metric": "taste_spearman (holdout faces) / pairs_acc (215 then/now pairs)",
           "curves": results}
    dst = data_path("metrics_dir", "recsys_sim.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["curves"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
