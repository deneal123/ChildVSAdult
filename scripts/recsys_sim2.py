"""Два открытых вопроса прототипа, которые я сам пометил как непроверенные.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_sim2.py

A. СМЕЩЕНИЕ ЭКСПОЗИЦИИ (жёстко). Прошлая симуляция была слишком мягкой: кандидаты приходили
   случайной пачкой по 40, что само давало разнообразие, и жадная политика не обвалилась.
   Здесь по-настоящему: приложение ранжирует ВЕСЬ каталог и показывает по softmax(score/τ);
   при τ->0 это жадная выдача. Голова учится ТОЛЬКО на показанном -> смещение.
   Проверяем, чинит ли его IPS-взвешивание (1/propensity с клиппингом).

B. КОНТАМИНАЦИЯ МЕТКИ. В дейтинге свайп — это не только лицо (био, вторые фото, настроение).
   Симулируем: свайп = смесь ИСТИННОГО отношения к лицу (твои оценки) и НЕ-лицевого фактора
   (реальная вовлечённость поста — мы доказали, что она про охват/подачу, а не про красоту).
   Свипаем долю α «решения про лицо» и смотрим, сколько свайпов нужно, чтобы выучить лицо.

Пишет metrics/recsys_sim2.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from age_gap import beauty, contrastive
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.recsys import BayesianLinearHead

log = get_logger(__name__)


def _z(v):
    return (v - np.mean(v)) / (np.std(v) + 1e-9)


def _load(args, device):
    cache = resolve_path("data_beauty", "cache", "recsys_features.npz")
    z = np.load(cache, allow_pickle=True)
    E, prior, y = z["E"], z["prior"], z["y"]
    # порядок строк = порядок ratings.jsonl (с фильтром по наличию кропа) — восстановим face_id
    rows = [r for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")]
    hd = data_path("data_dir", "interim", "faces_hires")
    rows = [r for r in rows if (hd / f"{r['face_id']}.jpg").exists()]
    assert len(rows) == len(y), "кеш не совпал с ratings.jsonl"
    fids = [r["face_id"] for r in rows]

    # НЕ-лицевой фактор: реальная вовлечённость поста этого лица
    ft = contrastive.build_face_table("vk").drop_duplicates("face_id").set_index("face_id")
    eng = np.array([float(ft.loc[f, "e_rate"]) if f in ft.index else np.nan for f in fids])
    m = np.isfinite(eng)
    eng[~m] = np.nanmedian(eng)
    return E, prior, y, eng


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", default="reports/rating/ratings.jsonl")
    ap.add_argument("--pca", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--prior-precision", type=float, default=300.0)
    args = ap.parse_args()

    device = beauty.pick_device()
    E, prior, y, eng = _load(args, device)

    rng0 = np.random.default_rng(0)
    perm = rng0.permutation(len(y))
    hold, pool = perm[:400], perm[400:]
    sc = StandardScaler().fit(E[pool])
    pca = PCA(args.pca, random_state=0, whiten=True).fit(sc.transform(E[pool]))
    Z = pca.transform(sc.transform(E))

    ckpts = [0, 50, 100, 200, 400, 600]
    ckpts = [c for c in ckpts if c <= min(args.n, len(pool))]
    results = {}

    # ---------------- A. смещение экспозиции + IPS ----------------
    for tau in [0.05, 0.3, 1.0]:
        for ips in [False, True]:
            curves = {c: [] for c in ckpts}
            for seed in range(args.seeds):
                rng = np.random.default_rng(200 + seed)
                truth = (y >= 3.0).astype(float)
                truth = np.where(rng.random(len(y)) < 0.05, 1 - truth, truth)
                a, b = np.polyfit(prior[pool], truth[pool], 1)
                pr = a * prior + b
                head = BayesianLinearHead(args.pca, prior_precision=args.prior_precision)
                remaining = list(pool)
                for step in range(max(ckpts) + 1):
                    if step in curves:
                        curves[step].append(float(spearmanr(
                            y[hold], pr[hold] + head.predict(Z[hold])[0]).statistic))
                    if not remaining or step == max(ckpts):
                        break
                    idx = np.array(remaining)
                    s = pr[idx] + head.predict(Z[idx])[0]          # приложение ранжирует ВЕСЬ каталог
                    p = np.exp((s - s.max()) / tau)
                    p /= p.sum()                                    # softmax-политика показа
                    k = int(rng.choice(len(idx), p=p))
                    j = int(idx[k])
                    w = min(1.0 / max(p[k], 1e-6) / len(idx), 20.0) if ips else 1.0  # IPS + клиппинг
                    head.update(Z[j], truth[j] - pr[j], weight=w)
                    remaining.remove(j)
            key = f"A/tau{tau:g}/{'ips' if ips else 'naive'}"
            results[key] = [{"n": c, "taste": round(float(np.mean(curves[c])), 4)} for c in ckpts]
            log.info("%-22s %s", key, " ".join(f"n={r['n']}:{r['taste']:.3f}" for r in results[key]))

    # ---------------- B. контаминация метки (лицо vs контекст) ----------------
    zy, ze = _z(y), _z(eng)
    for alpha in [1.0, 0.7, 0.5, 0.3]:
        curves = {c: [] for c in ckpts}
        for seed in range(args.seeds):
            rng = np.random.default_rng(300 + seed)
            latent = alpha * zy + (1 - alpha) * ze          # свайп = лицо + не-лицевой фактор
            pswipe = 1.0 / (1.0 + np.exp(-2.0 * (latent - np.quantile(latent, 0.77))))
            truth = (rng.random(len(y)) < pswipe).astype(float)
            a, b = np.polyfit(prior[pool], truth[pool], 1)
            pr = a * prior + b
            head = BayesianLinearHead(args.pca, prior_precision=args.prior_precision)
            order = rng.permutation(pool)                    # случайный показ (чистый случай)
            for step in range(max(ckpts) + 1):
                if step in curves:
                    curves[step].append(float(spearmanr(
                        y[hold], pr[hold] + head.predict(Z[hold])[0]).statistic))
                if step >= len(order) or step == max(ckpts):
                    break
                j = int(order[step])
                head.update(Z[j], truth[j] - pr[j])
            # метрика — Spearman с ИСТИННЫМ отношением к лицу (y), а не со свайпом
        key = f"B/alpha{alpha:g}"
        results[key] = [{"n": c, "taste": round(float(np.mean(curves[c])), 4)} for c in ckpts]
        log.info("%-22s %s", key, " ".join(f"n={r['n']}:{r['taste']:.3f}" for r in results[key]))

    out = {"pca": args.pca, "prior_precision": args.prior_precision, "seeds": args.seeds,
           "A_note": "softmax(score/tau) по всему каталогу; tau->0 = жадная выдача; ips = 1/propensity (clip 20)",
           "B_note": "свайп = alpha*лицо + (1-alpha)*вовлечённость(не-лицевой фактор); метрика — Spearman с ИСТИННЫМ вкусом к лицу",
           "curves": results}
    dst = data_path("metrics_dir", "recsys_sim2.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["curves"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
