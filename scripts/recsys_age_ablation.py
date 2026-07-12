"""Возраст и пол как ЯВНЫЕ входные фичи головы. Помогают ли поверх эмбеддинга?

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_age_ablation.py

Возраст мы измеряли (InsightFace age_est), но в голову никогда не подавали. Beauty-модель при этом
ловила заметную age-корреляцию (−0.19 / −0.26), т.е. возраст небезразличен.

Ключевой вопрос: он уже закодирован в CLIP/SigLIP-эмбеддинге (и тогда явная фича бесполезна) —
или несёт что-то сверх? Проверяем, а не гадаем.

Варианты (все — residual поверх популяционного приора):
  emb                 — текущий лучший ансамбль
  emb + age           — + стандартизованный возраст
  emb + age + age²    — нелинейность по возрасту (вкус может иметь оптимум, а не монотонность)
  emb + gender
  emb + age + gender
  age (без эмбеддинга) — сколько вкуса объясняет ОДИН возраст (sanity)
  age + gender (без эмбеддинга)

Оценка: OOF-вкус на 2339 метках + 1179 пар (SE 0.0116).
Пишет metrics/recsys_age_ablation.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)

ENS = ["dino_b", "clip_b32", "clip_L14", "siglip_b"]      # лучший ансамбль


def _agegender(face_ids, root):
    """face_id -> (age_est, gender) из face_genderage.jsonl указанного data-корня."""
    src = resolve_path(root, "interim", "face_genderage.jsonl")
    m = {r["face_id"]: (float(r["age_est"]), int(r["gender"])) for r in read_jsonl(src)}
    age = np.array([m.get(f, (np.nan, np.nan))[0] for f in face_ids])
    gen = np.array([m.get(f, (np.nan, np.nan))[1] for f in face_ids])
    med = np.nanmedian(age)
    age = np.where(np.isfinite(age), age, med)
    gen = np.where(np.isfinite(gen), gen, 0.0)
    return age, gen


def _oof(E, y, prior, grid, n_extra=0):
    """PCA только по эмбеддинг-части; extra-фичи (возраст/пол) добавляются КАК ЕСТЬ."""
    def fit(tr, te, k, a):
        Eemb, Eext = (E[:, :E.shape[1] - n_extra], E[:, E.shape[1] - n_extra:]) if n_extra else (E, None)
        if Eemb.shape[1] > 0:
            sc = StandardScaler().fit(Eemb[tr])
            kk = min(k, len(tr) - 1, Eemb.shape[1])
            pca = PCA(kk, random_state=0, whiten=True).fit(sc.transform(Eemb[tr]))
            Ztr, Zte = pca.transform(sc.transform(Eemb[tr])), pca.transform(sc.transform(Eemb[te]))
        else:
            Ztr = np.zeros((len(tr), 0))
            Zte = np.zeros((len(te), 0))
        if n_extra:
            se = StandardScaler().fit(Eext[tr])
            Ztr = np.hstack([Ztr, se.transform(Eext[tr])])
            Zte = np.hstack([Zte, se.transform(Eext[te])])
        c, d = np.polyfit(prior[tr], y[tr], 1)
        rg = Ridge(alpha=a).fit(Ztr, y[tr] - (c * prior[tr] + d))
        return (c * prior[te] + d) + rg.predict(Zte)

    oof, picked = np.full(len(y), np.nan), []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(E):
        best, bs = None, -np.inf
        for k, a in grid:
            s = np.mean([spearmanr(y[tr[i2]], fit(tr[i1], tr[i2], k, a)).statistic
                         for i1, i2 in KFold(3, shuffle=True, random_state=1).split(tr)])
            if s > bs:
                bs, best = s, (k, a)
        picked.append(best)
        oof[te] = fit(tr, te, *best)
    return float(spearmanr(y, oof).statistic), max(picked, key=picked.count), fit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", default="reports/rating/ratings.jsonl")
    ap.add_argument("--pairs", default="reports/rating/pairs_all.jsonl")
    args = ap.parse_args()

    z = np.load(resolve_path("data_beauty", "cache", "recsys_features.npz"), allow_pickle=True)
    prior, y, prior_p, pair_ids = z["prior"], z["y"], z["prior_p"], list(z["pair_ids"])
    em = np.load(resolve_path("data_beauty", "cache", "embeds_multi.npz"))
    E0 = np.hstack([em[k] for k in ENS])
    Ep0 = np.hstack([em[k + "|p"] for k in ENS])

    rows = [r for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")]
    hd = data_path("data_dir", "interim", "faces_hires")
    rows = [r for r in rows if (hd / f"{r['face_id']}.jpg").exists()]
    fids = [r["face_id"] for r in rows]

    age, gen = _agegender(fids, "data_natural")           # размеченные лица — natural
    age_p, gen_p = _agegender(pair_ids, "data")           # парные лица — then/now
    log.info("возраст: natural med=%.1f | then/now med=%.1f", np.median(age), np.median(age_p))
    log.info("СЫРАЯ связь твоего вкуса с возрастом: Spearman=%.3f", spearmanr(y, age).statistic)

    def col(*a):
        return np.column_stack(a) if a else np.zeros((0, 0))

    VARIANTS = {
        "emb (текущий)":        (E0, Ep0, 0),
        "emb + age":            (np.hstack([E0, col(age)]), np.hstack([Ep0, col(age_p)]), 1),
        "emb + age + age²":     (np.hstack([E0, col(age, age ** 2)]),
                                 np.hstack([Ep0, col(age_p, age_p ** 2)]), 2),
        "emb + gender":         (np.hstack([E0, col(gen)]), np.hstack([Ep0, col(gen_p)]), 1),
        "emb + age + gender":   (np.hstack([E0, col(age, gen)]), np.hstack([Ep0, col(age_p, gen_p)]), 2),
        "ТОЛЬКО age":           (col(age), col(age_p), 1),
        "ТОЛЬКО age + gender":  (col(age, gen), col(age_p, gen_p), 2),
    }
    GRID = [(k, a) for k in [40, 80, 160] for a in [50, 200, 800]]
    prs = [r for r in read_jsonl(resolve_path(args.pairs)) if r.get("winner")]
    pid = {f: k for k, f in enumerate(pair_ids)}

    def predict_pairs(E, Ep, n_extra, g):
        """Голова на ВСЕХ метках -> скор для парных лиц."""
        Ee, Ex = (E[:, :E.shape[1] - n_extra], E[:, E.shape[1] - n_extra:]) if n_extra else (E, None)
        Pe, Px = (Ep[:, :Ep.shape[1] - n_extra], Ep[:, Ep.shape[1] - n_extra:]) if n_extra else (Ep, None)
        if Ee.shape[1] > 0:
            sc = StandardScaler().fit(Ee)
            pca = PCA(min(g[0], Ee.shape[1]), random_state=0, whiten=True).fit(sc.transform(Ee))
            Ztr, Zte = pca.transform(sc.transform(Ee)), pca.transform(sc.transform(Pe))
        else:
            Ztr, Zte = np.zeros((len(y), 0)), np.zeros((len(Ep), 0))
        if n_extra:
            se = StandardScaler().fit(Ex)
            Ztr = np.hstack([Ztr, se.transform(Ex)])
            Zte = np.hstack([Zte, se.transform(Px)])
        c, d = np.polyfit(prior, y, 1)
        rg = Ridge(alpha=g[1]).fit(Ztr, y - (c * prior + d))
        return (c * prior_p + d) + rg.predict(Zte)

    results = {}
    for name, (E, Ep, n_extra) in VARIANTS.items():
        taste, g, _ = _oof(E, y, prior, GRID, n_extra)
        sp = predict_pairs(E, Ep, n_extra, g)
        ok = [r for r in prs if r["a"] in pid and r["b"] in pid and sp[pid[r["a"]]] != sp[pid[r["b"]]]]
        acc = float(np.mean([1.0 if (sp[pid[r["a"]]] > sp[pid[r["b"]]]) == (r["winner"] == r["a"]) else 0.0
                             for r in ok]))
        results[name] = {"taste_oof": round(taste, 4), "pairs": round(acc, 4), "dim": int(E.shape[1])}
        log.info("%-22s taste=%.4f  pairs=%.4f  (dim=%d)", name, taste, acc, E.shape[1])

    results["_ref"] = {"prior_pairs": 0.6158, "ceiling_taste": 0.850, "pairs_SE": 0.0116,
                       "raw_taste_vs_age_spearman": round(float(spearmanr(y, age).statistic), 4)}
    dst = data_path("metrics_dir", "recsys_age_ablation.json")
    dst.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
