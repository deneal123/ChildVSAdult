"""Добиваем две двери, открытые multiphoto-экспериментом.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_frame.py

ДВЕРЬ 1 — АГРЕГАЦИЯ АНКЕТЫ. Усреднение эмбеддинга по фото человека дало +0.117 на
персональной задаче (0.494 -> 0.611). Но среднее — наивнейший вариант. В знакомствах анкету
часто судят по ЛУЧШЕМУ фото, а не по среднему. Сравниваем:
  пул ЭМБЕДДИНГОВ: mean / max / mean+max
  пул СКОРОВ (практичнее: есть покадровый скорер, чем свести?): mean / max / top-2 / median / min
«max по скорам» = «анкета стоит столько, сколько её лучшее фото» — это проверяемая гипотеза
о том, как человек читает анкету, а не просто трюк.

ДВЕРЬ 2 — КАДРОВЫЙ СИГНАЛ. Разложение дисперсии показало: «какой кадр» весит столько же,
сколько «кто это» (0.217 против 0.213), а модель берёт лишь половину доступного
(внутриперсонный Spearman 0.355 при потолке 0.715).

Причина, вероятно, архитектурная: generic-энкодеры ИНВАРИАНТНЫ к кадровым факторам by design
(DINOv2 учился быть устойчивым к аугментациям — то есть специально ВЫБРАСЫВАТЬ резкость,
экспозицию, кадрирование). Значит их надо подать ЯВНО:
  sharpness (Laplacian var), яркость, контраст, насыщенность, colorfulness,
  доля кадра под лицом, поворот головы (yaw/roll по лендмаркам), уверенность детектора.

Метрика успеха здесь — не общий вкус, а ВНУТРИПЕРСОННЫЙ Spearman (потолок 0.715):
именно он показывает, ловим ли мы кадр, а не личность.

Пишет metrics/recsys_frame.json.
"""

from __future__ import annotations

import argparse
import collections
import json

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from age_gap import contrastive
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)

ENC = ["dino_b", "clip_b32", "clip_L14", "siglip_b"]
GRID = [(k, a) for k in [40, 80, 160] for a in [50, 200, 800]]
QNAMES = ["sharpness", "brightness", "contrast", "saturation", "colorfulness",
          "face_frac", "yaw", "roll", "det_score"]


def _quality(fids, hd) -> np.ndarray:
    """Явные КАДРОВЫЕ фичи: то, к чему generic-энкодер инвариантен по построению."""
    meta = {f["face_id"]: f for f in read_jsonl(data_path("data_dir", "interim", "faces.jsonl"))}
    Q = np.zeros((len(fids), len(QNAMES)))
    for i, fid in enumerate(fids):
        img = cv2.imread(str(hd / f"{fid}.jpg"))
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        b, gr, r = img[..., 0].astype(float), img[..., 1].astype(float), img[..., 2].astype(float)
        rg, yb = np.abs(r - gr), np.abs(0.5 * (r + gr) - b)          # Hasler-Süsstrunk colorfulness
        m = meta.get(fid, {})
        bb = m.get("bbox") or [0, 0, 1, 1]
        lm = m.get("landmarks")
        yaw = roll = 0.0
        if lm and len(lm) >= 5:                                       # 5 точек: глаза, нос, углы рта
            p = np.asarray(lm, float)
            eye_c, eye_d = (p[0] + p[1]) / 2, np.linalg.norm(p[1] - p[0]) + 1e-6
            yaw = float((p[2][0] - eye_c[0]) / eye_d)                 # нос смещён от центра глаз -> поворот
            roll = float(np.arctan2(p[1][1] - p[0][1], p[1][0] - p[0][0]))
        w, h = max(bb[2] - bb[0], 1), max(bb[3] - bb[1], 1)
        Q[i] = [float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.mean()), float(g.std()),
                float(hsv[..., 1].mean()), float(rg.std() + yb.std() + 0.3 * (rg.mean() + yb.mean())),
                float(w * h), yaw, roll, float(m.get("det_score", 0.0))]
    Q[:, 0] = np.log1p(Q[:, 0])                                       # резкость: тяжёлый хвост
    Q[:, 5] = np.log1p(Q[:, 5])
    return Q


def _oof(X, y, prior, groups):
    def fit(tr, te, k, a):
        sc = StandardScaler().fit(X[tr])
        pca = PCA(min(k, len(tr) - 1, X.shape[1]), random_state=0, whiten=True).fit(sc.transform(X[tr]))
        c, d = np.polyfit(prior[tr], y[tr], 1)
        rg = Ridge(alpha=a).fit(pca.transform(sc.transform(X[tr])), y[tr] - (c * prior[tr] + d))
        return (c * prior[te] + d) + rg.predict(pca.transform(sc.transform(X[te])))

    oof = np.full(len(y), np.nan)
    cv = GroupKFold(5) if groups is not None else KFold(5, shuffle=True, random_state=0)
    for tr, te in (cv.split(X, y, groups) if groups is not None else cv.split(X)):
        inner = (GroupKFold(3).split(X[tr], y[tr], groups[tr]) if groups is not None
                 else KFold(3, shuffle=True, random_state=1).split(tr))
        folds = [(tr[i1], tr[i2]) for i1, i2 in inner]
        best, bs = None, -np.inf
        for k, a in GRID:
            s = np.mean([spearmanr(y[b], fit(a_, b, k, a)).statistic for a_, b in folds])
            if s > bs:
                bs, best = s, (k, a)
        oof[te] = fit(tr, te, *best)
    return oof


def _split_perf(y, pred, groups):
    by = collections.defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    yw, pw, yb, pb = [], [], [], []
    for ix in (v for v in by.values() if len(v) >= 2):
        yy, pp = y[ix], pred[ix]
        yw += list(yy - yy.mean())
        pw += list(pp - pp.mean())
        yb.append(yy.mean())
        pb.append(pp.mean())
    return float(spearmanr(yb, pb).statistic), float(spearmanr(yw, pw).statistic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", default="reports/rating/ratings.jsonl")
    args = ap.parse_args()

    hd = data_path("data_dir", "interim", "faces_hires")
    Z = dict(np.load(resolve_path("data_beauty", "cache", "embeds_person.npz"), allow_pickle=True))
    fids = list(Z["fids"])
    EMB = np.hstack([Z[k] for k in ENC])
    PRI = Z["prior"]
    idx = {f: i for i, f in enumerate(fids)}

    ft = contrastive.build_face_table("vk").drop_duplicates("face_id").set_index("face_id")
    pers = np.array([ft.loc[f, "post_id"] for f in fids])
    lab = {r["face_id"]: float(r["score"])
           for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")}
    lab = {f: s for f, s in lab.items() if f in idx}

    qc = resolve_path("data_beauty", "cache", "frame_quality.npz")
    if qc.exists() and list(np.load(qc, allow_pickle=True)["fids"]) == fids:
        Q = np.load(qc)["Q"]
    else:
        Q = _quality(fids, hd)
        np.savez(qc, fids=np.array(fids), Q=Q)
    log.info("кадровые фичи: %s", Q.shape)

    by_p = collections.defaultdict(list)
    for i, p in enumerate(pers):
        by_p[p].append(i)
    li = np.array([idx[f] for f in lab])
    y = np.array([lab[f] for f in lab])
    g, prior = pers[li], PRI[li]
    res = {}

    # ================= ДВЕРЬ 2: кадровые фичи (покадровая задача) =================
    base = _oof(EMB[li], y, prior, g)
    b0, w0 = _split_perf(y, base, g)
    res["D2_кадровые_фичи (покадровая, GroupKFold)"] = {}
    VAR = {
        "emb (текущий)":        EMB[li],
        "emb + КАДРОВЫЕ":       np.hstack([EMB[li], Q[li]]),
        "emb + КАДР относит. своего среднего": np.hstack(
            [EMB[li], Q[li] - np.vstack([Q[by_p[p]].mean(0) for p in pers[li]])]),
        "ТОЛЬКО кадровые":      Q[li],
    }
    for name, X in VAR.items():
        oof = _oof(X, y, prior, g)
        t = float(spearmanr(y, oof).statistic)
        b, w = _split_perf(y, oof, g)
        res["D2_кадровые_фичи (покадровая, GroupKFold)"][name] = {
            "taste": round(t, 4), "между людьми": round(b, 4),
            "ВНУТРИ человека": round(w, 4), "dim": int(X.shape[1])}
        log.info("D2 %-38s taste=%.4f  между=%.4f  ВНУТРИ=%.4f", name, t, b, w)
    # какая из кадровых фич вообще связана с внутриперсонным остатком?
    lab_pos = collections.defaultdict(list)                    # человек -> позиции в массиве меток
    for k, i in enumerate(li):
        lab_pos[pers[i]].append(k)
    mp = [v for v in lab_pos.values() if len(v) >= 2]
    dev_y = np.concatenate([y[v] - y[v].mean() for v in mp])
    dev_q = np.vstack([Q[li][v] - Q[li][v].mean(0) for v in mp])
    res["D2_какая_фича_гонит_кадр (Spearman с внутриперс. остатком)"] = {
        n: round(float(spearmanr(dev_q[:, j], dev_y).statistic), 4) for j, n in enumerate(QNAMES)}
    log.info("D2 связь фич с кадровым остатком: %s",
             res["D2_какая_фича_гонит_кадр (Spearman с внутриперс. остатком)"])

    # ================= ДВЕРЬ 1: агрегация анкеты (персональная задача) =================
    lab_by_p = collections.defaultdict(list)
    for f, s in lab.items():
        lab_by_p[pers[idx[f]]].append((idx[f], s))
    pm = {p: v for p, v in lab_by_p.items() if len(v) >= 2}
    pk = sorted(pm)
    yb = np.array([np.mean([s for _, s in pm[p]]) for p in pk])
    prb = np.array([np.mean([PRI[i] for i, _ in pm[p]]) for p in pk])
    allph = [by_p[p] for p in pk]                      # ВСЕ фото анкеты (развёртываемо)

    # (а) пул эмбеддингов
    AGG = {
        "emb: одно фото":  np.array([EMB[ix[0]] for ix in allph]),
        "emb: mean":       np.array([EMB[ix].mean(0) for ix in allph]),
        "emb: max":        np.array([EMB[ix].max(0) for ix in allph]),
        "emb: mean+max":   np.array([np.concatenate([EMB[ix].mean(0), EMB[ix].max(0)]) for ix in allph]),
        "emb: mean + КАДРОВЫЕ(mean,max)": np.array(
            [np.concatenate([EMB[ix].mean(0), Q[ix].mean(0), Q[ix].max(0)]) for ix in allph]),
    }
    res["D1_агрегация_анкеты (таргет = средняя оценка человека)"] = {"_n_людей": len(pk)}
    for name, X in AGG.items():
        t = float(spearmanr(yb, _oof(X, yb, prb, None)).statistic)
        res["D1_агрегация_анкеты (таргет = средняя оценка человека)"][name] = round(t, 4)
        log.info("D1 %-34s spearman=%.4f", name, t)

    # (б) пул СКОРОВ покадровой модели — практичнее: скорер уже есть, надо лишь свести.
    # Скор нужен для ВСЕХ фото анкеты, включая неразмеченные. Считаем без утечки:
    # в каждом фолде голова учится на размеченных лицах ДРУГИХ людей и скорит ВСЕ фото тестовых.
    s_all = np.full(len(fids), np.nan)
    for tr, te in GroupKFold(5).split(EMB[li], y, g):
        sc = StandardScaler().fit(EMB[li][tr])
        pca = PCA(80, random_state=0, whiten=True).fit(sc.transform(EMB[li][tr]))
        c, d = np.polyfit(prior[tr], y[tr], 1)
        rg = Ridge(alpha=200).fit(pca.transform(sc.transform(EMB[li][tr])),
                                  y[tr] - (c * prior[tr] + d))
        tgt = sorted({i for p in set(pers[li][te]) for i in by_p[p]})   # ВСЕ фото тестовых людей
        s_all[tgt] = (c * PRI[tgt] + d) + rg.predict(pca.transform(sc.transform(EMB[tgt])))

    POOL = {
        "скоры: mean":    lambda v: float(np.mean(v)),
        "скоры: MAX (анкета = лучшее фото)": lambda v: float(np.max(v)),
        "скоры: top-2":   lambda v: float(np.mean(sorted(v)[-2:])),
        "скоры: median":  lambda v: float(np.median(v)),
        "скоры: MIN (анкета = худшее фото)": lambda v: float(np.min(v)),
    }
    res["D1_пул_скоров (все фото анкеты, leak-free)"] = {}
    for name, fn in POOL.items():
        v = np.array([fn(s_all[ix]) for ix in allph])
        t = float(spearmanr(yb, v).statistic)
        res["D1_пул_скоров (все фото анкеты, leak-free)"][name] = round(t, 4)
        log.info("D1 %-36s spearman=%.4f", name, t)

    # (в) КОНТРОЛЬ КОНФАУНДА. max по k выборкам МЕХАНИЧЕСКИ растёт с k, а число фото в анкете
    # коррелирует с оценкой -> «победа max» может быть замаскированным «сколько фото».
    # Фиксируем ровно K фото у каждого: тогда max берётся по одинаковому k и не может кодировать его.
    nph = np.array([len(ix) for ix in allph])
    res["D1_КОНТРОЛЬ_конфаунда"] = {
        "spearman(число фото, оценка)": round(float(spearmanr(nph, yb).statistic), 4)}
    K = 3
    keep = [j for j, ix in enumerate(allph) if len(ix) >= K]
    for tag, agg in [("mean", lambda e: e.mean(0)), ("max", lambda e: e.max(0))]:
        ss = []
        for seed in range(3):                      # 3 случайные тройки фото -> усредняем
            r = np.random.default_rng(seed)
            X = np.array([agg(EMB[list(r.choice(allph[j], K, replace=False))]) for j in keep])
            ss.append(float(spearmanr(yb[keep], _oof(X, yb[keep], prb[keep], None)).statistic))
        res["D1_КОНТРОЛЬ_конфаунда"][f"{tag} при ровно {K} фото"] = {
            "среднее": round(float(np.mean(ss)), 4), "по сэмплам": [round(s, 4) for s in ss]}
        log.info("D1-контроль %-5s при K=%d: %.4f %s", tag, K, np.mean(ss), [round(s, 3) for s in ss])
    res["D1_КОНТРОЛЬ_конфаунда"]["_n_анкет"] = len(keep)

    res["_ref"] = {"потолок_общий": 0.821, "потолок_внутри_человека": 0.715,
                   "база_покадровая": 0.5582, "база_персональная_одно_фото": 0.4944,
                   "база_персональная_mean": 0.5561, "внутриперс_база": round(w0, 4),
                   "между_база": round(b0, 4), "n_фото": len(fids)}
    dst = data_path("metrics_dir", "recsys_frame.json")
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("записано: %s", dst)


if __name__ == "__main__":
    main()
