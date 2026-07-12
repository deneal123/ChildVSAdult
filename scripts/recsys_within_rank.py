"""Живо ли направление «кадр»? Выделенный ВНУТРИПЕРСОННЫЙ ранкер против общей модели.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_within_rank.py

Разложение дисперсии показало: «какой кадр человека» весит столько же, сколько «кто это»
(0.217 против 0.213), а общая модель берёт лишь половину доступного кадрового сигнала
(внутриперсонный Spearman 0.355 при потолке 0.715). Явные кадровые фичи (резкость, свет,
ракурс) оказались НУЛЁМ — значит остаток не в качестве съёмки, а в чём-то тоньше.

Гипотеза: общая модель проваливает кадр потому, что училась на задаче «кто красивее»,
где кадровые различия — помеха. Нужен ранкер, обученный ТОЛЬКО на разностях фото ОДНОГО
человека: там «кто это» вычитается тождественно, и остаётся чистый кадр.

Прежде чем просить пользователя размечать внутриперсонные пары — проверяем на тех, что уже
есть (328 решительных пар из Likert-оценок), что направление живое.

Модели (все — на разности эмбеддингов d = emb(a) - emb(b), таргет = кто победил):
  общая модель (текущая)   — baseline: покадровый скор, сравниваем внутри человека
  ridge на разностях       — линейный ранкер, обученный на внутриперсонных парах
  + кадровые фичи          — d(emb) ⊕ d(качество)
  logreg на разностях      — то же, но классификация

Оценка: GroupKFold по ЧЕЛОВЕКУ (пары одного человека не делятся между train/test).
Антисимметрия: каждая пара подаётся в обе стороны (+d, -d), чтобы ранкер был честно
антисимметричен и не мог выучить смещение.

Пишет metrics/recsys_within_rank.json.
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from age_gap import contrastive
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)

ENC = ["dino_b", "clip_b32", "clip_L14", "siglip_b"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", default="reports/rating/ratings.jsonl")
    args = ap.parse_args()

    Z = dict(np.load(resolve_path("data_beauty", "cache", "embeds_person.npz"), allow_pickle=True))
    fids = list(Z["fids"])
    EMB = np.hstack([Z[k] for k in ENC])
    PRI = Z["prior"]
    Q = np.load(resolve_path("data_beauty", "cache", "frame_quality.npz"))["Q"]
    idx = {f: i for i, f in enumerate(fids)}

    ft = contrastive.build_face_table("vk").drop_duplicates("face_id").set_index("face_id")
    pers = np.array([ft.loc[f, "post_id"] for f in fids])
    lab = {r["face_id"]: float(r["score"])
           for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")}
    lab = {f: s for f, s in lab.items() if f in idx}

    # --- внутриперсонные пары: два фото ОДНОГО человека с РАЗНЫМИ оценками ---
    by_p = collections.defaultdict(list)
    for f, s in lab.items():
        by_p[pers[idx[f]]].append((idx[f], s))
    A, B, W, G = [], [], [], []
    for p, v in by_p.items():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                (ia, sa), (ib, sb) = v[i], v[j]
                if sa == sb:
                    continue
                A.append(ia)
                B.append(ib)
                W.append(1.0 if sa > sb else 0.0)
                G.append(p)
    A, B, W, G = np.array(A), np.array(B), np.array(W), np.array(G)
    log.info("внутриперсонных пар: %d у %d людей", len(A), len(set(G)))

    # --- baseline: ОБЩАЯ модель (покадровый residual-скор), применённая внутри человека ---
    li = np.array([idx[f] for f in lab])
    y = np.array([lab[f] for f in lab])
    gl, prior = pers[li], PRI[li]
    pos = {i: k for k, i in enumerate(li)}
    s_oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(EMB[li], y, gl):
        sc = StandardScaler().fit(EMB[li][tr])
        pca = PCA(80, random_state=0, whiten=True).fit(sc.transform(EMB[li][tr]))
        c, d = np.polyfit(prior[tr], y[tr], 1)
        rg = Ridge(alpha=200).fit(pca.transform(sc.transform(EMB[li][tr])), y[tr] - (c * prior[tr] + d))
        s_oof[te] = (c * prior[te] + d) + rg.predict(pca.transform(sc.transform(EMB[li][te])))
    base = float(np.mean([(s_oof[pos[a]] > s_oof[pos[b]]) == (w > .5) for a, b, w in zip(A, B, W, strict=True)]))

    # --- выделенные ВНУТРИПЕРСОННЫЕ ранкеры на разностях ---
    FEAT = {
        "ridge на разностях (emb)":      (EMB, None),
        "ridge на разностях (emb+кадр)": (EMB, Q),
        "ridge на разностях (только кадр)": (None, Q),
    }
    res = {"общая модель (baseline)": {"accuracy": round(base, 4), "n_пар": int(len(A)),
                                       "SE": round((0.25 / len(A)) ** 0.5, 4)}}

    def _pairfeat(E, Qq, a, b):
        parts = []
        if E is not None:
            parts.append(E[a] - E[b])
        if Qq is not None:
            parts.append(Qq[a] - Qq[b])
        return np.hstack(parts)

    for name, (E, Qq) in FEAT.items():
        oof = np.full(len(A), np.nan)
        for tr, te in GroupKFold(5).split(A, W, G):
            Xtr = np.vstack([_pairfeat(E, Qq, A[tr], B[tr]), _pairfeat(E, Qq, B[tr], A[tr])])
            ytr = np.concatenate([W[tr], 1 - W[tr]])                 # антисимметрия: обе стороны
            sc = StandardScaler().fit(Xtr)
            k = min(60, len(tr) - 1, Xtr.shape[1])
            pca = PCA(k, random_state=0, whiten=True).fit(sc.transform(Xtr))
            rg = Ridge(alpha=200).fit(pca.transform(sc.transform(Xtr)), ytr - 0.5)
            oof[te] = rg.predict(pca.transform(sc.transform(_pairfeat(E, Qq, A[te], B[te]))))
        acc = float(np.mean((oof > 0) == (W > .5)))
        res[name] = {"accuracy": round(acc, 4), "vs_baseline": round(acc - base, 4)}
        log.info("%-34s acc=%.4f  (baseline %.4f)", name, acc, base)

    # logreg-вариант (тот же вход, другая функция потерь)
    oof = np.full(len(A), np.nan)
    for tr, te in GroupKFold(5).split(A, W, G):
        Xtr = np.vstack([_pairfeat(EMB, None, A[tr], B[tr]), _pairfeat(EMB, None, B[tr], A[tr])])
        ytr = np.concatenate([W[tr], 1 - W[tr]])
        sc = StandardScaler().fit(Xtr)
        pca = PCA(min(60, len(tr) - 1), random_state=0, whiten=True).fit(sc.transform(Xtr))
        lr = LogisticRegression(C=0.05, max_iter=2000).fit(pca.transform(sc.transform(Xtr)), ytr)
        oof[te] = lr.predict_proba(pca.transform(sc.transform(_pairfeat(EMB, None, A[te], B[te]))))[:, 1]
    acc = float(np.mean((oof > 0.5) == (W > .5)))
    res["logreg на разностях (emb)"] = {"accuracy": round(acc, 4), "vs_baseline": round(acc - base, 4)}
    log.info("%-34s acc=%.4f", "logreg на разностях (emb)", acc)

    # --- РЕШАЮЩЕЕ: КРИВАЯ ОБУЧЕНИЯ ранкера. Он проигрывает общей модели потому, что сигнала
    # нет, или потому, что 328 пар на 3328 измерений — это просто мало? Растёт ли он с данными?
    # Плоская кривая -> направление мёртвое, разметку затевать незачем.
    # Растущая -> упирается в объём, и внутриперсонная разметка окупится.
    res["кривая_обучения_ранкера"] = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        accs = []
        for seed in range(5):                       # 5 подвыборок людей -> усредняем
            r = np.random.default_rng(seed)
            ppl = np.array(sorted(set(G)))
            take = set(r.choice(ppl, max(int(len(ppl) * frac), 5), replace=False))
            m = np.array([g in take for g in G])
            if m.sum() < 40:
                continue
            oo = np.full(int(m.sum()), np.nan)
            Am, Bm, Wm, Gm = A[m], B[m], W[m], G[m]
            for tr, te in GroupKFold(5).split(Am, Wm, Gm):
                Xtr = np.vstack([_pairfeat(EMB, None, Am[tr], Bm[tr]),
                                 _pairfeat(EMB, None, Bm[tr], Am[tr])])
                ytr = np.concatenate([Wm[tr], 1 - Wm[tr]])
                sc = StandardScaler().fit(Xtr)
                pca = PCA(min(60, len(tr) - 1), random_state=0, whiten=True).fit(sc.transform(Xtr))
                rg = Ridge(alpha=200).fit(pca.transform(sc.transform(Xtr)), ytr - 0.5)
                oo[te] = rg.predict(pca.transform(sc.transform(_pairfeat(EMB, None, Am[te], Bm[te]))))
            accs.append(float(np.mean((oo > 0) == (Wm > .5))))
        res["кривая_обучения_ранкера"][f"{int(frac * 100)}% пар (~{int(len(A) * frac)})"] = {
            "accuracy": round(float(np.mean(accs)), 4), "sd": round(float(np.std(accs)), 4)}
        log.info("кривая: %3d%% пар (~%3d) -> acc=%.4f ± %.3f",
                 int(frac * 100), int(len(A) * frac), np.mean(accs), np.std(accs))

    # --- АНСАМБЛЬ: общая модель ловит «кто», ранкер — «кадр». Дополняют ли они друг друга?
    # Если да, то даже нынешних 328 пар хватит для прироста ЗДЕСЬ И СЕЙЧАС.
    d_base = np.array([s_oof[pos[a]] - s_oof[pos[b]] for a, b in zip(A, B, strict=True)])
    oof_r = np.full(len(A), np.nan)
    for tr, te in GroupKFold(5).split(A, W, G):
        Xtr = np.vstack([_pairfeat(EMB, None, A[tr], B[tr]), _pairfeat(EMB, None, B[tr], A[tr])])
        ytr = np.concatenate([W[tr], 1 - W[tr]])
        sc = StandardScaler().fit(Xtr)
        pca = PCA(min(60, len(tr) - 1), random_state=0, whiten=True).fit(sc.transform(Xtr))
        rg = Ridge(alpha=200).fit(pca.transform(sc.transform(Xtr)), ytr - 0.5)
        oof_r[te] = rg.predict(pca.transform(sc.transform(_pairfeat(EMB, None, A[te], B[te]))))
    zb = d_base / (np.std(d_base) + 1e-9)
    zr = oof_r / (np.std(oof_r) + 1e-9)
    res["ансамбль (общая + ранкер)"] = {
        "корреляция предсказаний": round(float(np.corrcoef(zb, zr)[0, 1]), 4)}
    for w_ in (0.25, 0.5, 0.75):
        acc = float(np.mean(((1 - w_) * zb + w_ * zr > 0) == (W > .5)))
        res["ансамбль (общая + ранкер)"][f"вес ранкера {w_}"] = round(acc, 4)
        log.info("ансамбль: вес ранкера %.2f -> acc=%.4f  (общая %.4f)", w_, acc, base)

    # потолок: во что вообще упирается внутриперсонная точность при таком шуме разметки
    r1 = {r["face_id"]: r["score"]
          for r in read_jsonl(resolve_path("reports/rating/ratings.jsonl")) if r.get("score")}
    r2 = [r for r in read_jsonl(resolve_path("reports/rating/ratings_retest.jsonl")) if r.get("score")]
    dd = np.array([r["score"] - r1[r["face_id"]] for r in r2 if r["face_id"] in r1], float)
    res["_ref"] = {"внутриперс_потолок_spearman": 0.715, "sigma2_шума": round(float(np.var(dd, ddof=1) / 2), 3),
                   "дисперсия_КАДР": 0.217, "SE_точности": round((0.25 / len(A)) ** 0.5, 4)}
    dst = data_path("metrics_dir", "recsys_within_rank.json")
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("записано: %s", dst)


if __name__ == "__main__":
    main()
