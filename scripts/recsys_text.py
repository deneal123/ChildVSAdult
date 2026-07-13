"""Текст поста (самопрезентация человека) как фича. По сути — БИО из анкеты знакомств.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_text.py

Что это за текст. У 100% размеченных людей есть caption, медиана 1199 символов — это не подпись,
а самопрезентация: имя, возраст («мне 32 года»), рассказ о себе, просьба совета. Прямой аналог
БИО в приложении знакомств. Ни разу не использовалось.

МЕХАНИКА, задающая ожидания. Текст ОДИН на анкету, т.е. внутри человека он КОНСТАНТА. Значит на
кадровую дыру (внутриперс. Spearman 0.355, семь атак — семь нулей) он повлиять НЕ МОЖЕТ в
принципе. Бить он может только по «КТО ЭТО» (33% дисперсии, модель берёт 0.658) и по
ПЕРСОНАЛЬНОЙ задаче (0.556) — а она и есть развёртываемая.

Отсюда встроенный КОНТРОЛЬ КОРРЕКТНОСТИ: внутриперсонный Spearman обязан остаться на месте.
Если он заметно поедет — в пайплайне баг, и это будет видно.

Утечки нет: пользователь оценивал ТОЛЬКО кроп лица и текста не видел. Если текст всё же
предсказывает оценки — это содержательный результат: как человек себя описывает, коррелирует с
тем, как он выглядит.

Варианты:
  ПОКАДРОВАЯ:   emb / emb + текст / только текст
  ПЕРСОНАЛЬНАЯ: среднее по фото / + текст / только текст   <- главная, развёртываемая

Плюс: возраст ИЗ ТЕКСТА («мне 32 года») против оценки InsightFace — закрываем старый вопрос
«у нас в постах был возраст».

Пишет metrics/recsys_text.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import re

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from age_gap import beauty, contrastive
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)

ENC = ["dino_b", "clip_b32", "clip_L14", "siglip_b"]
TXT = "intfloat/multilingual-e5-base"                    # многоязычный, хорошо держит русский
GRID = [(k, a) for k in [40, 80, 160] for a in [50, 200, 800]]
AGE_RE = re.compile(r"(?:мне|мне\s+уже|возраст[:\s]+)\s*(\d{2})\s*(?:год|лет|года)?|(\d{2})\s*(?:год[аи]?|лет)\b",
                    re.IGNORECASE)
HAND = ["длина", "возраст_из_текста", "восклицаний", "вопросов", "заглавных", "цифр", "абзацев"]


def _age_from_text(t: str) -> float:
    for m in AGE_RE.finditer(t):
        v = m.group(1) or m.group(2)
        if v and 14 <= int(v) <= 80:
            return float(v)
    return np.nan


def _hand(t: str) -> list[float]:
    n = max(len(t), 1)
    return [np.log1p(len(t)), _age_from_text(t), t.count("!") / n * 100, t.count("?") / n * 100,
            sum(c.isupper() for c in t) / n, sum(c.isdigit() for c in t) / n, t.count("\n")]


@torch.no_grad()
def _embed_text(texts, device, batch=16):
    tok = AutoTokenizer.from_pretrained(TXT)
    net = AutoModel.from_pretrained(TXT).to(device).eval()
    out = np.zeros((len(texts), net.config.hidden_size), dtype=np.float32)
    for i in range(0, len(texts), batch):
        chunk = [f"query: {t[:3000]}" for t in texts[i:i + batch]]
        enc = tok(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            h = net(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        out[i:i + batch] = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()   # mean-pooling
        if i % 800 == 0 and i:
            log.info("текст: %d/%d", i, len(texts))
    torch.cuda.empty_cache()
    return out.astype(np.float64)


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

    device = beauty.pick_device()
    Z = dict(np.load(resolve_path("data_beauty", "cache", "embeds_person.npz"), allow_pickle=True))
    fids = list(Z["fids"])
    EMB = np.hstack([Z[k] for k in ENC])
    PRI = Z["prior"]
    idx = {f: i for i, f in enumerate(fids)}

    ft = contrastive.build_face_table("vk").drop_duplicates("face_id").set_index("face_id")
    pers = np.array([ft.loc[f, "post_id"] for f in fids])
    cap = {r["post_id"]: (r.get("caption") or "").strip()
           for r in read_jsonl(data_path("data_dir", "raw", "posts.jsonl"))}
    ppl = sorted(set(pers))
    log.info("людей %d, с текстом %d", len(ppl), sum(bool(cap.get(p)) for p in ppl))

    # текстовый эмбеддинг — ОДИН на человека, потом разворачиваем на его фото
    tc = resolve_path("data_beauty", "cache", "embeds_text.npz")
    if tc.exists() and list(np.load(tc, allow_pickle=True)["ppl"]) == ppl:
        T = np.load(tc)["T"]
    else:
        T = _embed_text([cap.get(p, "") for p in ppl], device)
        np.savez(tc, ppl=np.array(ppl), T=T)
    H = np.array([_hand(cap.get(p, "")) for p in ppl])
    age_txt = H[:, 1].copy()
    H[:, 1] = np.where(np.isfinite(H[:, 1]), H[:, 1], np.nanmedian(H[:, 1]))
    pi = {p: k for k, p in enumerate(ppl)}
    TXT_F = np.hstack([T, H])                                   # (людей, 768+7)
    log.info("текстовые фичи: %s", TXT_F.shape)

    lab = {r["face_id"]: float(r["score"])
           for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")}
    lab = {f: s for f, s in lab.items() if f in idx}
    li = np.array([idx[f] for f in lab])
    y = np.array([lab[f] for f in lab])
    g, prior = pers[li], PRI[li]
    TF = TXT_F[[pi[p] for p in g]]                              # текст, развёрнутый на лица

    res = {"A_покадровая (GroupKFold)": {}}
    for name, X in {"emb (baseline)": EMB[li],
                    "emb + текст": np.hstack([EMB[li], TF]),
                    "ТОЛЬКО текст": TF}.items():
        oof = _oof(X, y, prior, g)
        t = float(spearmanr(y, oof).statistic)
        b, w = _split_perf(y, oof, g)
        res["A_покадровая (GroupKFold)"][name] = {
            "taste": round(t, 4), "между людьми": round(b, 4),
            "ВНУТРИ человека": round(w, 4), "dim": int(X.shape[1])}
        log.info("A %-16s taste=%.4f  между=%.4f  ВНУТРИ=%.4f  (внутри обязан НЕ двигаться)",
                 name, t, b, w)

    # --- ПЕРСОНАЛЬНАЯ задача: развёртываемая. Таргет = средняя оценка человека ---
    by_p = collections.defaultdict(list)
    for i, p in enumerate(pers):
        by_p[p].append(i)
    lbp = collections.defaultdict(list)
    for f, s in lab.items():
        lbp[pers[idx[f]]].append((idx[f], s))
    pm = {p: v for p, v in lbp.items() if len(v) >= 2}
    pk = sorted(pm)
    yb = np.array([np.mean([s for _, s in pm[p]]) for p in pk])
    prb = np.array([np.mean([PRI[i] for i, _ in pm[p]]) for p in pk])
    Emean = np.array([EMB[by_p[p]].mean(0) for p in pk])
    Tp = TXT_F[[pi[p] for p in pk]]
    res["B_персональная (таргет = средняя оценка человека)"] = {"_n_людей": len(pk)}
    for name, X in {"среднее по фото (текущий)": Emean,
                    "среднее по фото + ТЕКСТ": np.hstack([Emean, Tp]),
                    "ТОЛЬКО текст": Tp}.items():
        t = float(spearmanr(yb, _oof(X, yb, prb, None)).statistic)
        res["B_персональная (таргет = средняя оценка человека)"][name] = round(t, 4)
        log.info("B %-26s spearman=%.4f", name, t)

    # --- возраст ИЗ ТЕКСТА против оценки InsightFace (старый долг) ---
    ga = {r["face_id"]: float(r["age_est"])
          for r in read_jsonl(data_path("data_dir", "interim", "face_genderage.jsonl"))}
    at = np.array([age_txt[pi[p]] for p in g])
    ae = np.array([ga.get(f, np.nan) for f in lab])
    ok = np.isfinite(at) & np.isfinite(ae)
    res["возраст_из_текста"] = {
        "нашёлся у % лиц": round(float(np.isfinite(at).mean() * 100), 1),
        "согласие с InsightFace (Spearman)": round(float(spearmanr(at[ok], ae[ok]).statistic), 4),
        "медиана в тексте": float(np.nanmedian(at)), "медиана InsightFace": float(np.nanmedian(ae)),
        "связь с ОЦЕНКОЙ (текстовый возраст)": round(float(spearmanr(at[ok], y[ok]).statistic), 4),
        "связь с ОЦЕНКОЙ (InsightFace)": round(float(spearmanr(ae[ok], y[ok]).statistic), 4)}
    log.info("возраст из текста: %s", res["возраст_из_текста"])

    # какие ручные текстовые признаки связаны с оценкой
    yp = np.array([np.mean([s for _, s in pm[p]]) for p in pk])
    Hp = H[[pi[p] for p in pk]]
    res["ручные_текстовые ↔ оценка человека"] = dict(sorted(
        {n: round(float(spearmanr(Hp[:, j], yp).statistic), 4) for j, n in enumerate(HAND)}.items(),
        key=lambda kv: -abs(kv[1])))

    res["_ref"] = {"внутриперс_база": 0.3555, "taste_база": 0.5582,
                   "персональная_база": 0.5561, "потолок_taste": 0.821}
    dst = data_path("metrics_dir", "recsys_text.json")
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("записано: %s", dst)


if __name__ == "__main__":
    main()
