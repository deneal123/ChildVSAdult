"""FER-энкодер в ансамбль: помогает ли представление, обученное различать ВЫРАЖЕНИЯ ЛИЦА?

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_fer.py

Дыра. Разложение дисперсии оценок: КТО 33% / КАДР 34% / ШУМ 33%. Модель берёт лишь половину
кадрового сигнала (внутриперс. Spearman 0.355 при потолке 0.715). Кадр устоял против ШЕСТИ атак:
скаляры качества съёмки, внутриперсонный ранкер, контекст остальных фото, парсинг лица, карта
парсинга, полный снимок.

Почему FER — принципиально иное. Все шесть атак кормили модель признаками, к которым
CLIP/DINOv2/SigLIP так или иначе уже инвариантны либо которые они уже кодируют. FER-энкодер
обучен РАЗЛИЧАТЬ выражения — то есть ровно тот сигнал, который generic-энкодеры
целенаправленно ВЫБРАСЫВАЮТ (DINOv2 учился быть устойчивым к аугментациям; CLIP — к тому, что
не влияет на подпись). Между фото одного человека лицо то же, а выражение — разное.

Берём ДВА независимых FER-энкодера (если сигнал реален, он не должен зависеть от одной модели):
  trpakov/vit-face-expression            7 эмоций, ViT-768
  dima806/facial_emotions_image_detection 7 эмоций, ViT-768
Из каждого — И вероятности эмоций, И внутренний CLS-эмбеддинг (второе важнее: это
представление, натренированное на выражениях, а не 7 итоговых чисел).

ЦЕЛЕВАЯ МЕТРИКА — ВНУТРИПЕРСОННЫЙ Spearman, не общий вкус. Если FER поднимет только «между
людьми», он попал не в ту дыру.

Плюс: какая эмоция гонит кадровый остаток — нравятся ли улыбающиеся кадры. Это содержательный
ответ, даже если прирост нулевой.

Пишет metrics/recsys_fer.json.
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModelForImageClassification

from age_gap import beauty, contrastive
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)

ENC = ["dino_b", "clip_b32", "clip_L14", "siglip_b"]
FER = {"fer_a": "trpakov/vit-face-expression",
       "fer_b": "dima806/facial_emotions_image_detection"}
GRID = [(k, a) for k in [40, 80, 160] for a in [50, 200, 800]]


@torch.no_grad()
def _embed_fer(mid, paths, device, batch=32):
    """-> (CLS-эмбеддинг 768, вероятности эмоций 7, имена эмоций)."""
    from PIL import Image
    proc = AutoImageProcessor.from_pretrained(mid)
    net = AutoModelForImageClassification.from_pretrained(mid).to(device).eval()
    names = [net.config.id2label[i] for i in range(net.config.num_labels)]
    E = np.zeros((len(paths), net.config.hidden_size), dtype=np.float32)
    P = np.zeros((len(paths), net.config.num_labels), dtype=np.float32)
    buf_i, buf_im = [], []

    def flush():
        if not buf_i:
            return
        px = torch.from_numpy(proc(images=buf_im, return_tensors="np")["pixel_values"]).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = net(pixel_values=px, output_hidden_states=True)
        E[buf_i] = out.hidden_states[-1][:, 0].float().cpu().numpy()      # CLS последнего слоя
        P[buf_i] = out.logits.softmax(-1).float().cpu().numpy()
        buf_i.clear()
        buf_im.clear()

    for i, p in enumerate(paths):
        try:
            buf_im.append(Image.open(p).convert("RGB"))
            buf_i.append(i)
        except Exception:  # noqa: BLE001
            continue
        if len(buf_i) >= batch:
            flush()
    flush()
    torch.cuda.empty_cache()
    return E.astype(np.float64), P.astype(np.float64), names


def _oof(X, y, prior, groups):
    def fit(tr, te, k, a):
        sc = StandardScaler().fit(X[tr])
        pca = PCA(min(k, len(tr) - 1, X.shape[1]), random_state=0, whiten=True).fit(sc.transform(X[tr]))
        c, d = np.polyfit(prior[tr], y[tr], 1)
        rg = Ridge(alpha=a).fit(pca.transform(sc.transform(X[tr])), y[tr] - (c * prior[tr] + d))
        return (c * prior[te] + d) + rg.predict(pca.transform(sc.transform(X[te])))

    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        folds = [(tr[i1], tr[i2]) for i1, i2 in GroupKFold(3).split(X[tr], y[tr], groups[tr])]
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
    hd = data_path("data_dir", "interim", "faces_hires")
    Z = dict(np.load(resolve_path("data_beauty", "cache", "embeds_person.npz"), allow_pickle=True))
    fids = list(Z["fids"])
    EMB = np.hstack([Z[k] for k in ENC])
    PRI = Z["prior"]
    idx = {f: i for i, f in enumerate(fids)}

    cache = resolve_path("data_beauty", "cache", "embeds_fer.npz")
    C = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    if list(C.get("fids", [])) != fids:
        C = {"fids": np.array(fids)}
    paths = [hd / f"{f}.jpg" for f in fids]
    for k, mid in FER.items():
        if k not in C:
            E, P, names = _embed_fer(mid, paths, device)
            C[k], C[k + "|p"], C[k + "|n"] = E, P, np.array(names)
            log.info("%-6s эмбеддинг %s, эмоции %s", k, E.shape, list(names))
            np.savez(cache, **C)

    FE = np.hstack([C[k] for k in FER])                       # CLS-эмбеддинги обоих FER (1536)
    FP = np.hstack([C[k + "|p"] for k in FER])                # вероятности эмоций (14)
    enames = [f"{k}:{n}" for k in FER for n in C[k + "|n"]]

    ft = contrastive.build_face_table("vk").drop_duplicates("face_id").set_index("face_id")
    pers = np.array([ft.loc[f, "post_id"] for f in fids])
    lab = {r["face_id"]: float(r["score"])
           for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")}
    lab = {f: s for f, s in lab.items() if f in idx}
    li = np.array([idx[f] for f in lab])
    y = np.array([lab[f] for f in lab])
    g, prior = pers[li], PRI[li]

    VAR = {
        "emb (baseline)":              EMB[li],
        "emb + эмоции (14 вероятн.)":  np.hstack([EMB[li], FP[li]]),
        "emb + FER-эмбеддинг (1536)":  np.hstack([EMB[li], FE[li]]),
        "emb + FER-эмбеддинг + эмоции": np.hstack([EMB[li], FE[li], FP[li]]),
        "ТОЛЬКО FER-эмбеддинг":        FE[li],
        "ТОЛЬКО эмоции (14)":          FP[li],
    }
    res = {"варианты (GroupKFold по человеку)": {}}
    for name, X in VAR.items():
        oof = _oof(X, y, prior, g)
        t = float(spearmanr(y, oof).statistic)
        b, w = _split_perf(y, oof, g)
        res["варианты (GroupKFold по человеку)"][name] = {
            "taste": round(t, 4), "между людьми": round(b, 4),
            "ВНУТРИ человека": round(w, 4), "dim": int(X.shape[1])}
        log.info("%-30s taste=%.4f  между=%.4f  ВНУТРИ=%.4f", name, t, b, w)

    # какая эмоция гонит КАДРОВЫЙ остаток (= что меняется между фото одного человека)
    lab_pos = collections.defaultdict(list)
    for k, i in enumerate(li):
        lab_pos[pers[i]].append(k)
    mp = [v for v in lab_pos.values() if len(v) >= 2]
    dev_y = np.concatenate([y[v] - y[v].mean() for v in mp])
    F = FP[li]
    dev_f = np.vstack([F[v] - F[v].mean(0) for v in mp])
    res["эмоция ↔ кадровый остаток"] = dict(sorted(
        {n: round(float(spearmanr(dev_f[:, j], dev_y).statistic), 4) for j, n in enumerate(enames)}.items(),
        key=lambda kv: -abs(kv[1])))
    res["эмоция ↔ оценка напрямую"] = dict(sorted(
        {n: round(float(spearmanr(F[:, j], y).statistic), 4) for j, n in enumerate(enames)}.items(),
        key=lambda kv: -abs(kv[1])))
    log.info("эмоция ↔ кадровый остаток: %s", res["эмоция ↔ кадровый остаток"])

    res["_ref"] = {"внутриперс_база": 0.3555, "внутриперс_потолок": 0.715,
                   "taste_база": 0.5582, "taste_потолок": 0.821,
                   "провалившиеся_атаки_на_кадр": ["качество съёмки", "внутриперс. ранкер",
                                                   "контекст фото", "парсинг лица",
                                                   "карта парсинга", "полный снимок"]}
    dst = data_path("metrics_dir", "recsys_fer.json")
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("записано: %s", dst)


if __name__ == "__main__":
    main()
