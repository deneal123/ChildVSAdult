"""ПОЛНЫЙ СНИМОК, а не кроп лица. Последнее непроверенное направление.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_fullphoto.py

Кадровый сигнал (34% дисперсии оценок, взята половина) устоял против ПЯТИ атак: скаляры
качества съёмки, выделенный внутриперсонный ранкер, контекст остальных фото, фичи парсинга
лица, пространственная карта парсинга. Внутриперсонный Spearman не двигается с 0.355 (потолок
0.715) ни от чего.

Общее у всех пяти атак: они кормили модель КРОПОМ ЛИЦА. Абляция масштаба кропа (0.15/0.4/0.8)
— тоже кропы лица. ПОЛНЫЙ СНИМОК модель не видела НИ РАЗУ.

Между двумя фото одного человека лицо одно и то же. Меняется всё остальное: одежда, поза,
фигура, фон, свет, селфи-в-зеркало против студийного кадра, композиция. Гипотеза: именно там
и сидит неизвлечённая половина кадрового сигнала — ВНЕ рамки лица.

Варианты (голова прежняя, GroupKFold по человеку):
  лицо (baseline)          — текущий ансамбль на кропе
  лицо + полный снимок
  только полный снимок
  полный снимок БЕЗ лица   — снимок с зачернённым боксом лица: остаётся чисто контекст.
                             Если и это работает — сигнал точно вне лица.

Целевая метрика — ВНУТРИПЕРСОННЫЙ Spearman (кадр), а не общий вкус.

Фичи только; никаких новых отображаемых артефактов (полные снимки никуда не выводятся).
Пишет metrics/recsys_fullphoto.json.
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
from transformers import AutoImageProcessor, CLIPModel, SiglipVisionModel

from age_gap import beauty, contrastive
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.common.schemas import RawPost

log = get_logger(__name__)

ENC = ["dino_b", "clip_b32", "clip_L14", "siglip_b"]           # ансамбль на ЛИЦЕ (из кеша)
FULL = {"clip_L14": ("openai/clip-vit-large-patch14", 16),      # энкодеры для ПОЛНОГО снимка
        "siglip_b": ("google/siglip-base-patch16-224", 32)}
GRID = [(k, a) for k in [40, 80, 160] for a in [50, 200, 800]]


@torch.no_grad()
def _embed_full(kind, items, device, blank_face=False):
    """items: список (путь_к_оригиналу, bbox). blank_face -> бокс лица зачернён."""
    from PIL import Image
    mid, batch = FULL[kind]
    proc = AutoImageProcessor.from_pretrained(mid)
    vm = (CLIPModel.from_pretrained(mid).vision_model if kind.startswith("clip")
          else SiglipVisionModel.from_pretrained(mid)).to(device).eval()
    out = np.zeros((len(items), vm.config.hidden_size), dtype=np.float32)
    buf_i, buf_im = [], []

    def flush():
        if not buf_i:
            return
        pv = torch.from_numpy(proc(images=buf_im, return_tensors="np")["pixel_values"]).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out[buf_i] = vm(pixel_values=pv).pooler_output.float().cpu().numpy()
        buf_i.clear()
        buf_im.clear()

    for i, (path, bb) in enumerate(items):
        if path is None:
            continue
        try:
            im = Image.open(path).convert("RGB")
            if blank_face and bb:
                a = np.array(im)
                x0, y0, x1, y1 = (int(max(v, 0)) for v in bb)
                a[y0:y1, x0:x1] = 0                              # лицо вырезано -> только контекст
                im = Image.fromarray(a)
            buf_im.append(im)
            buf_i.append(i)
        except Exception:  # noqa: BLE001
            continue
        if len(buf_i) >= batch:
            flush()
        if i % 2000 == 0 and i:
            log.info("%s%s: %d/%d", kind, " (без лица)" if blank_face else "", i, len(items))
    flush()
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
    Z = dict(np.load(resolve_path("data_beauty", "cache", "embeds_person.npz"), allow_pickle=True))
    fids = list(Z["fids"])
    FACE = np.hstack([Z[k] for k in ENC])
    PRI = Z["prior"]
    idx = {f: i for i, f in enumerate(fids)}

    # face_id -> (оригинал, bbox)
    photo = {}
    for row in read_jsonl(data_path("data_dir", "raw", "posts.jsonl")):
        for ph in RawPost.from_dict(row).photos:
            if ph.local_path:
                photo[ph.photo_id] = ph.local_path
    items = [(None, None)] * len(fids)
    for f in read_jsonl(data_path("data_dir", "interim", "faces.jsonl")):
        i = idx.get(f["face_id"])
        if i is not None:
            lp = photo.get(f.get("photo_id"))
            items[i] = (resolve_path(lp) if lp else None, f.get("bbox"))
    log.info("оригиналов найдено: %d/%d", sum(p is not None for p, _ in items), len(fids))

    cache = resolve_path("data_beauty", "cache", "embeds_full.npz")
    C = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    if list(C.get("fids", [])) != fids:
        C = {"fids": np.array(fids)}
    for k in FULL:
        for tag, blank in [("", False), ("|noface", True)]:
            key = k + tag
            if key not in C:
                C[key] = _embed_full(k, items, device, blank_face=blank)
                log.info("%-16s %s", key, C[key].shape)
                np.savez(cache, **C)

    F_full = np.hstack([C[k] for k in FULL])                     # полный снимок
    F_ctx = np.hstack([C[k + "|noface"] for k in FULL])          # снимок БЕЗ лица (чистый контекст)

    ft = contrastive.build_face_table("vk").drop_duplicates("face_id").set_index("face_id")
    pers = np.array([ft.loc[f, "post_id"] for f in fids])
    lab = {r["face_id"]: float(r["score"])
           for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")}
    lab = {f: s for f, s in lab.items() if f in idx}
    li = np.array([idx[f] for f in lab])
    y = np.array([lab[f] for f in lab])
    g, prior = pers[li], PRI[li]

    VAR = {
        "ЛИЦО (baseline)":            FACE[li],
        "лицо + ПОЛНЫЙ СНИМОК":       np.hstack([FACE[li], F_full[li]]),
        "лицо + снимок БЕЗ ЛИЦА":     np.hstack([FACE[li], F_ctx[li]]),
        "ТОЛЬКО полный снимок":       F_full[li],
        "ТОЛЬКО контекст (без лица)": F_ctx[li],
    }
    res = {"варианты (GroupKFold по человеку)": {}}
    for name, X in VAR.items():
        oof = _oof(X, y, prior, g)
        t = float(spearmanr(y, oof).statistic)
        b, w = _split_perf(y, oof, g)
        res["варианты (GroupKFold по человеку)"][name] = {
            "taste": round(t, 4), "между людьми": round(b, 4),
            "ВНУТРИ человека": round(w, 4), "dim": int(X.shape[1])}
        log.info("%-28s taste=%.4f  между=%.4f  ВНУТРИ=%.4f", name, t, b, w)

    res["_ref"] = {"внутриперс_база": 0.3555, "внутриперс_потолок": 0.715,
                   "taste_база": 0.5582, "taste_потолок": 0.821,
                   "провалившиеся_атаки_на_кадр": ["качество съёмки", "внутриперс. ранкер",
                                                   "контекст фото", "парсинг", "карта парсинга"]}
    dst = data_path("metrics_dir", "recsys_fullphoto.json")
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("записано: %s", dst)


if __name__ == "__main__":
    main()
