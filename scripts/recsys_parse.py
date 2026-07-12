"""Сегментация лица (face parsing) как фичи: геометрия, симметрия, мимика, окклюзия.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_parse.py

Дыра, в которую бьём. Разложение дисперсии оценок: КТО 33% / КАДР 34% / ШУМ 33%. Модель берёт
лишь половину кадрового сигнала (внутриперс. Spearman 0.355 при потолке 0.715). Уже проверено
и НЕ сработало: 9 скаляров качества съёмки (резкость/свет/ракурс), выделенный внутриперсонный
ранкер, контекст остальных фото.

Почему парсинг — другое. BiSeNet/SegFormer на CelebAMask-HQ даёт 19 классов (кожа, нос, глаза,
брови, губы, рот, уши, волосы, очки, шляпа, шея, одежда). Из них извлекается то, чего в
скалярах качества не было:
  ГЕОМЕТРИЯ  — доли площадей частей лица, пропорции (классические предикторы красоты)
  СИММЕТРИЯ  — flip-IoU маски лица, асимметрия глаз/бровей (тоже классика)
  МИМИКА     — открыт ли рот (видны зубы = улыбка), раскрытость глаз (щурится/моргнул)
  ПОЗА       — сколько ушей видно, смещение центроида, вертикаль глаз
  ОККЛЮЗИЯ   — волосы поперёк лица, очки, шляпа
Последние три группы МЕНЯЮТСЯ МЕЖДУ КАДРАМИ ОДНОГО ЧЕЛОВЕКА — то есть целятся ровно в дыру.

Варианты (голова прежняя: residual-ridge поверх приора, GroupKFold по человеку):
  emb                     — текущий ансамбль (baseline)
  emb + площади (19)
  emb + площади + геометрия/симметрия/мимика
  emb + карта разметки    — пространственная раскладка 8x8x19 (форма, а не только площади)
  только парсинг          — сколько вытягивает один парсинг
Плюс: связь КАЖДОЙ фичи с внутриперсонным остатком — что именно меняется между кадрами.

Пишет metrics/recsys_parse.json.
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
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from age_gap import beauty, contrastive
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)

ENC = ["dino_b", "clip_b32", "clip_L14", "siglip_b"]
SEG = "jonathandinu/face-parsing"
GRID = [(k, a) for k in [40, 80, 160] for a in [50, 200, 800]]
CLS = ["background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear",
       "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r", "neck_l", "neck", "cloth"]
FACE = [1, 2, 4, 5, 6, 7, 10, 11, 12]                 # кожа+нос+глаза+брови+рот+губы
DNAMES = ["face_frac", "eyes/face", "nose/face", "lips/face", "brows/face", "mouth/face",
          "рот_открыт(зубы)", "губы_верх/низ", "глаза_раскрытость", "асимм_глаз", "асимм_бровей",
          "асимм_ушей", "СИММЕТРИЯ(flip-IoU)", "центр_dx", "центр_dy", "вертикаль_глаз",
          "волосы_на_лице", "очки", "шляпа", "фон"]
G = 8                                                  # сетка карты разметки


@torch.no_grad()
def _parse(paths, device, batch=6):
    """-> (P: доли площадей 19, D: производные, M: карта G×G×19)."""
    from PIL import Image
    proc = SegformerImageProcessor.from_pretrained(SEG)
    net = SegformerForSemanticSegmentation.from_pretrained(SEG).to(device).eval()
    P = np.zeros((len(paths), 19))
    D = np.zeros((len(paths), len(DNAMES)))
    M = np.zeros((len(paths), G * G * 19), dtype=np.float32)
    buf_i, buf_im = [], []

    def flush():
        if not buf_i:
            return
        px = proc(images=buf_im, return_tensors="pt")["pixel_values"].to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            lg = net(pixel_values=px).logits
        lg = torch.nn.functional.interpolate(lg.float(), size=(256, 256), mode="bilinear")
        seg = lg.argmax(1).cpu().numpy()                                    # (b,256,256)
        for b, i in enumerate(buf_i):
            s = seg[b]
            oh = np.stack([(s == c) for c in range(19)]).astype(np.float32)  # (19,256,256)
            P[i] = oh.reshape(19, -1).mean(1)
            M[i] = oh.reshape(19, G, 256 // G, G, 256 // G).mean((2, 4)).transpose(1, 2, 0).ravel()
            D[i] = _derive(oh, P[i])
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
        if i % 1000 == 0 and i:
            log.info("парсинг: %d/%d", i, len(paths))
    flush()
    torch.cuda.empty_cache()
    return P, D, M


def _derive(oh, p):
    """Геометрия / симметрия / мимика / поза / окклюзия из one-hot маски (19,256,256)."""
    e = 1e-6
    face = oh[FACE].sum(0)                                     # маска лица
    ff = float(face.mean())
    eyes, brows = p[4] + p[5], p[6] + p[7]
    lips, ears = p[11] + p[12], p[8] + p[9]
    mouth_tot = p[10] + lips
    # раскрытость глаз: доля заполнения bbox глаз (щурится -> низкая)
    em = (oh[4] + oh[5]) > 0
    if em.any():
        ys, xs = np.nonzero(em)
        eye_fill = float(em.sum() / max((np.ptp(ys) + 1) * (np.ptp(xs) + 1), 1))
        eye_y = float(ys.mean() / 256)
    else:
        eye_fill = eye_y = 0.0
    # СИММЕТРИЯ: совпадение маски лица с её зеркалом
    fm = face > 0.5
    inter = float((fm & fm[:, ::-1]).sum())
    sym = inter / max(float((fm | fm[:, ::-1]).sum()), 1.0)
    if fm.any():
        ys, xs = np.nonzero(fm)
        dx, dy = float(xs.mean() / 256 - 0.5), float(ys.mean() / 256 - 0.5)
    else:
        dx = dy = 0.0
    # волосы поперёк лица: волосы внутри bbox лица сверх верхней трети
    hair_on = float((oh[13] * face.sum(0, keepdims=True).clip(0, 1)).mean()) if fm.any() else 0.0
    return [ff, eyes / (ff + e), p[2] / (ff + e), lips / (ff + e), brows / (ff + e),
            p[10] / (ff + e), p[10] / (mouth_tot + e), p[11] / (p[12] + e), eye_fill,
            abs(p[4] - p[5]) / (eyes + e), abs(p[6] - p[7]) / (brows + e),
            abs(p[8] - p[9]) / (ears + e), sym, dx, dy, eye_y, hair_on, p[3], p[14], p[0]]


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

    pc = resolve_path("data_beauty", "cache", "face_parse.npz")
    if pc.exists() and list(np.load(pc, allow_pickle=True)["fids"]) == fids:
        z = np.load(pc)
        P, D, M = z["P"], z["D"], z["M"]
    else:
        P, D, M = _parse([hd / f"{f}.jpg" for f in fids], device)
        np.savez(pc, fids=np.array(fids), P=P, D=D, M=M)
    log.info("парсинг: площади %s, производные %s, карта %s", P.shape, D.shape, M.shape)

    ft = contrastive.build_face_table("vk").drop_duplicates("face_id").set_index("face_id")
    pers = np.array([ft.loc[f, "post_id"] for f in fids])
    lab = {r["face_id"]: float(r["score"])
           for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")}
    lab = {f: s for f, s in lab.items() if f in idx}
    li = np.array([idx[f] for f in lab])
    y = np.array([lab[f] for f in lab])
    g, prior = pers[li], PRI[li]

    PD = np.hstack([P, D])
    VAR = {
        "emb (baseline)":            EMB[li],
        "emb + площади (19)":        np.hstack([EMB[li], P[li]]),
        "emb + площади + геометрия": np.hstack([EMB[li], PD[li]]),
        "emb + карта разметки 8x8":  np.hstack([EMB[li], M[li]]),
        "emb + всё вместе":          np.hstack([EMB[li], PD[li], M[li]]),
        "ТОЛЬКО парсинг (пл+геом)":  PD[li],
        "ТОЛЬКО карта разметки":     M[li],
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

    # что именно меняется между кадрами одного человека?
    lab_pos = collections.defaultdict(list)
    for k, i in enumerate(li):
        lab_pos[pers[i]].append(k)
    mp = [v for v in lab_pos.values() if len(v) >= 2]
    dev_y = np.concatenate([y[v] - y[v].mean() for v in mp])
    names = CLS + DNAMES
    F = PD[li]
    dev_f = np.vstack([F[v] - F[v].mean(0) for v in mp])
    corr = {n: round(float(spearmanr(dev_f[:, j], dev_y).statistic), 4) for j, n in enumerate(names)}
    res["связь_с_кадровым_остатком"] = dict(sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:12])
    res["связь_с_оценкой_напрямую"] = dict(sorted(
        {n: round(float(spearmanr(F[:, j], y).statistic), 4) for j, n in enumerate(names)}.items(),
        key=lambda kv: -abs(kv[1]))[:12])
    log.info("топ связей с кадровым остатком: %s", res["связь_с_кадровым_остатком"])

    res["_ref"] = {"внутриперс_база": 0.3555, "внутриперс_потолок": 0.715,
                   "taste_база": 0.5582, "дисперсия_КАДР": 0.217}
    dst = data_path("metrics_dir", "recsys_parse.json")
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("записано: %s", dst)


if __name__ == "__main__":
    main()
