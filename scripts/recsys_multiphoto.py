"""Несколько фото одного человека: агрегация, контекст, выбор кадра.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_multiphoto.py

Предыстория. Разложение дисперсии оценок пользователя (несмещённое, шум взят из ретеста)
показало неожиданное:

    КТО   — между людьми            0.213   33%
    КАДР  — какое фото человека     0.217   34%     <-- столько же, сколько «кто»!
    ШУМ   — неточность разметки     0.207   33%

Выбор кадра значит РОВНО СТОЛЬКО ЖЕ, сколько выбор человека. И модель берёт лишь половину
кадрового сигнала (внутриперсонный Spearman 0.355 при потолке 0.715). Значит недостающая
треть до потолка — не «предел входа», а неизвлечённый кадровый сигнал.

Плюс всплыла УТЕЧКА: все прошлые абляции считались обычным KFold, а фото одного человека
попадали и в train, и в test. Честный GroupKFold по человеку: 0.5679 -> 0.5421.
Здесь везде GroupKFold.

Три задачи:
  A. ПОКАДРОВАЯ (как раньше, таргет — оценка кадра): помогает ли контекст остальных фото
     человека? Ключевой вариант — «отклонение от собственного среднего»: явная кодировка
     того, чем ЭТОТ кадр отличается от типичного кадра этого же человека.
  B. ПЕРСОНАЛЬНАЯ (таргет — средняя оценка человека; именно она релевантна для знакомств,
     где свайпают анкету, а не фото): бьёт ли усреднение эмбеддингов одно фото?
  C. ВЫБОР КАДРА (продуктовая): умеет ли модель выбрать, какое из фото человека
     пользователю понравится больше? Это «какое фото ставить главным в анкете».

Пишет metrics/recsys_multiphoto.json.
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
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel, CLIPModel, SiglipVisionModel

from age_gap import beauty, contrastive
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)

ENCODERS = {                                  # лучший ансамбль (id, батч)
    "dino_b":   ("facebook/dinov2-base", 32),
    "clip_b32": ("openai/clip-vit-base-patch32", 32),
    "clip_L14": ("openai/clip-vit-large-patch14", 16),
    "siglip_b": ("google/siglip-base-patch16-224", 32),
}
GRID = [(k, a) for k in [40, 80, 160] for a in [50, 200, 800]]


@torch.no_grad()
def _embed(kind: str, paths, device) -> np.ndarray:
    from PIL import Image
    mid, batch = ENCODERS[kind]
    proc = AutoImageProcessor.from_pretrained(mid)
    if kind.startswith("clip"):
        vm = CLIPModel.from_pretrained(mid).vision_model
    elif kind.startswith("siglip"):
        vm = SiglipVisionModel.from_pretrained(mid)
    else:
        vm = AutoModel.from_pretrained(mid)
    vm = vm.to(device).eval()
    out = np.zeros((len(paths), vm.config.hidden_size), dtype=np.float32)
    buf_i, buf_im = [], []

    def flush():
        if not buf_i:
            return
        pv = torch.from_numpy(proc(images=buf_im, return_tensors="np")["pixel_values"]).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out[buf_i] = vm(pixel_values=pv).pooler_output.float().cpu().numpy()
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
    return out.astype(np.float64)


def _oof(X, y, prior, groups):
    """Residual-ridge поверх приора. GroupKFold по человеку — без утечки фото одного лица."""
    def fit(tr, te, k, a):
        sc = StandardScaler().fit(X[tr])
        pca = PCA(min(k, len(tr) - 1, X.shape[1]), random_state=0, whiten=True).fit(sc.transform(X[tr]))
        c, d = np.polyfit(prior[tr], y[tr], 1)
        rg = Ridge(alpha=a).fit(pca.transform(sc.transform(X[tr])), y[tr] - (c * prior[tr] + d))
        return (c * prior[te] + d) + rg.predict(pca.transform(sc.transform(X[te])))

    oof = np.full(len(y), np.nan)
    cv = GroupKFold(5) if groups is not None else KFold(5, shuffle=True, random_state=0)
    split = cv.split(X, y, groups) if groups is not None else cv.split(X)
    for tr, te in split:
        g_tr = groups[tr] if groups is not None else None
        inner = (GroupKFold(3).split(X[tr], y[tr], g_tr) if groups is not None
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
    """Разложить качество модели: МЕЖДУ людьми (кто) и ВНУТРИ человека (кадр)."""
    by = collections.defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    multi = [ix for ix in by.values() if len(ix) >= 2]
    yw, pw, yb, pb = [], [], [], []
    for ix in multi:
        yy, pp = y[ix], pred[ix]
        yw += list(yy - yy.mean())
        pw += list(pp - pp.mean())
        yb.append(yy.mean())
        pb.append(pp.mean())
    return (float(spearmanr(yb, pb).statistic) if len(yb) > 2 else float("nan"),
            float(spearmanr(yw, pw).statistic) if len(yw) > 2 else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", default="reports/rating/ratings.jsonl")
    ap.add_argument("--prior-weights", default="data_beauty/weights/beauty_dinov2.pt")
    args = ap.parse_args()

    device = beauty.pick_device()
    hd = data_path("data_dir", "interim", "faces_hires")

    # --- кто есть кто: все взрослые фото людей, у которых есть хоть одна метка ---
    ft = contrastive.build_face_table("vk").drop_duplicates("face_id")
    lab = {r["face_id"]: float(r["score"])
           for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")}
    lab = {f: s for f, s in lab.items() if (hd / f"{f}.jpg").exists()}
    persons = {r.post_id for r in ft.itertuples() if r.face_id in lab}   # пост == личность
    rows = [(r.face_id, r.post_id) for r in ft.itertuples()
            if r.post_id in persons and (hd / f"{r.face_id}.jpg").exists()]
    fids = [f for f, _ in rows]
    pers = np.array([p for _, p in rows])
    idx = {f: i for i, f in enumerate(fids)}
    log.info("людей %d | всех их фото %d | из них размечено %d", len(persons), len(fids), len(lab))

    # --- эмбеддинги + приор для ВСЕХ фото (кешируем: пересчёт дорогой) ---
    cache = resolve_path("data_beauty", "cache", "embeds_person.npz")
    Z = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    if list(Z.get("fids", [])) != fids:                       # состав поменялся -> пересчёт
        Z = {"fids": np.array(fids)}
    paths = [hd / f"{f}.jpg" for f in fids]
    for k in ENCODERS:
        if k not in Z:
            Z[k] = _embed(k, paths, device)
            log.info("%-9s %s", k, Z[k].shape)
            np.savez(cache, **Z)
    if "prior" not in Z:
        ck = torch.load(resolve_path(args.prior_weights), map_location=device, weights_only=False)
        bb = ck.get("backbone", "dinov2")
        m = beauty.BeautyRegressor(bb, unfreeze_top=int(ck["unfreeze_vision"])).to(device)
        m.load_state_dict(ck["state_dict"])
        Z["prior"] = beauty.score_paths(m, paths, device, float(ck["mu"]), float(ck["sd"]),
                                        bb, batch=64).astype(np.float64)
        np.savez(cache, **Z)
        del m
        torch.cuda.empty_cache()

    EMB = np.hstack([Z[k] for k in ENCODERS])                 # (8742, 3328)
    PRI = Z["prior"]

    # --- средний эмбеддинг человека (по ВСЕМ его фото, включая неразмеченные) ---
    by_p = collections.defaultdict(list)
    for i, p in enumerate(pers):
        by_p[p].append(i)
    MEAN = np.zeros_like(EMB)
    LOO = np.zeros_like(EMB)                                  # среднее ОСТАЛЬНЫХ фото (leave-one-out)
    for _p, ix in by_p.items():
        m = EMB[ix].mean(0)
        MEAN[ix] = m
        n = len(ix)
        LOO[ix] = m if n == 1 else (m * n - EMB[ix]) / (n - 1)

    # ================= A. ПОКАДРОВАЯ задача =================
    li = np.array([idx[f] for f in lab])                      # индексы размеченных лиц
    y = np.array([lab[f] for f in lab])
    g = pers[li]
    prior = PRI[li]

    VAR_A = {
        "1 фото (текущий)":              EMB[li],
        "+ среднее человека":            np.hstack([EMB[li], MEAN[li]]),
        "+ среднее ОСТАЛЬНЫХ фото":      np.hstack([EMB[li], LOO[li]]),
        "+ ОТКЛОНЕНИЕ от своего среднего": np.hstack([EMB[li], EMB[li] - MEAN[li]]),
        "ТОЛЬКО среднее человека":       MEAN[li],
    }
    res = {"A_покадровая (GroupKFold)": {}}
    for name, X in VAR_A.items():
        oof = _oof(X, y, prior, g)
        t = float(spearmanr(y, oof).statistic)
        b, w = _split_perf(y, oof, g)
        res["A_покадровая (GroupKFold)"][name] = {
            "taste": round(t, 4), "между людьми": round(b, 4), "внутри человека": round(w, 4),
            "dim": int(X.shape[1])}
        log.info("A %-34s taste=%.4f  между=%.4f  внутри=%.4f", name, t, b, w)

    # ================= B. ПЕРСОНАЛЬНАЯ задача =================
    # только люди с >=2 размеченными фото: таргет = средняя оценка (надёжнее одиночной)
    lab_by_p = collections.defaultdict(list)
    for f, s in lab.items():
        lab_by_p[pers[idx[f]]].append((idx[f], s))
    pm = {p: v for p, v in lab_by_p.items() if len(v) >= 2}
    pk = sorted(pm)
    yb = np.array([np.mean([s for _, s in pm[p]]) for p in pk])
    prb = np.array([np.mean([PRI[i] for i, _ in pm[p]]) for p in pk])
    first = np.array([sorted(i for i, _ in pm[p])[0] for p in pk])           # одно (первое) фото
    VAR_B = {
        "одно фото":                 EMB[first],
        "среднее по размеченным":    np.array([EMB[[i for i, _ in pm[p]]].mean(0) for p in pk]),
        "среднее по ВСЕМ фото":      np.array([EMB[by_p[p]].mean(0) for p in pk]),
    }
    res["B_персональная (таргет=средняя оценка человека)"] = {"_n_людей": len(pk)}
    for name, X in VAR_B.items():
        oof = _oof(X, yb, prb, None)
        t = float(spearmanr(yb, oof).statistic)
        res["B_персональная (таргет=средняя оценка человека)"][name] = {"spearman": round(t, 4)}
        log.info("B %-24s spearman=%.4f", name, t)

    # ================= C. ВЫБОР КАДРА =================
    # среди фото ОДНОГО человека: угадывает ли модель, какое понравилось больше?
    oof_best = _oof(VAR_A["+ ОТКЛОНЕНИЕ от своего среднего"], y, prior, g)
    oof_base = _oof(VAR_A["1 фото (текущий)"], y, prior, g)
    pos = {i: k for k, i in enumerate(li)}
    res["C_выбор_кадра (пары ВНУТРИ человека)"] = {}
    for nm, oo in [("1 фото (текущий)", oof_base), ("+ ОТКЛОНЕНИЕ", oof_best)]:
        ok = tot = 0
        for _p, v in pm.items():
            for a in range(len(v)):
                for b_ in range(a + 1, len(v)):
                    (ia, sa), (ib, sb) = v[a], v[b_]
                    if sa == sb:
                        continue
                    tot += 1
                    ok += int((oo[pos[ia]] > oo[pos[ib]]) == (sa > sb))
        acc = ok / tot if tot else float("nan")
        res["C_выбор_кадра (пары ВНУТРИ человека)"][nm] = {
            "accuracy": round(acc, 4), "n_пар": tot, "SE": round((0.25 / tot) ** 0.5, 4)}
        log.info("C %-20s acc=%.4f  (n=%d)", nm, acc, tot)

    res["_ref"] = {
        "дисперсия_КТО": 0.213, "дисперсия_КАДР": 0.217, "дисперсия_ШУМ": 0.207,
        "потолок_общий": 0.821, "потолок_внутри_человека": 0.715,
        "утечка_KFold_vs_GroupKFold": "0.5679 -> 0.5421",
    }
    dst = data_path("metrics_dir", "recsys_multiphoto.json")
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("записано: %s", dst)


if __name__ == "__main__":
    main()
