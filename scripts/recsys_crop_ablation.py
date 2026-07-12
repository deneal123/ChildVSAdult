"""Шаг 2: МАСШТАБ КРОПА и TTA. Проверяем вход, а не энкодер.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_crop_ablation.py

Масштабирование энкодеров исчерпано (вкус упёрся в ~0.565 на любом эмбеддинге). Но margin=0.4
для кропа был взят С ПОТОЛКА и ни разу не проверялся — а это реальное информационное горлышко:
узкий кроп = только лицо; широкий = причёска, плечи, кадрирование. Если вкус живёт там,
никакой энкодер этого не компенсирует, потому что информации нет во ВХОДЕ.

Сравниваем:
  margin 0.15 / 0.40 (текущий) / 0.80 — по отдельности
  конкатенация трёх масштабов
  + flip-TTA (усреднение эмбеддинга с зеркальным)

Энкодер фиксирован (clip_L14 — сильный и стабильный), чтобы изолировать эффект кропа.
Оценка на надёжном тесте: 1179 пар (SE 0.0116) + OOF-вкус на 2339 метках.
Пишет metrics/recsys_crop_ablation.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, CLIPModel

from age_gap import beauty
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)

ENCODER = "openai/clip-vit-large-patch14"
SCALES = {"m015": "_m015", "m040": "", "m080": "_m08"}   # суффиксы папок faces_hires<sfx>


@torch.no_grad()
def _embed(paths, device, flip=False, batch=16) -> np.ndarray:
    from PIL import Image, ImageOps
    proc = AutoImageProcessor.from_pretrained(ENCODER)
    vm = CLIPModel.from_pretrained(ENCODER).vision_model.to(device).eval()
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
        if p is None or not p.exists():
            continue
        try:
            im = Image.open(p).convert("RGB")
            buf_im.append(ImageOps.mirror(im) if flip else im)
            buf_i.append(i)
        except Exception:  # noqa: BLE001
            continue
        if len(buf_i) >= batch:
            flush()
    flush()
    torch.cuda.empty_cache()          # модель освободится при выходе из функции
    return out.astype(np.float64)


def _oof(E, y, prior, grid):
    def fit(tr, te, k, a):
        sc = StandardScaler().fit(E[tr])
        kk = min(k, len(tr) - 1, E.shape[1])
        pca = PCA(kk, random_state=0, whiten=True).fit(sc.transform(E[tr]))
        Ztr, Zte = pca.transform(sc.transform(E[tr])), pca.transform(sc.transform(E[te]))
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
    return float(spearmanr(y, oof).statistic), max(picked, key=picked.count)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", default="reports/rating/ratings.jsonl")
    ap.add_argument("--pairs", default="reports/rating/pairs_all.jsonl")
    args = ap.parse_args()

    device = beauty.pick_device()
    z = np.load(resolve_path("data_beauty", "cache", "recsys_features.npz"), allow_pickle=True)
    prior, y, prior_p, pair_ids = z["prior"], z["y"], z["prior_p"], list(z["pair_ids"])

    rows = [r for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")]
    base = data_path("data_dir", "interim", "faces_hires")
    rows = [r for r in rows if (base / f"{r['face_id']}.jpg").exists()]
    fids = [r["face_id"] for r in rows]
    prs = [r for r in read_jsonl(resolve_path(args.pairs)) if r.get("winner")]
    pid = {f: k for k, f in enumerate(pair_ids)}

    cache = resolve_path("data_beauty", "cache", "embeds_scales.npz")
    EMB = {k: v for k, v in np.load(cache).items()} if cache.exists() else {}
    for name, sfx in SCALES.items():
        for tag, flip in [("", False), ("_flip", True)]:
            key = name + tag
            if key in EMB:
                continue
            lp = [data_path("data_dir", "interim", f"faces_hires{sfx}") / f"{f}.jpg" for f in fids]
            pp = [resolve_path("data", "interim", f"faces_hires{sfx}") / f"{i}.jpg" for i in pair_ids]
            EMB[key] = _embed(lp, device, flip=flip)
            EMB[key + "|p"] = _embed(pp, device, flip=flip)
            log.info("%-10s %s", key, EMB[key].shape)
            np.savez(cache, **EMB)

    def tta(name):   # усреднение с зеркальным
        return ((EMB[name] + EMB[name + "_flip"]) / 2.0,
                (EMB[name + "|p"] + EMB[name + "_flip|p"]) / 2.0)

    VARIANTS = {
        "margin 0.15":            (EMB["m015"], EMB["m015|p"]),
        "margin 0.40 (текущий)":  (EMB["m040"], EMB["m040|p"]),
        "margin 0.80":            (EMB["m080"], EMB["m080|p"]),
        "0.40 + flip-TTA":        tta("m040"),
        "concat 3 масштабов":     (np.hstack([EMB[k] for k in ["m015", "m040", "m080"]]),
                                   np.hstack([EMB[k + "|p"] for k in ["m015", "m040", "m080"]])),
    }
    t3 = [tta(k) for k in ["m015", "m040", "m080"]]
    VARIANTS["concat 3 масштабов + TTA"] = (np.hstack([a for a, _ in t3]), np.hstack([b for _, b in t3]))

    GRID = [(k, a) for k in [40, 80, 160] for a in [50, 200, 800]]
    results = {}
    for name, (E, Ep) in VARIANTS.items():
        taste, g = _oof(E, y, prior, GRID)
        sc = StandardScaler().fit(E)
        pca = PCA(min(g[0], E.shape[1]), random_state=0, whiten=True).fit(sc.transform(E))
        c, d = np.polyfit(prior, y, 1)
        rg = Ridge(alpha=g[1]).fit(pca.transform(sc.transform(E)), y - (c * prior + d))
        s = (c * prior_p + d) + rg.predict(pca.transform(sc.transform(Ep)))
        ok = [r for r in prs if r["a"] in pid and r["b"] in pid and s[pid[r["a"]]] != s[pid[r["b"]]]]
        acc = float(np.mean([1.0 if (s[pid[r["a"]]] > s[pid[r["b"]]]) == (r["winner"] == r["a"]) else 0.0
                             for r in ok]))
        results[name] = {"taste_oof": round(taste, 4), "pairs": round(acc, 4), "dim": int(E.shape[1])}
        log.info("%-28s taste=%.4f  pairs=%.4f  (dim=%d)", name, taste, acc, E.shape[1])

    results["_ref"] = {"prior_pairs": 0.6158, "best_ensemble_taste": 0.5679,
                       "best_ensemble_pairs": 0.7710, "ceiling_taste": 0.850, "pairs_SE": 0.0116}
    dst = data_path("metrics_dir", "recsys_crop_ablation.json")
    dst.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
