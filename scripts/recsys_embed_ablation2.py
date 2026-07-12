"""Раунд 2 по эмбеддингу: сильнее энкодеры + ансамбли. Только реальные метки.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_embed_ablation2.py

Эмбеддинг — единственный рычаг, который сработал (concat(CLIP-B/32 + DINOv2): +26% вкуса),
и test-retest показал реальный запас (0.50 -> потолок 0.85). Значит давим сюда.

Кандидаты: DINOv2-base | CLIP-B/32 | CLIP-L/14 (крупнее) | SigLIP-base — и их конкатенации.
Голова: линейный residual-ridge поверх популяционного приора (ablation показал, что нелинейность
не помогает). Гиперпараметры подбираются ВЛОЖЕННО внутри train-фолда.

Метрики: OOF Spearman с 2339 реальными оценками (5-fold) + pairwise accuracy на 215 парах.
Пишет metrics/recsys_embed_ablation2.json.
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
from transformers import AutoImageProcessor, AutoModel, CLIPModel, SiglipVisionModel

from age_gap import beauty
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)

ENCODERS = {
    "dino_b": "facebook/dinov2-base",
    "clip_b32": "openai/clip-vit-base-patch32",
    "clip_L14": "openai/clip-vit-large-patch14",
    "siglip_b": "google/siglip-base-patch16-224",
}


@torch.no_grad()
def _embed(kind: str, paths, device, batch: int = 32) -> np.ndarray:
    from PIL import Image
    mid = ENCODERS[kind]
    proc = AutoImageProcessor.from_pretrained(mid)
    if kind.startswith("clip"):
        vm = CLIPModel.from_pretrained(mid).vision_model
    elif kind.startswith("siglip"):
        vm = SiglipVisionModel.from_pretrained(mid)
    else:
        vm = AutoModel.from_pretrained(mid)
    vm = vm.to(device).eval()
    dim = vm.config.hidden_size
    out = np.zeros((len(paths), dim), dtype=np.float32)
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
        if p is None:
            continue
        try:
            buf_im.append(Image.open(p).convert("RGB"))
            buf_i.append(i)
        except Exception:  # noqa: BLE001
            continue
        if len(buf_i) >= batch:
            flush()
    flush()
    del vm
    torch.cuda.empty_cache()
    return out.astype(np.float64)


def _oof(E, y, prior, grid):
    """OOF Spearman с вложенным подбором (pca, alpha) внутри train-фолда."""
    def fit(tr, te, k, a):
        sc = StandardScaler().fit(E[tr])
        kk = min(k, len(tr) - 1, E.shape[1])
        pca = PCA(kk, random_state=0, whiten=True).fit(sc.transform(E[tr]))
        Ztr, Zte = pca.transform(sc.transform(E[tr])), pca.transform(sc.transform(E[te]))
        c, d = np.polyfit(prior[tr], y[tr], 1)
        rg = Ridge(alpha=a).fit(Ztr, y[tr] - (c * prior[tr] + d))
        return (c * prior[te] + d) + rg.predict(Zte)

    oof = np.full(len(y), np.nan)
    picked = []
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
    ap.add_argument("--pairs", default="reports/rating/pairs.jsonl")
    args = ap.parse_args()

    device = beauty.pick_device()
    z = np.load(resolve_path("data_beauty", "cache", "recsys_features.npz"), allow_pickle=True)
    prior, y, prior_p, pair_ids = z["prior"], z["y"], z["prior_p"], list(z["pair_ids"])

    rows = [r for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")]
    hd = data_path("data_dir", "interim", "faces_hires")
    rows = [r for r in rows if (hd / f"{r['face_id']}.jpg").exists()]
    paths = [hd / f"{r['face_id']}.jpg" for r in rows]
    tn = resolve_path("data", "interim", "faces_hires")
    ppaths = [p if (p := tn / f"{i}.jpg").exists() else None for i in pair_ids]
    prs = [r for r in read_jsonl(resolve_path(args.pairs)) if r.get("winner")]
    pid = {f: k for k, f in enumerate(pair_ids)}

    cache = resolve_path("data_beauty", "cache", "embeds_multi.npz")
    if cache.exists():
        c = np.load(cache)
        EMB = {k: c[k] for k in c.files}
        log.info("multi-embed cache HIT (%d массивов)", len(EMB))
    else:
        EMB = {}
        for k in ENCODERS:
            EMB[k] = _embed(k, paths, device)
            EMB[k + "|p"] = _embed(k, ppaths, device)
            log.info("%-9s %s", k, EMB[k].shape)
        np.savez(cache, **EMB)

    COMBOS = [
        ("dino_b",), ("clip_b32",), ("clip_L14",), ("siglip_b",),
        ("dino_b", "clip_b32"),                       # текущий лучший
        ("dino_b", "clip_L14"),
        ("clip_L14", "siglip_b"),
        ("dino_b", "clip_b32", "clip_L14"),
        ("dino_b", "clip_b32", "clip_L14", "siglip_b"),
        ("clip_b32", "clip_L14", "siglip_b"),          # без DINOv2 (он слабейший)
        ("clip_b32", "siglip_b"),
    ]
    GRID = [(k, a) for k in [40, 80, 160] for a in [50, 200, 800]]

    results = {}
    for combo in COMBOS:
        E = np.hstack([EMB[k] for k in combo])
        Ep = np.hstack([EMB[k + "|p"] for k in combo])
        taste, g = _oof(E, y, prior, GRID)
        # пары: обучаем на всех метках с выбранным (pca, alpha)
        sc = StandardScaler().fit(E)
        pca = PCA(min(g[0], E.shape[1]), random_state=0, whiten=True).fit(sc.transform(E))
        c, d = np.polyfit(prior, y, 1)
        rg = Ridge(alpha=g[1]).fit(pca.transform(sc.transform(E)), y - (c * prior + d))
        s = (c * prior_p + d) + rg.predict(pca.transform(sc.transform(Ep)))
        ok = [r for r in prs if s[pid[r["a"]]] != s[pid[r["b"]]]]
        acc = float(np.mean([1.0 if (s[pid[r["a"]]] > s[pid[r["b"]]]) == (r["winner"] == r["a"]) else 0.0
                             for r in ok]))
        name = "+".join(combo)
        results[name] = {"taste_oof": round(taste, 4), "pairs": round(acc, 4),
                         "dim": int(E.shape[1]), "chosen_pca_alpha": list(g)}
        log.info("%-38s taste=%.4f  pairs=%.4f  (dim=%d)", name, taste, acc, E.shape[1])

    results["_ref"] = {"prior_only_pairs": 0.6884, "ceiling_taste": 0.850}
    dst = data_path("metrics_dir", "recsys_embed_ablation2.json")
    dst.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
