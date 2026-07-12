"""H3: какой ЗАМОРОЖЕННЫЙ эмбеддинг лучше персонализируется? (на РЕАЛЬНЫХ метках, без симуляции)

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_embed_ablation.py

Гипотеза: энкодер beauty-модели дообучен предсказывать СКАЛЯР красоты и мог схлопнуть именно те
направления, в которых различаются индивидуальные вкусы. Тогда сырой (не beauty) эмбеддинг
персонализируется лучше.

Сравниваем источники эмбеддинга:
  beauty_dinov2 — энкодер нашей SCUT-модели (текущий выбор)
  dinov2_raw    — DINOv2 без beauty-дообучения
  clip_raw      — CLIP ViT-B/32 (на SCUT он был сильнее: 0.929 vs 0.902)
  concat        — dinov2_raw + clip_raw

Метрики — ЧЕСТНЫЕ, на реальных данных (никакой симуляции):
  * OOF Spearman residual-головы с твоими 1619 оценками (5-fold);
  * pairwise accuracy на 215 парах then/now (независимый домен).
Приор (популяционный beauty-скор) во всех вариантах ОДИН И ТОТ ЖЕ — меняется только эмбеддинг головы.

Пишет metrics/recsys_embed_ablation.json.
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
from transformers import AutoImageProcessor, AutoModel, CLIPModel

from age_gap import beauty
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


@torch.no_grad()
def _embed_raw(kind, paths, device, batch=64):
    """Сырые (не beauty) эмбеддинги: pooler_output бэкбона."""
    from PIL import Image
    mid = {"dinov2_raw": "facebook/dinov2-base", "clip_raw": "openai/clip-vit-base-patch32"}[kind]
    proc = AutoImageProcessor.from_pretrained(mid)
    if kind.startswith("clip"):
        vm = CLIPModel.from_pretrained(mid).vision_model.to(device).eval()
    else:
        vm = AutoModel.from_pretrained(mid).to(device).eval()
    out = []
    for i in range(0, len(paths), batch):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        pv = torch.from_numpy(proc(images=ims, return_tensors="np")["pixel_values"]).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out.append(vm(pixel_values=pv).pooler_output.float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def _head_eval(E, y, prior, n_pca, alpha):
    """residual-голова на эмбеддинге -> OOF Spearman с реальными оценками."""
    oof = np.full(len(y), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(E):
        sc = StandardScaler().fit(E[tr])
        k = min(n_pca, len(tr) - 1, E.shape[1])
        pca = PCA(k, random_state=0, whiten=True).fit(sc.transform(E[tr]))
        Ztr, Zte = pca.transform(sc.transform(E[tr])), pca.transform(sc.transform(E[te]))
        a, b = np.polyfit(prior[tr], y[tr], 1)
        rg = Ridge(alpha=alpha).fit(Ztr, y[tr] - (a * prior[tr] + b))
        oof[te] = (a * prior[te] + b) + rg.predict(Zte)
    return float(spearmanr(y, oof).statistic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", default="reports/rating/ratings.jsonl")
    ap.add_argument("--pairs", default="reports/rating/pairs.jsonl")
    ap.add_argument("--pca", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=200.0)
    args = ap.parse_args()

    device = beauty.pick_device()
    z = np.load(resolve_path("data_beauty", "cache", "recsys_features.npz"), allow_pickle=True)
    E_beauty, prior, y = z["E"], z["prior"], z["y"]
    Ep_beauty, prior_p, pair_ids = z["Ep"], z["prior_p"], list(z["pair_ids"])

    rows = [r for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")]
    hd = data_path("data_dir", "interim", "faces_hires")
    rows = [r for r in rows if (hd / f"{r['face_id']}.jpg").exists()]
    paths = [hd / f"{r['face_id']}.jpg" for r in rows]
    tn = resolve_path("data", "interim", "faces_hires")
    ppaths = [tn / f"{i}.jpg" for i in pair_ids]
    prs = [r for r in read_jsonl(resolve_path(args.pairs)) if r.get("winner")]
    pid = {f: k for k, f in enumerate(pair_ids)}

    cache = resolve_path("data_beauty", "cache", "raw_embeds.npz")
    if cache.exists():
        rc = np.load(cache)
        raw = {k: rc[k] for k in rc.files}
        log.info("raw embed cache HIT")
    else:
        raw = {}
        for kind in ["dinov2_raw", "clip_raw"]:
            raw[kind] = _embed_raw(kind, paths, device)
            raw[kind + "_pairs"] = _embed_raw(kind, ppaths, device)
            log.info("%s: %s", kind, raw[kind].shape)
        np.savez(cache, **raw)

    variants = {
        "beauty_dinov2 (текущий)": (E_beauty, Ep_beauty),
        "dinov2_raw": (raw["dinov2_raw"], raw["dinov2_raw_pairs"]),
        "clip_raw": (raw["clip_raw"], raw["clip_raw_pairs"]),
        "concat(dinov2+clip)": (np.hstack([raw["dinov2_raw"], raw["clip_raw"]]),
                                np.hstack([raw["dinov2_raw_pairs"], raw["clip_raw_pairs"]])),
    }

    results = {}
    for name, (E, Ep) in variants.items():
        taste = _head_eval(E, y, prior, args.pca, args.alpha)
        # pairs: голова обучена на ВСЕХ метках, приор — тот же популяционный
        sc = StandardScaler().fit(E)
        pca = PCA(min(args.pca, E.shape[1]), random_state=0, whiten=True).fit(sc.transform(E))
        a, b = np.polyfit(prior, y, 1)
        rg = Ridge(alpha=args.alpha).fit(pca.transform(sc.transform(E)), y - (a * prior + b))
        s = (a * prior_p + b) + rg.predict(pca.transform(sc.transform(Ep)))
        ok = [r for r in prs if np.isfinite(s[pid[r["a"]]]) and np.isfinite(s[pid[r["b"]]])
              and s[pid[r["a"]]] != s[pid[r["b"]]]]
        acc = float(np.mean([1.0 if (s[pid[r["a"]]] > s[pid[r["b"]]]) == (r["winner"] == r["a"]) else 0.0
                             for r in ok]))
        results[name] = {"taste_oof_spearman": round(taste, 4), "pairs_accuracy": round(acc, 4),
                         "dim": int(E.shape[1])}
        log.info("%-26s taste_OOF=%.4f  pairs=%.4f  (dim=%d)", name, taste, acc, E.shape[1])

    results["_reference"] = {"prior_only_pairs": 0.6884, "full_finetune_pairs": 0.7209}
    dst = data_path("metrics_dir", "recsys_embed_ablation.json")
    dst.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
