"""Пересобрать канонический кеш фич рексиса на ЛУЧШЕМ эмбеддинге (concat CLIP+DINOv2).

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_build_features.py

Ablation на реальных метках показал: энкодер beauty-модели схлопывает направления
индивидуального вкуса (он учился предсказывать СКАЛЯР). Замороженный concat(DINOv2+CLIP)
даёт заметно лучшую персонализацию:
    beauty_dinov2  taste 0.433 / pairs 0.763
    concat         taste 0.503 / pairs 0.791

Кеш data_beauty/cache/recsys_features.npz — интерфейс для всех recsys_*-скриптов, поэтому
достаточно пересобрать его, и весь пайплайн поедет на лучшем эмбеддинге.

Ключи (не меняются): E, prior, y, Ep, prior_p, pair_ids
  E/Ep     — ЭМБЕДДИНГ головы (теперь concat, 1536-d)
  prior    — популяционный beauty-скор (по-прежнему из SCUT-модели; это ПРИОР, не эмбеддинг)
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel, CLIPModel

from age_gap import beauty
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)

BACKBONES = {"dinov2": "facebook/dinov2-base", "clip": "openai/clip-vit-base-patch32"}


@torch.no_grad()
def _embed(kind: str, paths, device, batch: int = 64) -> np.ndarray:
    from PIL import Image
    mid = BACKBONES[kind]
    proc = AutoImageProcessor.from_pretrained(mid)
    vm = (CLIPModel.from_pretrained(mid).vision_model if kind == "clip"
          else AutoModel.from_pretrained(mid)).to(device).eval()
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
    return out.astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", default="reports/rating/ratings.jsonl")
    ap.add_argument("--pairs", default="reports/rating/pairs.jsonl")
    ap.add_argument("--prior-weights", default="data_beauty/weights/beauty_dinov2.pt")
    args = ap.parse_args()

    device = beauty.pick_device()

    # --- размеченные лица natural ---
    rows = [r for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")]
    hd = data_path("data_dir", "interim", "faces_hires")
    rows = [r for r in rows if (hd / f"{r['face_id']}.jpg").exists()]
    paths = [hd / f"{r['face_id']}.jpg" for r in rows]
    y = np.array([float(r["score"]) for r in rows])

    # --- лица из парного теста (then/now) ---
    prs = [r for r in read_jsonl(resolve_path(args.pairs)) if r.get("winner")]
    ids = sorted({r["a"] for r in prs} | {r["b"] for r in prs})
    tn = resolve_path("data", "interim", "faces_hires")
    ppaths = [p if (p := tn / f"{i}.jpg").exists() else None for i in ids]

    # --- ЭМБЕДДИНГ головы: concat(DINOv2, CLIP) ---
    E = np.hstack([_embed("dinov2", paths, device), _embed("clip", paths, device)])
    Ep = np.hstack([_embed("dinov2", ppaths, device), _embed("clip", ppaths, device)])
    log.info("эмбеддинг головы: %s (labeled) / %s (pairs)", E.shape, Ep.shape)

    # --- ПРИОР: популяционный beauty-скор (остаётся из SCUT-модели) ---
    ck = torch.load(resolve_path(args.prior_weights), map_location=device, weights_only=False)
    bb = ck.get("backbone", "dinov2")
    m = beauty.BeautyRegressor(bb, unfreeze_top=int(ck["unfreeze_vision"])).to(device)
    m.load_state_dict(ck["state_dict"])
    mu, sd = float(ck["mu"]), float(ck["sd"])
    prior = beauty.score_paths(m, paths, device, mu, sd, bb, batch=64).astype(np.float64)
    prior_p = beauty.score_paths(m, ppaths, device, mu, sd, bb, batch=64).astype(np.float64)

    cache = resolve_path("data_beauty", "cache", "recsys_features.npz")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, E=E, prior=prior, y=y, Ep=Ep, prior_p=prior_p, pair_ids=np.array(ids))
    print(f"OK: {cache} | меток={len(y)} эмбеддинг={E.shape[1]}-d пар-лиц={len(ids)}")


if __name__ == "__main__":
    main()
