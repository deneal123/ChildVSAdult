"""Оценить ЛЮБУЮ beauty-модель на человеческой парной разметке (тест на VK-домене).

    uv run python scripts/beauty_eval_pairs.py --weights data_beauty/weights/beauty_dinov2.pt
    uv run python scripts/beauty_eval_pairs.py --weights data_beauty/weights/beauty_dinov2_adapted.pt

Главная метрика — pairwise accuracy: доля пар, где порядок модели совпал с человеческим выбором
(0.5 = случайно). Пары — лица then/now (hi-res кропы). Пишет metrics/beauty_eval_<weights>.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from age_gap import beauty
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--pairs", default="reports/rating/pairs.jsonl")
    args = ap.parse_args()

    device = beauty.pick_device()
    wpath = resolve_path(args.weights)
    ckpt = torch.load(wpath, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "clip")
    model = beauty.BeautyRegressor(backbone, unfreeze_top=int(ckpt["unfreeze_vision"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    mu, sd = float(ckpt.get("mu", 0.0)), float(ckpt.get("sd", 1.0))
    log.info("Веса: %s (backbone=%s)", wpath.name, backbone)

    rows = [r for r in read_jsonl(resolve_path(args.pairs)) if r.get("winner")]
    ids = sorted({r["a"] for r in rows} | {r["b"] for r in rows})
    hd = resolve_path("data", "interim", "faces_hires")  # пары — из then/now
    paths = [hd / f"{i}.jpg" for i in ids]
    sc = beauty.score_paths(model, [p if p.exists() else None for p in paths], device, mu, sd, backbone, batch=64)
    smap = dict(zip(ids, sc, strict=True))

    used = [r for r in rows if np.isfinite(smap[r["a"]]) and np.isfinite(smap[r["b"]])
            and smap[r["a"]] != smap[r["b"]]]
    acc = float(np.mean([1.0 if (smap[r["a"]] > smap[r["b"]]) == (r["winner"] == r["a"]) else 0.0 for r in used]))

    out = {"weights": wpath.name, "backbone": backbone, "n_decisive_pairs": len(rows),
           "n_pairs_used": len(used), "pairwise_accuracy": round(acc, 4),
           "beauty_range": [round(float(np.nanmin(sc)), 2), round(float(np.nanmax(sc)), 2),
                            round(float(np.nanmean(sc)), 2)],
           "note": "0.5 = случайно; baseline (SCUT DINOv2, без адаптации) = 0.688"}
    dst = data_path("metrics_dir", f"beauty_eval_{wpath.stem}.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
