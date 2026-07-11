"""Triplet/contrastive метрик-лернинг на лицах по таргету лайк/просмотр (research-эксперимент).

    ENV_FOR_DYNACONF=natural uv run python scripts/contrastive_train.py
    ENV_FOR_DYNACONF=natural uv run python scripts/contrastive_train.py --shuffle   # негативный контроль

Учит визуальный энкодер (CLIP, только лица, без текста) разносить эмбеддинги по нормализованному
лайк/просмотр; оценка — per-post Spearman vs e_rate с GroupKFold(person_id). Пишет metrics/contrastive.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from age_gap import beauty, contrastive
from age_gap.common.io import data_path, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def _pixels(df, batch: int = 256) -> np.ndarray:
    cache = resolve_path("data_beauty", "cache", f"contrastive_pixels_{data_path('data_dir').name}.npz")
    ids = df["face_id"].to_numpy()
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["face_ids"]) == list(ids):
            log.info("face pixel cache HIT: %s", cache.name)
            return z["pixels"]
    px = beauty.clip_pixels_from_crops(df["crop_path"].tolist(), batch=batch)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, pixels=px, face_ids=ids)
    log.info("face pixel cache: %d -> %s", len(ids), cache.name)
    return px


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="vk")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--unfreeze-vision", type=int, default=2)
    ap.add_argument("--shuffle", action="store_true", help="негативный контроль (перемешать таргет)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    device = beauty.pick_device()
    df = contrastive.build_face_table(args.platform)
    px = _pixels(df)

    res = contrastive.run_cv(
        df, px, device, n_splits=2 if args.quick else args.folds,
        unfreeze_top=args.unfreeze_vision, epochs=3 if args.quick else args.epochs,
        batch=args.batch, shuffle_target=args.shuffle)

    out = {
        "platform": args.platform, "n_faces": int(len(df)), "n_posts": int(df["post_id"].nunique()),
        "n_persons": int(df["person_id"].nunique()), "device": str(device),
        "unfreeze_vision": int(args.unfreeze_vision), "epochs": int(3 if args.quick else args.epochs),
        "target": "z(log like/view), per-face from post", **res,
    }
    name = "contrastive_shuffled.json" if args.shuffle else "contrastive.json"
    dst = data_path("metrics_dir", name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print(json.dumps({
        "mean_spearman_post_e_rate": round(out["mean_spearman"], 4),
        "std": round(out["std_spearman"], 4),
        "shuffled": bool(args.shuffle),
        "per_fold": [round(f["spearman_post_e_rate"], 3) for f in res["per_fold"]],
        "metrics": str(dst),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
