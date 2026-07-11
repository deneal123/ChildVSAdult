"""Обучение предиктора красоты лица на РЕАЛЬНЫХ человеческих оценках (SCUT-FBP5500).

    uv run python scripts/beauty_train.py                       # CLIP ft4, 5-fold
    uv run python scripts/beauty_train.py --unfreeze-vision 0   # frozen CLIP + голова (baseline)
    uv run python scripts/beauty_train.py --save-full           # + дообучить на всех данных и сохранить веса

Метрика — Pearson/Spearman vs человеческие оценки (осмысленный потолок, в отличие от VK-прокси).
Пишет metrics/beauty_scut.json. Веса/кеши — под data_beauty/ (в .gitignore).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from age_gap import beauty
from age_gap.common.io import data_path, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--unfreeze-vision", type=int, default=4, help="сколько верхних блоков CLIP дообучать (0=frozen)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--save-full", action="store_true", help="дообучить на всех данных и сохранить веса для VK")
    ap.add_argument("--quick", action="store_true", help="2 фолда, 3 эпохи")
    args = ap.parse_args()

    device = beauty.pick_device()
    ds, y, meta = beauty.load_scut()
    px = beauty.precompute_pixels(ds)
    log.info("device=%s | n=%d unfreeze=%d", device, len(y), args.unfreeze_vision)

    res = beauty.kfold_eval(
        px, y, device, n_splits=2 if args.quick else args.folds,
        unfreeze_top=args.unfreeze_vision, epochs=3 if args.quick else args.epochs,
        batch=args.batch, lr=args.lr, lr_backbone=args.lr_backbone,
    )
    res.pop("oof")

    out = {
        "dataset": beauty.SCUT_DATASET, "clip_model": beauty.CLIP_MODEL,
        "n": int(len(y)), "unfreeze_vision": int(args.unfreeze_vision),
        "epochs": int(3 if args.quick else args.epochs), "device": str(device),
        "per_fold": res["per_fold"], "cv_mean_std": res["mean_std"], "oof_overall": res["oof_overall"],
    }
    dst = data_path("metrics_dir", "beauty_scut.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    if args.save_full:
        log.info("Дообучение на ВСЕХ данных для применения к VK...")
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        n_val = max(1, int(0.1 * len(y)))
        va, core = idx[:n_val], idx[n_val:]
        mu, sd = float(y[core].mean()), float(y[core].std()) + 1e-8
        model = beauty.CLIPBeauty(unfreeze_top=args.unfreeze_vision).to(device)
        dl_tr = beauty._loader(px[core], (y[core] - mu) / sd, args.batch, True)
        dl_va = beauty._loader(px[va], (y[va] - mu) / sd, args.batch, False)
        beauty._train(model, dl_tr, dl_va, (y[va] - mu) / sd, device,
                      3 if args.quick else args.epochs, args.lr, args.lr_backbone, 0.05)
        wpath = resolve_path("data_beauty", "weights", "clip_beauty.pt")
        wpath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                    "unfreeze_vision": args.unfreeze_vision}, wpath)
        log.info("Веса сохранены: %s", wpath)

    print(json.dumps({
        "CV_pearson": round(out["cv_mean_std"]["pearson"]["mean"], 4),
        "CV_spearman": round(out["cv_mean_std"]["spearman"]["mean"], 4),
        "CV_mae": round(out["cv_mean_std"]["mae"]["mean"], 4),
        "oof_pearson": round(out["oof_overall"]["pearson"], 4),
        "metrics": str(dst),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
