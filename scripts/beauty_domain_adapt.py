"""Self-supervised доменная адаптация DINOv2 на VK-лицах (SimSiam), без меток.

    uv run python scripts/beauty_domain_adapt.py --epochs 3 --batch 32

Берёт hi-res кропы взрослых лиц ОБОИХ пабликов (data + data_natural), адаптирует верхние блоки
энкодера и сохраняет data_beauty/weights/dinov2_adapted.pt. Дальше:
    uv run python scripts/beauty_train.py --backbone dinov2 --init-encoder data_beauty/weights/dinov2_adapted.pt --save-full
"""

from __future__ import annotations

import argparse

import torch

from age_gap import domain_adapt as da
from age_gap.beauty import pick_device
from age_gap.common.io import resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", default=["data", "data_natural"], help="data-корни с faces_hires")
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--unfreeze-top", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--limit", type=int, default=0, help=">0: ограничить число лиц (быстрая проба)")
    args = ap.parse_args()

    device = pick_device()
    paths = da.face_crop_paths(args.dirs)
    if args.limit:
        paths = paths[:: max(1, len(paths) // args.limit)][: args.limit]
    log.info("Лиц для адаптации: %d (dirs=%s) device=%s", len(paths), args.dirs, device)

    out = da.adapt(paths, device, backbone=args.backbone, unfreeze_top=args.unfreeze_top,
                   epochs=args.epochs, batch=args.batch, lr=args.lr)

    wpath = resolve_path("data_beauty", "weights", "dinov2_adapted.pt")
    wpath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, wpath)
    log.info("Адаптированный энкодер сохранён: %s", wpath)
    print(f"OK: {wpath} (лиц={out['n_faces']}, epochs={out['epochs']})")


if __name__ == "__main__":
    main()
