"""Прогнать SCUT-FBP5500 через НАШ face-пайплайн (detect -> norm_crop 112) и закешировать.

Отдельный процесс: тут работает InsightFace (onnxruntime-GPU); обучение CLIP (torch-CUDA)
запускается уже другим процессом, читающим кеш, — так избегаем конфликта двух CUDA-рантаймов.

    uv run python scripts/beauty_align_scut.py
"""

from __future__ import annotations

import argparse

from age_gap import beauty
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda", help="cuda|cpu для детектора")
    args = ap.parse_args()
    ds, _, _ = beauty.load_scut()
    px = beauty.precompute_pixels_aligned(ds, device_str=args.device)
    log.info("Готово: aligned pixel cache, shape=%s", px.shape)


if __name__ == "__main__":
    main()
