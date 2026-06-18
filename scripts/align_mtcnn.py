"""CLI: построить MTCNN-выровненные кропы (faces_mtcnn) для facenet.

    uv run python scripts/align_mtcnn.py            # CPU (по умолчанию, не мешает GPU-обучению)
    uv run python scripts/align_mtcnn.py --device cuda
"""

from __future__ import annotations

import argparse

from age_gap.preprocessing.mtcnn_align import build_mtcnn_crops


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MTCNN-aligned crops for facenet")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--margin", type=int, default=0)
    args = parser.parse_args()
    n = build_mtcnn_crops(image_size=args.image_size, margin=args.margin, device=args.device)
    print(f"MTCNN crops built: {n}")


if __name__ == "__main__":
    main()
