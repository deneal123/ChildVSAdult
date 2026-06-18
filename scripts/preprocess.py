"""CLI: детекция, выравнивание и скоринг лиц.

Пример:
    uv run python scripts/preprocess.py
"""

from __future__ import annotations

import argparse

from age_gap.preprocessing.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect/align/score faces -> FaceCrop JSONL")
    parser.add_argument(
        "--posts", default=None, help="Путь к posts.jsonl (по умолчанию data/raw/posts.jsonl)"
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Устройство детекции: auto (по умолч.) | cpu (стабильно) | cuda",
    )
    args = parser.parse_args()

    new = run(posts_file=args.posts, device=args.device)
    print(f"Готово: обработано новых лиц={new} (запускайте повторно для догона остатка).")


if __name__ == "__main__":
    main()
