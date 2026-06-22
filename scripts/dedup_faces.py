"""CLI: дедуп near-duplicate кадров внутри личности (Part 3) — детекция тривиальных позитивов.

Находит кадры с косинусом >= threshold внутри каждой группы (почти дубликаты), пишет плоский
список избыточных лиц (`redundant_faces.jsonl`) и печатает статистику. Этот список затем исключается
из позитивов при пересборке пар.

    uv run python scripts/dedup_faces.py --threshold 0.97
"""

from __future__ import annotations

import argparse

from age_gap.datasets.dedup import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Near-duplicate face dedup within identities")
    parser.add_argument("--threshold", type=float, default=0.97)
    args = parser.parse_args()
    redundant = run(threshold=args.threshold)
    flat = sum(len(v) for v in redundant.values())
    print(f"\nГрупп с near-dup кадрами: {len(redundant)}; избыточных лиц: {flat}")


if __name__ == "__main__":
    main()
