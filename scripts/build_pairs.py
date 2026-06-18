"""CLI: генерация пар из готовых групп личностей.

Сначала постройте группы: scripts/build_groups.py (резюмируемо, с возрастом).

    uv run python scripts/build_pairs.py --neg-per-pos 1
"""

from __future__ import annotations

import argparse

from age_gap.datasets.pair_builder import build_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build positive/negative pairs from groups")
    parser.add_argument("--neg-per-pos", type=int, default=1, help="Негативов на один позитив")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = build_pairs(n_negatives_per_positive=args.neg_per_pos, seed=args.seed)
    pos = sum(p.label == 1 for p in pairs)
    print(f"Готово: пар={len(pairs)} (pos={pos}, neg={len(pairs) - pos}).")


if __name__ == "__main__":
    main()
