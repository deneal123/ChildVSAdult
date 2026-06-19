"""CLI: leakage-safe сплит пар по identity_group_id.

Пример:
    uv run python scripts/split.py
"""

from __future__ import annotations

import argparse

from age_gap.datasets.splits import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe split by identity_group_id")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--neg-per-pos", type=float, default=1.0, help="баланс негативов на позитив в каждом сплите"
    )
    parser.add_argument(
        "--age-matched-neg",
        action="store_true",
        help="негативы из одного возрастного бакета (разные люди) — §5.1, identity-only",
    )
    parser.add_argument(
        "--by-wall",
        choices=["A", "B"],
        default=None,
        help="cross-wall: указанная стена → train, другая → test (§5.5 генерализация)",
    )
    args = parser.parse_args()

    group_split = run(
        ratios=(args.train, args.val, args.test),
        seed=args.seed,
        neg_per_pos=args.neg_per_pos,
        age_matched_neg=args.age_matched_neg,
        by_wall=args.by_wall,
    )
    print(f"Готово: назначен сплит для {len(group_split)} групп.")


if __name__ == "__main__":
    main()
