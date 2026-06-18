"""CLI: hard-negative mining по baseline-эмбеддингам (MVP-4).

Дозаписывает hard-негативы в pairs.jsonl. После запуска пересоберите сплит:
    uv run python scripts/mine_hard_negatives.py --top-k 5
    uv run python scripts/split.py
"""

from __future__ import annotations

import argparse

from age_gap.datasets.hard_negatives import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives from baseline embeddings")
    parser.add_argument("--top-k", type=int, default=5, help="Соседей-негативов на одно лицо")
    parser.add_argument(
        "--max", type=int, default=None, help="Лимит hard-негативов (оставляет самые сложные)"
    )
    args = parser.parse_args()

    added = run(top_k=args.top_k, max_total=args.max)
    print(f"Готово: добавлено hard-негативов={added}. Пересоберите сплит: scripts/split.py")


if __name__ == "__main__":
    main()
