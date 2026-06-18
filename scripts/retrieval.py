"""CLI: retrieval-оценка (Rank-1 / Recall@K / MRR / median rank).

uv run python scripts/retrieval.py                       # baseline-эмбеддинги, test
uv run python scripts/retrieval.py --embeddings cache/embeddings/adapter_arcface.npz
"""

from __future__ import annotations

import argparse

from age_gap.evaluation.retrieval import evaluate_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-age retrieval metrics")
    parser.add_argument("--embeddings", default=None, help="Путь к .npz (по умолч. baseline)")
    parser.add_argument("--split", default="test", help="Сплит (test/val/train/all)")
    args = parser.parse_args()

    split = None if args.split == "all" else args.split
    res = evaluate_retrieval(embeddings_file=args.embeddings, split=split)
    if res.get("n_queries", 0) == 0:
        print("Нет данных для retrieval.")
        return
    print(f"queries={int(res['n_queries'])}")
    for k in ("rank1", "recall@1", "recall@5", "recall@10", "mrr", "median_rank"):
        if k in res:
            print(f"  {k:<12}{res[k]:.4f}")


if __name__ == "__main__":
    main()
