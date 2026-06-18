"""CLI: baseline-бенчмарк верификации по ArcFace-эмбеддингам.

Пример:
    uv run python scripts/benchmark.py            # все пары
    uv run python scripts/benchmark.py --split test
"""

from __future__ import annotations

import argparse

from age_gap.evaluation.benchmark import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline verification benchmark")
    parser.add_argument(
        "--split", default=None, help="Оценить только указанный сплит (train/val/test)"
    )
    parser.add_argument("--embeddings", default=None, help="Путь к .npz с эмбеддингами")
    args = parser.parse_args()

    results = run(split=args.split, embeddings_file=args.embeddings)
    if not results:
        print("Нет данных для оценки.")
        return
    print(f"{'group':<16}{'n_pos':>7}{'n_neg':>7}{'ROC-AUC':>10}{'EER':>8}")
    for name, m in results.items():
        print(
            f"{name:<16}{int(m['n_pos']):>7}{int(m['n_neg']):>7}"
            f"{m['roc_auc']:>10.4f}{m['eer']:>8.4f}"
        )


if __name__ == "__main__":
    main()
