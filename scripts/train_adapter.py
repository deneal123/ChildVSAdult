"""CLI: обучить кросс-возрастной adapter и сравнить с baseline.

Поток: train -> apply -> benchmark(baseline) vs benchmark(adapter).
Требует ml-зависимости: uv sync --extra ml

Пример:
    uv run python scripts/train_adapter.py --epochs 50
"""

from __future__ import annotations

import argparse

from age_gap.common.io import data_path
from age_gap.evaluation.benchmark import run as benchmark
from age_gap.training.train import apply_adapter, train_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cross-age adapter and compare to baseline")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--gap-weight", type=float, default=2.0, help="Вес больших возрастных разрывов в лоссе"
    )
    parser.add_argument("--large-gap-threshold", type=int, default=15)
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument(
        "--eval-split", default="test", help="Сплит для честной оценки (по умолч. test)"
    )
    args = parser.parse_args()

    train_adapter(
        epochs=args.epochs,
        lr=args.lr,
        margin=args.margin,
        batch_size=args.batch_size,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        patience=args.patience,
        gap_weight=args.gap_weight,
        large_gap_threshold=args.large_gap_threshold,
        residual=not args.no_residual,
    )
    adapter_emb = apply_adapter()

    split = args.eval_split
    base = benchmark(split=split, metrics_out=data_path("metrics_dir", "baseline_arcface.json"))
    adapt = benchmark(
        split=split,
        embeddings_file=str(adapter_emb),
        metrics_out=data_path("metrics_dir", "adapter_arcface.json"),
    )

    # Сравнение на held-out сплите: overall + ключевой бакет больших разрывов 25+.
    print(f"\nОценка на split='{split}':")
    print(f"{'group':<16}{'model':<10}{'ROC-AUC':>10}{'EER':>8}{'TAR@FAR=0.01':>14}")
    for group in ("overall", "age_gap:25-200"):
        for name, res in (("baseline", base), ("adapter", adapt)):
            m = res.get(group)
            if not m:
                continue
            print(
                f"{group:<16}{name:<10}{m['roc_auc']:>10.4f}{m['eer']:>8.4f}{m['tar@far=0.01']:>14.4f}"
            )


if __name__ == "__main__":
    main()
