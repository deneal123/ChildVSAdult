"""CLI: закон масштабирования — прирост от пар vs доля обучающих личностей (Q1).

Для каждой доли train-личностей: пересплит (val/test фиксированы), дообучение facenet, оценка
FG-NET large-gap и our.25+. Показывает, сколько данных нужно и выходит ли кривая на плато.
В конце восстанавливает полный сплит.

    uv run python scripts/scaling_law.py --fracs 0.1 0.25 0.5 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.datasets.splits import run as split_run
from age_gap.evaluation.external_suite import eval_all
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import finetune, load_finetuned

_KEYS = ("fgnet.large_gap", "fgnet.roc", "our.25+", "our.overall")


def main() -> None:
    parser = argparse.ArgumentParser(description="Data scaling law (gain vs #train identities)")
    parser.add_argument("--backbone", default="facenet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch_device()
    models = data_path("models_dir")
    bb = args.backbone

    frozen = eval_all(make_backbone(bb, pretrained=True).to(device).eval(), device)
    print("frozen baseline: " + " ".join(f"{k}={frozen.get(k, float('nan')):.4f}" for k in _KEYS), flush=True)
    rows: list[tuple[float, dict[str, float]]] = []
    for f in args.fracs:
        split_run(neg_per_pos=1.0, seed=args.seed, train_frac=f)
        ckpt = finetune(
            epochs=args.epochs,
            lr=args.lr,
            trainable_scope="head",
            backbone_name=bb,
            seed=args.seed,
            ckpt_out=Path(str(models / f"bb_{bb}_scale.pt")),
        )
        m = eval_all(load_finetuned(ckpt, device), device)
        rows.append((f, m))
        # Инкрементальный вывод: переживает обрыв (не терять уже посчитанные доли).
        print(
            f"[scaling frac={f:.2f}] " + " ".join(f"{k}={m.get(k, float('nan')):.4f}" for k in _KEYS),
            flush=True,
        )

    split_run(neg_per_pos=1.0, seed=args.seed)  # восстановить полный сплит

    print(f"\n{bb}: scaling law (frozen baseline + дообучение на доле train-личностей)")
    print(f"{'frac':>6}" + "".join(f"{k:>18}" for k in _KEYS))
    print(f"{'frozen':>6}" + "".join(f"{frozen.get(k, float('nan')):>18.4f}" for k in _KEYS))
    for f, m in rows:
        print(f"{f:>6.2f}" + "".join(f"{m.get(k, float('nan')):>18.4f}" for k in _KEYS))


if __name__ == "__main__":
    main()
