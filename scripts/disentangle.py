"""CLI: age-disentanglement vs обычное дообучение (Фаза 9 / §5.2).

Сравнивает frozen | +pairs | +pairs+disentangle на одном leakage-safe сплите по внешним
бенчмаркам (FG-NET large-gap — ключевой). Отвечает: помогает ли явное удаление возраста сверх пар.

    uv run python scripts/disentangle.py --epochs 10 --lambda-age 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.evaluation.external_suite import eval_all, print_table
from age_gap.models.backbones import make_backbone
from age_gap.training.disentangle import train_disentangle
from age_gap.training.finetune import finetune, load_finetuned


def main() -> None:
    parser = argparse.ArgumentParser(description="Age-disentanglement vs plain fine-tune")
    parser.add_argument("--backbone", default="facenet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lambda-age", type=float, default=0.3)
    parser.add_argument(
        "--skip-pairs", action="store_true", help="не переобучать +pairs, взять готовый bb_<bb>_pairs.pt"
    )
    args = parser.parse_args()

    device = torch_device()
    models = data_path("models_dir")
    bb = args.backbone

    frozen = make_backbone(bb, pretrained=True).to(device).eval()
    ck_pairs = Path(str(models / f"bb_{bb}_pairs.pt"))
    if not args.skip_pairs:
        ck_pairs = finetune(
            epochs=args.epochs,
            lr=args.lr,
            trainable_scope="head",
            backbone_name=bb,
            ckpt_out=ck_pairs,
        )
    ck_dis = train_disentangle(
        backbone_name=bb,
        epochs=args.epochs,
        lr=args.lr,
        lambda_age=args.lambda_age,
        ckpt_out=Path(str(models / f"bb_{bb}_disentangle.pt")),
    )

    variants = [
        (f"{bb}:frozen", frozen),
        (f"{bb}:+pairs", load_finetuned(ck_pairs, device)),
        (f"{bb}:+pairs+disent", load_finetuned(ck_dis, device)),
    ]
    results = {label: eval_all(b, device) for label, b in variants}  # type: ignore[arg-type]
    print_table(results, [v[0] for v in variants])


if __name__ == "__main__":
    main()
