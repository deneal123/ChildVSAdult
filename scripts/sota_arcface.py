"""CLI: SOTA-objective baseline (E21) — ArcFace-классификация vs наш pair-contrastive.

Обучает facenet ArcFace-margin классификацией по личности на наших данных и сравнивает на внешних
бенчмарках с frozen и с +pairs (наш контрастив, bb_<bb>_seed42.pt). Один backbone, один протокол.

    uv run python scripts/sota_arcface.py --epochs 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.evaluation.external_suite import eval_all, print_table
from age_gap.models.backbones import make_backbone
from age_gap.training.arcface_train import train_arcface
from age_gap.training.finetune import load_finetuned


def main() -> None:
    parser = argparse.ArgumentParser(description="ArcFace-classification SOTA baseline vs pairs")
    parser.add_argument("--backbone", default="facenet")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=32.0)
    parser.add_argument(
        "--skip-train", action="store_true", help="взять готовый bb_<bb>_arcface.pt"
    )
    args = parser.parse_args()

    device = torch_device()
    models = data_path("models_dir")
    bb = args.backbone

    ck_arc = Path(str(models / f"bb_{bb}_arcface.pt"))
    if not args.skip_train:
        ck_arc = train_arcface(
            backbone_name=bb, epochs=args.epochs, margin=args.margin, scale=args.scale
        )

    results = {
        "frozen": eval_all(make_backbone(bb, pretrained=True).to(device).eval(), device),
        "+pairs": eval_all(
            load_finetuned(Path(str(models / f"bb_{bb}_seed42.pt")), device), device
        ),
        "+arcface": eval_all(load_finetuned(ck_arc, device), device),
    }
    print_table(results, ["frozen", "+pairs", "+arcface"])


if __name__ == "__main__":
    main()
