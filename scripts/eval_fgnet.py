"""CLI: внешний cross-age бенчмарк FG-NET — frozen vs fine-tuned facenet.

Чистая атрибуция: FG-NET — ДРУГОЙ домен, чем VK, поэтому прирост здесь изолирует
«обучение кросс-возрасту» от адаптации к VK-домену.

    uv run python scripts/eval_fgnet.py
"""

from __future__ import annotations

import argparse

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.evaluation.fgnet import evaluate, prepare_crops
from age_gap.models.facenet import FaceNetBackbone
from age_gap.training.finetune import load_finetuned


def main() -> None:
    parser = argparse.ArgumentParser(description="FG-NET frozen vs fine-tuned facenet")
    parser.add_argument("--ckpt", default=None, help="Чекпойнт (по умолч. facenet_finetuned.pt)")
    args = parser.parse_args()

    device = torch_device()
    prepare_crops()  # детекция+выравнивание (кэшируется)

    frozen = evaluate(FaceNetBackbone(pretrained="casia-webface").to(device).eval(), device)
    ckpt = args.ckpt or data_path("models_dir", "facenet_finetuned.pt")
    after = evaluate(load_finetuned(ckpt, device), device)

    print(f"\nFG-NET (внешний cross-age){'':<4}{'frozen':>10}{'finetuned':>11}{'Δ':>9}")
    for metric in ("accuracy_10fold", "roc_auc", "large_gap_auc", "tar@far=0.01"):
        b, a = frozen[metric], after[metric]
        print(f"{metric:<28}{b:>10.4f}{a:>11.4f}{a - b:>+9.4f}")


if __name__ == "__main__":
    main()
