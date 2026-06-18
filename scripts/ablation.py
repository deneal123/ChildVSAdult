"""CLI: сводная ablation-таблица (требование paper-протокола, docs/deep-research.md).

Один слабый бэкбон (facenet casia-webface), варианты обучения на НАШИХ парах, оценка на
внешних cross-age бенчмарках (LFW, AgeDB-30, CALFW, FG-NET) и нашем held-out test:

    frozen | +natural pairs | +pairs+age-anchor-reg

Отвечает на главный вопрос статьи: что даёт выигрыш — сами реальные пары или возрастные якоря.
(Строка «+comments apparent-age» недоступна: сервисный VK-токен блокирует wall.getComments.)

    uv run python scripts/ablation.py --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.evaluation.external_suite import eval_all, print_table
from age_gap.models.facenet import FaceNetBackbone
from age_gap.training.finetune import finetune, load_finetuned


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: frozen vs +pairs vs +pairs+age-reg")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    device = torch_device()
    models = data_path("models_dir")

    # Обучаем варианты (мягко: head-only). Разница только в age-anchor regularization.
    ck_pairs = finetune(
        epochs=args.epochs,
        lr=args.lr,
        trainable_scope="head",
        gap_weight=0.0,
        ckpt_out=Path(str(models / "abl_pairs.pt")),
    )
    ck_agereg = finetune(
        epochs=args.epochs,
        lr=args.lr,
        trainable_scope="head",
        gap_weight=2.0,
        ckpt_out=Path(str(models / "abl_pairs_agereg.pt")),
    )

    variants = [
        ("frozen", FaceNetBackbone(pretrained="casia-webface").to(device).eval()),
        ("+pairs", load_finetuned(ck_pairs, device)),
        ("+pairs+agereg", load_finetuned(ck_agereg, device)),
    ]
    results = {label: eval_all(b, device) for label, b in variants}
    print_table(results, [v[0] for v in variants])


if __name__ == "__main__":
    main()
