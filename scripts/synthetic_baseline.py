"""CLI: synthetic-ageing baseline vs обучение на РЕАЛЬНЫХ парах (docs/deep-research.md).

Сравнивает на внешних cross-age бенчмарках:
    frozen | +real pairs | +synthetic(proxy) | +synthetic(FRAN)

Отвечает на вопрос статьи: реальные longitudinal пары полезнее синтетического старения?
``--aging`` выбирает источник синтетики: ``proxy`` (доменная пертурбация, без весов),
``fran`` (реальная aging-модель FRAN, веса с HuggingFace), ``both`` (обе строки в таблице).

    uv run python scripts/synthetic_baseline.py --epochs 10 --aging both
"""

from __future__ import annotations

import argparse
from pathlib import Path

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.evaluation.external_suite import eval_all, print_table
from age_gap.models.aging import make_aging
from age_gap.models.facenet import FaceNetBackbone
from age_gap.training.finetune import finetune, load_finetuned
from age_gap.training.synthetic import train_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic-ageing vs real-pairs baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--aging", choices=["proxy", "fran", "both"], default="both")
    args = parser.parse_args()

    device = torch_device()
    models = data_path("models_dir")

    real_ckpt = finetune(
        epochs=args.epochs,
        lr=args.lr,
        trainable_scope="head",
        gap_weight=0.0,
        ckpt_out=Path(str(models / "abl_pairs.pt")),
    )

    variants = [
        ("frozen", FaceNetBackbone(pretrained="casia-webface").to(device).eval()),
        ("+real_pairs", load_finetuned(real_ckpt, device)),
    ]

    kinds = ["proxy", "fran"] if args.aging == "both" else [args.aging]
    labels = {"proxy": "+syn_proxy", "fran": "+syn_fran"}
    ckpts = {"proxy": "facenet_synthetic.pt", "fran": "facenet_syn_fran.pt"}
    for kind in kinds:
        ckpt = train_synthetic(
            epochs=args.epochs,
            lr=args.lr,
            trainable_scope="head",
            aging=make_aging(kind),
            ckpt_out=Path(str(models / ckpts[kind])),
        )
        variants.append((labels[kind], load_finetuned(ckpt, device)))

    results = {label: eval_all(b, device) for label, b in variants}
    print_table(results, [v[0] for v in variants])


if __name__ == "__main__":
    main()
