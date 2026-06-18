"""CLI: честный протокол (frozen → дообучение на НАШИХ парах → внешние бенчмарки) для
НЕСКОЛЬКИХ backbone разной силы — для строгой публикации.

Строит для каждого backbone строки `frozen` и `+real_pairs` и печатает таблицу. Цель —
показать, как выигрыш от наших same-post пар зависит от силы backbone (слабые facenet /
ArcFace-CASIA vs сильный ArcFace-r100).

    uv run python scripts/backbones_benchmark.py --epochs 10 --lr 3e-5
    uv run python scripts/backbones_benchmark.py --backbones facenet arcface_r50_casia
"""

from __future__ import annotations

import argparse
from pathlib import Path

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.evaluation.external_suite import eval_all, print_table
from age_gap.models.backbones import BACKBONES, make_backbone
from age_gap.training.finetune import finetune, load_finetuned


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-backbone honest-protocol benchmark")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--scope", default="head", choices=["head", "tail", "full"])
    parser.add_argument("--crops", default="faces", help="подкаталог кропов (faces/faces_mtcnn)")
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="не переобучать: оценить уже сохранённые bb_<name>_pairs.pt (возобновление после сбоя)",
    )
    args = parser.parse_args()

    device = torch_device()
    models = data_path("models_dir")

    variants: list[tuple[str, object]] = []
    for name in args.backbones:
        ckpt_path = Path(str(models / f"bb_{name}_pairs.pt"))
        if args.eval_only and not ckpt_path.exists():
            print(f"[skip] {name}: нет чекпойнта {ckpt_path}")
            continue
        frozen = make_backbone(name, pretrained=True).to(device).eval()
        # Оценка нашего сплита идёт на том же наборе кропов, что и обучение (--crops),
        # чтобы ablation выравнивания (faces vs faces_mtcnn) был честным и для frozen.
        frozen.crops_dir = args.crops  # type: ignore[assignment]
        variants.append((f"{name}:frozen", frozen))
        if args.eval_only:
            ckpt = ckpt_path  # возобновление: используем готовый чекпойнт без дообучения
        else:
            ckpt = finetune(
                epochs=args.epochs,
                lr=args.lr,
                trainable_scope=args.scope,
                gap_weight=0.0,
                backbone_name=name,
                crops_dir=args.crops,
                ckpt_out=ckpt_path,
            )
        variants.append((f"{name}:+pairs", load_finetuned(ckpt, device)))

    results = {label: eval_all(b, device) for label, b in variants}  # type: ignore[arg-type]
    print_table(results, [v[0] for v in variants])


if __name__ == "__main__":
    main()
