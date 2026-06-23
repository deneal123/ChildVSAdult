"""CLI: SOTA-objective baseline (E21) --- margin-softmax classification vs our pair-contrastive.

Trains facenet with ArcFace / CosFace / SphereFace identity classification on our mined
identities and compares on external benchmarks with frozen and with +pairs (our contrastive,
bb_<bb>_seed42.pt). One backbone, one trainable scope, one evaluation protocol --- so the only
thing that varies is the training objective. Question: does a classification SOTA objective beat
a simple contrastive on our sparse few-shot identities (2-3 photos)?

    uv run python scripts/sota_arcface.py --objectives arcface cosface sphereface
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, resolve_path
from age_gap.evaluation.external_suite import eval_all, print_table
from age_gap.models.backbones import make_backbone
from age_gap.training.arcface_train import train_arcface
from age_gap.training.finetune import load_finetuned


def main() -> None:
    parser = argparse.ArgumentParser(description="Margin-softmax SOTA objectives vs pairs")
    parser.add_argument("--backbone", default="facenet")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--scale", type=float, default=32.0)
    parser.add_argument("--objectives", nargs="+", default=["arcface", "cosface", "sphereface"])
    parser.add_argument("--skip-train", action="store_true", help="reuse bb_<bb>_<obj>.pt")
    args = parser.parse_args()

    device = torch_device()
    models = data_path("models_dir")
    bb = args.backbone

    results = {
        "frozen": eval_all(make_backbone(bb, pretrained=True).to(device).eval(), device),
        "+pairs": eval_all(load_finetuned(Path(str(models / f"bb_{bb}_seed42.pt")), device), device),
    }
    order = ["frozen", "+pairs"]
    for obj in args.objectives:
        ck = Path(str(models / f"bb_{bb}_{obj}.pt"))
        if not args.skip_train:
            ck = train_arcface(backbone_name=bb, loss_type=obj, epochs=args.epochs, scale=args.scale)
        results[f"+{obj}"] = eval_all(load_finetuned(ck, device), device)
        order.append(f"+{obj}")

    print_table(results, order)
    dst = resolve_path("docs", "sota_objectives.json")
    dst.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
