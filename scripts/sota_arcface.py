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
from age_gap.common.io import data_path
from age_gap.common.manifest import write_experiment_manifest
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
    parser.add_argument(
        "--manifest-only", action="store_true", help="bind existing metrics and checkpoints"
    )
    args = parser.parse_args()

    models = data_path("models_dir")
    bb = args.backbone
    dst = Path(str(data_path("metrics_dir", "sota_objectives.json")))
    checkpoints = [
        Path(str(models / f"bb_{bb}_seed42.pt")),
        *[Path(str(models / f"bb_{bb}_{obj}.pt")) for obj in args.objectives],
    ]

    if args.manifest_only:
        results = json.loads(dst.read_text(encoding="utf-8"))
        _write_manifest(dst, results, checkpoints, args)
        print(f"wrote {dst.with_suffix('.manifest.json')}")
        return

    device = torch_device()

    results = {
        "frozen": eval_all(make_backbone(bb, pretrained=True).to(device).eval(), device),
        "+pairs": eval_all(load_finetuned(Path(str(models / f"bb_{bb}_seed42.pt")), device), device),
    }
    order = ["frozen", "+pairs"]
    for obj, expected_checkpoint in zip(args.objectives, checkpoints[1:], strict=True):
        ck = expected_checkpoint
        if not args.skip_train:
            ck = train_arcface(backbone_name=bb, loss_type=obj, epochs=args.epochs, scale=args.scale)
        results[f"+{obj}"] = eval_all(load_finetuned(ck, device), device)
        order.append(f"+{obj}")

    print_table(results, order)
    dst.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_manifest(dst, results, checkpoints, args)
    print(f"wrote {dst}")


def _write_manifest(
    dst: Path,
    results: dict,
    checkpoints: list[Path],
    args: argparse.Namespace,
) -> None:
    external = Path(str(data_path("data_dir", "external")))
    write_experiment_manifest(
        dst.with_suffix(".manifest.json"),
        experiment="margin-softmax-objectives-vs-pair-contrastive",
        parameters={
            "backbone": args.backbone,
            "epochs": args.epochs,
            "scale": args.scale,
            "objectives": args.objectives,
            "reused_checkpoints": args.skip_train,
            "manifest_only": args.manifest_only,
        },
        metrics=results,
        inputs=[
            Path(str(data_path("metrics_dir", "model_inventory.json"))),
            Path(str(data_path("data_dir", "processed", "pairs.jsonl"))),
            *checkpoints,
            *[
                path
                for path in (
                    external / "lfw_aligned.npz",
                    external / "agedb_30.bin",
                    external / "calfw.bin",
                    external / "fgnet_crops.npz",
                    external / "cacd_vs_aligned.npz",
                )
                if path.is_file()
            ],
        ],
        outputs=[dst],
    )


if __name__ == "__main__":
    main()
