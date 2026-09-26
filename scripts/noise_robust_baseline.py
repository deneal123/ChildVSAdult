"""CLI: noise-robust comparator --- sub-center ArcFace vs our contrastive and vanilla ArcFace.

Sub-center ArcFace (Deng et al., ECCV 2020) keeps K centroids per identity and max-pools them at
forward time, so label-noisy / low-quality crops are routed to off-centers instead of corrupting
the dominant centroid. It is the canonical baseline designed for *noisy web face labels* --- exactly
our mined-supervision regime. We train it on the SAME mined identities, the same weak backbone, the
same trainable scope and the same evaluation as ``+pairs`` (our contrastive) and ``+arcface`` (vanilla
margin), so the only thing that varies is explicit label-noise handling. Question: does a noise-robust
objective change the large-gap conclusion, or does the data still dominate?

    uv run python scripts/noise_robust_baseline.py --sub-centers 3
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
    parser = argparse.ArgumentParser(description="Sub-center ArcFace noise-robust comparator")
    parser.add_argument("--backbone", default="facenet")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--scale", type=float, default=32.0)
    parser.add_argument("--sub-centers", type=int, default=3)
    parser.add_argument("--skip-train", action="store_true", help="reuse bb_<bb>_arcface_sc<k>.pt")
    parser.add_argument(
        "--manifest-only", action="store_true", help="bind existing metrics and checkpoints"
    )
    args = parser.parse_args()

    device = torch_device()
    models = data_path("models_dir")
    bb = args.backbone
    k = args.sub_centers
    ck = Path(str(models / f"bb_{bb}_arcface_sc{k}.pt"))
    dst = Path(str(data_path("metrics_dir", "noise_robust_baseline.json")))
    if args.manifest_only:
        results = json.loads(dst.read_text(encoding="utf-8"))
        _write_manifest(dst, results, models, bb, ck, args)
        print(f"wrote {dst.with_suffix('.manifest.json')}")
        return

    # Comparators reuse already-trained checkpoints; only the sub-center model is (re)trained.
    results = {
        "frozen": eval_all(make_backbone(bb, pretrained=True).to(device).eval(), device),
        "+pairs": eval_all(load_finetuned(Path(str(models / f"bb_{bb}_seed42.pt")), device), device),
        "+arcface": eval_all(load_finetuned(Path(str(models / f"bb_{bb}_arcface.pt")), device), device),
    }
    order = ["frozen", "+pairs", "+arcface"]

    if not args.skip_train:
        ck = train_arcface(
            backbone_name=bb, loss_type="arcface", epochs=args.epochs, scale=args.scale,
            sub_centers=k,
        )
    tag = f"+arcface_sc{k}"
    results[tag] = eval_all(load_finetuned(ck, device), device)
    order.append(tag)

    print_table(results, order)
    dst.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_manifest(dst, results, models, bb, ck, args)
    print(f"wrote {dst}")


def _write_manifest(
    dst: Path,
    results: dict,
    models: Path,
    backbone: str,
    checkpoint: Path,
    args: argparse.Namespace,
) -> None:
    pairs_file = Path(str(data_path("data_dir", "processed", "pairs.jsonl")))
    pair_ckpt = Path(str(models / f"bb_{backbone}_seed42.pt"))
    arcface_ckpt = Path(str(models / f"bb_{backbone}_arcface.pt"))
    write_experiment_manifest(
        dst.with_suffix(".manifest.json"),
        experiment="subcenter-arcface-noise-robust-comparator",
        parameters={
            "backbone": backbone,
            "epochs": args.epochs,
            "scale": args.scale,
            "sub_centers": args.sub_centers,
            "skip_train": args.skip_train,
            "manifest_only": args.manifest_only,
        },
        metrics=results,
        inputs=[pairs_file, pair_ckpt, arcface_ckpt, checkpoint],
        outputs=[dst],
    )


if __name__ == "__main__":
    main()
