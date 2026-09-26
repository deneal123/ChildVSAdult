"""Evaluate a frozen or fine-tuned backbone on canonical CACD-VS with a manifest."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.common.manifest import write_experiment_manifest
from age_gap.evaluation.benchmark_external import evaluate_cacd_vs
from age_gap.models.backbones import BACKBONES, make_backbone
from age_gap.training.finetune import load_finetuned


def _evaluate_one(
    *, backbone: str, checkpoint: Path | None, label: str, device: str, cache: Path
) -> tuple[Path, dict]:
    actual_backbone = backbone
    if checkpoint is not None:
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
        actual_backbone = str(metadata.get("backbone", "facenet"))
    model = (
        load_finetuned(checkpoint, device)
        if checkpoint is not None
        else make_backbone(backbone, pretrained=True).to(device).eval()
    )
    metrics = evaluate_cacd_vs(model, device, cache)
    output = Path(str(data_path("metrics_dir", "cacd_vs", f"{label}.json")))
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backbone": actual_backbone,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "protocol": "official identity-disjoint 10 folds",
        "metrics": metrics,
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    inputs: list[Path] = [cache]
    if checkpoint:
        inputs.append(checkpoint)
    write_experiment_manifest(
        output.with_suffix(".manifest.json"),
        experiment="cacd-vs-verification",
        parameters={
            "backbone": actual_backbone,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "protocol": "official identity-disjoint 10 folds",
        },
        metrics=metrics,
        inputs=inputs,
        outputs=[output],
    )
    print(json.dumps(payload, indent=2))
    print(f"OK: {output}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output, metrics


def _aggregate(rows: dict[str, dict]) -> dict[str, dict[str, float | int]]:
    common = set.intersection(*(set(row) for row in rows.values())) if rows else set()
    out: dict[str, dict[str, float | int]] = {}
    for metric in sorted(common):
        if metric.endswith("_ci95") or metric == "n_pairs":
            continue
        values = np.asarray([row[metric] for row in rows.values()], dtype=float)
        out[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n_seeds": len(values),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=list(BACKBONES), default="adaface_ir101")
    parser.add_argument("--checkpoint", type=Path, help="one checkpoint (legacy/single-run mode)")
    parser.add_argument("--checkpoints", nargs="+", type=Path, help="multi-seed checkpoints")
    parser.add_argument("--label", help="artifact label; defaults to frozen or checkpoint stem")
    args = parser.parse_args()
    if args.checkpoint and args.checkpoints:
        parser.error("use either --checkpoint or --checkpoints")
    if args.label and args.checkpoints:
        parser.error("--label is only valid for a frozen or single-checkpoint run")

    device = torch_device()
    cache = Path(str(data_path("data_dir", "external", "cacd_vs_aligned.npz")))
    checkpoints = args.checkpoints or ([args.checkpoint] if args.checkpoint else [None])
    rows: dict[str, dict] = {}
    outputs: list[Path] = []
    for checkpoint in checkpoints:
        label = args.label or (checkpoint.stem if checkpoint else f"{args.backbone}_frozen")
        output, metrics = _evaluate_one(
            backbone=args.backbone,
            checkpoint=checkpoint,
            label=label,
            device=device,
            cache=cache,
        )
        outputs.append(output)
        rows[label] = metrics

    if args.checkpoints:
        summary = Path(str(data_path("metrics_dir", "cacd_vs", "fine_tuned_summary.json")))
        payload = {
            "protocol": "official identity-disjoint 10 folds",
            "checkpoints": [str(item) for item in args.checkpoints],
            "metrics": _aggregate(rows),
        }
        summary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        write_experiment_manifest(
            summary.with_suffix(".manifest.json"),
            experiment="cacd-vs-fine-tuned-multiseed-summary",
            parameters={"checkpoint_count": len(args.checkpoints)},
            metrics=payload["metrics"],
            inputs=outputs,
            outputs=[summary],
        )
        print(f"summary: {summary}")


if __name__ == "__main__":
    main()
