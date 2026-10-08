"""Fresh CUDA fixed10 native FaceNet cell; never overwrite CPU campaign."""

import argparse
import importlib
import json
from pathlib import Path

import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.run_oriented_campaign import PROTOCOL, paths_from, training_contract, verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--arm", choices=["low", "cross"], default="low")
    parser.add_argument("--seed", choices=[42, 1, 2], type=int, default=42)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh CUDA cell output required")
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU fallback")
    preflight = PROJECT_ROOT / "metrics/oriented_training_preflight_20261003/summary.manifest.json"
    native = verified(preflight, "oriented-uniform-training-preflight")
    inputs = list(dict.fromkeys([preflight, Path(__file__), *paths_from(native)]))
    before = [file_record(path) for path in inputs]
    pairs = (
        PROJECT_ROOT
        / f"metrics/oriented_exposure_matching_20261003/private/{args.arm}_candidate_arm.jsonl"
    )
    args.out.mkdir(parents=True)
    checkpoint = args.out / "private" / f"{args.arm}_s{args.seed}.pt"
    trainer = importlib.import_module("age_gap.training.finetune")
    original = trainer.torch_device
    trainer.torch_device = lambda: "cuda"
    torch.cuda.reset_peak_memory_stats()
    try:
        trainer.finetune(
            epochs=10,
            lr=3e-5,
            batch_size=64,
            margin=0.3,
            patience=10,
            trainable_scope="head",
            gap_weight=0.0,
            backbone_name="facenet",
            crops_dir="faces",
            ckpt_out=checkpoint,
            seed=args.seed,
            pairs_file=str(pairs),
            checkpoint_selection="last_epoch",
            batchnorm_policy="frozen_all",
        )
        torch.cuda.synchronize()
    finally:
        trainer.torch_device = original
    component = checkpoint.with_suffix(".manifest.json")
    actual = verified(component, "pair-contrastive-backbone-finetune")
    training_contract(actual, pairs, args.seed)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(not torch.isfinite(value).all() for value in saved["state_dict"].values()):
        raise ValueError("nonfinite checkpoint")
    if [file_record(path) for path in inputs] != before:
        raise RuntimeError("bound inputs changed")
    summary = dict(
        training_complete=True,
        completed_cells=1,
        evaluation_complete=False,
        publication_ready=False,
        device=torch.cuda.get_device_name(0),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        history=saved["history"],
        limitation="one CUDA cell; CPU/GPU results not pooled or assumed numerically equivalent",
    )
    result = args.out / "summary.json"
    result.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_experiment_manifest(
        args.out / "summary.manifest.json",
        experiment="oriented-native-cuda-training-cell",
        parameters=PROTOCOL
        | dict(
            device="cuda",
            seed=args.seed,
            arm=args.arm,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
        ),
        metrics=summary,
        inputs=inputs,
        outputs=[result, checkpoint, component],
    )


if __name__ == "__main__":
    main()
