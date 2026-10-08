"""Fresh fixed8 AdaFace CUDA training cell; evaluation/diagnostics remain separate."""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.run_oriented_campaign import verified


def contract(component, pairs, checkpoint, *, scope, lr, seed):
    expected = dict(
        backbone="adaface_ir101",
        epochs_requested=8,
        epochs_executed=8,
        selected_epoch=8,
        checkpoint_selection="last_epoch",
        batchnorm_policy="frozen_all",
        trainable_scope=scope,
        learning_rate=lr,
        batch_size=16,
        margin=0.3,
        gap_weight=0.0,
        crops_dir="faces",
        seed=seed,
    )
    if any(component["parameters"].get(k) != v for k, v in expected.items()):
        raise ValueError("fixed8 training contract differs")
    if (
        Path(component["parameters"].get("pairs_file", "")).resolve() != pairs.resolve()
        or file_record(pairs) not in component["inputs"]
        or file_record(checkpoint) not in component["outputs"]
    ):
        raise ValueError("training input/checkpoint linkage differs")
    history = component["metrics"].get("history", [])
    if [row.get("epoch") for row in history] != list(range(1, 9)):
        raise ValueError("eight ordered completed epochs required")
    for row in history:
        if any(
            not np.isfinite(row.get(k, np.nan))
            for k in ("training_loss", "validation_auc", "mean_gradient_norm", "epoch_seconds")
        ):
            raise ValueError("finite complete training diagnostics required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scope", choices=["head", "tail", "full"], default="head")
    parser.add_argument("--lr", type=float, choices=[1e-6, 1e-5], default=1e-6)
    parser.add_argument("--seed", type=int, choices=[42, 1, 2], default=42)
    parser.add_argument("--negative", choices=["random", "lookalike"], default="random")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh CUDA training destination required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    root = PROJECT_ROOT
    pairs = root / (
        "data/processed/pairs.jsonl"
        if args.negative == "random"
        else "data/processed/experiments/pairs_lookalike.jsonl"
    )
    # Never mine or rebuild shared inputs as a side effect of a training launch.
    if not pairs.is_file():
        raise FileNotFoundError(pairs)
    if not args.execute:
        print("CUDA and pair file available; no crop preflight or training performed")
        return
    torch.set_num_threads(1)
    trainer = importlib.import_module("age_gap.training.finetune")
    from age_gap.models.backbones import make_backbone

    model = make_backbone("adaface_ir101", pretrained=True)
    train = trainer.ImagePairDataset(
        "train", pairs_file=str(pairs), preprocess=trainer._bb_prep(model)
    )
    val = trainer.ImagePairDataset("val", pairs_file=str(pairs), preprocess=trainer._bb_prep(model))
    if not len(train) or not len(val):
        raise ValueError("nonempty train and validation required")
    crops = sorted({p for ds in (train, val) for item in ds._items for p in item[:2]})
    counts = dict(train_pairs=len(train), validation_pairs=len(val), unique_crop_files=len(crops))
    del model, train, val
    inputs = list(
        dict.fromkeys(
            [
                Path(__file__),
                pairs,
                root / "models/adaface_ir101.pt",
                root / "metrics/model_inventory.json",
                root / "src/age_gap/settings/settings.toml",
                root / "pyproject.toml",
                root / "uv.lock",
                root / "scripts/run_oriented_campaign.py",
                root / "scripts/run_restricted_matched_campaign.py",
                root / "scripts/evaluate_oriented_cuda_v1.py",
                *sorted((root / "src/age_gap").rglob("*.py")),
                *crops,
            ]
        )
    )
    before = [file_record(p) for p in inputs]
    args.out.mkdir(parents=True)
    checkpoint = args.out / "private/checkpoint.pt"
    original = trainer.torch_device
    trainer.torch_device = lambda: "cuda"
    torch.cuda.reset_peak_memory_stats()
    try:
        trainer.finetune(
            epochs=8,
            patience=8,
            batch_size=16,
            lr=args.lr,
            margin=0.3,
            trainable_scope=args.scope,
            gap_weight=0.0,
            backbone_name="adaface_ir101",
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
    contract(actual, pairs, checkpoint, scope=args.scope, lr=args.lr, seed=args.seed)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("selected_epoch") != 8 or any(
        not torch.isfinite(v).all() for v in saved["state_dict"].values()
    ):
        raise ValueError("finite completed checkpoint required")
    del saved
    if before != [file_record(p) for p in inputs]:
        raise RuntimeError("training inputs changed; no completion binding published")
    metrics = dict(
        training_complete=True,
        evaluation_complete=False,
        publication_ready=False,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        history=actual["metrics"]["history"],
        **counts,
        limitation="one native fixed-budget training cell; no mechanism conclusion before shared evaluation/representation diagnostics and completed matrix",
    )
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = args.out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="adaface-fixed8-native-cuda-training-cell",
        parameters=actual["parameters"]
        | dict(
            device="cuda",
            threads=1,
            negative=args.negative,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(),
            cudnn_version=torch.backends.cudnn.version(),
        ),
        metrics=metrics,
        inputs=inputs,
        outputs=[summary, checkpoint, component],
    )
    from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs

    validate_written_inputs(target, before)


if __name__ == "__main__":
    main()
