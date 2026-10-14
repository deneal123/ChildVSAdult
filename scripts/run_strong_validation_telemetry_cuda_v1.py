"""Fresh fixed8 AdaFace cell with same-pass validation telemetry, separate schema.

Source preparation only until independently accepted native execution. Existing
v1 cells/evaluators must not consume this distinct experiment by implication.
"""

import argparse
import ast
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.run_oriented_campaign import verified
from scripts.strong_validation_telemetry_v1 import (
    VERSION,
    ValidationCollector,
    installed_validation_collector,
)


def local_script_ancestry(script, root):
    """Literal local script imports only; not dynamic/runtime dependency closure.

    All src/age_gap files are bound separately. No imported helper is executed
    while assembling this list. Include imports inside function bodies too.
    """
    pending = [Path(script).resolve()]
    found = set()
    base = (root / "scripts").resolve()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if not path.is_relative_to(base) or not path.is_file():
            raise ValueError("local script dependency unavailable")
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module == "scripts":
                    modules = ["scripts." + item.name for item in node.names]
                else:
                    modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            for module in modules:
                if module.startswith("scripts."):
                    target = root.joinpath(*module.split(".")).with_suffix(".py")
                    pending.append(target.resolve())
    package = base / "__init__.py"
    if package.is_file():
        found.add(package)
    return sorted(found)


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


def validation_contract(records, history, count):
    """Refuse missing/extra epochs, nonfinite accounting or AUC/schema drift."""
    fields = {
        "epoch",
        "validation_loss",
        "validation_numerator",
        "validation_denominator",
        "validation_pairs",
        "validation_auc",
    }
    if type(count) is not int or count <= 0 or len(records) != 8 or len(history) != 8:
        raise ValueError("eight complete validation records required")
    for i, (row, old) in enumerate(zip(records, history, strict=True), 1):
        if set(row) != fields or type(row["epoch"]) is not int or row["epoch"] != i:
            raise ValueError("validation epoch/schema differs")
        if type(row["validation_pairs"]) is not int or row["validation_pairs"] != count:
            raise ValueError("validation pair count differs")
        for key in fields - {"epoch", "validation_pairs"}:
            if type(row[key]) is not float or not np.isfinite(row[key]):
                raise ValueError("finite floating validation diagnostics required")
        if row["validation_denominator"] != float(count):
            raise ValueError("gap_weight0 validation must have unit weight mass")
        if row["validation_loss"] != row["validation_numerator"] / row["validation_denominator"]:
            raise ValueError("global validation loss accounting differs")
        if old.get("epoch") != i or row["validation_auc"] != old.get("validation_auc"):
            raise ValueError("same-forward validation AUC differs")


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
    if set(val.labels) != {0, 1}:
        raise ValueError("two-class validation required before training")
    crops = sorted({p for ds in (train, val) for item in ds._items for p in item[:2]})
    counts = dict(train_pairs=len(train), validation_pairs=len(val), unique_crop_files=len(crops))
    del model, train, val
    inputs = list(
        dict.fromkeys(
            [
                Path(__file__),
                root / "scripts/strong_validation_telemetry_v1.py",
                pairs,
                root / "models/adaface_ir101.pt",
                root / "metrics/model_inventory.json",
                root / "src/age_gap/settings/settings.toml",
                root / "pyproject.toml",
                root / "uv.lock",
                root / "scripts/run_oriented_campaign.py",
                root / "scripts/run_restricted_matched_campaign.py",
                root / "scripts/evaluate_oriented_cuda_v1.py",
                *local_script_ancestry(Path(__file__), root),
                *sorted((root / "src/age_gap").rglob("*.py")),
                *crops,
            ]
        )
    )
    before = [file_record(p) for p in inputs]
    args.out.mkdir(parents=True)
    checkpoint = args.out / "private/checkpoint.pt"
    collector = ValidationCollector()
    original = trainer.torch_device
    trainer.torch_device = lambda: "cuda"
    torch.cuda.reset_peak_memory_stats()
    try:
        with installed_validation_collector(trainer, collector):
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
    validation_contract(collector.records, actual["metrics"]["history"], counts["validation_pairs"])
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
        validation_history=collector.records,
        validation_loss_scope="global_weight_mass_from_same_validation_forward",
        training_loss_scope="pair_count_weighted_batch_means; global equivalence only for gap_weight0",
        extra_validation_iterators=0,
        native_bitwise_equivalence_verified=False,
        **counts,
        limitation="one native fixed-budget training cell; no mechanism conclusion before shared evaluation/representation diagnostics and completed matrix",
    )
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = args.out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="adaface-fixed8-native-cuda-validation-telemetry-v1",
        parameters=actual["parameters"]
        | dict(
            telemetry_version=VERSION,
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


def cli():
    """Do not echo sensitive crop/weight paths from exception bodies.

    Native launches must still capture all third-party logs privately; this does
    not certify that OpenCV/framework logging itself is disclosure-safe.
    """
    try:
        main()
    except Exception:
        print(
            json.dumps(
                dict(training_complete=False, reason="native-producer-refusal-no-private-details")
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
