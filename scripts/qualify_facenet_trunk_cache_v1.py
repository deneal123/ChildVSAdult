"""Isolated LOWseed42 ten-epoch cached/native checkpoint comparison."""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.facenet_cached_runtime_v1 import cached_runtime
from scripts.probe_facenet_head_cache_v1 import ATOL, RTOL, compare
from scripts.run_oriented_campaign import PROTOCOL, verified


def compare_checkpoints(reference, candidate):
    for checkpoint in (reference, candidate):
        if (
            checkpoint.get("selected_epoch") != 10
            or checkpoint.get("checkpoint_selection") != "last_epoch"
        ):
            raise ValueError("completed fixed10/last checkpoints required")
        if [row["epoch"] for row in checkpoint["history"]] != list(range(1, 11)):
            raise ValueError("ten sequential completed epochs required")
    if reference["state_dict"].keys() != candidate["state_dict"].keys():
        raise ValueError("checkpoint state keys differ")
    state = {}
    for key, value in reference["state_dict"].items():
        other = candidate["state_dict"][key]
        state[key] = (
            compare(value, other)
            if value.is_floating_point()
            else {
                "within_tolerance": torch.equal(value, other),
                "bitwise_equal": torch.equal(value, other),
                "max_abs": 0 if torch.equal(value, other) else None,
            }
        )
    history = []
    for a, b in zip(reference["history"], candidate["history"], strict=True):
        history.append(
            {
                key: compare(
                    torch.tensor(a[key], dtype=torch.float64),
                    torch.tensor(b[key], dtype=torch.float64),
                )
                for key in ("training_loss", "validation_auc", "mean_gradient_norm")
            }
        )
    passed = all(row["within_tolerance"] for row in state.values()) and all(
        row["within_tolerance"] for epoch in history for row in epoch.values()
    )
    return dict(reference_cell_equivalence_passed=passed, state=state, history=history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh qualification output required")
    torch.set_num_threads(1)
    root = PROJECT_ROOT
    cache_path = root / "metrics/facenet_private_trunk_cache_20261004/summary.manifest.json"
    cache = verified(cache_path, "facenet-private-frozen-trunk-cache")
    if cache["metrics"].get("cache_complete") is not True:
        raise ValueError("completed bank required")
    reference_manifest = (
        root
        / "metrics/oriented_uniform_fixed10_bn_frozen_recovery_20261003_2209/low_s42.manifest.json"
    )
    reference = verified(reference_manifest, "oriented-source-crop-bound-training-cell")
    if reference["metrics"].get("training_complete") is not True:
        raise ValueError("completed native reference required")
    parameters = reference["parameters"]
    for key in (
        "backbone",
        "epochs_requested",
        "epochs_executed",
        "selected_epoch",
        "checkpoint_selection",
        "batchnorm_policy",
        "trainable_scope",
        "learning_rate",
        "batch_size",
        "margin",
        "gap_weight",
        "crops_dir",
    ):
        if parameters.get(key) != PROTOCOL[key]:
            raise ValueError(f"reference protocol mismatch: {key}")
    if parameters.get("seed") != 42:
        raise ValueError("LOW seed42 reference required")
    reference_checkpoint = (
        root / "models/oriented_uniform_fixed10_bn_frozen_recovery_20261003_2209/low_s42.pt"
    )
    if file_record(reference_checkpoint) not in reference["outputs"]:
        raise ValueError("reference checkpoint not bound")
    bank_dir = cache_path.parent / "private"
    features = np.load(bank_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    if list(features.shape) != cache["metrics"]["shape"]:
        raise ValueError("bank shape mismatch")
    rows = {}
    for expected, line in enumerate(
        (bank_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        record = json.loads(line)
        path = (root / record["crop"]["path"]).resolve()
        if (
            record["feature_row"] != expected
            or path in rows
            or record["crop"] not in cache["inputs"]
        ):
            raise ValueError("bank index provenance mismatch")
        rows[path] = expected
    if len(rows) != len(features):
        raise ValueError("bank index coverage mismatch")
    helpers = [
        Path(__file__),
        *[root / f"scripts/facenet_cached_{name}_v1.py" for name in ("runtime", "dataset", "head")],
    ]
    inputs = list(
        dict.fromkeys(
            [
                cache_path,
                reference_manifest,
                reference_checkpoint,
                *helpers,
                *[root / row["path"] for row in cache["inputs"] + cache["outputs"]],
            ]
        )
    )
    before = [file_record(path) for path in inputs]
    trainer = importlib.import_module("age_gap.training.finetune")
    args.out.mkdir(parents=True)
    checkpoint = args.out / "private" / "low_s42.pt"
    with cached_runtime(trainer, features, rows):
        trainer.finetune(
            epochs=10,
            lr=3e-5,
            batch_size=64,
            margin=0.3,
            trainable_scope="head",
            gap_weight=0.0,
            backbone_name="facenet",
            crops_dir="faces",
            ckpt_out=checkpoint,
            seed=42,
            checkpoint_selection="last_epoch",
            batchnorm_policy="frozen_all",
            pairs_file=str(
                root / "metrics/oriented_exposure_matching_20261003/private/low_candidate_arm.jsonl"
            ),
        )
    if [file_record(path) for path in inputs] != before:
        raise RuntimeError("qualification inputs changed")
    summary = compare_checkpoints(
        torch.load(reference_checkpoint, weights_only=True, map_location="cpu"),
        torch.load(checkpoint, weights_only=True, map_location="cpu"),
    )
    summary.update(
        training_complete=True,
        accelerator_qualified=False,
        scientific_evaluation_complete=False,
        publication_ready=False,
        limitation="one reference cell only; full optimizer/RNG trace not present in original checkpoint",
    )
    output = args.out / "summary.json"
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_experiment_manifest(
        args.out / "summary.manifest.json",
        experiment="facenet-cached-reference-cell-comparison",
        parameters=dict(
            seed=42,
            epochs=10,
            rtol=RTOL,
            atol=ATOL,
            scope="LOW reference only; no scientific campaign replacement",
        ),
        metrics=summary,
        inputs=inputs,
        outputs=[output, checkpoint, checkpoint.with_suffix(".manifest.json")],
    )


if __name__ == "__main__":
    main()
