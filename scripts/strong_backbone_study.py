"""Resumable AdaFace IR-101 mechanism study requested by the T-BIOM review.

The full protocol crosses learning rate, trainable scope, negative type, and
three seeds. Every run writes a checkpoint, metrics JSON, and provenance
manifest, so a long GPU campaign can be resumed without repeating completed
cells.

    uv run python scripts/strong_backbone_study.py --prepare-only
    uv run python scripts/strong_backbone_study.py --max-runs 1 --epochs 1
    uv run python scripts/strong_backbone_study.py --repair-manifests
    uv run python scripts/strong_backbone_study.py
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.common.manifest import file_record, write_experiment_manifest
from age_gap.datasets.hard_negatives import run as mine_hard_negatives
from age_gap.datasets.splits import run as build_split
from age_gap.evaluation.external_suite import eval_all
from age_gap.evaluation.representation_diagnostics import compare_representations
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import ImagePairDataset, finetune, load_finetuned

BACKBONE = "adaface_ir101"
DEFAULT_LRS = [1e-6, 1e-5]
DEFAULT_SCOPES = ["head", "tail", "full"]
DEFAULT_SEEDS = [42, 1, 2]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite(payload: object) -> object:
    if isinstance(payload, dict):
        return {key: _finite(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_finite(value) for value in payload]
    if isinstance(payload, float) and not np.isfinite(payload):
        return None
    return payload


def prepare_pair_variants() -> dict[str, Path]:
    canonical = Path(str(data_path("data_dir", "processed", "pairs.jsonl")))
    experiment_dir = Path(str(data_path("data_dir", "processed", "experiments")))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    lookalike = experiment_dir / "pairs_lookalike.jsonl"
    if not lookalike.exists():
        mined = experiment_dir / "pairs_lookalike_unsplit.jsonl"
        mine_hard_negatives(
            pairs_file=str(canonical),
            pairs_out=str(mined),
            top_k=5,
            max_total=40_000,
        )
        build_split(
            pairs_file=str(mined),
            pairs_out=str(lookalike),
            split_map_out=str(experiment_dir / "lookalike_group_splits.jsonl"),
            seed=42,
        )
        write_experiment_manifest(
            lookalike.with_suffix(".manifest.json"),
            experiment="lookalike-negative-pair-variant",
            parameters={"miner_top_k": 5, "max_total": 40_000, "split_seed": 42},
            metrics={},
            inputs=[canonical],
            outputs=[lookalike],
        )
    return {"random": canonical, "lookalike": lookalike}


def _aggregate(
    result_dir: Path,
    *,
    epochs: int = 8,
    negatives: list[str] | None = None,
    scopes: list[str] | None = None,
    learning_rates: list[float] | None = None,
    seeds: list[int] | None = None,
) -> Path:
    negatives = list(negatives or ["random", "lookalike"])
    scopes = list(scopes or DEFAULT_SCOPES)
    learning_rates = list(learning_rates or DEFAULT_LRS)
    seeds = list(seeds or DEFAULT_SEEDS)
    run_paths = sorted(
        path
        for path in result_dir.glob("run_*.json")
        if not path.name.endswith(".manifest.json")
    )
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in run_paths]
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if "negative_type" not in row or "metrics" not in row:
            continue
        row_epochs = row.get("epochs", len(row.get("training_history", [])))
        key = f"e{row_epochs}|{row['negative_type']}|{row['scope']}|{row['learning_rate']:.0e}"
        grouped.setdefault(key, []).append(row)
    expected_run_ids = [
        f"e{epochs}_{negative}_{scope}_lr{learning_rate:.0e}_s{seed}"
        for negative in negatives
        for scope in scopes
        for learning_rate in learning_rates
        for seed in seeds
    ]
    completed_run_ids = {
        str(row["run_id"])
        for row in rows
        if row.get("epochs") == epochs and row.get("run_id") in expected_run_ids
    }
    missing_run_ids = [run_id for run_id in expected_run_ids if run_id not in completed_run_ids]
    summary: dict[str, object] = {
        "backbone": BACKBONE,
        "campaign": {
            "epochs": epochs,
            "negatives": negatives,
            "scopes": scopes,
            "learning_rates": learning_rates,
            "seeds": seeds,
            "expected_runs": len(expected_run_ids),
            "completed_runs": len(completed_run_ids),
            "remaining_runs": len(missing_run_ids),
            "complete": not missing_run_ids,
            "missing_run_ids": missing_run_ids,
        },
        "cells": {},
    }
    for key, cell in sorted(grouped.items()):
        metric_names = sorted(
            set().union(*(row["metrics"].keys() for row in cell))
        )
        aggregates = {}
        for metric in metric_names:
            values = [row["metrics"].get(metric) for row in cell]
            numeric = [float(value) for value in values if isinstance(value, (int, float)) and np.isfinite(value)]
            if numeric:
                aggregates[metric] = {
                    "mean": float(np.mean(numeric)),
                    "std": float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0,
                    "n": len(numeric),
                }
        delta_names = sorted(set().union(*(row.get("metric_deltas", {}).keys() for row in cell)))
        delta_aggregates = {}
        for metric in delta_names:
            values = [row.get("metric_deltas", {}).get(metric) for row in cell]
            numeric = [float(value) for value in values if value is not None]
            if numeric:
                delta_aggregates[metric] = {
                    "mean": float(np.mean(numeric)),
                    "std": float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0,
                    "n": len(numeric),
                }
        epoch_numbers = sorted(
            {
                int(epoch["epoch"])
                for row in cell
                for epoch in row.get("training_history", [])
            }
        )
        trajectory = []
        for epoch_number in epoch_numbers:
            epoch_rows = [
                epoch
                for row in cell
                for epoch in row.get("training_history", [])
                if int(epoch["epoch"]) == epoch_number
            ]
            epoch_summary: dict[str, object] = {"epoch": epoch_number, "n": len(epoch_rows)}
            for field in ("training_loss", "validation_auc", "mean_gradient_norm"):
                values = [float(epoch[field]) for epoch in epoch_rows if epoch.get(field) is not None]
                if values:
                    epoch_summary[field] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    }
            trajectory.append(epoch_summary)
        best_epochs = [
            max(row.get("training_history", []), key=lambda epoch: epoch["validation_auc"])["epoch"]
            for row in cell
            if row.get("training_history")
        ]
        summary["cells"][key] = {
            "seeds": [row["seed"] for row in cell],
            "metrics": aggregates,
            "metric_deltas": delta_aggregates,
            "training_trajectory": trajectory,
            "best_validation_epochs": best_epochs,
            "selected_epochs": [row.get("selected_epoch") for row in cell],
            "executed_epochs": [row.get("epochs_executed", len(row.get("training_history", []))) for row in cell],
            "checkpoint_selection": [row.get("checkpoint_selection", "legacy best_val") for row in cell],
            "batchnorm_policy": [row.get("batchnorm_policy", "legacy adapt_all") for row in cell],
        }
    destination = result_dir / "summary.json"
    _write_json(destination, _finite(summary))
    provenance_inputs: list[Path] = list(run_paths)
    for candidate in (
        Path(str(data_path("models_dir", "adaface_ir101.pt"))),
        Path(str(data_path("metrics_dir", "model_inventory.json"))),
    ):
        if candidate.is_file():
            provenance_inputs.append(candidate)
    write_experiment_manifest(
        destination.with_suffix(".manifest.json"),
        experiment="strong-backbone-mechanism-summary",
        parameters={
            "backbone": BACKBONE,
            "epochs": epochs,
            "expected_runs": len(expected_run_ids),
            "completed_runs": len(completed_run_ids),
            "complete": not missing_run_ids,
        },
        metrics=_finite(summary),
        inputs=provenance_inputs,
        outputs=[destination],
    )
    return destination


def _repair_completed_manifests(
    result_dir: Path, models_dir: Path, variants: dict[str, Path]
) -> int:
    """Rebind completed run/frozen artifacts to the exact base weight inventory."""
    pretrained_weight = Path(str(data_path("models_dir", "adaface_ir101.pt")))
    model_inventory = Path(str(data_path("metrics_dir", "model_inventory.json")))
    provenance_inputs = [path for path in (pretrained_weight, model_inventory) if path.is_file()]
    repaired = 0

    for negative_type, pairs in variants.items():
        frozen_path = result_dir / f"frozen_{negative_type}.json"
        if not frozen_path.is_file():
            continue
        payload = json.loads(frozen_path.read_text(encoding="utf-8"))
        manifest = result_dir / f"frozen_{negative_type}.manifest.json"
        existing = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
        write_experiment_manifest(
            manifest,
            experiment="strong-backbone-frozen-baseline",
            parameters={"backbone": BACKBONE, "negative_type": negative_type},
            metrics=payload["metrics"],
            inputs=[pairs, *provenance_inputs],
            outputs=[frozen_path],
            command=existing.get("command", ["scripts/strong_backbone_study.py"]),
        )
        repaired += 1

    for result_path in sorted(result_dir.glob("run_*.json")):
        if result_path.name.endswith(".manifest.json"):
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not all(key in result for key in ("run_id", "negative_type", "metrics")):
            continue
        checkpoint = models_dir / f"{result['run_id']}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint missing for {result_path}: {checkpoint}")
        pairs = variants[result["negative_type"]]
        manifest = result_path.with_suffix(".manifest.json")
        existing = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
        old_parameters = existing.get("parameters", {})
        parameters = {
            "backbone": result.get("backbone", BACKBONE),
            "negative_type": result["negative_type"],
            "scope": result["scope"],
            "learning_rate": result["learning_rate"],
            "seed": result["seed"],
            "epochs": result.get("epochs", len(result.get("training_history", []))),
            "patience": old_parameters.get("patience", 3),
            "batch_size": old_parameters.get("batch_size", 16),
        }
        write_experiment_manifest(
            manifest,
            experiment="strong-backbone-mechanism-study",
            parameters=parameters,
            metrics=result["metrics"],
            inputs=[pairs, *provenance_inputs],
            outputs=[checkpoint, result_path],
            command=existing.get("command", ["scripts/strong_backbone_study.py"]),
        )
        repaired += 1
    return repaired


def _validate_completed_run(
    *,
    manifest_path: Path,
    expected_parameters: dict[str, object],
    pairs: Path,
    provenance_inputs: list[Path],
    result_path: Path,
    run_id: str,
) -> None:
    """Reject a resume skip unless the manifest still describes these exact artifacts."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot validate completed run manifest {manifest_path}: {exc}") from exc

    if manifest.get("experiment") != "strong-backbone-mechanism-study":
        raise RuntimeError(f"unexpected experiment type in {manifest_path}; refusing resume skip")
    if manifest.get("parameters") != expected_parameters:
        raise RuntimeError(
            f"run parameters do not match {manifest_path}; refusing resume skip. "
            "Use the original settings or a fresh --checkpoint-dir for a new run."
        )

    expected_inputs = [file_record(path) for path in [pairs, *provenance_inputs]]
    recorded_inputs = manifest.get("inputs")
    if recorded_inputs != expected_inputs:
        raise RuntimeError(
            f"input paths or SHA-256 checksums changed for {manifest_path}; "
            "refusing resume skip. Rebuild the affected experiment under a fresh output path."
        )

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise RuntimeError(f"invalid output records in {manifest_path}; refusing resume skip")
    output_by_path = {item.get("path"): item for item in outputs if isinstance(item, dict)}
    result_record = file_record(result_path)
    recorded_result = output_by_path.get(result_record["path"])
    if recorded_result != result_record:
        raise RuntimeError(
            f"result file is missing or its SHA-256 differs from {manifest_path}; "
            "refusing resume skip. Preserve the files and rerun in a fresh output directory."
        )

    checkpoint_records = [
        item
        for item in outputs
        if isinstance(item, dict) and item.get("path") != result_record["path"]
    ]
    if len(checkpoint_records) != 1:
        raise RuntimeError(f"invalid checkpoint record in {manifest_path}; refusing resume skip")
    checkpoint_record = checkpoint_records[0]
    checkpoint_path = Path(str(checkpoint_record.get("path", "")))
    if not checkpoint_path.is_absolute():
        from age_gap.common.io import PROJECT_ROOT

        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if checkpoint_path.name != f"{run_id}.pt":
        raise RuntimeError(f"unexpected checkpoint path in {manifest_path}; refusing resume skip")
    if file_record(checkpoint_path) != checkpoint_record:
        raise RuntimeError(
            f"checkpoint is missing or its SHA-256 differs from {manifest_path}; "
            "refusing resume skip. Preserve the files and rerun in a fresh output directory."
        )


def _ensure_run_outputs_clear(
    *,
    run_id: str,
    result_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    default_checkpoint_path: Path,
) -> None:
    """Never replace an orphan result or checkpoint as a side effect of resuming."""
    if result_path.exists() or manifest_path.exists():
        raise RuntimeError(
            f"incomplete result artifacts exist for {run_id}; no files were overwritten. "
            "Inspect and preserve them; this command cannot safely resume or replace a partial result."
        )
    if checkpoint_path.exists():
        hint = (
            " Choose a fresh --checkpoint-dir to preserve the existing checkpoint."
            if checkpoint_path.resolve() == default_checkpoint_path.resolve()
            else " Choose another --checkpoint-dir; the selected checkpoint path already exists."
        )
        raise RuntimeError(
            f"an orphan checkpoint exists for {run_id} at {checkpoint_path}; "
            "no files were overwritten." + hint
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lrs", nargs="+", type=float, default=DEFAULT_LRS)
    parser.add_argument("--scopes", nargs="+", choices=DEFAULT_SCOPES, default=DEFAULT_SCOPES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--negatives", nargs="+", choices=["random", "lookalike"], default=["random", "lookalike"]
    )
    parser.add_argument("--max-runs", type=int, help="pilot/debug limit; completed runs do not count")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "write new run checkpoints here. Use a fresh directory to preserve an orphan "
            "checkpoint from an interrupted or pilot run."
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--result-dir", type=Path, help="separate corrected campaign metrics from legacy runs")
    parser.add_argument("--checkpoint-selection", choices=["best_val", "last_epoch"], default="best_val")
    parser.add_argument("--batchnorm-policy", choices=["adapt_all", "frozen_all"], default="adapt_all")
    parser.add_argument(
        "--repair-manifests",
        action="store_true",
        help="rewrite manifests for completed runs with current provenance inputs; do not train",
    )
    args = parser.parse_args()
    if args.result_dir is None and (
        args.checkpoint_selection != "best_val" or args.batchnorm_policy != "adapt_all"
    ):
        parser.error("changed training protocol requires a separate --result-dir")

    variants = prepare_pair_variants()
    models_dir = Path(str(data_path("models_dir", "strong_backbone_study")))
    run_models_dir = args.checkpoint_dir or models_dir
    result_dir = args.result_dir or Path(str(data_path("metrics_dir", "strong_backbone_study")))
    models_dir.mkdir(parents=True, exist_ok=True)
    run_models_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.repair_manifests:
        repaired = _repair_completed_manifests(result_dir, models_dir, variants)
        summary = _aggregate(
            result_dir,
            epochs=args.epochs,
            negatives=args.negatives,
            scopes=args.scopes,
            learning_rates=args.lrs,
            seeds=args.seeds,
        )
        print(f"repaired {repaired} manifests; summary: {summary}")
        return
    if args.prepare_only:
        for name, path in variants.items():
            print(f"{name}: {path}")
        return

    device = torch_device()
    pretrained_weight = Path(str(data_path("models_dir", "adaface_ir101.pt")))
    model_inventory = Path(str(data_path("metrics_dir", "model_inventory.json")))
    provenance_inputs = [path for path in (pretrained_weight, model_inventory) if path.is_file()]
    completed_now = 0

    for negative_type in args.negatives:
        pairs = variants[negative_type]
        baseline_pairs = variants["random"] if args.result_dir is not None else pairs
        cell_provenance_inputs = provenance_inputs + (
            [baseline_pairs] if baseline_pairs != pairs else []
        )
        frozen_path = result_dir / f"frozen_{negative_type}.json"
        if frozen_path.exists():
            frozen_metrics = json.loads(frozen_path.read_text(encoding="utf-8"))["metrics"]
        else:
            print(f"[baseline] frozen {negative_type}")
            frozen_model = make_backbone(BACKBONE, pretrained=True).to(device).eval()
            frozen_metrics = _finite(eval_all(frozen_model, device, pairs_file=str(baseline_pairs)))
            _write_json(
                frozen_path,
                {"backbone": BACKBONE, "negative_type": negative_type, "metrics": frozen_metrics},
            )
            write_experiment_manifest(
                result_dir / f"frozen_{negative_type}.manifest.json",
                experiment="strong-backbone-frozen-baseline",
                parameters={"backbone": BACKBONE, "negative_type": negative_type},
                metrics=frozen_metrics,
                inputs=[pairs, *cell_provenance_inputs],
                outputs=[frozen_path],
            )
            del frozen_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for scope in args.scopes:
            for learning_rate in args.lrs:
                for seed in args.seeds:
                    run_id = (
                        f"e{args.epochs}_{negative_type}_{scope}_lr{learning_rate:.0e}_s{seed}"
                    )
                    result_path = result_dir / f"run_{run_id}.json"
                    run_manifest = result_dir / f"run_{run_id}.manifest.json"
                    if result_path.exists() and run_manifest.exists():
                        _validate_completed_run(
                            manifest_path=run_manifest,
                            expected_parameters={
                                "backbone": BACKBONE,
                                "negative_type": negative_type,
                                "scope": scope,
                                "learning_rate": learning_rate,
                                "seed": seed,
                                "epochs": args.epochs,
                                "patience": args.patience,
                                "batch_size": args.batch_size,
                                **({"checkpoint_selection": args.checkpoint_selection,
                                    "batchnorm_policy": args.batchnorm_policy}
                                   if args.result_dir is not None else {}),
                            },
                            pairs=pairs,
                            provenance_inputs=cell_provenance_inputs,
                            result_path=result_path,
                            run_id=run_id,
                        )
                        print(f"[resume] {run_id}")
                        continue
                    checkpoint = run_models_dir / f"{run_id}.pt"
                    _ensure_run_outputs_clear(
                        run_id=run_id,
                        result_path=result_path,
                        manifest_path=run_manifest,
                        checkpoint_path=checkpoint,
                        default_checkpoint_path=models_dir / f"{run_id}.pt",
                    )
                    if args.max_runs is not None and completed_now >= args.max_runs:
                        summary = _aggregate(
                            result_dir,
                            epochs=args.epochs,
                            negatives=args.negatives,
                            scopes=args.scopes,
                            learning_rates=args.lrs,
                            seeds=args.seeds,
                        )
                        print(f"pilot limit reached; summary: {summary}")
                        return

                    print(f"[run] {run_id}")
                    finetune(
                        epochs=args.epochs,
                        patience=args.patience,
                        batch_size=args.batch_size,
                        lr=learning_rate,
                        trainable_scope=scope,
                        backbone_name=BACKBONE,
                        ckpt_out=checkpoint,
                        seed=seed,
                        pairs_file=str(pairs),
                        checkpoint_selection=args.checkpoint_selection,
                        batchnorm_policy=args.batchnorm_policy,
                    )
                    frozen = make_backbone(BACKBONE, pretrained=True).to(device).eval()
                    tuned = load_finetuned(checkpoint, device)
                    # All training-negative variants use the same canonical held-out pairs.
                    # Legacy per-variant test metrics are deliberately not used for new cells.
                    evaluation_pairs = variants["random"] if args.result_dir is not None else pairs
                    metrics = eval_all(tuned, device, pairs_file=str(evaluation_pairs))
                    validation = ImagePairDataset(
                        "val", pairs_file=str(pairs), preprocess=tuned.preprocess
                    )
                    diagnostics = compare_representations(
                        frozen,
                        tuned,
                        validation,
                        device,
                        batch_size=args.batch_size,
                        seed=seed,
                        split_label="val",
                    )
                    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    result = _finite(
                        {
                            "run_id": run_id,
                            "backbone": BACKBONE,
                            "negative_type": negative_type,
                            "scope": scope,
                            "learning_rate": learning_rate,
                            "seed": seed,
                            "epochs": args.epochs,
                            "epochs_executed": len(checkpoint_data.get("history", [])),
                            "selected_epoch": checkpoint_data.get("selected_epoch"),
                            "checkpoint_selection": args.checkpoint_selection,
                            "batchnorm_policy": args.batchnorm_policy,
                            "evaluation_pair_source": str(evaluation_pairs),
                            "metrics": {**metrics, **diagnostics},
                            "frozen_metrics": frozen_metrics,
                            "metric_deltas": {
                                key: value - frozen_metrics[key]
                                for key, value in metrics.items()
                                if value is not None and frozen_metrics.get(key) is not None
                            },
                            "training_history": checkpoint_data.get("history", []),
                        }
                    )
                    _write_json(result_path, result)
                    write_experiment_manifest(
                        run_manifest,
                        experiment="strong-backbone-mechanism-study",
                        parameters={
                            "backbone": BACKBONE,
                            "negative_type": negative_type,
                            "scope": scope,
                            "learning_rate": learning_rate,
                            "seed": seed,
                            "epochs": args.epochs,
                            "patience": args.patience,
                            "batch_size": args.batch_size,
                            **({"checkpoint_selection": args.checkpoint_selection,
                                "batchnorm_policy": args.batchnorm_policy}
                               if args.result_dir is not None else {}),
                        },
                        metrics=result["metrics"],
                        inputs=[pairs, *cell_provenance_inputs],
                        outputs=[checkpoint, result_path],
                    )
                    completed_now += 1
                    _aggregate(
                        result_dir,
                        epochs=args.epochs,
                        negatives=args.negatives,
                        scopes=args.scopes,
                        learning_rates=args.lrs,
                        seeds=args.seeds,
                    )
                    del frozen, tuned, validation, checkpoint_data
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    summary = _aggregate(
        result_dir,
        epochs=args.epochs,
        negatives=args.negatives,
        scopes=args.scopes,
        learning_rates=args.lrs,
        seeds=args.seeds,
    )
    print(f"complete; summary: {summary}")


if __name__ == "__main__":
    main()
