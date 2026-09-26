"""Re-evaluate completed strong-backbone checkpoints on endpoint-age-matched FG-NET pairs.

This creates a separate artifact and never rewrites the original training result JSON or
manifest. Example pilot:

    uv run python scripts/reevaluate_strong_backbone_fgnet.py --max-checkpoints 1 \
        --output metrics/strong_backbone_study/fgnet_endpoint_age_matched_pilot.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.common.manifest import sha256_file, write_experiment_manifest
from age_gap.evaluation.benchmark_external import accuracy_10fold, pair_scores
from age_gap.evaluation.fgnet import load_pairs
from age_gap.evaluation.metrics import verification_metrics
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import load_finetuned

BACKBONE = "adaface_ir101"
PROTOCOL = "endpoint_age_matched"
LARGE_GAP_THRESHOLD = 25


def discover_completed_checkpoints(
    result_dir: Path | str, models_dir: Path | str
) -> list[dict[str, Any]]:
    """Find run records with their matching checkpoint; ignore incomplete runs."""
    results = Path(result_dir)
    models = Path(models_dir)
    completed: list[dict[str, Any]] = []
    for result_path in sorted(results.glob("run_*.json")):
        if result_path.name.endswith(".manifest.json"):
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id = result.get("run_id")
        if not isinstance(run_id, str) or not all(
            key in result for key in ("seed", "negative_type", "scope", "learning_rate")
        ):
            continue
        checkpoint = models / f"{run_id}.pt"
        if not checkpoint.is_file():
            continue
        completed.append(
            {
                "run_id": run_id,
                "result_path": result_path,
                "result_manifest": result_path.with_suffix(".manifest.json"),
                "checkpoint": checkpoint,
                "metadata": result,
            }
        )
    return completed


def load_matched_fgnet_pairs(
    cache: Path | str, *, seed: int, endpoint_age_tolerance: int
) -> tuple[list, list, np.ndarray, dict[str, object]]:
    """Load the corrected protocol explicitly, independent of the legacy default."""
    images_a, images_b, labels, _gaps, metadata = load_pairs(
        cache,
        seed=seed,
        protocol=PROTOCOL,
        endpoint_age_tolerance=endpoint_age_tolerance,
        return_metadata=True,
    )
    return images_a, images_b, labels, metadata


def _metric_groups(
    scores: np.ndarray, labels: np.ndarray, stratum_age_gap: np.ndarray
) -> dict[str, dict[str, float]]:
    groups = {
        "overall": np.ones(len(labels), dtype=bool),
        "large_gap_25_plus": stratum_age_gap >= LARGE_GAP_THRESHOLD,
    }
    metrics: dict[str, dict[str, float]] = {}
    for name, mask in groups.items():
        group_labels = labels[mask]
        positive_count = int((group_labels == 1).sum())
        negative_count = int((group_labels == 0).sum())
        if positive_count != negative_count:
            raise ValueError(
                f"Matched FG-NET {name} must be balanced; got {positive_count} positives "
                f"and {negative_count} negatives."
            )
        values = verification_metrics(scores[mask], group_labels)
        values["accuracy_10fold"] = accuracy_10fold(scores[mask], group_labels)
        metrics[name] = values
    return metrics


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def reevaluate(
    *,
    result_dir: Path | str,
    models_dir: Path | str,
    output: Path | str,
    seed: int = 42,
    endpoint_age_tolerance: int = 2,
    device: str | None = None,
    max_checkpoints: int | None = None,
    overwrite: bool = False,
) -> Path:
    destination = Path(output)
    manifest_path = destination.with_suffix(".manifest.json")
    if not overwrite and (destination.exists() or manifest_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing evaluation artifact: {destination}")

    checkpoints = discover_completed_checkpoints(result_dir, models_dir)
    if max_checkpoints is not None:
        if max_checkpoints < 1:
            raise ValueError("max_checkpoints must be positive")
        checkpoints = checkpoints[:max_checkpoints]
    if not checkpoints:
        raise FileNotFoundError("No completed strong-backbone checkpoints found")

    cache = Path(str(data_path("data_dir", "external", "fgnet_crops.npz")))
    base_weight = Path(str(data_path("models_dir", "adaface_ir101.pt")))
    inventory = Path(str(data_path("metrics_dir", "model_inventory.json")))
    evaluation_device = device or torch_device()
    images_a, images_b, labels, metadata = load_matched_fgnet_pairs(
        cache, seed=seed, endpoint_age_tolerance=endpoint_age_tolerance
    )
    labels = np.asarray(labels, dtype=np.int64)
    stratum_gaps = np.asarray(metadata["stratum_age_gap"], dtype=np.int64)
    observed_gaps = np.asarray(metadata["observed_age_gap"], dtype=np.int64)
    frozen = make_backbone(BACKBONE, pretrained=True).to(evaluation_device).eval()
    frozen_scores = pair_scores(frozen, images_a, images_b, evaluation_device, rgb=False)
    frozen_metrics = _metric_groups(frozen_scores, labels, stratum_gaps)
    rows: list[dict[str, Any]] = []
    try:
        for item in checkpoints:
            tuned = load_finetuned(item["checkpoint"], evaluation_device)
            tuned_scores = pair_scores(tuned, images_a, images_b, evaluation_device, rgb=False)
            tuned_metrics = _metric_groups(tuned_scores, labels, stratum_gaps)
            rows.append(
                {
                    "run_id": item["run_id"],
                    "evaluation_protocol": PROTOCOL,
                    "pair_seed": seed,
                    "endpoint_age_tolerance": endpoint_age_tolerance,
                    "negative_type": item["metadata"]["negative_type"],
                    "scope": item["metadata"]["scope"],
                    "learning_rate": item["metadata"]["learning_rate"],
                    "training_seed": item["metadata"]["seed"],
                    "checkpoint_path": item["checkpoint"].name,
                    "checkpoint_sha256": sha256_file(item["checkpoint"]),
                    "frozen_metrics": frozen_metrics,
                    "tuned_metrics": tuned_metrics,
                }
            )
            del tuned, tuned_scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        del frozen, frozen_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result: dict[str, Any] = {
        "backbone": BACKBONE,
        "protocol": PROTOCOL,
        "pair_seed": seed,
        "endpoint_age_tolerance": endpoint_age_tolerance,
        "large_gap_threshold": LARGE_GAP_THRESHOLD,
        "pair_counts": {
            "total": int(len(labels)),
            "positive": int((labels == 1).sum()),
            "negative": int((labels == 0).sum()),
            "large_gap_total": int((stratum_gaps >= LARGE_GAP_THRESHOLD).sum()),
            "large_gap_positive": int(((labels == 1) & (stratum_gaps >= LARGE_GAP_THRESHOLD)).sum()),
            "large_gap_negative": int(((labels == 0) & (stratum_gaps >= LARGE_GAP_THRESHOLD)).sum()),
        },
        "coverage": {
            "positive_retained": int(metadata["n_positive_retained"]),
            "positive_source": int(metadata["n_positive_source"]),
            "positive_unmatched": int(metadata["n_positive_unmatched"]),
            "positive_fraction": float(metadata["positive_coverage"]),
            "large_gap_positive_retained": int(metadata["n_large_gap_positive_retained"]),
            "large_gap_positive_source": int(metadata["n_large_gap_positive_source"]),
            "large_gap_positive_unmatched": int(metadata["n_large_gap_positive_unmatched"]),
            "large_gap_positive_fraction": float(metadata["large_gap_positive_coverage"]),
        },
        "age_gap_diagnostics": {
            "positive_mean": float(observed_gaps[labels == 1].mean()),
            "negative_mean": float(observed_gaps[labels == 0].mean()),
            "negative_match_max_endpoint_error": int(
                np.asarray(metadata["negative_endpoint_match_error"], dtype=np.int64).max(initial=0)
            ),
        },
        "frozen_baseline": frozen_metrics,
        "checkpoint_count": len(rows),
        "checkpoints": rows,
    }
    _write_json(destination, result)

    provenance_inputs: list[Path] = [cache]
    if base_weight.is_file():
        provenance_inputs.append(base_weight)
    if inventory.is_file():
        provenance_inputs.append(inventory)
    for item in checkpoints:
        provenance_inputs.extend([item["checkpoint"], item["result_path"]])
        if item["result_manifest"].is_file():
            provenance_inputs.append(item["result_manifest"])
    write_experiment_manifest(
        manifest_path,
        experiment="strong-backbone-fgnet-endpoint-age-matched-reevaluation",
        parameters={
            "backbone": BACKBONE,
            "protocol": PROTOCOL,
            "pair_seed": seed,
            "endpoint_age_tolerance": endpoint_age_tolerance,
            "large_gap_threshold": LARGE_GAP_THRESHOLD,
            "checkpoint_count": len(rows),
            "device": evaluation_device,
        },
        metrics={
            "pair_counts": result["pair_counts"],
            "coverage": result["coverage"],
            "frozen_baseline": frozen_metrics,
        },
        inputs=provenance_inputs,
        outputs=[destination],
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--endpoint-age-tolerance", type=int, default=2)
    parser.add_argument("--device")
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result_dir = Path(str(data_path("metrics_dir", "strong_backbone_study")))
    models_dir = Path(str(data_path("models_dir", "strong_backbone_study")))
    output = args.output or result_dir / "fgnet_endpoint_age_matched.json"
    saved = reevaluate(
        result_dir=result_dir,
        models_dir=models_dir,
        output=output,
        seed=args.seed,
        endpoint_age_tolerance=args.endpoint_age_tolerance,
        device=args.device,
        max_checkpoints=args.max_checkpoints,
        overwrite=args.overwrite,
    )
    print(f"saved FG-NET matched-protocol evaluation: {saved}")


if __name__ == "__main__":
    main()
