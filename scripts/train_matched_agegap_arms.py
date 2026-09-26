"""Train a resumable matched LOW-vs-CROSS age-gap ablation.

Each arm/seed has its own checkpoint and evaluation record. Both arms use the same
FaceNet initialization, optimizer budget, validation/test rows, and matched FG-NET
pair set. Canonical pair files are read only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, read_jsonl
from age_gap.common.manifest import sha256_file, write_experiment_manifest
from age_gap.evaluation.benchmark_external import pair_scores
from age_gap.evaluation.fgnet import load_pairs
from age_gap.evaluation.metrics import eer, roc_auc, tar_at_far
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import evaluate_our_split, finetune, load_finetuned

BACKBONE = "facenet"
FGNET_PROTOCOL = "endpoint_age_matched"
FGNET_LARGE_GAP = 25
DEFAULT_SEEDS = [42, 1, 2]


def _digest_rows(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_arm_files(
    low_path: Path | str,
    cross_path: Path | str,
    *,
    group_to_person: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Check matched-arm labels and that both arms carry identical held-out rows."""
    paths = {"low": Path(low_path), "cross": Path(cross_path)}
    loaded = {name: list(read_jsonl(path)) for name, path in paths.items()}
    if group_to_person is not None:
        unknown_groups = sorted(
            {
                str(row[f"identity_group_{side}"])
                for rows in loaded.values()
                for row in rows
                for side in ("a", "b")
                if row.get(f"identity_group_{side}") is not None
                and str(row[f"identity_group_{side}"]) not in group_to_person
            }
        )
        if unknown_groups:
            raise ValueError(
                f"person_clusters mapping is missing {len(unknown_groups)} identity groups"
            )
    heldout_by_arm: dict[str, dict[str, list[dict[str, Any]]]] = {}
    counts: dict[str, dict[str, int]] = {}
    train_people_by_arm: dict[str, set[str]] = {}
    positive_people_by_arm: dict[str, set[str]] = {}

    def row_people(row: dict[str, Any]) -> set[str]:
        people = set()
        for side in ("a", "b"):
            group = row.get(f"identity_group_{side}")
            if group is not None:
                key = str(group)
                people.add((group_to_person or {}).get(key, key))
        return people

    for arm, rows in loaded.items():
        train_rows = [row for row in rows if row.get("split") == "train"]
        positives = [row for row in train_rows if row.get("label") == 1]
        negatives = [row for row in train_rows if row.get("label") == 0]
        if not positives or len(positives) != len(negatives):
            raise ValueError(f"{arm} arm must have non-empty, class-balanced training rows")
        if arm == "low" and any(not 1 <= int(row.get("age_gap", -1)) <= 2 for row in positives):
            raise ValueError("LOW arm contains a positive outside its 1-2 year gap definition")
        if arm == "cross" and any(int(row.get("age_gap", -1)) < 25 for row in positives):
            raise ValueError("CROSS arm contains a positive below its 25+ year gap definition")
        if any(row.get("label") not in (0, 1) for row in train_rows):
            raise ValueError(f"{arm} arm contains an invalid training label")
        train_people = set().union(*(row_people(row) for row in train_rows))
        positive_people = set().union(*(row_people(row) for row in positives))
        heldout_rows = [row for row in rows if row.get("split") in {"val", "test"}]
        heldout_people = set().union(*(row_people(row) for row in heldout_rows))
        if train_people & heldout_people:
            raise ValueError(f"{arm} arm has train-person overlap with val/test")
        train_people_by_arm[arm] = train_people
        positive_people_by_arm[arm] = positive_people
        heldout_by_arm[arm] = {
            split: [row for row in rows if row.get("split") == split]
            for split in ("val", "test")
        }
        if any(not heldout_by_arm[arm][split] for split in ("val", "test")):
            raise ValueError(f"{arm} arm must contain non-empty val and test rows")
        counts[arm] = {
            "train_positive": len(positives),
            "train_negative": len(negatives),
            "val_pairs": len(heldout_by_arm[arm]["val"]),
            "test_pairs": len(heldout_by_arm[arm]["test"]),
            "train_unique_people": len(train_people),
            "positive_unique_people": len(positive_people),
            "person_groups_without_map": sum(
                1
                for row in train_rows
                for side in ("a", "b")
                if row.get(f"identity_group_{side}") is not None
                and str(row[f"identity_group_{side}"]) not in (group_to_person or {})
            ),
        }
    if train_people_by_arm["low"] & train_people_by_arm["cross"]:
        raise ValueError("LOW and CROSS train-person sets overlap")
    if positive_people_by_arm["low"] & positive_people_by_arm["cross"]:
        raise ValueError("LOW and CROSS positive-person sets overlap")
    for split in ("val", "test"):
        if _digest_rows(heldout_by_arm["low"][split]) != _digest_rows(
            heldout_by_arm["cross"][split]
        ):
            raise ValueError(f"LOW and CROSS arms do not have identical {split} rows")
    return {
        "counts": counts,
        "identity_mapping_available": bool(group_to_person),
        "heldout_sha256": {
            split: _digest_rows(heldout_by_arm["low"][split]) for split in ("val", "test")
        },
    }


def _pair_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "n_pairs": float(len(labels)),
        "n_positive": float((labels == 1).sum()),
        "n_negative": float((labels == 0).sum()),
        "roc_auc": roc_auc(scores, labels),
        "eer": eer(scores, labels),
        "tar@far=0.01": tar_at_far(scores, labels, 0.01),
    }


def load_matched_fgnet(
    cache: Path | str, *, seed: int, endpoint_age_tolerance: int
) -> tuple[list, list, np.ndarray, np.ndarray, dict[str, Any]]:
    images_a, images_b, labels, _gaps, metadata = load_pairs(
        cache,
        seed=seed,
        protocol=FGNET_PROTOCOL,
        endpoint_age_tolerance=endpoint_age_tolerance,
        return_metadata=True,
    )
    labels = np.asarray(labels, dtype=np.int64)
    stratum_gaps = np.asarray(metadata["stratum_age_gap"], dtype=np.int64)
    if int((labels == 1).sum()) != int((labels == 0).sum()):
        raise ValueError("Matched FG-NET overall pairs must have equal positive and negative counts")
    large = stratum_gaps >= FGNET_LARGE_GAP
    if int((labels[large] == 1).sum()) != int((labels[large] == 0).sum()):
        raise ValueError("Matched FG-NET 25+ pairs must have equal positive and negative counts")
    metadata = {**metadata, "stratum_age_gap": stratum_gaps}
    return images_a, images_b, labels, large, metadata


def _fgnet_metrics(
    backbone: torch.nn.Module,
    device: str,
    pair_data: tuple[list, list, np.ndarray, np.ndarray, dict[str, Any]],
) -> dict[str, Any]:
    images_a, images_b, labels, large_mask, metadata = pair_data
    scores = pair_scores(backbone, images_a, images_b, device, rgb=False)
    return _fgnet_metrics_from_scores(scores, labels, large_mask, metadata)


def _fgnet_metrics_from_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    large_mask: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "overall": _pair_metrics(scores, labels),
        "large_gap_25_plus": _pair_metrics(scores[large_mask], labels[large_mask]),
        "coverage": {
            "retained_positive": int(metadata["n_positive_retained"]),
            "source_positive": int(metadata["n_positive_source"]),
            "unmatched_positive": int(metadata["n_positive_unmatched"]),
            "positive_fraction": float(metadata["positive_coverage"]),
            "large_gap_retained_positive": int(metadata["n_large_gap_positive_retained"]),
            "large_gap_source_positive": int(metadata["n_large_gap_positive_source"]),
            "large_gap_unmatched_positive": int(metadata["n_large_gap_positive_unmatched"]),
            "large_gap_positive_fraction": float(metadata["large_gap_positive_coverage"]),
        },
    }


def _paired_subject_bootstrap_cross_minus_low(
    low_scores: np.ndarray,
    cross_scores: np.ndarray,
    labels: np.ndarray,
    subject_a: np.ndarray,
    subject_b: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired identity-cluster bootstrap for CROSS minus LOW ROC-AUC."""
    low = np.asarray(low_scores, dtype=float)
    cross = np.asarray(cross_scores, dtype=float)
    y = np.asarray(labels, dtype=np.int64)
    a, b = np.asarray(subject_a), np.asarray(subject_b)
    if low.ndim != 1 or any(value.shape != low.shape for value in (cross, y, a, b)):
        raise ValueError("paired arm scores, labels, and subject IDs must be equal-length vectors")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("both labels are required for paired subject bootstrap")
    if np.any((y == 1) & (a != b)) or np.any((y == 0) & (a == b)):
        raise ValueError("subject endpoints contradict FG-NET pair labels")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    subjects, inverse = np.unique(np.concatenate((a, b)), return_inverse=True)
    endpoint_a = inverse[: len(y)]
    endpoint_b = inverse[len(y) :]
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        sampled = rng.integers(len(subjects), size=len(subjects))
        multiplicity = np.bincount(sampled, minlength=len(subjects))
        weights = multiplicity[endpoint_a] * np.where(y == 1, 1, multiplicity[endpoint_b])
        if any(not np.any(weights[y == label]) for label in (0, 1)):
            continue
        low_auc = roc_auc_score(y, low, sample_weight=weights)
        cross_auc = roc_auc_score(y, cross, sample_weight=weights)
        deltas.append(float(cross_auc - low_auc))
    if not deltas:
        raise ValueError("no valid paired subject-bootstrap resamples")
    point_delta = float(roc_auc_score(y, cross) - roc_auc_score(y, low))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "method": "paired subject-cluster bootstrap; dyad multiplicity product; percentile 95% CI",
        "delta_cross_minus_low_auc": point_delta,
        "ci95": [float(lo), float(hi)],
        "n_subjects": int(len(subjects)),
        "n_pairs": int(len(y)),
        "n_valid_resamples": len(deltas),
        "n_requested_resamples": n_boot,
        "seed": seed,
    }


def _safe_run_id(
    arm: str,
    epochs: int,
    learning_rate: float,
    scope: str,
    batch: int,
    seed: int,
    checkpoint_selection: str = "best_val",
) -> str:
    policy = "_last" if checkpoint_selection == "last_epoch" else ""
    return f"{arm}_facenet_e{epochs}_lr{learning_rate:.0e}_{scope}_b{batch}_s{seed}{policy}"


def _checkpoint_has_matching_training_manifest(
    checkpoint: Path,
    arm_path: Path,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    trainable_scope: str,
    batch_size: int,
    patience: int,
    checkpoint_selection: str = "best_val",
) -> bool:
    """Only reuse an orphan checkpoint when finetune completed and recorded it."""
    manifest_path = checkpoint.with_suffix(".manifest.json")
    if not checkpoint.is_file() or not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        parameters = payload["parameters"]
        input_hashes = {row["sha256"] for row in payload["inputs"]}
        output_hashes = {row["sha256"] for row in payload["outputs"]}
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    expected = {
        "backbone": BACKBONE,
        "epochs_requested": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "margin": 0.3,
        "patience": patience,
        "trainable_scope": trainable_scope,
        "gap_weight": 0.0,
        "crops_dir": "faces",
        "seed": seed,
        "checkpoint_selection": checkpoint_selection,
    }
    return (
        payload.get("experiment") == "pair-contrastive-backbone-finetune"
        and all(
            parameters.get(key, "best_val" if key == "checkpoint_selection" else None) == value
            for key, value in expected.items()
        )
        and sha256_file(arm_path) in input_hashes
        and sha256_file(checkpoint) in output_hashes
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _flatten_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in payload.items():
        if key == "coverage" or key in {"n_pairs", "n_positive", "n_negative"}:
            continue
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, name))
        elif isinstance(value, (int, float)) and np.isfinite(value):
            flat[name] = float(value)
    return flat


def _aggregate_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, dict[int, dict[str, float]]] = {"low": {}, "cross": {}}
    for row in rows:
        arm = str(row["arm"])
        by_arm[arm][int(row["seed"])] = _flatten_metrics(row["metrics"])
    aggregate: dict[str, Any] = {"arms": {}, "paired_cross_minus_low": {}}
    for arm, seed_rows in by_arm.items():
        names = sorted(set().union(*(values.keys() for values in seed_rows.values()))) if seed_rows else []
        aggregate["arms"][arm] = {}
        for name in names:
            values = [seed_rows[seed][name] for seed in sorted(seed_rows) if name in seed_rows[seed]]
            aggregate["arms"][arm][name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
                "values": values,
            }
    common_seeds = sorted(set(by_arm["low"]) & set(by_arm["cross"]))
    if common_seeds:
        common_metrics = sorted(
            set.intersection(
                *(set(by_arm[arm][seed]) for arm in ("low", "cross") for seed in common_seeds)
            )
        )
        for name in common_metrics:
            deltas = [
                by_arm["cross"][seed][name] - by_arm["low"][seed][name]
                for seed in common_seeds
            ]
            aggregate["paired_cross_minus_low"][name] = {
                "mean": float(np.mean(deltas)),
                "std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "n": len(deltas),
                "seeds": common_seeds,
                "values": deltas,
            }
    return aggregate


def run_experiment(
    *,
    low_arm: Path | str,
    cross_arm: Path | str,
    models_dir: Path | str,
    result_dir: Path | str,
    fgnet_cache: Path | str,
    seeds: list[int] = DEFAULT_SEEDS,
    epochs: int = 10,
    learning_rate: float = 3e-5,
    trainable_scope: str = "head",
    batch_size: int = 64,
    patience: int = 3,
    checkpoint_selection: str = "best_val",
    endpoint_age_tolerance: int = 2,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 0,
    device: str | None = None,
    max_runs: int | None = None,
) -> Path:
    low_arm, cross_arm = Path(low_arm), Path(cross_arm)
    models_dir, result_dir = Path(models_dir), Path(result_dir)
    arm_paths = {"low": low_arm, "cross": cross_arm}
    for path in arm_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a non-empty list of unique integers")
    if checkpoint_selection not in {"best_val", "last_epoch"}:
        raise ValueError("checkpoint_selection must be 'best_val' or 'last_epoch'")
    if bootstrap_resamples < 1 or bootstrap_seed < 0:
        raise ValueError("bootstrap_resamples must be positive and bootstrap_seed non-negative")
    if max_runs is not None and max_runs < 1:
        raise ValueError("max_runs must be positive")

    models_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    evaluation_device = device or torch_device()
    arm_hashes = {arm: sha256_file(path) for arm, path in arm_paths.items()}
    model_inventory = Path(str(data_path("metrics_dir", "model_inventory.json")))
    person_clusters_path = Path(str(data_path("data_dir", "processed", "person_clusters.jsonl")))
    group_to_person = {
        str(row["identity_group_id"]): str(row["person_id"])
        for row in read_jsonl(person_clusters_path)
        if row.get("identity_group_id") is not None and row.get("person_id") is not None
    }
    validation = validate_arm_files(
        low_arm, cross_arm, group_to_person=group_to_person
    )
    person_clusters_sha = (
        sha256_file(person_clusters_path) if person_clusters_path.is_file() else None
    )
    inventory_payload = (
        json.loads(model_inventory.read_text(encoding="utf-8"))
        if model_inventory.is_file()
        else {}
    )
    facenet_artifact = (
        inventory_payload.get("models", {}).get("facenet_casia", {}).get("artifact", {})
    )
    base_weight = Path(facenet_artifact["path"]) if facenet_artifact.get("path") else None
    base_weight_sha = (
        sha256_file(base_weight) if base_weight is not None and base_weight.is_file() else None
    )
    inventory_sha = sha256_file(model_inventory) if model_inventory.is_file() else None
    fgnet_cache = Path(fgnet_cache)
    fgnet_cache_sha = sha256_file(fgnet_cache)
    run_configs = {
        (arm, seed): {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "trainable_scope": trainable_scope,
            "batch_size": batch_size,
            "patience": patience,
            "checkpoint_selection": checkpoint_selection,
            "margin": 0.3,
            "gap_weight": 0.0,
            "arm_file_sha256": arm_hashes[arm],
            "base_weight_sha256": base_weight_sha,
            "model_inventory_sha256": inventory_sha,
            "person_clusters_sha256": person_clusters_sha,
            "fgnet_cache_sha256": fgnet_cache_sha,
            "fgnet_protocol": FGNET_PROTOCOL,
            "fgnet_pair_seed": 42,
            "fgnet_endpoint_age_tolerance": endpoint_age_tolerance,
        }
        for arm in ("low", "cross")
        for seed in seeds
    }
    completed_records: dict[tuple[str, int], dict[str, Any]] = {}
    reusable_checkpoints: set[tuple[str, int]] = set()
    for (arm, seed), config in run_configs.items():
        run_id = _safe_run_id(
            arm, epochs, learning_rate, trainable_scope, batch_size, seed, checkpoint_selection
        )
        checkpoint = models_dir / f"{run_id}.pt"
        result_path = result_dir / f"{run_id}.json"
        manifest_path = result_dir / f"{run_id}.manifest.json"
        if checkpoint.is_file() and result_path.is_file() and manifest_path.is_file():
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if any(
                saved.get(key, "best_val" if key == "checkpoint_selection" else None) != value
                for key, value in config.items()
            ):
                raise ValueError(f"Existing completed run has stale inputs or parameters: {run_id}")
            if saved.get("checkpoint_sha256") != sha256_file(checkpoint):
                raise ValueError(f"Existing completed run checkpoint hash mismatch: {run_id}")
            completed_records[(arm, seed)] = saved
        elif checkpoint.exists():
            if result_path.exists() or manifest_path.exists():
                raise FileExistsError(f"Incomplete result artifact exists for {run_id}")
            if _checkpoint_has_matching_training_manifest(
                checkpoint,
                arm_paths[arm],
                seed=seed,
                epochs=epochs,
                learning_rate=learning_rate,
                trainable_scope=trainable_scope,
                batch_size=batch_size,
                patience=patience,
                checkpoint_selection=checkpoint_selection,
            ):
                reusable_checkpoints.add((arm, seed))
            else:
                raise FileExistsError(
                    f"Checkpoint exists without a matching completed training manifest; "
                    f"refusing to overwrite: {checkpoint}"
                )
        elif result_path.exists() or manifest_path.exists():
            raise FileExistsError(f"Incomplete result artifact exists for {run_id}")

    # Reuse the original baseline when resuming; compute it once for a new campaign.
    frozen_baseline = next(
        (row["frozen_baseline"] for row in completed_records.values()), None
    )
    frozen_fgnet_data = load_matched_fgnet(
        fgnet_cache, seed=42, endpoint_age_tolerance=endpoint_age_tolerance
    )
    fgnet_labels = np.asarray(frozen_fgnet_data[2], dtype=np.int64)
    fgnet_large_mask = np.asarray(frozen_fgnet_data[3], dtype=bool)
    fgnet_metadata = frozen_fgnet_data[4]
    fgnet_subject_a = np.asarray(fgnet_metadata["subject_a"])
    fgnet_subject_b = np.asarray(fgnet_metadata["subject_b"])
    if frozen_baseline is None:
        frozen = make_backbone(BACKBONE, pretrained=True).to(evaluation_device).eval()
        frozen_baseline = {
            "fgnet": _fgnet_metrics(frozen, evaluation_device, frozen_fgnet_data),
            "internal": {
                split: evaluate_our_split(
                    frozen,
                    evaluation_device,
                    split=split,
                    pairs_file=str(low_arm),
                    batch_size=batch_size,
                )
                for split in ("val", "test")
            },
        }
        del frozen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_records: list[dict[str, Any]] = []
    calls = 0
    for seed in seeds:
        for arm in ("low", "cross"):
            run_id = _safe_run_id(
                arm, epochs, learning_rate, trainable_scope, batch_size, seed, checkpoint_selection
            )
            checkpoint = models_dir / f"{run_id}.pt"
            result_path = result_dir / f"{run_id}.json"
            manifest_path = result_dir / f"{run_id}.manifest.json"
            if (arm, seed) in completed_records:
                run_records.append(completed_records[(arm, seed)])
                continue
            if max_runs is not None and calls >= max_runs:
                break

            if (arm, seed) not in reusable_checkpoints:
                finetune(
                    epochs=epochs,
                    lr=learning_rate,
                    batch_size=batch_size,
                    patience=patience,
                    margin=0.3,
                    trainable_scope=trainable_scope,
                    gap_weight=0.0,
                    backbone_name=BACKBONE,
                    crops_dir="faces",
                    ckpt_out=checkpoint,
                    seed=seed,
                    pairs_file=str(arm_paths[arm]),
                    checkpoint_selection=checkpoint_selection,
                )
            tuned = load_finetuned(checkpoint, evaluation_device)
            metrics = {
                "fgnet": _fgnet_metrics(tuned, evaluation_device, frozen_fgnet_data),
                "internal": {
                    split: evaluate_our_split(
                        tuned,
                        evaluation_device,
                        split=split,
                        pairs_file=str(arm_paths[arm]),
                        batch_size=batch_size,
                    )
                    for split in ("val", "test")
                },
            }
            del tuned
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result: dict[str, Any] = {
                "run_id": run_id,
                "arm": arm,
                "backbone": BACKBONE,
                "seed": seed,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "trainable_scope": trainable_scope,
                "batch_size": batch_size,
                "patience": patience,
                "checkpoint_selection": checkpoint_selection,
                "selected_epoch": epochs if checkpoint_selection == "last_epoch" else None,
                "margin": 0.3,
                "gap_weight": 0.0,
                "fgnet_protocol": FGNET_PROTOCOL,
                "fgnet_pair_seed": 42,
                "fgnet_endpoint_age_tolerance": endpoint_age_tolerance,
                "arm_file_sha256": arm_hashes[arm],
                "checkpoint_sha256": sha256_file(checkpoint),
                "base_weight_sha256": base_weight_sha,
                "model_inventory_sha256": inventory_sha,
                "person_clusters_sha256": person_clusters_sha,
                "fgnet_cache_sha256": fgnet_cache_sha,
                "arm_counts": validation["counts"][arm],
                "heldout_sha256": validation["heldout_sha256"],
                "frozen_baseline": frozen_baseline,
                "metrics": metrics,
            }
            _write_json(result_path, result)
            manifest_inputs = [arm_paths[arm], checkpoint, fgnet_cache]
            if base_weight is not None and base_weight.is_file():
                manifest_inputs.append(base_weight)
            if model_inventory.is_file():
                manifest_inputs.append(model_inventory)
            if person_clusters_path.is_file():
                manifest_inputs.append(person_clusters_path)
            write_experiment_manifest(
                manifest_path,
                experiment="matched-low-vs-cross-agegap-facenet-run",
                parameters={
                    "arm": arm,
                    "backbone": BACKBONE,
                    "seed": seed,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "trainable_scope": trainable_scope,
                    "batch_size": batch_size,
                    "patience": patience,
                    "checkpoint_selection": checkpoint_selection,
                    "margin": 0.3,
                    "gap_weight": 0.0,
                    "fgnet_protocol": FGNET_PROTOCOL,
                    "fgnet_pair_seed": 42,
                    "fgnet_endpoint_age_tolerance": endpoint_age_tolerance,
                    "arm_file_sha256": arm_hashes[arm],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                },
                metrics={"frozen_baseline": frozen_baseline, "metrics": metrics},
                inputs=manifest_inputs,
                outputs=[result_path],
            )
            run_records.append(result)
            calls += 1
            if max_runs is not None and calls >= max_runs:
                break
        if max_runs is not None and calls >= max_runs:
            break

    paired_subject_ci: dict[str, dict[str, Any]] = {}
    scores_by_arm_seed: dict[tuple[str, int], np.ndarray] = {}
    for row in run_records:
        key = (str(row["arm"]), int(row["seed"]))
        checkpoint = models_dir / f"{row['run_id']}.pt"
        tuned = load_finetuned(checkpoint, evaluation_device)
        scores_by_arm_seed[key] = pair_scores(
            tuned,
            frozen_fgnet_data[0],
            frozen_fgnet_data[1],
            evaluation_device,
            rgb=False,
        )
        del tuned
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for seed in seeds:
        low_scores = scores_by_arm_seed.get(("low", seed))
        cross_scores = scores_by_arm_seed.get(("cross", seed))
        if low_scores is None or cross_scores is None:
            continue
        paired_subject_ci[str(seed)] = {
            stratum: _paired_subject_bootstrap_cross_minus_low(
                low_scores[mask],
                cross_scores[mask],
                fgnet_labels[mask],
                fgnet_subject_a[mask],
                fgnet_subject_b[mask],
                n_boot=bootstrap_resamples,
                seed=bootstrap_seed + seed,
            )
            for stratum, mask in (
                ("overall", np.ones_like(fgnet_large_mask, dtype=bool)),
                ("large_gap_25_plus", fgnet_large_mask),
            )
        }

    summary_path = result_dir / "summary.json"
    summary = {
        "backbone": BACKBONE,
        "arms": ["low", "cross"],
        "seeds": seeds,
        "training_protocol": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "trainable_scope": trainable_scope,
            "batch_size": batch_size,
            "patience": patience,
            "checkpoint_selection": checkpoint_selection,
            "margin": 0.3,
            "gap_weight": 0.0,
        },
        "fgnet_protocol": {
            "name": FGNET_PROTOCOL,
            "pair_seed": 42,
            "endpoint_age_tolerance": endpoint_age_tolerance,
            "large_gap_threshold": FGNET_LARGE_GAP,
            "paired_subject_bootstrap_resamples": bootstrap_resamples,
            "paired_subject_bootstrap_seed_offset": bootstrap_seed,
        },
        "completed_run_count": len(run_records),
        "expected_run_count": 2 * len(seeds),
        "complete": len(run_records) == 2 * len(seeds),
        "frozen_baseline": frozen_baseline,
        "aggregates": _aggregate_runs(run_records),
        "paired_subject_bootstrap_cross_minus_low": paired_subject_ci,
        "paired_subject_bootstrap_pending_seeds": [
            seed for seed in seeds if str(seed) not in paired_subject_ci
        ],
        "runs": [
            {
                "run_id": row["run_id"],
                "arm": row["arm"],
                "seed": row["seed"],
                "selected_epoch": row.get("selected_epoch"),
                "arm_file_sha256": row["arm_file_sha256"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "metrics": row["metrics"],
            }
            for row in run_records
        ],
    }
    _write_json(summary_path, summary)
    summary_inputs = [
        *arm_paths.values(),
        fgnet_cache,
        person_clusters_path,
        *(
            [model_inventory] if model_inventory.is_file() else []
        ),
        *(
            [base_weight] if base_weight is not None and base_weight.is_file() else []
        ),
        *[
            artifact
            for row in run_records
            for artifact in (
                models_dir / f"{row['run_id']}.pt",
                result_dir / f"{row['run_id']}.json",
                result_dir / f"{row['run_id']}.manifest.json",
            )
        ],
    ]
    write_experiment_manifest(
        summary_path.with_suffix(".manifest.json"),
        experiment="matched-low-vs-cross-agegap-facenet-summary",
        parameters=summary["training_protocol"] | summary["fgnet_protocol"],
        metrics={
            "complete": summary["complete"],
            "completed_run_count": len(run_records),
            "aggregates": summary["aggregates"],
            "paired_subject_bootstrap_cross_minus_low": paired_subject_ci,
        },
        inputs=summary_inputs,
        outputs=[summary_path],
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-arm", type=Path, default=data_path("data_dir", "interim", "matched_agegap_arms", "low_arm.jsonl"))
    parser.add_argument("--cross-arm", type=Path, default=data_path("data_dir", "interim", "matched_agegap_arms", "cross_arm.jsonl"))
    parser.add_argument("--fgnet-cache", type=Path, default=data_path("data_dir", "external", "fgnet_crops.npz"))
    parser.add_argument("--models-dir", type=Path, default=data_path("models_dir", "matched_agegap_arms"))
    parser.add_argument("--result-dir", type=Path, default=data_path("metrics_dir", "matched_agegap_arms"))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--trainable-scope", choices=["head", "tail", "full"], default="head")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--checkpoint-selection", choices=["best_val", "last_epoch"], default="best_val")
    parser.add_argument("--endpoint-age-tolerance", type=int, default=2)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    summary = run_experiment(
        low_arm=args.low_arm,
        cross_arm=args.cross_arm,
        models_dir=args.models_dir,
        result_dir=args.result_dir,
        fgnet_cache=args.fgnet_cache,
        seeds=args.seeds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        trainable_scope=args.trainable_scope,
        batch_size=args.batch_size,
        patience=args.patience,
        checkpoint_selection=args.checkpoint_selection,
        endpoint_age_tolerance=args.endpoint_age_tolerance,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        device=args.device,
        max_runs=args.max_runs,
    )
    print(f"matched LOW/CROSS result summary: {summary}")


if __name__ == "__main__":
    main()
