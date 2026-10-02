"""Re-evaluate saved common-protocol MTLFace/CACon checkpoints on matched FG-NET.

CANDIDATE artifact (see ``Result.md``). This script is not part of the canonical tree yet; the
root task integrates it as ``scripts/reevaluate_comparators_fgnet.py``.

It evaluates the three saved ``MTLFace-CP`` and three saved ``CACon-CP`` checkpoints, plus the
frozen common backbone they share, under the corrected *endpoint-age-matched* FG-NET protocol.
It reuses the corrected utilities rather than re-deriving them:

* ``scripts.reevaluate_strong_backbone_fgnet`` for the explicit matched pair loader and the
  balanced overall / 25+ metric groups;
* ``age_gap.evaluation.subject_bootstrap`` for paired subject-resampled CIs and LOSO;
* ``age_gap.evaluation.benchmark_external`` for the backbone-aware preprocessing / embedding.

Design constraints (requested by the task):

* CPU only; ``torch`` thread count is capped at 2 and CUDA is hidden before torch import.
* Every unique cached FG-NET crop is embedded exactly once per model; pair scores are then read
  off the cached embedding matrices (no duplicate pair inference).
* The public JSON carries only aggregate statistics (never raw scores, subject IDs or images).
* On a canonical run the per-model embedding matrices are cached under the git-ignored ``data/``
  tree so later analyses can reuse them without re-encoding faces.
* Legacy provenance is reported honestly: published ``fgnet.*`` numbers used the old
  ``legacy_random`` protocol, and any checkpoint whose training manifest is missing is marked as
  such instead of being presented as reproduced.

Example canonical invocation (run by the root task, not by this candidate):

    uv run python scripts/reevaluate_comparators_fgnet.py
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# Models and tensors are explicitly placed on CPU; importing this module does not alter GPU visibility.


def _ensure_project_root_on_path() -> Path:
    """Make ``age_gap`` and the canonical ``scripts`` package importable anywhere.

    The script is authored under the artifact tree but is meant to run from the repository root
    (canonically ``scripts/reevaluate_comparators_fgnet.py``). Walk up from this file until a
    directory that has both ``src/age_gap`` and ``scripts`` is found and prepend it to
    ``sys.path``; fall back to the current working directory so the root can also invoke it from
    a checkout root.
    """
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents, Path.cwd().resolve()]
    for candidate in candidates:
        if (candidate / "src" / "age_gap").is_dir() and (candidate / "scripts").is_dir():
            src = candidate / "src"
            for entry in (str(candidate), str(src)):
                if entry not in sys.path:
                    sys.path.insert(0, entry)
            return candidate
    return here.parent


_PROJECT_ROOT = _ensure_project_root_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from age_gap.common.io import data_path  # noqa: E402
from age_gap.common.manifest import sha256_file, write_experiment_manifest  # noqa: E402
from age_gap.evaluation.benchmark_external import _embed, _preprocess_fn  # noqa: E402
from age_gap.evaluation.subject_bootstrap import (  # noqa: E402
    leave_one_subject_out_auc,
    paired_subject_bootstrap_auc,
)
from age_gap.models.backbones import make_backbone  # noqa: E402
from age_gap.training.sota_common import MTLFaceCommonModel  # noqa: E402

# Reuse the corrected matched-protocol loader and metric groups instead of re-deriving them.
from scripts.reevaluate_strong_backbone_fgnet import (  # noqa: E402
    LARGE_GAP_THRESHOLD,
    PROTOCOL,
    _metric_groups,
    load_matched_fgnet_pairs,
)

EXPERIMENT = "comparator-fgnet-endpoint-age-matched-reevaluation"
FROZEN_BACKBONE = "arcface_r50_casia"
METHODS = ("mtlface", "cacon")
METHOD_DISPLAY = {"mtlface": "mtlface-common-protocol", "cacon": "cacon-common-protocol"}
DEFAULT_SEEDS = (42, 1, 2)
STRATA = ("overall", "large_gap_25plus")
PUBLIC_METRICS = ("roc_auc", "eer", "tar@far=0.01", "tar@far=0.001", "accuracy_10fold")
MAX_THREADS = 2


# --------------------------------------------------------------------------------------
# CPU / device handling
# --------------------------------------------------------------------------------------
def configure_cpu_threads(limit: int = MAX_THREADS) -> str:
    """Force deterministic CPU execution and cap intra-op threads at ``limit`` (<=2)."""
    threads = max(1, min(int(limit), MAX_THREADS, os.cpu_count() or 1))
    torch.set_num_threads(threads)
    with contextlib.suppress(RuntimeError):
        # can only be set before any parallel work; safe to skip if already initialised
        torch.set_num_interop_threads(1)
    return "cpu"


# --------------------------------------------------------------------------------------
# Unique-crop pooling and embedding (each unique crop is encoded once per model)
# --------------------------------------------------------------------------------------
def _crop_digest(image: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).digest()


def build_unique_pool(*image_lists: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Pool the unique crops across all endpoint lists and map each endpoint to its slot.

    Identical crops (same bytes) are stored once, so a model never encodes the same face twice.
    """
    seen: dict[bytes, int] = {}
    unique: list[np.ndarray] = []
    index_lists: list[np.ndarray] = []
    for images in image_lists:
        index = np.empty(len(images), dtype=np.int64)
        for position, image in enumerate(images):
            key = _crop_digest(image)
            slot = seen.get(key)
            if slot is None:
                slot = len(unique)
                seen[key] = slot
                unique.append(image)
            index[position] = slot
        index_lists.append(index)
    return unique, index_lists


def embed_unique_crops(
    model: torch.nn.Module,
    unique_crops: list[np.ndarray],
    device: str,
    *,
    batch_size: int = 128,
) -> np.ndarray:
    """Encode each pooled crop once with the model's own preprocessing."""
    prep = _preprocess_fn(model)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parts = []
    for start in range(0, len(unique_crops), batch_size):
        batch = torch.from_numpy(np.stack([prep(image, bgr=True) for image in unique_crops[start:start + batch_size]]))
        parts.append(_embed(model, batch, device))
    return np.concatenate(parts)


def pair_scores_from_embeddings(
    embeddings: np.ndarray, index_a: np.ndarray, index_b: np.ndarray
) -> np.ndarray:
    """Cosine similarity of L2-normalised embeddings for aligned endpoint index arrays."""
    return (embeddings[index_a] * embeddings[index_b]).sum(axis=1)


def score_unique_crops(
    model: torch.nn.Module,
    unique_crops: list[np.ndarray],
    index_a: np.ndarray,
    index_b: np.ndarray,
    device: str,
    *,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = embed_unique_crops(model, unique_crops, device, batch_size=batch_size)
    return pair_scores_from_embeddings(embeddings, index_a, index_b), embeddings


# --------------------------------------------------------------------------------------
# Checkpoint loading (dispatch over the common-protocol wrappers)
# --------------------------------------------------------------------------------------
def load_common_checkpoint(
    checkpoint: Path | str, device: str, *, backbone_name: str | None = None
) -> tuple[torch.nn.Module, dict]:
    """Rebuild the exact module the common protocol trained and saved.

    The MTLFace common checkpoint stores its backbone state dict under the ``"backbone"``
    key (the ``**best`` spread shadows the name), so the backbone name is not recoverable
    from that file alone. Callers must pass the name recorded in the run manifest/CLI.
    """
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"checkpoint is not a mapping: {checkpoint}")
    fmt = data.get("format")
    saved_name = data.get("backbone") if isinstance(data.get("backbone"), str) else None
    resolved_name = backbone_name or saved_name
    if not resolved_name:
        raise ValueError(f"cannot determine backbone name for checkpoint: {checkpoint}")
    if saved_name is not None and backbone_name is not None and saved_name != backbone_name:
        raise ValueError(
            f"checkpoint backbone {saved_name!r} contradicts requested {backbone_name!r}"
        )
    if fmt == "mtlface-common-v1":
        model = MTLFaceCommonModel(
            make_backbone(resolved_name, pretrained=False), int(data["dimension"])
        )
        model.backbone.load_state_dict(data["backbone"])
        model.separation.load_state_dict(data["separation"])
        module: torch.nn.Module = model
    elif fmt == "cacon-common-v1":
        backbone = make_backbone(resolved_name, pretrained=False)
        backbone.load_state_dict(data["state_dict"])
        module = backbone
    else:
        raise ValueError(f"unsupported common-protocol format {fmt!r}: {checkpoint}")
    return module.to(device).eval(), {
        "format": str(fmt),
        "backbone": resolved_name,
        "backbone_name_in_checkpoint": saved_name,
    }


# --------------------------------------------------------------------------------------
# Discovery and provenance
# --------------------------------------------------------------------------------------
def discover_common_protocol_runs(
    result_dir: Path | str,
    models_dir: Path | str,
    *,
    backbone: str,
    epochs: int,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, list[dict[str, Any]]]:
    """Locate the saved run JSON, training manifest and checkpoint for every method/seed."""
    results = Path(result_dir)
    models = Path(models_dir)
    discovered: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        rows: list[dict[str, Any]] = []
        for seed in seeds:
            run_id = f"{method}_{backbone}_e{epochs}_s{seed}"
            result_path = results / f"{run_id}.json"
            manifest_path = results / f"{run_id}.manifest.json"
            checkpoint = models / f"{run_id}.pt"
            missing = [
                str(path)
                for path in (result_path, checkpoint)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(f"incomplete common-protocol run {run_id}: {missing}")
            rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "run_id": run_id,
                    "result_path": result_path,
                    "manifest_path": manifest_path,
                    "checkpoint": checkpoint,
                }
            )
        discovered[method] = rows
    return discovered


def _manifest_checkpoint_match(manifest: dict, checkpoint: Path, digest: str) -> bool:
    for record in manifest.get("outputs", []):
        if not isinstance(record, dict):
            continue
        if Path(str(record.get("path", ""))).name == checkpoint.name and record.get(
            "sha256"
        ) == digest:
            return True
    return False


def training_provenance(record: dict[str, Any]) -> dict[str, Any]:
    """Describe a checkpoint's provenance without over-claiming reproduction."""
    checkpoint = Path(record["checkpoint"])
    digest = sha256_file(checkpoint)
    manifest_path = Path(record["manifest_path"])
    provenance: dict[str, Any] = {
        "run_id": record["run_id"],
        "seed": record["seed"],
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": digest,
        "training_manifest": manifest_path.name if manifest_path.is_file() else None,
    }
    if not manifest_path.is_file():
        provenance.update(
            {
                "provenance_status": "legacy: training manifest absent",
                "manifest_checkpoint_match": None,
            }
        )
        return provenance
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    match = _manifest_checkpoint_match(manifest, checkpoint, digest)
    provenance.update(
        {
            "provenance_status": (
                "verified: common-protocol training manifest binds this checkpoint"
                if match
                else "manifest present but does not bind this checkpoint digest"
            ),
            "manifest_checkpoint_match": bool(match),
            "manifest_experiment": manifest.get("experiment"),
        }
    )
    return provenance


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"mean": None, "std": None, "n_seeds": 0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "n_seeds": int(array.size),
    }


def aggregate_seed_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean/std across seeds for public metrics and for the paired delta vs frozen."""
    aggregate: dict[str, dict[str, Any]] = {}
    for stratum in STRATA:
        stratum_aggregate: dict[str, Any] = {}
        for metric in PUBLIC_METRICS:
            values = [float(row["metrics"][stratum][metric]) for row in rows]
            stratum_aggregate[metric] = _mean_std(values)
        deltas = [float(row["paired_vs_frozen"][stratum]["delta_auc"]) for row in rows]
        stratum_aggregate["delta_auc_vs_frozen"] = _mean_std(deltas)
        aggregate[stratum] = stratum_aggregate
    return aggregate


def cross_method_rows(per_seed_scores: dict[int, dict[str, np.ndarray]], meta: dict[str, Any],
                      *, bootstrap_seed: int, n_boot: int) -> list[dict[str, Any]]:
    """Paired subject bootstrap of CACon minus MTLFace for every seed and stratum."""
    labels = meta["labels"]
    subject_a = meta["subject_a"]
    subject_b = meta["subject_b"]
    rows: list[dict[str, Any]] = []
    for seed in sorted(per_seed_scores):
        scores = per_seed_scores[seed]
        for stratum in STRATA:
            mask = meta["stratum_masks"][stratum]
            interval = paired_subject_bootstrap_auc(
                scores["mtlface"][mask],
                scores["cacon"][mask],
                labels[mask],
                subject_a[mask],
                subject_b[mask],
                n_boot=n_boot,
                seed=bootstrap_seed,
            )
            rows.append(
                {
                    "seed": seed,
                    "stratum": stratum,
                    "a": METHOD_DISPLAY["mtlface"],
                    "b": METHOD_DISPLAY["cacon"],
                    "delta_direction": "b_minus_a",
                    "delta_auc": interval.delta_auc,
                    "delta_ci95": list(interval.delta_ci95),
                    "n_subjects": interval.n_subjects,
                    "n_pairs": interval.n_pairs,
                }
            )
    return rows


def aggregate_cross_method(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for stratum in STRATA:
        deltas = [float(row["delta_auc"]) for row in rows if row["stratum"] == stratum]
        aggregate[stratum] = {
            "delta_auc": _mean_std(deltas),
            "all_seeds_ci95_exclude_zero": all(
                row["delta_ci95"][0] > 0 or row["delta_ci95"][1] < 0
                for row in rows
                if row["stratum"] == stratum
            ),
        }
    return aggregate


# --------------------------------------------------------------------------------------
# Private embedding cache
# --------------------------------------------------------------------------------------
def save_embedding_cache(
    destination_dir: Path | str,
    embeddings: dict[str, np.ndarray],
    unique_crops: list[np.ndarray],
) -> Path:
    """Persist per-model embedding matrices under an ignored directory (private cache)."""
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "fgnet_comparator_embeddings.npz"
    digests = np.asarray([_crop_digest(image).hex() for image in unique_crops], dtype="U64")
    np.savez_compressed(path, crop_sha256=digests, **embeddings)
    return path


# --------------------------------------------------------------------------------------
# Public payload assembly
# --------------------------------------------------------------------------------------
def build_public_payload(
    *,
    pair_counts: dict[str, int],
    coverage: dict[str, Any],
    negative_seed: int,
    endpoint_age_tolerance: int,
    bootstrap_seed: int,
    n_boot: int,
    frozen_metrics: dict[str, dict[str, float]],
    method_rows: dict[str, list[dict[str, Any]]],
    cross_rows: list[dict[str, Any]],
    legacy_note: str,
) -> dict[str, Any]:
    comparators: dict[str, Any] = {}
    for method, rows in method_rows.items():
        comparators[METHOD_DISPLAY[method]] = {
            "runs": rows,
            "aggregate_across_seeds": aggregate_seed_rows(rows),
        }
    return {
        "experiment": EXPERIMENT,
        "protocol": PROTOCOL,
        "endpoint_age_tolerance": int(endpoint_age_tolerance),
        "negative_seed": int(negative_seed),
        "bootstrap_seed": int(bootstrap_seed),
        "n_bootstrap": int(n_boot),
        "large_gap_threshold": LARGE_GAP_THRESHOLD,
        "strata": list(STRATA),
        "device": "cpu",
        "torch_threads": MAX_THREADS,
        "pair_counts": pair_counts,
        "coverage": coverage,
        "frozen_common_backbone": {
            "backbone": FROZEN_BACKBONE,
            "metrics": frozen_metrics,
        },
        "comparators": comparators,
        "cross_method": {
            "rows": cross_rows,
            "aggregate_across_seeds": aggregate_cross_method(cross_rows),
        },
        "provenance_notes": {
            "training_benchmark_identity_independence": "unverified; candidate adjudication pending",
            "seed_intervals": "same FG-NET subjects reused; intervals are not independent replications",
            "legacy_protocol_note": legacy_note,
            "unique_crop_embedding": (
                "each unique cached FG-NET crop is embedded once per model; pair scores are "
                "cosine similarities read from the cached embedding matrices"
            ),
            "private_embeddings": (
                "embedding matrices are cached under the git-ignored data/ tree and never "
                "included in this public aggregate"
            ),
        },
    }


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def reevaluate(
    *,
    result_dir: Path | str,
    models_dir: Path | str,
    output: Path | str,
    backbone: str = FROZEN_BACKBONE,
    epochs: int = 8,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    endpoint_age_tolerance: int = 2,
    negative_seed: int = 42,
    bootstrap_seed: int = 0,
    n_boot: int = 2000,
    embeddings_dir: Path | str | None = None,
    overwrite: bool = False,
) -> Path:
    if n_boot < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("positive bootstrap count and distinct seeds required")
    destination = Path(output)
    manifest_path = destination.with_suffix(".manifest.json")
    if not overwrite and (destination.exists() or manifest_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing evaluation artifact: {destination}")
    device = configure_cpu_threads()

    runs = discover_common_protocol_runs(
        result_dir, models_dir, backbone=backbone, epochs=epochs, seeds=seeds
    )
    cache = Path(str(data_path("data_dir", "external", "fgnet_crops.npz")))
    base_weight = Path(str(data_path("models_dir", f"{backbone}.pth")))
    inventory = Path(str(data_path("metrics_dir", "model_inventory.json")))

    images_a, images_b, labels, metadata = load_matched_fgnet_pairs(
        cache, seed=negative_seed, endpoint_age_tolerance=endpoint_age_tolerance
    )
    labels = np.asarray(labels, dtype=np.int64)
    stratum_gaps = np.asarray(metadata["stratum_age_gap"], dtype=np.int64)
    meta = {
        "labels": labels,
        "subject_a": np.asarray(metadata["subject_a"]),
        "subject_b": np.asarray(metadata["subject_b"]),
        "stratum_masks": {
            "overall": np.ones(len(labels), dtype=bool),
            "large_gap_25plus": stratum_gaps >= LARGE_GAP_THRESHOLD,
        },
    }
    unique_crops, (index_a, index_b) = build_unique_pool(images_a, images_b)

    embeddings_cache: dict[str, np.ndarray] = {}
    frozen = make_backbone(backbone, pretrained=True).to(device).eval()
    frozen_scores, frozen_embeddings = score_unique_crops(
        frozen, unique_crops, index_a, index_b, device
    )
    embeddings_cache[f"frozen_{backbone}"] = frozen_embeddings
    frozen_groups = _metric_groups(frozen_scores, labels, stratum_gaps)
    frozen_metrics = {"overall": frozen_groups["overall"],
                      "large_gap_25plus": frozen_groups["large_gap_25_plus"]}
    del frozen

    per_seed_scores: dict[int, dict[str, np.ndarray]] = {}
    method_rows: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        rows: list[dict[str, Any]] = []
        for record in runs[method]:
            seed = record["seed"]
            model, checkpoint_info = load_common_checkpoint(
                record["checkpoint"], device, backbone_name=backbone
            )
            scores, embeddings = score_unique_crops(
                model, unique_crops, index_a, index_b, device
            )
            embeddings_cache[f"{method}_e{epochs}_s{seed}"] = embeddings
            groups = _metric_groups(scores, labels, stratum_gaps)
            metrics = {"overall": groups["overall"], "large_gap_25plus": groups["large_gap_25_plus"]}
            paired: dict[str, Any] = {}
            loso: dict[str, Any] = {}
            for stratum in STRATA:
                mask = meta["stratum_masks"][stratum]
                interval = paired_subject_bootstrap_auc(
                    frozen_scores[mask],
                    scores[mask],
                    labels[mask],
                    meta["subject_a"][mask],
                    meta["subject_b"][mask],
                    n_boot=n_boot,
                    seed=bootstrap_seed,
                )
                paired[stratum] = {
                    "frozen_auc": interval.frozen_auc,
                    "tuned_auc": interval.tuned_auc,
                    "delta_auc": interval.delta_auc,
                    "delta_ci95": list(interval.delta_ci95),
                    "n_subjects": interval.n_subjects,
                    "n_pairs": interval.n_pairs,
                }
                sensitivity = leave_one_subject_out_auc(
                    frozen_scores[mask],
                    scores[mask],
                    labels[mask],
                    meta["subject_a"][mask],
                    meta["subject_b"][mask],
                )
                loso[stratum] = {
                    "minimum_delta_auc": sensitivity.minimum_delta_auc,
                    "maximum_delta_auc": sensitivity.maximum_delta_auc,
                    "median_delta_auc": sensitivity.median_delta_auc,
                    "n_evaluable_subjects": sensitivity.n_evaluable_subjects,
                    "n_subjects": sensitivity.n_subjects,
                }
            rows.append(
                {
                    "seed": seed,
                    "run_id": record["run_id"],
                    "checkpoint": checkpoint_info,
                    "provenance": training_provenance(record),
                    "metrics": metrics,
                    "paired_vs_frozen": paired,
                    "leave_one_subject_out_vs_frozen": loso,
                }
            )
            per_seed_scores.setdefault(seed, {})[method] = scores
            del model
        method_rows[method] = rows

    cross_rows = cross_method_rows(
        per_seed_scores, meta, bootstrap_seed=bootstrap_seed, n_boot=n_boot
    )
    pair_counts = {
        "total": int(len(labels)),
        "positive": int((labels == 1).sum()),
        "negative": int((labels == 0).sum()),
        "large_gap_total": int((stratum_gaps >= LARGE_GAP_THRESHOLD).sum()),
        "large_gap_positive": int(((labels == 1) & (stratum_gaps >= LARGE_GAP_THRESHOLD)).sum()),
        "large_gap_negative": int(((labels == 0) & (stratum_gaps >= LARGE_GAP_THRESHOLD)).sum()),
    }
    coverage = {
        "positive_retained": int(metadata["n_positive_retained"]),
        "positive_source": int(metadata["n_positive_source"]),
        "positive_unmatched": int(metadata["n_positive_unmatched"]),
        "positive_fraction": float(metadata["positive_coverage"]),
        "large_gap_positive_retained": int(metadata["n_large_gap_positive_retained"]),
        "large_gap_positive_source": int(metadata["n_large_gap_positive_source"]),
        "large_gap_positive_fraction": float(metadata["large_gap_positive_coverage"]),
        "unique_crops_embedded": len(unique_crops),
    }
    legacy_note = (
        "Published fgnet.* numbers were produced with the legacy_random negative protocol. "
        "These endpoint-age-matched re-evaluations are a corrected protocol and are not "
        "directly comparable to those legacy numbers. Common-protocol checkpoints with a "
        "matching training manifest are controlled reimplementations, not reproductions of "
        "published leaderboard results."
    )
    payload = build_public_payload(
        pair_counts=pair_counts,
        coverage=coverage,
        negative_seed=negative_seed,
        endpoint_age_tolerance=endpoint_age_tolerance,
        bootstrap_seed=bootstrap_seed,
        n_boot=n_boot,
        frozen_metrics=frozen_metrics,
        method_rows=method_rows,
        cross_rows=cross_rows,
        legacy_note=legacy_note,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    outputs: list[Path] = [destination]
    if embeddings_dir is not None:
        outputs.append(save_embedding_cache(embeddings_dir, embeddings_cache, unique_crops))

    inputs: list[Path] = [cache]
    if base_weight.is_file():
        inputs.append(base_weight)
    if inventory.is_file():
        inputs.append(inventory)
    for method in METHODS:
        for record in runs[method]:
            inputs.append(record["checkpoint"])
            if Path(record["manifest_path"]).is_file():
                inputs.append(Path(record["manifest_path"]))

    write_experiment_manifest(
        manifest_path,
        experiment=EXPERIMENT,
        parameters={
            "protocol": PROTOCOL,
            "backbone": backbone,
            "epochs": epochs,
            "seeds": list(seeds),
            "endpoint_age_tolerance": endpoint_age_tolerance,
            "negative_seed": negative_seed,
            "bootstrap_seed": bootstrap_seed,
            "n_boot": n_boot,
            "device": "cpu",
            "torch_threads": MAX_THREADS,
        },
        metrics={
            "pair_counts": pair_counts,
            "coverage": coverage,
            "frozen_common_backbone": frozen_metrics,
        },
        inputs=inputs,
        outputs=outputs,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default=FROZEN_BACKBONE)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--endpoint-age-tolerance", type=int, default=2)
    parser.add_argument("--negative-seed", type=int, default=42)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path(str(data_path("data_dir", "interim", "comparator_fgnet_embeddings"))),
        help="git-ignored directory for the private per-model embedding cache",
    )
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result_dir = Path(str(data_path("metrics_dir", "sota_common_protocol")))
    models_dir = Path(str(data_path("models_dir", "sota_common_protocol")))
    output = args.output or Path(
        str(data_path("metrics_dir", "comparator_fgnet_endpoint_age_matched.json"))
    )
    saved = reevaluate(
        result_dir=result_dir,
        models_dir=models_dir,
        output=output,
        backbone=args.backbone,
        epochs=args.epochs,
        seeds=tuple(args.seeds),
        endpoint_age_tolerance=args.endpoint_age_tolerance,
        negative_seed=args.negative_seed,
        bootstrap_seed=args.bootstrap_seed,
        n_boot=args.n_bootstrap,
        embeddings_dir=None if args.no_embeddings else args.embeddings_dir,
        overwrite=args.overwrite,
    )
    print(f"saved comparator FG-NET re-evaluation: {saved}")


if __name__ == "__main__":
    sys.exit(main())
