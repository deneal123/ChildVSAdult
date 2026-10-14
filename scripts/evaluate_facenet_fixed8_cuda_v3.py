"""Fresh native FaceNet fixed8 full-pool CUDA evaluation, root-corrected E421.

New v2 training histories only, not historical recovery. Strict native bindings,
full650 FG-NET pool with canonical matching coverage (not all raw positives),
per-fixed-checkpoint subject intervals; optional retrospective frozen/last loss.
No CPU neural fallback, no implicit download, no overwrite, no mechanism claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs
from scripts.run_facenet_validation_cuda_v2 import (
    CASIA_WEIGHT_NAME,
    FIXED_BUDGET,
    SEEDS,
    inventory_weight_record,
    require_pretrained_weight,
    resolve_facenet_casia_weight,
    state_dict_digest,
    validate_weight_record,
)
from scripts.run_oriented_campaign import paths_from, verified
from scripts.shared_byte_integrity_v2 import parse_manifest_strict
from scripts.strong_subject_inference_v1 import infer

#: Native experiment whose completed cells only may be evaluated here.
NATIVE_EXPERIMENT = "facenet-fixed8-validation-loss-v2"
#: Distinct new evaluation experiment identifier.
OUTPUT_EXPERIMENT = "facenet-fixed8-validation-full-fgnet-cuda-v3"
PROPOSAL_VERSION = "weak-fixed8-evaluator-v3-root-e421"

FGNET_CACHE = PROJECT_ROOT / "data/external/fgnet_crops.npz"
INVENTORY = PROJECT_ROOT / "metrics/model_inventory.json"
NATIVE_PAIRS = PROJECT_ROOT / "data/processed/pairs.jsonl"
NATIVE_SUMMARY_NAME = "summary.json"
NATIVE_MANIFEST_NAME = "summary.manifest.json"
NATIVE_LIMITATION = "single fresh fixed8 cell; no legacy weak recovery claim"
EXPECTED_FGNET_CROPS = 650

#: Native parameter contract shared by every accepted weak v2 cell, derived from the
#: producer's ``FIXED_BUDGET`` so it cannot silently drift.  ``learning_rate`` and ``seed``
#: are checked separately; the shared learning rate is enforced across all bindings in
#: :func:`require_shared_init`.
BASE_CONTRACT: dict[str, Any] = {
    **{key: value for key, value in FIXED_BUDGET.items() if key != "learning_rate"},
    "device": "cuda",
    "threads": 1,
}
ALLOWED_LRS = (1e-6, 1e-5)

#: Exact native standalone-summary projection (source-defined; NOT the metric projection).
NATIVE_SUMMARY_KEYS = (
    "experiment",
    "seed",
    "history",
    "training_complete",
    "evaluation_complete",
    "publication_ready",
    "limitation",
)
#: Exact native manifest-metrics projection.
NATIVE_METRICS_KEYS = (
    "history",
    "training_complete",
    "evaluation_complete",
    "publication_ready",
    "selected_epoch",
)

#: Full fifteen-field source-defined epoch schema.
HISTORY_INT_FIELDS = (
    "epoch",
    "train_batches",
    "train_pairs",
    "validation_pairs",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)
HISTORY_FLOAT_FIELDS = (
    "train_loss",
    "validation_loss",
    "train_numerator",
    "train_denominator",
    "validation_numerator",
    "validation_denominator",
    "validation_auc",
    "mean_gradient_norm",
    "epoch_seconds",
)
HISTORY_FIELDS = (*HISTORY_INT_FIELDS, *HISTORY_FLOAT_FIELDS)

#: ROC-v2 protocol hard-coded by ``scripts.reevaluate_fgnet_metrics_v2.protocol``.
ROC_V2_PROTOCOL_SEED = 42
ROC_V2_TOLERANCE = 2
#: Shared-subject bootstrap settings passed to the existing per-checkpoint ``infer``.
N_BOOT = 2000
BOOTSTRAP_SEED = 0
#: Evaluation batch size, matching the native training batch size.
BATCH_SIZE = 16
#: Native contrastive margin, reused for the retrospective validation-loss diagnostic.
MARGIN = 0.3

#: Canonical sources whose exact bytes are bound into the evaluation manifest inputs.
SOURCE_DEPENDENCIES = (
    "scripts/shared_byte_integrity_v2.py",
    "scripts/run_facenet_validation_cuda_v2.py",
    "scripts/evaluate_strong_cuda_fgnet_v1.py",
    "scripts/evaluate_oriented_cuda_v1.py",
    "scripts/reevaluate_fgnet_metrics_v2.py",
    "scripts/strong_subject_inference_v1.py",
    "scripts/benchmark_metrics_v2.py",
    "scripts/verification_metrics_v2.py",
    "scripts/run_oriented_campaign.py",
    "scripts/run_restricted_matched_campaign.py",
    "scripts/measure_checkpoint_validation_cuda_v2.py",
    "tests/test_facenet_validation_cuda_v2.py",
)


# --------------------------------------------------------------------------------------
# Pure contract helpers (fully testable without CUDA, models, weights or native data)
# --------------------------------------------------------------------------------------
def strict_equal(actual: Any, expected: Any) -> bool:
    """Typed equality: ``True`` never equals ``1`` and ``1`` never equals ``1.0``."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(
            strict_equal(actual[key], value) for key, value in expected.items()
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            strict_equal(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def freeze_value(value: Any) -> Any:
    """Order-sensitive, type-sensitive JSON-comparable view for byte-stable comparison."""
    if isinstance(value, dict):
        return {key: freeze_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [freeze_value(item) for item in value]
    return value


def canonical_bindings(paths) -> list[Path]:
    """Accept exactly one or three explicit native run directories, no duplicate path."""
    resolved = [Path(path) for path in paths]
    if len(resolved) not in (1, 3):
        raise ValueError("exactly one or three --bindings are required")
    keys = [str(path.resolve()) for path in resolved]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate --bindings path")
    return resolved


def require_seed_set(seeds) -> list[int]:
    """Require canonical explicit seeds: any one of 42/1/2, or exactly all three."""
    seeds = list(seeds)
    if any(type(seed) is not int or seed not in SEEDS for seed in seeds):
        raise ValueError("native seeds must be the canonical 42/1/2")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate native seed binding")
    if len(seeds) != 1 and set(seeds) != set(SEEDS):
        raise ValueError("three bindings must be exactly seeds 42/1/2")
    return seeds


def _is_number(value: Any) -> bool:
    return type(value) in (int, float) and not isinstance(value, bool)


def finite_history(history, label: str = "history") -> list:
    """Full fifteen-field source-defined epoch schema, ordered 1..8, finite and consistent.

    Enforces typed integer epoch/order/counts, typed finite float metrics, AUC in ``[0, 1]``,
    closure of the published losses to their global numerator/denominator, constant counts
    and positive masses across epochs, and — because the fixed8 budget has ``gap_weight == 0``
    — denominators equal to pair counts and training batches equal to the batch ceiling.
    """
    if not isinstance(history, list) or len(history) != 8:
        raise ValueError(f"{label} must be a list of eight epochs")
    if [row.get("epoch") if isinstance(row, dict) else None for row in history] != list(
        range(1, 9)
    ):
        raise ValueError(f"{label} must carry ordered native integer epochs 1..8")
    for row in history:
        if not isinstance(row, dict) or set(row) != set(HISTORY_FIELDS):
            raise ValueError(f"{label} epoch must carry exactly the fifteen native fields")
        if type(row["epoch"]) is not int or row["epoch"] not in range(1, 9):
            raise ValueError(f"{label} epoch must be a native int in 1..8")
        for key in HISTORY_INT_FIELDS:
            if type(row[key]) is not int:
                raise ValueError(f"{label} count {key!r} must be a native int")
        for key in HISTORY_FLOAT_FIELDS:
            value = row[key]
            if type(value) is not float or not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} metric {key!r} must be a finite number")
        auc = float(row["validation_auc"])
        if not 0.0 <= auc <= 1.0:
            raise ValueError(f"{label} validation AUC must lie in [0, 1]")
        for count in ("train_pairs", "validation_pairs", "train_batches"):
            if row[count] <= 0:
                raise ValueError(f"{label} count {count!r} must be positive")
        for mass in ("train_denominator", "validation_denominator"):
            if float(row[mass]) <= 0:
                raise ValueError(f"{label} weight mass {mass!r} must be positive")
        # Published losses are exactly the global numerator/denominator quotients.
        for loss_key, num_key, den_key in (
            ("train_loss", "train_numerator", "train_denominator"),
            ("validation_loss", "validation_numerator", "validation_denominator"),
        ):
            quotient = float(row[num_key]) / float(row[den_key])
            if not math.isclose(float(row[loss_key]), quotient, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"{label} {loss_key!r} is not its global quotient")
        # gap_weight == 0 => unweighted masses equal pair counts; batches equal the ceiling.
        if float(row["train_denominator"]) != row["train_pairs"]:
            raise ValueError(f"{label} zero gap weight must make train mass equal pair count")
        if float(row["validation_denominator"]) != row["validation_pairs"]:
            raise ValueError(f"{label} zero gap weight must make val mass equal pair count")
        if row["train_batches"] != -(-row["train_pairs"] // BATCH_SIZE):
            raise ValueError(f"{label} train_batches must equal ceil(train_pairs / batch)")
        if row["epoch_seconds"] <= 0 or row["peak_allocated_bytes"] < 0 or row["peak_reserved_bytes"] < row["peak_allocated_bytes"]:
            raise ValueError("native duration/memory accounting differs")
    first = history[0]
    for row in history:
        for key in (
            "train_pairs",
            "validation_pairs",
            "train_batches",
            "train_denominator",
            "validation_denominator",
        ):
            if row[key] != first[key]:
                raise ValueError(f"{label} {key!r} must be constant across epochs")
    return history


def require_native_summary(summary: dict) -> dict:
    """Exact source-defined native standalone-summary projection."""
    if not isinstance(summary, dict) or set(summary) != set(NATIVE_SUMMARY_KEYS):
        raise ValueError("native summary must carry the exact seven native keys in order")
    if summary["experiment"] != NATIVE_EXPERIMENT:
        raise ValueError("native summary experiment differs")
    if type(summary["seed"]) is not int or summary["seed"] not in SEEDS:
        raise ValueError("native summary seed must be one of 42/1/2")
    finite_history(summary["history"], "native summary")
    if summary["training_complete"] is not True:
        raise ValueError("native summary must record TRUE training completion")
    if summary["evaluation_complete"] is not False or summary["publication_ready"] is not False:
        raise ValueError("native summary must record evaluation/publication incomplete")
    if summary["limitation"] != NATIVE_LIMITATION:
        raise ValueError("native summary limitation differs from the source constant")
    return summary


def native_contract(native: dict, run_directory: Path, weight_record: dict) -> dict:
    """Validate one completed weak v2 native cell; return its selected last checkpoint."""
    if not isinstance(native, dict) or native.get("experiment") != NATIVE_EXPERIMENT:
        raise ValueError(
            "completed native weak fixed8 v2 cell required; legacy weak recovery and "
            "pair-contrastive-backbone-finetune lineage are not accepted"
        )
    parameters = native.get("parameters")
    metrics = native.get("metrics")
    if not isinstance(parameters, dict) or not isinstance(metrics, dict):
        raise ValueError("native parameters/metrics mappings required")
    for key, value in BASE_CONTRACT.items():
        if not strict_equal(parameters.get(key), value):
            raise ValueError(f"native parameter {key!r} differs from the fixed8 contract")
    lr = parameters.get("learning_rate")
    if type(lr) is not float or lr not in ALLOWED_LRS:
        raise ValueError("native learning rate must be 1e-6 or 1e-5")
    seed = parameters.get("seed")
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("native seed must be one of 42/1/2")
    if not strict_equal(parameters.get("pairs_file"), str(NATIVE_PAIRS.resolve())):
        raise ValueError("native pairs_file must be the exact canonical random-negative file")
    if metrics.get("training_complete") is not True:
        raise ValueError("TRUE native training completion required")
    if type(metrics.get("selected_epoch")) is not int or metrics["selected_epoch"] != 8:
        raise ValueError("native selected epoch must be the native int eight")
    if set(metrics) != set(NATIVE_METRICS_KEYS):
        raise ValueError("native metrics must carry the exact five native keys in order")
    if metrics.get("evaluation_complete") is not False or metrics.get("publication_ready") is not False:
        raise ValueError("native metrics must record evaluation/publication incomplete")
    history = finite_history(metrics.get("history"), "native")

    # The actual cached weight record (not a matching hash string in parameters) must be a
    # declared native input; the same must hold for the canonical pair file.
    inputs = native.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("native declared inputs required")
    if not any(record == weight_record for record in inputs):
        raise ValueError("native inputs do not bind the exact cached weight record")
    weight_sha = weight_record.get("sha256")
    if parameters.get("pretrained_weight_sha256") != weight_sha:
        raise ValueError("native declared weight digest differs from the cached weight record")
    if file_record(NATIVE_PAIRS) not in inputs:
        raise ValueError("native inputs do not bind the canonical pair file")

    # The original standalone summary must be a declared native output, and the source-defined
    # projection must match; re-hash originals (already done by ``verified``) before load.
    checkpoint = Path(run_directory) / "private/checkpoint.pt"
    summary_path = Path(run_directory) / NATIVE_SUMMARY_NAME
    outputs = native.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("native declared outputs required")
    if file_record(checkpoint) not in outputs:
        raise ValueError("native manifest does not bind its last checkpoint")
    if file_record(summary_path) not in outputs:
        raise ValueError("native manifest does not bind its original standalone summary")
    summary, _, _ = parse_manifest_strict(summary_path.read_bytes())
    require_native_summary(summary)
    expected_summary = dict(experiment=NATIVE_EXPERIMENT, seed=seed, history=history,
                            training_complete=True, evaluation_complete=False,
                            publication_ready=False, limitation=NATIVE_LIMITATION)
    if not strict_equal(summary, expected_summary):
        raise ValueError("native summary differs from its source-defined projection")
    if not isinstance(parameters.get("model_init_state_sha256"), str):
        raise ValueError("native model init digest required")

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_contract(saved, history)
    return {"seed": seed, "learning_rate": lr, "checkpoint": checkpoint, "history": history}


def checkpoint_contract(saved: dict, native_history: list) -> dict:
    """Pure weights-only checkpoint contract matching the native producer semantics.

    Requires the native backbone/crops/BatchNorm metadata, a native int ``selected_epoch``
    equal to eight, ``last_epoch`` selection, a nonempty finite tensor state and a history
    strictly equal to the native manifest history.  Errors never name tensors.
    """
    if not isinstance(saved, dict) or "state_dict" not in saved:
        raise ValueError("checkpoint must be a mapping with a state_dict")
    if saved.get("backbone") != "facenet" or saved.get("crops_dir") != "faces":
        raise ValueError("checkpoint must declare the facenet backbone and faces crops")
    if saved.get("batchnorm_policy") != "frozen_all":
        raise ValueError("checkpoint must declare the frozen-all BatchNorm policy")
    if type(saved.get("selected_epoch")) is not int or saved.get("selected_epoch") != 8:
        raise ValueError("checkpoint selected_epoch must be the native int eight")
    if saved.get("checkpoint_selection") != "last_epoch":
        raise ValueError("checkpoint checkpoint_selection must be last_epoch")
    history = finite_history(saved.get("history"), "checkpoint")
    if not strict_equal(history, native_history):
        raise ValueError("checkpoint history differs from the native manifest history")
    state = saved["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint state_dict must be a nonempty mapping")
    for tensor in state.values():
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError("checkpoint contains a non-tensor or non-finite state entry")
    return saved


def require_shared_init(natives, weight_record: dict) -> dict:
    """Every provided training must share one byte-bound weight and frozen initialisation.

    All provided cells must also use the same exact learning rate, so only the seed varies.
    """
    weight_shas, init_shas, rates = set(), set(), set()
    for native in natives:
        parameters = native["parameters"]
        declared = parameters.get("pretrained_weight_sha256")
        if declared != weight_record.get("sha256"):
            raise ValueError("declared pretrained weight digest differs from the cached weight")
        init = parameters.get("model_init_state_sha256")
        if not isinstance(init, str) or not init:
            raise ValueError("model init digest required for shared initialisation")
        weight_shas.add(declared)
        init_shas.add(init)
        rates.add(parameters.get("learning_rate"))
    if len(weight_shas) != 1 or len(init_shas) != 1:
        raise ValueError("all provided training cells must share one frozen initialisation")
    if len(rates) != 1:
        raise ValueError("all provided training cells must share one exact learning rate")
    return {
        "pretrained_weight_sha256": next(iter(weight_shas)),
        "model_init_state_sha256": next(iter(init_shas)),
        "learning_rate": next(iter(rates)),
    }


def aggregate_metrics(result: dict) -> dict:
    """Manifest metrics is the full aggregate result, never a bookkeeping projection."""
    return result


def require_full_pool(crops, subjects) -> None:
    """The full actual aligned BGR112 FG-NET pool is mandatory (no row dropping)."""
    if (
        not isinstance(crops, np.ndarray)
        or crops.dtype != np.uint8
        or crops.shape != (len(subjects), 112, 112, 3)
    ):
        raise ValueError("full aligned BGR112 crop pool required")
    if len(crops) != EXPECTED_FGNET_CROPS:
        raise ValueError("the full actual FG-NET crop pool is required")


def validate_embeddings(vectors: np.ndarray, n_crops: int) -> np.ndarray:
    """Complete finite unit 512-dimensional embeddings for every full-pool crop."""
    if (
        not isinstance(vectors, np.ndarray)
        or vectors.shape != (n_crops, 512)
        or vectors.dtype.kind != "f"
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    ):
        raise ValueError("complete finite unit 512-d embeddings required for every crop")
    return vectors


def coverage_complete(coverage: dict) -> bool:
    """Validate canonical retained/source accounting, allowing unmatched positives."""
    keys = {"source_positives", "retained_positives", "source_25plus_positives", "retained_25plus_positives"}
    if type(coverage) is not dict or set(coverage) != keys:
        return False
    if any(type(v) is not int or v < 0 for v in coverage.values()):
        return False
    return (0 < coverage["retained_positives"] <= coverage["source_positives"]
            and 0 < coverage["retained_25plus_positives"] <= coverage["source_25plus_positives"]
            <= coverage["source_positives"]
            and coverage["retained_25plus_positives"] <= coverage["retained_positives"])


def validation_measurement(measured: dict) -> dict:
    """Source-bound retrospective frozen/last validation-loss diagnostic (not history)."""
    required = (
        "validation_loss",
        "validation_numerator",
        "validation_denominator",
        "validation_auc",
        "validation_pairs",
    )
    if not isinstance(measured, dict) or any(key not in measured for key in required):
        raise ValueError("validation measurement lacks a canonical field")
    if (type(measured["validation_pairs"]) is not int
            or any(type(measured[k]) is not float or not math.isfinite(measured[k]) or measured[k] < 0
                   for k in required if k != "validation_pairs")):
        raise ValueError("native validation measurement types required")
    denominator = measured["validation_denominator"]
    if denominator != measured["validation_pairs"]:
        raise ValueError("zero-gap validation mass/count differs")
    if denominator <= 0 or measured["validation_pairs"] <= 0:
        raise ValueError("validation measurement requires positive mass and count")
    if not math.isclose(
        float(measured["validation_loss"]),
        float(measured["validation_numerator"]) / denominator,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("validation measurement loss is not its global quotient")
    if not 0.0 <= float(measured["validation_auc"]) <= 1.0:
        raise ValueError("validation measurement AUC must lie in [0, 1]")
    return {
        "validation_loss": float(measured["validation_loss"]),
        "validation_numerator": float(measured["validation_numerator"]),
        "validation_denominator": denominator,
        "validation_auc": float(measured["validation_auc"]),
        "validation_pairs": int(measured["validation_pairs"]),
        "binding": "retrospective frozen/last validation diagnostic; not a historical trajectory",
    }


def source_paths() -> list[Path]:
    """Canonical producer/evaluator sources bound into the manifest inputs."""
    return [Path(__file__), *(PROJECT_ROOT / rel for rel in SOURCE_DEPENDENCIES)]


def native_manifests(bindings) -> list[Path]:
    """The original native completion manifests themselves, one per binding."""
    return [native_manifest_path(run_directory(binding)) for binding in bindings]


def build_inputs(*, bindings, natives, weight: Path, cache: Path, extra=()) -> list[Path]:
    """Deduplicated original-byte provenance including each original native manifest."""
    return list(
        dict.fromkeys(
            [
                *source_paths(),
                Path(weight),
                Path(cache),
                INVENTORY,
                NATIVE_PAIRS,
                *native_manifests(bindings),
                *extra,
                *(p for native in natives for p in paths_from(native)),
            ]
        )
    )


def snapshot(paths) -> list:
    """Full file records for every declared input (before/after stability)."""
    return [file_record(path) for path in paths]


def run_directory(binding) -> Path:
    """Resolve a binding that may be the run directory or its completion manifest path."""
    path = Path(binding)
    if path.name in (NATIVE_MANIFEST_NAME, NATIVE_SUMMARY_NAME):
        return path.parent
    return path


def native_manifest_path(directory: Path) -> Path:
    return Path(directory) / NATIVE_MANIFEST_NAME


def native_summary_path(directory: Path) -> Path:
    return Path(directory) / NATIVE_SUMMARY_NAME


# --------------------------------------------------------------------------------------
# CUDA seams (lazy imports; monkeypatchable in synthetic tests)
# --------------------------------------------------------------------------------------
def build_architecture() -> Any:
    """Cached pretrained FaceNet architecture; the weight cache was already verified."""
    from age_gap.models.backbones import make_backbone

    return make_backbone("facenet", pretrained=True)


def load_checkpoint_state(model: Any, checkpoint: Path) -> Any:
    """Load the full checkpoint state with ``strict=True``; never silently partial."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model


def embed_pool(model, crops, *, device: str, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Full aligned BGR112 pool -> model-specific preprocessing -> unit embeddings."""
    rows = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            batch = np.stack(
                [model.preprocess(image, True) for image in crops[start : start + batch_size]]
            )
            rows.append(model(torch.from_numpy(batch).to(device)).detach().cpu().numpy())
    return np.concatenate(rows)


def measure_validation_loss(model, pairs: Path) -> dict:
    """Canonical ``val`` split through the existing ``evaluate_epoch`` (batch16, margin 0.3)."""
    from age_gap.training.finetune import ImagePairDataset
    from scripts.run_facenet_validation_cuda_v2 import evaluate_epoch

    dataset = ImagePairDataset(
        split="val",
        pairs_file=str(pairs),
        preprocess=lambda image: model.preprocess(image, True),
        crops_dir="faces",
    )
    if not len(dataset):
        raise ValueError("canonical validation split is empty")
    return evaluate_epoch(
        model, dataset, device="cuda", batch_size=BATCH_SIZE, margin=MARGIN
    )


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def readiness(bindings, *, weight: Path, weight_record: dict):
    """Verify every declared native input/output and the shared frozen init before CUDA."""
    natives, selected, seeds = [], {}, []
    for binding in bindings:
        directory = run_directory(binding)
        native = verified(native_manifest_path(directory), NATIVE_EXPERIMENT)
        cell = native_contract(native, directory, weight_record)
        seed = cell["seed"]
        if seed in selected:
            raise ValueError("duplicate native seed binding")
        selected[seed] = cell["checkpoint"]
        natives.append(native)
        seeds.append(seed)
    require_seed_set(seeds)
    shared = require_shared_init(natives, weight_record)
    return natives, selected, shared


def run(args) -> Path:
    if args.out.exists():
        raise FileExistsError("fresh evaluation output required; never overwrite results")
    bindings = canonical_bindings(args.bindings)
    weight = require_pretrained_weight(resolve_facenet_casia_weight())
    weight_record = validate_weight_record(weight, inventory_weight_record(INVENTORY))
    natives, selected, shared = readiness(bindings, weight=weight, weight_record=weight_record)
    if not args.execute:
        print(
            f"dry-run: {OUTPUT_EXPERIMENT} {len(selected)} completed weak fixed8 v2 cell(s) "
            f"verified; no CUDA probe, no model load, no inference, no output"
        )
        return Path()

    if not torch.cuda.is_available():  # explicit re-check; no CPU fallback exists
        raise RuntimeError("CUDA required; no CPU fallback for the weak fixed8 evaluator")

    from scripts.reevaluate_fgnet_metrics_v2 import protocol

    torch.set_num_threads(1)
    cache = FGNET_CACHE
    inputs = build_inputs(bindings=bindings, natives=natives, weight=weight, cache=cache)
    before = snapshot(inputs)

    with np.load(cache, allow_pickle=False) as data:
        crops, subjects, ages = data["crops"], data["subjects"], data["ages"]
    require_full_pool(crops, subjects)
    left, right, labels, gaps, coverage = protocol(subjects, ages)
    if not coverage_complete(coverage):
        raise ValueError("canonical protocol coverage accounting differs")

    # The frozen initialisation actually reconstructed in memory must match the declared
    # shared digest before the first neural forward, not merely across metadata.
    frozen = build_architecture()
    frozen_digest = state_dict_digest(frozen.state_dict())
    if frozen_digest != shared["model_init_state_sha256"]:
        raise ValueError("reconstructed frozen initialisation differs from the declared digest")
    frozen = frozen.to("cuda").eval()

    keys = ["frozen", *[f"tuned_s{seed}" for seed in sorted(selected)]]
    lookup = {f"tuned_s{seed}": selected[seed] for seed in selected}
    embeddings, scores, measurements = {}, {}, {}
    torch.cuda.reset_peak_memory_stats()
    for key in keys:
        if key == "frozen":
            model = frozen
        else:
            model = load_checkpoint_state(build_architecture(), lookup[key]).to("cuda").eval()
        vectors = validate_embeddings(embed_pool(model, crops, device="cuda"), len(crops))
        embeddings[key] = vectors
        scores[key] = np.sum(vectors[left] * vectors[right], axis=1)
        if args.measure_validation_loss:
            measurements[key] = validation_measurement(measure_validation_loss(model, NATIVE_PAIRS))
        if key == "frozen":
            del frozen
        del model
        torch.cuda.empty_cache()
        print(f"full FG-NET pool embedded: {key}", flush=True)

    def stratum(mask):
        return infer(
            {key: value[mask] for key, value in scores.items()},
            labels[mask],
            subjects[left][mask],
            subjects[right][mask],
        )

    result = {
        "experiment": OUTPUT_EXPERIMENT,
        "evaluator_version": PROPOSAL_VERSION,
        "bindings": [str(run_directory(binding)) for binding in bindings],
        "actual_cells": list(keys[1:]),
        "seeds": sorted(selected),
        "protocol": "original full-pool endpoint_age_matched; seed42/tolerance2",
        "overall": stratum(np.ones(len(labels), bool)),
        "large_gap_25plus": stratum(gaps >= 25),
        "coverage": coverage,
        "shared_initialization": shared,
        "n_crops": int(len(crops)),
        "validation_loss_measured": bool(args.measure_validation_loss),
        "validation_loss_diagnostic": measurements,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "execution_complete": True,
        "publication_ready": False,
        "limitation": (
            "individual fixed native last checkpoints; per-checkpoint subject bootstrap and "
            "LOO only; no ensemble, no seed-population interval, no legacy history, no causal "
            "or mechanism claim; validation-loss diagnostic is retrospective and not a "
            "recovered training trajectory"
        ),
    }

    if snapshot(inputs) != before:
        raise RuntimeError("evaluation inputs changed before output publication")

    private = args.out / "private"
    private.mkdir(parents=True)
    outputs = []
    for key, vectors in embeddings.items():
        path = private / f"embeddings_{key}.npz"
        np.savez_compressed(path, indices=np.arange(len(crops)), embeddings=vectors)
        outputs.append(path)
    stored = private / "scores.npz"
    np.savez_compressed(
        stored,
        **scores,
        labels=labels,
        left=left,
        right=right,
        subject_a=subjects[left],
        subject_b=subjects[right],
        source_gap=gaps,
    )
    outputs.append(stored)

    summary = args.out / NATIVE_SUMMARY_NAME
    summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = args.out / NATIVE_MANIFEST_NAME
    write_experiment_manifest(
        target,
        experiment=OUTPUT_EXPERIMENT,
        parameters=dict(
            device="cuda",
            threads=1,
            batch_size=BATCH_SIZE,
            protocol_seed=ROC_V2_PROTOCOL_SEED,
            tolerance=ROC_V2_TOLERANCE,
            n_boot=N_BOOT,
            bootstrap_seed=BOOTSTRAP_SEED,
            image_inference=True,
            evaluator_version=PROPOSAL_VERSION,
            native_experiment=NATIVE_EXPERIMENT,
            pretrained_weight_name=CASIA_WEIGHT_NAME,
            actual_cells=list(keys[1:]),
            validation_loss_measured=bool(args.measure_validation_loss),
        ),
        metrics=aggregate_metrics(result),
        inputs=inputs,
        outputs=[summary, *outputs],
    )
    validate_written_inputs(target, before)

    # The published metrics must be the full aggregate result, equal to the standalone summary.
    published = json.loads(target.read_text(encoding="utf-8"))["metrics"]
    if not strict_equal(published, result):
        target.unlink()
        raise RuntimeError("manifest metrics are not the full aggregate result")
    return target


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bindings",
        type=Path,
        nargs="+",
        required=True,
        help="one or three completed facenet-fixed8-validation-loss-v2 run directories",
    )
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--measure-validation-loss", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as error:
        raise SystemExit(f"weak fixed8 evaluation refused: {type(error).__name__}") from None


if __name__ == "__main__":
    main()
