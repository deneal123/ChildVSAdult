"""Retrospective frozen-init vs selected-last-checkpoint internal validation loss.

CUDA-only measurement producer for ONE already-completed native fixed8 cell
(``adaface-fixed8-native-cuda-training-cell`` or ``facenet-fixed8-validation-loss-v2``).
It re-validates the native completion binding, reconstructs the *frozen pretrained init*
and applies the *native last checkpoint* with a strict weights-only load, then scores the
same declared validation rows twice (no optimizer, no shuffling, float32, unclipped
cosine scores) and reports global numerator/denominator plus a pair-level conditional
paired bootstrap CI.

Scope limits: this is a retrospective measurement of two states of one native cell. It
does NOT recover historical epoch losses, does NOT reselect a checkpoint, performs no new
external evaluation and makes no mechanism claim.

Importing this module must never probe CUDA, load a model or read the pairs file; all of
that happens inside :func:`run`. Without ``--execute`` the CLI is a pure dry run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest

EXPERIMENT = "fixed-checkpoint-validation-loss-cuda-v2"
ADA_EXPERIMENT = "adaface-fixed8-native-cuda-training-cell"
FACENET_EXPERIMENT = "facenet-fixed8-validation-loss-v2"
NATIVE_EXPERIMENTS = (ADA_EXPERIMENT, FACENET_EXPERIMENT)
COMPONENT_EXPERIMENT = "pair-contrastive-backbone-finetune"

# Native experiment -> (backbone, inventory key). Every accepted native cell must be an
# exact member of this table; nothing is inferred from the checkpoint contents.
NATIVE_BACKBONES = {
    ADA_EXPERIMENT: ("adaface_ir101", "adaface_ir101"),
    FACENET_EXPERIMENT: ("facenet", "facenet_casia"),
}
INVENTORY_KEYS = {"facenet": "facenet_casia", "adaface_ir101": "adaface_ir101"}
ADA_WEIGHT_REL = "models/adaface_ir101.pt"

FIXED_EPOCHS = 8
BATCH_SIZE = 16
MARGIN = 0.3
GAP_WEIGHT = 0.0
CROPS_DIR = "faces"
DEFAULT_CROPS_DIR = CROPS_DIR
AUC_REPRODUCTION_TOL = 1e-4
BOOTSTRAP_SEED = 42
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_ALPHA = 0.05
MIN_VALID_DRAW_FRACTION = 0.5
ALLOWED_SCOPES = ("head", "tail", "full")
ALLOWED_LRS = (1e-6, 1e-5)
ALLOWED_SEEDS = (42, 1, 2)
LIMITATION = (
    "retrospective measurement of two states of one native cell; no historical epoch "
    "recovery, no checkpoint reselection, no new external evaluation, no mechanism claim"
)


# --------------------------------------------------------------------------------------
# Device / seam wrappers (imported lazily so module import never probes CUDA)
# --------------------------------------------------------------------------------------
def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback for fixed-checkpoint validation")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA reported available but no device is visible")


def make_backbone(name: str, pretrained: bool = True):
    from age_gap.models.backbones import make_backbone as _make

    return _make(name, pretrained=pretrained)


def build_dataset(split: str, pairs_file: Path, model, crops_dir: str):
    from age_gap.training.finetune import ImagePairDataset, _bb_prep

    return ImagePairDataset(
        split=split,
        pairs_file=str(pairs_file),
        preprocess=_bb_prep(model),
        crops_dir=crops_dir,
    )


def crop_path(face_id: str, crops_dir: str) -> Path:
    from age_gap.training.finetune import _crop_path

    return _crop_path(face_id, crops_dir)


def infer_scores(model, dataset, *, device: str, batch_size: int, margin: float) -> dict:
    """Fixed no-shuffle float32 CUDA pass; raw cosine scores, no clipping, no optimizer."""
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    scores: list[Any] = []
    labels: list[Any] = []
    weights: list[Any] = []
    with torch.no_grad():
        for ta, tb, y, w in loader:
            za, zb = model(ta.to(device)), model(tb.to(device))
            cos = (za * zb).sum(dim=-1)
            scores.append(cos.detach().to(torch.float32).cpu())
            labels.append(y.detach().to(torch.float32).cpu())
            weights.append(w.detach().to(torch.float32).cpu())
    if not scores:
        raise RuntimeError("empty validation inference; no rows scored")
    return dict(
        scores=torch.cat(scores).numpy(),
        labels=torch.cat(labels).numpy(),
        weights=torch.cat(weights).numpy(),
        n_pairs=int(sum(t.numel() for t in scores)),
    )


def roc_auc(scores, labels) -> float:
    from age_gap.evaluation.metrics import roc_auc as _roc_auc

    return float(_roc_auc(np.asarray(scores), np.asarray(labels)))


def validate_written_inputs(target, expected) -> None:
    from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs as _validate

    return _validate(target, expected)


def verify_records(records) -> None:
    from scripts.run_restricted_matched_campaign import verify_records as _verify

    _verify(records)


def factory_weight_path(backbone_name: str) -> Path:
    """The exact file the real pretrained factory reads (no substitution allowed)."""
    if backbone_name == "facenet":
        from scripts.run_facenet_validation_cuda_v2 import resolve_facenet_casia_weight

        return Path(resolve_facenet_casia_weight())
    if backbone_name == "adaface_ir101":
        return Path(PROJECT_ROOT) / ADA_WEIGHT_REL
    raise ValueError(f"no known pretrained factory path for backbone {backbone_name!r}")


def check_device_binding(native: dict) -> None:
    """Require the measured environment to match the native GPU/torch/cudnn binding."""
    import torch

    params = native["parameters"]
    checks = {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": torch.cuda.get_device_name(),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    for key, actual in checks.items():
        declared = params.get(key)
        if declared is not None and declared != actual:
            raise RuntimeError(f"{key} differs from the native binding: {declared!r} != {actual!r}")


# --------------------------------------------------------------------------------------
# Small typed guards
# --------------------------------------------------------------------------------------
def _guard(params: dict, key: str, types, *, optional: bool = False):
    if key not in params or params[key] is None:
        if optional:
            return None
        raise ValueError(f"native parameter {key!r} missing")
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, types):
        raise ValueError(
            f"native parameter {key!r} has wrong type {type(value).__name__} (true typeguard)"
        )
    return value


def _guard_choice(params: dict, key: str, allowed, *, optional: bool = False):
    value = _guard(params, key, (int, float), optional=optional)
    if value is None:
        return None
    if not any(value == candidate for candidate in allowed):
        raise ValueError(f"native parameter {key!r}={value!r} outside {allowed!r}")
    return value


def _guard_int_choice(params: dict, key: str, allowed, *, optional: bool = False):
    """Strict integer choice: rejects floats (``42.0``) and bools (``True``)."""
    value = _guard(params, key, int, optional=optional)
    if value is None:
        return None
    if not any(value == candidate for candidate in allowed):
        raise ValueError(f"native parameter {key!r}={value!r} outside {allowed!r}")
    return value


def _finite(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} is not numeric: {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} is not finite: {value!r}")
    return out


# --------------------------------------------------------------------------------------
# Native contract / ancestry binding
# --------------------------------------------------------------------------------------
def state_dict_digest(state: dict) -> str:
    """Byte-exact reuse of the canonical native initialisation digest (no local variant)."""
    from scripts.run_facenet_validation_cuda_v2 import state_dict_digest as _canonical

    return _canonical(state)


def classify_native(native: dict) -> str:
    experiment = native.get("experiment")
    if experiment not in NATIVE_EXPERIMENTS:
        raise ValueError(f"unsupported native experiment {experiment!r}; fixed8 CUDA cell required")
    return experiment


def native_contract(native: dict) -> tuple[dict, list]:
    """Re-validate the frozen fixed8 last-checkpoint contract from the native manifest."""
    experiment = classify_native(native)
    backbone, inventory_key = NATIVE_BACKBONES[experiment]
    params = native.get("parameters")
    if not isinstance(params, dict):
        raise ValueError("native manifest has no parameter mapping")
    if params.get("backbone") != backbone:
        raise ValueError(
            f"native experiment {experiment!r} must carry backbone {backbone!r}, "
            f"got {params.get('backbone')!r}"
        )
    if INVENTORY_KEYS[backbone] != inventory_key:  # pragma: no cover - static consistency
        raise ValueError("inventory table drifted from the native experiment table")

    for key in ("epochs_requested", "epochs_executed", "selected_epoch", "batch_size"):
        value = _guard(params, key, int)
        if value != (FIXED_EPOCHS if key != "batch_size" else BATCH_SIZE):
            raise ValueError(f"native fixed8 contract differs at {key!r}")
    if _guard(params, "checkpoint_selection", str) != "last_epoch":
        raise ValueError("native fixed8 contract differs at 'checkpoint_selection'")
    if _guard(params, "batchnorm_policy", str) != "frozen_all":
        raise ValueError("native fixed8 contract differs at 'batchnorm_policy'")
    if _guard(params, "crops_dir", str) != CROPS_DIR:
        raise ValueError("native fixed8 contract differs at 'crops_dir'")
    if _guard(params, "device", str) != "cuda":
        raise ValueError("native binding device must be 'cuda'")
    _guard(params, "threads", int)
    if _guard(params, "margin", float) != MARGIN:
        raise ValueError("native fixed8 contract differs at 'margin'")
    if _guard(params, "gap_weight", float) != GAP_WEIGHT:
        raise ValueError("native fixed8 contract differs at 'gap_weight'")
    _guard_choice(params, "learning_rate", ALLOWED_LRS)
    scope = _guard(params, "trainable_scope", str)
    if scope not in ALLOWED_SCOPES:
        raise ValueError(f"native trainable_scope={scope!r} outside {ALLOWED_SCOPES!r}")
    _guard_int_choice(params, "seed", ALLOWED_SEEDS)
    _guard(params, "pairs_file", str)
    negative = _guard(params, "negative", str, optional=True)
    if negative is not None and negative not in ("random", "lookalike"):
        raise ValueError(f"native negative={negative!r} outside ('random', 'lookalike')")
    for key in ("torch_version", "cuda_version", "gpu_name"):
        _guard(params, key, str)

    metrics = native.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("training_complete") is not True:
        raise ValueError("native training_complete flag required")
    history = metrics.get("history")
    if not isinstance(history, list):
        raise ValueError("native history must be a list")
    if [row.get("epoch") for row in history] != list(range(1, FIXED_EPOCHS + 1)):
        raise ValueError("native must carry eight ordered completed epochs")
    for row in history:
        loss_key = "train_loss" if "train_loss" in row else "training_loss"
        for key in (loss_key, "validation_auc", "mean_gradient_norm", "epoch_seconds"):
            if key not in row:
                raise ValueError(f"native history lacks {key!r} at epoch {row.get('epoch')}")
            _finite(row[key], f"native {key} at epoch {row.get('epoch')}")
        if "validation_loss" in row:
            _finite(row["validation_loss"], f"native validation_loss at {row.get('epoch')}")
    return params, history


def _resolve(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def find_checkpoint(native: dict) -> Path:
    points = [p for p in (_resolve(r["path"]) for r in native["outputs"]) if p.suffix == ".pt"]
    if len(points) != 1:
        raise ValueError(f"exactly one native checkpoint .pt required; found {len(points)}")
    if not points[0].is_file():
        raise FileNotFoundError(points[0])
    return points[0]


def native_declared_row_count(native: dict) -> int | None:
    """Validate the caller-declared native val row count *before* any int coercion.

    The native manifest may (or may not) declare ``metrics.validation_pairs``. When it does,
    the raw JSON value must already be a genuine positive ``int`` (not a bool, float, string
    or numeric string): silently coercing ``"9158"`` or ``9158.4`` would let a mistyped or
    truncated declaration compare equal to a differently-sized enumeration.
    """
    metrics = native.get("metrics")
    if not isinstance(metrics, dict) or "validation_pairs" not in metrics:
        return None
    declared = metrics["validation_pairs"]
    if isinstance(declared, bool) or not isinstance(declared, int):
        raise ValueError(
            f"native declared validation_pairs must be a real int, got {type(declared).__name__}"
        )
    if declared <= 0:
        raise ValueError(f"native declared validation_pairs must be positive, got {declared}")
    return declared


def load_native(train_dir: Path | str) -> dict:
    manifest = Path(train_dir) / "summary.manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"native completion manifest missing: {manifest}")

    # Earliest possible provenance snapshot of the parent manifest itself, taken BEFORE the
    # file is parsed. It is carried verbatim (never re-derived later) so that any mutation of
    # the parent metadata (history/parameters) after parsing is detected at every stage.
    manifest_record = file_record(manifest)

    native = json.loads(manifest.read_text(encoding="utf-8"))
    classify_native(native)
    verify_records(native["inputs"] + native["outputs"])
    params, history = native_contract(native)
    declared_rows = native_declared_row_count(native)
    checkpoint = find_checkpoint(native)

    component = None
    if native["experiment"] == ADA_EXPERIMENT:
        from scripts.run_oriented_campaign import verified

        component_path = checkpoint.with_suffix(".manifest.json")
        # The component provenance must be an output the parent cell actually published.
        if file_record(component_path) not in native["outputs"]:
            raise ValueError("native cell does not declare the finetune component as its output")
        component = verified(component_path, COMPONENT_EXPERIMENT)
        verify_component_binding(component, checkpoint, params, parent_history=history)

    # The parent manifest must not have changed while it was being read/parsed/verified.
    if file_record(manifest) != manifest_record:
        raise RuntimeError("native completion manifest changed while being loaded")

    return dict(
        manifest=manifest,
        manifest_record=manifest_record,
        native=native,
        parameters=params,
        history=history,
        declared_rows=declared_rows,
        checkpoint=checkpoint,
        component=component,
        experiment=native["experiment"],
    )


COMPONENT_KEYS = (
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
    "seed",
)


def verify_component_binding(
    component: dict, checkpoint: Path, params: dict, *, parent_history: list | None = None
) -> None:
    """The finetune component must bind this exact checkpoint, budget and history."""
    if file_record(checkpoint) not in component["outputs"]:
        raise ValueError("component manifest does not bind the native checkpoint")
    cparams = component.get("parameters", {})
    for key in COMPONENT_KEYS:
        if cparams.get(key) != params.get(key):
            raise ValueError(f"component parameter {key!r} differs from the native cell")
    if Path(str(cparams.get("pairs_file", ""))).resolve() != Path(
        str(params.get("pairs_file", ""))
    ).resolve():
        raise ValueError("component pairs_file differs from the native cell")
    if file_record(_resolve(str(params["pairs_file"]))) not in component["inputs"]:
        raise ValueError("component manifest does not declare the pairs file it trained on")
    history = component.get("metrics", {}).get("history", [])
    if [row.get("epoch") for row in history] != list(range(1, FIXED_EPOCHS + 1)):
        raise ValueError("component manifest must carry eight ordered completed epochs")
    if parent_history is not None and history != parent_history:
        raise ValueError("component history differs from the parent cell history")


def bind_pretrained_weight(backbone_name: str, native: dict) -> dict:
    """Exact factory weight file must exist and match inventory + native input records.

    Nothing here may be satisfied by a bare SHA membership: the resolved factory path, the
    inventory record path, the full ``file_record`` (path+bytes+sha256) declared by the
    native manifest, and the inventory bytes/sha256 all have to agree.
    """
    key = INVENTORY_KEYS.get(backbone_name)
    if key is None:
        raise ValueError(f"no inventory key for backbone {backbone_name!r}")
    inventory = Path(PROJECT_ROOT) / "metrics/model_inventory.json"
    payload = json.loads(inventory.read_text(encoding="utf-8"))
    try:
        declared = payload["models"][key]["artifact"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"model inventory lacks {key} artifact: {exc}") from None
    if not isinstance(declared, dict) or not {"path", "bytes", "sha256"} <= set(declared):
        raise RuntimeError(f"model inventory {key} artifact record is incomplete")

    weight = Path(factory_weight_path(backbone_name))
    if not weight.is_file():
        raise FileNotFoundError(f"pretrained weight missing at {weight}; refusing implicit download")
    inventory_path = _resolve(declared["path"])
    if inventory_path.resolve() != weight.resolve():
        raise RuntimeError(
            f"inventory path {inventory_path} is not the exact factory path {weight} "
            "for the pretrained backbone"
        )
    record = file_record(weight)
    if record["bytes"] != declared["bytes"] or record["sha256"] != declared["sha256"]:
        raise RuntimeError("pretrained weight bytes/sha differ from model_inventory.json")

    native_inputs = native["native"]["inputs"]
    if record not in native_inputs:
        raise RuntimeError(
            "bound pretrained weight record (path+bytes+sha) is not a declared native input"
        )
    expected = native["parameters"].get("pretrained_weight_sha256")
    if expected is not None and expected != record["sha256"]:
        raise RuntimeError("native manifest pretrained weight sha differs from the bound file")
    return dict(path=str(weight), inventory_key=key, record=record, inventory_record=declared)


def declared_val_rows(pairs_file: Path, crops_dir: str) -> list:
    """Enumerate every declared val row; refuse any missing crop instead of dropping it."""
    from age_gap.common.io import read_jsonl
    from age_gap.common.schemas import Pair

    rows: list = []
    missing: list = []
    for raw in read_jsonl(pairs_file):
        pair = Pair.from_dict(raw)
        if pair.split != "val":
            continue
        ca, cb = crop_path(pair.face_a, crops_dir), crop_path(pair.face_b, crops_dir)
        if not (ca.is_file() and cb.is_file()):
            missing.append((pair.pair_id, str(ca), str(cb)))
        rows.append((pair, ca, cb))
    if missing:
        raise RuntimeError(
            f"{len(missing)} declared validation crops are missing; no row dropping allowed"
        )
    if not rows:
        raise ValueError("no declared validation rows")
    return rows


# --------------------------------------------------------------------------------------
# Ancestry: full native + component records, plus local code/crop coverage
# --------------------------------------------------------------------------------------
def build_ancestry_records(
    *,
    script,
    native: dict,
    component: dict | None,
    checkpoint,
    weight,
    pairs,
    rows,
    root=None,
    parent_manifest=None,
    parent_record: dict | None = None,
) -> list:
    """Full deduplicated ancestry: parent manifest snapshot, native/component records, code, crops.

    The declared upstream records are carried verbatim (not re-derived) so that a later
    re-hash can detect any mutation of the very bytes the native cell published. When the
    caller loaded the cell through :func:`load_native`, ``parent_record`` is the earliest
    snapshot of ``summary.manifest.json`` and ``parent_manifest`` is its path; both are added
    explicitly so that editing the parent metadata (history/parameters) is caught.
    """
    root = Path(root) if root is not None else Path(PROJECT_ROOT)
    records: list = []
    seen: dict[str, dict] = {}

    def add(record: dict) -> None:
        if not isinstance(record, dict) or not {"path", "bytes", "sha256"} <= set(record):
            raise ValueError(f"incomplete ancestry record: {record!r}")
        raw = Path(record["path"])
        key = str(raw if raw.is_absolute() else (root / raw).resolve())
        if key in seen:
            if seen[key] != record:
                raise RuntimeError(f"conflicting ancestry records for {record['path']!r}")
            return
        seen[key] = dict(record)
        records.append(dict(record))

    # Parent completion manifest itself: verbatim earliest snapshot, never re-derived.
    if parent_manifest is not None or parent_record is not None:
        if parent_manifest is None or parent_record is None:
            raise ValueError("parent_manifest and parent_record must be supplied together")
        if file_record(Path(parent_manifest)) != parent_record:
            raise RuntimeError("native completion manifest changed before ancestry snapshot")
        add(parent_record)

    for record in native["inputs"]:
        add(record)
    for record in native["outputs"]:
        add(record)
    if component is not None:
        for record in component["inputs"] + component["outputs"]:
            add(record)

    extras = [
        Path(script).resolve(),
        Path(checkpoint).resolve(),
        Path(weight["path"]).resolve(),
        (root / "metrics/model_inventory.json").resolve(),
        Path(pairs).resolve(),
        (root / "pyproject.toml").resolve(),
        (root / "uv.lock").resolve(),
        (root / "src/age_gap/settings/settings.toml").resolve(),
        *sorted((root / "src/age_gap").rglob("*.py")),
        (root / "scripts/run_strong_backbone_cuda_cell_v1.py").resolve(),
        (root / "scripts/run_facenet_validation_cuda_v2.py").resolve(),
        (root / "scripts/validation_pair_loss_v1.py").resolve(),
        (root / "scripts/run_oriented_campaign.py").resolve(),
        (root / "scripts/run_restricted_matched_campaign.py").resolve(),
        (root / "scripts/evaluate_oriented_cuda_v1.py").resolve(),
        *(crop for _pair, ca, cb in rows for crop in (Path(ca), Path(cb))),
    ]
    for path in extras:
        if not Path(path).is_file():
            raise FileNotFoundError(f"declared ancestry input missing: {path}")
        add(file_record(path))
    return records


def check_ancestry(records: list, stage: str) -> None:
    """Re-hash the whole ancestry and compare against the declared record snapshot."""
    try:
        verify_records(records)
    except ValueError as exc:
        raise RuntimeError(f"declared ancestry changed {stage}: {exc}") from None


# --------------------------------------------------------------------------------------
# Strict checkpoint loading + measurement math
# --------------------------------------------------------------------------------------
def load_checkpoint(checkpoint: Path) -> dict:
    """Weights-only load: the checkpoint is untrusted data, never pickled objects."""
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or "state_dict" not in saved:
        raise RuntimeError("checkpoint must be a mapping with a state_dict")
    return saved


def _history_rows(rows: list, label: str) -> list:
    if not isinstance(rows, list):
        raise RuntimeError(f"{label} history must be a list")
    if [row.get("epoch") for row in rows] != list(range(1, FIXED_EPOCHS + 1)):
        raise RuntimeError(f"{label} history must carry ordered epochs 1..{FIXED_EPOCHS}")
    for row in rows:
        loss_key = "train_loss" if "train_loss" in row else "training_loss"
        for key in (loss_key, "validation_auc", "mean_gradient_norm", "epoch_seconds"):
            if key not in row:
                raise RuntimeError(f"{label} history lacks {key!r}")
            try:
                _finite(row[key], f"{label} {key}")
            except ValueError as exc:
                raise RuntimeError(str(exc)) from None
    return rows


def verify_completed_checkpoint(saved: dict, *, parent_history: list) -> None:
    """Saved metadata must match the parent cell, not merely be independently plausible."""
    import torch

    if saved.get("selected_epoch") != FIXED_EPOCHS:
        raise RuntimeError(f"checkpoint selected_epoch must be {FIXED_EPOCHS}")
    if saved.get("checkpoint_selection") != "last_epoch":
        raise RuntimeError("checkpoint checkpoint_selection must be last_epoch")
    checkpoint_history = _history_rows(saved.get("history"), "checkpoint")
    if checkpoint_history != parent_history:
        raise RuntimeError("checkpoint history differs from the parent cell history")
    state = saved.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("checkpoint state_dict is empty")
    for name, tensor in state.items():
        if not isinstance(name, str):
            raise RuntimeError(f"checkpoint state_dict key {name!r} is not a string")
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"checkpoint state_dict entry {name!r} is not a tensor")
        if tensor.is_complex():
            raise RuntimeError(f"checkpoint tensor {name!r} has unsupported complex dtype")
        if not (tensor.is_floating_point() or tensor.dtype in (torch.int64, torch.int32,
                                                               torch.int16, torch.int8,
                                                               torch.uint8, torch.bool)):
            raise RuntimeError(f"checkpoint tensor {name!r} has unsupported dtype {tensor.dtype}")
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"non-finite checkpoint tensor: {name}")


def load_tuned_strict(model, state: dict) -> None:
    """Strict load onto the matching initialised architecture (never strict=False)."""
    model.load_state_dict(state, strict=True)
    model.eval()


def _dataset_crop_paths(dataset) -> list:
    items = getattr(dataset, "_items", None)
    if not items:
        raise RuntimeError(
            "validation dataset exposes no ordered _items; exact row coverage cannot be verified"
        )
    return [(Path(item[0]), Path(item[1])) for item in items]


def verify_dataset_coverage(dataset, rows: list) -> None:
    """Exact ordered crop paths, not just a matching row count."""
    got = _dataset_crop_paths(dataset)
    want = [(Path(ca), Path(cb)) for _pair, ca, cb in rows]
    if len(got) != len(want):
        raise RuntimeError(f"validation dataset dropped rows: {len(got)} of {len(want)} declared")
    for index, (observed, declared) in enumerate(zip(got, want, strict=True)):
        if (
            observed[0].resolve() != declared[0].resolve()
            or observed[1].resolve() != declared[1].resolve()
        ):
            raise RuntimeError(f"dataset row {index} crop paths differ from the declared val row")


def declared_labels_weights(rows: list) -> tuple[np.ndarray, np.ndarray]:
    from age_gap.training.dataset import _pair_weight

    labels = np.asarray([float(pair.label) for pair, _a, _b in rows], dtype=np.float32)
    weights = np.asarray(
        [_pair_weight(int(pair.label), pair.age_gap, GAP_WEIGHT) for pair, _a, _b in rows],
        dtype=np.float32,
    )
    return labels, weights


def verify_arm_vectors(produced: dict, rows: list, name: str) -> None:
    """Inference label/weight vectors must equal the declared rows, not merely each other."""
    want_labels, want_weights = declared_labels_weights(rows)
    got_labels = np.asarray(produced["labels"], dtype=np.float32)
    got_weights = np.asarray(produced["weights"], dtype=np.float32)
    if got_labels.shape != want_labels.shape or not np.array_equal(got_labels, want_labels):
        raise RuntimeError(f"{name} arm labels do not match the declared validation rows")
    if got_weights.shape != want_weights.shape or not np.allclose(
        got_weights, want_weights, rtol=0.0, atol=0.0
    ):
        raise RuntimeError(f"{name} arm weights do not match the declared validation rows")


def pair_loss_numpy(scores, labels, margin: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return labels * (1.0 - scores) + (1.0 - labels) * np.maximum(scores - margin, 0.0)


def measure_pair_loss(scores, labels, weights, margin: float) -> dict:
    """Canonical global pair-weighted numerator/denominator (never mean of batch means)."""
    from scripts.validation_pair_loss_v1 import measure

    result = measure(scores, labels, margin=margin, weights=weights)
    if not isinstance(result, dict):
        raise RuntimeError("canonical measure() did not return a mapping")
    numerator = _finite(result.get("numerator"), "measure numerator")
    denominator = _finite(result.get("denominator"), "measure denominator")
    loss = _finite(result.get("validation_loss"), "measure validation_loss")
    if denominator <= 0:
        raise RuntimeError("measure denominator must be positive")
    if not math.isclose(numerator / denominator, loss, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("measure numerator/denominator do not reproduce validation_loss")
    out = dict(result)
    out.update(numerator=numerator, denominator=denominator, validation_loss=loss)
    return out


def check_auc_reproduction(scores, labels, native_auc, tol: float = AUC_REPRODUCTION_TOL) -> float:
    native_auc = _finite(native_auc, "native validation AUC")
    tol = _finite(tol, "AUC reproduction tolerance")
    if not (0.0 <= native_auc <= 1.0):
        raise RuntimeError(f"native validation AUC out of range: {native_auc!r}")
    if not (0.0 < tol <= 1.0):
        raise RuntimeError(f"AUC reproduction tolerance out of range: {tol!r}")
    auc = roc_auc(scores, labels)
    if not math.isfinite(auc) or not (0.0 <= auc <= 1.0):
        raise RuntimeError(f"tuned validation AUC invalid: {auc!r}")
    if abs(auc - native_auc) > tol:
        raise RuntimeError(
            f"tuned valAUC {auc!r} does not reproduce native {native_auc!r} within {tol}"
        )
    return auc


def paired_bootstrap_ci(
    per_pair_frozen,
    per_pair_tuned,
    weights,
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
    alpha: float = BOOTSTRAP_ALPHA,
) -> dict:
    """Paired pair-level conditional bootstrap CI for delta = tuned_loss - frozen_loss.

    Resampling unit is the validation *pair*; both arms use the same indices. This is
    conditional on the observed fixed validation split and is NOT a subject/person or
    training-seed population interval. Zero-mass resamples are counted as unavailable
    rather than silently turning into NaN; too few valid draws refuse to report a CI.
    """
    per_frozen = np.asarray(per_pair_frozen, dtype=float)
    per_tuned = np.asarray(per_pair_tuned, dtype=float)
    mass = np.asarray(weights, dtype=float)
    if per_frozen.ndim != 1 or per_frozen.size == 0:
        raise ValueError("per-pair losses must be nonempty 1-D arrays")
    if not (per_frozen.shape == per_tuned.shape == mass.shape):
        raise ValueError("aligned per-pair losses and weights required")
    if not (np.isfinite(per_frozen).all() and np.isfinite(per_tuned).all()):
        raise ValueError("finite per-pair losses required")
    if not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("finite nonnegative weights required")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("bootstrap draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    if isinstance(alpha, bool) or not isinstance(alpha, float) or not math.isfinite(alpha):
        raise ValueError("bootstrap alpha must be a finite float")
    if not (0.0 < alpha < 1.0):
        raise ValueError("bootstrap alpha must be strictly between 0 and 1")

    total_mass = float(mass.sum())
    if not math.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("positive finite total weight required")

    n = per_frozen.size
    delta_observed = float(((per_tuned - per_frozen) * mass).sum() / total_mass)
    if not math.isfinite(delta_observed):
        raise ValueError("observed delta is not finite")
    frozen_loss = float((per_frozen * mass).sum() / total_mass)
    tuned_loss = float((per_tuned * mass).sum() / total_mass)

    rng = np.random.default_rng(seed)
    deltas = np.full(draws, np.nan)
    for index in range(draws):
        take = rng.integers(0, n, size=n)
        w = mass[take]
        denominator = float(w.sum())
        if not math.isfinite(denominator) or denominator <= 0:
            continue
        frozen_draw = float((per_frozen[take] * w).sum() / denominator)
        tuned_draw = float((per_tuned[take] * w).sum() / denominator)
        if not (math.isfinite(frozen_draw) and math.isfinite(tuned_draw)):
            continue
        deltas[index] = tuned_draw - frozen_draw

    valid = deltas[np.isfinite(deltas)]
    valid_draws = int(valid.size)
    unavailable = int(draws - valid_draws)
    minimum = max(1, int(math.ceil(draws * MIN_VALID_DRAW_FRACTION)))
    if valid_draws < minimum:
        raise RuntimeError(
            f"insufficient valid bootstrap draws ({valid_draws}/{draws}); refusing to report a CI"
        )
    low, high = np.percentile(valid, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return dict(
        requested_draws=draws,
        valid_draws=valid_draws,
        unavailable_draws=unavailable,
        valid_draw_fraction=valid_draws / draws,
        draws=draws,
        seed=seed,
        alpha=alpha,
        resampling_unit="pair",
        conditional="paired pair-level CI on the fixed validation split",
        not_a="subject/person or training-seed population interval",
        loss_frozen=frozen_loss,
        loss_tuned=tuned_loss,
        delta_observed=delta_observed,
        delta_mean=delta_observed,
        delta_bootstrap_mean=float(valid.mean()),
        delta_ci_low=float(low),
        delta_ci_high=float(high),
    )


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def command_line(args) -> list:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--train",
        str(args.train),
        "--out",
        str(args.out),
        "--execute",
    ]


def run(args) -> Path:
    import torch

    if args.out.exists():
        raise FileExistsError(f"fresh output directory required: {args.out}")

    native = load_native(args.train)
    params = native["parameters"]
    backbone_name = params["backbone"]
    weight = bind_pretrained_weight(backbone_name, native)
    require_cuda()
    check_device_binding(native["native"])

    pairs = Path(str(params["pairs_file"]))
    if not pairs.is_absolute():
        pairs = (PROJECT_ROOT / pairs).resolve()
    if not pairs.is_file():
        raise FileNotFoundError(f"declared native pair file missing: {pairs}")
    crops_dir = params["crops_dir"]
    rows = declared_val_rows(pairs, crops_dir)
    # Type/positivity already validated pre-coercion in load_native; compare verbatim here.
    expected_native = native.get("declared_rows")
    if expected_native is not None and expected_native != len(rows):
        raise RuntimeError(
            f"declared validation rows {len(rows)} != native validation_pairs {expected_native}"
        )

    # Snapshot the full ancestry BEFORE any model construction / preprocessing. The parent
    # completion manifest is included from the earliest snapshot taken during load_native.
    records = build_ancestry_records(
        script=Path(__file__),
        native=native["native"],
        component=native["component"],
        checkpoint=native["checkpoint"],
        weight=weight,
        pairs=pairs,
        rows=rows,
        parent_manifest=native["manifest"],
        parent_record=native["manifest_record"],
    )
    check_ancestry(records, "before model construction")

    saved = load_checkpoint(native["checkpoint"])
    verify_completed_checkpoint(saved, parent_history=native["history"])
    checkpoint_digest = state_dict_digest(saved["state_dict"])

    frozen = make_backbone(backbone_name, pretrained=True)
    init_digest = state_dict_digest(frozen.state_dict())
    native_init = params.get("model_init_state_sha256")
    if native_init is not None and native_init != init_digest:
        raise RuntimeError("re-initialised architecture differs from the native init digest")
    tuned = copy.deepcopy(frozen)
    load_tuned_strict(tuned, saved["state_dict"])

    dataset = build_dataset("val", pairs, frozen, crops_dir)
    verify_dataset_coverage(dataset, rows)
    check_ancestry(records, "after model load")

    device = "cuda"
    torch.set_num_threads(1)
    frozen = frozen.to(device).eval()
    tuned = tuned.to(device).eval()

    torch.cuda.reset_peak_memory_stats()
    frozen_out = infer_scores(frozen, dataset, device=device, batch_size=BATCH_SIZE, margin=MARGIN)
    tuned_out = infer_scores(tuned, dataset, device=device, batch_size=BATCH_SIZE, margin=MARGIN)
    torch.cuda.synchronize()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())

    for name, produced in (("frozen", frozen_out), ("tuned", tuned_out)):
        if produced["n_pairs"] != len(rows):
            raise RuntimeError(f"{name} arm scored {produced['n_pairs']} of {len(rows)} rows")
        verify_arm_vectors(produced, rows, name)
    if not np.array_equal(frozen_out["labels"], tuned_out["labels"]) or not np.allclose(
        frozen_out["weights"], tuned_out["weights"], rtol=0, atol=0
    ):
        raise RuntimeError("frozen and tuned arms did not use identical rows/weights")

    frozen_loss = measure_pair_loss(
        frozen_out["scores"], frozen_out["labels"], frozen_out["weights"], MARGIN
    )
    tuned_loss = measure_pair_loss(
        tuned_out["scores"], tuned_out["labels"], tuned_out["weights"], MARGIN
    )
    frozen_auc = roc_auc(frozen_out["scores"], frozen_out["labels"])
    if not math.isfinite(frozen_auc) or not (0.0 <= frozen_auc <= 1.0):
        raise RuntimeError(f"frozen validation AUC invalid: {frozen_auc!r}")
    native_auc = float(native["history"][-1]["validation_auc"])
    tuned_auc = check_auc_reproduction(tuned_out["scores"], tuned_out["labels"], native_auc)

    per_frozen = pair_loss_numpy(frozen_out["scores"], frozen_out["labels"], MARGIN)
    per_tuned = pair_loss_numpy(tuned_out["scores"], tuned_out["labels"], MARGIN)
    bootstrap = paired_bootstrap_ci(per_frozen, per_tuned, frozen_out["weights"])

    check_ancestry(records, "after inference")

    args.out.mkdir(parents=True)
    private_dir = args.out / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    npz = private_dir / "scores.npz"
    np.savez(
        npz,
        scores_frozen=np.asarray(frozen_out["scores"], dtype=np.float32),
        scores_tuned=np.asarray(tuned_out["scores"], dtype=np.float32),
        labels=np.asarray(frozen_out["labels"], dtype=np.int8),
        weights=np.asarray(frozen_out["weights"], dtype=np.float32),
        pair_ids=np.asarray([pair.pair_id for pair, _a, _b in rows], dtype=np.str_),
    )

    delta = float(tuned_loss["validation_loss"] - frozen_loss["validation_loss"])
    cudnn_version = torch.backends.cudnn.version()
    summary_payload = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "execution_complete": True,
        "evaluation_complete": False,
        "publication_ready": False,
        "native_experiment": native["experiment"],
        "backbone": backbone_name,
        "crops_dir": crops_dir,
        "margin": MARGIN,
        "batch_size": BATCH_SIZE,
        "gap_weight": GAP_WEIGHT,
        "epochs": {"requested": 8, "executed": 8, "selected": 8},
        "n_validation_rows_declared": len(rows),
        "n_validation_rows_native": expected_native,
        "n_validation_rows_scored": int(frozen_out["n_pairs"]),
        "validation_rows_dropped": 0,
        "frozen": {
            "numerator": frozen_loss["numerator"],
            "denominator": frozen_loss["denominator"],
            "validation_loss": frozen_loss["validation_loss"],
            "validation_auc": frozen_auc,
        },
        "tuned": {
            "numerator": tuned_loss["numerator"],
            "denominator": tuned_loss["denominator"],
            "validation_loss": tuned_loss["validation_loss"],
            "validation_auc": tuned_auc,
            "native_final_epoch_validation_auc": native_auc,
            "auc_abs_delta": abs(tuned_auc - native_auc),
        },
        "loss_delta_tuned_minus_frozen": delta,
        "paired_bootstrap_ci": bootstrap,
        "binding": {
            "pretrained_weight_sha256": weight["record"]["sha256"],
            "pretrained_weight_bytes": weight["record"]["bytes"],
            "pretrained_weight_path": weight["path"],
            "pretrained_weight_inventory_key": weight["inventory_key"],
            "model_init_state_sha256": init_digest,
            "native_model_init_state_sha256": native_init,
            "checkpoint_sha256": file_record(native["checkpoint"])["sha256"],
            "checkpoint_state_dict_sha256": checkpoint_digest,
            "native_seed": params.get("seed"),
            "native_learning_rate": params.get("learning_rate"),
            "native_trainable_scope": params.get("trainable_scope"),
            "native_batchnorm_policy": params.get("batchnorm_policy"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": cudnn_version,
            "gpu_name": torch.cuda.get_device_name(),
            "device": device,
            "threads": 1,
        },
        "peaks": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "privateness": {"raw_pair_scores": "private/scores.npz; never in this summary"},
        "scope": {
            "retrospective_measurement": True,
            "historical_epoch_loss_recovery": False,
            "checkpoint_reselection": False,
            "new_external_evaluation": False,
            "mechanism_claim": False,
        },
        "limitation": LIMITATION,
    }
    summary = args.out / "summary.json"
    summary.write_text(
        json.dumps(summary_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    target = args.out / "summary.manifest.json"
    try:
        write_experiment_manifest(
            target,
            experiment=EXPERIMENT,
            parameters={
                "backbone": backbone_name,
                "crops_dir": crops_dir,
                "margin": MARGIN,
                "batch_size": BATCH_SIZE,
                "gap_weight": GAP_WEIGHT,
                "epochs_requested": 8,
                "epochs_executed": 8,
                "selected_epoch": 8,
                "checkpoint_selection": "last_epoch",
                "batchnorm_policy": params.get("batchnorm_policy"),
                "trainable_scope": params.get("trainable_scope"),
                "learning_rate": params.get("learning_rate"),
                "seed": params.get("seed"),
                "negative": params.get("negative"),
                "train_experiment": native["experiment"],
                "pairs_file": str(pairs),
                "checkpoint": str(native["checkpoint"]),
                "component_manifest": (
                    str(native["checkpoint"].with_suffix(".manifest.json"))
                    if native["component"] is not None
                    else None
                ),
                "pretrained_weight_path": weight["path"],
                "pretrained_weight_sha256": weight["record"]["sha256"],
                "model_init_state_sha256": init_digest,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_alpha": BOOTSTRAP_ALPHA,
                "auc_reproduction_tol": AUC_REPRODUCTION_TOL,
                "device": device,
                "threads": 1,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cudnn_version": cudnn_version,
                "gpu_name": torch.cuda.get_device_name(),
                "native_seed": params.get("seed"),
                "command": " ".join(shlex.quote(part) for part in command_line(args)),
                "python_version": platform.python_version(),
                "ancestry_input_count": len(records),
            },
            metrics=summary_payload,
            inputs=[record["path"] for record in records],
            outputs=[summary, npz],
            command=command_line(args),
        )
        validate_written_inputs(target, records)
    except BaseException:
        # Never leave a stale "completed" marker from this invocation.
        if target.exists():
            target.unlink()
        raise
    return target


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="completed native run dir")
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            f"dry-run: {EXPERIMENT} train={args.train} out={args.out}; "
            "no CUDA probe, no model load, no data preflight, no completion manifest"
        )
        return
    print(run(args))


if __name__ == "__main__":
    main()
