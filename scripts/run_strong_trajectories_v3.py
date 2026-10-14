"""Versioned aggregation of accepted fixed8 native AdaFace histories.

Canonical bounded history aggregation. It never trains, evaluates, loads
tensors/weights/images or touches live outputs; it reads only declared
manifest/aggregate bytes and its own producing sources.

Recovery v3 closes seven holes found in v2:

1. The published manifest previously omitted every native record input because
   one ``seen`` set was shared between record targets and the extra input list.
   v3 builds the input declaration from the phase-1 snapshot keys and asserts
   that every alias, native source and output role appears exactly once.
2. Missing producing dependencies were silently filtered and ``sources=[]``
   bypassed them. v3 has a mandatory dependency set (own module, reducer,
   shared-byte helper, manifest, io and settings) resolved/contained/preflighted
   before any native read; extra dependencies add to it and can never replace it.
3. The fresh destination is canonicalized, contained and re-resolved so a
   retargeted output root fails closed.
4. A binding alias was resolved once and then forgotten. v3 preflights every
   binding alias, re-resolves the declared spelling before each read, and keeps
   the alias binding inside the shared-byte phases.
5. Input records were re-hashed after phase 2, so changed native bytes could be
   absorbed. v3 derives the expected records from the preserved phase-1 snapshot
   and requires exact equality plus a fresh post-write rehash.
6. The written manifest was compared as parsed arrays only. v3 strictly parses
   the published bytes and freshly hashes the published files.
7. History fields are grounded in the trainer SOURCE schema
   (``src/age_gap/training/finetune.py`` history append), not in any corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.shared_byte_integrity_v1 import (
    ByteIntegrityError,
    RecordSnapshot,
    audited_records,
    parse_manifest_strict,
    rehash_phase,
)
from scripts.strong_training_trajectory_v1 import summarize

__all__ = ["TrajectoryError", "aggregate", "main"]

NATIVE_EXPERIMENT = "adaface-fixed8-native-cuda-training-cell"
EXPERIMENT = "strong-native-training-trajectories-v3"
BACKBONE = "adaface_ir101"
EPOCHS = 8
FROZEN_BN = "frozen_all"
CROPS = "faces"
NEGATIVES = ("random", "lookalike")
SCOPES = ("head", "tail", "full")
LEARNING_RATES = (1e-6, 1e-5)
SEEDS = (42, 1, 2)
# Grounded in src/age_gap/training/finetune.py (single history.append dict):
#   epoch, training_loss, validation_auc, mean_gradient_norm, epoch_seconds
HISTORY_FIELDS = (
    "epoch",
    "training_loss",
    "validation_auc",
    "mean_gradient_norm",
    "epoch_seconds",
)
MEASUREMENTS = ("training_loss", "validation_auc", "mean_gradient_norm", "epoch_seconds")
NONNEGATIVE = ("training_loss", "mean_gradient_norm", "epoch_seconds")
OUTPUT_ROLES = frozenset({"summary.json", "checkpoint.pt", "checkpoint.manifest.json"})
RECORD_KEYS = frozenset({"path", "bytes", "sha256"})
# Mandatory producing dependencies. Always resolved/contained/preflighted; an
# extra ``dependencies`` argument can only add to this set. The relative set is
# fixed in code and checked against the dependency root, so no argument can
# shrink or bypass it.
MANDATORY_DEPENDENCIES = (
    "scripts/strong_training_trajectory_v1.py",
    "scripts/shared_byte_integrity_v1.py",
    "src/age_gap/common/manifest.py",
    "src/age_gap/common/io.py",
    "src/age_gap/settings/__init__.py",
    "src/age_gap/settings/config.py",
)


class TrajectoryError(Exception):
    """One rejected declaration/contract, worded so private paths never leak."""

    def __init__(self, stage: str, subject: str, detail: str) -> None:
        super().__init__(f"{stage}: {subject}: {detail}")
        self.stage = stage
        self.subject = subject
        self.detail = detail


def _fail(stage: str, subject: str, detail: str) -> None:
    raise TrajectoryError(stage, subject, detail)


def _canonical(path: object) -> Path | None:
    """Resolve without ever surfacing an OSError carrying a raw path."""
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Typed, fail-closed field gates (bool is never an int, NaN/Inf never a number).
# ---------------------------------------------------------------------------


def _int(value: object, subject: str, field: str, stage: str = "contract") -> int:
    if type(value) is not int:
        _fail(stage, subject, f"reason=type field={field} expected=int")
    return value


def _num(value: object, subject: str, field: str, stage: str = "contract") -> float:
    if type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _fail(stage, subject, f"reason=type field={field} expected=finite-number")
    return value


def _str(value: object, subject: str, field: str, stage: str = "contract") -> str:
    if type(value) is not str:
        _fail(stage, subject, f"reason=type field={field} expected=str")
    return value


def _obj(value: object, subject: str, field: str, stage: str = "contract") -> dict:
    if type(value) is not dict:
        _fail(stage, subject, f"reason=type field={field} expected=object")
    return value


def _require(condition: bool, stage: str, subject: str, detail: str) -> None:
    if not condition:
        _fail(stage, subject, detail)


# ---------------------------------------------------------------------------
# Mandatory producing dependencies (never bypassed, never silently filtered).
# ---------------------------------------------------------------------------


def _digest_file(path: Path) -> tuple[int, str]:
    import hashlib

    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _relative_record(path: Path, base: Path) -> dict:
    size, sha = _digest_file(path)
    return {"path": Path(path).relative_to(base).as_posix(), "bytes": size, "sha256": sha}


def _dependency_declarations(extra, *, base=None, own_module=None) -> list:
    """Resolve/contain/preflight every dependency before any read.

    The mandatory set is fixed in code and always enforced against the
    dependency root; ``extra`` can only add more files. Synthetic callers may
    inject ``base``/``own_module`` so the same checks bind to their fixture
    root; production binds them to the repository root.
    """
    root = Path(base) if base is not None else PROJECT_ROOT
    own = _canonical(own_module if own_module is not None else Path(__file__))
    if own is None or not own.is_file() or not own.is_relative_to(root):
        _fail("dependencies", "own-module", "reason=unresolvable")
    own_alias = Path(own_module if own_module is not None else __file__).absolute()
    paths = [(own_alias, own)]
    for index, relative in enumerate(MANDATORY_DEPENDENCIES):
        target = _canonical(root / relative)
        if target is None or not target.is_file() or not target.is_relative_to(root):
            _fail("dependencies", f"mandatory[{index}]", "reason=missing-or-outside-root")
        paths.append((root / relative, target))
    for index, extra_path in enumerate(extra or ()):
        alias = Path(extra_path)
        if not alias.is_absolute():
            alias = root / alias
        target = _canonical(alias)
        if target is None or not target.is_file() or not target.is_relative_to(root):
            _fail("dependencies", f"extra[{index}]", "reason=missing-or-outside-root")
        paths.append((alias, target))
    records = []
    seen = set()
    for index, (path, target) in enumerate(paths):
        token = os.path.normcase(str(path))
        if token in seen:
            continue
        seen.add(token)
        if _canonical(path) != target:
            _fail("dependencies", f"source[{index}]", "reason=alias-retargeted")
        records.append(_relative_record(path, root))
    return records


# ---------------------------------------------------------------------------
# Binding aliases (declared spelling preserved, re-resolved before each read).
# ---------------------------------------------------------------------------


class _BindingAlias:
    __slots__ = ("declared", "key", "target")

    def __init__(self, declared: Path, target: Path, base: Path) -> None:
        self.declared = declared
        self.target = target
        self.key = declared.relative_to(base).as_posix()


def _binding_alias(binding: object, base: Path, index: int) -> _BindingAlias:
    subject = f"binding[{index}]"
    declared = Path(binding)
    if not declared.is_absolute():
        declared = base / declared
    if not declared.is_relative_to(base):
        _fail("preflight", subject, "reason=binding-alias-outside-root")
    target = _canonical(declared)
    if target is None or not target.is_file() or not target.is_relative_to(base):
        _fail("preflight", subject, "reason=binding-outside-root-or-missing")
    return _BindingAlias(declared, target, base)


def _read_binding(alias: _BindingAlias, index: int) -> bytes:
    subject = f"binding[{index}]"
    current = _canonical(alias.declared)
    if current is None or str(current).lower() != str(alias.target).lower():
        _fail("preflight", subject, "reason=binding-retargeted")
    try:
        return alias.declared.read_bytes()
    except (OSError, RuntimeError, ValueError):
        _fail("preflight", subject, "reason=unreadable-binding")


# ---------------------------------------------------------------------------
# Native manifest declaration preflight (shape only; no byte streaming).
# ---------------------------------------------------------------------------


def _record_shape(record: object, subject: str) -> dict:
    if type(record) is not dict or set(record) != RECORD_KEYS:
        _fail("preflight", subject, "reason=record-shape")
    return record


def _leaf(record: dict) -> str:
    return str(record.get("path", "")).replace("\\", "/").rsplit("/", 1)[-1]


def _preflight_manifest(data: object, index: int) -> tuple[list, list]:
    subject = f"binding[{index}]"
    if type(data) is not dict:
        _fail("preflight", subject, "reason=manifest-shape")
    _require(type(data.get("schema_version")) is int and data["schema_version"] == 1,
             "preflight", subject, "reason=schema-version")
    _require(data.get("experiment") == NATIVE_EXPERIMENT, "preflight", subject,
             "reason=wrong-experiment")
    inputs = data.get("inputs")
    outputs = data.get("outputs")
    _require(type(inputs) is list and bool(inputs), "preflight", subject, "reason=inputs-shape")
    _require(type(outputs) is list and len(outputs) == 3, "preflight", subject,
             "reason=outputs-shape")
    for position, record in enumerate(inputs):
        _record_shape(record, f"{subject}.input[{position}]")
    leaves = set()
    for position, record in enumerate(outputs):
        _record_shape(record, f"{subject}.output[{position}]")
        leaves.add(_leaf(record))
    _require(leaves == OUTPUT_ROLES, "preflight", subject, "reason=output-roles")
    _require("parameters" in data and "metrics" in data, "preflight", subject,
             "reason=missing-parameters-or-metrics")
    return inputs, outputs


# ---------------------------------------------------------------------------
# Native fixed8 contract and ordered history validation.
# ---------------------------------------------------------------------------


def _validate_history(history: object, subject: str) -> list:
    if type(history) is not list or len(history) != EPOCHS:
        _fail("history", subject, "reason=history-shape")
    for position, row in enumerate(history):
        rsub = f"{subject}.history[{position}]"
        if type(row) is not dict or set(row) != set(HISTORY_FIELDS):
            _fail("history", rsub, "reason=history-row-shape")
        epoch = _int(row["epoch"], rsub, "epoch", "history")
        if epoch != position + 1:
            _fail("history", rsub, "reason=epoch-order")
        for key in MEASUREMENTS:
            _num(row[key], rsub, key, "history")
        if not 0.0 <= row["validation_auc"] <= 1.0:
            _fail("history", rsub, "reason=auc-range")
        for key in NONNEGATIVE:
            if row[key] < 0:
                _fail("history", rsub, f"reason=negative field={key}")
    return history


def _validate_contract(data: dict, index: int) -> tuple[dict, list]:
    subject = f"binding[{index}]"
    params = _obj(data.get("parameters"), subject, "parameters")
    metrics = _obj(data.get("metrics"), subject, "metrics")

    if "validation_loss" in params or "validation_loss" in metrics:
        _fail("contract", subject, "reason=fabricated-validation-loss")

    _require(_str(params.get("backbone"), subject, "backbone") == BACKBONE,
             "contract", subject, "reason=contract field=backbone")
    _require(_int(params.get("epochs_requested"), subject, "epochs_requested") == EPOCHS,
             "contract", subject, "reason=contract field=epochs_requested")
    _require(_int(params.get("epochs_executed"), subject, "epochs_executed") == EPOCHS,
             "contract", subject, "reason=contract field=epochs_executed")
    _require(_int(params.get("selected_epoch"), subject, "selected_epoch") == EPOCHS,
             "contract", subject, "reason=contract field=selected_epoch")
    _require(_str(params.get("checkpoint_selection"), subject, "checkpoint_selection") == "last_epoch",
             "contract", subject, "reason=contract field=checkpoint_selection")
    _require(_str(params.get("batchnorm_policy"), subject, "batchnorm_policy") == FROZEN_BN,
             "contract", subject, "reason=contract field=batchnorm_policy")
    _require(_int(params.get("batch_size"), subject, "batch_size") == 16,
             "contract", subject, "reason=contract field=batch_size")
    _require(_num(params.get("margin"), subject, "margin") == 0.3,
             "contract", subject, "reason=contract field=margin")
    _require(_num(params.get("gap_weight"), subject, "gap_weight") == 0.0,
             "contract", subject, "reason=contract field=gap_weight")
    _require(_str(params.get("crops_dir"), subject, "crops_dir") == CROPS,
             "contract", subject, "reason=contract field=crops_dir")
    _require(_str(params.get("device"), subject, "device") == "cuda",
             "contract", subject, "reason=contract field=device")
    _require(_int(params.get("threads"), subject, "threads") == 1,
             "contract", subject, "reason=contract field=threads")
    _str(params.get("pairs_file"), subject, "pairs_file")

    negative = _str(params.get("negative"), subject, "negative")
    scope = _str(params.get("trainable_scope"), subject, "trainable_scope")
    learning_rate = _num(params.get("learning_rate"), subject, "learning_rate")
    seed = _int(params.get("seed"), subject, "seed")
    _require(negative in NEGATIVES, "contract", subject, "reason=cell field=negative")
    _require(scope in SCOPES, "contract", subject, "reason=cell field=trainable_scope")
    _require(learning_rate in LEARNING_RATES, "contract", subject, "reason=cell field=learning_rate")
    _require(seed in SEEDS, "contract", subject, "reason=cell field=seed")

    _require(metrics.get("training_complete") is True, "contract", subject,
             "reason=metric field=training_complete")
    _require(metrics.get("evaluation_complete") is False, "contract", subject,
             "reason=metric field=evaluation_complete")
    _require(metrics.get("publication_ready") is False, "contract", subject,
             "reason=metric field=publication_ready")
    for key in ("train_pairs", "validation_pairs", "unique_crop_files"):
        if _int(metrics.get(key), subject, key) <= 0:
            _fail("contract", subject, f"reason=metric field={key}")
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        if _num(metrics.get(key), subject, key) < 0:
            _fail("contract", subject, f"reason=metric field={key}")

    history = _validate_history(metrics.get("history"), subject)
    cell = dict(negative=negative, trainable_scope=scope, learning_rate=learning_rate, seed=seed)
    return cell, history


# ---------------------------------------------------------------------------
# Curves (native metrics only) and reuse of the immutable descriptive reducer.
# ---------------------------------------------------------------------------


def _curve(history: list, key: str) -> dict:
    values = [row[key] for row in history]
    return {
        "first": values[0],
        "last": values[-1],
        "change": values[-1] - values[0],
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def _cell(cell: dict, history: list) -> dict:
    return {
        "version": "strong-training-trajectory-v3",
        "cell": cell,
        "epochs": EPOCHS,
        "selected_epoch": EPOCHS,
        "checkpoint_selection": "last_epoch",
        "curves": {key: _curve(history, key) for key in MEASUREMENTS},
        "validation_loss": None,
        "validation_loss_status": (
            "not collected by the fixed8 native trainer; cannot be inferred from validation AUC"
        ),
        "historical_loss_available": False,
        "checkpoint_reselection": False,
        "seed_population_claim": False,
        "mechanism_claim": False,
        "publication_clearance": False,
        "descriptive": summarize(history),
        "limitations": [
            "curves are descriptive recorded native metrics only",
            "peak AUC is diagnostic, never a selected checkpoint",
            "no historical loss, no seed-population, mechanism or publication clearance",
        ],
    }


# ---------------------------------------------------------------------------
# Destination guard (fresh, contained, retarget-resistant).
# ---------------------------------------------------------------------------


def _prepare_destination(destination: Path, base: Path, *, resolver=None) -> Path:
    resolve = resolver or (lambda item: Path(item).resolve())
    if destination.exists():
        raise FileExistsError("fresh destination required")
    parent = destination.parent
    parent_target = _canonical(parent)
    if parent_target is None or not parent_target.is_dir() or not parent_target.is_relative_to(base):
        _fail("output", "destination", "reason=outside-root-or-missing-parent")
    expected = parent_target / destination.name
    if expected.exists():
        raise FileExistsError("fresh destination required")
    destination.mkdir(parents=False, exist_ok=False)
    actual = resolve(destination)
    if _canonical(actual) is None or Path(actual) != expected or not Path(actual).is_dir():
        _fail("output", "destination", "reason=retargeted-after-creation")
    return Path(actual)


# ---------------------------------------------------------------------------
# Input snapshot -> manifest record assertions.
# ---------------------------------------------------------------------------


def _snapshot_value(native: RecordSnapshot, deps: RecordSnapshot, base: Path,
                    dep_base: Path, path: Path) -> tuple[int, str] | None:
    resolved = _canonical(path)
    if resolved is None:
        return None
    for snapshot, root in ((native, base), (deps, dep_base)):
        try:
            key = os.path.normcase(str(resolved.relative_to(root))).replace("\\", "/")
        except ValueError:
            continue
        record = snapshot.get(key)
        if record is not None:
            return record["bytes"], record["sha256"]
    return None


def _expected_inputs(input_paths: list, native: RecordSnapshot, deps: RecordSnapshot,
                     base: Path, dep_base: Path) -> list:
    expected = []
    seen = set()
    for path in input_paths:
        resolved = _canonical(path)
        token = str(resolved).lower() if resolved else str(path).lower()
        if token in seen:
            _fail("publication", "inputs", "reason=duplicate-input-path")
        seen.add(token)
        snapshot = _snapshot_value(native, deps, base, dep_base, path)
        if snapshot is None:
            _fail("publication", "inputs", "reason=input-not-in-phase1-snapshot")
        try:
            current = file_record(path)
        except (OSError, RuntimeError, ValueError):
            _fail("publication", "inputs", "reason=unreadable-declared-input")
        if (current["bytes"], current["sha256"]) != snapshot:
            _fail("publication", "inputs", "reason=input-changed-after-phase1")
        expected.append({"path": current["path"], "bytes": snapshot[0], "sha256": snapshot[1]})
    return expected


def _expected_outputs(summary: Path) -> list:
    try:
        return [file_record(summary)]
    except (OSError, RuntimeError, ValueError):
        _fail("publication", "outputs", "reason=unreadable-summary")


def _exact(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_exact(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _validate_published_manifest(target: Path, expected_inputs: list, expected_outputs: list,
                                 output_paths: list) -> None:
    """Strict-parse the published bytes and require exact input/output equality."""
    try:
        raw = Path(target).read_bytes()
    except (OSError, RuntimeError, ValueError):
        _fail("publication", "manifest", "reason=unreadable-manifest")
    data, _size, _sha = parse_manifest_strict(raw)
    if type(data) is not dict:
        _fail("publication", "manifest", "reason=manifest-shape")
    if not _exact(data.get("inputs"), expected_inputs):
        _fail("publication", "manifest", "reason=inputs-changed-during-publication")
    if not _exact(data.get("outputs"), expected_outputs):
        _fail("publication", "manifest", "reason=outputs-changed-during-publication")
    if data.get("experiment") != EXPERIMENT:
        _fail("publication", "manifest", "reason=wrong-experiment")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        _fail("publication", "manifest", "reason=schema-version")
    # Freshly hash the final published output files (no parsed-array shortcut).
    final = []
    for path in output_paths:
        try:
            final.append(file_record(path))
        except (OSError, RuntimeError, ValueError):
            _fail("publication", "outputs", "reason=published-output-unreadable")
    if final != expected_outputs:
        _fail("publication", "outputs", "reason=published-output-changed-after-write")
    summary_data, _, _ = parse_manifest_strict(output_paths[0].read_bytes())
    if not _exact(data.get("metrics"), summary_data):
        _fail("publication", "manifest", "reason=metrics-differ-from-summary")


def _guard_destination(alias: Path, target: Path, base: Path) -> None:
    if (_canonical(alias) != target or _canonical(target) != target
            or not target.is_relative_to(base) or not target.is_dir()):
        _fail("output", "destination", "reason=retargeted-before-or-after-write")


def _discard_own_marker(target: Path, destination: Path, base: Path) -> None:
    # Only a completion marker in this invocation's fresh, still-bound directory.
    if (_canonical(destination) == destination and destination.is_relative_to(base)
            and _canonical(target) == target):
        target.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def aggregate(binding_paths, out, *, root=PROJECT_ROOT, dependencies=None,
              dependency_root=None, own_module=None, destination_resolver=None) -> int:
    """Aggregate accepted fixed8 native histories and write one fresh manifest."""
    binding_list = list(binding_paths or ())
    if not binding_list:
        _fail("preflight", "bindings", "reason=empty-bindings")
    base = Path(root).resolve()
    dep_root = Path(dependency_root).resolve() if dependency_root is not None else PROJECT_ROOT
    destination = Path(out)
    if destination.exists():
        raise FileExistsError("fresh destination required")

    # --- mandatory producing dependencies preflight (before any native read) --
    dep_records = _dependency_declarations(dependencies, base=dep_root, own_module=own_module)
    dep_declarations = [dict(record) for record in dep_records]
    dep_before = audited_records(dep_declarations, dep_root)

    # --- binding alias preflight, then re-resolved reads --------------------
    aliases = [_binding_alias(binding, base, index) for index, binding in enumerate(binding_list)]
    keys = [str(alias.target).lower() for alias in aliases]
    if len(set(keys)) != len(keys):
        _fail("preflight", "bindings", "reason=duplicate-binding")

    declarations: list = []
    parsed: list = []
    for index, alias in enumerate(aliases):
        raw = _read_binding(alias, index)
        data, size, sha = parse_manifest_strict(raw)
        inputs, outputs = _preflight_manifest(data, index)
        declarations.append({"path": alias.key, "bytes": size, "sha256": sha})
        declarations.extend(inputs)
        declarations.extend(outputs)
        parsed.append((index, data, inputs, outputs))

    # --- full-byte phase 1: aliases rejected, shared targets streamed once --
    native_before = audited_records(declarations, base)

    # --- contract + history validation, curve derivation -------------------
    cells: dict = {}
    for index, data, _inputs, _outputs in parsed:
        cell, history = _validate_contract(data, index)
        key = f"{cell['negative']}|{cell['trainable_scope']}|lr{cell['learning_rate']:g}|seed{cell['seed']}"
        if key in cells:
            _fail("aggregation", f"binding[{index}]", "reason=duplicate-native-cell")
        cells[key] = _cell(cell, history)

    # --- full-byte phase 2: fresh re-stream, alias binding preserved -------
    native_after = rehash_phase(native_before, base)
    dep_after = rehash_phase(dep_before, dep_root)

    # --- input declaration from the phase-1 snapshot (no omission, no adopt) -
    input_paths: list = []
    seen: set = set()
    for key in native_after:
        path = base / key
        token = str(path).lower()
        if token not in seen:
            seen.add(token)
            input_paths.append(path)
    for key in dep_after:
        path = dep_root / key
        token = str(path).lower()
        if token not in seen:
            seen.add(token)
            input_paths.append(path)
    input_paths.sort(key=lambda item: str(item).lower())
    expected_inputs = _expected_inputs(input_paths, native_before, dep_before, base, dep_root)

    metrics = dict(
        execution_complete=True,
        publication_ready=False,
        cell_count=len(cells),
        cells=cells,
        validation_loss=None,
        validation_loss_status=(
            "not collected by the fixed8 native trainer; cannot be inferred from validation AUC"
        ),
        historical_loss_available=False,
        checkpoint_reselection=False,
        seed_population_claim=False,
        mechanism_claim=False,
        publication_clearance=False,
        distinct_byte_targets=len(input_paths),
        limitation=(
            "descriptive completed native histories only; the matrix is not established as "
            "complete; no validation loss, seed-population, mechanism or publication decision"
        ),
    )

    resolved_destination = _prepare_destination(destination, base, resolver=destination_resolver)
    summary = resolved_destination / "summary.json"
    target = resolved_destination / "summary.manifest.json"

    _guard_destination(destination, resolved_destination, base)
    summary.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    expected_outputs = _expected_outputs(summary)

    try:
        _guard_destination(destination, resolved_destination, base)
        write_experiment_manifest(
            target,
            experiment=EXPERIMENT,
            parameters=dict(
                epochs=EPOCHS,
                checkpoint_selection="last_epoch",
                aggregation="separate-source-and-native-fresh-byte-phases",
                binding_count=len(aliases),
                distinct_byte_targets=len(input_paths),
            ),
            metrics=metrics,
            inputs=input_paths,
            outputs=[summary],
        )
        # --- fresh complete after-write guard from the phase-1 snapshots ----
        rehash_phase(native_before, base)
        rehash_phase(dep_before, dep_root)
        _guard_destination(destination, resolved_destination, base)
        _validate_published_manifest(target, expected_inputs, expected_outputs, [summary])
    except Exception:
        _discard_own_marker(target, resolved_destination, base)
        raise
    return len(cells)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    # Containment boundary; defaults to the canonical project root. It never
    # disables the containment check and exists so synthetic fixtures bound the
    # same check inside their own directory.
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        count = aggregate(args.bindings, args.out, root=args.root)
    except FileExistsError:
        print(json.dumps({"ok": False, "stage": "output", "subject": "out",
                          "detail": "reason=fresh-destination-required"}))
        return 1
    except ByteIntegrityError as error:
        print(json.dumps({"ok": False, "stage": error.stage, "subject": error.subject,
                          "detail": error.detail}))
        return 1
    except TrajectoryError as error:
        print(json.dumps({"ok": False, "stage": error.stage, "subject": error.subject,
                          "detail": error.detail}))
        return 1
    except Exception:  # noqa: BLE001 - never surface a private path or digest
        print(json.dumps({"ok": False, "stage": "internal", "subject": "runner",
                          "detail": "reason=fail-closed"}))
        return 1
    print(json.dumps({"ok": True, "cells": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
