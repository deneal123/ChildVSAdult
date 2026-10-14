"""Retrospective frozen-init vs selected-last-checkpoint validation loss, v5 (versioned).

Root-reviewed hardening of ``scripts/measure_checkpoint_validation_cuda_v2.py``.
Measurement semantics are unchanged: CUDA-only, frozen pretrained init vs the native last
checkpoint, no optimizer, no shuffling, float32, unclipped cosine, global pair-weighted
loss and a pair-level conditional bootstrap CI. No claim that historical epoch losses are
recovered and no mechanism claim. Import and execution without ``--execute`` never probe CUDA, load a model
or read data.

v5 keeps every v4 hardening and closes the six remaining v4 review holes exactly:

1. ``freeze_ancestry`` freezes the *original declared* byte claims (schema/containment
   preflight, then one fresh stream per physical target). It re-resolves every original
   alias spelling before AND after the initial hashing, rejects a retarget, and verifies
   the declared claims against the freshly read bytes, so new bytes are never silently
   blessed. External (outside-root) targets are streamed and checked at freeze; nothing is
   cached across phases.
2. The parsed model inventory record is pinned once (exact bytes + exact parsed payload +
   exact declared artifact record) and that pinned object — not a re-parsed copy — is
   carried into ancestry and the outside-root allowlist. The finetune component is pinned
   the same way (alias spelling + parsed record).
3. The outside-root allowlist is built from the pinned strict artifact record; bytes are a
   strict non-bool int and the digest is a strict 64-lowerhex string, never a cast.
4. The owned output is guarded by a frozen lexical spelling, a fresh resolved path and a
   directory lease (device+inode). Before and after every write (``mkdir`` for the owned
   dir and the private dir, ``np.savez``, the summary write, the manifest write) both the
   directory lease and each exact output leaf are re-guarded. A retarget fails with an
   empty external directory untouched, and cleanup only ever removes owned physical paths.
5. Publication compares inputs and outputs with recursive type-exact equality against the
   earliest claims, verifies the manifest ``experiment`` and ``schema_version``, and
   re-hashes the whole frozen ancestry after the writes.
6. ``command_line`` records the true OS-level invocation from ``sys.orig_argv`` when
   CPython exposes it (module/alias invocations included) and otherwise reconstructs it,
   describing which happened honestly.

This version imports immutable v2 for measurement math. Native CUDA measurement remains pending.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
import platform
import re
import shlex
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

import scripts.measure_checkpoint_validation_cuda_v2 as v2
import scripts.shared_byte_integrity_v2 as sbi
from age_gap.common.io import PROJECT_ROOT, data_path
from age_gap.common.manifest import SCHEMA_VERSION, file_record, write_experiment_manifest

EXPERIMENT = "fixed-checkpoint-validation-loss-cuda-v5"
LIMITATION = (
    "retrospective measurement of two states of one native cell; no historical epoch "
    "recovery, no checkpoint reselection, no new external evaluation, no mechanism claim"
)
CLOSURE_GUARD = 200
_HEX = frozenset("0123456789abcdef")
_RECORD_KEYS = {"path", "bytes", "sha256"}
_REASON_RE = re.compile(r"reason=[a-z0-9-]+")
_OUTPUT_LEAVES = ("summary.json", "summary.manifest.json", "private", "private/scores.npz")

MANDATORY_SEEDS = (
    "scripts.measure_checkpoint_validation_cuda_v5",
    "scripts.measure_checkpoint_validation_cuda_v2",
    "scripts.shared_byte_integrity_v2",
    "scripts.validation_pair_loss_v1",
    "age_gap.common.manifest",
    "age_gap.common.io",
    "age_gap.settings",
)
SCRIPT_SEEDS = (
    "scripts.run_facenet_validation_cuda_v2",
    "scripts.run_oriented_campaign",
    "scripts.run_restricted_matched_campaign",
    "scripts.evaluate_oriented_cuda_v1",
)


def _fail(stage: str, subject: str, reason: str) -> None:
    raise sbi.ByteIntegrityError(stage, subject, reason)


def _target(spelling: str, base: Path) -> Path:
    raw = Path(spelling)
    return (raw if raw.is_absolute() else base / raw).resolve()


# --------------------------------------------------------------------------------------
# Strict numerics, recursive type-exact equality, strict JSON and strict records
# --------------------------------------------------------------------------------------
def _finite_number(value: Any, name: str) -> float:
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real int/float, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _strict_epoch(value: Any, allowed: tuple[int, ...]) -> int:
    if type(value) is not int:
        raise ValueError(f"history epoch must be a strict int, got {type(value).__name__}")
    if value not in allowed:
        raise ValueError("history epoch outside the fixed8 range")
    return value


def _type_exact_equal(left: Any, right: Any, context: str) -> None:
    """Recursive type-exact equality; nested bool never equals nested int."""
    if type(left) is not type(right):
        raise ValueError(f"{context} differs type-exactly")
    if isinstance(left, dict):
        if set(left) != set(right):
            raise ValueError(f"{context} keys differ")
        for key in left:
            _type_exact_equal(left[key], right[key], f"{context}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"{context} length differs")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _type_exact_equal(a, b, f"{context}[{index}]")
    elif left != right:
        raise ValueError(f"{context} value differs")


def load_strict_json(path: Path | str) -> tuple[Any, dict]:
    """Parse strict JSON and snapshot the SAME bytes (duplicate keys / non-finite rejected)."""
    raw = Path(path).read_bytes()
    data, size, digest = sbi.parse_manifest_strict(raw)
    record = file_record(path)
    if record["bytes"] != size or record["sha256"] != digest:
        raise RuntimeError("strict-parsed bytes changed before snapshot")
    return data, record


def strict_record(record: Any, subject: str) -> dict:
    """A JSON record is accepted only with the exact schema and strict scalar types."""
    if type(record) is not dict or set(record) != _RECORD_KEYS:
        _fail("record", subject, "reason=record-shape")
    path, size, sha = record["path"], record["bytes"], record["sha256"]
    if type(path) is not str:
        _fail("record", subject, "reason=path-type")
    if type(size) is bool or type(size) is not int or size < 0:
        _fail("record", subject, "reason=bytes-int")
    if type(sha) is not str or len(sha) != 64 or any(c not in _HEX for c in sha):
        _fail("record", subject, "reason=sha256-hex")
    return {"path": path, "bytes": size, "sha256": sha}


def strict_records(records: Any, stage: str) -> tuple[dict, ...]:
    if type(records) is not list:
        _fail(stage, "records", "reason=type expected=list")
    return tuple(strict_record(record, f"record[{index}]") for index, record in enumerate(records))


def _records_type_exact(written: Any, expected: list, context: str) -> None:
    if type(written) is not list or len(written) != len(expected):
        raise RuntimeError(f"{context} count differs from the earliest claims")
    for index, (a, b) in enumerate(zip(written, expected, strict=True)):
        _type_exact_equal(
            strict_record(a, f"{context}[{index}]"),
            strict_record(b, f"{context}[{index}]"),
            f"{context}[{index}]",
        )


# --------------------------------------------------------------------------------------
# Native contract / component binding (strict)
# --------------------------------------------------------------------------------------
def native_contract_v5(native: dict) -> tuple[dict, list]:
    params, history = v2.native_contract(native)
    for field in ("epochs_requested", "epochs_executed", "selected_epoch", "batch_size", "seed"):
        if type(params.get(field)) is not int:
            raise ValueError(f"native {field} must be a strict integer")
    for field in ("learning_rate", "margin", "gap_weight"):
        _finite_number(params.get(field), f"native {field}")
    for row in history:
        _strict_epoch(row.get("epoch"), tuple(range(1, v2.FIXED_EPOCHS + 1)))
        loss_key = "train_loss" if "train_loss" in row else "training_loss"
        for key in (loss_key, "validation_auc", "mean_gradient_norm", "epoch_seconds"):
            if key not in row:
                raise ValueError(f"native history lacks {key!r}")
            _finite_number(row[key], f"native {key}")
        if "validation_loss" in row:
            _finite_number(row["validation_loss"], "native validation_loss")
    return params, history


def verify_component_binding_v5(
    component: dict, checkpoint: Path, params: dict, *, parent_history: list | None = None
) -> None:
    v2.verify_component_binding(component, checkpoint, params, parent_history=parent_history)
    _type_exact_equal(component.get("parameters", {}), params, "component parameters")
    if parent_history is not None:
        _type_exact_equal(
            component.get("metrics", {}).get("history", []), parent_history, "component history"
        )


# --------------------------------------------------------------------------------------
# Factory path + PINNED inventory (exact bytes, exact parsed payload, exact record)
# --------------------------------------------------------------------------------------
def factory_weight_path_v5(backbone_name: str) -> Path:
    if backbone_name not in v2.INVENTORY_KEYS:
        raise ValueError(f"no known pretrained factory path for backbone {backbone_name!r}")
    if backbone_name == "adaface_ir101":
        return Path(data_path("models_dir", "adaface_ir101.pt"))
    return v2.factory_weight_path(backbone_name)


@dataclass(frozen=True)
class PinnedJson:
    """A parsed JSON document pinned by exact bytes, exact payload and one strict record."""

    path: str
    record: dict
    parsed: Any


def pin_json(path: Path | str) -> PinnedJson:
    parsed, record = load_strict_json(path)
    return PinnedJson(str(path), record, parsed)


def verify_pinned_json(pin: PinnedJson, stage: str) -> None:
    """Re-read and re-parse; the pinned payload must be unchanged, type-exactly."""
    raw = Path(pin.path).read_bytes()
    if file_record(pin.path) != pin.record:
        _fail(stage, "pinned-json", "reason=bytes-changed")
    parsed, size, digest = sbi.parse_manifest_strict(raw)
    if size != pin.record["bytes"] or digest != pin.record["sha256"]:
        _fail(stage, "pinned-json", "reason=digest-changed")
    _type_exact_equal(parsed, pin.parsed, f"pinned-json {stage}")


@dataclass(frozen=True)
class InventoryPin:
    """The single pinned model inventory: exact bytes, exact key record, exact target."""

    pin: PinnedJson
    key: str
    declared: dict
    target: Path


def pin_inventory(path: Path | str, key: str) -> InventoryPin:
    pin = pin_json(path)
    if type(pin.parsed) is not dict or type(pin.parsed.get("models")) is not dict:
        raise RuntimeError("model inventory must be a JSON object with a models map")
    model = pin.parsed["models"].get(key)
    if type(model) is not dict or "artifact" not in model:
        raise RuntimeError(f"model inventory has no artifact for key {key}")
    declared = strict_record(model["artifact"], f"inventory.{key}.artifact")
    target = _target(declared["path"], Path(PROJECT_ROOT).resolve())
    return InventoryPin(pin, key, declared, target)


def allowed_outside_weight(pin: InventoryPin) -> dict[str, tuple]:
    """The ONLY outside-root targets permitted: the exact pinned factory weight record."""
    return {os.path.normcase(str(pin.target)): (pin.declared["bytes"], pin.declared["sha256"])}


def bind_pretrained_weight_v5(backbone_name: str, native: dict, pin: InventoryPin) -> dict:
    expected_key = v2.INVENTORY_KEYS.get(backbone_name)
    if expected_key is None:
        raise ValueError(f"no inventory key for backbone {backbone_name!r}")
    if pin.key != expected_key:
        raise RuntimeError("pinned inventory key does not match the native backbone")
    weight = factory_weight_path_v5(backbone_name)
    if not weight.is_file():
        raise FileNotFoundError(
            f"pretrained weight missing; refusing implicit download ({pin.key})"
        )
    if pin.target != weight.resolve():
        raise RuntimeError(f"inventory path is not the exact factory path for {pin.key}")
    record = file_record(weight)
    if record["bytes"] != pin.declared["bytes"] or record["sha256"] != pin.declared["sha256"]:
        raise RuntimeError("pretrained weight bytes/sha differ from the pinned inventory")
    if record not in native["native"]["inputs"]:
        raise RuntimeError("bound pretrained weight record is not a declared native input")
    return {"path": str(weight.resolve()), "record": record, "inventory_key": pin.key}


# --------------------------------------------------------------------------------------
# Frozen ancestry: original alias guard + fresh per-physical stream, no blessed new bytes
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FrozenAncestry:
    claims: tuple  # ORIGINAL declared records, carried verbatim
    aliases: tuple  # every distinct original spelling -> physical key
    physical: tuple  # physical key -> (bytes, sha256) actually read at freeze
    contained_snapshot: Any  # shared_byte_integrity_v2 snapshot of the contained subset
    outside_targets: tuple  # (physical key, bytes, sha256) for the bound external weight


def _guard_aliases(aliases: tuple, base: Path, stage: str) -> None:
    """Fresh whole re-resolution of every ORIGINAL declared spelling, no cache."""
    for index, (spelling, key) in enumerate(aliases):
        if os.path.normcase(str(_target(spelling, base))) != key:
            _fail(stage, f"alias[{index}]", "reason=declared-target-changed")


def _containment_preflight(claims: tuple, base: Path, allowed: dict) -> None:
    for index, record in enumerate(claims):
        target = _target(record["path"], base)
        if not target.is_relative_to(base):
            key = os.path.normcase(str(target))
            if key not in allowed:
                _fail("preflight", f"record[{index}]", "reason=path-escapes-root")
            if (record["bytes"], record["sha256"]) != allowed[key]:
                _fail("preflight", f"record[{index}]", "reason=outside-not-bound")


def freeze_ancestry(records: list, root, allowed_outside: dict | None = None) -> FrozenAncestry:
    """Freeze the original declared claims, then verify them against freshly read bytes."""
    base = Path(root).resolve()
    allowed = dict(allowed_outside or {})
    claims = strict_records(records, "preflight")
    _containment_preflight(claims, base, allowed)

    aliases: list[tuple[str, str]] = []
    seen_spellings: set[str] = set()
    targets: dict[str, tuple] = {}
    for record in claims:
        spelling = str(record["path"])
        key = os.path.normcase(str(_target(spelling, base)))
        declared = (record["bytes"], record["sha256"])
        if key in targets and targets[key] != declared:
            _fail("freeze", "record", "reason=aliases-disagree-on-bytes-or-hash")
        targets.setdefault(key, declared)
        if spelling not in seen_spellings:
            aliases.append((spelling, key))
            seen_spellings.add(spelling)
    aliases_t = tuple(aliases)
    _guard_aliases(aliases_t, base, "before initial hashing")

    contained: list[dict] = []
    contained_seen: set[str] = set()
    outside: list[tuple] = []
    for key, (size, sha) in targets.items():
        target = Path(key)
        if target.is_relative_to(base):
            norm = os.path.normcase(target.relative_to(base).as_posix()).replace("\\", "/")
            if norm not in contained_seen:
                contained_seen.add(norm)
                contained.append({"path": norm, "bytes": size, "sha256": sha})
        else:
            outside.append((key, size, sha))

    # External targets are checked AT FREEZE, against the pinned allowlist only.
    for key, size, sha in outside:
        try:
            actual = sbi._stream_file(Path(key))
        except (OSError, RuntimeError, ValueError):
            _fail("freeze", "outside-target", "reason=unreadable")
        if actual != (size, sha):
            _fail("freeze", "outside-target", "reason=byte-drift")

    snapshot = sbi.audited_records(contained, base) if contained else None
    _guard_aliases(aliases_t, base, "after initial hashing")

    physical = []
    for key, (size, sha) in sorted(targets.items()):
        if not Path(key).is_relative_to(base):
            physical.append((key, size, sha))
            continue
        norm = os.path.normcase(Path(key).relative_to(base).as_posix()).replace("\\", "/")
        actual = snapshot[norm]
        actual = (actual["bytes"], actual["sha256"])
        if actual != (size, sha):  # never silently bless new bytes
            _fail("freeze", "record", "reason=byte-drift")
        physical.append((key, actual[0], actual[1]))
    return FrozenAncestry(claims, aliases_t, tuple(physical), snapshot, tuple(outside))


def fresh_check(frozen: FrozenAncestry, root, stage: str) -> None:
    """Re-resolve every original alias and re-stream every physical target (no cache)."""
    base = Path(root).resolve()
    _guard_aliases(frozen.aliases, base, stage)
    if frozen.contained_snapshot is not None:
        sbi.rehash_phase(frozen.contained_snapshot, base)
    for key, size, sha in frozen.outside_targets:
        try:
            actual = sbi._stream_file(Path(key))
        except (OSError, RuntimeError, ValueError):
            _fail(stage, "outside-target", "reason=unreadable")
        if actual != (size, sha):
            _fail(stage, "outside-target", "reason=byte-drift")
    _guard_aliases(frozen.aliases, base, stage)


def declared_val_rows_v5(pairs_file: Path, crops_dir: str) -> tuple[list, dict]:
    earliest = file_record(pairs_file)
    rows = v2.declared_val_rows(pairs_file, crops_dir)
    if file_record(pairs_file) != earliest:
        raise RuntimeError("pairs file changed during validation-row enumeration")
    return rows, earliest


def verify_pairs(pairs_file: Path, earliest: dict, stage: str) -> None:
    if file_record(pairs_file) != earliest:
        raise RuntimeError(f"pairs file changed {stage}")


# --------------------------------------------------------------------------------------
# Producing-source closure (AST, mandatory seeds, fail-closed guard)
# --------------------------------------------------------------------------------------
def _module_file(name: str, root: Path) -> Path | None:
    if name.startswith("scripts."):
        candidate = root / "scripts" / (name.split(".", 1)[1].replace(".", "/") + ".py")
        return candidate if candidate.is_file() else None
    if name == "age_gap" or name.startswith("age_gap."):
        module = root / "src" / "age_gap"
        for part in name.split(".")[1:]:
            module = module / part
        package = module / "__init__.py"
        if package.is_file():
            return package
        flat = Path(str(module) + ".py")
        return flat if flat.is_file() else None
    return None


def _py_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        raise RuntimeError(f"cannot parse producing source {path.name}") from exc
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return found


def _resolve_seed(name: str, root: Path, self_seed: str) -> Path | None:
    path = _module_file(name, root)
    if path is None and name == self_seed:
        return Path(__file__)  # proposal not yet promoted into scripts/
    return path


def producing_source_paths(root: Path | None = None) -> list[Path]:
    root = Path(root) if root is not None else Path(PROJECT_ROOT)
    self_seed = f"scripts.{Path(__file__).stem}"
    mandatory: set[Path] = set()
    for seed in MANDATORY_SEEDS:
        path = _resolve_seed(seed, root, self_seed)
        if path is None:
            raise FileNotFoundError(f"mandatory producing-source seed missing: {seed}")
        mandatory.add(path.resolve())
    queue = list(MANDATORY_SEEDS) + list(SCRIPT_SEEDS)
    seen: set[str] = set()
    files: list[Path] = []
    while queue:
        if len(files) >= CLOSURE_GUARD:
            raise RuntimeError("producing-source closure guard exhausted")
        name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        path = _resolve_seed(name, root, self_seed)
        if path is None:
            continue
        files.append(path.resolve())
        for imported in _py_imports(path):
            if imported.startswith(("scripts.", "age_gap")) and imported not in seen:
                queue.append(imported)
    if not mandatory <= set(files):
        raise RuntimeError("producing-source closure dropped a mandatory seed")
    files.extend(sorted((root / "src" / "age_gap").rglob("*.py")))
    return files


def build_ancestry_records_v5(
    *,
    script,
    native,
    component,
    checkpoint,
    weight,
    pairs_record,
    rows,
    parent_record,
    inventory_pin: InventoryPin,
    root=None,
) -> list:
    root = Path(root) if root is not None else Path(PROJECT_ROOT)
    records: list = [dict(parent_record)]
    records += [dict(r) for r in native["inputs"]]
    records += [dict(r) for r in native["outputs"]]
    if component is not None:
        records += [dict(r) for r in component["inputs"] + component["outputs"]]
    extras = [
        Path(script).resolve(),
        Path(checkpoint).resolve(),
        Path(weight["path"]).resolve(),
        Path(inventory_pin.pin.path).resolve(),
        (root / "pyproject.toml").resolve(),
        (root / "uv.lock").resolve(),
        (root / "src/age_gap/settings/settings.toml").resolve(),
        *producing_source_paths(root),
        *(crop for _pair, ca, cb in rows for crop in (Path(ca), Path(cb))),
    ]
    for path in extras:
        if not Path(path).is_file():
            raise FileNotFoundError(f"declared ancestry input missing: {Path(path).name}")
        records.append(file_record(path))
    # The inventory is carried from the PIN, never re-hashed into a fresh claim.
    records = [
        r
        for r in records
        if os.path.normcase(str(_target(r["path"], root)))
        != os.path.normcase(str(_target(inventory_pin.pin.record["path"], root)))
    ]
    records.append(dict(inventory_pin.pin.record))
    records.append(dict(pairs_record))  # earliest pinned pairs record, verbatim
    return records


# --------------------------------------------------------------------------------------
# Output guard: frozen spelling + fresh resolution + directory lease + exact leaves
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OutputGuard:
    lexical: str
    resolved: str
    parent_resolved: str
    lease: tuple | None  # (st_dev, st_ino) of the owned directory once created
    ancestry_targets: tuple


def make_output_guard(out_lexical, root, ancestry_targets: list | tuple = ()) -> OutputGuard:
    base = Path(root).resolve()
    lexical = Path(out_lexical)
    resolved = lexical.resolve()
    if not (resolved == base or base in resolved.parents):
        raise ValueError("output directory must be contained in the project root")
    targets = tuple(os.path.normcase(str(Path(t))) for t in ancestry_targets)
    for target in targets:
        if (
            resolved == Path(target)
            or Path(target) in resolved.parents
            or resolved in Path(target).parents
        ):
            raise ValueError("output directory aliases a declared ancestry target")
    return OutputGuard(str(lexical), str(resolved), str(resolved.parent.resolve()), None, targets)


def _lease(path: Path) -> tuple:
    info = os.stat(path)
    return (info.st_dev, info.st_ino)


def guard_output(guard: OutputGuard, frozen: FrozenAncestry | None, stage: str) -> Path:
    lexical = Path(guard.lexical)
    if str(lexical.resolve()) != guard.resolved:
        raise RuntimeError(f"output directory retargeted {stage}")
    if str(lexical.parent.resolve()) != guard.parent_resolved:
        raise RuntimeError(f"output parent retargeted {stage}")
    resolved = Path(guard.resolved)
    if guard.lease is not None and (not resolved.is_dir() or _lease(resolved) != guard.lease):
        raise RuntimeError(f"output directory lease broken {stage}")
    for target in guard.ancestry_targets:
        if (
            resolved == Path(target)
            or Path(target) in resolved.parents
            or resolved in Path(target).parents
        ):
            raise RuntimeError(f"output aliases a declared ancestry target {stage}")
    if frozen is not None:
        for key, _size, _sha in frozen.physical:
            if (
                resolved == Path(key)
                or Path(key) in resolved.parents
                or resolved in Path(key).parents
            ):
                raise RuntimeError(f"output aliases a declared ancestry target {stage}")
    return resolved


def guard_leaf(guard: OutputGuard, leaf, stage: str) -> Path:
    """Each exact output leaf must stay inside the leased owned directory."""
    owned = Path(guard.resolved)
    lexical = Path(leaf).absolute()
    resolved = lexical.resolve()
    # Do not resolve the allowance through an attacker-controlled leaf/junction.
    # The exact expected physical paths were fixed by the owned directory lease.
    allowed = {os.path.normcase(str(owned / name)) for name in _OUTPUT_LEAVES}
    if (
        os.path.normcase(str(lexical)) not in allowed
        or resolved != lexical
        or not resolved.is_relative_to(owned)
    ):
        raise RuntimeError(f"output leaf outside the owned directory {stage}")
    if guard.lease is not None and (not owned.is_dir() or _lease(owned) != guard.lease):
        raise RuntimeError(f"output directory lease broken {stage}")
    if resolved.is_file() and resolved.stat().st_nlink != 1:
        raise RuntimeError(f"output leaf has additional hard links {stage}")
    return resolved


def lease_output(guard: OutputGuard, frozen: FrozenAncestry | None) -> OutputGuard:
    """Create the owned directory, then freeze its lease BEFORE any leaf is written."""
    owned = guard_output(guard, frozen, "before create")
    owned.mkdir(parents=True)
    leased = replace(guard, lease=_lease(owned))
    guard_output(leased, frozen, "after create")
    return leased


# --------------------------------------------------------------------------------------
# Publication: earliest claims, deep equality, experiment/schema, exact closure
# --------------------------------------------------------------------------------------
def _write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def publish_manifest(
    target: Path,
    *,
    summary: Path,
    npz: Path,
    frozen: FrozenAncestry,
    summary_payload: dict,
    params_out: dict,
    guard: OutputGuard,
    root,
    cli: list,
    claimed_outputs: tuple,
) -> None:
    base = Path(root).resolve()
    guard_output(guard, frozen, "before publish")
    guard_leaf(guard, target, "before publish")
    summary_record, npz_record = claimed_outputs
    try:
        write_experiment_manifest(
            target,
            experiment=EXPERIMENT,
            parameters=params_out,
            metrics=summary_payload,
            inputs=[str(_target(claim["path"], base)) for claim in frozen.claims],
            outputs=[summary, npz],
            command=cli,
        )
        guard_output(guard, frozen, "after manifest write")
        guard_leaf(guard, target, "after manifest write")
        written, _rec = load_strict_json(target)
        if written.get("experiment") != EXPERIMENT:
            raise RuntimeError("written manifest has the wrong experiment")
        if (
            type(written.get("schema_version")) is not int
            or written["schema_version"] != SCHEMA_VERSION
        ):
            raise RuntimeError("written manifest has the wrong schema version")
        _type_exact_equal(written.get("parameters"), params_out, "manifest parameters")
        _type_exact_equal(written.get("metrics"), summary_payload, "manifest metrics")
        _records_type_exact(written.get("inputs"), list(frozen.claims), "manifest inputs")
        _records_type_exact(
            written.get("outputs"), [summary_record, npz_record], "manifest outputs"
        )
        # Post-write output mutation: bytes must still match the earliest claims.
        if file_record(summary) != summary_record or file_record(npz) != npz_record:
            raise RuntimeError("written output bytes changed after creation")
        guard_output(guard, frozen, "before summary read")
        guard_leaf(guard, summary, "before summary read")
        summary_written, _srec = load_strict_json(summary)
        _type_exact_equal(summary_written, summary_payload, "summary")
        fresh_check(frozen, base, "after writes")
        guard_output(guard, frozen, "after writes")
    except BaseException:
        try:
            guard_output(guard, frozen, "cleanup")  # guard BEFORE touching any leaf
            owned_leaf = guard_leaf(guard, summary, "cleanup")
            if owned_leaf.exists():
                failed = dict(summary_payload)
                failed["status"] = "failed-write-check"
                failed["execution_complete"] = False
                _write_summary(owned_leaf, failed)
            manifest_leaf = guard_leaf(guard, target, "cleanup")
            if manifest_leaf.exists():
                manifest_leaf.unlink()
        except Exception:
            pass
        raise


def command_line(args) -> tuple[list, dict]:
    """The true OS-level argv when CPython exposes it, otherwise an honest reconstruction."""
    reconstructed = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--train",
        str(args.train),
        "--out",
        str(args.out),
        "--execute",
    ]
    true_argv = list(getattr(sys, "orig_argv", None) or [])
    if true_argv and Path(true_argv[0]).stem.lower().startswith("python"):
        return true_argv, {
            "command_source": "sys.orig_argv",
            "command_reconstructed": False,
            "command_argv": true_argv,
        }
    return reconstructed, {
        "command_source": "reconstructed",
        "command_reconstructed": True,
        "command_argv": reconstructed,
    }


# --------------------------------------------------------------------------------------
# Orchestration (measurement semantics unchanged)
# --------------------------------------------------------------------------------------
def run(args) -> Path:
    import torch

    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    root = Path(PROJECT_ROOT).resolve()

    native = load_native_v5(args.train, root)
    params = native["parameters"]
    backbone_name = params["backbone"]
    weight = bind_pretrained_weight_v5(backbone_name, native, native["inventory_pin"])
    v2.require_cuda()
    v2.check_device_binding(native["native"])

    pairs = Path(str(params["pairs_file"]))
    if not pairs.is_absolute():
        pairs = (root / pairs).resolve()
    if not pairs.is_file():
        raise FileNotFoundError("declared native pair file missing")
    rows, pairs_record = declared_val_rows_v5(pairs, params["crops_dir"])
    expected_native = native.get("declared_rows")
    if expected_native is not None and expected_native != len(rows):
        raise RuntimeError("declared validation rows differ from native validation_pairs")

    records = build_ancestry_records_v5(
        script=Path(__file__),
        native=native["native"],
        component=native["component"],
        checkpoint=native["checkpoint"],
        weight=weight,
        pairs_record=pairs_record,
        rows=rows,
        parent_record=native["manifest_record"],
        inventory_pin=native["inventory_pin"],
        root=root,
    )
    frozen = freeze_ancestry(records, root, allowed_outside_weight(native["inventory_pin"]))
    verify_pairs(pairs, pairs_record, "before ancestry")
    verify_native_pins(native, "before model construction")

    # Output guard is frozen from the lexical spelling + the resolved fresh directory. The
    # directory is NOT created until the lease step just before the first write.
    owned_targets = [claim["path"] for claim in frozen.claims]
    guard = make_output_guard(args.out, root, owned_targets)

    saved = v2.load_checkpoint(native["checkpoint"])
    v2.verify_completed_checkpoint(saved, parent_history=native["history"])
    if type(saved.get("selected_epoch")) is not int or saved["selected_epoch"] != v2.FIXED_EPOCHS:
        raise RuntimeError("checkpoint selected_epoch must be a strict int 8")
    _type_exact_equal(saved.get("history"), native["history"], "checkpoint history")
    checkpoint_digest = v2.state_dict_digest(saved["state_dict"])
    frozen_model = v2.make_backbone(backbone_name, pretrained=True)
    init_digest = v2.state_dict_digest(frozen_model.state_dict())
    native_init = params.get("model_init_state_sha256")
    if native_init is not None and native_init != init_digest:
        raise RuntimeError("re-initialised architecture differs from the native init digest")
    tuned = copy.deepcopy(frozen_model)
    v2.load_tuned_strict(tuned, saved["state_dict"])
    dataset = v2.build_dataset("val", pairs, frozen_model, params["crops_dir"])
    v2.verify_dataset_coverage(dataset, rows)
    verify_pairs(pairs, pairs_record, "before inference")
    fresh_check(frozen, root, "before inference")
    verify_native_pins(native, "before inference")

    device = "cuda"
    torch.set_num_threads(1)
    frozen_model, tuned = frozen_model.to(device).eval(), tuned.to(device).eval()
    torch.cuda.reset_peak_memory_stats()
    frozen_out = v2.infer_scores(
        frozen_model, dataset, device=device, batch_size=v2.BATCH_SIZE, margin=v2.MARGIN
    )
    tuned_out = v2.infer_scores(
        tuned, dataset, device=device, batch_size=v2.BATCH_SIZE, margin=v2.MARGIN
    )
    torch.cuda.synchronize()
    peaks = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }

    for name, produced in (("frozen", frozen_out), ("tuned", tuned_out)):
        if produced["n_pairs"] != len(rows):
            raise RuntimeError(f"{name} arm scored a different number of rows")
        v2.verify_arm_vectors(produced, rows, name)
    frozen_loss = v2.measure_pair_loss(
        frozen_out["scores"], frozen_out["labels"], frozen_out["weights"], v2.MARGIN
    )
    tuned_loss = v2.measure_pair_loss(
        tuned_out["scores"], tuned_out["labels"], tuned_out["weights"], v2.MARGIN
    )
    frozen_auc = v2.roc_auc(frozen_out["scores"], frozen_out["labels"])
    native_auc = _finite_number(native["history"][-1]["validation_auc"], "native validation AUC")
    tuned_auc = v2.check_auc_reproduction(tuned_out["scores"], tuned_out["labels"], native_auc)
    bootstrap = v2.paired_bootstrap_ci(
        v2.pair_loss_numpy(frozen_out["scores"], frozen_out["labels"], v2.MARGIN),
        v2.pair_loss_numpy(tuned_out["scores"], tuned_out["labels"], v2.MARGIN),
        frozen_out["weights"],
    )
    verify_pairs(pairs, pairs_record, "after inference")
    fresh_check(frozen, root, "after inference")
    verify_native_pins(native, "after inference")

    # Fresh owned output: freeze the lease, then guard the private dir and each leaf.
    guard = lease_output(guard, frozen)
    owned = Path(guard.resolved)
    private = owned / "private"
    guard_output(guard, frozen, "before private dir")
    guard_leaf(guard, private, "before private dir")
    private.mkdir(parents=True)
    guard_output(guard, frozen, "after private dir")
    guard_leaf(guard, private, "after private dir")
    npz = private / "scores.npz"
    guard_leaf(guard, npz, "before scores write")
    np.savez(
        npz,
        scores_frozen=np.asarray(frozen_out["scores"], dtype=np.float32),
        scores_tuned=np.asarray(tuned_out["scores"], dtype=np.float32),
        labels=np.asarray(frozen_out["labels"], dtype=np.int8),
        weights=np.asarray(frozen_out["weights"], dtype=np.float32),
        pair_ids=np.asarray([pair.pair_id for pair, _a, _b in rows], dtype=np.str_),
    )
    guard_output(guard, frozen, "after scores write")
    guard_leaf(guard, npz, "after scores write")

    summary = owned / "summary.json"
    guard_leaf(guard, summary, "before summary write")
    cudnn_version = torch.backends.cudnn.version()
    delta = float(tuned_loss["validation_loss"] - frozen_loss["validation_loss"])
    cli, cli_meta = command_line(args)
    summary_payload = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "execution_complete": True,
        "evaluation_complete": False,
        "publication_ready": False,
        "native_experiment": native["experiment"],
        "backbone": backbone_name,
        "crops_dir": params["crops_dir"],
        "margin": v2.MARGIN,
        "batch_size": v2.BATCH_SIZE,
        "gap_weight": v2.GAP_WEIGHT,
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
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": cudnn_version,
            "gpu_name": torch.cuda.get_device_name(),
            "device": device,
            "threads": 1,
        },
        "peaks": peaks,
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
    _write_summary(summary, summary_payload)
    guard_output(guard, frozen, "after summary write")
    guard_leaf(guard, summary, "after summary write")
    claimed_outputs = (file_record(summary), file_record(npz))
    target = owned / "summary.manifest.json"
    params_out = dict(params)
    params_out.update(
        experiment=EXPERIMENT,
        command=" ".join(shlex.quote(p) for p in cli),
        python_version=platform.python_version(),
        ancestry_input_count=len(records),
        **cli_meta,
    )
    publish_manifest(
        target,
        summary=summary,
        npz=npz,
        frozen=frozen,
        summary_payload=summary_payload,
        params_out=params_out,
        guard=guard,
        root=root,
        cli=cli,
        claimed_outputs=claimed_outputs,
    )
    return target


def load_native_v5(train_dir: Path | str, root: Path) -> dict:
    manifest = Path(train_dir) / "summary.manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("native completion manifest missing")
    native, manifest_record = load_strict_json(manifest)
    v2.classify_native(native)
    backbone = v2.NATIVE_BACKBONES[native["experiment"]][0]
    inventory_pin = pin_inventory(
        Path(root) / "metrics/model_inventory.json", v2.INVENTORY_KEYS[backbone]
    )
    allowed = allowed_outside_weight(inventory_pin)
    _containment_preflight(
        strict_records(native["inputs"] + native["outputs"], "native"),
        Path(root).resolve(),
        allowed,
    )
    v2.verify_records(native["inputs"] + native["outputs"])
    params, history = native_contract_v5(native)
    declared_rows = v2.native_declared_row_count(native)
    checkpoint = v2.find_checkpoint(native)

    component_pin = None
    if native["experiment"] == v2.ADA_EXPERIMENT:
        component_path = checkpoint.with_suffix(".manifest.json")
        if file_record(component_path) not in native["outputs"]:
            raise ValueError("native cell does not declare the finetune component as its output")
        component_pin = pin_json(component_path)
        if component_pin.parsed.get("experiment") != v2.COMPONENT_EXPERIMENT:
            raise ValueError("wrong finetune component experiment")
        _containment_preflight(
            strict_records(
                component_pin.parsed["inputs"] + component_pin.parsed["outputs"], "component"
            ),
            Path(root).resolve(),
            allowed,
        )
        v2.verify_records(component_pin.parsed["inputs"] + component_pin.parsed["outputs"])
        verify_component_binding_v5(
            component_pin.parsed, checkpoint, params, parent_history=history
        )

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
        component=(component_pin.parsed if component_pin else None),
        component_pin=component_pin,
        inventory_pin=inventory_pin,
        experiment=native["experiment"],
    )


def verify_native_pins(native: dict, stage: str) -> None:
    """Re-verify the pinned native manifest, inventory and component after each phase."""
    if file_record(native["manifest"]) != native["manifest_record"]:
        raise RuntimeError(f"native manifest changed {stage}")
    verify_pinned_json(native["inventory_pin"].pin, stage)
    if native["component_pin"] is not None:
        verify_pinned_json(native["component_pin"], stage)


def _redact(exc: BaseException) -> str:
    detail = getattr(exc, "detail", None)
    match = _REASON_RE.search(detail) if isinstance(detail, str) else None
    reason = match.group(0) if match else "reason=internal"
    return f"{type(exc).__name__}: {reason}"


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(f"dry-run: {EXPERIMENT}; no CUDA probe, no model load, no data preflight")
        return
    try:
        run(args)
    except BaseException as exc:
        raise SystemExit(_redact(exc)) from None
    print(f"{EXPERIMENT}: completed")


if __name__ == "__main__":
    main()
