"""Hardened stdlib verifier for exactly one native fixed8 AdaFace CUDA cell.

Recovery v3. Supersedes recovery v2 (kept read-only). Five v2 holes are closed:

1. Privacy used two rules: labels redacted identity-like names but the sensitivity
   test that decides whether a detail may carry a SHA-256 did not, so a mismatch on
   a present identity-named file leaked its digest. v3 uses ONE classification for
   both the label and the detail: only declared public code/aggregate paths may
   appear or carry a digest, everything else is redacted. Identity-like names are
   checked on EVERY path component (parents included), declared and resolved
   (alias/symlink) locations are both classified, and the actual cell-root basename
   is never emitted: on a name mismatch only the derived canonical cell label is
   reported.
2. Percentile bootstrap CIs need not contain the original point and a nonlinear
   leave-one-out range need not contain the full-sample delta. The three
   containment checks are removed; finite, ordered, in-domain ranges and the exact
   arithmetic identities remain.
3. The second-pass snapshot deduplicated by literal path, letting a later manifest
   hash silently overwrite an earlier disagreement. v3 canonicalizes every record
   path, rejects inconsistent aliases before dedup, and requires each manifest's
   own single-read (bytes, sha256) to equal EVERY parent-declared record for the
   same resolved path.
4. Schema gaps: history loss/norm/duration must be nonnegative, AUC values in
   [0, 1], counts positive integers, and every role experiment string typed. A
   resolve() OSError now yields a normalized privacy-safe failure, never a raw path.
5. Native role sets, parameter/metric field names and stratum contract are grounded
   in the IMMUTABLE producer sources only (no real manifests are read):
     - scripts/run_strong_backbone_cuda_cell_v1.py:30-47 (fixed8 training contract
       fields), :145-175 (train summary/metrics, device/negative parameters, outputs)
     - src/age_gap/training/finetune.py:264-271 (history fields), :330-352 (component
       parameters, metrics best_validation_auc/selected_epoch/history, outputs=[ckpt_out])
     - scripts/evaluate_strong_cuda_fgnet_v1.py:145-166 (external results/coverage),
       :167-185 (private outputs), :189-203 (external parameters, outputs)
     - scripts/strong_subject_inference_v1.py:71-80 (ROC-v2 metric fields),
       :81-101 (metrics block keys, LOO counts, bootstrap policy strings)
   No native field is invented beyond those lines.

Coordinator port adds sensitive-component precedence, normalized resolution failures,
pre-read manifest containment, second-pass containment, and the complete fixed-budget
parameter and pair-source linkage. No Torch, no GPU, no training or
inference, no weight or array decoding, no real-data mutation. Output is a single
per-cell acceptance object; no cumulative counters, matrix totals, mechanism,
identity, ethics or publication fields are ever emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

NEGATIVES = ("lookalike", "random")
SCOPES = ("full", "head", "tail")
LEARNING_RATES = (1e-6, 1e-5)
SEEDS = (1, 2, 42)
EPOCHS = 8
BACKBONE = "adaface_ir101"
FROZEN = "frozen"
TRAIN_EXPERIMENT = "adaface-fixed8-native-cuda-training-cell"
CHECKPOINT_EXPERIMENT = "pair-contrastive-backbone-finetune"
EXTERNAL_EXPERIMENT = "adaface-fixed8-cuda-full-fgnet-roc-v2"
HISTORY_FIELDS = ("training_loss", "validation_auc", "mean_gradient_norm", "epoch_seconds")
NONNEGATIVE_HISTORY = ("training_loss", "mean_gradient_norm", "epoch_seconds")
METRICS = ("roc_auc", "eer_interpolated", "eer_discrete_minimax", "tar@far=0.01", "tar@far=0.001")
METRIC_KEYS = {"point", "ci95", "delta_vs_frozen", "delta_ci95", "loo_delta_min", "loo_delta_max"}
# (stratum, pairs, subjects): observed immutable native contract for the one cell.
STRATA = (("overall", 5308, 82), ("large_gap_25plus", 430, 75))
BOOTSTRAP_POSITIVE = "owner multiplicity once"
BOOTSTRAP_NEGATIVE = "endpoint multiplicity product"
CROP_CACHE_REL = "data/external/fgnet_crops.npz"
COVERAGE_FIELDS = ("source_positives", "retained_positives", "source_25plus_positives",
                   "retained_25plus_positives")
# role -> (manifest leaf relative to the cell root, required experiment)
ROLES = {
    "train": ("train/summary.manifest.json", TRAIN_EXPERIMENT),
    "checkpoint": ("train/private/checkpoint.manifest.json", CHECKPOINT_EXPERIMENT),
    "external": ("fgnet/summary.manifest.json", EXTERNAL_EXPERIMENT),
}
# Non-private structural leaves of the cell root: aggregates/manifests, safe to name.
SAFE_CELL_LEAVES = frozenset({
    "train/summary.json", "train/summary.manifest.json",
    "fgnet/summary.json", "fgnet/summary.manifest.json",
})
# The entire public-code/aggregate allowlist. Nothing else may ever be echoed.
PUBLIC_FILES = frozenset({"pyproject.toml", "uv.lock", "metrics/model_inventory.json"})
PUBLIC_PREFIXES = ("scripts/", "src/", "tests/")
# Any path part that makes a record private, data-bearing or identity-bearing.
SENSITIVE_PARTS = frozenset({
    "private", "data", "raw", "interim", "processed", "faces", "crops", "external",
    "identity", "identities", "social", "person", "persons", "embeddings", "weights",
    "checkpoints", "cache",
})
_OUTSIDE = "<redacted:outside-path>"
_SENSITIVE = "<redacted:sensitive-path>"
_IDENTITY = "<redacted:identity-like>"
_TYPES = {"int": int, "float": (int, float), "str": str, "bool": bool, "dict": dict, "list": list}


class VerificationError(Exception):
    """One rejected check, worded so private material can never leak."""

    def __init__(self, stage: str, subject: str, detail: str) -> None:
        super().__init__(f"{stage}: {subject}: {detail}")
        self.stage, self.subject, self.detail = stage, subject, detail


def cell_name(negative: str, scope: str, lr, seed) -> str:
    """Derive the canonical cell label; reject anything off the declared spec."""
    if negative not in NEGATIVES or scope not in SCOPES:
        raise VerificationError("spec", "arguments", "unsupported negative or scope")
    if type(seed) is not int or seed not in SEEDS:
        raise VerificationError("spec", "seed", "seed must be one of the declared integers")
    if type(lr) is bool or not isinstance(lr, (int, float)) or not math.isfinite(float(lr)):
        raise VerificationError("spec", "learning_rate", "learning_rate must be a finite number")
    if float(lr) not in LEARNING_RATES:
        raise VerificationError("spec", "learning_rate", "unsupported learning rate")
    return f"{negative}_{scope}_lr{float(lr):.0e}_s{seed}"


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest_file(path: Path) -> tuple[int, str]:
    """Stream a file through SHA-256; no size/mtime shortcut is ever accepted."""
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _safe_resolve(path: Path) -> Path | None:
    """Resolve without ever surfacing an OSError carrying a raw path."""
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _rel(root: Path, path: Path) -> str:
    """Repo-relative logical path, or an absolute path when outside the repo."""
    resolved = _safe_resolve(path)
    if resolved is None:
        return _OUTSIDE
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _norm(text: object) -> str:
    return str(text).replace("\\", "/")


def _parts(text: str) -> tuple[str, ...]:
    return tuple(part for part in text.split("/") if part not in ("", "."))


def _identity_like(name: str) -> bool:
    return sum(character.isdigit() for character in name) >= 6


def _outside(text: str, parts: tuple[str, ...]) -> bool:
    return text.startswith("/") or (len(text) > 1 and text[1] == ":") or ".." in parts


def _is_public(raw: str, parts: tuple[str, ...]) -> bool:
    """Only declared public code and aggregate files may ever be echoed."""
    if raw in PUBLIC_FILES:
        return True
    if len(parts) >= 2 and raw.startswith(PUBLIC_PREFIXES):
        return parts[-1].endswith((".py", ".toml"))
    return False


def _classify(text: object, cell: str | None, prefix: str | None) -> tuple[str, bool]:
    """Single redaction classification shared by labels and details.

    Returns ``(label, public)``. ``public`` is True only for the allowlisted
    public code/aggregate files; those may carry a digest in a failure detail.
    """
    raw = _norm(text)
    parts = _parts(raw)
    if raw == _OUTSIDE or _outside(raw, parts):
        return _OUTSIDE, False
    if prefix and (raw == prefix or raw.startswith(prefix + "/")):
        rest = raw[len(prefix):].lstrip("/")
        if rest in SAFE_CELL_LEAVES:
            return f"{cell or '<cell>'}/{rest}", True
        return _SENSITIVE, False
    if {part.lower() for part in parts} & SENSITIVE_PARTS:
        return _SENSITIVE, False
    if any(_identity_like(part) for part in parts):
        return _IDENTITY, False
    if _is_public(raw, parts):
        return raw, True
    return _SENSITIVE, False


def _label(text: object, cell: str | None = None, prefix: str | None = None) -> str:
    return _classify(text, cell, prefix)[0]


def _sensitive(text: object, cell: str | None = None, prefix: str | None = None) -> bool:
    return not _classify(text, cell, prefix)[1]


def _classify_pair(declared: object, resolved: object, cell: str | None,
                   prefix: str | None) -> tuple[str, bool]:
    """Classify a record using BOTH its declared and its resolved location.

    A public alias that resolves into a sensitive directory (or vice versa) is
    treated as sensitive, so symlinks and aliases cannot smuggle material out.
    """
    declared_label, declared_public = _classify(declared, cell, prefix)
    resolved_label, resolved_public = _classify(resolved, cell, prefix)
    if declared_public and resolved_public:
        return declared_label, True
    return (declared_label if not declared_public else resolved_label), False


def record_of(path: Path, root: Path) -> dict[str, object]:
    """Declared record for a real file, mirroring the canonical manifest schema."""
    size, sha = _digest_file(path)
    return {"path": _rel(root, path), "bytes": size, "sha256": sha}


def _typed(value: object, kind: str, label: str, stage: str):
    """Fail-closed type gate: bool never passes as int/float, NaN/Inf never as float."""
    if kind in ("int", "bool", "str"):
        if type(value) is not _TYPES[kind]:
            raise VerificationError(stage, label, f"expected {kind}, got {type(value).__name__}")
    elif not isinstance(value, _TYPES[kind]):
        raise VerificationError(stage, label, f"expected {kind}, got {type(value).__name__}")
    if kind == "float":
        if type(value) is bool:
            raise VerificationError(stage, label, "expected number, got bool")
        if not math.isfinite(float(value)):
            raise VerificationError(stage, label, "non-finite number")
    return value


def _number(value: object, label: str, stage: str) -> float:
    return float(_typed(value, "float", label, stage))


def _unit(value: object, label: str, stage: str) -> float:
    number = _number(value, label, stage)
    if not 0.0 <= number <= 1.0:
        raise VerificationError(stage, label, "must be within [0, 1]")
    return number


def _nonnegative(value: object, label: str, stage: str) -> float:
    number = _number(value, label, stage)
    if number < 0.0:
        raise VerificationError(stage, label, "must be nonnegative")
    return number


def _positive_int(value: object, label: str, stage: str) -> int:
    number = _typed(value, "int", label, stage)
    if number <= 0:
        raise VerificationError(stage, label, "must be a positive integer")
    return number


def _count(value: object, label: str, stage: str) -> int:
    number = _typed(value, "int", label, stage)
    if number < 0:
        raise VerificationError(stage, label, "must be a nonnegative integer")
    return number


def _bounds(value: object, label: str, stage: str) -> tuple[float, float]:
    pair = _typed(value, "list", label, stage)
    if len(pair) != 2:
        raise VerificationError(stage, label, "expected exactly two bounds")
    low = _number(pair[0], f"{label}[0]", stage)
    high = _number(pair[1], f"{label}[1]", stage)
    if low > high:
        raise VerificationError(stage, label, "bounds must be ordered low<=high")
    return low, high


def _parse_json(raw: bytes) -> object:
    """Strict JSON: reject duplicate keys and NaN/Infinity constants."""

    def no_duplicates(pairs):
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate keys")
        return dict(pairs)

    def reject_constant(name):
        raise ValueError(f"non-standard constant {name!r}")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates,
                      parse_constant=reject_constant)


def _inside_target(path: Path, root: Path, stage: str) -> Path:
    """Fail before opening anything when resolution fails or escapes the root."""
    target = _safe_resolve(path)
    if target is None or not target.is_relative_to(root):
        raise VerificationError(stage, _OUTSIDE, "reason=path-escapes-or-unresolvable")
    return target


def _read_bytes(path: Path, relative: str, stage: str, cell: str, prefix: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except (OSError, RuntimeError, ValueError):
        raise VerificationError(stage, _label(relative, cell, prefix), "reason=unreadable") from None


def _malformed(relative: str, size: int, sha: str, cell: str, prefix: str) -> VerificationError:
    label = _label(relative, cell, prefix)
    if _sensitive(relative, cell, prefix):
        return VerificationError("malformed-manifest", label, f"reason=malformed bytes={size}")
    return VerificationError("malformed-manifest", label,
                             f"reason=malformed bytes={size} sha256={sha}")


def _load_manifest(path: Path, root: Path, cell: str, prefix: str) -> tuple[dict, str, int, str]:
    """Hash and parse the SAME bytes: one read, then sha256, then json."""
    target = _inside_target(path, root, "manifest-path")
    relative = _rel(root, target)
    raw = _read_bytes(target, relative, "manifest-missing", cell, prefix)
    size, sha = len(raw), _digest(raw)
    try:
        data = _parse_json(raw)
    except (UnicodeDecodeError, ValueError):
        raise _malformed(relative, size, sha, cell, prefix) from None
    manifest = _typed(data, "dict", _label(relative, cell, prefix), "manifest-shape")
    for key in ("experiment", "parameters", "metrics", "inputs", "outputs"):
        if key not in manifest:
            raise VerificationError("manifest-shape", _label(relative, cell, prefix),
                                    f"missing {key!r}")
    return manifest, relative, size, sha


def _record_target(root: Path, relative: str) -> Path:
    return Path(relative) if Path(relative).is_absolute() else root / relative


def _check_manifest_binding(relative: str, size: int, sha: str,
                            declared: dict[str, set[tuple[int, str]]], cell: str,
                            prefix: str) -> None:
    """A parsed manifest's own bytes must match EVERY declared record at its path."""
    if declared.get(relative) and declared[relative] != {(size, sha)}:
        raise VerificationError("manifest-hash-conflict", _label(relative, cell, prefix),
                                "parsed manifest bytes disagree with a declared record")


def _verify_records(manifest: dict, role: str, root: Path, cell: str, prefix: str,
                    declared: dict[str, set[tuple[int, str]]],
                    binding: dict[str, set[tuple[str, str]]]) -> list[tuple[str, int, str]]:
    """Re-hash every declared record; canonicalize, detect aliases, redact failures."""
    snapshot: list[tuple[str, int, str]] = []
    for kind in ("inputs", "outputs"):
        stage = f"{role}-{kind}"
        records = _typed(manifest.get(kind), "list", kind, stage)
        for index, raw in enumerate(records):
            record = _typed(raw, "dict", f"{kind}[{index}]", stage)
            if set(record) != {"path", "bytes", "sha256"}:
                raise VerificationError(stage, f"{kind}[{index}]",
                                        "reason=record-shape expected path, bytes, sha256")
            relative = _typed(record["path"], "str", "path", stage)
            size = _typed(record["bytes"], "int", "record bytes", stage)
            sha = _typed(record["sha256"], "str", "record sha256", stage)
            target = _record_target(root, relative)
            resolved = _safe_resolve(target)
            public = False
            if resolved is None:
                label, public = _classify_pair(relative, _OUTSIDE, cell, prefix)
            else:
                resolved_relative = _rel(root, resolved)
                label, public = _classify_pair(relative, resolved_relative, cell, prefix)
            if size < 0 or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise VerificationError(stage, label, f"reason=malformed-record bytes={size}")
            if resolved is None or not resolved.is_relative_to(root):
                raise VerificationError("record-escape", label, "reason=path-escapes-root")
            key = _rel(root, resolved)
            # Canonicalize, then reject inconsistent aliases BEFORE any disk
            # comparison or dedup, so a disagreement can never be overwritten.
            if (size, sha) not in declared.setdefault(key, set()) and declared[key]:
                raise VerificationError("record-alias-conflict", _label(key, cell, prefix),
                                        "reason=aliases-disagree-on-bytes-or-hash")
            declared[key].add((size, sha))
            binding.setdefault(key, set()).add((role, kind))
            try:
                actual = _digest_file(resolved)
            except OSError:
                raise VerificationError("record-missing", label,
                                        f"reason=unreadable declared_bytes={size}") from None
            if actual != (size, sha):
                if public:
                    detail = (f"reason=byte-drift declared=({size},{sha}) "
                              f"actual=({actual[0]},{actual[1]}) same_size={actual[0] == size}")
                else:
                    detail = (f"reason=byte-drift declared_bytes={size} actual_bytes={actual[0]} "
                              f"same_size={actual[0] == size}")
                raise VerificationError(stage, label, detail)
            snapshot.append((key, size, sha))
    return snapshot


def _check_history(history: list, stage: str) -> None:
    if len(history) != EPOCHS:
        raise VerificationError(stage, "history", f"expected {EPOCHS} rows, got {len(history)}")
    required = {"epoch", *HISTORY_FIELDS}
    for index, raw in enumerate(history):
        row = _typed(raw, "dict", f"history[{index}]", stage)
        if set(row) != required:
            raise VerificationError(stage, f"history[{index}]", "unexpected history fields")
        if _typed(row.get("epoch"), "int", f"history[{index}].epoch", stage) != index + 1:
            raise VerificationError(stage, f"history[{index}].epoch", f"expected {index + 1}")
        _unit(row.get("validation_auc"), f"history[{index}].validation_auc", stage)
        for field in NONNEGATIVE_HISTORY:
            _nonnegative(row.get(field), f"history[{index}].{field}", stage)


def _check_case(train: dict, checkpoint: dict, negative: str, scope: str, lr: float,
                seed: int) -> None:
    stage = "case-identity"
    expected = {"backbone": BACKBONE, "epochs_requested": EPOCHS, "epochs_executed": EPOCHS,
                "selected_epoch": EPOCHS, "checkpoint_selection": "last_epoch",
                "batchnorm_policy": "frozen_all", "trainable_scope": scope,
                "learning_rate": float(lr), "seed": seed, "batch_size": 16,
                "margin": 0.3, "gap_weight": 0.0, "crops_dir": "faces"}
    text_keys = {"backbone", "checkpoint_selection", "batchnorm_policy", "trainable_scope",
                 "crops_dir"}
    for manifest, name in ((train, "train"), (checkpoint, "checkpoint")):
        wanted = TRAIN_EXPERIMENT if name == "train" else CHECKPOINT_EXPERIMENT
        if _typed(manifest["experiment"], "str", f"{name}.experiment", stage) != wanted:
            raise VerificationError(stage, f"{name}.experiment", "unexpected experiment")
        params = _typed(manifest.get("parameters"), "dict", f"{name}.parameters", stage)
        for key, target in expected.items():
            kind = ("float" if key in {"learning_rate", "margin", "gap_weight"}
                    else ("str" if key in text_keys else "int"))
            got = _typed(params.get(key), kind, f"{name}.{key}", stage)
            if got != target:
                raise VerificationError(stage, f"{name}.{key}", f"expected {target!r}")
    if _typed(train["parameters"].get("negative"), "str", "train.negative", stage) != negative:
        raise VerificationError(stage, "train.negative", "does not match the derived cell")
    if _typed(train["parameters"].get("device"), "str", "train.device", stage) != "cuda":
        raise VerificationError(stage, "train.device", "native cell must run on cuda")
    metrics = _typed(train.get("metrics"), "dict", "train.metrics", stage)
    for key, target in (("training_complete", True), ("evaluation_complete", False),
                        ("publication_ready", False)):
        if _typed(metrics.get(key), "bool", f"train.{key}", stage) is not target:
            raise VerificationError(stage, f"train.{key}", f"expected {target}")
    for key in ("train_pairs", "validation_pairs", "unique_crop_files"):
        _positive_int(metrics.get(key), f"train.{key}", stage)
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        _nonnegative(metrics.get(key), f"train.{key}", stage)
    history = _typed(metrics.get("history"), "list", "train.history", stage)
    _check_history(history, stage)
    checkpoint_metrics = _typed(checkpoint.get("metrics"), "dict", "checkpoint.metrics", stage)
    if _typed(checkpoint_metrics.get("selected_epoch"), "int", "checkpoint.selected_epoch",
              stage) != EPOCHS:
        raise VerificationError(stage, "checkpoint.selected_epoch", f"expected {EPOCHS}")
    if _typed(checkpoint_metrics.get("history"), "list", "checkpoint.history", stage) != history:
        raise VerificationError(stage, "checkpoint.history", "must equal the training history")
    _unit(checkpoint_metrics.get("best_validation_auc"), "checkpoint.best_validation_auc", stage)


def _metric(entry: object, label: str, stage: str, *, tuned: bool) -> tuple[float, float]:
    """ROC-v2 metric fields; finite ordered in-domain ranges, no CI containment."""
    metric = _typed(entry, "dict", label, stage)
    if set(metric) != METRIC_KEYS:
        raise VerificationError(stage, label, "reason=metric-shape expected the six ROC-v2 fields")
    point = _unit(metric["point"], f"{label}.point", stage)
    low, high = _bounds(metric["ci95"], f"{label}.ci95", stage)
    if low < 0.0 or high > 1.0:
        raise VerificationError(stage, f"{label}.ci95", "must be within [0, 1]")
    delta = _number(metric["delta_vs_frozen"], f"{label}.delta_vs_frozen", stage)
    if not -1.0 <= delta <= 1.0:
        raise VerificationError(stage, f"{label}.delta_vs_frozen", "must be within [-1, 1]")
    delta_low, delta_high = _bounds(metric["delta_ci95"], f"{label}.delta_ci95", stage)
    if delta_low < -1.0 or delta_high > 1.0:
        raise VerificationError(stage, f"{label}.delta_ci95", "must be within [-1, 1]")
    if not tuned:
        if delta != 0.0 or delta_low != 0.0 or delta_high != 0.0:
            raise VerificationError(stage, label, "frozen baseline must self-delta to exactly zero")
        if metric["loo_delta_min"] is not None or metric["loo_delta_max"] is not None:
            raise VerificationError(stage, label, "frozen baseline carries no LOO delta")
        return point, delta
    loo_low = _number(metric["loo_delta_min"], f"{label}.loo_delta_min", stage)
    loo_high = _number(metric["loo_delta_max"], f"{label}.loo_delta_max", stage)
    if not -1.0 <= loo_low <= 1.0 or not -1.0 <= loo_high <= 1.0:
        raise VerificationError(stage, f"{label}.loo_delta_min", "must be within [-1, 1]")
    if loo_low > loo_high:
        raise VerificationError(stage, f"{label}.loo_delta_min", "must be <= loo_delta_max")
    return point, delta


def _check_metrics_block(block: object, name: str, pairs: int, subjects: int, cell: str) -> None:
    stage = "external-metrics"
    section = _typed(block, "dict", f"external.{name}", stage)
    for key, target in (("n_pairs", pairs), ("n_subjects", subjects),
                        ("actual_tuned_checkpoint_count", 1), ("loo_valid", subjects),
                        ("loo_unavailable", 0)):
        if _typed(section.get(key), "int", f"{name}.{key}", stage) != target:
            raise VerificationError(stage, f"{name}.{key}", f"expected {target!r}")
    if _typed(section.get("metric_version"), "str", f"{name}.metric_version",
              stage) != "empirical-roc-v2":
        raise VerificationError(stage, f"{name}.metric_version", "unexpected metric version")
    if _typed(section.get("publication_ready"), "bool", f"{name}.publication_ready",
              stage) is not False:
        raise VerificationError(stage, f"{name}.publication_ready", "must stay false")
    _typed(section.get("scope"), "str", f"{name}.scope", stage)
    _typed(section.get("training_identity_independence"), "str",
           f"{name}.training_identity_independence", stage)
    models = _typed(section.get("models"), "dict", f"{name}.models", stage)
    if set(models) != {FROZEN, cell}:
        raise VerificationError(stage, f"{name}.models", "must hold exactly frozen and the derived cell")
    frozen = _typed(models[FROZEN], "dict", f"{name}.models.{FROZEN}", stage)
    tuned = _typed(models[cell], "dict", f"{name}.models.{cell}", stage)
    for metric in METRICS:
        frozen_point, frozen_delta = _metric(frozen.get(metric), f"{name}.{FROZEN}.{metric}",
                                             stage, tuned=False)
        point, delta = _metric(tuned.get(metric), f"{name}.{cell}.{metric}", stage, tuned=True)
        if frozen_delta != 0.0 or delta != point - frozen_point:
            raise VerificationError(stage, f"{name}.{cell}.{metric}",
                                    "tuned delta must equal point minus the shared frozen point")
    bootstrap = _typed(section.get("bootstrap"), "dict", f"{name}.bootstrap", stage)
    for key in ("requested", "valid"):
        if _typed(bootstrap.get(key), "int", f"{name}.bootstrap.{key}", stage) != 2000:
            raise VerificationError(stage, f"{name}.bootstrap.{key}", "expected 2000")
    if _typed(bootstrap.get("seed"), "int", f"{name}.bootstrap.seed", stage) != 0:
        raise VerificationError(stage, f"{name}.bootstrap.seed", "expected shared protocol seed 0")
    for key in ("shared_draws_across_models", "thresholds_reselected"):
        if _typed(bootstrap.get(key), "bool", f"{name}.bootstrap.{key}", stage) is not True:
            raise VerificationError(stage, f"{name}.bootstrap.{key}", "must be true")
    if _typed(bootstrap.get("positive_weight"), "str", f"{name}.bootstrap.positive_weight",
              stage) != BOOTSTRAP_POSITIVE:
        raise VerificationError(stage, f"{name}.bootstrap.positive_weight", "unexpected policy")
    if _typed(bootstrap.get("negative_weight"), "str", f"{name}.bootstrap.negative_weight",
              stage) != BOOTSTRAP_NEGATIVE:
        raise VerificationError(stage, f"{name}.bootstrap.negative_weight", "unexpected policy")


def _check_external(external: dict, cell: str) -> None:
    stage = "external-identity"
    if _typed(external["experiment"], "str", "external.experiment", stage) != EXTERNAL_EXPERIMENT:
        raise VerificationError(stage, "external.experiment", "unexpected experiment")
    params = _typed(external.get("parameters"), "dict", "external.parameters", stage)
    for key, target in (("device", "cuda"), ("protocol_seed", 42), ("tolerance", 2),
                        ("n_boot", 2000), ("bootstrap_seed", 0)):
        kind = "str" if key == "device" else "int"
        if _typed(params.get(key), kind, f"external.{key}", stage) != target:
            raise VerificationError(stage, f"external.{key}", f"expected {target!r}")
    if _typed(params.get("actual_cells"), "list", "external.actual_cells", stage) != [cell]:
        raise VerificationError(stage, "external.actual_cells", "must name exactly the derived cell")
    metrics = _typed(external.get("metrics"), "dict", "external.metrics", stage)
    if _typed(metrics.get("execution_complete"), "bool", "external.execution_complete",
              stage) is not True:
        raise VerificationError(stage, "external.execution_complete", "external run must be complete")
    if _typed(metrics.get("publication_ready"), "bool", "external.publication_ready",
              stage) is not False:
        raise VerificationError(stage, "external.publication_ready", "must stay false")
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        _nonnegative(metrics.get(key), f"external.{key}", stage)
    _typed(metrics.get("limitation"), "str", "external.limitation", stage)
    coverage = _typed(metrics.get("coverage"), "dict", "external.coverage", stage)
    if set(coverage) != set(COVERAGE_FIELDS):
        raise VerificationError(stage, "external.coverage", "unexpected coverage fields")
    for key in COVERAGE_FIELDS:
        _count(coverage.get(key), f"external.coverage.{key}", stage)
    for name, pairs, subjects in STRATA:
        _check_metrics_block(metrics.get(name), name, pairs, subjects, cell)


def _summary_metrics(relative: str, root: Path, cell: str, prefix: str) -> object:
    target = _inside_target(root / relative, root, "summary-path")
    raw = _read_bytes(target, relative, "summary-missing", cell, prefix)
    try:
        return _parse_json(raw)
    except (UnicodeDecodeError, ValueError):
        raise VerificationError("summary-malformed", _label(relative, cell, prefix),
                                "reason=malformed") from None


def _rehash_snapshot(snapshot: list[tuple[str, int, str]], root: Path, cell: str,
                     prefix: str) -> None:
    """Second pass: catch same-size mutation between the two hashing passes."""
    for relative, size, sha in snapshot:
        target = _inside_target(root / relative, root, "snapshot-drift")
        label, public = _classify_pair(relative, _rel(root, target), cell, prefix)
        try:
            actual = _digest_file(target)
        except OSError:
            detail = "reason=drift-missing"
            if public:
                detail += f" declared_bytes={size}"
            raise VerificationError("snapshot-drift", label, detail) from None
        if actual != (size, sha):
            if public:
                detail = (f"reason=drift declared=({size},{sha}) actual=({actual[0]},{actual[1]}) "
                          f"same_size={actual[0] == size}")
            else:
                detail = (f"reason=drift declared_bytes={size} actual_bytes={actual[0]} "
                          f"same_size={actual[0] == size}")
            raise VerificationError("snapshot-drift", label, detail)


def verify_cell(root: Path, cell_root: Path, *, negative: str, scope: str, lr, seed: int,
                _hook=None) -> dict:
    """Verify one native cell and return its per-cell acceptance object."""
    cell = cell_name(negative, scope, lr, seed)
    root = _safe_resolve(Path(root))
    if root is None:
        raise VerificationError("root", _OUTSIDE, "reason=root-unresolvable")
    candidate = Path(cell_root)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = _safe_resolve(candidate)
    # Containment is established before any manifest byte is read. On any resolve
    # failure the actual path is never echoed.
    if resolved is None:
        raise VerificationError("cell-root", _OUTSIDE, "reason=cell-root-unresolvable")
    if not resolved.is_relative_to(root):
        raise VerificationError("cell-root", _OUTSIDE, "reason=cell-root-outside-repo")
    if resolved.name != cell:
        # The actual mismatched basename is never emitted; the derived label is safe.
        raise VerificationError("cell-name", cell,
                                "cell root basename must equal the derived canonical label")
    if not resolved.is_dir():
        raise VerificationError("cell-root", _OUTSIDE, "reason=cell-root-missing")
    prefix = resolved.relative_to(root).as_posix()
    train_summary = f"{prefix}/train/summary.json"
    train_manifest = f"{prefix}/train/summary.manifest.json"
    checkpoint_pt = f"{prefix}/train/private/checkpoint.pt"
    checkpoint_manifest = f"{prefix}/train/private/checkpoint.manifest.json"
    external_summary = f"{prefix}/fgnet/summary.json"
    scores = f"{prefix}/fgnet/private/scores.npz"
    frozen_embeddings = f"{prefix}/fgnet/private/frozen_embeddings.npz"
    cell_embeddings = f"{prefix}/fgnet/private/{cell}_embeddings.npz"

    manifests: dict[str, dict] = {}
    parsed: dict[str, tuple[str, int, str]] = {}
    # Preflight ALL role paths before opening the first manifest.
    for leaf, _experiment in ROLES.values():
        _inside_target(resolved / leaf, root, "manifest-path")
    for role, (leaf, experiment) in ROLES.items():
        manifest, relative, size, sha = _load_manifest(resolved / leaf, root, cell, prefix)
        if manifest["experiment"] != experiment:
            raise VerificationError("native-roles", _label(relative, cell, prefix),
                                    "unexpected experiment for role")
        manifests[role] = manifest
        parsed[role] = (relative, size, sha)
    train, checkpoint, external = manifests["train"], manifests["checkpoint"], manifests["external"]
    declared: dict[str, set[tuple[int, str]]] = {}
    binding: dict[str, set[tuple[str, str]]] = {}
    snapshot: list[tuple[str, int, str]] = []
    for role in ROLES:
        snapshot += _verify_records(manifests[role], role, root, cell, prefix, declared, binding)
    # Every parsed manifest's own single-read (bytes, sha256) must equal EVERY
    # parent-declared record for the same resolved path; disagreement is fatal.
    for relative, size, sha in parsed.values():
        _check_manifest_binding(relative, size, sha, declared, cell, prefix)
        declared.setdefault(relative, set()).add((size, sha))
        snapshot.append((relative, size, sha))
    unique = {relative: (size, sha) for relative, size, sha in snapshot}
    snapshot = [(relative, size, sha) for relative, (size, sha) in unique.items()]

    def of(role: str, kind: str) -> set[str]:
        return {key for key, roles in binding.items() if (role, kind) in roles}

    # Summaries are exact bound outputs; checkpoint manifest and weights are bound roles.
    if of("train", "outputs") != {train_summary, checkpoint_pt, checkpoint_manifest}:
        raise VerificationError("role-outputs", "train outputs",
                                "must be exactly summary.json, checkpoint.pt, checkpoint.manifest.json")
    if of("checkpoint", "outputs") != {checkpoint_pt}:
        raise VerificationError("role-outputs", "checkpoint outputs", "must be exactly checkpoint.pt")
    if of("external", "outputs") != {external_summary, scores, frozen_embeddings, cell_embeddings}:
        raise VerificationError("role-outputs", "external outputs",
                                "must be exactly the summary and the three npz artifacts")
    external_inputs = of("external", "inputs")
    for parent, key in (("train manifest", train_manifest), ("checkpoint manifest", checkpoint_manifest),
                        ("checkpoint", checkpoint_pt), ("crop cache", CROP_CACHE_REL)):
        if key not in external_inputs:
            raise VerificationError("parent-linkage", _label(key, cell, prefix),
                                    f"external inputs must bind the {parent}")
    if train_summary not in declared:
        raise VerificationError("summary-binding", _label(train_summary, cell, prefix),
                                "summary must be a bound output")
    if external_summary not in declared:
        raise VerificationError("summary-binding", _label(external_summary, cell, prefix),
                                "summary must be a bound output")

    pair_source = ("data/processed/pairs.jsonl" if negative == "random"
                   else "data/processed/experiments/pairs_lookalike.jsonl")
    for role in ("train", "checkpoint"):
        params = _typed(manifests[role]["parameters"], "dict", role, "pair-source")
        raw_pair = _typed(params.get("pairs_file"), "str", "pairs_file", "pair-source")
        target = _inside_target(_record_target(root, raw_pair), root, "pair-source")
        if _rel(root, target) != pair_source or pair_source not in of(role, "inputs"):
            raise VerificationError("pair-source", _SENSITIVE, "reason=pair-source-linkage")

    _check_case(train, checkpoint, negative, scope, float(lr), seed)
    _check_external(external, cell)
    if _summary_metrics(train_summary, root, cell, prefix) != train["metrics"]:
        raise VerificationError("summary-mismatch", _label(train_summary, cell, prefix),
                                "manifest metrics differ from the standalone summary")
    if _summary_metrics(external_summary, root, cell, prefix) != external["metrics"]:
        raise VerificationError("summary-mismatch", _label(external_summary, cell, prefix),
                                "manifest metrics differ from the standalone summary")
    if _hook is not None:
        _hook()
    _rehash_snapshot(snapshot, root, cell, prefix)
    return {"cell": cell, "accepted": True, "records_verified": len(snapshot), "snapshot_stable": True}


def _learning_rate(text: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value not in LEARNING_RATES:
        raise argparse.ArgumentTypeError(f"unsupported learning rate: {text!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify one native fixed8 AdaFace CUDA cell.")
    parser.add_argument("--root", type=Path, default=None, help="repo root (default: cwd)")
    parser.add_argument("--cell-root", type=Path, required=True,
                        help="cell directory inside the repo root")
    parser.add_argument("--negative", choices=list(NEGATIVES), required=True)
    parser.add_argument("--scope", choices=list(SCOPES), required=True)
    parser.add_argument("--lr", type=_learning_rate, required=True)
    parser.add_argument("--seed", type=int, choices=list(SEEDS), required=True)
    args = parser.parse_args(argv)
    root = args.root or Path.cwd()
    cell = f"{args.negative}_{args.scope}_lr{args.lr:.0e}_s{args.seed}"
    try:
        result = verify_cell(root, args.cell_root, negative=args.negative, scope=args.scope,
                             lr=args.lr, seed=args.seed)
    except VerificationError as error:
        print(json.dumps({"cell": cell, "accepted": False, "stage": error.stage,
                          "subject": error.subject, "detail": error.detail}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
