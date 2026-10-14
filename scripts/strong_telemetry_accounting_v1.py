"""Stdlib-only telemetry accounting validator; no native byte acceptance.

Scope
-----
This module is a *draft* pure-dict accounting checker for the native manifest of the
separately versioned telemetry experiment::

    adaface-fixed8-native-cuda-validation-telemetry-v1

It is deliberately NOT a full evaluator and NOT a full byte/reader. It performs no
file I/O, imports no ``torch``/``numpy``/model runner, executes no native code, and
reads no metrics/data/checkpoints/weights. Callers pass already-decoded plain
``dict`` objects (a manifest's ``experiment``/``parameters``/``metrics`` surface).

Two different claims, kept distinct
-----------------------------------
1. **Source-import closure (static):** which source files a producer *declares* it
   depends on. This is a byte-binding/bookkeeping claim. ``required_train_producer_bindings``
   below only states the expected set; it does not resolve or hash anything.
2. **Native acceptance (dynamic):** whether an observed native artifact actually
   satisfies the schema/accounting contract. ``validate_telemetry_record`` checks the
   decoded values only. It cannot prove a genuine torch forward pass, cannot replay
   neurons, and cannot recompute a statistical metric.

Honesty boundary (must not be overstated)
-----------------------------------------
Because this is stdlib-only, every accepted identity is an *accounting* identity on
already-recorded numbers. In particular this module verifies
``validation_loss == validation_numerator / validation_denominator`` but cannot
verify that the numerator is the true weighted loss of a real validation forward.
It therefore always reports ``neural_replay: False`` and ``statistical_replay: False``.

No invented magnitude bounds
----------------------------
The producer (``scripts/run_strong_validation_telemetry_cuda_v1.py:96-121``) only
requires the four magnitudes to be *finite* and the denominators/loss identity to
hold. Floating unit-vector dot products can exceed 1 by a rounding ulp so positive-pair loss terms can be slightly negative. This validator intentionally does NOT add an exact nonnegative-loss
bound, an upper loss bound, or any tolerance that the producer does not itself
enforce. Only AUC is unit-bounded because ``roc_auc`` is defined on ``[0, 1]``.
"""

from __future__ import annotations

import math

# --- Exact native literals (as-of this audit) ---------------------------------

# scripts/run_strong_validation_telemetry_cuda_v1.py:245
TELEMETRY_TRAIN_EXPERIMENT = "adaface-fixed8-native-cuda-validation-telemetry-v1"
# scripts/verify_strong_native_cell_v1.py:62 (checkpoint role is unchanged)
CHECKPOINT_EXPERIMENT = "pair-contrastive-backbone-finetune"
# scripts/strong_validation_telemetry_v1.py:18
TELEMETRY_VERSION = "strong-same-validation-pass-telemetry-v1"
# scripts/run_strong_validation_telemetry_cuda_v1.py:233-234
VALIDATION_LOSS_SCOPE = "global_weight_mass_from_same_validation_forward"
TRAINING_LOSS_SCOPE = "pair_count_weighted_batch_means; global equivalence only for gap_weight0"
# scripts/run_strong_validation_telemetry_cuda_v1.py:249-250
TELEMETRY_DEVICE = "cuda"
TELEMETRY_THREADS = 1
TELEMETRY_NEGATIVES = ("random", "lookalike")

EPOCHS = 8
# scripts/run_strong_validation_telemetry_cuda_v1.py:235-236
EXTRA_VALIDATION_ITERATORS = 0
NATIVE_BITWISE_EQUIVALENCE_VERIFIED = False

# scripts/run_strong_validation_telemetry_cuda_v1.py:98-105
VALIDATION_HISTORY_FIELDS = frozenset(
    {
        "epoch",
        "validation_loss",
        "validation_numerator",
        "validation_denominator",
        "validation_pairs",
        "validation_auc",
    }
)
_VALIDATION_FLOAT_FIELDS = (
    "validation_loss",
    "validation_numerator",
    "validation_denominator",
    "validation_auc",
)

# scripts/verify_strong_native_cell_v1.py:64
HISTORY_FIELDS = (
    "training_loss",
    "validation_auc",
    "mean_gradient_norm",
    "epoch_seconds",
)
# scripts/verify_strong_native_cell_v1.py:65
NONNEGATIVE_HISTORY = ("training_loss", "mean_gradient_norm", "epoch_seconds")

# Minimal declared producer bindings for the telemetry train role. Mirrors
# scripts/run_strong_validation_telemetry_cuda_v1.py:168-181 (NOT a hash check).
REQUIRED_TRAIN_PRODUCER_BINDINGS = (
    "scripts/run_strong_validation_telemetry_cuda_v1.py",
    "scripts/strong_validation_telemetry_v1.py",
    "models/adaface_ir101.pt",
    "metrics/model_inventory.json",
    "src/age_gap/training/finetune.py",
    "src/age_gap/training/dataset.py",
    "src/age_gap/training/losses.py",
    "src/age_gap/evaluation/metrics.py",
)


class TelemetryAccountingError(ValueError):
    """Raised when a decoded telemetry surface violates the accounting contract."""

    def __init__(self, where: str, message: str) -> None:
        super().__init__(f"{where}: {message}")
        self.where = where
        self.message = message


# --- strict native-typing helpers ---------------------------------------------


def _require_mapping(value, where: str) -> dict:
    if type(value) is not dict:
        raise TelemetryAccountingError(where, "expected a JSON object")
    return value


def _require_list(value, where: str) -> list:
    if type(value) is not list:
        raise TelemetryAccountingError(where, "expected a JSON array")
    return value


def _native_int(value, where: str) -> int:
    # bool is a subclass of int; JSON booleans must not pass as integers.
    if type(value) is not int:
        raise TelemetryAccountingError(where, f"expected native int, got {type(value).__name__}")
    return value


def _native_float(value, where: str) -> float:
    if type(value) is not float:
        raise TelemetryAccountingError(where, f"expected native float, got {type(value).__name__}")
    if not math.isfinite(value):
        raise TelemetryAccountingError(where, "expected finite float")
    return value


def _native_bool(value, where: str) -> bool:
    if type(value) is not bool:
        raise TelemetryAccountingError(where, f"expected native bool, got {type(value).__name__}")
    return value


def _unit(value, where: str) -> float:
    number = _native_float(value, where)
    if not 0.0 <= number <= 1.0:
        raise TelemetryAccountingError(where, "expected value within [0, 1]")
    return number


# --- checks on the decoded surfaces -------------------------------------------


def validate_experiment(manifest) -> None:
    """Check the top-level native experiment identifier."""
    record = _require_mapping(manifest, "manifest")
    got = record.get("experiment")
    if got != TELEMETRY_TRAIN_EXPERIMENT:
        raise TelemetryAccountingError("manifest.experiment", "unexpected experiment")


def validate_history(history) -> None:
    """Check the base 4-field training history (unchanged by telemetry)."""
    rows = _require_list(history, "history")
    if len(rows) != EPOCHS:
        raise TelemetryAccountingError("history", f"expected {EPOCHS} rows, got {len(rows)}")
    required = {"epoch", *HISTORY_FIELDS}
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, f"history[{index}]")
        if set(row) != required:
            raise TelemetryAccountingError(f"history[{index}]", "unexpected history fields")
        if _native_int(row.get("epoch"), f"history[{index}].epoch") != index + 1:
            raise TelemetryAccountingError(f"history[{index}].epoch", f"expected {index + 1}")
        _unit(row.get("validation_auc"), f"history[{index}].validation_auc")
        for field in NONNEGATIVE_HISTORY:
            value = _native_float(row.get(field), f"history[{index}].{field}")
            if value < 0.0:
                raise TelemetryAccountingError(f"history[{index}].{field}", "expected nonnegative")


def validate_validation_history(validation_history, history, count: int) -> None:
    """Check the eight native telemetry rows against the paired base history.

    ``count`` is the expected validation pair count (``len(val)`` / ``counts['validation_pairs']``).
    """
    if type(count) is not int or count <= 0:
        raise TelemetryAccountingError("count", "expected a positive native int")
    rows = _require_list(validation_history, "validation_history")
    base = _require_list(history, "history")
    if len(rows) != EPOCHS or len(base) != EPOCHS:
        raise TelemetryAccountingError(
            "validation_history", "expected eight validation rows and eight history rows"
        )
    for index, (row, old) in enumerate(zip(rows, base, strict=True), 1):
        label = f"validation_history[{index - 1}]"
        row = _require_mapping(row, label)
        old = _require_mapping(old, f"history[{index - 1}]")
        if set(row) != VALIDATION_HISTORY_FIELDS:
            raise TelemetryAccountingError(label, "unexpected validation-history fields")
        if _native_int(row.get("epoch"), f"{label}.epoch") != index:
            raise TelemetryAccountingError(f"{label}.epoch", f"expected {index}")
        if _native_int(row.get("validation_pairs"), f"{label}.validation_pairs") != count:
            raise TelemetryAccountingError(f"{label}.validation_pairs", f"expected {count}")
        for field in _VALIDATION_FLOAT_FIELDS:
            _native_float(row.get(field), f"{label}.{field}")
        # Unit weight mass: dataset._pair_weight returns exactly 1.0 for gap_weight 0.
        if row["validation_denominator"] != float(count):
            raise TelemetryAccountingError(f"{label}.validation_denominator", "unit mass expected")
        # Strict accounting identity, exactly as the producer checks it. No
        # magnitude bound is invented here (see module docstring).
        if row["validation_loss"] != row["validation_numerator"] / row["validation_denominator"]:
            raise TelemetryAccountingError(f"{label}.validation_loss", "loss accounting differs")
        # Same-forward AUC: telemetry AUC must equal the base history AUC exactly.
        if old.get("epoch") != index:
            raise TelemetryAccountingError(f"history[{index - 1}].epoch", "epoch order differs")
        _unit(row["validation_auc"], f"{label}.validation_auc")
        if row["validation_auc"] != old.get("validation_auc"):
            raise TelemetryAccountingError(f"{label}.validation_auc", "same-forward AUC differs")


def validate_telemetry_parameters(parameters) -> None:
    """Check the telemetry parameter block (version/device/threads/negative/versions)."""
    params = _require_mapping(parameters, "parameters")
    if params.get("telemetry_version") != TELEMETRY_VERSION:
        raise TelemetryAccountingError(
            "parameters.telemetry_version", "unexpected telemetry version"
        )
    if type(params.get("gap_weight")) is not float or params["gap_weight"] != 0.0:
        raise TelemetryAccountingError("parameters.gap_weight", "unit-weight policy required")
    if params.get("device") != TELEMETRY_DEVICE:
        raise TelemetryAccountingError("parameters.device", f"expected {TELEMETRY_DEVICE!r}")
    if _native_int(params.get("threads"), "parameters.threads") != TELEMETRY_THREADS:
        raise TelemetryAccountingError("parameters.threads", f"expected {TELEMETRY_THREADS}")
    if params.get("negative") not in TELEMETRY_NEGATIVES:
        raise TelemetryAccountingError("parameters.negative", "unexpected negative mode")
    for key in ("torch_version", "cuda_version", "gpu_name"):
        if type(params.get(key)) is not str or not params[key]:
            raise TelemetryAccountingError(f"parameters.{key}", "expected nonempty string")
    cudnn = params.get("cudnn_version")
    if cudnn is not None and type(cudnn) is not int:
        raise TelemetryAccountingError("parameters.cudnn_version", "expected int or null")


def validate_summary_metrics(metrics, count: int) -> None:
    """Check the telemetry metric block: scopes, flags, pair count, and history pairing."""
    block = _require_mapping(metrics, "metrics")
    for key, expected in (
        ("training_complete", True),
        ("evaluation_complete", False),
        ("publication_ready", False),
    ):
        if type(block.get(key)) is not bool or block[key] is not expected:
            raise TelemetryAccountingError("metrics." + key, "completion/scope flag differs")
    if block.get("validation_loss_scope") != VALIDATION_LOSS_SCOPE:
        raise TelemetryAccountingError("metrics.validation_loss_scope", "unexpected scope literal")
    if block.get("training_loss_scope") != TRAINING_LOSS_SCOPE:
        raise TelemetryAccountingError("metrics.training_loss_scope", "unexpected scope literal")
    if (
        _native_int(block.get("extra_validation_iterators"), "metrics.extra_validation_iterators")
        != EXTRA_VALIDATION_ITERATORS
    ):
        raise TelemetryAccountingError(
            "metrics.extra_validation_iterators", f"expected {EXTRA_VALIDATION_ITERATORS}"
        )
    flag = _native_bool(
        block.get("native_bitwise_equivalence_verified"),
        "metrics.native_bitwise_equivalence_verified",
    )
    if flag is not NATIVE_BITWISE_EQUIVALENCE_VERIFIED:
        raise TelemetryAccountingError(
            "metrics.native_bitwise_equivalence_verified", "must be False (no bitwise claim)"
        )
    if _native_int(block.get("validation_pairs"), "metrics.validation_pairs") != count:
        raise TelemetryAccountingError("metrics.validation_pairs", f"expected {count}")
    validate_history(block.get("history"))
    validate_validation_history(block.get("validation_history"), block.get("history"), count)


def validate_telemetry_record(manifest, count: int) -> dict:
    """Run the full decoded-surface accounting check and return an honest report.

    Returns a report dict that records what was and was not verified. It never
    asserts native, neural, statistical, or bitwise equivalence.
    """
    validate_experiment(manifest)
    record = _require_mapping(manifest, "manifest")
    validate_telemetry_parameters(record.get("parameters"))
    validate_summary_metrics(record.get("metrics"), count)
    rows = _require_list(record["metrics"]["validation_history"], "validation_history")
    return {
        "experiment": TELEMETRY_TRAIN_EXPERIMENT,
        "telemetry_version": TELEMETRY_VERSION,
        "epochs": len(rows),
        "validation_pairs": count,
        "native_bitwise_equivalence_verified": NATIVE_BITWISE_EQUIVALENCE_VERIFIED,
        "accounting_checked": True,
        "neural_replay": False,
        "statistical_replay": False,
        "source_import_closure_checked": False,
    }


def required_train_producer_bindings() -> tuple:
    """Return the expected static producer bindings for the telemetry train role.

    This is a *declaration* of the expected set only; resolving and hashing those
    files belongs to a separate byte reader, which this draft does not implement.
    """
    return REQUIRED_TRAIN_PRODUCER_BINDINGS
