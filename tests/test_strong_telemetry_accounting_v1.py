"""Generated synthetic tests for the draft telemetry accounting validator (e472).

These tests are stdlib + pytest only. They import no ``torch``/``numpy``, perform
no file I/O, read no native metrics/data/checkpoints/weights, and never import or
monkeypatch the canonical producer, helper, verifier, evaluator, or reader. The
validator is exercised as a pure set of dict functions using synthetic in-memory
manifests. All pytest temp/cache output is directed under this task directory by
the caller via ``--basetemp``; these tests need no temp files themselves.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts import strong_telemetry_accounting_v1 as V

_MODULE_PATH = Path(V.__file__)


COUNT = 53
AUC = [0.5, 0.55, 0.6, 0.62, 0.7, 0.73, 0.8, 0.81]


def _history():
    return [
        {
            "epoch": i,
            "training_loss": 0.5 + 0.01 * i,
            "validation_auc": AUC[i - 1],
            "mean_gradient_norm": 0.1 * i,
            "epoch_seconds": 1.0 + i,
        }
        for i in range(1, V.EPOCHS + 1)
    ]


def _validation_history(count=COUNT, numerator=2.0, aucs=None):
    aucs = AUC if aucs is None else aucs
    rows = []
    for i in range(1, V.EPOCHS + 1):
        num = numerator + i
        den = float(count)
        rows.append(
            {
                "epoch": i,
                "validation_loss": num / den,
                "validation_numerator": num,
                "validation_denominator": den,
                "validation_pairs": count,
                "validation_auc": aucs[i - 1],
            }
        )
    return rows


def _manifest(count=COUNT, **overrides):
    metrics = {
        "training_complete": True,
        "evaluation_complete": False,
        "publication_ready": False,
        "history": _history(),
        "validation_history": _validation_history(count),
        "validation_loss_scope": V.VALIDATION_LOSS_SCOPE,
        "training_loss_scope": V.TRAINING_LOSS_SCOPE,
        "extra_validation_iterators": 0,
        "native_bitwise_equivalence_verified": False,
        "validation_pairs": count,
    }
    parameters = {
        "telemetry_version": V.TELEMETRY_VERSION,
        "device": "cuda",
        "gap_weight": 0.0,
        "threads": 1,
        "negative": "random",
        "torch_version": "2.5.1+cu121",
        "cuda_version": "12.1",
        "gpu_name": "Synthetic GPU",
        "cudnn_version": 90100,
    }
    metrics.update(overrides.pop("metrics", {}))
    parameters.update(overrides.pop("parameters", {}))
    manifest = {
        "experiment": V.TELEMETRY_TRAIN_EXPERIMENT,
        "parameters": parameters,
        "metrics": metrics,
    }
    manifest.update(overrides)
    return manifest


# --- happy path ---------------------------------------------------------------


def test_valid_manifest_is_accepted():
    report = V.validate_telemetry_record(_manifest(), COUNT)
    assert report["epochs"] == 8
    assert report["validation_pairs"] == COUNT
    assert report["accounting_checked"] is True
    assert report["neural_replay"] is False
    assert report["statistical_replay"] is False
    assert report["source_import_closure_checked"] is False


def test_cudnn_version_may_be_null():
    V.validate_telemetry_record(_manifest(parameters={"cudnn_version": None}), COUNT)


# --- exact literals -----------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("experiment", "adaface-fixed8-native-cuda-training-cell"),
        ("experiment", "pair-contrastive-backbone-finetune"),
        ("experiment", None),
    ],
)
def test_experiment_must_be_telemetry_id(field, value):
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_experiment({**_manifest(), field: value})


@pytest.mark.parametrize(
    "key,value",
    [
        ("telemetry_version", "strong-same-validation-pass-telemetry-v1 "),
        ("telemetry_version", "other-v1"),
        ("device", "cpu"),
        ("device", "cuda:0"),
        ("threads", 2),
        ("threads", True),
        ("negative", "other"),
        ("torch_version", ""),
        ("cuda_version", 12.1),
        ("gpu_name", None),
        ("cudnn_version", "90100"),
    ],
)
def test_parameter_block_rejects_drift(key, value):
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_telemetry_parameters(_manifest(parameters={key: value})["parameters"])


def test_scope_literals_exact():
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_summary_metrics(
            _manifest(metrics={"validation_loss_scope": "other"})["metrics"], COUNT
        )
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_summary_metrics(
            _manifest(metrics={"training_loss_scope": "other"})["metrics"], COUNT
        )


def test_flags_must_be_exact_native_types():
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_summary_metrics(
            _manifest(metrics={"extra_validation_iterators": 0.0})["metrics"], COUNT
        )
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_summary_metrics(
            _manifest(metrics={"native_bitwise_equivalence_verified": 0})["metrics"], COUNT
        )
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_summary_metrics(
            _manifest(metrics={"native_bitwise_equivalence_verified": True})["metrics"], COUNT
        )


# --- eight ordered rows, exact native types -----------------------------------


def test_validation_history_requires_eight_rows():
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_validation_history(_validation_history()[:7], _history(), COUNT)


def test_validation_history_rejects_extra_field():
    rows = _validation_history()
    rows[0]["extra"] = 1.0
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_validation_history(rows, _history(), COUNT)


def test_validation_history_rejects_missing_field():
    rows = _validation_history()
    del rows[0]["validation_auc"]
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_validation_history(rows, _history(), COUNT)


def test_validation_history_rejects_epoch_order():
    rows = _validation_history()
    rows[3]["epoch"] = 9
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_validation_history(rows, _history(), COUNT)


@pytest.mark.parametrize("field", ["epoch", "validation_pairs"])
def test_integer_fields_must_be_native_int(field):
    rows = _validation_history()
    rows[0][field] = float(rows[0][field])
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_validation_history(rows, _history(), COUNT)


@pytest.mark.parametrize(
    "field",
    ["validation_loss", "validation_numerator", "validation_denominator", "validation_auc"],
)
def test_float_fields_must_be_native_float_not_int(field):
    rows = _validation_history()
    rows[0][field] = int(rows[0][field])
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_validation_history(rows, _history(), COUNT)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_rejected(bad):
    rows = _validation_history()
    rows[0]["validation_numerator"] = bad
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_validation_history(rows, _history(), COUNT)


# --- count / mass / loss identity / AUC coupling ------------------------------


def test_denominator_must_equal_count():
    rows = _validation_history()
    rows[0]["validation_denominator"] = float(COUNT + 1)
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_validation_history(rows, _history(), COUNT)


def test_validation_pairs_must_equal_count():
    rows = _validation_history()
    rows[0]["validation_pairs"] = COUNT + 1
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_validation_history(rows, _history(), COUNT)


def test_loss_must_equal_numerator_over_denominator():
    rows = _validation_history()
    rows[0]["validation_loss"] = rows[0]["validation_loss"] + 1e-9
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_validation_history(rows, _history(), COUNT)


def test_auc_must_equal_history_auc():
    rows = _validation_history()
    rows[0]["validation_auc"] = 0.5123
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_validation_history(rows, _history(), COUNT)


def test_auc_is_unit_bounded():
    rows = _validation_history(aucs=[1.2] + AUC[1:])
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_validation_history(rows, _history(), COUNT)


def test_metrics_validation_pairs_must_equal_count():
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_summary_metrics(
            _manifest(metrics={"validation_pairs": COUNT + 1})["metrics"], COUNT
        )


# --- no invented magnitude bound (root-review correction) ---------------------


def test_slightly_negative_loss_is_accounting_accepted():
    """Floating cosine rounding may push a near-zero loss below exact zero.

    The producer enforces only finiteness and the accounting identity, so this
    validator must not invent a nonnegative/upper loss gate.
    """
    rows = _validation_history()
    num = -1e-9
    den = float(COUNT)
    rows[0]["validation_numerator"] = num
    rows[0]["validation_denominator"] = den
    rows[0]["validation_loss"] = num / den
    V.validate_validation_history(rows, _history(), COUNT)


def test_loss_above_two_is_accounting_accepted():
    """No upper bound is invented; only finite + identity are required."""
    rows = _validation_history(numerator=500.0)
    V.validate_validation_history(rows, _history(), COUNT)


def test_all_records_are_exactly_reproducible_float_roundtrips():
    rows = _validation_history()
    for row in rows:
        assert row["validation_loss"] == row["validation_numerator"] / row["validation_denominator"]


# --- base history still checked -----------------------------------------------


def test_base_history_requires_four_fields():
    history = _history()
    history[0]["extra"] = 1.0
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_history(history)


def test_base_history_requires_nonnegative_magnitudes():
    history = _history()
    history[0]["mean_gradient_norm"] = -0.5
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_history(history)


def test_base_history_epochs_ordered():
    history = _history()
    history[5]["epoch"] = 7
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_history(history)


# --- source closure is a declaration, not a hash ------------------------------


def test_required_bindings_are_declared_not_verified():
    required = V.required_train_producer_bindings()
    assert "scripts/run_strong_validation_telemetry_cuda_v1.py" in required
    assert "scripts/strong_validation_telemetry_v1.py" in required
    # Old train producer must not be required by the telemetry train role.
    assert "scripts/run_strong_backbone_cuda_cell_v1.py" not in required


def test_validator_module_does_not_import_torch_or_numpy():
    source = _MODULE_PATH.read_text(encoding="utf-8")
    for banned in ("import torch", "import numpy", "from torch", "from numpy"):
        assert banned not in source


def test_report_never_claims_bitwise_or_replay():
    report = V.validate_telemetry_record(_manifest(), COUNT)
    assert report["native_bitwise_equivalence_verified"] is False
    assert report["neural_replay"] is False
    assert report["statistical_replay"] is False
    assert math.isfinite(report["epochs"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("training_complete", False),
        ("training_complete", 1),
        ("evaluation_complete", True),
        ("publication_ready", True),
    ],
)
def test_completion_and_publication_flags_refused(key, value):
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_telemetry_record(_manifest(metrics={key: value}), COUNT)


@pytest.mark.parametrize("value", [0, False, 1.0, None])
def test_unit_weight_policy_requires_native_zero_float(value):
    with pytest.raises(V.TelemetryAccountingError):
        V.validate_telemetry_record(_manifest(parameters={"gap_weight": value}), COUNT)


def test_refusal_never_echoes_untrusted_experiment():
    with pytest.raises(V.TelemetryAccountingError) as caught:
        V.validate_experiment({"experiment": "synthetic-private-marker"})
    assert "synthetic-private-marker" not in str(caught.value)
