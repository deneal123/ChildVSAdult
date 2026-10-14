"""Synthetic CPU-only fixture tests for the proposed weak fixed8 CUDA evaluator (v2).

Task ``weak-fixed8-evaluator-e405``, check date 2026-10-09.  Companion to
``evaluate_facenet_fixed8_cuda_v2.py`` (v1 files preserved byte-for-byte).

No GPU, CUDA probe, native inference, training, model load, crop-cache read, weight read,
download or large hashing happens here.  Every prerequisite is a synthetic object built to the
**real source-defined schema** or a ``monkeypatch`` stub.  The fixtures deliberately use the
full fifteen-field native epoch history, the exact seven-key native summary, explicit native
input records (weight + pairs) and native output records (checkpoint + summary) — not the v1
short-history / unbound placeholders.

The tests assert the v2 corrections:

* ``manifest.metrics`` is the full aggregate result (real ``overall`` / ``large_gap_25plus``
  blocks), equal to the standalone summary — not a seven-key projection;
* each original native ``summary.manifest.json`` is bound as an input;
* the exact canonical pairs file, the cached weight record among native inputs, the original
  summary among native outputs, and the source-defined native summary projection are required;
* the full fifteen-field history schema, typed and numerically closed, is enforced;
* the checkpoint requires facenet/faces/frozen_all/native-int-8/last_epoch and the exact history;
* all seeds share one exact learning rate; the full 650-crop pool and coverage are required;
* no CUDA probe or model forward occurs without ``--execute``; existing output is never
  overwritten; error messages never leak tensor names.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import evaluate_facenet_fixed8_cuda_v4 as evaluator

WEIGHT_SHA = "a" * 64
INIT_SHA = "b" * 64
PAIRS_PATH = "data/processed/pairs.jsonl"


# --------------------------------------------------------------------------- #
# realistic full-schema synthetic builders
# --------------------------------------------------------------------------- #
def synthetic_history(train_pairs=4, validation_pairs=4):
    """Eight epochs of the exact fifteen-field source-defined schema, all consistent."""
    rows = []
    for epoch in range(1, 9):
        train_loss = 1.0 / epoch
        validation_loss = 1.0 / (epoch + 1)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_numerator": train_loss * train_pairs,
                "train_denominator": float(train_pairs),
                "mean_gradient_norm": 0.01,
                "train_batches": -(-train_pairs // evaluator.BATCH_SIZE),
                "train_pairs": train_pairs,
                "validation_loss": validation_loss,
                "validation_numerator": validation_loss * validation_pairs,
                "validation_denominator": float(validation_pairs),
                "validation_auc": 0.5 + epoch / 100.0,
                "validation_pairs": validation_pairs,
                "epoch_seconds": float(epoch),
                "peak_allocated_bytes": 1000 * epoch,
                "peak_reserved_bytes": 2000 * epoch,
            }
        )
    return rows


def synthetic_summary(seed=42, history=None):
    return {
        "experiment": evaluator.NATIVE_EXPERIMENT,
        "seed": seed,
        "history": copy.deepcopy(history if history is not None else synthetic_history()),
        "training_complete": True,
        "evaluation_complete": False,
        "publication_ready": False,
        "limitation": evaluator.NATIVE_LIMITATION,
    }


def synthetic_checkpoint(history=None):
    return {
        "state_dict": {"last_linear.weight": torch.ones(2, 2)},
        "history": copy.deepcopy(history if history is not None else synthetic_history()),
        "selected_epoch": 8,
        "checkpoint_selection": "last_epoch",
        "backbone": "facenet",
        "crops_dir": "faces",
        "batchnorm_policy": "frozen_all",
    }


def weight_record():
    return {
        "path": "models/casia.pt",
        "bytes": 123,
        "sha256": WEIGHT_SHA,
    }


def pairs_record():
    return {
        "path": PAIRS_PATH,
        "bytes": 456,
        "sha256": "f" * 64,
    }


def synthetic_native(seed=42, lr=1e-6, tmp_path=None):
    run = Path(tmp_path) / f"run_s{seed}"
    history = synthetic_history()
    native = {
        "experiment": evaluator.NATIVE_EXPERIMENT,
        "parameters": {
            **evaluator.BASE_CONTRACT,
            "learning_rate": lr,
            "seed": seed,
            "model_init_state_sha256": INIT_SHA,
            "pretrained_weight_sha256": WEIGHT_SHA,
            "pairs_file": str((evaluator.PROJECT_ROOT / PAIRS_PATH).resolve()),
        },
        "metrics": {
            "history": history,
            "training_complete": True,
            "evaluation_complete": False,
            "publication_ready": False,
            "selected_epoch": 8,
        },
        "inputs": [weight_record(), pairs_record()],
        "outputs": [
            {"path": str(run / "private/checkpoint.pt")},
            {"path": str(run / "summary.json")},
        ],
    }
    return native, run


def _bind_native_files(monkeypatch, saved, summary):
    """Stub file_record so native outputs 'match', and torch.load for the checkpoint."""
    monkeypatch.setattr(evaluator, "file_record", lambda path: pairs_record() if Path(path) == evaluator.NATIVE_PAIRS else {"path": str(path)})
    monkeypatch.setattr(evaluator.torch, "load", lambda *a, **k: saved)
    import json as _json

    monkeypatch.setattr(
        evaluator.Path,
        "read_bytes",
        lambda self: _json.dumps(summary).encode("utf-8"),
        raising=False,
    )


def _accept_native(monkeypatch, tmp_path, saved=None, summary=None):
    native, run = synthetic_native(tmp_path=tmp_path)
    _bind_native_files(
        monkeypatch, saved or synthetic_checkpoint(), summary or synthetic_summary()
    )
    return native, run


# --------------------------------------------------------------------------- #
# bindings / seeds
# --------------------------------------------------------------------------- #
def test_single_and_triple_bindings_accepted(tmp_path):
    assert len(evaluator.canonical_bindings([tmp_path / "a"])) == 1
    assert len(evaluator.canonical_bindings([tmp_path / f"s{s}" for s in (42, 1, 2)])) == 3


@pytest.mark.parametrize("count", [0, 2, 4])
def test_wrong_binding_count_rejected(count):
    with pytest.raises(ValueError, match="one or three"):
        evaluator.canonical_bindings([Path(f"run{i}") for i in range(count)])


def test_duplicate_binding_path_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        evaluator.canonical_bindings([Path("run"), Path("other"), Path("run")])


@pytest.mark.parametrize("seed", [42, 1, 2])
def test_single_canonical_seed_accepted(seed):
    assert evaluator.require_seed_set([seed]) == [seed]


def test_full_seed_set_accepted():
    assert set(evaluator.require_seed_set([42, 1, 2])) == {42, 1, 2}


@pytest.mark.parametrize(
    "seeds", [[42, 1], [42, 1, 2, 3], [42, 1, 1], [True], [42.0], [0], [-1], ["42"]]
)
def test_bad_seed_set_rejected(seeds):
    with pytest.raises(ValueError, match="seed"):
        evaluator.require_seed_set(seeds)


# --------------------------------------------------------------------------- #
# typed equality
# --------------------------------------------------------------------------- #
def test_strict_equal_never_confuses_bool_and_number():
    assert evaluator.strict_equal(1, 1)
    assert not evaluator.strict_equal(True, 1)
    assert not evaluator.strict_equal(1, 1.0)
    assert not evaluator.strict_equal(1.0, 1)
    assert evaluator.strict_equal({"a": [1, 2]}, {"a": [1, 2]})
    assert not evaluator.strict_equal({"a": [1, 2]}, {"a": [2, 1]})


def test_freeze_value_sorts_mappings_and_keeps_sequence_order():
    assert evaluator.freeze_value({"b": 1, "a": 2}) == {"a": 2, "b": 1}
    assert evaluator.freeze_value([1, 2]) != evaluator.freeze_value([2, 1])


# --------------------------------------------------------------------------- #
# full fifteen-field history schema
# --------------------------------------------------------------------------- #
def test_finite_history_accepts_full_schema():
    assert len(evaluator.finite_history(synthetic_history())) == 8


def test_history_field_set_is_exactly_fifteen_source_fields():
    assert set(synthetic_history()[0]) == set(evaluator.HISTORY_FIELDS)
    assert len(evaluator.HISTORY_FIELDS) == 15


@pytest.mark.parametrize(
    "mutate",
    [
        "short",
        "extra_field",
        "missing_field",
        "epoch_order",
        "epoch_bool",
        "count_float",
        "count_bool",
        "nonfinite_loss",
        "auc_out_of_range",
        "auc_bool",
        "quotient_mismatch",
        "mass_not_count",
        "batches_not_ceiling",
        "varying_pairs",
        "zero_mass",
    ],
)
def test_finite_history_rejects_schema_violations(mutate):
    history = synthetic_history()
    if mutate == "short":
        history = history[:7]
    elif mutate == "extra_field":
        history[0]["extra"] = 1
    elif mutate == "missing_field":
        del history[0]["validation_auc"]
    elif mutate == "epoch_order":
        history[3]["epoch"] = 9
    elif mutate == "epoch_bool":
        history[2]["epoch"] = True
    elif mutate == "count_float":
        history[1]["train_pairs"] = 4.0
    elif mutate == "count_bool":
        history[1]["train_batches"] = True
    elif mutate == "nonfinite_loss":
        history[5]["validation_loss"] = float("nan")
    elif mutate == "auc_out_of_range":
        history[4]["validation_auc"] = 1.5
    elif mutate == "auc_bool":
        history[4]["validation_auc"] = True
    elif mutate == "quotient_mismatch":
        history[2]["validation_loss"] = 0.123
    elif mutate == "mass_not_count":
        history[3]["train_denominator"] = 3.0
    elif mutate == "batches_not_ceiling":
        history[0]["train_batches"] = 2
    elif mutate == "varying_pairs":
        history[6]["train_pairs"] = 2
        history[6]["train_numerator"] = history[6]["train_loss"] * 2
    elif mutate == "zero_mass":
        history[7]["validation_denominator"] = 0.0
    with pytest.raises(ValueError):
        evaluator.finite_history(history)


def test_history_batch_ceiling_uses_batch16():
    history = synthetic_history(train_pairs=17)
    assert history[0]["train_batches"] == 2
    evaluator.finite_history(history)
    bad = synthetic_history(train_pairs=17)
    bad[0]["train_batches"] = 1
    with pytest.raises(ValueError, match="ceil"):
        evaluator.finite_history(bad)


# --------------------------------------------------------------------------- #
# native summary projection
# --------------------------------------------------------------------------- #
def test_require_native_summary_accepts_exact_seven_keys():
    assert tuple(evaluator.require_native_summary(synthetic_summary())) == (
        evaluator.NATIVE_SUMMARY_KEYS
    )


@pytest.mark.parametrize(
    "mutate",
    ["extra", "missing", "experiment", "seed", "history", "evaluation", "publication", "limitation"],
)
def test_require_native_summary_rejects_drift(mutate):
    summary = synthetic_summary()
    if mutate == "extra":
        summary["extra"] = 1
    elif mutate == "missing":
        del summary["limitation"]
    elif mutate == "experiment":
        summary["experiment"] = "pair-contrastive-backbone-finetune"
    elif mutate == "seed":
        summary["seed"] = 3
    elif mutate == "history":
        summary["history"][0]["train_loss"] = 0.999
    elif mutate == "evaluation":
        summary["evaluation_complete"] = True
    elif mutate == "publication":
        summary["publication_ready"] = True
    else:
        summary["limitation"] = "recovered legacy weak history"
    with pytest.raises(ValueError):
        evaluator.require_native_summary(summary)


# --------------------------------------------------------------------------- #
# native contract
# --------------------------------------------------------------------------- #
def test_native_contract_accepts_fresh_weak_v2(monkeypatch, tmp_path):
    native, run = _accept_native(monkeypatch, tmp_path)
    cell = evaluator.native_contract(native, run, weight_record())
    assert cell["seed"] == 42
    assert cell["learning_rate"] == 1e-6
    assert cell["checkpoint"] == run / "private/checkpoint.pt"


@pytest.mark.parametrize(
    "experiment",
    [
        "pair-contrastive-backbone-finetune",
        "facenet-weak-legacy-recovery-v1",
        "facenet-fixed8-validation-loss-v1",
    ],
)
def test_native_contract_rejects_other_lineages(monkeypatch, tmp_path, experiment):
    native, run = _accept_native(monkeypatch, tmp_path)
    native["experiment"] = experiment
    with pytest.raises(ValueError, match="weak fixed8 v2"):
        evaluator.native_contract(native, run, weight_record())


@pytest.mark.parametrize(
    "key,value",
    [
        ("trainable_scope", "full"),
        ("trainable_scope", "tail"),
        ("epochs_executed", 8.0),
        ("batch_size", 32),
        ("batchnorm_policy", "train"),
        ("checkpoint_selection", "best"),
        ("device", "cpu"),
        ("gap_weight", 0.5),
        ("margin", 0.5),
    ],
)
def test_native_contract_rejects_parameter_drift(monkeypatch, tmp_path, key, value):
    native, run = _accept_native(monkeypatch, tmp_path)
    native["parameters"][key] = value
    with pytest.raises(ValueError):
        evaluator.native_contract(native, run, weight_record())


@pytest.mark.parametrize("lr", [3e-5, 1e-7, 0, None, True])
def test_native_contract_rejects_bad_lr(monkeypatch, tmp_path, lr):
    native, run = _accept_native(monkeypatch, tmp_path)
    native["parameters"]["learning_rate"] = lr
    with pytest.raises(ValueError, match="learning rate"):
        evaluator.native_contract(native, run, weight_record())


@pytest.mark.parametrize("seed", [3, 0, True, "42"])
def test_native_contract_rejects_bad_seed(monkeypatch, tmp_path, seed):
    native, run = _accept_native(monkeypatch, tmp_path)
    native["parameters"]["seed"] = seed
    with pytest.raises(ValueError, match="seed"):
        evaluator.native_contract(native, run, weight_record())


@pytest.mark.parametrize(
    "pairs", [None, "data/processed/experiments/pairs_lookalike.jsonl", "/tmp/pairs.jsonl"]
)
def test_native_contract_requires_exact_canonical_pairs(monkeypatch, tmp_path, pairs):
    native, run = _accept_native(monkeypatch, tmp_path)
    native["parameters"]["pairs_file"] = pairs
    with pytest.raises(ValueError, match="pairs_file"):
        evaluator.native_contract(native, run, weight_record())


def test_native_contract_requires_true_completion(monkeypatch, tmp_path):
    native, run = _accept_native(monkeypatch, tmp_path)
    native["metrics"]["training_complete"] = False
    with pytest.raises(ValueError, match="completion"):
        evaluator.native_contract(native, run, weight_record())


def test_native_contract_requires_native_int_selected_epoch(monkeypatch, tmp_path):
    native, run = _accept_native(monkeypatch, tmp_path)
    native["metrics"]["selected_epoch"] = 8.0
    with pytest.raises(ValueError, match="native int eight"):
        evaluator.native_contract(native, run, weight_record())


def test_native_contract_requires_exact_metrics_keys(monkeypatch, tmp_path):
    native, run = _accept_native(monkeypatch, tmp_path)
    native["metrics"]["extra"] = 1
    with pytest.raises(ValueError, match="five native keys"):
        evaluator.native_contract(native, run, weight_record())


def test_native_contract_rejects_completed_evaluation_flag(monkeypatch, tmp_path):
    native, run = _accept_native(monkeypatch, tmp_path)
    native["metrics"]["evaluation_complete"] = True
    with pytest.raises(ValueError, match="incomplete"):
        evaluator.native_contract(native, run, weight_record())


def test_native_contract_requires_weight_record_among_inputs(monkeypatch, tmp_path):
    native, run = _accept_native(monkeypatch, tmp_path)
    native["inputs"] = [pairs_record()]
    with pytest.raises(ValueError, match="cached weight record"):
        evaluator.native_contract(native, run, weight_record())


def test_native_contract_requires_pairs_among_inputs(monkeypatch, tmp_path):
    native, run = _accept_native(monkeypatch, tmp_path)
    native["inputs"] = [weight_record()]
    with pytest.raises(ValueError, match="pair file"):
        evaluator.native_contract(native, run, weight_record())


def test_native_contract_rejects_hash_string_without_input_record(monkeypatch, tmp_path):
    """A matching parameters hash is not sufficient; the actual input record is required."""
    native, run = _accept_native(monkeypatch, tmp_path)
    native["inputs"] = [pairs_record()]  # parameters still carry the right sha
    with pytest.raises(ValueError, match="cached weight record"):
        evaluator.native_contract(native, run, weight_record())


def test_native_contract_requires_bound_checkpoint(monkeypatch, tmp_path):
    native, run = _accept_native(monkeypatch, tmp_path)
    native["outputs"] = [native["outputs"][1]]
    with pytest.raises(ValueError, match="checkpoint"):
        evaluator.native_contract(native, run, weight_record())


def test_native_contract_requires_bound_original_summary(monkeypatch, tmp_path):
    native, run = _accept_native(monkeypatch, tmp_path)
    native["outputs"] = [native["outputs"][0]]
    with pytest.raises(ValueError, match="standalone summary"):
        evaluator.native_contract(native, run, weight_record())


def test_native_contract_propagates_bad_original_summary(monkeypatch, tmp_path):
    bad = synthetic_summary()
    bad["limitation"] = "recovered legacy weak history"
    native, run = _accept_native(monkeypatch, tmp_path, summary=bad)
    with pytest.raises(ValueError, match="limitation"):
        evaluator.native_contract(native, run, weight_record())


# --------------------------------------------------------------------------- #
# weights-only checkpoint contract
# --------------------------------------------------------------------------- #
def test_checkpoint_contract_accepts_last_epoch():
    history = synthetic_history()
    assert evaluator.checkpoint_contract(synthetic_checkpoint(history), history)


@pytest.mark.parametrize(
    "mutate",
    ["backbone", "crops", "batchnorm", "epoch_float", "selection", "history", "empty", "nonfinite", "no_state"],
)
def test_checkpoint_contract_rejects_mismatch(mutate):
    history = synthetic_history()
    saved = synthetic_checkpoint(history)
    if mutate == "backbone":
        saved["backbone"] = "adaface_ir101"
    elif mutate == "crops":
        saved["crops_dir"] = "faces_mtcnn"
    elif mutate == "batchnorm":
        saved["batchnorm_policy"] = "train"
    elif mutate == "epoch_float":
        saved["selected_epoch"] = 8.0
    elif mutate == "selection":
        saved["checkpoint_selection"] = "best"
    elif mutate == "history":
        saved["history"][0]["train_loss"] = 0.123
    elif mutate == "empty":
        saved["state_dict"] = {}
    elif mutate == "nonfinite":
        saved["state_dict"] = {"layer.weight": torch.tensor([float("nan")])}
    else:
        del saved["state_dict"]
    with pytest.raises(ValueError):
        evaluator.checkpoint_contract(saved, history)


def test_checkpoint_nonfinite_error_does_not_leak_tensor_names():
    saved = synthetic_checkpoint()
    saved["state_dict"] = {"secret.head.weight": torch.tensor([float("inf")])}
    with pytest.raises(ValueError) as excinfo:
        evaluator.checkpoint_contract(saved, synthetic_history())
    assert "secret.head.weight" not in str(excinfo.value)


# --------------------------------------------------------------------------- #
# shared frozen initialisation + shared exact LR
# --------------------------------------------------------------------------- #
def _native(seed, init=INIT_SHA, weight=WEIGHT_SHA, lr=1e-6):
    return {
        "parameters": {
            "seed": seed,
            "model_init_state_sha256": init,
            "pretrained_weight_sha256": weight,
            "learning_rate": lr,
        }
    }


def test_shared_init_accepts_identical_bytes_and_lr():
    shared = evaluator.require_shared_init(
        [_native(42), _native(1), _native(2)], weight_record()
    )
    assert shared["model_init_state_sha256"] == INIT_SHA
    assert shared["pretrained_weight_sha256"] == WEIGHT_SHA
    assert shared["learning_rate"] == 1e-6


def test_shared_init_rejects_differing_init():
    with pytest.raises(ValueError, match="share one frozen initialisation"):
        evaluator.require_shared_init([_native(42), _native(1, init="d" * 64)], weight_record())


def test_shared_init_rejects_differing_weight():
    with pytest.raises(ValueError, match="cached weight"):
        evaluator.require_shared_init(
            [_native(42), _native(1, weight="e" * 64)], weight_record()
        )


def test_shared_init_rejects_differing_learning_rate():
    with pytest.raises(ValueError, match="exact learning rate"):
        evaluator.require_shared_init(
            [_native(42, lr=1e-6), _native(1, lr=1e-5)], weight_record()
        )


# --------------------------------------------------------------------------- #
# aggregate metrics = full result (v2 correction #1)
# --------------------------------------------------------------------------- #
def test_aggregate_metrics_is_full_result_not_projection():
    result = {
        "experiment": "x",
        "overall": {"frozen": {"roc_auc": 0.5}},
        "large_gap_25plus": {"frozen": {"roc_auc": 0.6}},
    }
    assert evaluator.aggregate_metrics(result) == result
    assert "overall" in evaluator.aggregate_metrics(result)
    assert "large_gap_25plus" in evaluator.aggregate_metrics(result)


# --------------------------------------------------------------------------- #
# full pool + coverage
# --------------------------------------------------------------------------- #
def _pool(n):
    return np.zeros((n, 112, 112, 3), dtype=np.uint8)


def test_require_full_pool_accepts_650():
    evaluator.require_full_pool(_pool(650), np.zeros(650, int))


@pytest.mark.parametrize("n", [649, 651, 0])
def test_require_full_pool_rejects_non_650(n):
    with pytest.raises(ValueError, match="650|full actual"):
        evaluator.require_full_pool(_pool(n), np.zeros(n, int))


def test_require_full_pool_rejects_bad_dtype_and_shape():
    with pytest.raises(ValueError, match="BGR112"):
        evaluator.require_full_pool(np.zeros((650, 112, 112, 3), np.float32), np.zeros(650, int))
    with pytest.raises(ValueError, match="BGR112"):
        evaluator.require_full_pool(np.zeros((650, 160, 160, 3), np.uint8), np.zeros(650, int))


def test_coverage_accounting_allows_canonical_unmatched_positives():
    full = {
        "source_positives": 10,
        "retained_positives": 10,
        "source_25plus_positives": 4,
        "retained_25plus_positives": 4,
    }
    assert evaluator.coverage_complete(full)
    dropped = dict(full, retained_positives=9)
    assert evaluator.coverage_complete(dropped)
    assert not evaluator.coverage_complete(dict(full, retained_positives=11))
    assert not evaluator.coverage_complete(dict(full, source_positives=0))


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
def _unit(n, dim=512, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_validate_embeddings_accepts_full_unit_pool():
    assert evaluator.validate_embeddings(_unit(650), 650).shape == (650, 512)


@pytest.mark.parametrize("mutate", ["shape", "nonfinite", "unnormalized", "dtype"])
def test_validate_embeddings_rejects_incomplete(mutate):
    v = _unit(10)
    if mutate == "shape":
        v = v[:9]
    elif mutate == "nonfinite":
        v[3, 0] = float("nan")
    elif mutate == "unnormalized":
        v[2] *= 2
    else:
        v = (v * 100).astype(np.int64)
    with pytest.raises(ValueError, match="512-d"):
        evaluator.validate_embeddings(v, 10)


# --------------------------------------------------------------------------- #
# validation-loss diagnostic (v2 optional section)
# --------------------------------------------------------------------------- #
def test_validation_measurement_accepts_canonical_fields():
    measured = {
        "validation_loss": 0.25,
        "validation_numerator": 1.0,
        "validation_denominator": 4.0,
        "validation_auc": 0.75,
        "validation_pairs": 4,
    }
    out = evaluator.validation_measurement(measured)
    assert out["validation_pairs"] == 4
    assert "retrospective" in out["binding"]


@pytest.mark.parametrize(
    "mutate", ["missing", "zero_mass", "quotient", "auc"]
)
def test_validation_measurement_rejects_invalid(mutate):
    measured = {
        "validation_loss": 0.25,
        "validation_numerator": 1.0,
        "validation_denominator": 4.0,
        "validation_auc": 0.75,
        "validation_pairs": 4,
    }
    if mutate == "missing":
        del measured["validation_auc"]
    elif mutate == "zero_mass":
        measured["validation_denominator"] = 0.0
    elif mutate == "quotient":
        measured["validation_loss"] = 0.9
    else:
        measured["validation_auc"] = 1.2
    with pytest.raises(ValueError):
        evaluator.validation_measurement(measured)


# --------------------------------------------------------------------------- #
# provenance / inputs (v2 correction #2)
# --------------------------------------------------------------------------- #
def test_source_paths_bind_original_producer_and_strong_evaluator():
    paths = {p.as_posix() for p in evaluator.source_paths()}
    assert any(p.endswith("scripts/run_facenet_validation_cuda_v2.py") for p in paths)
    assert any(p.endswith("scripts/evaluate_strong_cuda_fgnet_v1.py") for p in paths)
    assert any(p.endswith("tests/test_facenet_validation_cuda_v2.py") for p in paths)


def test_native_manifests_resolves_one_manifest_per_binding(tmp_path):
    bindings = [tmp_path / f"s{s}" for s in (42, 1, 2)]
    manifests = evaluator.native_manifests(bindings)
    assert manifests == [b / "summary.manifest.json" for b in bindings]


def test_build_inputs_binds_each_original_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluator, "source_paths", lambda: [tmp_path / "src.py"])
    monkeypatch.setattr(
        evaluator, "paths_from", lambda native: [tmp_path / "ancestry.py", tmp_path / "src.py"]
    )
    bindings = [tmp_path / "s42"]
    inputs = [Path(i) for i in evaluator.build_inputs(
        bindings=bindings,
        natives=[object()],
        weight=tmp_path / "w.pt",
        cache=tmp_path / "crops.npz",
    )]
    assert len(inputs) == len(set(inputs))
    assert tmp_path / "s42" / "summary.manifest.json" in inputs
    assert evaluator.NATIVE_PAIRS in inputs
    assert tmp_path / "ancestry.py" in inputs


def test_run_directory_accepts_dir_or_manifest(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    assert evaluator.run_directory(directory) == directory
    assert evaluator.run_directory(directory / "summary.manifest.json") == directory


# --------------------------------------------------------------------------- #
# fail-closed execution and no-overwrite
# --------------------------------------------------------------------------- #
def _stub_weight_and_readiness(monkeypatch, tmp_path, natives=()):
    monkeypatch.setattr(evaluator, "require_pretrained_weight", lambda path: Path(path))
    monkeypatch.setattr(evaluator, "resolve_facenet_casia_weight", lambda: tmp_path / "w.pt")
    monkeypatch.setattr(evaluator, "inventory_weight_record", lambda path: {})
    monkeypatch.setattr(
        evaluator, "validate_weight_record", lambda w, r: {"sha256": WEIGHT_SHA}
    )
    monkeypatch.setattr(
        evaluator, "readiness", lambda bindings, **kwargs: (list(natives), {}, {})
    )


def test_existing_output_directory_is_never_overwritten(monkeypatch, tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("do not touch", encoding="utf-8")
    args = SimpleNamespace(
        bindings=[tmp_path / "run"], out=out, execute=False, measure_validation_loss=False
    )
    with pytest.raises(FileExistsError, match="never overwrite"):
        evaluator.run(args)
    assert sentinel.read_text(encoding="utf-8") == "do not touch"


def test_dry_run_probes_no_cuda_and_writes_nothing(monkeypatch, tmp_path):
    _stub_weight_and_readiness(monkeypatch, tmp_path)

    def forbidden():
        raise AssertionError("dry run must not probe CUDA")

    monkeypatch.setattr(evaluator.torch.cuda, "is_available", forbidden)
    args = SimpleNamespace(
        bindings=[tmp_path / "run"],
        out=tmp_path / "fresh",
        execute=False,
        measure_validation_loss=False,
    )
    assert evaluator.run(args) == Path()
    assert not args.out.exists()


def test_execute_requires_cuda_with_no_cpu_fallback(monkeypatch, tmp_path):
    _stub_weight_and_readiness(monkeypatch, tmp_path)
    monkeypatch.setattr(evaluator.torch.cuda, "is_available", lambda: False)
    args = SimpleNamespace(
        bindings=[tmp_path / "run"],
        out=tmp_path / "fresh",
        execute=True,
        measure_validation_loss=False,
    )
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        evaluator.run(args)
    assert not args.out.exists()
    assert not (args.out / "summary.json").exists()


# --------------------------------------------------------------------------- #
# embed_pool batching (CPU synthetic tensors, no download)
# --------------------------------------------------------------------------- #
class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(3, 512)

    def preprocess(self, image, bgr=True):
        return np.zeros((3, 160, 160), dtype=np.float32)

    def forward(self, x):
        z = self.net(x.mean(dim=(2, 3)))
        return torch.nn.functional.normalize(z, dim=-1)


def test_embed_pool_covers_every_crop_and_returns_unit_rows():
    model = _FakeModel().eval()
    crops = np.zeros((5, 112, 112, 3), dtype=np.uint8)
    vectors = evaluator.embed_pool(model, crops, device="cpu", batch_size=2)
    assert vectors.shape == (5, 512)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_load_checkpoint_state_is_strict(monkeypatch):
    """v2 correction #5: full checkpoint state loads with strict=True."""
    model = torch.nn.Linear(2, 2)
    calls = {}

    def fake_load_state_dict(state, strict=False):
        calls["strict"] = strict
        return torch.nn.modules.module._IncompatibleKeys([], [])

    monkeypatch.setattr(evaluator.torch, "load", lambda *a, **k: {"state_dict": {}})
    monkeypatch.setattr(model, "load_state_dict", fake_load_state_dict)
    evaluator.load_checkpoint_state(model, Path("ckpt.pt"))
    assert calls["strict"] is True


@pytest.mark.parametrize("with_validation", [False, True])
def test_mocked_complete_run_publishes_all_cells_without_lifetime_error(
    monkeypatch, tmp_path, with_validation
):
    """Control-flow fixture only: no real weights, inference or scientific evidence."""
    import json

    from scripts import reevaluate_fgnet_metrics_v2 as protocol_module

    _stub_weight_and_readiness(monkeypatch, tmp_path)
    selected = {42: tmp_path / "fake-checkpoint.pt"}
    monkeypatch.setattr(
        evaluator, "readiness",
        lambda *a, **k: ([], selected, {"model_init_state_sha256": "fixture"}),
    )
    cache = tmp_path / "fixture.npz"
    np.savez_compressed(
        cache, crops=np.zeros((650, 112, 112, 3), dtype=np.uint8),
        subjects=np.arange(650), ages=np.full(650, 30),
    )
    monkeypatch.setattr(evaluator, "FGNET_CACHE", cache)
    monkeypatch.setattr(evaluator, "build_inputs", lambda **k: [cache])
    monkeypatch.setattr(evaluator, "snapshot", lambda paths: [{"fixture": "stable"}])
    coverage = {
        "source_positives": 2659, "retained_positives": 2654,
        "source_25plus_positives": 215,
        "retained_25plus_positives": 215,
    }
    assert evaluator.coverage_complete(coverage)
    monkeypatch.setattr(
        protocol_module, "protocol",
        lambda *a: (np.array([0, 1]), np.array([2, 3]),
                    np.array([1, 0]), np.array([30, 30]), coverage),
    )

    class FixtureModel:
        def state_dict(self):
            return {}

        def to(self, device):
            assert device == "cuda"
            return self

        def eval(self):
            return self

    monkeypatch.setattr(evaluator, "build_architecture", FixtureModel)
    monkeypatch.setattr(evaluator, "state_dict_digest", lambda s: "fixture")
    monkeypatch.setattr(evaluator, "load_checkpoint_state", lambda model, p: model)
    vectors = np.zeros((650, 512), dtype=np.float32)
    vectors[:, 0] = 1.0
    monkeypatch.setattr(evaluator, "embed_pool", lambda *a, **k: vectors.copy())
    calls = []
    monkeypatch.setattr(evaluator, "infer", lambda scores, *a, **kw: {
        "fixture_only": True, "models": sorted(scores),
    })
    monkeypatch.setattr(evaluator, "measure_validation_loss", lambda *a: {
        "validation_loss": 0.25, "validation_numerator": 1.0,
        "validation_denominator": 4.0, "validation_auc": 0.75,
        "validation_pairs": 4,
    })
    monkeypatch.setattr(evaluator.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(evaluator.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(evaluator.torch.cuda, "empty_cache", lambda: calls.append("release"))
    monkeypatch.setattr(evaluator.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(evaluator.torch.cuda, "max_memory_reserved", lambda: 0)

    def fixture_manifest(target, **kwargs):
        target.write_text(json.dumps({"metrics": kwargs["metrics"]}), encoding="utf-8")

    monkeypatch.setattr(evaluator, "write_experiment_manifest", fixture_manifest)
    monkeypatch.setattr(evaluator, "validate_written_inputs", lambda *a: None)
    out = tmp_path / "fresh"
    target = evaluator.run(SimpleNamespace(
        bindings=[tmp_path / "s42"], out=out, execute=True,
        measure_validation_loss=with_validation,
    ))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert json.loads(target.read_text(encoding="utf-8"))["metrics"] == summary
    assert summary["actual_cells"] == ["tuned_s42"]
    assert summary["validation_loss_measured"] is with_validation
    assert len(summary["validation_loss_diagnostic"]) == (2 if with_validation else 0)
    assert calls == ["release", "release"]
    assert len(list((out / "private").glob("embeddings_*.npz"))) == 2
    assert summary["publication_ready"] is False


def test_declared_bootstrap_values_match_executed_current_defaults():
    """Current v3 defaults coincide; future changes must not silently drift."""
    import inspect

    signature = inspect.signature(evaluator.infer)
    assert signature.parameters["n_boot"].default == evaluator.N_BOOT
    assert signature.parameters["seed"].default == evaluator.BOOTSTRAP_SEED


def test_source_protocol_constants_match_current_manifest_parameters(monkeypatch):
    from scripts import reevaluate_fgnet_metrics_v2 as source

    calls = {}

    def matching(subjects, ages, positives, **kwargs):
        calls.update(kwargs)
        return [(0, 2, 3, 30, 0, 0)]

    monkeypatch.setattr(source, "_match_negatives", matching)
    source.protocol(np.array([0, 0, 1, 2]), np.array([5, 35, 6, 36]))
    assert calls == {"seed": evaluator.ROC_V2_PROTOCOL_SEED,
                     "tolerance": evaluator.ROC_V2_TOLERANCE}



def test_v4_declares_known_protocol_import_helpers_and_v3_ancestry():
    required = {"scripts/fgnet_retrieval_study.py", "scripts/render_fgnet_evidence.py",
                "scripts/evaluate_facenet_fixed8_cuda_v3.py"}
    assert required <= set(evaluator.SOURCE_DEPENDENCIES)
    assert evaluator.OUTPUT_EXPERIMENT.endswith("-v4")


def test_v4_inference_keywords_are_explicit_not_default_dependent():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(evaluator.run))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "infer"]
    assert len(calls) == 1
    assert {kw.arg: kw.value.id for kw in calls[0].keywords} == {
        "n_boot": "N_BOOT", "seed": "BOOTSTRAP_SEED"}


def test_v4_dry_run_wording_discloses_cpu_deserialization():
    import inspect

    source = inspect.getsource(evaluator.run)
    assert "weights-only CPU checkpoint deserialization" in source
    assert "no neural forward" in source
    assert "no model load" not in source


def test_v4_transitive_literal_script_imports_have_declared_source_files():
    """Static project-helper coverage only, not runtime/dynamic-package certification."""
    import ast

    bound = {path.resolve() for path in evaluator.source_paths()}
    pending, seen = list(bound), set()
    while pending:
        path = pending.pop()
        if path in seen or path.suffix != ".py":
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
                if node.module == "scripts":
                    modules.extend(f"scripts.{alias.name}" for alias in node.names)
            for module in modules:
                if module.startswith("scripts."):
                    helper = evaluator.PROJECT_ROOT.joinpath(*module.split(".")).with_suffix(".py")
                    if helper.is_file():
                        assert helper.resolve() in bound
                        pending.append(helper.resolve())
