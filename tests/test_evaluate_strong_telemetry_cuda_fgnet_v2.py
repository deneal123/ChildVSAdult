"""Synthetic tests for the draft SEPARATE telemetry evaluator (source only).

No native model/data/checkpoint/weight access, no training, no inference, no
network. Only generated in-memory fixtures plus monkeypatched ``verified`` and
``file_record`` (and ``contract`` for readiness isolation). Numerical main() is
never executed past the CUDA guard.
"""

from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path

import pytest

from scripts import evaluate_strong_telemetry_cuda_fgnet_v2 as ev
from scripts import strong_telemetry_accounting_v1 as V

DRAFT = Path(ev.__file__)

LEGACY_NATIVE = "adaface-fixed8-native-cuda-training-cell"
LEGACY_EXTERNAL = "adaface-fixed8-cuda-full-fgnet-roc-v2"
COUNT = 53
AUC = [0.50, 0.55, 0.60, 0.62, 0.70, 0.73, 0.80, 0.81]


def rec(path):
    return {"path": str(path), "bytes": 1, "sha256": "a" * 64}


def histories():
    base = [
        {"epoch": i, "training_loss": 0.5 + 0.01 * i, "validation_auc": AUC[i - 1],
         "mean_gradient_norm": 0.1 * i, "epoch_seconds": 1.0 + i}
        for i in range(1, V.EPOCHS + 1)
    ]
    tel = []
    for i in range(1, V.EPOCHS + 1):
        num = 2.0 + i
        den = float(COUNT)
        tel.append({"epoch": i, "validation_loss": num / den, "validation_numerator": num,
                    "validation_denominator": den, "validation_pairs": COUNT,
                    "validation_auc": AUC[i - 1]})
    return base, tel


def telemetry_metrics():
    base, tel = histories()
    return {
        "training_complete": True,
        "evaluation_complete": False,
        "publication_ready": False,
        "history": base,
        "validation_history": tel,
        "validation_loss_scope": V.VALIDATION_LOSS_SCOPE,
        "training_loss_scope": V.TRAINING_LOSS_SCOPE,
        "extra_validation_iterators": 0,
        "native_bitwise_equivalence_verified": False,
        "validation_pairs": COUNT,
    }


def telemetry_parameters():
    return {
        "telemetry_version": V.TELEMETRY_VERSION,
        "device": "cuda",
        "gap_weight": 0.0,
        "threads": 1,
        "negative": "random",
        "torch_version": "2.12.0+cu132",
        "cuda_version": "13.2",
        "gpu_name": "Synthetic GPU",
        "cudnn_version": 90100,
    }


def _component_params():
    return {
        "backbone": "adaface_ir101", "epochs_requested": 8, "epochs_executed": 8,
        "selected_epoch": 8, "checkpoint_selection": "last_epoch",
        "batchnorm_policy": "frozen_all", "trainable_scope": "head",
        "learning_rate": 1e-6, "batch_size": 16, "margin": 0.3, "gap_weight": 0.0,
        "crops_dir": "faces", "seed": 42, "pairs_file": "data/processed/pairs.jsonl",
    }


def fixture(monkeypatch):
    native_path = Path("cell/summary.manifest.json")
    checkpoint = native_path.parent / "private/checkpoint.pt"
    component_path = checkpoint.with_suffix(".manifest.json")
    component = {
        "schema_version": 1,
        "experiment": ev.COMPONENT_EXPERIMENT,
        "parameters": _component_params(),
        "inputs": [],
        "outputs": [rec(checkpoint)],
    }
    producer_inputs = [rec(rel) for rel in V.required_train_producer_bindings()
                       if rel != "models/adaface_ir101.pt"]
    producer_inputs.append(rec(ev.PROJECT_ROOT / "models/adaface_ir101.pt"))
    native = {
        "schema_version": 1,
        "experiment": ev.TELEMETRY_NATIVE_EXPERIMENT,
        "parameters": _component_params() | telemetry_parameters(),
        "metrics": telemetry_metrics(),
        "inputs": producer_inputs,
        "outputs": [rec(checkpoint), rec(component_path)],
    }
    seen = []

    def fake_verified(path, experiment):
        seen.append((path, experiment))
        if path == native_path:
            return native
        if path == component_path:
            return component
        raise FileNotFoundError("unknown synthetic manifest")

    monkeypatch.setattr(ev, "verified", fake_verified)
    monkeypatch.setattr(ev, "file_record", lambda p: rec(p))
    monkeypatch.setattr(ev, "contract", lambda *a, **k: None)
    return native_path, native, component, seen


def test_readiness_accepts_linked_telemetry_cell(monkeypatch):
    path, native, _, seen = fixture(monkeypatch)
    selected, natives = ev.readiness([path])
    assert list(selected) == ["random_head_lr1e-06_s42"]
    assert natives == [native]
    assert (path, ev.TELEMETRY_NATIVE_EXPERIMENT) in seen
    assert all(experiment != LEGACY_NATIVE for _, experiment in seen)


def test_readiness_invokes_telemetry_accounting_checker(monkeypatch):
    path, native, _, _ = fixture(monkeypatch)
    native["metrics"]["validation_history"][0:1] = []  # seven rows now
    with pytest.raises(V.TelemetryAccountingError):
        ev.readiness([path])


def test_readiness_rejects_incomplete_producer_bindings(monkeypatch):
    path, native, _, _ = fixture(monkeypatch)
    native["inputs"] = [r for r in native["inputs"]
                        if not r["path"].endswith("strong_validation_telemetry_v1.py")]
    with pytest.raises(ValueError, match="producer source bindings incomplete"):
        ev.readiness([path])


@pytest.mark.parametrize("change", ["incomplete", "cpu", "weights", "unbound",
                                    "mismatch", "input", "duplicate"])
def test_readiness_rejects_invalid_linkage(monkeypatch, change):
    path, native, component, _ = fixture(monkeypatch)
    bindings = [path]
    if change == "incomplete":
        native["metrics"]["training_complete"] = False
    elif change == "cpu":
        native["parameters"]["device"] = "cpu"
    elif change == "weights":
        native["inputs"] = [r for r in native["inputs"]
                            if "adaface_ir101.pt" not in r["path"]]
    elif change == "unbound":
        native["outputs"].pop()
    elif change == "mismatch":
        component["parameters"]["seed"] = 1
    elif change == "input":
        component["inputs"] = [rec("unbound-extra")]
    else:
        bindings.append(path)
    with pytest.raises(ValueError):
        ev.readiness(bindings)


def test_main_no_execute_prints_and_skips_cuda(monkeypatch, capsys):
    path, _, _, _ = fixture(monkeypatch)
    monkeypatch.setattr(ev.torch.cuda, "is_available",
                        lambda: pytest.fail("CUDA must not be probed without --execute"))
    monkeypatch.setattr(sys, "argv", ["ev2", "--bindings", str(path),
                                      "--out", "only/no-execute-out"])
    ev.main()
    assert "no inference" in capsys.readouterr().out


def test_main_execute_requires_cuda_without_fallback(monkeypatch, tmp_path):
    path, _, _, _ = fixture(monkeypatch)
    monkeypatch.setattr(ev.torch.cuda, "is_available", lambda: False)
    out = tmp_path / "fresh-out"
    monkeypatch.setattr(sys, "argv", ["ev2", "--bindings", str(path),
                                      "--out", str(out), "--execute"])
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        ev.main()
    assert not out.exists()


def test_main_binds_all_sources_before_reading_cache():
    tree = ast.parse(DRAFT.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    bind = next(n for n in ast.walk(main) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "inputs" for t in n.targets)
                and isinstance(n.value, ast.Call)
                and getattr(n.value.func, "id", None) == "evaluation_inputs")
    before = next(n for n in ast.walk(main) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "before" for t in n.targets))
    cache_read = next(n for n in ast.walk(main) if isinstance(n, ast.With))
    assert bind.lineno < before.lineno < cache_read.lineno


def test_main_preserves_cuda_guard_literal():
    source = DRAFT.read_text(encoding="utf-8")
    assert 'RuntimeError("CUDA required; no CPU fallback")' in source
    assert "torch.cuda.is_available()" in source


def test_evaluation_inputs_bind_helper_ancestry_and_src_tree():
    native = {"inputs": [rec("a")], "outputs": [rec("b")]}
    inputs = ev.evaluation_inputs(
        DRAFT, ev.PROJECT_ROOT / ev.CROP_CACHE_REL, [native], [Path("cell/summary.manifest.json")],
        ev.PROJECT_ROOT, ancestry=lambda script, root: [root / "scripts/synthetic_ancestor.py"])
    names = {p.name for p in inputs}
    assert {"strong_telemetry_accounting_v1.py", "strong_validation_telemetry_v1.py",
            "run_strong_validation_telemetry_cuda_v1.py", "synthetic_ancestor.py"} <= names
    src_prefix = str(ev.PROJECT_ROOT / "src/age_gap")
    assert any(str(p).startswith(src_prefix) for p in inputs)
    assert str(DRAFT) in {str(p) for p in inputs}


def test_distinct_literals_and_no_live_legacy_experiment():
    assert ev.TELEMETRY_NATIVE_EXPERIMENT == V.TELEMETRY_TRAIN_EXPERIMENT
    assert ev.TELEMETRY_NATIVE_EXPERIMENT != LEGACY_NATIVE
    assert ev.TELEMETRY_EXTERNAL_EXPERIMENT != LEGACY_EXTERNAL
    tree = ast.parse(DRAFT.read_text(encoding="utf-8"))
    values = {n.value for n in ast.walk(tree)
              if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert LEGACY_NATIVE not in values
    assert LEGACY_EXTERNAL not in values


def test_accounting_report_keys_are_not_invented(monkeypatch):
    path, native, _, _ = fixture(monkeypatch)
    report = V.validate_telemetry_record(native, native["metrics"]["validation_pairs"])
    assert set(report) == {
        "experiment", "telemetry_version", "epochs", "validation_pairs",
        "native_bitwise_equivalence_verified", "accounting_checked",
        "neural_replay", "statistical_replay", "source_import_closure_checked",
    }
    assert report["accounting_checked"] is True
    assert report["native_bitwise_equivalence_verified"] is False
    assert report["neural_replay"] is False
    assert report["statistical_replay"] is False


def test_readiness_does_not_mutate_native_fixture(monkeypatch):
    path, native, _, _ = fixture(monkeypatch)
    snapshot = copy.deepcopy(native)
    ev.readiness([path])
    assert native == snapshot


@pytest.mark.parametrize("key,value", [
    ("seed", True), ("seed", 42.0), ("learning_rate", "1e-6"),
    ("epochs_requested", 8.0), ("epochs_executed", 8.0),
    ("selected_epoch", 8.0), ("batch_size", 16.0),
    ("margin", "0.3"), ("gap_weight", False), ("gap_weight", 0),
    ("checkpoint_selection", "validation_best"),
    ("batchnorm_policy", "train"), ("scope", "tail"),
])
def test_readiness_rejects_shadow_or_untyped_fixed_metadata(monkeypatch, key, value):
    path, native, _, _ = fixture(monkeypatch)
    native["parameters"][key] = value
    with pytest.raises(ValueError):
        ev.readiness([path])


@pytest.mark.parametrize("which,value", [
    ("native", True), ("native", 1.0), ("component", True), ("component", 1.0),
])
def test_readiness_requires_typed_manifest_versions(monkeypatch, which, value):
    path, native, component, _ = fixture(monkeypatch)
    (native if which == "native" else component)["schema_version"] = value
    with pytest.raises(ValueError, match="typed"):
        ev.readiness([path])


def test_producer_source_aliases_are_resolved_without_mutation(monkeypatch):
    _, native, _, _ = fixture(monkeypatch)
    record = native["inputs"][0]
    record["path"] = str(Path("scripts/../scripts") / Path(record["path"]).name)
    native["inputs"].append(dict(record))
    original = copy.deepcopy(native)
    assert len(ev.declared_producer_sources(native, ev.PROJECT_ROOT)) == 8
    assert native == original


def test_conflicting_producer_source_aliases_are_refused(monkeypatch):
    _, native, _, _ = fixture(monkeypatch)
    alias = dict(native["inputs"][0])
    alias["path"] = str(Path("scripts/../scripts") / Path(alias["path"]).name)
    alias["sha256"] = "b" * 64
    native["inputs"].append(alias)
    with pytest.raises(ValueError, match="conflicting"):
        ev.declared_producer_sources(native, ev.PROJECT_ROOT)


@pytest.mark.parametrize("field,value", [("path", None), ("bytes", True),
                                       ("sha256", "z" * 64)])
def test_producer_source_record_shape_is_refused(monkeypatch, field, value):
    _, native, _, _ = fixture(monkeypatch)
    native["inputs"][0][field] = value
    with pytest.raises(ValueError, match="invalid byte record"):
        ev.declared_producer_sources(native, ev.PROJECT_ROOT)


def test_current_producer_source_claim_must_match(monkeypatch):
    _, native, _, _ = fixture(monkeypatch)
    native["inputs"][0]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="source bytes differ"):
        ev.declared_producer_sources(native, ev.PROJECT_ROOT)


def test_real_static_script_ancestry_reads_generated_sources_only(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    entry = scripts / "entry.py"
    child = scripts / "child.py"
    leaf = scripts / "leaf.py"
    package = scripts / "__init__.py"
    entry.write_text("from scripts import child\ndef unused():\n import scripts.leaf\n",
                     encoding="utf-8")
    child.write_text("from scripts.leaf import value\n", encoding="utf-8")
    leaf.write_text("raise RuntimeError('must not execute generated imports')\n",
                    encoding="utf-8")
    package.write_text("", encoding="utf-8")
    assert set(ev.local_script_ancestry(entry, tmp_path)) == {entry, child, leaf, package}


def test_cli_failure_does_not_disclose_private_material(monkeypatch, capsys):
    private = "private-sensitive-example/checkpoint.pt"

    def failed_main():
        raise FileNotFoundError(private)

    monkeypatch.setattr(ev, "main", failed_main)
    assert ev.cli() == 1
    captured = capsys.readouterr()
    assert private not in captured.out + captured.err
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "execution_complete": False,
        "reason": "telemetry-evaluator-refusal-no-private-details",
    }


def test_real_component_contract_on_generated_files(monkeypatch, tmp_path):
    # No fake contract or real model: original linkage helper reads generated
    # tiny files, and the telemetry accounting helper validates generated rows.
    root = tmp_path
    for name in V.required_train_producer_bindings():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic producer binding\n")
    pairs = root / "data/processed/pairs.jsonl"
    pairs.parent.mkdir(parents=True)
    pairs.write_text("synthetic pair linkage only\n", encoding="utf-8")
    native_path = root / "cell/summary.manifest.json"
    checkpoint = native_path.parent / "private/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint; never load\n")
    component_path = checkpoint.with_suffix(".manifest.json")
    component_path.write_text("synthetic component bytes\n", encoding="utf-8")
    component = {
        "schema_version": 1, "experiment": ev.COMPONENT_EXPERIMENT,
        "parameters": _component_params() | {"pairs_file": str(pairs)},
        "metrics": {"history": histories()[0]},
        "inputs": [ev.file_record(pairs)],
        "outputs": [ev.file_record(checkpoint)],
    }
    native = {
        "schema_version": 1, "experiment": ev.TELEMETRY_NATIVE_EXPERIMENT,
        "parameters": component["parameters"] | telemetry_parameters(),
        "metrics": telemetry_metrics(),
        "inputs": [ev.file_record(root / name) for name in V.required_train_producer_bindings()]
                  + component["inputs"],
        "outputs": [ev.file_record(checkpoint), ev.file_record(component_path)],
    }
    monkeypatch.setattr(ev, "PROJECT_ROOT", root)

    def verified(path, experiment):
        if path == native_path and experiment == ev.TELEMETRY_NATIVE_EXPERIMENT:
            return native
        if path == component_path and experiment == ev.COMPONENT_EXPERIMENT:
            return component
        pytest.fail("unexpected generated manifest lookup")

    monkeypatch.setattr(ev, "verified", verified)
    assert list(ev.readiness([native_path])[0]) == ["random_head_lr1e-06_s42"]
    component["metrics"]["history"].pop()
    with pytest.raises(ValueError, match="eight ordered"):
        ev.readiness([native_path])
