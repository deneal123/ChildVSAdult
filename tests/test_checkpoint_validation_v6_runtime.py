"""Runtime test for the root-authored V6 wrapper ``scripts/measure_checkpoint_validation_cuda_v6.py``.

This file is written to be canonical-ready for ``tests/test_checkpoint_validation_v6_runtime.py``:
it reuses the existing canonical V5 positive fixture
(``tests/test_checkpoint_validation_v5_positive_runtime.py``) through ``importlib.util`` and
only imports repository modules by their canonical ``scripts.`` / ``age_gap.`` names.

What is genuinely exercised:
  * ``v6.run(args)`` on a fresh, real ``isolated_v5()`` module (distinct from canonical V5);
  * the real directory lease, leaf guards, NPZ write, ``summary.json`` publication,
    ``summary.manifest.json`` write and the strict-JSON type-exact readback;
  * the real linear pair-loss arithmetic on tiny distinct frozen/tuned scores;
  * ``verify_component_binding_v6`` accepting exactly the seven permitted native-wrapper
    extras and rejecting every illegal shape.

What is mocked: native cell load, ancestry closure, pretrained-weight path, checkpoint load,
backbone/dataset/inference and the CUDA probes. No GPU call, no native model, no network, no
real data and no CPU neural training happens. This is NOT native CUDA loss evidence and does
not recover any historical epoch loss or prove physical native source closure.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from age_gap.common import io as ag_io
from age_gap.common import manifest as ag_manifest
from age_gap.common.manifest import file_record
from scripts import measure_checkpoint_validation_cuda_v2 as v2
from scripts import measure_checkpoint_validation_cuda_v5 as v5
from scripts import measure_checkpoint_validation_cuda_v6 as v6

_FIXTURE_NAME = "_v5_positive_runtime_fixture"


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("project root with pyproject.toml not found")


def _load_v5_fixture():
    """Reuse the canonical V5 positive fixture as a module (no scratch import path)."""
    path = _repo_root() / "tests" / "test_checkpoint_validation_v5_positive_runtime.py"
    if not path.is_file():
        pytest.skip(f"canonical V5 positive fixture missing: {path}")
    spec = importlib.util.spec_from_file_location(_FIXTURE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_FIXTURE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_FIXTURE_NAME, None)
        raise
    return module


def _rec(path: Path) -> dict:
    return dict(file_record(path))


def _prepared_isolated_run(monkeypatch, tmp_path) -> tuple:
    """Build the synthetic tree, point the fixture at the isolated V5, install the seams."""
    fixture = _load_v5_fixture()
    root = tmp_path / "root"
    # ``file_record`` relativizes declared paths against ``age_gap.common.*.PROJECT_ROOT``.
    # The canonical fixture already redirects ``v5.PROJECT_ROOT`` to the synthetic root; we
    # redirect the two ``age_gap`` bindings the same way so every record path is consistently
    # root-relative (the synthetic root is a real directory on disk). The fixture passes with
    # pytest's default outside-project basetemp; inside-project basetemp needs this mirror.
    monkeypatch.setattr(ag_io, "PROJECT_ROOT", root)
    monkeypatch.setattr(ag_manifest, "PROJECT_ROOT", root)
    fx = fixture._write_synthetic_fixture(root)
    fx["params"].update(negative="random", cudnn_version=91200)
    iso = v6.isolated_v5()
    # The fixture module's own ``v5`` global must be the isolated module BEFORE its seam
    # installer runs, so every patch lands on the isolated copy only.
    monkeypatch.setattr(fixture, "v5", iso)
    fixture._install_fakes(monkeypatch, fx)
    return fixture, fx, iso


def _component_for(fx: dict, *, history: list | None = None) -> dict:
    """A real file-backed finetune component sharing exactly the fixed8 contract."""
    cparams = {key: fx["params"][key] for key in v2.COMPONENT_KEYS}
    cparams["pairs_file"] = fx["params"]["pairs_file"]
    cparams["model_init_state_sha256"] = fx["params"]["model_init_state_sha256"]
    return {
        "outputs": [_rec(fx["checkpoint"])],
        "inputs": [_rec(fx["pairs"])],
        "parameters": cparams,
        "metrics": {"history": copy.deepcopy(fx["history"] if history is None else history)},
    }


def _attach_component(monkeypatch, fx: dict, iso, component: dict) -> Path:
    """Wrap the fixture's native loader so the isolated V6 binding runs inside run()."""
    component_path = fx["root"] / "native/component.manifest.json"
    component_path.write_text(json.dumps(component, indent=2) + "\n", encoding="utf-8")
    inner = iso.load_native_v5

    def loader_with_component(train_dir, root_arg):
        native = inner(train_dir, root_arg)
        native["component"] = component
        native["component_pin"] = iso.pin_json(component_path)
        # Mirror the real load_native_v5 ADA branch: bind the component to the parent.
        iso.verify_component_binding_v5(
            component,
            native["checkpoint"],
            native["parameters"],
            parent_history=native["history"],
        )
        return native

    monkeypatch.setattr(iso, "load_native_v5", loader_with_component)
    return component_path


# --------------------------------------------------------------------------------------
# 1. Full V6 run() through the isolated V5: real guards, math, publication, readback.
# --------------------------------------------------------------------------------------
def test_v6_full_run_publishes_isolated_real_outputs(tmp_path, monkeypatch):
    canonical_v5_exp = v5.EXPERIMENT
    canonical_v5_seeds = tuple(v5.MANDATORY_SEEDS)
    v5_source_before = Path(v5.__file__).read_text(encoding="utf-8")
    v2_source_before = Path(v2.__file__).read_text(encoding="utf-8")

    _, fx, iso = _prepared_isolated_run(monkeypatch, tmp_path)

    # The isolated module is a fresh copy: canonical globals are untouched.
    assert iso is not v5
    assert iso.EXPERIMENT == v6.EXPERIMENT == "fixed-checkpoint-validation-loss-cuda-v6"
    assert "scripts.measure_checkpoint_validation_cuda_v6" in iso.MANDATORY_SEEDS
    assert "scripts.measure_checkpoint_validation_cuda_v6" not in canonical_v5_seeds

    out = fx["root"] / "out"
    monkeypatch.setattr(v6, "isolated_v5", lambda: iso)
    target = v6.run(SimpleNamespace(train=fx["root"] / "native", out=out))

    assert target == out / "summary.manifest.json"
    summary = out / "summary.json"
    npz = out / "private/scores.npz"
    assert target.is_file() and summary.is_file() and npz.is_file()

    # Real NPZ write, distinct frozen/tuned score vectors, private pair ids only.
    with np.load(npz) as data:
        assert list(data.files) == [
            "scores_frozen",
            "scores_tuned",
            "labels",
            "weights",
            "pair_ids",
        ]
        assert data["scores_frozen"].dtype == np.float32
        assert data["pair_ids"].tolist() == ["p1", "p2"]
        assert not np.array_equal(data["scores_frozen"], data["scores_tuned"])

    manifest = json.loads(target.read_text(encoding="utf-8"))
    written_summary = json.loads(summary.read_text(encoding="utf-8"))

    # V6 experiment identity plus JSON-stable, type-exact strict readback.
    assert manifest["experiment"] == v6.EXPERIMENT
    assert written_summary["experiment"] == v6.EXPERIMENT
    assert manifest["metrics"] == written_summary

    # Real linear pair-loss arithmetic on the tiny distinct scores.
    assert written_summary["frozen"]["validation_loss"] == pytest.approx(0.25)
    assert written_summary["tuned"]["validation_loss"] == pytest.approx(0.05)
    assert written_summary["loss_delta_tuned_minus_frozen"] == pytest.approx(-0.20)

    # Honest, JSON-stable scope flags: nothing is claimed beyond a retrospective measurement.
    assert written_summary["status"] == "completed"
    assert written_summary["execution_complete"] is True
    assert written_summary["evaluation_complete"] is False
    assert written_summary["publication_ready"] is False
    scope = written_summary["scope"]
    assert scope["retrospective_measurement"] is True
    assert scope["historical_epoch_loss_recovery"] is False
    assert scope["checkpoint_reselection"] is False
    assert scope["new_external_evaluation"] is False
    assert scope["mechanism_claim"] is False

    # Earliest-claim byte closure on outputs and inputs.
    assert [r["sha256"] for r in manifest["outputs"]] == [
        _rec(summary)["sha256"],
        _rec(npz)["sha256"],
    ]
    assert [r["sha256"] for r in manifest["inputs"]] == [r["sha256"] for r in fx["records_used"]]

    # Canonical V5 globals/constants/source and V2 source remain unchanged.
    assert canonical_v5_exp == v5.EXPERIMENT
    assert tuple(v5.MANDATORY_SEEDS) == canonical_v5_seeds
    assert Path(v5.__file__).read_text(encoding="utf-8") == v5_source_before
    assert Path(v2.__file__).read_text(encoding="utf-8") == v2_source_before


# --------------------------------------------------------------------------------------
# 2. Full V6 run() WITH a component: the seven native-wrapper extras are accepted.
# --------------------------------------------------------------------------------------
def test_v6_full_run_accepts_seven_native_wrapper_extras(tmp_path, monkeypatch):
    _, fx, iso = _prepared_isolated_run(monkeypatch, tmp_path)
    component = _component_for(fx)
    _attach_component(monkeypatch, fx, iso, component)

    # The parent carries exactly the permitted extras beyond the shared contract.
    parent_extras = set(fx["params"]) - set(component["parameters"])
    assert parent_extras == v6.NATIVE_WRAPPER_FIELDS

    monkeypatch.setattr(v6, "isolated_v5", lambda: iso)
    target = v6.run(SimpleNamespace(train=fx["root"] / "native", out=fx["root"] / "out"))
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8"))["experiment"] == v6.EXPERIMENT


# --------------------------------------------------------------------------------------
# 3. verify_component_binding_v6: positive acceptance and every illegal shape rejected.
# --------------------------------------------------------------------------------------
def test_v6_binding_accepts_exact_contract(monkeypatch, tmp_path):
    _, fx, iso = _prepared_isolated_run(monkeypatch, tmp_path)
    component = _component_for(fx)
    # No exception: every shared parameter is type-exact, extras are the permitted set.
    iso.verify_component_binding_v5(
        component, fx["checkpoint"], fx["params"], parent_history=fx["history"]
    )


def test_v6_binding_rejects_unknown_parent_extra(monkeypatch, tmp_path):
    _, fx, iso = _prepared_isolated_run(monkeypatch, tmp_path)
    component = _component_for(fx)
    params = dict(fx["params"])
    params["not_a_permitted_field"] = 1
    with pytest.raises(ValueError, match="undeclared extra native-wrapper parameter"):
        iso.verify_component_binding_v5(
            component, fx["checkpoint"], params, parent_history=fx["history"]
        )


def test_v6_binding_rejects_missing_component_parameter(monkeypatch, tmp_path):
    _, fx, iso = _prepared_isolated_run(monkeypatch, tmp_path)
    component = _component_for(fx)
    component["parameters"]["component_only_marker"] = 1
    with pytest.raises(ValueError, match="native omits a component parameter"):
        iso.verify_component_binding_v5(
            component, fx["checkpoint"], fx["params"], parent_history=fx["history"]
        )


def test_v6_binding_rejects_bool_int_mismatch(monkeypatch, tmp_path):
    _, fx, iso = _prepared_isolated_run(monkeypatch, tmp_path)
    component = _component_for(fx)
    component["parameters"]["batch_size"] = 1
    params = dict(fx["params"])
    params["batch_size"] = True  # 1 == True, so only the type-exact check can catch this
    with pytest.raises(ValueError, match="differs type-exactly"):
        iso.verify_component_binding_v5(
            component, fx["checkpoint"], params, parent_history=fx["history"]
        )


def test_v6_binding_rejects_changed_history(monkeypatch, tmp_path):
    _, fx, iso = _prepared_isolated_run(monkeypatch, tmp_path)
    component = _component_for(fx)
    component["metrics"]["history"][7]["validation_auc"] = 0.5
    with pytest.raises(ValueError, match="history"):
        iso.verify_component_binding_v5(
            component, fx["checkpoint"], fx["params"], parent_history=fx["history"]
        )


def test_v6_binding_rejects_type_changed_history(monkeypatch, tmp_path):
    _, fx, iso = _prepared_isolated_run(monkeypatch, tmp_path)
    component = _component_for(fx)
    # True == 1.0 under value equality, so only V6's type-exact history guard catches it.
    component["metrics"]["history"][0]["validation_auc"] = True
    with pytest.raises(ValueError, match="differs type-exactly"):
        iso.verify_component_binding_v5(
            component, fx["checkpoint"], fx["params"], parent_history=fx["history"]
        )


# --------------------------------------------------------------------------------------
# 4. Source-only isolation / closure declaration (cannot prove physical native closure).
# --------------------------------------------------------------------------------------
def test_v6_source_isolation_and_closure_declaration(tmp_path, monkeypatch):
    canonical_v5_exp = v5.EXPERIMENT
    v5_source_before = Path(v5.__file__).read_text(encoding="utf-8")
    v2_source_before = Path(v2.__file__).read_text(encoding="utf-8")

    iso = v6.isolated_v5()

    # The wrapper declares itself as the running source and pins the immutable V5.
    assert iso.__file__ == v6.__file__
    assert iso.EXPERIMENT == v6.EXPERIMENT
    assert "scripts.measure_checkpoint_validation_cuda_v6" in iso.MANDATORY_SEEDS
    closure = {p.resolve() for p in iso.producing_source_paths()}
    assert Path(v6.__file__).resolve() in closure
    assert Path(v5.__file__).resolve() in closure

    # Isolation up to source identity (physical native closure cannot be proven here).
    assert iso is not v5
    assert iso.verify_component_binding_v5 is v6.verify_component_binding_v6
    assert v5.verify_component_binding_v5 is not v6.verify_component_binding_v6
    assert canonical_v5_exp == v5.EXPERIMENT
    assert Path(v5.__file__).read_text(encoding="utf-8") == v5_source_before
    assert Path(v2.__file__).read_text(encoding="utf-8") == v2_source_before
