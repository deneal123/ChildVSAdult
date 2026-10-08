from copy import deepcopy
from pathlib import Path

import pytest

from scripts import evaluate_strong_cuda_fgnet_v1 as evaluator


def setup(monkeypatch):
    path = Path("cell/summary.manifest.json")
    checkpoint = path.parent / "private/checkpoint.pt"
    component_path = checkpoint.with_suffix(".manifest.json")
    component = dict(
        parameters=dict(trainable_scope="head", learning_rate=1e-6, seed=42),
        inputs=[dict(path="pairs")],
        outputs=[dict(path=str(checkpoint))],
    )
    native = dict(
        parameters=component["parameters"] | dict(device="cuda", negative="random"),
        metrics=dict(training_complete=True),
        inputs=deepcopy(component["inputs"])
        + [dict(path=str(evaluator.PROJECT_ROOT / "models/adaface_ir101.pt"))],
        outputs=[dict(path=str(checkpoint)), dict(path=str(component_path))],
    )
    monkeypatch.setattr(evaluator, "verified", lambda p, e: native if p == path else component)
    monkeypatch.setattr(evaluator, "file_record", lambda p: dict(path=str(p)))
    calls = []
    monkeypatch.setattr(evaluator, "contract", lambda *args, **kwargs: calls.append((args, kwargs)))
    return path, native, component, calls


def test_single_native_cell_linked(monkeypatch):
    path, _, _, calls = setup(monkeypatch)
    selected, natives = evaluator.readiness([path])
    assert list(selected) == ["random_head_lr1e-06_s42"]
    assert len(natives) == len(calls) == 1
    assert calls[0][1] == dict(scope="head", lr=1e-6, seed=42)


@pytest.mark.parametrize(
    "change", ["incomplete", "cpu", "unbound", "mismatch", "input", "duplicate", "weights"]
)
def test_invalid_native_linkage_rejected(monkeypatch, change):
    path, native, component, _ = setup(monkeypatch)
    bindings = [path]
    if change == "incomplete":
        native["metrics"]["training_complete"] = False
    elif change == "cpu":
        native["parameters"]["device"] = "cpu"
    elif change == "unbound":
        native["outputs"].pop()
    elif change == "mismatch":
        component["parameters"]["seed"] = 1
    elif change == "input":
        native["inputs"] = []
    elif change == "weights":
        native["inputs"][-1]["sha256"] = "different initialization"
    else:
        bindings.append(path)
    with pytest.raises(ValueError):
        evaluator.readiness(bindings)


def test_missing_native_completion_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluator.readiness([tmp_path / "summary.manifest.json"])
