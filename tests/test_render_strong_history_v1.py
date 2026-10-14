"""Synthetic, CPU-only tests for render_strong_history_v1.

No torch, no weights/images, no CUDA, no network, no live output. Every fixture lives
inside a temporary directory and the mandatory dependency closure is bound to that
root via ``own_module`` (the same synthetic injection the v3 runner documents).

Run from the repository root:
    .venv/Scripts/python.exe -m pytest \
        tests/test_render_strong_history_v1.py -q
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import render_strong_history_v1 as R
from scripts.strong_training_trajectory_v1 import summarize

MANDATORY = R.MANDATORY_DEPENDENCIES
KEYS = ("random|head|lr1e-06|seed42", "random|tail|lr1e-05|seed1")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(root: Path, relative: str, data: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def record(root: Path, relative: str) -> dict:
    data = (root / relative).read_bytes()
    return {"path": relative.replace("\\", "/"), "bytes": len(data), "sha256": sha(data)}


def history(base: float = 0.5) -> list:
    return [
        {
            "epoch": e,
            "training_loss": base / e,
            "validation_auc": 0.5 + e / 100.0,
            "mean_gradient_norm": 0.1 + e / 1000.0,
            "epoch_seconds": 1.0 + e / 100.0,
        }
        for e in range(1, 9)
    ]


def curve(values: list) -> dict:
    return {
        "first": values[0],
        "last": values[-1],
        "change": values[-1] - values[0],
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def cell(key: str, values: dict, descriptive: dict) -> dict:
    negative, scope, lr, seed = key.split("|")
    return {
        "version": R.CELL_VERSION,
        "cell": {
            "negative": negative,
            "trainable_scope": scope,
            "learning_rate": float(lr[2:]),
            "seed": int(seed[4:]),
        },
        "epochs": 8,
        "selected_epoch": 8,
        "checkpoint_selection": "last_epoch",
        "curves": {m: curve(values[m]) for m in R.MEASUREMENTS},
        "validation_loss": None,
        "checkpoint_reselection": False,
        "historical_loss_available": False,
        "publication_clearance": False,
        "seed_population_claim": False,
        "mechanism_claim": False,
        "descriptive": descriptive,
    }


def summary_payload(keys=KEYS, mutate=None) -> dict:
    cells = {}
    for key in keys:
        hist = history()
        values = {m: [row[m] for row in hist] for m in R.MEASUREMENTS}
        if mutate:
            values = mutate(key, values)
        cells[key] = cell(key, values, summarize(hist))
    return {
        "execution_complete": True,
        "publication_ready": False,
        "cell_count": len(cells),
        "cells": cells,
        "validation_loss": None,
        "historical_loss_available": False,
        "checkpoint_reselection": False,
        "seed_population_claim": False,
        "mechanism_claim": False,
        "publication_clearance": False,
        "limitation": "x",
    }


class Fixture:
    """One synthetic project root with the mandatory closure and one v3 aggregate."""

    def __init__(self, root: Path, *, payload=None, keys=KEYS):
        self.root = root
        for index, relative in enumerate((*MANDATORY, "own/renderer.py")):
            write(root, relative, f"# stub {index}\n".encode())
        write(root, "data/native.bin", b"NATIVE-BYTES")
        self.payload = payload if payload is not None else summary_payload(keys)
        write(root, "cell/summary.json", json.dumps(self.payload).encode())
        manifest = {
            "schema_version": 1,
            "experiment": R.NATIVE_EXPERIMENT,
            "parameters": {"epochs": 8},
            "metrics": self.payload,
            "inputs": [record(root, "data/native.bin")],
            "outputs": [record(root, "cell/summary.json")],
        }
        write(root, "cell/summary.manifest.json", json.dumps(manifest).encode())
        self.own = root / "own/renderer.py"

    def render(self, out="out", expected=KEYS, **overrides):
        kwargs = dict(
            root=self.root,
            own_module=self.own,
            figure_writer=lambda payload, path: path.write_bytes(b"%PDF-1.4"),
        )
        kwargs.update(overrides)
        return R.render(
            self.root / "cell/summary.json",
            self.root / "cell/summary.manifest.json",
            list(expected),
            self.root / out,
            **kwargs,
        )


def test_success_writes_fresh_summary_table_figure_and_manifest(tmp_path):
    fixture = Fixture(tmp_path / "root")
    payload = fixture.render()
    out = fixture.root / "out"
    assert [row["key"] for row in payload["rows"]] == sorted(KEYS)
    assert payload["membership"]["declared_cell_count"] == 2
    assert payload["membership"]["matrix_cell_count"] == 36
    assert payload["validation_loss"] is None and payload["seed_mean"] is None
    assert payload["aggregate_seed_mean"] is False and payload["saturation_claim"] is False
    assert "validation loss was not collected" in (out / "history_table.tex").read_text()
    assert "diagnostic AUC peak is not" in (out / "history_table.tex").read_text()
    row = next(r for r in payload["rows"] if r["key"] == KEYS[0])
    assert row["training_loss_first"] == pytest.approx(0.5)
    assert row["training_loss_last"] == pytest.approx(0.0625)
    assert row["epoch_minutes_total"] == pytest.approx(
        sum(1.0 + e / 100.0 for e in range(1, 9)) / 60.0
    )
    manifest = json.loads((out / "history.manifest.json").read_text())
    assert manifest["experiment"] == R.EXPERIMENT
    assert manifest["parameters"]["declared_cell_count"] == 2
    assert manifest["metrics"]["rows"] == payload["rows"]


def test_membership_is_explicit_and_exact(tmp_path):
    fixture = Fixture(tmp_path / "root")
    with pytest.raises(R.RenderError) as missing:
        fixture.render(expected=(KEYS[0],))
    assert missing.value.stage == "membership"
    with pytest.raises(R.RenderError) as duplicate:
        fixture.render(expected=(KEYS[0], KEYS[0]))
    assert duplicate.value.detail == "reason=duplicate-expected-cell"
    with pytest.raises(R.RenderError) as spelling:
        fixture.render(expected=("random|head|lr1e-06|seed99",))
    assert spelling.value.detail == "reason=cell field=seed"
    with pytest.raises(R.RenderError) as empty:
        fixture.render(expected=())
    assert empty.value.detail == "reason=empty-or-wrong-type"


def test_schema_and_missing_epochs_are_rejected(tmp_path):
    short = summary_payload()
    key = KEYS[0]
    short["cells"][key]["curves"]["training_loss"]["values"] = [0.5] * 7
    fixture = Fixture(tmp_path / "root", payload=short)
    with pytest.raises(R.RenderError) as error:
        fixture.render()
    assert error.value.stage == "history" and error.value.detail.startswith("reason=history-shape")


def test_bool_and_nonfinite_measurements_are_rejected(tmp_path):
    payload = summary_payload()
    payload["cells"][KEYS[0]]["curves"]["validation_auc"]["values"][3] = True
    fixture = Fixture(tmp_path / "root", payload=payload)
    with pytest.raises(R.RenderError) as error:
        fixture.render()
    assert error.value.stage == "history" and error.value.detail.endswith("expected=num")
    # NaN cannot even be serialized, so the strict parser is what rejects it upstream.
    with pytest.raises(R.ByteIntegrityError):
        R.parse_manifest_strict(b'{"a": NaN}')


def test_missing_mandatory_dependency_fails_source_closure(tmp_path):
    fixture = Fixture(tmp_path / "root")
    (fixture.root / MANDATORY[0]).unlink()
    with pytest.raises(R.RenderError) as error:
        fixture.render()
    assert (
        error.value.stage == "dependencies"
        and error.value.detail == "reason=missing-or-outside-root"
    )


def test_changed_declared_input_is_rejected(tmp_path):
    fixture = Fixture(tmp_path / "root")
    write(fixture.root, "data/native.bin", b"NATIVE-BYTES-TAMPERED")
    with pytest.raises(R.ByteIntegrityError):
        fixture.render()


def test_output_tampered_after_manifest_is_rejected(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path / "root")
    real = R.write_experiment_manifest

    def tamper(target, **kwargs):
        path = real(target, **kwargs)
        (fixture.root / "out/history_table.tex").write_text("TAMPERED")
        return path

    monkeypatch.setattr(R, "write_experiment_manifest", tamper)
    with pytest.raises(R.RenderError) as error:
        fixture.render()
    assert error.value.stage == "publication"
    assert error.value.detail == "reason=published-output-changed-after-write"
    assert not (fixture.root / "out/history.manifest.json").exists()


def test_unexpected_cell_in_summary_is_rejected(tmp_path):
    fixture = Fixture(
        tmp_path / "root", payload=summary_payload(keys=(*KEYS, "random|full|lr1e-05|seed2"))
    )
    with pytest.raises(R.RenderError) as error:
        fixture.render()
    assert error.value.detail == "reason=declared-membership-mismatch"


def test_nonfresh_destination_is_rejected(tmp_path):
    fixture = Fixture(tmp_path / "root")
    (fixture.root / "out").mkdir()
    with pytest.raises(FileExistsError):
        fixture.render()


def test_parsed_manifest_bytes_cannot_be_replaced_before_preflight(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path / "root")
    real = R._dependency_declarations

    def mutate(*args, **kwargs):
        target = fixture.root / "cell/summary.manifest.json"
        target.write_bytes(target.read_bytes() + b"\n")
        return real(*args, **kwargs)

    monkeypatch.setattr(R, "_dependency_declarations", mutate)
    with pytest.raises(R.ByteIntegrityError):
        fixture.render()
    assert not (fixture.root / "out").exists()


def test_native_input_mutation_during_render_invalidates_receipt(tmp_path):
    fixture = Fixture(tmp_path / "root")

    def mutate(payload, path):
        path.write_bytes(b"%PDF-1.4")
        (fixture.root / "data/native.bin").write_bytes(b"MODIFIED")

    with pytest.raises(R.ByteIntegrityError):
        fixture.render(figure_writer=mutate)
    assert not (fixture.root / "out/history.manifest.json").exists()
    assert (
        json.loads((fixture.root / "out/summary.json").read_text())["execution_complete"] is False
    )


def test_nested_bool_parameter_mutation_is_rejected(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path / "root")
    real = R.write_experiment_manifest

    def mutate(target, **kwargs):
        result = real(target, **kwargs)
        data = json.loads(target.read_text())
        data["parameters"]["epochs"] = True
        target.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(R, "write_experiment_manifest", mutate)
    with pytest.raises(R.RenderError, match="parameters-changed"):
        fixture.render()
    assert not (fixture.root / "out/history.manifest.json").exists()


def test_figure_failure_leaves_only_incomplete_summary(tmp_path):
    fixture = Fixture(tmp_path / "root")

    def fail(payload, path):
        raise RuntimeError("private filename must not be printed")

    with pytest.raises(RuntimeError):
        fixture.render(figure_writer=fail)
    assert (
        json.loads((fixture.root / "out/summary.json").read_text())["execution_complete"] is False
    )
    assert not (fixture.root / "out/history.manifest.json").exists()


def test_additional_output_is_not_accepted_as_native_summary(tmp_path):
    fixture = Fixture(tmp_path / "root")
    target = fixture.root / "cell/summary.manifest.json"
    data = json.loads(target.read_text())
    data["outputs"].append(record(fixture.root, "data/native.bin"))
    target.write_text(json.dumps(data))
    with pytest.raises(R.RenderError, match="outputs-shape"):
        fixture.render()


def test_cli_unexpected_error_is_private(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError("PRIVATE-CAPTION-NAME")

    monkeypatch.setattr(R, "render", fail)
    assert (
        R.main(["--summary", "x", "--manifest", "y", "--expected-cell", KEYS[0], "--out", "z"]) == 1
    )
    assert "PRIVATE-CAPTION-NAME" not in capsys.readouterr().out
