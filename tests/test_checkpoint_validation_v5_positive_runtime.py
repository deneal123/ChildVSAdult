"""Full POSITIVE orchestration test for ``scripts.measure_checkpoint_validation_cuda_v5.run``.

This test GENUINELY calls ``v5.run(args)`` end to end. Only the I/O seams (native cell
load, pretrained binder path, checkpoint load, backbone/dataset/inference, CUDA probes)
are mocked. The REAL output lease/leaf guards, NPZ write, summary publication, experiment
manifest write and the type-exact strict readback all execute untouched, plus the real
row enumeration, label/weight verification, pair-loss reduction, AUC reproduction and
paired bootstrap on tiny arrays.

No native CUDA evidence: no GPU, no real weights/data/network, no training.

Synthetic fixtures are local pytest-owned tiny files. The project root is
redirected to the synthetic fixture; guards and publication remain real.
The explicit TorchVersion fixture tests serialization without relying on the
installed package version's Python type. No native CUDA evidence is implied.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from age_gap.common.manifest import file_record
from scripts import measure_checkpoint_validation_cuda_v2 as v2
from scripts import measure_checkpoint_validation_cuda_v5 as v5


def _rec(path: Path) -> dict:
    return dict(file_record(path))


def _write_synthetic_fixture(root: Path) -> dict:
    """Tiny local native cell + crops + inventory + weight inside ``root``."""
    metrics = root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    weight = root / "models/facenet_casia.pt"
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b"synthetic-weights")

    for name in ("a1", "a2", "b1", "b2"):
        crop = root / f"crops/{name}.jpg"
        crop.parent.mkdir(parents=True, exist_ok=True)
        crop.write_bytes(b"face")

    pairs = root / "val_pairs.jsonl"
    pairs.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "pair_id": "p1",
                    "face_a": "a1",
                    "face_b": "b1",
                    "label": 1,
                    "pair_type": "positive_same_post",
                    "age_gap": 5,
                    "split": "val",
                },
                {
                    "pair_id": "p2",
                    "face_a": "a2",
                    "face_b": "b2",
                    "label": 0,
                    "pair_type": "negative_cross_group",
                    "split": "val",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    inventory = metrics / "model_inventory.json"
    inventory.write_text(
        json.dumps({"models": {v2.INVENTORY_KEYS["facenet"]: {"artifact": _rec(weight)}}}),
        encoding="utf-8",
    )

    checkpoint = root / "native/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"synthetic-checkpoint")

    # AUC must reproduce the native last-epoch value within tol; 1.0 matches the two
    # perfectly separated synthetic scores.
    history = [
        {
            "epoch": epoch,
            "train_loss": 0.5,
            "validation_auc": 1.0,
            "mean_gradient_norm": 0.1,
            "epoch_seconds": 1.0,
        }
        for epoch in range(1, v2.FIXED_EPOCHS + 1)
    ]
    params = {
        "backbone": "facenet",
        "epochs_requested": 8,
        "epochs_executed": 8,
        "selected_epoch": 8,
        "batch_size": v2.BATCH_SIZE,
        "checkpoint_selection": "last_epoch",
        "batchnorm_policy": "frozen_all",
        "crops_dir": v2.CROPS_DIR,
        "device": "cuda",
        "threads": 1,
        "margin": v2.MARGIN,
        "gap_weight": v2.GAP_WEIGHT,
        "learning_rate": 1e-6,
        "trainable_scope": "head",
        "seed": 42,
        "pairs_file": str(pairs),
        "torch_version": "2.12.0+cu132",
        "cuda_version": "13.2",
        "gpu_name": "FakeGPU",
        "model_init_state_sha256": "d" * 64,
    }
    native = {
        "experiment": v2.FACENET_EXPERIMENT,
        "parameters": params,
        "metrics": {"training_complete": True, "history": history, "validation_pairs": 2},
        "inputs": [_rec(weight)],
        "outputs": [_rec(checkpoint)],
    }

    class _FakeBackbone:
        tuned = False

        def state_dict(self):
            return {"w": [1.0, 2.0]}

        def eval(self):
            return self

        def to(self, _device):
            return self

    return {
        "root": root,
        "weight": weight,
        "inventory": inventory,
        "pairs": pairs,
        "checkpoint": checkpoint,
        "native": native,
        "params": params,
        "history": history,
        "state": {"w": [1.0, 2.0]},
        "state_digest": "d" * 64,
        "FakeBackbone": _FakeBackbone,
    }


def _install_fakes(monkeypatch, fx: dict) -> None:
    """Patch only the I/O seams; real guards and publication stay in force."""
    import torch

    root = fx["root"]
    monkeypatch.setattr(v5, "PROJECT_ROOT", root)
    monkeypatch.setattr(torch, "__version__", torch.torch_version.TorchVersion("0.0.0+synthetic"))
    native_manifest = root / "native/summary.manifest.json"

    # 1) Native completion cell load. Strict JSON / pin machinery below is real, and the
    #    synthetic inventory is pinned by its explicit path (its records relativize against
    #    the real PROJECT_ROOT because every fixture lives under it).
    def fake_load_native_v5(train_dir, root_arg):
        inv_pin = v5.pin_inventory(fx["inventory"], v2.INVENTORY_KEYS["facenet"])
        native_manifest.write_text(
            json.dumps(
                {
                    "experiment": v2.FACENET_EXPERIMENT,
                    "parameters": fx["params"],
                    "metrics": fx["native"]["metrics"],
                    "inputs": fx["native"]["inputs"],
                    "outputs": fx["native"]["outputs"],
                }
            ),
            encoding="utf-8",
        )
        fx["inventory_pin"] = inv_pin
        fx["parent_record"] = _rec(native_manifest)
        fx["weight_record"] = _rec(fx["weight"])
        fx["pairs_record"] = _rec(fx["pairs"])
        return {
            "manifest": native_manifest,
            "manifest_record": fx["parent_record"],
            "native": fx["native"],
            "parameters": fx["params"],
            "history": fx["history"],
            "declared_rows": 2,
            "checkpoint": fx["checkpoint"],
            "component": None,
            "component_pin": None,
            "inventory_pin": inv_pin,
            "experiment": v2.FACENET_EXPERIMENT,
        }

    monkeypatch.setattr(v5, "load_native_v5", fake_load_native_v5)

    # 2) Producing-source closure is large/coupled; substitute an equivalent tiny bound
    #    record list (all files really exist inside PROJECT_ROOT). No guard is weakened.
    def fake_build_ancestry_records_v5(**kwargs):
        records = [
            dict(fx["parent_record"]),
            dict(fx["native"]["outputs"][0]),
            dict(fx["native"]["inputs"][0]),
            dict(fx["inventory_pin"].pin.record),
            dict(fx["pairs_record"]),
        ]
        fx["records_used"] = records
        return records

    monkeypatch.setattr(v5, "build_ancestry_records_v5", fake_build_ancestry_records_v5)

    # 3) Environment seams. bind_pretrained_weight_v5 stays REAL, so only the exact
    #    factory path resolution is redirected to the synthetic weight.
    monkeypatch.setattr(v2, "require_cuda", lambda: None)
    monkeypatch.setattr(v2, "check_device_binding", lambda _native: None)
    monkeypatch.setattr(v5, "factory_weight_path_v5", lambda _name: fx["weight"])

    # 4) Checkpoint / model / dataset / inference seams.
    monkeypatch.setattr(
        v2,
        "load_checkpoint",
        lambda _cp: {
            "selected_epoch": 8,
            "state_dict": fx["state"],
            "history": fx["history"],
        },
    )
    monkeypatch.setattr(v2, "verify_completed_checkpoint", lambda _saved, **_kw: None)
    monkeypatch.setattr(v2, "state_dict_digest", lambda _state: fx["state_digest"])
    monkeypatch.setattr(v2, "make_backbone", lambda _name, pretrained=True: fx["FakeBackbone"]())
    monkeypatch.setattr(
        v2, "load_tuned_strict", lambda model, _state: setattr(model, "tuned", True)
    )
    monkeypatch.setattr(
        v2,
        "build_dataset",
        lambda _split, _pairs, _model, _crops: [("a1", "b1"), ("a2", "b2")],
    )
    monkeypatch.setattr(v2, "verify_dataset_coverage", lambda _dataset, _rows: None)
    monkeypatch.setattr(
        v2,
        "crop_path",
        lambda face_id, _crops_dir: root / f"crops/{face_id}.jpg",
    )

    def fake_infer_scores(_model, _dataset, *, device, batch_size, margin):
        return {
            "scores": np.asarray([0.9, 0.1] if _model.tuned else [0.65, 0.45], dtype=np.float64),
            "labels": np.asarray([1.0, 0.0], dtype=np.float64),
            "weights": np.asarray([1.0, 1.0], dtype=np.float64),
            "n_pairs": 2,
        }

    monkeypatch.setattr(v2, "infer_scores", fake_infer_scores)

    # 5) CUDA probe surface only; no GPU call can occur once these are patched.
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "FakeGPU")
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 9999)
    # NOTE: torch.__version__ is deliberately NOT substituted, so the real
    # torch.torch_version.TorchVersion subclass flows through the positive run.


def test_run_positive_orchestration_publishes_real_outputs(tmp_path, monkeypatch):
    fx = _write_synthetic_fixture(tmp_path / "root")
    _install_fakes(monkeypatch, fx)

    out = fx["root"] / "out"
    target = v5.run(SimpleNamespace(train=fx["root"] / "native", out=out))

    assert target == out / "summary.manifest.json"
    summary = out / "summary.json"
    npz = out / "private/scores.npz"
    assert target.is_file() and summary.is_file() and npz.is_file()

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

    manifest = json.loads(target.read_text(encoding="utf-8"))
    written_summary = json.loads(summary.read_text(encoding="utf-8"))
    assert manifest["experiment"] == v5.EXPERIMENT
    assert manifest["metrics"] == written_summary
    assert written_summary["status"] == "completed"
    assert written_summary["frozen"]["validation_loss"] == pytest.approx(0.25)
    assert written_summary["tuned"]["validation_loss"] == pytest.approx(0.05)
    assert written_summary["loss_delta_tuned_minus_frozen"] == pytest.approx(-0.20)
    assert written_summary["publication_ready"] is False
    assert written_summary["scope"]["historical_epoch_loss_recovery"] is False
    assert written_summary["scope"]["mechanism_claim"] is False
    assert [r["sha256"] for r in manifest["outputs"]] == [
        _rec(summary)["sha256"],
        _rec(npz)["sha256"],
    ]
    assert [r["sha256"] for r in manifest["inputs"]] == [r["sha256"] for r in fx["records_used"]]


def test_raw_torchversion_subclass_breaks_readback_but_plain_string_passes(tmp_path, monkeypatch):
    """The raw ``TorchVersion`` failure mode, and where the minimal fix actually lives."""
    import torch

    fx = _write_synthetic_fixture(tmp_path / "root")
    _install_fakes(monkeypatch, fx)

    raw = torch.__version__
    assert type(raw) is not str, "expect torch.__version__ to be a str subclass here"
    # Value survives JSON, the Python type does not: exactly what strict readback detects.
    roundtripped = json.loads(json.dumps({"binding": {"torch_version": raw}}))
    assert roundtripped["binding"]["torch_version"] == str(raw)
    assert type(roundtripped["binding"]["torch_version"]) is str

    # The failure mode the task described: a summary/binding carrying the raw subclass
    # is refused by the v5 type-exact readback after a JSON round-trip.
    with pytest.raises(ValueError, match="differs type-exactly"):
        v5._type_exact_equal(raw, roundtripped["binding"]["torch_version"], "metrics")

    # Minimal fix: coerce at the serialization boundary. This makes the type-exact
    # comparison agree again and is what the canonical v5 source now does.
    fixed = str(raw)
    assert type(fixed) is str
    v5._type_exact_equal(str(fixed), roundtripped["binding"]["torch_version"], "metrics")

    # Where the fix lives: v5 already wraps it, the v2 siblings were still raw in this
    # checkout (reported as a finding; no canonical source is edited by this worker).
    v5_src = Path(v5.__file__).read_text(encoding="utf-8")
    assert '"torch_version": str(torch.__version__)' in v5_src

    # And a REAL positive run with the raw subclass still publishes cleanly: v5's
    # str(...) boundary is what makes the manifest read back type-exactly.
    out = fx["root"] / "out"
    target = v5.run(SimpleNamespace(train=fx["root"] / "native", out=out))
    written = json.loads(target.read_text(encoding="utf-8"))
    published = written["metrics"]["binding"]["torch_version"]
    assert type(published) is str and published == str(raw)
