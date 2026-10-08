"""Synthetic CPU guards for the retrospective fixed-checkpoint measurement producer.

These tests never load real weights, real face data or a GPU. Where a safety boundary can
be exercised with real upstream code (``file_record``/``verify_records``/``measure``/
``ImagePairDataset``/``state_dict_digest``), the real function is used against tiny local
files instead of being mocked. Only genuinely external effects (CUDA device, backbone
construction, score inference) are monkeypatched. Passing these is NOT native CUDA
evidence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

PRODUCER = Path(__file__).resolve().parents[1] / "scripts" / "measure_checkpoint_validation_cuda_v2.py"
_spec = importlib.util.spec_from_file_location(
    "measure_checkpoint_validation_cuda_v2", PRODUCER
)
producer = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = producer
_spec.loader.exec_module(producer)

EPOCH_KEYS = ("training_loss", "validation_auc", "mean_gradient_norm", "epoch_seconds")


# --------------------------------------------------------------------------------------
# Import / dry-run: no CUDA probe, no model load, no data preflight, no completion
# --------------------------------------------------------------------------------------
def test_import_does_not_probe_gpu(monkeypatch):
    calls = []

    def boom():
        calls.append(True)
        raise AssertionError("import-time CUDA probe")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    fresh_spec = importlib.util.spec_from_file_location("mcvc_reload", PRODUCER)
    fresh = importlib.util.module_from_spec(fresh_spec)
    fresh_spec.loader.exec_module(fresh)
    assert calls == []


def test_dry_run_touches_nothing(tmp_path, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise AssertionError("dry-run must not probe CUDA or load models/data")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    monkeypatch.setattr(producer, "load_native", boom)
    monkeypatch.setattr(producer, "make_backbone", boom)
    monkeypatch.setattr(producer, "declared_val_rows", boom)
    out = tmp_path / "out"
    producer.main(["--train", str(tmp_path / "native"), "--out", str(out)])
    assert "dry-run" in capsys.readouterr().out
    assert not out.exists()


# --------------------------------------------------------------------------------------
# Native fixed8 contract
# --------------------------------------------------------------------------------------
def _params(**over):
    base = dict(
        backbone="facenet",
        epochs_requested=8,
        epochs_executed=8,
        selected_epoch=8,
        checkpoint_selection="last_epoch",
        batchnorm_policy="frozen_all",
        trainable_scope="head",
        learning_rate=1e-6,
        batch_size=16,
        margin=0.3,
        gap_weight=0.0,
        crops_dir="faces",
        seed=42,
        device="cuda",
        threads=1,
        pairs_file="data/processed/pairs.jsonl",
        torch_version="2.12.0+cu132",
        cuda_version="13.2",
        gpu_name="Fake GPU",
    )
    base.update(over)
    return base


def _history(epochs=8, **over):
    rows = []
    for e in range(1, epochs + 1):
        row = {"epoch": e, **{k: 0.5 for k in EPOCH_KEYS}}
        row.update(over)
        rows.append(row)
    return rows


def _native(experiment=producer.FACENET_EXPERIMENT, **over):
    backbone = "adaface_ir101" if experiment == producer.ADA_EXPERIMENT else "facenet"
    params = _params(backbone=backbone)
    params.update(over)
    history = _history()
    if backbone == "adaface_ir101":
        for row in history:
            row["training_loss"] = row.pop("training_loss", 0.5) if "training_loss" in row else 0.5
    payload = {
        "experiment": experiment,
        "parameters": params,
        "metrics": {"training_complete": True, "history": history},
    }
    return payload


def test_classify_native_rejects_other_experiments():
    assert producer.classify_native(_native()) == producer.FACENET_EXPERIMENT
    assert producer.classify_native(_native(producer.ADA_EXPERIMENT)) == producer.ADA_EXPERIMENT
    with pytest.raises(ValueError, match="unsupported native experiment"):
        producer.classify_native({"experiment": "pair-contrastive-backbone-finetune"})


@pytest.mark.parametrize(
    "experiment,mutate",
    [
        (producer.FACENET_EXPERIMENT, {"epochs_requested": 4}),
        (producer.FACENET_EXPERIMENT, {"epochs_executed": 7}),
        (producer.FACENET_EXPERIMENT, {"selected_epoch": 3}),
        (producer.FACENET_EXPERIMENT, {"checkpoint_selection": "best_val"}),
        (producer.FACENET_EXPERIMENT, {"batchnorm_policy": "adapt_all"}),
        (producer.FACENET_EXPERIMENT, {"batch_size": 32}),
        (producer.FACENET_EXPERIMENT, {"margin": 0.5}),
        (producer.FACENET_EXPERIMENT, {"gap_weight": 1.0}),
        (producer.FACENET_EXPERIMENT, {"backbone": "arcface_r50_casia"}),
        (producer.FACENET_EXPERIMENT, {"device": "cpu"}),
        (producer.FACENET_EXPERIMENT, {"crops_dir": "faces_mtcnn"}),
        (producer.FACENET_EXPERIMENT, {"learning_rate": 1e-3}),
        (producer.FACENET_EXPERIMENT, {"trainable_scope": "middle"}),
        (producer.FACENET_EXPERIMENT, {"seed": 7}),
        (producer.ADA_EXPERIMENT, {"backbone": "facenet"}),
    ],
)
def test_native_contract_fails_closed_on_any_drift(experiment, mutate):
    with pytest.raises(ValueError, match="contract differs|outside|must carry|device"):
        producer.native_contract(_native(experiment, **mutate))


@pytest.mark.parametrize(
    "mutate",
    [
        {"epochs_requested": "8"},
        {"epochs_executed": 8.0},
        {"selected_epoch": None},
        {"batch_size": True},
        {"margin": "0.3"},
        {"gap_weight": None},
        {"learning_rate": "1e-6"},
        {"trainable_scope": 1},
        {"seed": "42"},
        {"device": 0},
        {"crops_dir": None},
        {"threads": "1"},
        {"torch_version": 2.12},
    ],
)
def test_native_contract_true_typeguards(mutate):
    native = _native()
    native["parameters"].update(mutate)
    with pytest.raises(ValueError, match="wrong type|missing"):
        producer.native_contract(native)


def test_native_contract_requires_eight_finite_epochs():
    native = _native()
    native["metrics"]["history"] = _history(3)
    with pytest.raises(ValueError, match="eight ordered"):
        producer.native_contract(native)
    native = _native()
    native["metrics"]["history"][5]["validation_auc"] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        producer.native_contract(native)
    native = _native()
    native["metrics"]["training_complete"] = False
    with pytest.raises(ValueError, match="training_complete"):
        producer.native_contract(native)


def test_native_contract_accepts_strong_cell_history_key():
    # The strong cell writes "training_loss"; the facenet v2 cell writes "train_loss".
    native = _native(producer.ADA_EXPERIMENT)
    rows = producer.native_contract(native)[1]
    assert all("training_loss" in row for row in rows)
    native = _native()
    for row in native["metrics"]["history"]:
        row["train_loss"] = row.pop("training_loss")
    assert producer.native_contract(native)[1][0]["train_loss"] == 0.5


def test_native_contract_negative_is_a_string_choice():
    # Regression: 'negative' is a real native *string* (random|lookalike) and must not be
    # routed through the numeric choice guard.
    native = _native()
    native["parameters"]["negative"] = "lookalike"
    producer.native_contract(native)
    native["parameters"]["negative"] = "mined"
    with pytest.raises(ValueError, match="outside"):
        producer.native_contract(native)
    native["parameters"]["negative"] = 1
    with pytest.raises(ValueError, match="wrong type"):
        producer.native_contract(native)


@pytest.mark.parametrize("seed", [42.0, 1.0, 2.0, True, False])
def test_native_contract_seed_must_be_real_int(seed):
    # 42.0 == 42 numerically but is NOT a genuine integer seed; bools are not ints either.
    native = _native()
    native["parameters"]["seed"] = seed
    with pytest.raises(ValueError, match="wrong type"):
        producer.native_contract(native)


def test_native_contract_seed_valid_ints_pass():
    for seed in (42, 1, 2):
        native = _native()
        native["parameters"]["seed"] = seed
        assert producer.native_contract(native)[0]["seed"] == seed


@pytest.mark.parametrize(
    "declared,match",
    [
        ("9158", "must be a real int"),
        (9158.0, "must be a real int"),
        (9158.4, "must be a real int"),
        (True, "must be a real int"),
        (-1, "must be positive"),
        (0, "must be positive"),
    ],
)
def test_native_declared_row_count_guards_before_coercion(declared, match):
    native = {"metrics": {"validation_pairs": declared}}
    with pytest.raises(ValueError, match=match):
        producer.native_declared_row_count(native)


def test_native_declared_row_count_absent_is_none_and_valid_int_passes():
    assert producer.native_declared_row_count({"metrics": {}}) is None
    assert producer.native_declared_row_count({}) is None
    assert producer.native_declared_row_count({"metrics": {"validation_pairs": 9158}}) == 9158


# --------------------------------------------------------------------------------------
# Real tiny load_native (re-hash of declared records, single checkpoint, component check)
# --------------------------------------------------------------------------------------
def _write_native(tmp_path, *, experiment=producer.FACENET_EXPERIMENT, epoch_key=None):
    if epoch_key is None:
        epoch_key = (
            "training_loss" if experiment == producer.ADA_EXPERIMENT else "train_loss"
        )
    run_dir = tmp_path / "native"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "private/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    history = [
        {"epoch": e, epoch_key: 0.5, "validation_auc": 0.7,
         "mean_gradient_norm": 0.1, "epoch_seconds": 1.0}
        for e in range(1, 9)
    ]
    torch.save(
        {"state_dict": {"net.weight": torch.zeros(1)}, "history": history,
         "selected_epoch": 8, "checkpoint_selection": "last_epoch"},
        checkpoint,
    )
    payload = _native(experiment)
    payload["metrics"]["history"] = history
    payload["metrics"]["validation_pairs"] = 4
    from age_gap.common.manifest import write_experiment_manifest

    manifest = run_dir / "summary.manifest.json"
    write_experiment_manifest(
        manifest, experiment=experiment, parameters=payload["parameters"],
        metrics=payload["metrics"], inputs=[checkpoint], outputs=[checkpoint],
    )
    return run_dir, checkpoint


def test_load_native_rehashes_records_and_finds_one_checkpoint(tmp_path):
    run_dir, checkpoint = _write_native(tmp_path)
    loaded = producer.load_native(run_dir)
    assert loaded["checkpoint"] == checkpoint.resolve()
    assert loaded["experiment"] == producer.FACENET_EXPERIMENT
    assert loaded["component"] is None
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="size/checksum changed|unavailable"):
        producer.load_native(run_dir)


def test_load_native_requires_component_to_be_parent_output(tmp_path):
    run_dir, checkpoint = _write_native(tmp_path, experiment=producer.ADA_EXPERIMENT)
    from age_gap.common.manifest import file_record, write_experiment_manifest

    payload = _native(producer.ADA_EXPERIMENT)
    params = payload["parameters"]
    parent_history = json.loads(
        (run_dir / "summary.manifest.json").read_text(encoding="utf-8"))["metrics"]["history"]
    project_pairs = Path(str(params["pairs_file"]))
    if not project_pairs.is_absolute():
        project_pairs = Path(producer.PROJECT_ROOT) / project_pairs
    component = checkpoint.with_suffix(".manifest.json")
    write_experiment_manifest(
        component, experiment=producer.COMPONENT_EXPERIMENT,
        parameters=params,
        metrics={"history": parent_history}, inputs=[project_pairs], outputs=[checkpoint],
    )
    # The parent cell does not declare the component as an output -> refuse.
    with pytest.raises(ValueError, match="does not declare the finetune component"):
        producer.load_native(run_dir)
    # Declaring it makes the pair load, and the real upstream record check still applies.
    payload = json.loads((run_dir / "summary.manifest.json").read_text(encoding="utf-8"))
    payload["outputs"] = payload["outputs"] + [file_record(component)]
    (run_dir / "summary.manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    loaded = producer.load_native(run_dir)
    assert loaded["component"] is not None


def test_find_checkpoint_requires_exactly_one(tmp_path):
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    native = {"outputs": [{"path": str(a)}, {"path": str(b)}]}
    with pytest.raises(ValueError, match="exactly one"):
        producer.find_checkpoint(native)
    with pytest.raises(FileNotFoundError):
        producer.find_checkpoint({"outputs": [{"path": str(tmp_path / "missing.pt")}]})


# --------------------------------------------------------------------------------------
# Pretrained weight binding (no implicit download, no SHA-only membership)
# --------------------------------------------------------------------------------------
def _weight_env(tmp_path, monkeypatch, *, pairs_file=None):
    from age_gap.common.manifest import file_record

    if pairs_file is None:
        pairs_file, _ = _pairs_file(tmp_path)
    weight = tmp_path / "casia.pt"
    weight.write_bytes(b"fake-casia-weights")
    inventory = tmp_path / "metrics/model_inventory.json"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(json.dumps({"models": {"facenet_casia": {
        "artifact": file_record(weight)}}}), encoding="utf-8")
    monkeypatch.setattr(producer, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(producer, "factory_weight_path", lambda name: weight)
    record = file_record(weight)
    native = {
        "parameters": {"pretrained_weight_sha256": record["sha256"],
                       "pairs_file": str(pairs_file)},
        "native": {"inputs": [record]},
    }
    return weight, inventory, record, native


def test_bind_pretrained_weight_matches_factory_inventory_and_native(tmp_path, monkeypatch):
    _weight, _inv, record, native = _weight_env(tmp_path, monkeypatch)
    bound = producer.bind_pretrained_weight("facenet", native)
    assert bound["record"] == record
    assert bound["path"] == str(_weight)


def test_bind_pretrained_weight_rejects_sha_only_membership(tmp_path, monkeypatch):
    weight, inventory, record, native = _weight_env(tmp_path, monkeypatch)
    # Native declares only the sha256: SHA-only membership must no longer pass.
    native["native"]["inputs"] = [{"sha256": record["sha256"]}]
    with pytest.raises(RuntimeError, match="not a declared native input"):
        producer.bind_pretrained_weight("facenet", native)
    # A record with the right sha but a wrong path/bytes is also refused.
    native["native"]["inputs"] = [record | {"bytes": record["bytes"] + 1}]
    with pytest.raises(RuntimeError, match="not a declared native input"):
        producer.bind_pretrained_weight("facenet", native)
    native["native"]["inputs"] = [record | {"path": "other/casia.pt"}]
    with pytest.raises(RuntimeError, match="not a declared native input"):
        producer.bind_pretrained_weight("facenet", native)
    assert inventory.is_file()


def test_bind_pretrained_weight_fails_closed(tmp_path, monkeypatch):
    weight, inventory, record, native = _weight_env(tmp_path, monkeypatch)

    inventory.write_text(json.dumps({"models": {"facenet_casia": {
        "artifact": record | {"sha256": "0" * 64}}}}))
    with pytest.raises(RuntimeError, match="sha differ"):
        producer.bind_pretrained_weight("facenet", native)

    inventory.write_text(json.dumps({"models": {"facenet_casia": {"artifact": record}}}))
    with pytest.raises(RuntimeError, match="not a declared native input"):
        producer.bind_pretrained_weight("facenet", {"parameters": {}, "native": {"inputs": []}})

    inventory.write_text(json.dumps({"models": {}}))
    with pytest.raises(RuntimeError, match="lacks facenet_casia"):
        producer.bind_pretrained_weight("facenet", native)

    inventory.write_text(json.dumps({"models": {"facenet_casia": {
        "artifact": {"sha256": record["sha256"]}}}}))
    with pytest.raises(RuntimeError, match="incomplete"):
        producer.bind_pretrained_weight("facenet", native)

    inventory.write_text(json.dumps({"models": {"facenet_casia": {
        "artifact": record | {"path": str(tmp_path / "elsewhere.pt")}}}}))
    with pytest.raises(RuntimeError, match="not the exact factory path"):
        producer.bind_pretrained_weight("facenet", native)

    inventory.write_text(json.dumps({"models": {"facenet_casia": {"artifact": record}}}))
    weight.unlink()
    with pytest.raises(FileNotFoundError, match="refusing implicit download"):
        producer.bind_pretrained_weight("facenet", native)


def test_factory_weight_path_is_not_substitutable(tmp_path, monkeypatch):
    monkeypatch.setattr(producer, "PROJECT_ROOT", tmp_path)
    assert producer.factory_weight_path("adaface_ir101") == (tmp_path / "models/adaface_ir101.pt")
    sentinel = tmp_path / "cache/casia.pt"

    def fake_resolve(torch_home=None):
        return sentinel

    monkeypatch.setattr(producer, "factory_weight_path", producer.factory_weight_path)
    monkeypatch.setitem(
        sys.modules,
        "scripts.run_facenet_validation_cuda_v2",
        SimpleNamespace(resolve_facenet_casia_weight=fake_resolve),
    )
    assert producer.factory_weight_path("facenet") == sentinel
    with pytest.raises(ValueError, match="no known pretrained factory path"):
        producer.factory_weight_path("arcface_r50_casia")


def test_verify_component_binding_requires_same_checkpoint_budget_and_history(tmp_path):
    from age_gap.common.manifest import file_record, write_experiment_manifest

    pairs_file, _rows = _pairs_file(tmp_path)
    checkpoint = _fake_checkpoint(tmp_path / "checkpoint.pt")
    params = _params(pairs_file=str(pairs_file))
    history = _history()
    component_path = tmp_path / "checkpoint.manifest.json"
    write_experiment_manifest(
        component_path, experiment=producer.COMPONENT_EXPERIMENT, parameters=params,
        metrics={"history": history}, inputs=[pairs_file], outputs=[checkpoint],
    )
    component = json.loads(component_path.read_text(encoding="utf-8"))
    producer.verify_component_binding(component, checkpoint, params, parent_history=history)

    other = _fake_checkpoint(tmp_path / "other.pt")
    with pytest.raises(ValueError, match="does not bind"):
        producer.verify_component_binding(component, other, params, parent_history=history)
    bad = json.loads(json.dumps(component))
    bad["parameters"] = params | {"learning_rate": 1e-5}
    with pytest.raises(ValueError, match="learning_rate"):
        producer.verify_component_binding(bad, checkpoint, params, parent_history=history)
    bad = json.loads(json.dumps(component))
    bad["parameters"] = params | {"pairs_file": "data/processed/other.jsonl"}
    with pytest.raises(ValueError, match="pairs_file"):
        producer.verify_component_binding(bad, checkpoint, params, parent_history=history)
    bad = json.loads(json.dumps(component))
    bad["metrics"]["history"] = _history()
    bad["metrics"]["history"][0]["validation_auc"] = 0.9
    with pytest.raises(ValueError, match="history differs"):
        producer.verify_component_binding(bad, checkpoint, params, parent_history=history)
    bad = json.loads(json.dumps(component))
    bad["metrics"]["history"] = _history(3)
    with pytest.raises(ValueError, match="eight ordered"):
        producer.verify_component_binding(bad, checkpoint, params, parent_history=history)
    assert file_record(checkpoint)["sha256"]


# --------------------------------------------------------------------------------------
# Validation rows: no silent dropping, exact ordered coverage
# --------------------------------------------------------------------------------------
def _pairs_file(tmp_path, n=4):
    from age_gap.common.schemas import Pair

    rows = [
        Pair(pair_id=f"p{i}", face_a=f"a{i}", face_b=f"b{i}", label=i % 2,
             pair_type="positive_same_post", age_gap=10, split="val" if i != 2 else "train")
        for i in range(n)
    ]
    pairs_file = tmp_path / "pairs.jsonl"
    pairs_file.write_text("\n".join(json.dumps(p.to_dict()) for p in rows), encoding="utf-8")
    return pairs_file, rows


def _real_crops(tmp_path, monkeypatch, rows, *, drop=None):
    def fake_crop(face_id, crops_dir):
        return tmp_path / f"{face_id}.jpg"

    monkeypatch.setattr(producer, "crop_path", fake_crop)
    for p in rows:
        if p.face_a == drop:
            continue
        (tmp_path / f"{p.face_a}.jpg").write_bytes(b"i")
        (tmp_path / f"{p.face_b}.jpg").write_bytes(b"i")


def test_declared_val_rows_counts_all_and_refuses_missing(tmp_path, monkeypatch):
    pairs_file, rows = _pairs_file(tmp_path)
    _real_crops(tmp_path, monkeypatch, rows)
    out = producer.declared_val_rows(pairs_file, "faces")
    assert [p.pair_id for p, _a, _b in out] == ["p0", "p1", "p3"]

    (tmp_path / f"{rows[3].face_a}.jpg").unlink()
    with pytest.raises(RuntimeError, match="missing; no row dropping"):
        producer.declared_val_rows(pairs_file, "faces")


def test_verify_dataset_coverage_uses_real_imagepairdataset(tmp_path, monkeypatch):
    import age_gap.training.finetune as finetune
    from age_gap.training.finetune import ImagePairDataset

    pairs_file, rows = _pairs_file(tmp_path)
    _real_crops(tmp_path, monkeypatch, rows)

    # Route the REAL dataset's crop resolution onto our tiny files (real code path).
    monkeypatch.setattr(finetune, "_crop_path", producer.crop_path)
    declared = producer.declared_val_rows(pairs_file, "faces")
    dataset = ImagePairDataset(
        "val", pairs_file=str(pairs_file), preprocess=lambda img: img.transpose(2, 0, 1),
        crops_dir="faces",
    )
    assert len(dataset) == len(declared) == 3
    producer.verify_dataset_coverage(dataset, declared)  # sanity

    # Same count, reordered rows -> must be refused.
    dataset._items = list(reversed(dataset._items))
    with pytest.raises(RuntimeError, match="crop paths differ"):
        producer.verify_dataset_coverage(dataset, declared)

    # A duplicated row of the same count cannot pass either.
    dataset._items = [dataset._items[0], dataset._items[0], dataset._items[2]]
    with pytest.raises(RuntimeError, match="crop paths differ"):
        producer.verify_dataset_coverage(dataset, declared)

    # A genuinely dropped row must be caught by the count check.
    dataset._items = dataset._items[:2]
    with pytest.raises(RuntimeError, match="dropped rows"):
        producer.verify_dataset_coverage(dataset, declared)


def test_verify_arm_vectors_compares_against_declared_rows(tmp_path, monkeypatch):
    pairs_file, rows = _pairs_file(tmp_path)
    _real_crops(tmp_path, monkeypatch, rows)
    declared = producer.declared_val_rows(pairs_file, "faces")
    labels, weights = producer.declared_labels_weights(declared)
    good = {"labels": labels.copy(), "weights": weights.copy()}
    producer.verify_arm_vectors(good, declared, "frozen")  # sanity

    swapped = {"labels": labels[::-1].copy(), "weights": weights.copy()}
    with pytest.raises(RuntimeError, match="labels do not match"):
        producer.verify_arm_vectors(swapped, declared, "frozen")
    wrong_w = {"labels": labels.copy(), "weights": weights + 1.0}
    with pytest.raises(RuntimeError, match="weights do not match"):
        producer.verify_arm_vectors(wrong_w, declared, "tuned")


# --------------------------------------------------------------------------------------
# Strict checkpoint loading
# --------------------------------------------------------------------------------------
def _fake_checkpoint(path, epochs=8, bad=False, state=None, selection="last_epoch",
                     validation_auc=0.5):
    history = [{"epoch": e, "train_loss": 0.5, "validation_auc": validation_auc,
                "mean_gradient_norm": 0.1, "epoch_seconds": 1.0}
               for e in range(1, epochs + 1)]
    if bad:
        history[3]["validation_auc"] = float("inf")
    state = {"net.weight": torch.ones(2, 2)} if state is None else state
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "history": history, "selected_epoch": epochs,
                "checkpoint_selection": selection}, path)
    return path


def _checkpoint_history(path):
    return torch.load(path, map_location="cpu", weights_only=True)["history"]


def test_verify_completed_checkpoint_fail_closed(tmp_path):
    good = _fake_checkpoint(tmp_path / "good.pt")
    producer.verify_completed_checkpoint(
        torch.load(good, map_location="cpu", weights_only=True),
        parent_history=_checkpoint_history(good),
    )
    bad = _fake_checkpoint(tmp_path / "bad.pt", bad=True)
    with pytest.raises(RuntimeError, match="not finite"):
        producer.verify_completed_checkpoint(
            torch.load(bad, map_location="cpu", weights_only=True),
            parent_history=_checkpoint_history(bad),
        )
    short = _fake_checkpoint(tmp_path / "short.pt", epochs=3)
    with pytest.raises(RuntimeError, match="selected_epoch"):
        producer.verify_completed_checkpoint(
            torch.load(short, map_location="cpu", weights_only=True),
            parent_history=_checkpoint_history(short),
        )
    sel = _fake_checkpoint(tmp_path / "sel.pt", selection="best_val")
    with pytest.raises(RuntimeError, match="last_epoch"):
        producer.verify_completed_checkpoint(
            torch.load(sel, map_location="cpu", weights_only=True),
            parent_history=_checkpoint_history(sel),
        )
    state = _fake_checkpoint(tmp_path / "state.pt", state={"net.weight": "not-a-tensor"})
    with pytest.raises(RuntimeError, match="not a tensor"):
        producer.verify_completed_checkpoint(
            torch.load(state, map_location="cpu", weights_only=False),
            parent_history=_checkpoint_history(state),
        )
    nan = _fake_checkpoint(tmp_path / "nan.pt", state={"net.weight": torch.tensor([float("nan")])})
    with pytest.raises(RuntimeError, match="non-finite checkpoint tensor"):
        producer.verify_completed_checkpoint(
            torch.load(nan, map_location="cpu", weights_only=True),
            parent_history=_checkpoint_history(nan),
        )


def test_verify_completed_checkpoint_accepts_integer_buffers(tmp_path):
    # Real backbones carry integer BN buffers (e.g. num_batches_tracked); those are legal.
    state = {"net.weight": torch.ones(2, 2),
             "net.num_batches_tracked": torch.tensor(3, dtype=torch.int64)}
    path = _fake_checkpoint(tmp_path / "buffers.pt", state=state)
    producer.verify_completed_checkpoint(
        torch.load(path, map_location="cpu", weights_only=True),
        parent_history=_checkpoint_history(path),
    )
    complex_path = _fake_checkpoint(
        tmp_path / "complex.pt", state={"net.weight": torch.ones(2, dtype=torch.complex64)}
    )
    with pytest.raises(RuntimeError, match="unsupported complex dtype"):
        producer.verify_completed_checkpoint(
            torch.load(complex_path, map_location="cpu", weights_only=True),
            parent_history=_checkpoint_history(complex_path),
        )


def test_history_rows_rejects_inf_with_runtime_error(tmp_path):
    saved = torch.load(_fake_checkpoint(tmp_path / "inf.pt"), map_location="cpu", weights_only=True)
    saved["history"][2]["validation_auc"] = float("inf")
    with pytest.raises(RuntimeError, match="not finite"):
        producer.verify_completed_checkpoint(saved, parent_history=saved["history"])


def test_verify_completed_checkpoint_requires_parent_history_match(tmp_path):
    good = _fake_checkpoint(tmp_path / "good.pt")
    other = _history()
    other[0]["validation_auc"] = 0.6
    with pytest.raises(RuntimeError, match="history differs from the parent"):
        producer.verify_completed_checkpoint(
            torch.load(good, map_location="cpu", weights_only=True), parent_history=other
        )


def test_load_checkpoint_requires_state_dict(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"history": []}, path)
    with pytest.raises(RuntimeError, match="state_dict"):
        producer.load_checkpoint(path)


def test_load_tuned_strict_rejects_mismatched_architecture():
    model = nn.Linear(2, 2, bias=False)
    with pytest.raises(RuntimeError):
        producer.load_tuned_strict(model, {"net.weight": torch.ones(2, 2)})
    producer.load_tuned_strict(model, {"weight": torch.ones(2, 2)})
    assert not model.training


def test_state_dict_digest_is_canonical_and_value_sensitive():
    from scripts.run_facenet_validation_cuda_v2 import state_dict_digest as canonical

    state = {"a": torch.zeros(2), "b": torch.ones(2, 2)}
    assert producer.state_dict_digest(state) == canonical(state)
    reordered = {"b": state["b"], "a": state["a"]}
    assert producer.state_dict_digest(reordered) == producer.state_dict_digest(state)
    assert producer.state_dict_digest({"a": torch.ones(2)}) != producer.state_dict_digest(state)


# --------------------------------------------------------------------------------------
# Measurement math
# --------------------------------------------------------------------------------------
def test_measure_pair_loss_is_global_not_mean_of_batch_means():
    from scripts.validation_pair_loss_v1 import measure

    scores = np.array([0.5, 0.5, 0.5, -0.5], dtype=float)
    labels = np.array([1, 1, 1, 0], dtype=float)
    weights = np.array([1.0, 1.0, 1.0, 9.0], dtype=float)
    got = producer.measure_pair_loss(scores, labels, weights, producer.MARGIN)
    reference = measure(scores, labels, margin=producer.MARGIN, weights=weights)
    assert got == reference
    assert got["numerator"] == pytest.approx(1.5)
    assert got["denominator"] == pytest.approx(12.0)


def test_measure_pair_loss_rejects_out_of_range_unclipped_scores():
    with pytest.raises(ValueError):
        producer.measure_pair_loss([1.5], [1], [1.0], producer.MARGIN)


def test_measure_pair_loss_wrapper_checks_nan_outputs(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "scripts.validation_pair_loss_v1",
        SimpleNamespace(measure=lambda *a, **k: {
            "numerator": float("nan"), "denominator": 1.0, "validation_loss": float("nan")}),
    )
    with pytest.raises(ValueError, match="not finite"):
        producer.measure_pair_loss([0.0], [1], [1.0], producer.MARGIN)


def test_check_auc_reproduction_tolerance():
    native = 0.7
    scores = np.array([0.9, 0.8, 0.1, 0.0])
    labels = np.array([1, 1, 0, 0])
    producer.check_auc_reproduction(scores, labels, native, tol=1.0)
    with pytest.raises(RuntimeError, match="does not reproduce"):
        producer.check_auc_reproduction(scores, labels, native, tol=1e-9)
    with pytest.raises(RuntimeError, match="out of range"):
        producer.check_auc_reproduction(scores, labels, 1.5)
    with pytest.raises(RuntimeError, match="tolerance out of range"):
        producer.check_auc_reproduction(scores, labels, native, tol=0.0)


def test_paired_bootstrap_ci_is_pair_level_and_reproducible():
    frozen = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    tuned = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    weights = np.ones(6)
    first = producer.paired_bootstrap_ci(frozen, tuned, weights)
    second = producer.paired_bootstrap_ci(frozen, tuned, weights)
    assert first == second
    assert first["resampling_unit"] == "pair"
    assert "subject" in first["not_a"]
    assert first["delta_ci_low"] <= first["delta_observed"] <= first["delta_ci_high"]
    assert first["requested_draws"] == first["draws"] == producer.BOOTSTRAP_DRAWS
    assert first["valid_draws"] + first["unavailable_draws"] == producer.BOOTSTRAP_DRAWS


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"draws": 0}, ValueError),
        ({"draws": 1.5}, ValueError),
        ({"seed": -1}, ValueError),
        ({"alpha": 0.0}, ValueError),
        ({"alpha": 1.0}, ValueError),
        ({"alpha": "0.05"}, ValueError),
    ],
)
def test_paired_bootstrap_ci_argument_guards(kwargs, exc):
    frozen = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(exc):
        producer.paired_bootstrap_ci(frozen, frozen, np.ones(4), **kwargs)


def test_paired_bootstrap_ci_refuses_nonpositive_total_mass():
    frozen = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="positive finite total weight"):
        producer.paired_bootstrap_ci(frozen, frozen, np.zeros(2))
    with pytest.raises(ValueError, match="nonnegative"):
        producer.paired_bootstrap_ci(frozen, frozen, np.array([-1.0, 2.0]))
    with pytest.raises(ValueError, match="aligned"):
        producer.paired_bootstrap_ci(frozen, frozen, np.ones(3))
    with pytest.raises(ValueError, match="finite per-pair losses"):
        producer.paired_bootstrap_ci(np.array([np.nan, 0.1]), frozen, np.ones(2))
    with pytest.raises(ValueError, match="nonempty 1-D"):
        producer.paired_bootstrap_ci(np.array([[0.1, 0.2]]), np.array([[0.1, 0.2]]), np.ones(2))


def test_paired_bootstrap_ci_refuses_inadequate_valid_draws(monkeypatch):
    # Simulate nearly all resamples drawing only zero-weight rows: unavailable draws exceed
    # the allowed fraction, so a CI must not be reported.
    frozen = np.array([0.1, 0.2, 0.3, 0.4])
    weights = np.array([1.0, 1.0, 0.0, 0.0])
    good = np.random.default_rng(0)

    class DegenerateRng:
        def integers(self, low, high, size):
            return good.integers(2, 4, size)  # always zero-weight rows

    monkeypatch.setattr(producer.np.random, "default_rng", lambda seed: DegenerateRng())
    with pytest.raises(RuntimeError, match="insufficient valid bootstrap draws"):
        producer.paired_bootstrap_ci(frozen, frozen, weights, draws=16)


def test_gap_weight_zero_yields_unit_weights_and_helper_refuses_multiplier():
    from age_gap.training.dataset import _pair_weight

    assert producer.GAP_WEIGHT == 0.0
    assert _pair_weight(1, 25, producer.GAP_WEIGHT) == 1.0
    with pytest.raises(ValueError, match="nonnegative"):
        # A mis-wired helper that returned a negative multiplier must fail closed.
        producer.paired_bootstrap_ci(
            np.array([0.1, 0.2]), np.array([0.1, 0.2]), np.array([1.0, -1.0])
        )


def test_device_binding_matches_native(monkeypatch):
    import torch as _torch

    params = {"torch_version": _torch.__version__, "cuda_version": _torch.version.cuda,
              "gpu_name": "Fake GPU", "cudnn_version": _torch.backends.cudnn.version()}
    monkeypatch.setattr(_torch.cuda, "get_device_name", lambda *a, **k: "Fake GPU")
    producer.check_device_binding({"parameters": params})  # sanity
    monkeypatch.setattr(_torch.cuda, "get_device_name", lambda *a, **k: "Other GPU")
    with pytest.raises(RuntimeError, match="gpu_name differs"):
        producer.check_device_binding({"parameters": params})


# --------------------------------------------------------------------------------------
# End-to-end mocked run (fake backbone/inference; real tiny files and real record guards)
# --------------------------------------------------------------------------------------
class FakeBackbone(nn.Module):
    trainable_scopes = {"head": ("w",)}

    def __init__(self):
        super().__init__()
        self.net = nn.Linear(2, 2, bias=False)

    def to(self, device):
        assert device == "cuda"
        return self


def _fake_project_tree(root):
    """Minimal real project skeleton so ancestry construction reads actual files."""
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "metrics/model_inventory.json").write_text("{}", encoding="utf-8")
    (root / "pyproject.toml").write_text("# p\n", encoding="utf-8")
    (root / "uv.lock").write_text("# l\n", encoding="utf-8")
    settings = root / "src/age_gap/settings"
    settings.mkdir(parents=True, exist_ok=True)
    (settings / "settings.toml").write_text("# s\n", encoding="utf-8")
    (root / "src/age_gap/stub.py").write_text("# stub\n", encoding="utf-8")
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for name in (
        "run_strong_backbone_cuda_cell_v1.py",
        "run_facenet_validation_cuda_v2.py",
        "validation_pair_loss_v1.py",
        "run_oriented_campaign.py",
        "run_restricted_matched_campaign.py",
        "evaluate_oriented_cuda_v1.py",
    ):
        (scripts / name).write_text(f"# {name}\n", encoding="utf-8")
    return root


def _mocked_run_env(tmp_path, monkeypatch, *, scores=(0.0, 0.5), mismatch=False):
    """Wire a full run with real record guards over small real files."""
    from age_gap.common.manifest import file_record

    run_dir = tmp_path / "run"
    (run_dir / "private").mkdir(parents=True)
    checkpoint = run_dir / "private/checkpoint.pt"
    _fake_checkpoint(checkpoint)
    pairs_file, pairs = _pairs_file(tmp_path)

    def fake_crop(face_id, crops_dir):
        return tmp_path / f"{face_id}.jpg"

    monkeypatch.setattr(producer, "crop_path", fake_crop)
    for pair in pairs:
        for face in (pair.face_a, pair.face_b):
            (tmp_path / f"{face}.jpg").write_bytes(b"i")

    script = tmp_path / "proposal_script.py"
    script.write_text("# producer\n", encoding="utf-8")
    weight = tmp_path / "casia.pt"
    weight.write_bytes(b"fake-weights")
    _fake_project_tree(tmp_path)

    # A real parent completion manifest file so its record can be snapshotted and re-hashed.
    parent_manifest = run_dir / "summary.manifest.json"
    parent_manifest.write_text(
        json.dumps({"experiment": producer.FACENET_EXPERIMENT, "metrics": {"history": []}}),
        encoding="utf-8",
    )

    rows = producer.declared_val_rows(pairs_file, "faces")
    native = {
        "manifest": parent_manifest,
        "manifest_record": file_record(parent_manifest),
        "native": {"inputs": [file_record(weight)], "outputs": [],
                   "parameters": {"device": "cuda"},
                   "metrics": {"validation_pairs": len(rows)}},
        "parameters": _params(pairs_file=str(pairs_file), crops_dir="faces"),
        "history": _checkpoint_history(checkpoint),
        "declared_rows": len(rows),
        "checkpoint": checkpoint,
        "component": None,
        "experiment": producer.FACENET_EXPERIMENT,
    }

    class FakeDataset:
        def __init__(self):
            self._items = [(ca, cb, 0, 0, 1.0) for _p, ca, cb in rows]

        def __len__(self):
            return len(self._items)

        def __iter__(self):
            return iter(
                (torch.zeros(1), torch.zeros(1), torch.tensor(float(y)), torch.tensor(float(w)))
                for _a, _b, y, _g, w in self._items
            )

    dataset = FakeDataset()
    produced = {}
    frozen_count = len(rows) if not mismatch else len(rows) - 1

    def fake_make_backbone(name, pretrained=True):
        assert pretrained is True and name == "facenet"
        model = FakeBackbone()
        produced.setdefault("frozen", model)
        return model

    def fake_infer(model, ds, *, device, batch_size, margin):
        assert device == "cuda" and batch_size == producer.BATCH_SIZE
        assert margin == producer.MARGIN
        key = "frozen" if model is produced["frozen"] else "tuned"
        values = np.full(
            len(rows) if key == "tuned" else frozen_count,
            float(scores[0] if key == "frozen" else scores[1]), np.float32,
        )
        labels, weights = producer.declared_labels_weights(rows)
        return {"scores": values, "labels": labels, "weights": weights,
                "n_pairs": int(values.size)}

    def fake_ancestry(**kwargs):
        # Real record construction over the real tiny files.
        return real_build_ancestry(
            script=script, native=native["native"], component=None,
            checkpoint=checkpoint, weight={"path": str(weight)}, pairs=pairs_file,
            rows=rows, root=tmp_path,
            parent_manifest=native["manifest"], parent_record=native["manifest_record"],
        )

    real_build_ancestry = producer.build_ancestry_records
    monkeypatch.setattr(producer, "load_native", lambda train: native)
    monkeypatch.setattr(producer, "bind_pretrained_weight", lambda name, nat: {
        "path": str(weight), "inventory_key": "facenet_casia",
        "record": file_record(weight), "inventory_record": file_record(weight)})
    monkeypatch.setattr(producer, "require_cuda", lambda: None)
    monkeypatch.setattr(producer, "check_device_binding", lambda nat: None)
    monkeypatch.setattr(producer, "make_backbone", fake_make_backbone)
    monkeypatch.setattr(producer, "build_dataset", lambda split, pairs, model, crops: dataset)
    monkeypatch.setattr(producer, "infer_scores", fake_infer)
    monkeypatch.setattr(producer, "build_ancestry_records", fake_ancestry)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1234)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2345)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "Fake GPU")
    args = Namespace(train=run_dir, out=tmp_path / "out")
    return args, native, rows, dataset, [script, weight, pairs_file], mismatch


def test_mocked_run_writes_private_scores_and_aggregate_summary(tmp_path, monkeypatch):
    args, native, rows, dataset, inputs, _ = _mocked_run_env(
        tmp_path, monkeypatch, scores=(0.0, 0.5)
    )
    manifest = producer.run(args)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["experiment"] == producer.EXPERIMENT
    assert payload["metrics"]["n_validation_rows_declared"] == 3
    assert payload["metrics"]["n_validation_rows_scored"] == 3
    assert payload["metrics"]["validation_rows_dropped"] == 0
    assert payload["metrics"]["tuned"]["validation_auc"] == pytest.approx(0.5)
    assert payload["metrics"]["execution_complete"] is True
    assert payload["metrics"]["publication_ready"] is False
    assert payload["metrics"]["peaks"]["peak_allocated_bytes"] == 1234
    assert payload["metrics"]["paired_bootstrap_ci"]["requested_draws"] == 1000
    assert payload["parameters"]["seed"] == 42
    assert "--execute" in payload["parameters"]["command"]
    import shlex

    assert payload["parameters"]["command"] == " ".join(
        shlex.quote(part) for part in payload["command"]
    )

    # Every declared ancestry record was really re-hashed from disk.
    root = tmp_path
    for record in payload["inputs"]:
        path = Path(record["path"])
        path = path if path.is_absolute() else root / path
        assert producer.file_record(path) == record

    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    serialized = json.dumps(summary)
    assert "scores_frozen" not in serialized and "scores_tuned" not in serialized
    assert "pair_ids" not in serialized
    assert "p0" not in serialized
    assert summary["privateness"]["raw_pair_scores"].startswith("private/")
    npz = np.load(args.out / "private/scores.npz", allow_pickle=True)
    assert npz["scores_frozen"].tolist() == [0.0] * 3
    assert npz["scores_tuned"].tolist() == [0.5] * 3
    assert npz["pair_ids"].tolist() == [p.pair_id for p, _a, _b in rows]


def test_mocked_run_refuses_arm_row_mismatch(tmp_path, monkeypatch):
    args, native, rows, dataset, inputs, _ = _mocked_run_env(
        tmp_path, monkeypatch, scores=(0.0, 0.5), mismatch=True
    )
    with pytest.raises(RuntimeError, match="scored 2 of 3"):
        producer.run(args)
    assert not (args.out / "summary.manifest.json").exists()


def test_mocked_run_refuses_swapped_declared_labels(tmp_path, monkeypatch):
    args, native, rows, dataset, inputs, _ = _mocked_run_env(tmp_path, monkeypatch)
    labels, weights = producer.declared_labels_weights(rows)
    original_infer = producer.infer_scores

    def swapped_infer(model, ds, **kwargs):
        out = original_infer(model, ds, **kwargs)
        out["labels"] = labels[::-1].copy()
        return out

    monkeypatch.setattr(producer, "infer_scores", swapped_infer)
    with pytest.raises(RuntimeError, match="labels do not match"):
        producer.run(args)
    assert not (args.out / "summary.manifest.json").exists()


def test_mocked_run_aborts_when_source_mutates_after_native_check(tmp_path, monkeypatch):
    args, native, rows, dataset, inputs, _ = _mocked_run_env(tmp_path, monkeypatch)
    script = inputs[0]
    real_make = producer.make_backbone

    def mutating_make(name, pretrained=True):
        model = real_make(name, pretrained=pretrained)
        script.write_text("# mutated during model construction\n", encoding="utf-8")
        return model

    monkeypatch.setattr(producer, "make_backbone", mutating_make)
    with pytest.raises(RuntimeError, match="ancestry changed after model load"):
        producer.run(args)
    assert not (args.out / "summary.manifest.json").exists()


def test_mocked_run_aborts_when_weight_mutates_before_model_construction(tmp_path, monkeypatch):
    # The mutation lands AFTER the pre-construction ancestry check but BEFORE make_backbone:
    # only the post-load re-hash can catch it, so the pre-construction snapshot must be real.
    args, native, rows, dataset, inputs, _ = _mocked_run_env(tmp_path, monkeypatch)
    weight = inputs[1]
    real_load = producer.load_checkpoint

    def mutating_load(checkpoint):
        saved = real_load(checkpoint)
        weight.write_bytes(b"mutated-pretrained-weight")
        return saved

    monkeypatch.setattr(producer, "load_checkpoint", mutating_load)
    with pytest.raises(RuntimeError, match="ancestry changed after model load"):
        producer.run(args)
    assert not (args.out / "summary.manifest.json").exists()


def _mutate_parent_manifest(native):
    """Edit only the parent completion manifest; its referenced files stay byte-identical."""
    payload = json.loads(native["manifest"].read_text(encoding="utf-8"))
    payload["metrics"]["history"] = [{"epoch": 1, "validation_auc": 0.0}]
    native["manifest"].write_text(json.dumps(payload), encoding="utf-8")


def test_build_ancestry_records_rejects_parent_manifest_changed_before_snapshot(tmp_path, monkeypatch):
    args, native, rows, dataset, inputs, _ = _mocked_run_env(tmp_path, monkeypatch)
    _mutate_parent_manifest(native)
    with pytest.raises(RuntimeError, match="changed before ancestry snapshot"):
        producer.build_ancestry_records(
            script=inputs[0], native=native["native"], component=None,
            checkpoint=native["checkpoint"], weight={"path": str(inputs[1])},
            pairs=inputs[2], rows=rows, root=tmp_path,
            parent_manifest=native["manifest"], parent_record=native["manifest_record"],
        )


def test_load_native_snapshots_parent_manifest_and_carries_it_verbatim(tmp_path, monkeypatch):
    # Mutation happens *during* load (after parse/verify): only the post-load re-hash catches it.
    run_dir, checkpoint = _write_native(tmp_path)
    manifest = run_dir / "summary.manifest.json"
    real_contract = producer.native_contract

    def mutating_contract(native):
        out = real_contract(native)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["metrics"]["history"] = [{"epoch": 1, "train_loss": 0.0,
                                          "validation_auc": 0.0,
                                          "mean_gradient_norm": 0.0,
                                          "epoch_seconds": 0.0}]
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return out

    monkeypatch.setattr(producer, "native_contract", mutating_contract)
    with pytest.raises(RuntimeError, match="changed while being loaded"):
        producer.load_native(run_dir)


def test_mocked_run_aborts_when_parent_manifest_mutates_before_model(tmp_path, monkeypatch):
    args, native, rows, dataset, inputs, _ = _mocked_run_env(tmp_path, monkeypatch)
    _mutate_parent_manifest(native)
    with pytest.raises(RuntimeError, match="changed before ancestry snapshot"):
        producer.run(args)
    assert not (args.out / "summary.manifest.json").exists()


def test_mocked_run_aborts_when_parent_manifest_mutates_after_load(tmp_path, monkeypatch):
    args, native, rows, dataset, inputs, _ = _mocked_run_env(tmp_path, monkeypatch)
    real_load = producer.load_checkpoint

    def mutating_load(checkpoint):
        saved = real_load(checkpoint)
        _mutate_parent_manifest(native)
        return saved

    monkeypatch.setattr(producer, "load_checkpoint", mutating_load)
    with pytest.raises(RuntimeError, match="ancestry changed after model load"):
        producer.run(args)
    assert not (args.out / "summary.manifest.json").exists()


def test_mocked_run_aborts_when_parent_manifest_mutates_after_inference(tmp_path, monkeypatch):
    args, native, rows, dataset, inputs, _ = _mocked_run_env(tmp_path, monkeypatch)
    real_infer = producer.infer_scores
    calls = {"n": 0}

    def mutating_infer(model, ds, **kwargs):
        out = real_infer(model, ds, **kwargs)
        calls["n"] += 1
        if calls["n"] == 2:
            _mutate_parent_manifest(native)
        return out

    monkeypatch.setattr(producer, "infer_scores", mutating_infer)
    with pytest.raises(RuntimeError, match="ancestry changed after inference"):
        producer.run(args)
    assert not (args.out / "summary.manifest.json").exists()


def test_build_ancestry_records_carries_parent_and_component_records_verbatim(tmp_path, monkeypatch):
    from age_gap.common.manifest import file_record

    pairs_file, rows = _pairs_file(tmp_path)
    _real_crops(tmp_path, monkeypatch, rows)
    script = tmp_path / "proposal_script.py"
    script.write_text("# producer\n", encoding="utf-8")
    weight = tmp_path / "casia.pt"
    weight.write_bytes(b"w")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"ckpt")
    other_input = tmp_path / "upstream_input.json"
    other_input.write_text("{}", encoding="utf-8")
    _fake_project_tree(tmp_path)

    native = {
        "inputs": [file_record(other_input), file_record(weight)],
        "outputs": [file_record(checkpoint)],
    }
    component = {"inputs": [file_record(pairs_file)], "outputs": [file_record(checkpoint)]}
    declared = producer.declared_val_rows(pairs_file, "faces")
    records = producer.build_ancestry_records(
        script=script, native=native, component=component, checkpoint=checkpoint,
        weight={"path": str(weight)}, pairs=pairs_file, rows=declared, root=tmp_path,
    )
    paths = [record["path"] for record in records]
    # Parent + component records are carried verbatim, deduplicated.
    for record in native["inputs"] + native["outputs"] + component["inputs"]:
        assert record in records
    assert paths.count(file_record(checkpoint)["path"]) == 1
    # Every crop the declared rows need is present.
    for _pair, ca, cb in declared:
        assert file_record(ca)["path"] in paths
        assert file_record(cb)["path"] in paths
    # Mutating any declared upstream record is caught against the snapshot.
    other_input.write_text('{"changed": 1}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="declared ancestry changed"):
        producer.check_ancestry(records, "test stage")


def test_mocked_run_aborts_when_inputs_mutate_during_manifest_write(tmp_path, monkeypatch):
    args, native, rows, dataset, inputs, _ = _mocked_run_env(tmp_path, monkeypatch)
    source = inputs[0]
    writer = producer.write_experiment_manifest

    def changed_writer(path, **kwargs):
        source.write_text("# mutated at publication\n", encoding="utf-8")
        return writer(path, **kwargs)

    monkeypatch.setattr(producer, "write_experiment_manifest", changed_writer)
    with pytest.raises(RuntimeError, match="changed during manifest publication"):
        producer.run(args)
    assert not (args.out / "summary.manifest.json").exists()


def test_mocked_run_leaves_no_completion_marker_on_exception(tmp_path, monkeypatch):
    args, native, rows, dataset, inputs, _ = _mocked_run_env(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("synthetic publication failure")

    monkeypatch.setattr(producer, "validate_written_inputs", boom)
    with pytest.raises(RuntimeError, match="synthetic publication failure"):
        producer.run(args)
    assert not (args.out / "summary.manifest.json").exists()


def test_real_inference_loop_keeps_order_tail_and_disables_gradients():
    # Exercise the real DataLoader/inference path on prescribed CPU tensors only;
    # run() still requires CUDA. No optimizer, face data or pretrained weights.
    from torch.nn import functional as F
    from torch.utils.data import TensorDataset

    class UnitVectors(nn.Module):
        def forward(self, value):
            assert not self.training and not torch.is_grad_enabled()
            return F.normalize(value, dim=-1)

    left = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [1., -1.], [3., 4.]])
    right = torch.tensor([[1., 0.], [1., 0.], [1., -1.], [-1., 1.], [4., 3.]])
    labels = torch.tensor([1., 0., 0., 0., 1.])
    weights = torch.arange(1., 6.)
    dataset = TensorDataset(left, right, labels, weights)
    result = producer.infer_scores(
        UnitVectors(), dataset, device="cpu", batch_size=2, margin=producer.MARGIN
    )
    assert result["n_pairs"] == 5
    np.testing.assert_allclose(result["scores"], [1., 0., 0., -1., .96], atol=1e-6)
    np.testing.assert_array_equal(result["labels"], labels.numpy())
    np.testing.assert_array_equal(result["weights"], weights.numpy())
    assert result["scores"].dtype == np.float32


def test_real_inference_loop_refuses_empty_input():
    from torch.utils.data import TensorDataset

    empty = TensorDataset(torch.empty(0, 2), torch.empty(0, 2),
                          torch.empty(0), torch.empty(0))
    with pytest.raises(RuntimeError, match="empty validation inference"):
        producer.infer_scores(nn.Identity(), empty, device="cpu", batch_size=2,
                              margin=producer.MARGIN)


def test_cuda_requirement_has_no_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA required; no CPU fallback"):
        producer.require_cuda()
