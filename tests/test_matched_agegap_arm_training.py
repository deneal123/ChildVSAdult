from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import train_matched_agegap_arms as runner


class _FakeModel:
    def to(self, _device):
        return self

    def eval(self):
        return self


def _rows(arm: str) -> list[dict]:
    gap = 2 if arm == "low" else 30
    person = f"{arm}_person"
    return [
        {
            "pair_id": f"{arm}_positive",
            "face_a": f"{arm}_a",
            "face_b": f"{arm}_b",
            "label": 1,
            "age_a": 10,
            "age_b": 10 + gap,
            "age_gap": gap,
            "identity_group_a": person,
            "identity_group_b": person,
            "split": "train",
            "status": "ok",
        },
        {
            "pair_id": f"{arm}_negative",
            "face_a": f"{arm}_a",
            "face_b": f"{arm}_impostor",
            "label": 0,
            "age_a": 10,
            "age_b": 10 + gap,
            "age_gap": gap,
            "identity_group_a": person,
            "identity_group_b": f"{arm}_impostor",
            "split": "train",
            "status": "ok",
        },
        *[
            {
                "pair_id": f"common_{split}",
                "face_a": f"common_{split}_a",
                "face_b": f"common_{split}_b",
                "label": label,
                "age_a": 10,
                "age_b": 40 if label else 11,
                "age_gap": 30 if label else 1,
                "identity_group_a": f"{split}_person",
                "identity_group_b": f"{split}_person" if label else f"{split}_impostor",
                "split": split,
                "status": "ok",
            }
            for split in ("val", "test")
            for label in (1, 0)
        ],
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_arm_validator_checks_positive_strata_and_identical_heldout(tmp_path) -> None:
    low_path, cross_path = tmp_path / "low.jsonl", tmp_path / "cross.jsonl"
    low, cross = _rows("low"), _rows("cross")
    _write_jsonl(low_path, low)
    _write_jsonl(cross_path, cross)

    result = runner.validate_arm_files(low_path, cross_path)

    assert result["counts"]["low"]["train_positive"] == 1
    assert result["counts"]["cross"]["train_negative"] == 1
    assert set(result["heldout_sha256"]) == {"val", "test"}
    groups = {
        str(row[f"identity_group_{side}"])
        for row in low + cross
        for side in ("a", "b")
    }
    overlapping_map = {group: group for group in groups}
    overlapping_map["cross_person"] = "low_person"
    with pytest.raises(ValueError, match="train-person sets overlap"):
        runner.validate_arm_files(low_path, cross_path, group_to_person=overlapping_map)
    cross[-1] = {**cross[-1], "age_b": 39}
    _write_jsonl(cross_path, cross)
    with pytest.raises(ValueError, match="identical test rows"):
        runner.validate_arm_files(low_path, cross_path)


def test_runner_uses_private_arms_and_resumes_complete_arm_seed_runs(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    low_path = data_root / "interim/matched_agegap_arms/low_arm.jsonl"
    cross_path = data_root / "interim/matched_agegap_arms/cross_arm.jsonl"
    low_path.parent.mkdir(parents=True)
    _write_jsonl(low_path, _rows("low"))
    _write_jsonl(cross_path, _rows("cross"))
    person_clusters = data_root / "processed/person_clusters.jsonl"
    person_clusters.parent.mkdir(parents=True, exist_ok=True)
    all_rows = _rows("low") + _rows("cross")
    groups = sorted(
        {
            str(row[f"identity_group_{side}"])
            for row in all_rows
            for side in ("a", "b")
        }
    )
    person_clusters.write_text(
        "\n".join(
            json.dumps({"identity_group_id": group, "person_id": group}) for group in groups
        )
        + "\n",
        encoding="utf-8",
    )
    models_root, metrics_root = tmp_path / "models", tmp_path / "metrics"
    models_root.mkdir()
    metrics_root.mkdir()
    base_weight = tmp_path / "facenet-casia.pt"
    base_weight.write_bytes(b"casia base weights")
    inventory = metrics_root / "model_inventory.json"
    inventory.write_text(
        json.dumps(
            {"models": {"facenet_casia": {"artifact": {"path": str(base_weight)}}}}
        ),
        encoding="utf-8",
    )
    fgnet_cache = tmp_path / "fgnet.npz"
    fgnet_cache.write_bytes(b"fgnet cache")
    canonical = data_root / "processed/pairs.jsonl"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"canonical sentinel\n")
    original_canonical = canonical.read_bytes()
    trained: list[dict] = []

    def fake_data_path(kind: str, *parts: str) -> Path:
        if kind == "models_dir":
            return models_root.joinpath(*parts)
        if kind == "metrics_dir":
            return metrics_root.joinpath(*parts)
        return data_root.joinpath(*parts)

    def fake_finetune(**kwargs) -> Path:
        trained.append(kwargs)
        checkpoint = Path(kwargs["ckpt_out"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint:{Path(kwargs['pairs_file']).stem}".encode())
        return checkpoint

    metadata = {
        "n_positive_retained": 2,
        "n_positive_source": 2,
        "n_positive_unmatched": 0,
        "positive_coverage": 1.0,
        "n_large_gap_positive_retained": 1,
        "n_large_gap_positive_source": 1,
        "n_large_gap_positive_unmatched": 0,
        "large_gap_positive_coverage": 1.0,
        "stratum_age_gap": np.asarray([30, 2, 30, 2]),
        "subject_a": np.asarray(["p1", "p2", "n1", "n2"]),
        "subject_b": np.asarray(["p1", "p2", "n3", "n4"]),
    }
    pair_data = (
        [np.zeros((2, 2, 3), dtype=np.uint8)] * 4,
        [np.zeros((2, 2, 3), dtype=np.uint8)] * 4,
        np.asarray([1, 1, 0, 0]),
        np.asarray([True, False, True, False]),
        metadata,
    )
    monkeypatch.setattr(runner, "data_path", fake_data_path)
    monkeypatch.setattr(runner, "torch_device", lambda: "cpu")
    monkeypatch.setattr(runner, "make_backbone", lambda *_args, **_kwargs: _FakeModel())
    monkeypatch.setattr(runner, "finetune", fake_finetune)
    monkeypatch.setattr(runner, "load_finetuned", lambda *_args, **_kwargs: _FakeModel())
    monkeypatch.setattr(runner, "load_matched_fgnet", lambda *_args, **_kwargs: pair_data)
    monkeypatch.setattr(runner, "pair_scores", lambda *_args, **_kwargs: np.asarray([0.9, 0.8, 0.2, 0.1]))
    monkeypatch.setattr(
        runner,
        "evaluate_our_split",
        lambda _model, _device, split, **_kwargs: {"overall_auc": 0.75, "split": split},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    summary_path = runner.run_experiment(
        low_arm=low_path,
        cross_arm=cross_path,
        models_dir=models_root / "matched",
        result_dir=metrics_root / "matched",
        fgnet_cache=fgnet_cache,
        seeds=[42, 1],
        epochs=10,
        learning_rate=3e-5,
        trainable_scope="head",
        batch_size=64,
        patience=3,
        checkpoint_selection="last_epoch",
        device="cpu",
        bootstrap_resamples=25,
        max_runs=2,
    )

    assert canonical.read_bytes() == original_canonical
    assert len(trained) == 2
    assert [(Path(call["pairs_file"]).stem, call["seed"]) for call in trained] == [
        ("low_arm", 42),
        ("cross_arm", 42),
    ]
    assert {Path(call["pairs_file"]) for call in trained} == {low_path, cross_path}
    assert all(call["backbone_name"] == "facenet" for call in trained)
    assert all(call["seed"] == 42 and call["epochs"] == 10 for call in trained)
    assert all(call["lr"] == 3e-5 and call["trainable_scope"] == "head" for call in trained)
    assert all(call["batch_size"] == 64 and call["gap_weight"] == 0.0 for call in trained)
    assert all(call["checkpoint_selection"] == "last_epoch" for call in trained)
    result_files = sorted((metrics_root / "matched").glob("*.json"))
    result_files = [path for path in result_files if not path.name.endswith(".manifest.json")]
    result_files = [path for path in result_files if path.name != "summary.json"]
    assert len(result_files) == 2
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_files]
    assert {row["arm"] for row in results} == {"low", "cross"}
    assert all(row["fgnet_protocol"] == "endpoint_age_matched" for row in results)
    assert all(row["checkpoint_selection"] == "last_epoch" for row in results)
    assert all(row["selected_epoch"] == 10 for row in results)
    assert all(row["run_id"].endswith("_last") for row in results)
    assert all(row["person_clusters_sha256"] for row in results)
    assert all(row["metrics"]["fgnet"]["large_gap_25_plus"]["n_pairs"] == 2.0 for row in results)
    assert all(Path(path.with_suffix(".manifest.json")).is_file() for path in result_files)
    assert all("face_a" not in path.read_text(encoding="utf-8") for path in result_files)
    manifest = json.loads(result_files[0].with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert {record["sha256"] for record in manifest["inputs"]} >= {
        results[0]["arm_file_sha256"],
        results[0]["checkpoint_sha256"],
    }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["complete"] is False
    assert summary["paired_subject_bootstrap_pending_seeds"] == [1]
    assert summary["aggregates"]["paired_cross_minus_low"][
        "fgnet.large_gap_25_plus.roc_auc"
    ]["n"] == 1
    assert "42" in summary["paired_subject_bootstrap_cross_minus_low"]
    summary_manifest = json.loads(
        summary_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    summary_input_names = [Path(record["path"]).name for record in summary_manifest["inputs"]]
    assert fgnet_cache.name in summary_input_names
    assert "person_clusters.jsonl" in summary_input_names
    assert sum(name.endswith("_last.pt") for name in summary_input_names) == 2
    assert sum(name.endswith("_last.manifest.json") for name in summary_input_names) == 2

    resumed = runner.run_experiment(
        low_arm=low_path,
        cross_arm=cross_path,
        models_dir=models_root / "matched",
        result_dir=metrics_root / "matched",
        fgnet_cache=fgnet_cache,
        seeds=[42, 1],
        epochs=10,
        learning_rate=3e-5,
        trainable_scope="head",
        batch_size=64,
        patience=3,
        checkpoint_selection="last_epoch",
        device="cpu",
        bootstrap_resamples=25,
    )
    assert resumed == summary_path
    assert len(trained) == 4
    completed = json.loads(resumed.read_text(encoding="utf-8"))
    assert completed["complete"] is True
    assert completed["aggregates"]["paired_cross_minus_low"][
        "fgnet.large_gap_25_plus.roc_auc"
    ]["n"] == 2

    runner.run_experiment(
        low_arm=low_path,
        cross_arm=cross_path,
        models_dir=models_root / "matched",
        result_dir=metrics_root / "matched",
        fgnet_cache=fgnet_cache,
        seeds=[42, 1],
        epochs=10,
        learning_rate=3e-5,
        trainable_scope="head",
        batch_size=64,
        patience=3,
        checkpoint_selection="last_epoch",
        device="cpu",
        bootstrap_resamples=25,
    )
    assert len(trained) == 4


def test_orphan_checkpoint_is_not_overwritten(tmp_path, monkeypatch) -> None:
    low_path, cross_path = tmp_path / "low.jsonl", tmp_path / "cross.jsonl"
    _write_jsonl(low_path, _rows("low"))
    _write_jsonl(cross_path, _rows("cross"))
    models_dir, result_dir = tmp_path / "models", tmp_path / "metrics"
    models_dir.mkdir()
    result_dir.mkdir()
    data_root = tmp_path / "data"
    groups = sorted(
        {
            str(row[f"identity_group_{side}"])
            for row in (_rows("low") + _rows("cross"))
            for side in ("a", "b")
        }
    )
    cluster_path = data_root / "processed/person_clusters.jsonl"
    cluster_path.parent.mkdir(parents=True)
    _write_jsonl(
        cluster_path,
        [{"identity_group_id": group, "person_id": group} for group in groups],
    )
    base_weight = tmp_path / "base.pt"
    base_weight.write_bytes(b"base")
    inventory_path = result_dir / "model_inventory.json"
    inventory_path.write_text(
        json.dumps({"models": {"facenet_casia": {"artifact": {"path": str(base_weight)}}}}),
        encoding="utf-8",
    )
    cache = tmp_path / "fgnet.npz"
    cache.write_bytes(b"cache")
    checkpoint = models_dir / "low_facenet_e10_lr3e-05_head_b64_s42.pt"
    checkpoint.write_bytes(b"must-preserve")
    monkeypatch.setattr(
        runner,
        "data_path",
        lambda kind, *parts: (result_dir if kind == "metrics_dir" else data_root).joinpath(*parts),
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.run_experiment(
            low_arm=low_path,
            cross_arm=cross_path,
            models_dir=models_dir,
            result_dir=result_dir,
            fgnet_cache=cache,
            seeds=[42],
            device="cpu",
        )
    assert checkpoint.read_bytes() == b"must-preserve"


def test_subject_paired_bootstrap_delta_and_seed_are_deterministic() -> None:
    labels = np.asarray([1, 1, 0, 0, 1, 1, 0, 0])
    subjects_a = np.asarray(["p1", "p2", "n1", "n2", "p3", "p4", "n3", "n4"])
    subjects_b = np.asarray(["p1", "p2", "n5", "n6", "p3", "p4", "n7", "n8"])
    low = np.asarray([0.7, 0.6, 0.4, 0.3, 0.8, 0.5, 0.2, 0.1])
    cross = np.asarray([0.9, 0.7, 0.3, 0.2, 0.8, 0.6, 0.4, 0.1])
    first = runner._paired_subject_bootstrap_cross_minus_low(
        low, cross, labels, subjects_a, subjects_b, n_boot=100, seed=12
    )
    second = runner._paired_subject_bootstrap_cross_minus_low(
        low, cross, labels, subjects_a, subjects_b, n_boot=100, seed=12
    )
    assert first == second
    assert first["delta_cross_minus_low_auc"] == pytest.approx(0.0)
    assert first["n_valid_resamples"] > 0
