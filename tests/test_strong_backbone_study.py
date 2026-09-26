from __future__ import annotations

import json

import pytest

from age_gap.common.manifest import write_experiment_manifest
from scripts.strong_backbone_study import (
    _aggregate,
    _ensure_run_outputs_clear,
    _validate_completed_run,
)


def test_aggregate_marks_partial_campaign_and_lists_missing_run(tmp_path) -> None:
    run_id = "e8_random_head_lr1e-06_s42"
    (tmp_path / f"run_{run_id}.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "epochs": 8,
                "negative_type": "random",
                "scope": "head",
                "learning_rate": 1e-6,
                "seed": 42,
                "metrics": {"fgnet.large_gap": 0.9},
                "metric_deltas": {"fgnet.large_gap": -0.1},
                "training_history": [
                    {
                        "epoch": 1,
                        "training_loss": 0.4,
                        "validation_auc": 0.8,
                        "mean_gradient_norm": 0.2,
                    },
                    {
                        "epoch": 2,
                        "training_loss": 0.3,
                        "validation_auc": 0.9,
                        "mean_gradient_norm": 0.1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run_random_head_lr1e-06_s42.json").write_text(
        json.dumps(
            {
                "run_id": "random_head_lr1e-06_s42",
                "epochs": 1,
                "negative_type": "random",
                "scope": "head",
                "learning_rate": 1e-6,
                "seed": 42,
                "metrics": {"fgnet.large_gap": 0.8},
            }
        ),
        encoding="utf-8",
    )

    output = _aggregate(
        tmp_path,
        epochs=8,
        negatives=["random"],
        scopes=["head"],
        learning_rates=[1e-6],
        seeds=[42, 1],
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["campaign"] == {
        "epochs": 8,
        "negatives": ["random"],
        "scopes": ["head"],
        "learning_rates": [1e-6],
        "seeds": [42, 1],
        "expected_runs": 2,
        "completed_runs": 1,
        "remaining_runs": 1,
        "complete": False,
        "missing_run_ids": ["e8_random_head_lr1e-06_s1"],
    }
    assert summary["cells"]["e8|random|head|1e-06"]["metrics"]["fgnet.large_gap"] == {
        "mean": 0.9,
        "std": 0.0,
        "n": 1,
    }
    cell = summary["cells"]["e8|random|head|1e-06"]
    assert cell["best_validation_epochs"] == [2]
    assert cell["metric_deltas"]["fgnet.large_gap"] == {
        "mean": -0.1,
        "std": 0.0,
        "n": 1,
    }
    assert cell["training_trajectory"] == [
        {
            "epoch": 1,
            "n": 1,
            "training_loss": {"mean": 0.4, "std": 0.0},
            "validation_auc": {"mean": 0.8, "std": 0.0},
            "mean_gradient_norm": {"mean": 0.2, "std": 0.0},
        },
        {
            "epoch": 2,
            "n": 1,
            "training_loss": {"mean": 0.3, "std": 0.0},
            "validation_auc": {"mean": 0.9, "std": 0.0},
            "mean_gradient_norm": {"mean": 0.1, "std": 0.0},
        },
    ]
    assert output.with_suffix(".manifest.json").is_file()


def _write_valid_run_manifest(tmp_path, *, pair_file, base_weight, inventory, checkpoint, result):
    parameters = {
        "backbone": "adaface_ir101",
        "negative_type": "random",
        "scope": "full",
        "learning_rate": 1e-6,
        "seed": 42,
        "epochs": 8,
        "patience": 3,
        "batch_size": 16,
    }
    manifest = tmp_path / "run.manifest.json"
    write_experiment_manifest(
        manifest,
        experiment="strong-backbone-mechanism-study",
        parameters=parameters,
        metrics={},
        inputs=[pair_file, base_weight, inventory],
        outputs=[checkpoint, result],
    )
    return manifest, parameters


def test_completed_run_resume_validates_manifest_checksums(tmp_path) -> None:
    pair_file = tmp_path / "pairs.jsonl"
    base_weight = tmp_path / "base.pt"
    inventory = tmp_path / "inventory.json"
    checkpoint = tmp_path / "e8_random_full_lr1e-06_s42.pt"
    result = tmp_path / "run_e8_random_full_lr1e-06_s42.json"
    for path in (pair_file, base_weight, inventory, checkpoint, result):
        path.write_text(path.name, encoding="utf-8")
    manifest, parameters = _write_valid_run_manifest(
        tmp_path,
        pair_file=pair_file,
        base_weight=base_weight,
        inventory=inventory,
        checkpoint=checkpoint,
        result=result,
    )

    _validate_completed_run(
        manifest_path=manifest,
        expected_parameters=parameters,
        pairs=pair_file,
        provenance_inputs=[base_weight, inventory],
        result_path=result,
        run_id="e8_random_full_lr1e-06_s42",
    )

    checkpoint.write_text("changed checkpoint", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint is missing or its SHA-256 differs"):
        _validate_completed_run(
            manifest_path=manifest,
            expected_parameters=parameters,
            pairs=pair_file,
            provenance_inputs=[base_weight, inventory],
            result_path=result,
            run_id="e8_random_full_lr1e-06_s42",
        )


def test_completed_run_resume_rejects_parameter_or_input_drift(tmp_path) -> None:
    pair_file = tmp_path / "pairs.jsonl"
    base_weight = tmp_path / "base.pt"
    inventory = tmp_path / "inventory.json"
    checkpoint = tmp_path / "e8_random_full_lr1e-06_s42.pt"
    result = tmp_path / "run_e8_random_full_lr1e-06_s42.json"
    for path in (pair_file, base_weight, inventory, checkpoint, result):
        path.write_text(path.name, encoding="utf-8")
    manifest, parameters = _write_valid_run_manifest(
        tmp_path,
        pair_file=pair_file,
        base_weight=base_weight,
        inventory=inventory,
        checkpoint=checkpoint,
        result=result,
    )

    changed_parameters = {**parameters, "patience": 2}
    with pytest.raises(RuntimeError, match="run parameters do not match"):
        _validate_completed_run(
            manifest_path=manifest,
            expected_parameters=changed_parameters,
            pairs=pair_file,
            provenance_inputs=[base_weight, inventory],
            result_path=result,
            run_id="e8_random_full_lr1e-06_s42",
        )

    pair_file.write_text("changed input", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input paths or SHA-256 checksums changed"):
        _validate_completed_run(
            manifest_path=manifest,
            expected_parameters=parameters,
            pairs=pair_file,
            provenance_inputs=[base_weight, inventory],
            result_path=result,
            run_id="e8_random_full_lr1e-06_s42",
        )


def test_orphan_checkpoint_requires_fresh_checkpoint_path_without_overwrite(tmp_path) -> None:
    run_id = "e8_random_full_lr1e-06_s42"
    result = tmp_path / f"run_{run_id}.json"
    manifest = tmp_path / f"run_{run_id}.manifest.json"
    orphan = tmp_path / f"{run_id}.pt"
    fresh = tmp_path / "retry" / f"{run_id}.pt"
    orphan.write_text("pilot checkpoint", encoding="utf-8")

    with pytest.raises(RuntimeError, match="orphan checkpoint exists"):
        _ensure_run_outputs_clear(
            run_id=run_id,
            result_path=result,
            manifest_path=manifest,
            checkpoint_path=orphan,
            default_checkpoint_path=orphan,
        )
    assert orphan.read_text(encoding="utf-8") == "pilot checkpoint"

    _ensure_run_outputs_clear(
        run_id=run_id,
        result_path=result,
        manifest_path=manifest,
        checkpoint_path=fresh,
        default_checkpoint_path=orphan,
    )


def test_orphan_result_is_not_replaced_even_with_alternate_checkpoint_dir(tmp_path) -> None:
    result = tmp_path / "run.json"
    result.write_text("partial result", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete result artifacts"):
        _ensure_run_outputs_clear(
            run_id="run",
            result_path=result,
            manifest_path=tmp_path / "run.manifest.json",
            checkpoint_path=tmp_path / "retry" / "run.pt",
            default_checkpoint_path=tmp_path / "run.pt",
        )
