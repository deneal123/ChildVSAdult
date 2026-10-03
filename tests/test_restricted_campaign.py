import json
from copy import deepcopy

import cv2
import numpy as np
import pytest

from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.run_restricted_matched_campaign import (
    audit_crops,
    command,
    finalize,
    require_fresh,
    validate_completed,
    verify_records,
)


def test_audit_decodes_unique_crops_and_no_budget_shrink(tmp_path):
    for name in ("a", "b", "c"):
        assert cv2.imwrite(str(tmp_path / f"{name}.jpg"), np.zeros((112, 112, 3), np.uint8))
    rows = {"LOW": [{"face_a": "a", "face_b": "b"}],
            "CROSS": [{"face_a": "a", "face_b": "c"}]}
    records, audit = audit_crops(rows, resolver=lambda face: tmp_path / f"{face}.jpg")
    assert len(records) == 3
    assert audit["unique_crops"] == 3 and audit["missing_rows_dropped"] == 0
    assert audit["source_shape_counts"] == {"112x112": 3}
    verify_records(records)


@pytest.mark.parametrize("bad", ["missing", "undecodable", "changed"])
def test_bad_crop_refused(tmp_path, monkeypatch, bad):
    path = tmp_path / "a.jpg"
    if bad == "undecodable":
        path.write_bytes(b"not an image")
    elif bad == "changed":
        assert cv2.imwrite(str(path), np.zeros((112, 112, 3), np.uint8))
        real = cv2.imread
        def read_then_mutate(filename):
            image = real(filename)
            path.write_bytes(path.read_bytes() + b"modified")
            return image
        monkeypatch.setattr(cv2, "imread", read_then_mutate)
    with pytest.raises(ValueError):
        audit_crops({"LOW": [{"face_a": "a", "face_b": "a"}]}, resolver=lambda face: path)


def test_fresh_only_preserves_existing_checkpoint(tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    checkpoint = directory / "orphan.pt"
    checkpoint.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        require_fresh([directory])
    assert checkpoint.read_bytes() == b"preserve"


def test_command_fixed_full_budget_cpu(tmp_path):
    cmd = command(tmp_path / "arms", tmp_path / "models", tmp_path / "out")
    assert cmd[cmd.index("--seeds") + 1:cmd.index("--epochs")] == ["42", "1", "2"]
    assert cmd[cmd.index("--epochs") + 1] == "10"
    assert cmd[cmd.index("--checkpoint-selection") + 1] == "last_epoch"
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert "--max-runs" not in cmd


def completed_summary():
    return {"complete": True, "completed_run_count": 6, "expected_run_count": 6,
            "seeds": [42, 1, 2], "training_protocol": {"epochs": 10, "checkpoint_selection": "last_epoch"},
            "runs": [{"arm": arm, "seed": seed, "selected_epoch": 10,
                      "run_id": f"{arm}_{seed}"} for arm in ("low", "cross") for seed in (42, 1, 2)]}


@pytest.mark.parametrize("bad", ["incomplete", "seeds", "epochs", "bestval", "duplicate", "selected"])
def test_completed_guard_rejects_shortened_or_mismatched_campaign(bad):
    summary = completed_summary()
    if bad == "incomplete":
        summary["complete"] = False
    elif bad == "seeds":
        summary["seeds"] = [42]
    elif bad == "epochs":
        summary["training_protocol"]["epochs"] = 1
    elif bad == "bestval":
        summary["training_protocol"]["checkpoint_selection"] = "best_val"
    elif bad == "duplicate":
        summary["runs"][1] = deepcopy(summary["runs"][0])
    elif bad == "selected":
        summary["runs"][0]["selected_epoch"] = 8
    with pytest.raises(ValueError):
        validate_completed(summary)


def campaign_fixture(tmp_path):
    out, models = tmp_path / "out", tmp_path / "models"
    out.mkdir()
    models.mkdir()
    source = tmp_path / "source.py"
    source.write_bytes(b"original source")
    snapshot = out / "snapshot.jsonl"
    snapshot.write_bytes(b"private input records")
    audit = out / "preflight.json"
    audit.write_text("{}")
    preflight = write_experiment_manifest(out / "preflight.manifest.json", experiment="preflight",
        inputs=[source], outputs=[snapshot, audit], parameters={}, metrics={})
    summary = completed_summary()
    eval_inputs = []
    for row in summary["runs"]:
        checkpoint = models / f"{row['run_id']}.pt"
        checkpoint.write_bytes(b"synthetic checkpoint")
        params = {"seed": row["seed"], "epochs_executed": 10, "selected_epoch": 10,
                  "epochs_requested": 10, "checkpoint_selection": "last_epoch", "batchnorm_policy": "adapt_all"}
        write_experiment_manifest(checkpoint.with_suffix(".manifest.json"),
            experiment="pair-contrastive-backbone-finetune", inputs=[source], outputs=[checkpoint],
            parameters=params, metrics={})
        result = out / f"{row['run_id']}.json"
        result.write_text("{}")
        result_manifest = write_experiment_manifest(result.with_suffix(".manifest.json"), experiment="evaluation",
            inputs=[source, checkpoint], outputs=[result], parameters={}, metrics={})
        eval_inputs.extend([checkpoint, result, result_manifest])
    completed = out / "summary.json"
    completed.write_text(json.dumps(summary))
    write_experiment_manifest(completed.with_suffix(".manifest.json"), experiment="summary",
        inputs=eval_inputs, outputs=[completed], parameters={}, metrics={})
    kwargs = dict(records=[file_record(source), file_record(preflight)], preflight=preflight,
                  snapshot=snapshot, input_paths=[source], parameters={"training_completed": False})
    return out, models, source, kwargs


def test_final_binding_requires_real_training_budgets_and_all_hashes(tmp_path):
    out, models, _, kwargs = campaign_fixture(tmp_path)
    assert finalize(out, models, **kwargs) == out / "summary.json"
    manifest = json.loads((out / "campaign-bound.manifest.json").read_text())
    assert manifest["parameters"]["training_completed"] is True
    assert manifest["metrics"]["publication_ready"] is False
    verify_records(manifest["inputs"] + manifest["outputs"])


@pytest.mark.parametrize("attack", ["input", "snapshot", "checkpoint", "result", "training_budget"])
def test_final_binding_refuses_changed_inputs_or_partial_outputs(tmp_path, attack):
    out, models, source, kwargs = campaign_fixture(tmp_path)
    if attack == "input":
        source.write_bytes(b"changed source")
    elif attack == "snapshot":
        kwargs["snapshot"].write_bytes(b"tampered")
    elif attack == "checkpoint":
        next(models.glob("*.pt")).write_bytes(b"changed weights")
    elif attack == "result":
        (out / "low_42.json").write_text('{"different":true}')
    elif attack == "training_budget":
        path = next(models.glob("*.manifest.json"))
        payload = json.loads(path.read_text())
        payload["parameters"]["epochs_executed"] = 8
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        finalize(out, models, **kwargs)
    assert not (out / "campaign-bound.manifest.json").exists()
    assert json.loads((out / "campaign-status.json").read_text())["status"].startswith("failed")


def test_missing_or_modified_declared_file_refused(tmp_path):
    source = tmp_path / "data"
    source.write_bytes(b"data")
    records = [file_record(source)]
    source.unlink()
    with pytest.raises(ValueError):
        verify_records(records)


def test_late_input_mutation_removes_only_new_binding(tmp_path, monkeypatch):
    import scripts.run_restricted_matched_campaign as runner

    out, models, source, kwargs = campaign_fixture(tmp_path)
    writer = runner.write_experiment_manifest
    def mutate_after_binding(*args, **options):
        result = writer(*args, **options)
        source.write_bytes(b"changed during final binding")
        return result
    monkeypatch.setattr(runner, "write_experiment_manifest", mutate_after_binding)
    with pytest.raises(ValueError, match="changed"):
        finalize(out, models, **kwargs)
    assert not (out / "campaign-bound.manifest.json").exists()
    assert (out / "summary.json").is_file()
    assert len(list(models.glob("*.pt"))) == 6
