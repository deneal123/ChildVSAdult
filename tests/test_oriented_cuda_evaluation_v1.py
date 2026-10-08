import json
from pathlib import Path

import pytest

from scripts import evaluate_oriented_cuda_v1 as evaluator


def campaign():
    return dict(
        parameters=evaluator.PROTOCOL | dict(device="cuda"),
        metrics=dict(completed_cells=6, training_complete=True),
        outputs=[
            dict(path=f"{arm}_{seed}.json") for arm in ("low", "cross") for seed in evaluator.SEEDS
        ],
    )


def install(monkeypatch, native):
    def verified(path, experiment):
        if experiment == "oriented-uniform-cuda-serial-training":
            return native
        if experiment == "pair-contrastive-backbone-finetune":
            checkpoint = str(path).replace(".manifest.json", ".pt")
            return dict(outputs=[dict(path=checkpoint)])
        arm, seed = Path(path).stem.split("_")
        return dict(
            parameters=dict(arm=arm, seed=int(seed)),
            outputs=[
                dict(path=f"{arm}_{seed}.pt"),
                dict(path=str(evaluator.PROJECT_ROOT / f"{arm}_{seed}.manifest.json")),
            ],
        )

    monkeypatch.setattr(evaluator, "verified", verified)
    monkeypatch.setattr(evaluator, "check_cell", lambda *args: None)
    monkeypatch.setattr(evaluator, "file_record", lambda path: dict(path=str(path)))
    monkeypatch.setattr(evaluator, "training_contract", lambda *args: None)


def test_six_checkpoint_mapping(monkeypatch):
    install(monkeypatch, campaign())
    _, selected = evaluator.readiness(Path("campaign"))
    assert len(selected) == 6
    assert set(selected) == {f"{a}_s{s}" for a in ("low", "cross") for s in evaluator.SEEDS}


def test_missing_and_duplicate_rejected(monkeypatch):
    native = campaign()
    native["outputs"].pop()
    install(monkeypatch, native)
    with pytest.raises(ValueError, match="missing"):
        evaluator.readiness(Path("campaign"))
    native["outputs"].append(native["outputs"][0])
    with pytest.raises(ValueError, match="duplicate"):
        evaluator.readiness(Path("campaign"))


def test_cpu_campaign_rejected(monkeypatch):
    native = campaign()
    native["parameters"]["device"] = "cpu"
    install(monkeypatch, native)
    with pytest.raises(ValueError, match="protocol"):
        evaluator.readiness(Path("campaign"))


def test_changed_manifest_inputs_not_published(tmp_path):
    target = tmp_path / "summary.manifest.json"
    target.write_text(json.dumps(dict(inputs=[dict(sha256="changed")])), encoding="utf-8")
    with pytest.raises(RuntimeError, match="publication"):
        evaluator.validate_written_inputs(target, [dict(sha256="original")])
    assert not target.exists()


def test_unchanged_manifest_inputs_retained(tmp_path):
    target = tmp_path / "summary.manifest.json"
    records = [dict(sha256="original")]
    target.write_text(json.dumps(dict(inputs=records)), encoding="utf-8")
    evaluator.validate_written_inputs(target, records)
    assert target.exists()
