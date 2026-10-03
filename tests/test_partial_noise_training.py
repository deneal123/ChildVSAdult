import json

import pytest

from age_gap.common.manifest import write_experiment_manifest
from scripts.train_partial_noise import checked_manifest, verify_training, worker


def training_fixture(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    noisy = tmp_path / "noise.jsonl"
    noisy.write_text("{}")
    for seed in (42, 1, 2):
        checkpoint = models / f"partial_noise_s{seed}.pt"
        checkpoint.write_bytes(b"synthetic weights")
        params = {"seed": seed, "epochs_requested": 10, "epochs_executed": 10, "selected_epoch": 10,
                  "checkpoint_selection": "last_epoch", "batchnorm_policy": "adapt_all",
                  "backbone": "facenet", "trainable_scope": "head", "batch_size": 64,
                  "learning_rate": 3e-5, "margin": .3, "gap_weight": 0.0, "crops_dir": "faces",
                  "pairs_file": str(noisy)}
        write_experiment_manifest(checkpoint.with_suffix(".manifest.json"),
            experiment="pair-contrastive-backbone-finetune", parameters=params, metrics={},
            inputs=[noisy], outputs=[checkpoint])
    return models, noisy


def test_three_fixed_budget_training_manifests_required(tmp_path):
    models, noisy = training_fixture(tmp_path)
    assert len(verify_training(models, noisy)) == 3


@pytest.mark.parametrize("attack", ["seed", "selected", "executed", "bestval", "bn", "lr", "file",
                                   "checksum", "missing", "extra", "input"])
def test_stale_or_incomplete_training_not_accepted(tmp_path, attack):
    models, noisy = training_fixture(tmp_path)
    path = models / "partial_noise_s42.manifest.json"
    payload = json.loads(path.read_text())
    if attack in {"seed", "selected", "executed", "bestval", "bn", "lr", "file"}:
        key, value = {"seed": ("seed", 2), "selected": ("selected_epoch", 1),
                      "executed": ("epochs_executed", 8), "bestval": ("checkpoint_selection", "best_val"),
                      "bn": ("batchnorm_policy", "frozen_all"), "lr": ("learning_rate", .1),
                      "file": ("pairs_file", str(tmp_path / "other.jsonl"))}[attack]
        payload["parameters"][key] = value
        path.write_text(json.dumps(payload))
    elif attack == "checksum":
        (models / "partial_noise_s42.pt").write_bytes(b"changed weights")
    elif attack == "missing":
        path.unlink()
    elif attack == "extra":
        (models / "extra.manifest.json").write_text("{}")
    elif attack == "input":
        noisy.write_text('{"modified":true}')
    with pytest.raises(ValueError):
        verify_training(models, noisy)


def test_manifest_experiment_binding(tmp_path):
    models, _ = training_fixture(tmp_path)
    with pytest.raises(ValueError, match="unexpected"):
        checked_manifest(models / "partial_noise_s42.manifest.json", "wrong")


def test_worker_refuses_gpu_environment_before_training(tmp_path, monkeypatch):
    import torch

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CPU-only"):
        worker(tmp_path / "noise.jsonl", tmp_path / "models", tmp_path / "out")
    assert not (tmp_path / "models").exists()
