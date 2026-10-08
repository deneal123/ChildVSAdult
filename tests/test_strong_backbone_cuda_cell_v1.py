from pathlib import Path

import pytest

from scripts import run_strong_backbone_cuda_cell_v1 as runner


def fixture(monkeypatch):
    pairs, checkpoint = Path("pairs.jsonl"), Path("checkpoint.pt")
    monkeypatch.setattr(runner, "file_record", lambda p: dict(path=str(p)))
    component = dict(
        parameters=dict(
            backbone="adaface_ir101",
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
            pairs_file=str(pairs),
        ),
        inputs=[dict(path=str(pairs))],
        outputs=[dict(path=str(checkpoint))],
        metrics=dict(
            history=[
                dict(
                    epoch=i,
                    training_loss=0.3,
                    validation_auc=0.8,
                    mean_gradient_norm=0.1,
                    epoch_seconds=30.0,
                )
                for i in range(1, 9)
            ]
        ),
    )
    return component, pairs, checkpoint


def test_completed_native_contract(monkeypatch):
    component, pairs, checkpoint = fixture(monkeypatch)
    runner.contract(component, pairs, checkpoint, scope="head", lr=1e-6, seed=42)


@pytest.mark.parametrize(
    "change", ["epochs", "selection", "bn", "seed", "link", "history", "finite"]
)
def test_incomplete_or_incompatible_training_rejected(monkeypatch, change):
    component, pairs, checkpoint = fixture(monkeypatch)
    if change == "epochs":
        component["parameters"]["epochs_executed"] = 7
    elif change == "selection":
        component["parameters"]["checkpoint_selection"] = "best_val"
    elif change == "bn":
        component["parameters"]["batchnorm_policy"] = "adapt_all"
    elif change == "seed":
        component["parameters"]["seed"] = 1
    elif change == "link":
        component["outputs"] = []
    elif change == "history":
        component["metrics"]["history"].pop()
    else:
        component["metrics"]["history"][0]["mean_gradient_norm"] = float("nan")
    with pytest.raises(ValueError):
        runner.contract(component, pairs, checkpoint, scope="head", lr=1e-6, seed=42)
