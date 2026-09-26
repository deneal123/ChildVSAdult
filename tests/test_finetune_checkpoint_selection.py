from __future__ import annotations

from pathlib import Path

import torch

from age_gap.training import finetune as finetune_module


class _TinyDataset(torch.utils.data.Dataset):
    labels = [1, 0]

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        x = torch.tensor([float(index + 1)], dtype=torch.float32)
        y = torch.tensor(float(index == 0), dtype=torch.float32)
        weight = torch.tensor(1.0, dtype=torch.float32)
        return x, x + 0.1, y, weight


class _TinyBackbone(torch.nn.Module):
    trainable_scopes = {"tail": ("weight", "bias")}

    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.net(x), dim=-1)


def _run(tmp_path: Path, monkeypatch, selection: str) -> dict:
    aucs = iter([0.9, 0.8, 0.7])
    monkeypatch.setattr(finetune_module, "torch_device", lambda: "cpu")
    monkeypatch.setattr(finetune_module, "make_backbone", lambda *_a, **_k: _TinyBackbone())
    monkeypatch.setattr(finetune_module, "ImagePairDataset", lambda **_k: _TinyDataset())
    monkeypatch.setattr(finetune_module, "_val_auc", lambda *_a, **_k: next(aucs))
    monkeypatch.setattr(finetune_module, "write_experiment_manifest", lambda *_a, **_k: None)
    checkpoint = tmp_path / f"{selection}.pt"
    finetune_module.finetune(
        epochs=3,
        lr=1e-2,
        batch_size=2,
        patience=1,
        ckpt_out=checkpoint,
        checkpoint_selection=selection,
        pairs_file=str(tmp_path / "pairs.jsonl"),
    )
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def test_last_epoch_selects_final_epoch_despite_early_auc_peak(tmp_path, monkeypatch) -> None:
    payload = _run(tmp_path, monkeypatch, "last_epoch")

    assert [row["validation_auc"] for row in payload["history"]] == [0.9, 0.8, 0.7]
    assert payload["selected_epoch"] == 3
    assert payload["checkpoint_selection"] == "last_epoch"


def test_best_validation_remains_default_selection(tmp_path, monkeypatch) -> None:
    payload = _run(tmp_path, monkeypatch, "best_val")

    assert len(payload["history"]) == 2  # patience=1 retains legacy early stopping
    assert payload["selected_epoch"] == 1
    assert payload["checkpoint_selection"] == "best_val"
