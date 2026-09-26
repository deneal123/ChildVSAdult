from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts import scaling_multiseed


class _FakeModel:
    def to(self, _device):
        return self

    def eval(self):
        return self


def test_scaling_sweep_uses_fraction_files_without_rewriting_canonical(
    tmp_path: Path, monkeypatch
) -> None:
    canonical = tmp_path / "data/processed/pairs.jsonl"
    canonical.parent.mkdir(parents=True)
    canonical.write_text('{"canonical": true}\n', encoding="utf-8")
    inventory = tmp_path / "metrics/model_inventory.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text('{"models": {}}\n', encoding="utf-8")
    original = canonical.read_bytes()
    trained_on: list[Path] = []

    def fake_data_path(kind: str, *parts: str) -> Path:
        roots = {
            "models_dir": tmp_path / "models",
            "metrics_dir": tmp_path / "metrics",
            "data_dir": tmp_path / "data",
        }
        return roots[kind].joinpath(*parts)

    def fake_split_run(**kwargs) -> None:
        assert Path(kwargs["pairs_file"]) == canonical
        assert Path(kwargs["pairs_out"]) != canonical
        pairs_out = Path(kwargs["pairs_out"])
        split_out = Path(kwargs["split_map_out"])
        pairs_out.parent.mkdir(parents=True, exist_ok=True)
        split_out.parent.mkdir(parents=True, exist_ok=True)
        pairs_out.write_text('{"fraction": 0.1}\n', encoding="utf-8")
        split_out.write_text('{"identity_group_id": "x", "split": "train"}\n', encoding="utf-8")

    def fake_finetune(**kwargs) -> Path:
        trained_on.append(Path(kwargs["pairs_file"]))
        checkpoint = Path(kwargs["ckpt_out"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return checkpoint

    monkeypatch.setattr(scaling_multiseed, "FRACS", [0.1])
    monkeypatch.setattr(scaling_multiseed, "SEEDS", [42])
    monkeypatch.setattr(scaling_multiseed, "data_path", fake_data_path)
    monkeypatch.setattr(scaling_multiseed, "torch_device", lambda: "cpu")
    monkeypatch.setattr(scaling_multiseed, "split_run", fake_split_run)
    monkeypatch.setattr(scaling_multiseed, "finetune", fake_finetune)
    monkeypatch.setattr(scaling_multiseed, "make_backbone", lambda *_args, **_kwargs: _FakeModel())
    monkeypatch.setattr(scaling_multiseed, "load_finetuned", lambda *_args, **_kwargs: _FakeModel())
    monkeypatch.setattr(
        scaling_multiseed,
        "eval_all",
        lambda *_args, **_kwargs: {"fgnet.large_gap": 0.8, "our.25+": 0.7},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    scaling_multiseed.main()

    assert canonical.read_bytes() == original
    assert trained_on == [tmp_path / "data/processed/experiments/pairs_scaling_f0p1.jsonl"]
    result = json.loads((tmp_path / "metrics/scaling_multiseed.json").read_text())
    assert result["schema_version"] == 2
    assert result["fracs"]["0.1"]["pairs_file"] == "pairs_scaling_f0p1.jsonl"
    assert (tmp_path / "metrics/scaling_multiseed.manifest.json").is_file()
