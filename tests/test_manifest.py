from __future__ import annotations

import json
from pathlib import Path

from age_gap.common.manifest import file_record, sha256_file, write_experiment_manifest


def test_sha256_and_file_record(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("evidence\n", encoding="utf-8")

    assert sha256_file(source) == "ffb9a8be5f26ace0e4edc54d3112c480a187104286279284363f5896767e6ae3"
    record = file_record(source)
    assert record["bytes"] == 10
    assert record["sha256"] == sha256_file(source)


def test_write_experiment_manifest(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    output = tmp_path / "model.pt"
    manifest = tmp_path / "model.manifest.json"
    source.write_text("{}\n", encoding="utf-8")
    output.write_bytes(b"checkpoint")

    result = write_experiment_manifest(
        manifest,
        experiment="unit-test",
        parameters={"seed": 42},
        metrics={"val_auc": 0.9},
        inputs=[source],
        outputs=[output],
        command=["python", "experiment.py"],
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["experiment"] == "unit-test"
    assert payload["parameters"] == {"seed": 42}
    assert payload["metrics"] == {"val_auc": 0.9}
    assert payload["inputs"][0]["sha256"] == sha256_file(source)
    assert payload["outputs"][0]["sha256"] == sha256_file(output)
