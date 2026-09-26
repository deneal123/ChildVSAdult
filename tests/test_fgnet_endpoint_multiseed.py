"""Checks for the aggregate of three matched FG-NET evaluations."""

from __future__ import annotations

import json

import pytest

from age_gap.common.manifest import sha256_file
from scripts import summarize_fgnet_endpoint_multiseed as aggregate


def _fixture(tmp_path, *, corrupt_seed: int | None = None):
    paths = {}
    for seed, tuned in ((42, 0.84), (1, 0.85), (2, 0.86)):
        checkpoint = tmp_path / "models" / f"bb_facenet_seed{seed}.pt"
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        path = tmp_path / "metrics" / f"seed{seed}.json"
        path.parent.mkdir(exist_ok=True)
        row = {
            "protocol": "endpoint_age_matched",
            "endpoint_age_tolerance": 2,
            "negative_seed": 42,
            "n_bootstrap": 100,
        }
        for stratum in aggregate.STRATA:
            row[stratum] = {
                "class_counts": {"positive": 2, "negative": 2},
                "subject_bootstrap": {
                    "frozen_auc": 0.8,
                    "tuned_auc": tuned,
                    "delta_auc": tuned - 0.8,
                    "delta_ci95": [0.01, 0.09],
                },
            }
        path.write_text(json.dumps(row), encoding="utf-8")
        manifest = {
            "experiment": "fgnet-endpoint-age-matched-subject-statistics",
            "inputs": [
                {
                    "path": f"models/{checkpoint.name}",
                    "sha256": sha256_file(checkpoint),
                }
            ],
            "outputs": [
                {
                    "path": f"metrics/{path.name}",
                    "sha256": sha256_file(path) if seed != corrupt_seed else "invalid",
                }
            ],
        }
        path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        paths[seed] = path
    return paths


def test_three_seed_summary_checks_protocol_and_uses_sample_std(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate, "PROJECT_ROOT", tmp_path)
    result = aggregate.summarize(_fixture(tmp_path))
    large = result["strata"]["large_gap_25plus"]
    assert large["tuned_auc_mean"] == pytest.approx(0.85)
    assert large["delta_auc_std_across_seeds"] == pytest.approx(0.01)
    assert large["all_per_seed_delta_ci95_above_zero"]
    assert result["checkpoint_training_provenance"].startswith("legacy checkpoints")


def test_three_seed_summary_rejects_stale_output_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="output checksum mismatch"):
        aggregate.summarize(_fixture(tmp_path, corrupt_seed=1))
