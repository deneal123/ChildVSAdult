from __future__ import annotations

import json

import numpy as np

from scripts import reevaluate_strong_backbone_fgnet as reevaluation


def _write_run(path, run_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "seed": 42,
                "negative_type": "random",
                "scope": "head",
                "learning_rate": 1e-6,
            }
        ),
        encoding="utf-8",
    )


def test_discovery_only_returns_runs_with_matching_completed_checkpoints(tmp_path) -> None:
    result_dir = tmp_path / "metrics"
    models_dir = tmp_path / "models"
    result_dir.mkdir()
    models_dir.mkdir()
    _write_run(result_dir / "run_complete.json", "complete")
    _write_run(result_dir / "run_incomplete.json", "incomplete")
    (models_dir / "complete.pt").write_bytes(b"checkpoint")

    found = reevaluation.discover_completed_checkpoints(result_dir, models_dir)

    assert [row["run_id"] for row in found] == ["complete"]
    assert found[0]["checkpoint"].name == "complete.pt"


def test_loader_always_selects_corrected_protocol_explicitly(monkeypatch, tmp_path) -> None:
    seen = {}
    labels = np.asarray([1, 0], dtype=np.int64)
    metadata = {"protocol": "endpoint_age_matched", "stratum_age_gap": np.asarray([30, 30])}

    def fake_load_pairs(cache, **kwargs):
        seen["cache"] = cache
        seen.update(kwargs)
        return [np.zeros((2, 2, 3), dtype=np.uint8)] * 2, [
            np.zeros((2, 2, 3), dtype=np.uint8)
        ] * 2, labels, np.asarray([30, 30]), metadata

    monkeypatch.setattr(reevaluation, "load_pairs", fake_load_pairs)

    result = reevaluation.load_matched_fgnet_pairs(
        tmp_path / "cache.npz", seed=7, endpoint_age_tolerance=0
    )

    assert seen["protocol"] == "endpoint_age_matched"
    assert seen["seed"] == 7
    assert seen["endpoint_age_tolerance"] == 0
    assert seen["return_metadata"] is True
    assert result[3]["protocol"] == "endpoint_age_matched"


def test_metric_groups_use_source_gap_mask_and_require_balance() -> None:
    labels = np.tile(np.asarray([1, 0], dtype=np.int64), 10)
    scores = np.linspace(0.1, 0.9, 20)
    stratum_gap = np.asarray([30] * 10 + [10] * 10)

    metrics = reevaluation._metric_groups(scores, labels, stratum_gap)

    assert metrics["large_gap_25_plus"]["n_pairs"] == 10.0
    assert metrics["large_gap_25_plus"]["n_pos"] == 5.0
    assert metrics["large_gap_25_plus"]["n_neg"] == 5.0
