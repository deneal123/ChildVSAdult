import json

import pytest

from scripts.cross_source import _aggregate as aggregate_external
from scripts.eval_cacd_vs import _aggregate as aggregate_cacd
from scripts.eval_cross_platform import _aggregate as aggregate_pairs
from scripts.sota_common_protocol import _aggregate_completed_runs, _file_set_digest
from scripts.strong_backbone_study import _aggregate as aggregate_strong


def test_aggregate_pairs_excludes_frozen_and_reports_ci_width() -> None:
    rows = {
        "frozen": {"overall_auc": 0.70, "overall_ci95": [0.68, 0.72]},
        "seed1": {"overall_auc": 0.74, "overall_ci95": [0.72, 0.76], "n_pos": 10},
        "seed2": {"overall_auc": 0.76, "overall_ci95": [0.73, 0.77], "n_pos": 10},
    }

    result = aggregate_pairs(rows)

    assert result["overall_auc"]["mean"] == pytest.approx(0.75)
    assert result["overall_auc"]["std"] == pytest.approx(0.014142135623730963)
    assert result["overall_auc"]["n_seeds"] == 2
    assert result["overall_ci95_width"]["mean"] == pytest.approx(0.04)
    assert result["overall_ci95_width"]["max"] == pytest.approx(0.04)
    assert "n_pos" not in result


def test_aggregate_external_uses_only_common_metrics() -> None:
    rows = {
        "seed1": {"fgnet.large_gap": 0.80, "only_one": 1.0},
        "seed2": {"fgnet.large_gap": 0.84},
        "seed3": {"fgnet.large_gap": 0.82},
    }

    result = aggregate_external(rows)

    assert result["fgnet.large_gap"]["mean"] == 0.82
    assert result["fgnet.large_gap"]["n_seeds"] == 3
    assert "only_one" not in result


def test_aggregate_cacd_skips_pair_count_and_per_run_ci() -> None:
    rows = {
        "seed1": {"n_pairs": 4000, "roc_auc": 0.98, "roc_auc_ci95": [0.97, 0.99]},
        "seed2": {"n_pairs": 4000, "roc_auc": 0.99, "roc_auc_ci95": [0.98, 1.0]},
    }

    result = aggregate_cacd(rows)

    assert result["roc_auc"]["mean"] == pytest.approx(0.985)
    assert "n_pairs" not in result
    assert "roc_auc_ci95" not in result


def test_sota_summary_requires_and_aggregates_every_seed(tmp_path) -> None:
    for seed, auc in ((42, 0.80), (1, 0.82), (2, 0.84)):
        payload = {"run_id": f"mtlface_bb_e8_s{seed}", "metrics": {"auc": auc}}
        (tmp_path / f"mtlface_bb_e8_s{seed}.json").write_text(json.dumps(payload), encoding="utf-8")

    output = _aggregate_completed_runs(
        method="mtlface", seeds=[42, 1, 2], backbone="bb", epochs=8, results_dir=tmp_path
    )

    assert output is not None
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["metrics"]["auc"]["mean"] == pytest.approx(0.82)
    assert result["metrics"]["auc"]["n_seeds"] == 3
    assert output.with_suffix(".manifest.json").exists()


def test_file_set_digest_is_order_independent(tmp_path) -> None:
    left, right = tmp_path / "a.jpg", tmp_path / "b.jpg"
    left.write_bytes(b"a")
    right.write_bytes(b"bb")

    assert _file_set_digest([left, right]) == _file_set_digest([right, left])
    assert _file_set_digest([left, right])["bytes"] == 3


def test_strong_backbone_summary_has_manifest_and_sample_std(tmp_path) -> None:
    for seed, auc in ((1, 0.7), (2, 0.9)):
        row = {
            "negative_type": "random",
            "scope": "head",
            "learning_rate": 1e-6,
            "seed": seed,
            "epochs": 8,
            "metrics": {"auc": auc},
        }
        (tmp_path / f"run_{seed}.json").write_text(json.dumps(row), encoding="utf-8")

    output = aggregate_strong(tmp_path)

    result = json.loads(output.read_text(encoding="utf-8"))
    cell = result["cells"]["e8|random|head|1e-06"]["metrics"]["auc"]
    assert cell["mean"] == pytest.approx(0.8)
    assert cell["std"] == pytest.approx(0.1414213562)
    assert output.with_suffix(".manifest.json").exists()
