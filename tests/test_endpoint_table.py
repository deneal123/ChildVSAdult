from __future__ import annotations

import copy
import json

import pytest

from age_gap.common.manifest import write_experiment_manifest
from scripts.render_fgnet_endpoint_table import render_table, verified_payload


def _summary():
    row = {
        "frozen_auc": 0.8,
        "tuned_auc_mean": 0.85,
        "tuned_auc_std_across_seeds": 0.001,
        "delta_auc_mean": 0.05,
        "delta_auc_std_across_seeds": 0.001,
        "subject_bootstrap_by_seed": {
            str(seed): {"n_pairs": 430, "n_subjects": 75, "delta_ci95": [0.006, 0.09]}
            for seed in (42, 1, 2)
        },
    }
    return {
        "protocol": "endpoint_age_matched", "seeds": [42, 1, 2],
        "strata": {"large_gap_25plus": copy.deepcopy(row), "overall": copy.deepcopy(row)},
    }


def test_table_distinguishes_seed_variability_from_subject_ci():
    text = render_table(_summary())
    assert "0.8500" in text and "+0.0500" in text
    assert "not for the seed mean" in text
    assert "training manifests are absent" in text
    assert "overlap remains under review" in text


@pytest.mark.parametrize("change", ["protocol", "seeds", "subjects", "intervals"])
def test_refuses_noncomparable_or_incomplete_endpoint(change):
    summary = _summary()
    if change == "protocol":
        summary["protocol"] = "random"
    elif change == "seeds":
        summary["seeds"] = [42, 42, 1]
    elif change == "subjects":
        summary["strata"]["overall"]["subject_bootstrap_by_seed"]["1"]["n_subjects"] = 76
    else:
        del summary["strata"]["overall"]["subject_bootstrap_by_seed"]["1"]
    with pytest.raises(ValueError):
        render_table(summary)


def test_verified_payload_rejects_changed_input_and_output(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("original", encoding="utf-8")
    result = tmp_path / "summary.json"
    payload = _summary()
    result.write_text(json.dumps(payload), encoding="utf-8")
    write_experiment_manifest(
        result.with_suffix(".manifest.json"), experiment="test", parameters={},
        metrics=payload, inputs=[source], outputs=[result],
    )
    assert verified_payload(result, tmp_path) == payload
    source.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="input checksum"):
        verified_payload(result, tmp_path)
    source.write_text("original", encoding="utf-8")
    result.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="output checksum"):
        verified_payload(result, tmp_path)
