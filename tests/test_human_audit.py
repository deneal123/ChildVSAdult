from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from age_gap.evaluation.human_audit import binary_metrics, cohens_kappa, percentile_ci
from scripts.build_blinded_audit import (
    _assert_no_frozen_responses,
    _materialize_opaque_images,
    _sample_balanced_disjoint,
    _sample_minimum_overlap,
)
from scripts.score_blinded_audit import _source_components, score


def test_cohens_kappa_perfect_and_known_case() -> None:
    assert cohens_kappa(["a", "b"], ["a", "b"]) == pytest.approx(1.0)
    assert cohens_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == pytest.approx(0.5)


def test_binary_metrics() -> None:
    metrics = binary_metrics([True, True, False, False], [True, False, True, False])
    assert metrics == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_source_components_join_tasks_sharing_an_image() -> None:
    key = {
        "a": {"source_images": ["one.jpg", "shared.jpg"]},
        "b": {"source_images": ["shared.jpg", "two.jpg"]},
        "c": {"source_images": ["three.jpg"]},
    }

    components = {frozenset(component) for component in _source_components(list(key), key)}

    assert components == {frozenset({"a", "b"}), frozenset({"c"})}


def test_percentile_ci_is_deterministic_and_contains_mean() -> None:
    lo, hi = percentile_ci([0.0, 1.0, 1.0, 0.0], lambda x: x.mean(), iterations=200)
    assert 0 <= lo <= 0.5 <= hi <= 1


def test_age_audit_scores_multiple_frozen_systems_and_uncertain_gold(tmp_path) -> None:
    def write(name, rows):
        path = tmp_path / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    a = write("a.jsonl", [{"task_id": "x", "response": 10}, {"task_id": "y", "response": "uncertain"}])
    b = write("b.jsonl", [{"task_id": "x", "response": 10}, {"task_id": "y", "response": "uncertain"}])
    gold = write("gold.jsonl", [{"task_id": "x", "response": 10}, {"task_id": "y", "response": "uncertain"}])
    key = write(
        "key.jsonl",
        [
            {"task_id": "x", "audit_type": "age_extraction", "auto_age": 11},
            {"task_id": "y", "audit_type": "age_extraction", "auto_age": 20},
        ],
    )
    baselines = write(
        "baselines.jsonl",
        [
            {"task_id": "x", "regex_age": None, "local_age": 10},
            {"task_id": "y", "regex_age": 20, "local_age": 19},
        ],
    )

    result = score(a, b, gold, key, baselines)["audit"]["age_extraction"]

    assert result["n_numeric_gold"] == 1
    assert result["n_uncertain_gold"] == 1
    assert result["systems"]["gigachat"]["mae"] == 1
    assert result["systems"]["regex"]["n"] == 0
    assert result["systems"]["local"]["mae"] == 0


def test_audit_pack_materializes_opaque_image_paths(tmp_path: Path) -> None:
    source = tmp_path / "private-corpus-id.jpg"
    source.write_bytes(b"image-bytes")
    output = tmp_path / "pack"
    output.mkdir()
    tasks = [{"task_id": "age-opaque", "images": [str(source)], "response": None}]
    key = [{"task_id": "age-opaque", "audit_type": "age_extraction"}]

    manifest = _materialize_opaque_images(tasks, key, output, seed=7)

    assert tasks[0]["images"][0].startswith("images/")
    assert "private-corpus-id" not in tasks[0]["images"][0]
    assert (output / tasks[0]["images"][0]).read_bytes() == b"image-bytes"
    assert key[0]["source_images"] == [str(source)]
    assert json.loads(manifest.read_text(encoding="utf-8"))["n_files"] == 1


def test_audit_pack_refuses_to_overwrite_started_responses(tmp_path: Path) -> None:
    response = tmp_path / "annotator_a.jsonl"
    response.write_text(json.dumps({"task_id": "x", "response": "uncertain"}) + "\n")

    with pytest.raises(RuntimeError, match="refusing to overwrite frozen response"):
        _assert_no_frozen_responses([response])


def test_minimum_overlap_sampler_preserves_boundary_order() -> None:
    rows = [
        {"evidence_id": "closest", "images": ["a", "b"], "similarity": 0.9701},
        {"evidence_id": "next", "images": ["c", "d"], "similarity": 0.9698},
        {"evidence_id": "far", "images": ["e", "f"], "similarity": 0.91},
    ]

    selected = _sample_minimum_overlap(rows, 2, set())

    assert [row["evidence_id"] for row in selected] == ["closest", "next"]


def test_balanced_sampler_covers_both_automatic_labels() -> None:
    rows = [
        {
            "evidence_id": f"{label}-{index}",
            "images": [f"{label}-{index}.jpg"],
            "auto_label": label,
        }
        for label in ("same_identity", "different_identity")
        for index in range(4)
    ]

    selected = _sample_balanced_disjoint(rows, 6, np.random.default_rng(7), set())

    assert Counter(row["auto_label"] for row in selected) == {
        "same_identity": 3,
        "different_identity": 3,
    }


def test_human_audit_rejects_null_responses(tmp_path: Path) -> None:
    def write(name: str, response) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps({"task_id": "x", "response": response}) + "\n")
        return path

    a = write("a.jsonl", None)
    b = write("b.jsonl", "same_identity")
    gold = write("gold.jsonl", "same_identity")
    key = tmp_path / "key.jsonl"
    key.write_text(
        json.dumps(
            {
                "task_id": "x",
                "audit_type": "positive_identity",
                "auto_label": "same_identity",
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="null response"):
        score(a, b, gold, key)


def test_binary_audit_metrics_compare_automatic_label_to_gold(tmp_path: Path) -> None:
    def write(name: str, rows: list[dict]) -> Path:
        path = tmp_path / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    responses = [
        {"task_id": "x", "response": "same_identity"},
        {"task_id": "y", "response": "different_identity"},
        {"task_id": "z", "response": "different_identity"},
        {"task_id": "w", "response": "same_identity"},
    ]
    a = write("a.jsonl", responses)
    b = write("b.jsonl", responses)
    gold = write("gold.jsonl", responses)
    key = write(
        "key.jsonl",
        [
            {"task_id": "x", "audit_type": "positive_identity", "auto_label": "same_identity"},
            {"task_id": "y", "audit_type": "positive_identity", "auto_label": "same_identity"},
            {
                "task_id": "z",
                "audit_type": "positive_identity",
                "auto_label": "different_identity",
            },
            {
                "task_id": "w",
                "audit_type": "positive_identity",
                "auto_label": "different_identity",
            },
        ],
    )

    result = score(a, b, gold, key)["audit"]["positive_identity"]

    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["accuracy"] == pytest.approx(0.5)
    assert len(result["cohens_kappa_95ci"]) == 2
    assert len(result["precision_95ci"]) == 2
    assert len(result["recall_95ci"]) == 2
    assert len(result["f1_95ci"]) == 2


def test_positive_only_audit_reports_ppv_not_artificial_recall(tmp_path: Path) -> None:
    def write(name: str, rows: list[dict]) -> Path:
        path = tmp_path / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    responses = [
        {"task_id": "x", "response": "same_identity"},
        {"task_id": "y", "response": "different_identity"},
    ]
    a = write("a.jsonl", responses)
    b = write("b.jsonl", responses)
    gold = write("gold.jsonl", responses)
    key = write(
        "key.jsonl",
        [
            {"task_id": task_id, "audit_type": "positive_identity", "auto_label": "same_identity"}
            for task_id in ("x", "y")
        ],
    )

    result = score(a, b, gold, key)["audit"]["positive_identity"]

    assert result["positive_predictive_value"] == pytest.approx(0.5)
    assert "recall" not in result
    assert "f1" not in result
    assert result["sampling_design"].startswith("automatic-positive audit")
