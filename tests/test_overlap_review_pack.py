"""Synthetic, offline tests for build_overlap_review_pack.

No real crops, benchmarks, network or GPU are touched: every input is a tiny
synthetic fixture under ``tmp_path``. The tests pin the properties that make the pack
safe and useful: per-reviewer blinding with isolated directories, full-keep high-risk
strata, deterministic lower-band sampling with explicit inclusion probabilities,
endpoint reuse across strata, determinism, independent reviewer orders, blank
decisions, and explicit blockers when an image cannot be rebuilt.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import build_overlap_review_pack as borp

FORBIDDEN_REVIEWER_KEYS = {
    "cosine_similarity",
    "phash_distance",
    "rank",
    "match_type",
    "benchmark",
    "train_index",
    "benchmark_index",
    "train_face_id",
    "auto_label",
    "stratum",
    "candidate_source",
    "source_images",
}


def _crop(path: Path, value: int) -> np.ndarray:
    image = np.full((112, 112, 3), value, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return image


def _benchmark_npz(path: Path, images: list[np.ndarray], *, fgnet: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fgnet:
        np.savez_compressed(path, crops=np.stack(images))
    else:
        half = len(images) // 2
        np.savez_compressed(path, a=np.stack(images[:half]), b=np.stack(images[half:]))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _fixture(
    tmp_path: Path,
    *,
    bad_benchmark_index: bool = False,
    wrong_pixel_hash: bool = False,
    reuses_endpoint: bool = False,
) -> tuple[Path, Path]:
    audit_dir = tmp_path / "audit"
    external = tmp_path / "external"
    audit_dir.mkdir(parents=True, exist_ok=True)

    n_crops = 10
    inventory: list[dict] = []
    for index in range(n_crops):
        crop_path = audit_dir / "faces" / f"face-{index}.jpg"
        crop = _crop(crop_path, 10 * (index + 1))
        inventory.append(
            {
                "train_index": index,
                "face_id": f"face-{index}",
                "path": str(crop_path),
                "pixel_sha256": "bad" if (wrong_pixel_hash and index == 0) else borp.pixel_sha256(crop),
                "phash_hex": "0" * 16,
            }
        )
    bench = [np.full((112, 112, 3), 10 * (index + 1) + 1, dtype=np.uint8) for index in range(n_crops)]
    _benchmark_npz(external / "fgnet_crops.npz", bench, fgnet=True)
    _write_jsonl(audit_dir / "private_training_inventory.jsonl", inventory)

    _write_jsonl(
        audit_dir / "private_candidates.jsonl",
        [
            {
                "benchmark": "FG-NET",
                "match_type": "exact_decoded_pixels",
                "train_index": 0,
                "benchmark_index": 0,
                "phash_distance": 0,
            },
            {
                "benchmark": "FG-NET",
                "match_type": "phash_candidate",
                "train_index": 1,
                "benchmark_index": 1,
                "phash_distance": 2,
            },
        ],
    )

    high_benchmark_index = 999 if bad_benchmark_index else 2
    embedding_rows = [
        {
            "benchmark": "FG-NET",
            "train_index": 2,
            "benchmark_index": high_benchmark_index,
            "cosine_similarity": 0.85,
            "rank": 1,
            "match_type": "top_k",
        },
        {
            "benchmark": "FG-NET",
            "train_index": 3,
            "benchmark_index": 3,
            "cosine_similarity": 0.75,
            "rank": 2,
            "match_type": "cosine_threshold",
        },
        {
            "benchmark": "FG-NET",
            "train_index": 4,
            "benchmark_index": 4,
            "cosine_similarity": 0.65,
            "rank": -1,
            "match_type": "cosine_threshold",
        },
        {
            "benchmark": "FG-NET",
            "train_index": 5,
            "benchmark_index": 5,
            "cosine_similarity": 0.50,
            "rank": 3,
            "match_type": "top_k",
        },
    ]
    if reuses_endpoint:
        # Same train crop as the pHash candidate, cosine just under the high-risk line.
        embedding_rows.append(
            {
                "benchmark": "FG-NET",
                "train_index": 1,
                "benchmark_index": 6,
                "cosine_similarity": 0.75,
                "rank": 4,
                "match_type": "top_k",
            }
        )
    _write_jsonl(audit_dir / "private_embedding_candidates.jsonl", embedding_rows)
    return audit_dir, external


def _build(tmp_path: Path, name: str = "pack", **kwargs):
    audit_dir, external = _fixture(tmp_path, **kwargs)
    output = tmp_path / name
    result = borp.build_pack(output, audit_dir=audit_dir, external_dir=external, seed=7)
    return output, result


def _reviewer_rows(output: Path, reviewer: str) -> list[dict]:
    path = output / reviewer / "tasks.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_reviewer_dirs_are_isolated_and_blinded(tmp_path: Path) -> None:
    output, _ = _build(tmp_path)

    for reviewer in borp.REVIEWERS:
        entries = {item.name for item in (output / reviewer).iterdir()}
        assert entries == {"tasks.jsonl", "README.md", "images"}
        # No private artefact leaks into a reviewer directory.
        assert not (output / reviewer / "key.jsonl").exists()
        assert not (output / reviewer / "sampling_manifest.json").exists()

        images = list((output / reviewer / "images").glob("*.jpg"))
        assert images
        for image in images:
            assert image.name[:24].isalnum()

        for row in _reviewer_rows(output, reviewer):
            assert row["audit_type"] == "overlap_candidate"
            assert row["response"] is None and row["notes"] is None
            assert row["allowed"] == borp.ALLOWED
            assert set(row) == {"task_id", "audit_type", "images", "allowed", "response", "notes"}
            assert FORBIDDEN_REVIEWER_KEYS.isdisjoint(row)
            assert len(row["images"]) == 2
            for image in row["images"]:
                assert image.startswith("images/")
            assert "cosine_similarity" not in json.dumps(row)

    key = borp._read_jsonl(output / borp.PRIVATE_DIRNAME / "key.jsonl")
    assert any(row["cosine_similarity"] is not None for row in key)
    assert all(row["review_status"] == "unreviewed_candidate" for row in key)


def test_high_risk_strata_are_kept_fully_and_lower_bands_sampled(tmp_path: Path) -> None:
    output, result = _build(tmp_path)

    assert result["inclusion_probability"][borp.EXACT_STRATUM] == 1.0
    assert result["inclusion_probability"][borp.PHASH_STRATUM] == 1.0
    assert result["inclusion_probability"]["cosine_ge_0.80"] == 1.0
    assert result["strata"][borp.EXACT_STRATUM] == 1
    assert result["strata"][borp.PHASH_STRATUM] == 1
    assert result["strata"]["cosine_ge_0.80"] == 1
    assert result["strata"]["cosine_0.70_0.80"] == 1
    assert result["strata"]["cosine_0.60_0.70"] == 1
    assert result["strata"]["cosine_lt_0.60"] == 1

    sampling = json.loads(
        (output / borp.PRIVATE_DIRNAME / "sampling_manifest.json").read_text(encoding="utf-8")
    )
    assert sampling["high_risk_cosine_threshold"] == 0.8
    assert sampling["strata"]["cosine_ge_0.80"]["full_keep"] is True
    assert sampling["endpoint_reuse_policy"].startswith("never excluded")


def test_lower_band_inclusion_probability_is_deterministic_and_uniform(tmp_path: Path) -> None:
    audit_dir, external = _fixture(tmp_path)
    # 20 low-cosine candidates, distinct endpoints, target 20 keeps all => prob 1.0.
    rows = [
        {
            "benchmark": "FG-NET",
            "train_index": index,
            "benchmark_index": index,
            "cosine_similarity": 0.55,
            "rank": index,
            "match_type": "cosine_threshold",
        }
        for index in range(6, 20)
    ]
    _write_jsonl(audit_dir / "private_embedding_candidates.jsonl", rows)
    strata = borp._stratify(borp._embedding_candidates(audit_dir))
    picked, probability = borp._select_stratum("cosine_lt_0.60", strata["cosine_lt_0.60"], 7)
    assert probability == 1.0
    assert len(picked) == 14

    # Shrink the target to check the n/N inclusion probability and determinism.
    original = borp.STRATUM_TARGETS["cosine_lt_0.60"]
    borp.STRATUM_TARGETS["cosine_lt_0.60"] = 5
    try:
        picked, probability = borp._select_stratum("cosine_lt_0.60", strata["cosine_lt_0.60"], 7)
        again, _ = borp._select_stratum("cosine_lt_0.60", strata["cosine_lt_0.60"], 7)
    finally:
        borp.STRATUM_TARGETS["cosine_lt_0.60"] = original
    assert len(picked) == 5
    assert probability == pytest.approx(5 / 14, abs=1e-8)
    # Same seed => same sample.
    assert [row["evidence_id"] for row in picked] == [row["evidence_id"] for row in again]


def test_endpoint_reuse_across_strata_is_not_excluded(tmp_path: Path) -> None:
    output, result = _build(tmp_path, reuses_endpoint=True)

    key = borp._read_jsonl(output / borp.PRIVATE_DIRNAME / "key.jsonl")
    train_one = [row for row in key if row["train_index"] == 1]
    assert {row["stratum"] for row in train_one} == {borp.PHASH_STRATUM, "cosine_0.70_0.80"}
    assert result["n_tasks"] == 7  # 2 image + 1 high + 2 mid (incl. reused endpoint) + 2 lower


def test_pack_is_deterministic_for_the_same_seed(tmp_path: Path) -> None:
    first, first_result = _build(tmp_path, "pack_one")
    second, second_result = _build(tmp_path, "pack_two")

    assert first_result["n_tasks"] == second_result["n_tasks"]
    for reviewer in borp.REVIEWERS:
        assert (first / reviewer / "tasks.jsonl").read_text(encoding="utf-8") == (
            second / reviewer / "tasks.jsonl"
        ).read_text(encoding="utf-8")
    assert (first / borp.PRIVATE_DIRNAME / "key.jsonl").read_text(encoding="utf-8") == (
        second / borp.PRIVATE_DIRNAME / "key.jsonl"
    ).read_text(encoding="utf-8")


def test_reviewers_share_task_ids_in_independent_orders(tmp_path: Path) -> None:
    output, _ = _build(tmp_path)

    a = [row["task_id"] for row in _reviewer_rows(output, "reviewer_a")]
    b = [row["task_id"] for row in _reviewer_rows(output, "reviewer_b")]
    assert sorted(a) == sorted(b)
    assert len(set(a)) == len(a)
    assert a != b


def test_unresolvable_benchmark_index_becomes_an_explicit_blocker(tmp_path: Path) -> None:
    output, result = _build(tmp_path, bad_benchmark_index=True)

    assert result["blockers"]["benchmark_index_unresolvable"] == {"FG-NET": 1}
    blockers = json.loads((output / "blockers.json").read_text(encoding="utf-8"))
    assert blockers["status"] == "blocked"
    key = borp._read_jsonl(output / borp.PRIVATE_DIRNAME / "key.jsonl")
    assert all(row["benchmark_index"] != 999 for row in key)


def test_pixel_hash_mismatch_is_reported_not_silently_dropped(tmp_path: Path) -> None:
    output, result = _build(tmp_path, wrong_pixel_hash=True)

    assert result["blockers"]["unresolved_selected_candidates"] >= 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["reviewer_decisions_status"] == "pending"
    assert summary["adjudication_status"] == "pending"
    assert summary["human_identity_judgements"] == 0
    assert summary["confirmed_identity_overlaps"] is None


def test_refuses_to_overwrite_frozen_reviewer_responses(tmp_path: Path) -> None:
    output, _ = _build(tmp_path)

    reviewer_path = output / "reviewer_a" / "tasks.jsonl"
    rows = [json.loads(line) for line in reviewer_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["response"] = "same_identity"
    reviewer_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    audit_dir, external = _fixture(tmp_path / "second")
    with pytest.raises(RuntimeError, match="frozen response"):
        borp.build_pack(output, audit_dir=audit_dir, external_dir=external, seed=7)


def test_decision_and_adjudication_schemas_are_separate_and_blank(tmp_path: Path) -> None:
    output, _ = _build(tmp_path)

    decisions = json.loads((output / "decisions.schema.json").read_text(encoding="utf-8"))
    gold = json.loads((output / "adjudicated_gold.schema.json").read_text(encoding="utf-8"))

    assert decisions["properties"]["response"]["enum"] == [
        "same_identity",
        "different_identity",
        "uncertain",
        None,
    ]
    assert "adjudication_notes" in gold["properties"]
    assert "adjudication_notes" not in decisions["properties"]


def test_cosine_stratum_boundaries_are_inclusive_at_the_high_risk_edge() -> None:
    assert borp._cosine_stratum(0.80) == "cosine_ge_0.80"
    assert borp._cosine_stratum(0.7999) == "cosine_0.70_0.80"
    assert borp._cosine_stratum(0.70) == "cosine_0.70_0.80"
    assert borp._cosine_stratum(0.0) == "cosine_lt_0.60"
