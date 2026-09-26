"""Validate blinding, balance and file integrity of the prepared human-audit pack."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from age_gap.common.io import data_path, read_jsonl
from age_gap.common.manifest import sha256_file, write_experiment_manifest

EXPECTED_TYPES = {
    "age_extraction",
    "group_integrity",
    "positive_identity",
    "cluster_merge",
    "near_duplicate",
}
OPAQUE_IMAGE = re.compile(r"^images/[0-9a-f]{24}\.[a-z0-9]+$")
SOURCE_LEAK = re.compile(r"(?:[A-Za-z]:\\|(?:vk_|reddit_)?-?\d{5,}_\d{5,})")
FORBIDDEN_FIELDS = {"auto_age", "auto_source", "auto_label", "similarity", "source_images"}


def _rows(path: Path) -> list[dict]:
    return list(read_jsonl(path))


def validate(base: Path, expected_per_type: int) -> dict:
    annotator_a = base / "annotator_a.jsonl"
    annotator_b = base / "annotator_b.jsonl"
    key_path = base / "audit_key.jsonl"
    adjudication_template_path = base / "adjudicated_gold.template.jsonl"
    images_manifest_path = base / "images.manifest.json"
    age_baselines_path = base / "age_baseline_predictions.jsonl"
    a_rows, b_rows, key_rows = _rows(annotator_a), _rows(annotator_b), _rows(key_path)
    adjudication_rows = _rows(adjudication_template_path)
    by_a = {row["task_id"]: row for row in a_rows}
    by_b = {row["task_id"]: row for row in b_rows}
    by_key = {row["task_id"]: row for row in key_rows}

    if len(by_a) != len(a_rows) or len(by_b) != len(b_rows) or len(by_key) != len(key_rows):
        raise RuntimeError("duplicate task IDs in audit pack")
    if set(by_a) != set(by_b) or set(by_a) != set(by_key):
        raise RuntimeError("annotator/key task-ID sets differ")
    if {row["task_id"] for row in adjudication_rows} != set(by_a):
        raise RuntimeError("adjudication-template task IDs differ")
    if any(row.get("response") is not None for row in adjudication_rows):
        raise RuntimeError("adjudication template contains completed responses")
    if [row["task_id"] for row in a_rows] == [row["task_id"] for row in b_rows]:
        raise RuntimeError("annotator task orders are not independently shuffled")

    age_task_ids = {
        row["task_id"] for row in a_rows if row["audit_type"] == "age_extraction"
    }
    age_baselines = _rows(age_baselines_path)
    by_age_baseline = {row["task_id"]: row for row in age_baselines}
    if len(by_age_baseline) != len(age_baselines) or set(by_age_baseline) != age_task_ids:
        raise RuntimeError("age-baseline predictions do not match the frozen age-audit tasks")
    age_baseline_coverage = {
        name: sum(row.get(field) is not None for row in age_baselines) / len(age_baselines)
        for name, field in (
            ("gigachat", "giga_age"),
            ("regex", "regex_age"),
            ("local", "local_age"),
        )
    }

    counts = Counter(row["audit_type"] for row in a_rows)
    if set(counts) != EXPECTED_TYPES or any(counts[kind] != expected_per_type for kind in counts):
        raise RuntimeError(f"unexpected audit strata: {dict(counts)}")
    if any(row.get("response") is not None for row in [*a_rows, *b_rows]):
        raise RuntimeError("audit pack already contains non-null responses")
    if any(FORBIDDEN_FIELDS.intersection(row) for row in [*a_rows, *b_rows]):
        raise RuntimeError("automatic decision or source mapping leaked to annotator file")
    if SOURCE_LEAK.search(json.dumps([a_rows, b_rows], ensure_ascii=False)):
        raise RuntimeError("source path or corpus identifier leaked to annotator file")

    image_refs: list[str] = []
    for task_id in by_a:
        left = {key: value for key, value in by_a[task_id].items() if key != "notes"}
        right = {key: value for key, value in by_b[task_id].items() if key != "notes"}
        if left != right:
            raise RuntimeError(f"annotator task payloads differ for {task_id}")
        for relative in by_a[task_id].get("images", []):
            if not OPAQUE_IMAGE.fullmatch(relative):
                raise RuntimeError(f"non-opaque image reference: {relative}")
            path = (base / relative).resolve()
            if path.parent != (base / "images").resolve() or not path.is_file():
                raise RuntimeError(f"missing or unsafe audit image: {relative}")
            image_refs.append(relative)

    image_manifest = json.loads(images_manifest_path.read_text(encoding="utf-8"))
    manifest_files = {row["file"]: row for row in image_manifest["files"]}
    if len(manifest_files) != image_manifest["n_files"] or set(image_refs) != set(manifest_files):
        raise RuntimeError("audit image manifest does not match task references")
    for relative, record in manifest_files.items():
        path = base / relative
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"audit image checksum mismatch: {relative}")

    if not all("source_images" in row for row in key_rows):
        raise RuntimeError("closed key lacks source-image mapping")
    automatic_label_counts = {
        kind: dict(sorted(Counter(row.get("auto_label") for row in key_rows if row["audit_type"] == kind).items()))
        for kind in sorted(EXPECTED_TYPES - {"age_extraction"})
    }
    duplicate_labels = automatic_label_counts["near_duplicate"]
    minimum_per_side = max(1, expected_per_type // 5)
    if (
        duplicate_labels.get("near_duplicate", 0) < minimum_per_side
        or duplicate_labels.get("distinct_photo", 0) < minimum_per_side
    ):
        raise RuntimeError(
            "near-duplicate audit does not cover both sides of the production threshold: "
            f"{duplicate_labels}"
        )
    duplicate_similarities = np.asarray(
        [
            float(row["cosine_similarity"])
            for row in key_rows
            if row["audit_type"] == "near_duplicate"
        ],
        dtype=float,
    )
    if len(duplicate_similarities) != expected_per_type:
        raise RuntimeError("near-duplicate key lacks cosine similarities")
    duplicate_similarity_range = {
        "threshold": 0.97,
        "min": float(duplicate_similarities.min()),
        "p25": float(np.quantile(duplicate_similarities, 0.25)),
        "median": float(np.median(duplicate_similarities)),
        "p75": float(np.quantile(duplicate_similarities, 0.75)),
        "max": float(duplicate_similarities.max()),
    }
    integrity_labels = automatic_label_counts["group_integrity"]
    expected_integrity = {
        "different_identity": expected_per_type // 2,
        "same_identity": expected_per_type - expected_per_type // 2,
    }
    if integrity_labels != expected_integrity:
        raise RuntimeError(
            "group-integrity audit is not balanced across accepted/rejected groups: "
            f"{integrity_labels}"
        )
    source_occurrences = [source for row in key_rows for source in row["source_images"]]
    unique_sources = set(source_occurrences)
    source_types: dict[str, set[str]] = {}
    for row in key_rows:
        for source in row["source_images"]:
            source_types.setdefault(source, set()).add(row["audit_type"])
    cross_stratum_reuse = sum(len(types) > 1 for types in source_types.values())
    if cross_stratum_reuse:
        raise RuntimeError(f"source images reused across audit strata: {cross_stratum_reuse}")
    return {
        "valid": True,
        "annotators": 2,
        "tasks_per_annotator": len(a_rows),
        "adjudication_template_rows": len(adjudication_rows),
        "counts_by_type": dict(sorted(counts.items())),
        "automatic_label_counts_by_type": automatic_label_counts,
        "near_duplicate_similarity": duplicate_similarity_range,
        "age_baseline_coverage": age_baseline_coverage,
        "independently_shuffled": True,
        "responses_filled": 0,
        "opaque_image_files": len(manifest_files),
        "source_image_occurrences": len(source_occurrences),
        "unique_source_images": len(unique_sources),
        "repeated_source_occurrences": len(source_occurrences) - len(unique_sources),
        "cross_stratum_source_reuse": cross_stratum_reuse,
        "source_identifier_leaks": 0,
        "automatic_decision_leaks": 0,
        "image_checksum_mismatches": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=data_path("data_dir", "interim", "human_audit")
    )
    parser.add_argument("--expected-per-type", type=int, default=400)
    parser.add_argument(
        "--output", type=Path, default=data_path("metrics_dir", "human_audit_pack_validation.json")
    )
    args = parser.parse_args()
    result = validate(args.base, args.expected_per_type)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    inputs = [
        args.base / "annotator_a.jsonl",
        args.base / "annotator_b.jsonl",
        args.base / "audit_key.jsonl",
        args.base / "adjudicated_gold.template.jsonl",
        args.base / "images.manifest.json",
        args.base / "README.md",
        args.base / "age_baseline_predictions.jsonl",
        args.base / "age_baseline_predictions.manifest.json",
    ]
    write_experiment_manifest(
        args.output.with_suffix(".manifest.json"),
        experiment="blinded-human-audit-pack-validation",
        parameters={
            "expected_types": sorted(EXPECTED_TYPES),
            "expected_per_type": args.expected_per_type,
            "requires_empty_responses": True,
        },
        metrics=result,
        inputs=inputs,
        outputs=[args.output],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
