"""Read-only inventory before source-bound recognition-list construction."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from age_gap.common.io import read_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest


def audit(groups, splits, clusters):
    split_map, person_map = {}, {}
    violations = Counter()
    for rows, mapping, field in ((splits, split_map, "split"), (clusters, person_map, "person_id")):
        for row in rows:
            group, value = row["identity_group_id"], row[field]
            if group in mapping and mapping[group] != value:
                violations["conflicting_" + field + "_rows"] += 1
            mapping[group] = value
    people_splits, faces_people, faces_splits = defaultdict(set), defaultdict(set), defaultdict(set)
    counts, age_values, seen_groups = Counter(), defaultdict(set), set()
    for row in groups:
        group = row["identity_group_id"]
        if group in seen_groups:
            violations["duplicate_group_rows"] += 1
        seen_groups.add(group)
        counts["groups"] += 1
        if group not in split_map:
            violations["groups_without_split"] += 1
        if group not in person_map:
            violations["groups_without_recorded_person"] += 1
        split, person = split_map.get(group), person_map.get(group)
        if split not in ("train", "val", "test"):
            violations["groups_with_invalid_split"] += 1
        if person is not None and split is not None:
            people_splits[person].add(split)
        faces = row.get("faces", [])
        counts["face_occurrences"] += len(faces)
        counts["groups_with_at_least_two_faces"] += len(set(faces)) >= 2
        if len(faces) != len(set(faces)):
            violations["groups_with_duplicate_faces"] += 1
        for face in faces:
            if person is not None:
                faces_people[face].add(person)
            if split is not None:
                faces_splits[face].add(split)
        for label in row.get("age_labels", []):
            face, value = label.get("face_id"), label.get("age")
            if face is None:
                counts["unmapped_age_labels"] += 1
                continue
            if face not in faces:
                violations["age_labels_outside_group_faces"] += 1
            if value is None:
                counts["missing_age_labels"] += 1
            elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                violations["invalid_age_labels"] += 1
            else:
                age_values[face].add(value)
    violations["recorded_people_across_splits"] = sum(len(v) > 1 for v in people_splits.values())
    violations["faces_across_splits"] = sum(len(v) > 1 for v in faces_splits.values())
    violations["faces_with_multiple_recorded_people"] = sum(
        len(v) > 1 for v in faces_people.values()
    )
    violations["faces_with_conflicting_age_values"] = sum(len(v) > 1 for v in age_values.values())
    counts["unique_faces"] = len(faces_splits)
    counts["recorded_people"] = len(people_splits)
    return dict(
        counts=dict(counts),
        violations=dict(violations),
        input_contract_clean=not any(violations.values()),
        human_identity_audit_complete=False,
        crops_verified=False,
        training_complete=False,
        publication_ready=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    sources = [args.groups, args.splits, args.clusters, Path(__file__)]
    before = [file_record(path) for path in sources]
    result = audit(*(list(read_jsonl(path)) for path in sources[:3]))
    if before != [file_record(path) for path in sources]:
        raise RuntimeError("inputs changed during audit")
    args.out.mkdir(parents=True)
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        args.out / "summary.manifest.json",
        experiment="sota-recognition-input-audit",
        parameters={
            "split_policy": "declared-recorded-person",
            "age_policy": "audit-all-explicit-labels-no-adjudication",
        },
        metrics=result,
        inputs=sources,
        outputs=[summary],
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
