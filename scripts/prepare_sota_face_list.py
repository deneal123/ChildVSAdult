"""Prepare private recognition inputs under an explicit, outcome-free policy.

Conflicting retained-face ages become missing, never last-write-wins. Orphan
labels are ignored and counted; no source annotations are changed. This is a
new declared common protocol, not recovery of an old executed dataset.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from age_gap.common.io import read_jsonl, write_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.audit_sota_face_inputs import audit

POLICY = {
    "split": "train",
    "minimum_faces_per_group_before_and_after_crop_check": 2,
    "identity": "recorded-person-contiguous-sorted",
    "conflicting_age": "missing-no-last-write-wins",
    "orphan_age_labels": "ignore-count",
    "missing_crop": "exclude-count",
    "missing_person_mapping": "exclude-count-no-group-id-fallback",
    "human_identity_clearance": False,
}


def select(groups, splits, clusters, crop_resolver):
    inventory = audit(groups, splits, clusters)
    fatal = (
        "conflicting_split_rows",
        "conflicting_person_id_rows",
        "duplicate_group_rows",
        "groups_without_split",
        "groups_with_invalid_split",
        "groups_with_duplicate_faces",
        "recorded_people_across_splits",
        "faces_across_splits",
        "faces_with_multiple_recorded_people",
        "invalid_age_labels",
    )
    if any(inventory["violations"].get(key, 0) for key in fatal):
        raise ValueError("ambiguous recorded identity/split or malformed metadata")
    split_map = {r["identity_group_id"]: r["split"] for r in splits}
    person_map = {r["identity_group_id"]: r["person_id"] for r in clusters}
    counts, candidates, decisions = Counter(), [], []
    for row in sorted(groups, key=lambda r: r["identity_group_id"]):
        group = row["identity_group_id"]
        if split_map[group] != "train":
            counts["nontrain_groups"] += 1
            continue
        faces = row["faces"]
        ages = defaultdict(set)
        for label in row.get("age_labels", []):
            face, age = label.get("face_id"), label.get("age")
            if face not in faces:
                counts["train_unmapped_or_orphan_age_labels"] += 1
                continue
            if age is not None:
                ages[face].add(age)
        reason = None
        if group not in person_map:
            reason = "missing_recorded_person"
        elif len(faces) < 2:
            reason = "fewer_than_two_declared_faces"
        if reason:
            counts[reason + "_groups"] += 1
            decisions.append(dict(group=group, reason=reason, faces=len(faces)))
            continue
        retained = []
        for face in faces:
            path = crop_resolver(face)
            if path is None:
                counts["missing_crops"] += 1
                decisions.append(dict(group=group, face_id=face, reason="missing_crop"))
                continue
            values = sorted(ages[face])
            status = "conflict_masked" if len(values) > 1 else "explicit" if values else "missing"
            retained.append(
                dict(
                    face_id=face,
                    person_id=person_map[group],
                    group_id=group,
                    age=values[0] if len(values) == 1 else None,
                    age_status=status,
                    age_candidates=values,
                    crop_path=str(path),
                )
            )
        if len(retained) < 2:
            counts["fewer_than_two_existing_crops_groups"] += 1
            decisions.append(
                dict(group=group, reason="fewer_than_two_existing_crops", faces=len(retained))
            )
            continue
        candidates.extend(retained)
    people = sorted({r["person_id"] for r in candidates})
    identity = {person: index for index, person in enumerate(people)}
    candidates.sort(key=lambda r: (r["person_id"], r["face_id"]))
    for row in candidates:
        row["identity"] = identity[row["person_id"]]
        counts["retained_age_" + row["age_status"]] += 1
    counts["retained_images"], counts["retained_people"] = len(candidates), len(people)
    if not candidates:
        raise ValueError("no eligible recognition inputs")
    return candidates, decisions, dict(counts), inventory


def main():
    parser = argparse.ArgumentParser()
    for key in ("groups", "splits", "clusters", "crops", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    sources = [
        args.groups,
        args.splits,
        args.clusters,
        Path(__file__),
        Path("scripts/audit_sota_face_inputs.py"),
        Path("src/age_gap/common/io.py"),
        Path("src/age_gap/common/manifest.py"),
        Path("src/age_gap/settings/settings.toml"),
    ]
    before = [file_record(p) for p in sources]

    def crop(face):
        if not isinstance(face, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", face):
            raise ValueError("unsafe face ID")
        path = (args.crops / (face + ".jpg")).resolve()
        if not path.is_relative_to(args.crops.resolve()):
            raise ValueError("crop escapes declared cache")
        return path if path.is_file() else None

    rows, decisions, counts, inventory = select(*(list(read_jsonl(p)) for p in sources[:3]), crop)
    crop_paths = [Path(r["crop_path"]) for r in rows]
    crop_records = [file_record(p) for p in crop_paths]
    for path in crop_paths:
        image = cv2.imread(str(path))
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("undecodable retained crop")
    if before != [file_record(p) for p in sources] or crop_records != [
        file_record(p) for p in crop_paths
    ]:
        raise RuntimeError("inputs changed during preparation")
    args.out.mkdir(parents=True)
    private = args.out / "private"
    private.mkdir()
    faces, excluded = private / "faces.jsonl", private / "decisions.jsonl"
    write_jsonl(faces, rows)
    write_jsonl(excluded, decisions)
    summary = args.out / "summary.json"
    metrics = dict(
        counts=counts,
        inventory=inventory,
        preparation_complete=True,
        retained_crops_decoded=True,
        training_complete=False,
        human_identity_audit_complete=False,
        publication_ready=False,
    )
    summary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        args.out / "summary.manifest.json",
        experiment="sota-recognition-face-list",
        parameters=POLICY,
        metrics=metrics,
        inputs=sources + crop_paths,
        outputs=[summary, faces, excluded],
    )
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
