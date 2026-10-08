"""Bound recognition inputs and explicit epoch execution, not full MTLFace/FAS."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import file_record
from scripts.mtlface_training_v2 import FaceRecord, RecognitionDataset, recognition_step
from scripts.prepare_sota_face_list import POLICY


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_face_rows(rows, counts, bound_crops):
    """Validate the declared list, not human truth of identities or ages."""
    if len(rows) != counts["retained_images"] or not rows:
        raise ValueError("retained image count mismatch")
    people = sorted({row["person_id"] for row in rows})
    if len(people) != counts["retained_people"]:
        raise ValueError("retained identity count mismatch")
    identities = {person: index for index, person in enumerate(people)}
    seen_faces, seen_paths, ages = set(), set(), {"explicit": 0, "missing": 0, "conflict_masked": 0}
    records = []
    for row in rows:
        path = _resolve(row["crop_path"]).resolve()
        if row["face_id"] in seen_faces or path in seen_paths:
            raise ValueError("duplicate face/crop in recognition list")
        seen_faces.add(row["face_id"])
        seen_paths.add(path)
        if row["identity"] != identities[row["person_id"]]:
            raise ValueError("recorded-person label mapping mismatch")
        if path not in bound_crops:
            raise ValueError("crop not bound by manifest inputs")
        candidates = row["age_candidates"]
        if any(isinstance(age, bool) or not isinstance(age, int) or age < 0 for age in candidates):
            raise ValueError("invalid age candidate")
        if candidates != sorted(set(candidates)):
            raise ValueError("age candidates must be sorted unique")
        status = (
            "conflict_masked" if len(candidates) > 1 else "explicit" if candidates else "missing"
        )
        expected_age = candidates[0] if len(candidates) == 1 else None
        if row["age_status"] != status or row["age"] != expected_age:
            raise ValueError("age conflict policy mismatch")
        ages[status] += 1
        records.append(FaceRecord(path, row["identity"], row["age"]))
    if any(ages[status] != counts.get("retained_age_" + status, 0) for status in ages):
        raise ValueError("age-status count mismatch")
    return records


def load_bound_dataset(manifest_path, preprocess):
    manifest_path = Path(manifest_path).resolve()
    native = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        native.get("experiment") != "sota-recognition-face-list"
        or native.get("parameters") != POLICY
    ):
        raise ValueError("declared recognition-list protocol required")
    metrics = native["metrics"]
    if (
        metrics.get("preparation_complete") is not True
        or metrics.get("retained_crops_decoded") is not True
    ):
        raise ValueError("completed decoded preparation required")
    for record in native["inputs"] + native["outputs"]:
        if file_record(_resolve(record["path"])) != record:
            raise ValueError("recognition prerequisite hash mismatch")
    faces = manifest_path.parent / "private/faces.jsonl"
    if file_record(faces) not in native["outputs"]:
        raise ValueError("face-list output not bound")
    bound_crops = {
        _resolve(record["path"]).resolve()
        for record in native["inputs"]
        if record["path"].endswith(".jpg")
    }
    records = validate_face_rows(list(read_jsonl(faces)), metrics["counts"], bound_crops)
    return RecognitionDataset(records, preprocess), native


def run_recognition_epoch(
    model, identity_head, optimizer, dataset, *, batch_size, generator, device
):
    """Uniform shuffled full epoch; caller owns seeded, continuing RNG and LR.

    No early stopping or checkpoint selection. num_workers=0/drop_last=False
    are explicit common-protocol choices, not recovered author settings.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("positive integer batch size required")
    if not isinstance(generator, torch.Generator):
        raise ValueError("explicit continuing torch Generator required")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    totals, batches, samples = {}, 0, 0
    for batch in loader:
        batch = tuple(value.to(device) for value in batch)
        result = recognition_step(model, identity_head, optimizer, batch)
        size = len(batch[0])
        for name, value in result.items():
            totals[name] = totals.get(name, 0.0) + value * size
        batches += 1
        samples += size
    if samples != len(dataset) or not samples:
        raise ValueError("full nonempty dataset coverage required")
    return dict(
        samples=samples,
        batches=batches,
        sample_weighted_means={name: value / samples for name, value in totals.items()},
    )
