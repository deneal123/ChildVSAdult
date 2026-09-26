"""Backfill auditable image/face quality metadata without rerunning detection.

The historical ``faces.jsonl`` omitted raw resolution and blur even though they
were used during filtering. This resumable sidecar reconstructs those fields from
the stored source images and existing boxes/landmarks. It never changes canonical
labels or splits.

    uv run python scripts/build_face_quality_audit.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.manifest import write_experiment_manifest
from age_gap.preprocessing.quality import blur_variance


def _pose(landmarks: list[list[float]]) -> tuple[float | None, float | None]:
    if len(landmarks) < 5:
        return None, None
    points = np.asarray(landmarks, dtype=float)
    eye_center = (points[0] + points[1]) / 2
    eye_distance = np.linalg.norm(points[1] - points[0]) + 1e-6
    yaw = float((points[2][0] - eye_center[0]) / eye_distance)
    roll = float(np.arctan2(points[1][1] - points[0][1], points[1][0] - points[0][0]))
    return yaw, roll


def _audit_photo(task: tuple[str, list[dict], Path | None]) -> list[dict]:
    photo_id, faces, image_path = task
    image = cv2.imread(str(image_path)) if image_path and image_path.exists() else None
    image_height, image_width = image.shape[:2] if image is not None else (None, None)
    records: list[dict] = []
    for face in faces:
        bbox = face.get("bbox") or []
        face_width = face_height = blur = None
        if len(bbox) == 4:
            face_width = float(max(0.0, bbox[2] - bbox[0]))
            face_height = float(max(0.0, bbox[3] - bbox[1]))
            if image is not None:
                x1 = max(0, int(round(bbox[0])))
                y1 = max(0, int(round(bbox[1])))
                x2 = min(image_width, int(round(bbox[2])))
                y2 = min(image_height, int(round(bbox[3])))
                blur = blur_variance(image[y1:y2, x1:x2])
        yaw, roll = _pose(face.get("landmarks") or [])
        records.append(
            {
                "face_id": face["face_id"],
                "photo_id": photo_id,
                "is_usable": bool(face.get("is_usable")),
                "reject_reason": face.get("reject_reason"),
                "image_width": image_width,
                "image_height": image_height,
                "face_width": face_width,
                "face_height": face_height,
                "blur_var": blur,
                "det_score": face.get("det_score"),
                "face_quality_score": face.get("face_quality_score"),
                "pose_yaw_proxy": yaw,
                "pose_roll_rad": roll,
            }
        )
    return records


def main() -> None:
    posts_path = Path(str(data_path("data_dir", "raw", "posts.jsonl")))
    faces_path = Path(str(data_path("data_dir", "interim", "faces.jsonl")))
    output = Path(str(data_path("data_dir", "interim", "face_quality_audit.jsonl")))

    photo_paths: dict[str, Path] = {}
    for post in read_jsonl(posts_path):
        for photo in post.get("photos", []):
            if photo.get("local_path"):
                photo_paths[photo["photo_id"]] = Path(resolve_path(photo["local_path"]))

    by_photo: dict[str, list[dict]] = defaultdict(list)
    for face in read_jsonl(faces_path):
        by_photo[face["photo_id"]].append(face)
    completed = {row["face_id"] for row in read_jsonl(output)}

    output.parent.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            photo_id,
            [face for face in faces if face["face_id"] not in completed],
            photo_paths.get(photo_id),
        )
        for photo_id, faces in by_photo.items()
        if any(face["face_id"] not in completed for face in faces)
    ]
    written = 0
    with (
        output.open("a", encoding="utf-8") as stream,
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        for photo_index, records in enumerate(pool.map(_audit_photo, tasks), start=1):
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            if photo_index % 500 == 0:
                stream.flush()
                print(f"pending photos {photo_index}/{len(tasks)}; new face records {written}")

    total = sum(1 for _ in read_jsonl(output))
    if total != sum(len(faces) for faces in by_photo.values()):
        raise RuntimeError(f"Incomplete quality sidecar: {total} records for {len(by_photo)} photos")
    write_experiment_manifest(
        output.with_suffix(".manifest.json"),
        experiment="face-quality-metadata-backfill",
        parameters={"pose": "five-point yaw proxy and eye-line roll", "blur": "Laplacian variance"},
        metrics={"records": total},
        inputs=[posts_path, faces_path],
        outputs=[output],
    )
    print(f"OK: {output} ({total} records)")


if __name__ == "__main__":
    main()
