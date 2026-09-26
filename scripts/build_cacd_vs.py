"""Build the canonical aligned CACD-VS cache from the authors' archive.

Source and protocol: https://bcsiriuschen.github.io/CARC/

The official archive contains 4,000 pairs in ten contiguous folds. Within each
400-pair fold, the first 200 pairs are positive and the next 200 are negative.
This script applies the same RetinaFace + five-point ``norm_crop`` preprocessing
used for the study data and records the archive checksum in a manifest.

    uv run python scripts/build_cacd_vs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from age_gap.common.io import data_path
from age_gap.common.logging import get_logger
from age_gap.common.manifest import sha256_file, write_experiment_manifest
from age_gap.preprocessing.crop_align import align_face
from age_gap.preprocessing.detect import FaceDetector

log = get_logger(__name__)
EXPECTED_ARCHIVE_SHA256 = "555a12479fd96302d87d83c20c59898e17cb5c1b53a962b257722db7a4143393"


def _align_one(detector: FaceDetector, image: np.ndarray, size: int) -> tuple[np.ndarray, bool]:
    faces = detector.detect(image)
    if faces:
        face = max(
            faces,
            key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
        )
        return align_face(image, face.kps, image_size=size), True
    height, width = image.shape[:2]
    side = min(height, width)
    y0, x0 = (height - side) // 2, (width - side) // 2
    return cv2.resize(image[y0 : y0 + side, x0 : x0 + side], (size, size)), False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=112)
    args = parser.parse_args()

    external = Path(str(data_path("data_dir", "external")))
    archive = external / "CACD_VS.tar"
    source = external / "CACD_VS"
    destination = external / "cacd_vs_aligned.npz"
    cache_dir = external / f"cacd_vs_aligned_{args.size}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or not source.is_dir():
        raise FileNotFoundError("Expected data/external/CACD_VS.tar and extracted CACD_VS/")
    archive_sha256 = sha256_file(archive)
    if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(
            f"Unexpected CACD-VS archive checksum: {archive_sha256}; "
            f"expected {EXPECTED_ARCHIVE_SHA256}"
        )
    expected_names = {f"{pair_id:04d}_{side}.jpg" for pair_id in range(4000) for side in (0, 1)}
    actual_names = {path.name for path in source.glob("*.jpg")}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)[:5]
        extra = sorted(actual_names - expected_names)[:5]
        raise RuntimeError(f"Invalid CACD-VS contents; missing={missing}, extra={extra}")

    detector = FaceDetector()
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    labels: list[int] = []
    fold_ids: list[int] = []
    misses = 0
    for pair_id in range(4000):
        aligned_pair = []
        for side in (0, 1):
            stem = f"{pair_id:04d}_{side}"
            cached = cache_dir / f"{stem}.jpg"
            fallback = cache_dir / f"{stem}.fallback.jpg"
            cached_path = cached if cached.exists() else fallback if fallback.exists() else None
            if cached_path is not None:
                aligned = cv2.imread(str(cached_path))
                if aligned is None:
                    raise RuntimeError(f"Cannot decode aligned cache file {cached_path}")
                detected = cached_path == cached
            else:
                image = cv2.imread(str(source / f"{stem}.jpg"))
                if image is None:
                    raise RuntimeError(f"Cannot decode CACD-VS pair {pair_id}, side {side}")
                aligned, detected = _align_one(detector, image, args.size)
                output = cached if detected else fallback
                if not cv2.imwrite(str(output), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Cannot write aligned cache file {output}")
            aligned_pair.append(aligned)
            misses += int(not detected)
        first.append(aligned_pair[0])
        second.append(aligned_pair[1])
        within_fold = pair_id % 400
        labels.append(int(within_fold < 200))
        fold_ids.append(pair_id // 400)
        if (pair_id + 1) % 250 == 0:
            log.info("CACD-VS aligned %d/4000 pairs (detector misses=%d)", pair_id + 1, misses)

    np.savez_compressed(
        destination,
        a=np.stack(first),
        b=np.stack(second),
        issame=np.asarray(labels, dtype=np.int64),
        fold_ids=np.asarray(fold_ids, dtype=np.int64),
        miss_rate=np.float64(misses / 8000),
    )
    write_experiment_manifest(
        destination.with_suffix(".manifest.json"),
        experiment="cacd-vs-alignment-cache",
        parameters={
            "source_url": "https://bcsiriuschen.github.io/CARC/",
            "expected_archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "alignment": "InsightFace buffalo_l RetinaFace + five-point norm_crop",
            "image_size": args.size,
        },
        metrics={"pairs": 4000, "positive": 2000, "negative": 2000, "miss_rate": misses / 8000},
        inputs=[archive],
        outputs=[destination],
    )
    print(f"OK: {destination} (detector misses {misses}/8000)")


if __name__ == "__main__":
    main()
