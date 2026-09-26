"""Find exact and near-duplicate face images between mined data and benchmarks.

All outputs, including image references needed for manual candidate review, are
written below ignored ``data/interim/benchmark_overlap_audit``. The script does
not modify source images or benchmark caches. Perceptual-hash hits are candidates
only; this script never labels them as confirmed identity overlap.

Run with ``uv run python scripts/benchmark_image_overlap_audit.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from age_gap.common.io import PROJECT_ROOT, data_path, read_jsonl
from age_gap.common.manifest import sha256_file, write_experiment_manifest

DEFAULT_OUTPUT = Path("data/interim/benchmark_overlap_audit")
PHASH_MAX_DISTANCE = 4
_POPCOUNT = np.asarray([i.bit_count() for i in range(256)], dtype=np.uint8)


def pixel_sha256(image: np.ndarray) -> str:
    """Hash decoded pixels so equivalent encodings match exactly."""
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def perceptual_hash(image: np.ndarray) -> int:
    """Return a 64-bit DCT pHash (32x32 input, 8x8 low-frequency block)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8]
    values = low.flatten()[1:]
    threshold = float(np.median(values))
    bits = low.flatten() > threshold
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _resolve_image_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _training_images(max_train: int | None = None) -> list[tuple[str, Path]]:
    faces_by_id = {
        str(row["face_id"]): row
        for row in read_jsonl(data_path("data_dir", "interim", "faces.jsonl"))
        if row.get("is_usable") and row.get("face_crop_path")
    }
    group_path = data_path("data_dir", "processed", "identity_groups.jsonl")
    face_ids = sorted({str(face_id) for group in read_jsonl(group_path) for face_id in group["faces"]})
    records: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for face_id in face_ids:
        row = faces_by_id.get(face_id)
        if row is None:
            continue
        path = _resolve_image_path(str(row["face_crop_path"]))
        canonical = str(path.resolve()).casefold()
        if path.is_file() and canonical not in seen:
            seen.add(canonical)
            records.append((face_id, path))
    if max_train is not None:
        records = records[:max_train]
    return records


def _decode_bin(path: Path) -> list[np.ndarray]:
    # These local benchmark artifacts use the InsightFace verification format.
    with path.open("rb") as handle:
        bins, _labels = pickle.load(handle, encoding="bytes")
    images = [cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR) for blob in bins]
    return [image for image in images if image is not None]


def _benchmark_images(external: Path) -> tuple[dict[str, list[np.ndarray]], dict[str, Path]]:
    paths = {
        "FG-NET": external / "fgnet_crops.npz",
        "AgeDB-30": external / "agedb_30.bin",
        "CALFW": external / "calfw.bin",
        "LFW": external / "lfw_aligned.npz",
        "CACD-VS": external / "cacd_vs_aligned.npz",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing benchmark image artifacts: {missing}")

    output: dict[str, list[np.ndarray]] = {}
    for name, path in paths.items():
        if path.suffix == ".bin":
            output[name] = _decode_bin(path)
            continue
        with np.load(path, allow_pickle=False) as archive:
            arrays = [archive["crops"]] if name == "FG-NET" else [archive["a"], archive["b"]]
            output[name] = [image for array in arrays for image in array]
    return output, paths


def _block_index(hashes: list[int]) -> list[dict[int, list[int]]]:
    """Index eight bytes; distance <= 4 guarantees at least one shared block."""
    indexes: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(8)]
    for index, value in enumerate(hashes):
        for block in range(8):
            indexes[block][(value >> (block * 8)) & 0xFF].append(index)
    return indexes


def _candidate_pairs(
    train_hashes: list[int], benchmark_hashes: list[int], max_distance: int
) -> list[tuple[int, int, int]]:
    index = _block_index(train_hashes)
    candidates: list[tuple[int, int, int]] = []
    for benchmark_index, value in enumerate(benchmark_hashes):
        possible: set[int] = set()
        for block in range(8):
            possible.update(index[block].get((value >> (block * 8)) & 0xFF, ()))
        for train_index in possible:
            distance = hamming_distance(train_hashes[train_index], value)
            if distance <= max_distance:
                candidates.append((train_index, benchmark_index, distance))
    return candidates


def run_audit(output_dir: Path, *, max_train: int | None = None, max_distance: int = 4) -> dict[str, Any]:
    if not 0 <= max_distance <= 7:
        raise ValueError("pHash max distance must be in [0, 7] for complete block-index retrieval")
    output_dir.mkdir(parents=True, exist_ok=True)
    training = _training_images(max_train)
    if not training:
        raise RuntimeError("No usable mined face crops from identity_groups.jsonl were found")

    external = Path(str(data_path("data_dir", "external")))
    benchmarks, benchmark_paths = _benchmark_images(external)
    train_hashes: list[int] = []
    train_exact: dict[str, list[int]] = defaultdict(list)
    inventory_path = output_dir / "private_training_inventory.jsonl"
    with inventory_path.open("w", encoding="utf-8") as inventory:
        for face_id, path in training:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            train_index = len(train_hashes)
            exact = pixel_sha256(image)
            phash = perceptual_hash(image)
            train_hashes.append(phash)
            train_exact[exact].append(train_index)
            inventory.write(
                json.dumps(
                    {
                        "train_index": train_index,
                        "face_id": face_id,
                        "path": path.resolve().as_posix(),
                        "file_sha256": sha256_file(path),
                        "pixel_sha256": exact,
                        "phash_hex": f"{phash:016x}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # The inventory only contains entries that decoded successfully.
    inventory_rows = [json.loads(line) for line in inventory_path.read_text(encoding="utf-8").splitlines()]
    candidate_path = output_dir / "private_candidates.jsonl"
    summary: dict[str, Any] = {
        "status": "candidate_search_complete_manual_review_required",
        "confirmed_identity_overlaps": None,
        "verified_zero_overlap": False,
        "methods": {
            "exact": "SHA-256 over decoded pixel array (shape, dtype, bytes)",
            "perceptual": "64-bit DCT pHash; candidate if Hamming distance <= configured threshold",
            "perceptual_candidate_threshold": max_distance,
            "embedding_retrieval": "not_run",
        },
        "training_image_count": len(inventory_rows),
        "benchmarks": {},
        "private_candidate_file": candidate_path.name,
        "privacy": "Candidate file and inventory are private ignored artifacts; do not distribute publicly.",
        "interpretation": "Exact/pHash hits are image-level candidates only; no identity match is confirmed without blinded manual review.",
    }
    with candidate_path.open("w", encoding="utf-8") as candidates_file:
        for name, images in benchmarks.items():
            exact_hits = 0
            exact_pair_set: set[tuple[int, int]] = set()
            phash_candidates: list[tuple[int, int, int]] = []
            benchmark_hashes: list[int] = []
            for benchmark_index, image in enumerate(images):
                exact = pixel_sha256(image)
                for train_index in train_exact.get(exact, []):
                    exact_pair_set.add((train_index, benchmark_index))
                    candidates_file.write(
                        json.dumps(
                            {
                                "benchmark": name,
                                "match_type": "exact_decoded_pixels",
                                "train_index": train_index,
                                "benchmark_index": benchmark_index,
                                "phash_distance": 0,
                            }
                        )
                        + "\n"
                    )
                    exact_hits += 1
                benchmark_hashes.append(perceptual_hash(image))
            phash_candidates = [
                pair
                for pair in _candidate_pairs(train_hashes, benchmark_hashes, max_distance)
                if pair[:2] not in exact_pair_set
            ]
            for train_index, benchmark_index, distance in phash_candidates:
                candidates_file.write(
                    json.dumps(
                        {
                            "benchmark": name,
                            "match_type": "phash_candidate",
                            "train_index": train_index,
                            "benchmark_index": benchmark_index,
                            "phash_distance": distance,
                        }
                    )
                    + "\n"
                )
            summary["benchmarks"][name] = {
                "benchmark_image_count": len(images),
                "exact_image_candidate_pairs": exact_hits,
                "phash_candidate_pairs": len(phash_candidates),
                "manual_review_status": "pending",
                "confirmed_matches": None,
                "excluded_images": None,
                "unresolved_candidates": None,
            }

    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = output_dir / "manifest.json"
    source_metadata = [
        data_path("data_dir", "interim", "faces.jsonl"),
        data_path("data_dir", "processed", "identity_groups.jsonl"),
        *benchmark_paths.values(),
    ]
    write_experiment_manifest(
        manifest_path,
        experiment="mined-training-benchmark-image-overlap-candidates",
        parameters={
            "exact_match": "decoded-pixel SHA-256",
            "perceptual_hash": "DCT pHash 64-bit",
            "phash_max_hamming_distance": max_distance,
            "max_train_images": max_train,
            "embedding_retrieval": False,
            "manual_identity_confirmation": False,
            "private_outputs": True,
        },
        metrics={
            "training_images": len(inventory_rows),
            "benchmark_images": {key: len(value) for key, value in benchmarks.items()},
            "candidate_pairs": {
                key: value["phash_candidate_pairs"] + value["exact_image_candidate_pairs"]
                for key, value in summary["benchmarks"].items()
            },
            "confirmed_identity_overlaps": None,
            "verified_zero_overlap": False,
        },
        inputs=[*source_metadata, inventory_path],
        outputs=[report_path, candidate_path],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / DEFAULT_OUTPUT)
    parser.add_argument("--max-train", type=int, default=None, help="small reproducible smoke run")
    parser.add_argument("--phash-max-distance", type=int, default=PHASH_MAX_DISTANCE)
    args = parser.parse_args()
    summary = run_audit(args.output_dir, max_train=args.max_train, max_distance=args.phash_max_distance)
    print(json.dumps(summary["benchmarks"], ensure_ascii=False, indent=2))
    print(f"Private audit artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
