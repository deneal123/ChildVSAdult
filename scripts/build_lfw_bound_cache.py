"""Build a NEW CPU-only LFW cache with independently replayable row/source binding.

No legacy cache is read, modified, or retroactively assigned provenance. Official
pairs.txt order and contiguous folds are mandatory. Face data and binding rows stay
in a private directory; only the aggregate summary is publication-facing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, sha256_file, write_experiment_manifest


@dataclass(frozen=True)
class Pair:
    index: int
    fold: int
    label: int
    name_a: str
    name_b: str
    image_a: str
    image_b: str


def _image(name: str, instance: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
        raise ValueError("invalid person name in protocol")
    if not instance.isdigit() or not 1 <= int(instance) <= 9999:
        raise ValueError("invalid image instance in protocol")
    return f"{name}/{name}_{int(instance):04d}.jpg"


def parse_protocol(path: Path) -> tuple[list[Pair], int, int]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines or len(lines[0].split()) != 2:
        raise ValueError("View-2 header must declare folds and pairs per class/fold")
    try:
        folds, per_class = map(int, lines[0].split())
    except ValueError as exc:
        raise ValueError("noninteger protocol header") from exc
    if not 1 <= folds <= 100 or not 1 <= per_class <= 10000:
        raise ValueError("unsupported protocol dimensions")
    specs = lines[1:]
    if len(specs) != 2 * folds * per_class:
        raise ValueError("protocol row count disagrees with header")
    pairs = []
    for index, line in enumerate(specs):
        fields = line.split()
        label = int(index % (2 * per_class) < per_class)
        if label and len(fields) == 3:
            name_a, ia, ib = fields
            name_b = name_a
            if int(ia) == int(ib):
                raise ValueError("positive pair reuses the exact image")
        elif not label and len(fields) == 4:
            name_a, ia, name_b, ib = fields
            if name_a == name_b:
                raise ValueError("negative pair names the same person twice")
        else:
            raise ValueError("protocol class order is not positive/negative fold blocks")
        pairs.append(Pair(index, index // (2 * per_class), label, name_a, name_b,
                          _image(name_a, ia), _image(name_b, ib)))
    return pairs, folds, per_class


def source_path(raw_root: Path, image: str) -> Path:
    root = raw_root.resolve()
    path = (root / image).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("missing or escaped protocol image")
    return path


def decode_bgr(path: Path) -> np.ndarray:
    """Replicate local sklearn full-canvas/resize=1.0 float32 round-trip, streaming."""
    with Image.open(path) as im:
        if im.mode != "RGB" or im.size != (250, 250):
            raise ValueError("LFW source must be a 250x250 RGB image")
        pixels = np.asarray(im.crop((0, 0, 250, 250)).resize((250, 250)), dtype=np.float32)
    pixels /= 255.0
    pixels *= 255.0
    return cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2BGR)


def pixel_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def crop_with_landmarks(bgr: np.ndarray, landmarks, size: int) -> np.ndarray:
    if landmarks is None:
        h, w = bgr.shape[:2]
        width = min(h, w)
        y, x = (h - width) // 2, (w - width) // 2
        return cv2.resize(bgr[y:y + width, x:x + width], (size, size))
    from insightface.utils.face_align import norm_crop

    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("invalid alignment landmarks")
    return norm_crop(bgr, landmark=points, image_size=size, mode="arcface")


def make_cpu_detector(weights: Path, threads: int, det_size: int):
    import onnxruntime as ort
    from insightface.model_zoo.scrfd import SCRFD

    if not weights.is_file():
        raise FileNotFoundError("local detector weights are required; no download is allowed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(weights), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("detector must use CPU exclusively")
    model = SCRFD(model_file=str(weights), session=session)
    if not model.use_kps:
        raise ValueError("detector must expose five alignment landmarks")
    model.input_size = (det_size, det_size)
    model.det_thresh = 0.5

    def detect(bgr):
        boxes, points = model.detect(bgr, input_size=(det_size, det_size))
        if len(boxes) == 0:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return np.asarray(points[int(np.argmax(areas))], dtype=np.float32).tolist()

    return detect


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def construct(private: Path, pairs: list[Pair], protocol: Path, raw_root: Path,
              detector: Callable, *, size: int = 112) -> tuple[Path, Path, Path, dict]:
    """Instrument the actual construction, not a later annotation of an existing cache."""
    if private.exists() and any(private.iterdir()):
        raise FileExistsError("refusing nonempty construction directory")
    private.mkdir(parents=True, exist_ok=True)
    shape = (len(pairs), size, size, 3)
    a = np.lib.format.open_memmap(private / "a.npy", mode="w+", dtype=np.uint8, shape=shape)
    b = np.lib.format.open_memmap(private / "b.npy", mode="w+", dtype=np.uint8, shape=shape)
    sources, crops, rows = {}, {}, []
    for pair in pairs:
        row = {"index": pair.index, "label": pair.label, "fold": pair.fold}
        for side, name, image, array in (
            ("a", pair.name_a, pair.image_a, a), ("b", pair.name_b, pair.image_b, b)
        ):
            if image not in sources:
                raw = source_path(raw_root, image)
                before = sha256_file(raw)
                pixels = decode_bgr(raw)
                landmarks = detector(pixels)
                crop = crop_with_landmarks(pixels, landmarks, size)
                if crop.shape != (size, size, 3) or crop.dtype != np.uint8:
                    raise ValueError("invalid aligned crop shape/type")
                if sha256_file(raw) != before:
                    raise ValueError("source changed during decoding/alignment")
                sources[image] = {"source_sha256": before, "decoded_bgr_sha256": pixel_hash(pixels),
                                  "landmarks": landmarks, "crop_sha256": pixel_hash(crop),
                                  "detected": landmarks is not None}
                crops[image] = crop
            array[pair.index] = crops[image]
            # These are private linkage tokens, NOT a claim of anonymisation.
            row[f"subject_{side}"] = hashlib.sha256(name.encode("utf-8")).hexdigest()
            row[f"image_{side}"] = image
            row[f"crop_sha256_{side}"] = sources[image]["crop_sha256"]
        rows.append(row)
        if (pair.index + 1) % 100 == 0:
            print(f"aligned {pair.index + 1}/{len(pairs)} pairs; {len(sources)} unique images", flush=True)
    a.flush()
    b.flush()
    labels = np.array([p.label for p in pairs], dtype=np.int64)
    folds = np.array([p.fold for p in pairs], dtype=np.int64)
    misses = sum(not sources[image]["detected"] for p in pairs for image in (p.image_a, p.image_b))
    miss_rate = misses / (2 * len(pairs))
    rows_path, sources_path = private / "pair_rows.jsonl", private / "source_metadata.json"
    with rows_path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    _json(sources_path, sources)
    cache = private / "lfw_bound.npz"
    np.savez_compressed(cache, a=a, b=b, issame=labels, fold_ids=folds,
                        miss_rate=np.float64(miss_rate), binding_version=np.int64(1),
                        protocol_sha256=np.array(sha256_file(protocol)),
                        rows_sha256=np.array(sha256_file(rows_path)),
                        sources_sha256=np.array(sha256_file(sources_path)))
    del a, b, crops
    return cache, rows_path, sources_path, {
        "n_pairs": len(pairs), "n_subjects": len({p.name_a for p in pairs} | {p.name_b for p in pairs}),
        "n_unique_images": len(sources), "missing_detection_endpoints": misses,
        "miss_rate": miss_rate, "legacy_cache_replaced": False,
    }


def verify_binding(cache: Path, protocol: Path, raw_root: Path, *, replay: bool = True) -> dict:
    """Check EVERY endpoint against official order and independently replay source transforms.

    Does not claim historical origin of a legacy cache or training-identity independence.
    Full crop replay uses recorded landmarks and pinned alignment, not fresh detection.
    """
    pairs, folds, per_class = parse_protocol(protocol)
    private = cache.parent
    rows_path, sources_path = private / "pair_rows.jsonl", private / "source_metadata.json"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    if len(rows) != len(pairs):
        raise ValueError("row metadata count mismatch")
    required_images = {im for p in pairs for im in (p.image_a, p.image_b)}
    if set(sources) != required_images:
        raise ValueError("source metadata does not cover exact protocol image set")
    with np.load(cache, allow_pickle=False) as z:
        if int(z["binding_version"]) != 1 or str(z["protocol_sha256"]) != sha256_file(protocol):
            raise ValueError("protocol/cache binding mismatch")
        for key, path in (("rows_sha256", rows_path), ("sources_sha256", sources_path)):
            if str(z[key]) != sha256_file(path):
                raise ValueError("private metadata checksum mismatch")
        labels, fold_ids = z["issame"], z["fold_ids"]
        if not np.array_equal(labels, [p.label for p in pairs]):
            raise ValueError("label order mismatch")
        if not np.array_equal(fold_ids, [p.fold for p in pairs]):
            raise ValueError("official contiguous folds mismatch")
        miss_rate = float(z["miss_rate"])
        if not np.isfinite(miss_rate) or not 0 <= miss_rate <= 0.05:
            raise ValueError("detector misses exceed 5% or invalid miss rate")
        size = None
        for side in ("a", "b"):
            array = z[side]
            if array.dtype != np.uint8 or array.ndim != 4 or array.shape[0] != len(pairs) or array.shape[-1] != 3:
                raise ValueError("invalid endpoint array")
            if array.shape[1] != array.shape[2] or (size is not None and size != array.shape[1]):
                raise ValueError("inconsistent endpoint dimensions")
            size = array.shape[1]
            for pair, row, crop in zip(pairs, rows, array, strict=True):
                image = getattr(pair, f"image_{side}")
                name = getattr(pair, f"name_{side}")
                expected = {"index": pair.index, "label": pair.label, "fold": pair.fold,
                            f"image_{side}": image,
                            f"subject_{side}": hashlib.sha256(name.encode("utf-8")).hexdigest()}
                if any(row.get(key) != value for key, value in expected.items()):
                    raise ValueError("official person/image row mapping mismatch")
                digest = pixel_hash(crop)
                if digest != row[f"crop_sha256_{side}"] or digest != sources[image]["crop_sha256"]:
                    raise ValueError("endpoint crop pixel mismatch")
            del array
    computed_misses = sum(not sources[im]["detected"] for p in pairs for im in (p.image_a, p.image_b))
    if miss_rate != computed_misses / (2 * len(pairs)):
        raise ValueError("miss rate disagrees with endpoint detection records")
    for image, record in sources.items():
        raw = source_path(raw_root, image)
        if sha256_file(raw) != record["source_sha256"]:
            raise ValueError("raw source checksum mismatch")
        if bool(record["detected"]) != (record["landmarks"] is not None):
            raise ValueError("detection flag/landmarks mismatch")
        if replay:
            pixels = decode_bgr(raw)
            if pixel_hash(pixels) != record["decoded_bgr_sha256"]:
                raise ValueError("source decoder pixel mismatch")
            crop = crop_with_landmarks(pixels, record["landmarks"], size)
            if pixel_hash(crop) != record["crop_sha256"]:
                raise ValueError("independent source-transform replay mismatch")
    return {"row_binding_verified": True, "all_endpoints_checked": 2 * len(pairs),
            "unique_sources_checked": len(sources), "full_transform_replay": replay,
            "folds": folds, "pairs_per_class_per_fold": per_class,
            "training_identity_independence": "unverified"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    home = Path.home()
    parser.add_argument("--protocol", type=Path, default=home / "scikit_learn_data/lfw_home/pairs.txt")
    parser.add_argument("--raw-root", type=Path, default=home / "scikit_learn_data/lfw_home/lfw_funneled")
    parser.add_argument("--weights", type=Path, default=home / ".insightface/models/buffalo_l/det_10g.onnx")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "metrics/lfw_bound_cache_20261002")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    cv2.setNumThreads(args.threads)
    if args.verify_only:
        print(json.dumps(verify_binding(args.verify_only, args.protocol, args.raw_root)))
        return
    pairs, folds, per_class = parse_protocol(args.protocol)
    if (folds, per_class) != (10, 300):
        parser.error("production build requires full official 10x300 View-2 protocol")
    paths = [source_path(args.raw_root, image) for image in sorted({im for p in pairs for im in (p.image_a, p.image_b)})]
    if not args.execute:
        print(json.dumps({"status": "planned", "n_pairs": len(pairs), "n_unique_images": len(paths),
                          "legacy_cache_replaced": False, "device": "cpu"}))
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("choose a new output directory; existing results are preserved")
    if not args.weights.is_file():
        raise FileNotFoundError("local detector weights missing")
    import insightface.model_zoo.scrfd as detector_module
    import insightface.utils.face_align as alignment_module

    inputs = [args.protocol, args.weights, Path(__file__), Path(detector_module.__file__),
              Path(alignment_module.__file__), *paths]
    original_records = [file_record(path) for path in inputs]
    detector = make_cpu_detector(args.weights, args.threads, 640)
    cache, rows, sources, summary = construct(args.out / "private", pairs, args.protocol, args.raw_root, detector)
    summary.update(verify_binding(cache, args.protocol, args.raw_root))
    if [file_record(path) for path in inputs] != original_records:
        raise ValueError("construction inputs changed; refusing a completed manifest")
    summary.update({"source": "LFW funneled named images; official pairs.txt",
                    "claims_legacy_cache_binding": False, "publication_ready": False})
    result = args.out / "lfw_bound_cache.json"
    _json(result, summary)
    params = {"threads": args.threads, "device": "cpu", "providers": ["CPUExecutionProvider"],
              "detector": "SCRFD det_10g", "det_size": 640, "det_threshold": 0.5,
              "nms_threshold": 0.4, "alignment_mode": "arcface", "image_size": 112,
              "decoder": "PIL RGB full 250px + resize 1.0 + sklearn float32 round-trip -> BGR",
              "fallback": "central square resize; reject miss_rate > 0.05",
              "fold_assignment": "official contiguous positive/negative blocks",
              "versions": {package: version(package) for package in ("insightface", "onnxruntime", "Pillow", "numpy", "opencv-python")}}
    manifest = write_experiment_manifest(args.out / "lfw_bound_cache.manifest.json",
                                        experiment="lfw_bound_cache", parameters=params, metrics=summary,
                                        inputs=inputs,
                                        outputs=[result, cache, rows, sources, cache.parent / "a.npy", cache.parent / "b.npy"])
    if json.loads(manifest.read_text(encoding="utf-8"))["inputs"] != original_records:
        manifest.unlink()
        raise ValueError("inputs changed while writing final manifest")
    print(f"completed: {result}; all 12000 endpoints checked, legacy untouched", flush=True)


if __name__ == "__main__":
    main()
