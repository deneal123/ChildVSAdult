"""Private pre-dropout feature bank; not a qualified cached trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from age_gap.models.facenet import FaceNetBackbone
from age_gap.training.finetune import (
    ImagePairDataset,
    _apply_batchnorm_policy,
    _bb_prep,
    _set_trainable,
)
from scripts.probe_facenet_head_cache_v1 import capture_features
from scripts.run_oriented_campaign import verified


def collect_pool(pair_paths, dataset_factory=ImagePairDataset):
    if set(pair_paths) != {"low", "cross"}:
        raise ValueError("both oriented arms required")
    paths, counts, heldout = set(), {}, None
    for arm in ("low", "cross"):
        counts[arm] = {}
        for split, expected in (("train", 1200), ("val", 9158)):
            dataset = dataset_factory(split, pairs_file=str(pair_paths[arm]), crops_dir="faces")
            if len(dataset) != expected:
                raise ValueError("source row coverage changed")
            if split == "val":
                if heldout is not None and dataset._items != heldout:
                    raise ValueError("held-out row order/content differs")
                heldout = dataset._items
            counts[arm][split] = len(dataset)
            paths.update(path.resolve() for item in dataset._items for path in item[:2])
    return sorted(paths), counts


def pixel_digest(image):
    if image.shape != (3, 160, 160) or image.dtype != np.float32 or not np.isfinite(image).all():
        raise ValueError("finite native float32CHW pixels required")
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh cache directory required; partial banks not resumed")
    torch.set_num_threads(1)
    preflight = PROJECT_ROOT / "metrics/oriented_training_preflight_20261003/summary.manifest.json"
    probe = (
        PROJECT_ROOT / "metrics/facenet_balanced_head_cache_probe_20261004/summary.manifest.json"
    )
    native = verified(preflight, "oriented-uniform-training-preflight")
    diagnostic = verified(probe, "facenet-balanced-head-cache-numerical-probe")
    if diagnostic["metrics"].get("bounded_tolerance_passed") is not True:
        raise ValueError("bounded balanced-label comparison required")
    pair_paths = {
        arm: PROJECT_ROOT
        / f"metrics/oriented_exposure_matching_20261003/private/{arm}_candidate_arm.jsonl"
        for arm in ("low", "cross")
    }
    paths, counts = collect_pool(pair_paths)
    upstream = [
        PROJECT_ROOT / record["path"]
        for record in native["inputs"]
        if not record["path"].startswith("data/interim/faces/")
    ]
    inputs = list(
        dict.fromkeys(
            [
                preflight,
                probe,
                Path(__file__),
                PROJECT_ROOT / "scripts/probe_facenet_head_cache_v1.py",
                *upstream,
                *paths,
            ]
        )
    )
    before = {str(path): file_record(path) for path in inputs}
    native_records = {str(PROJECT_ROOT / record["path"]): record for record in native["inputs"]}
    for path in paths:
        if native_records.get(str(path)) != before[str(path)]:
            raise ValueError("crop absent or changed in native preflight")
    for record in [*native["inputs"], *diagnostic["inputs"]]:
        path = PROJECT_ROOT / record["path"]
        if str(path) in before and before[str(path)] != record:
            raise ValueError("bound source changed")
    torch.manual_seed(42)
    base = FaceNetBackbone()
    _set_trainable(base, "head")
    base.train()
    _apply_batchnorm_policy(base, "frozen_all")
    initial_state = state_digest(base)
    prep = _bb_prep(base)
    private = args.out / "private"
    private.mkdir(parents=True)
    array_path = private / "features.npy"
    features = np.lib.format.open_memmap(
        array_path, mode="w+", dtype=np.float32, shape=(len(paths), 1792, 1, 1)
    )
    rows = []
    for start in range(0, len(paths), 64):
        images = []
        for index, path in enumerate(paths[start : start + 64], start):
            image = cv2.imread(str(path))
            if image is None or image.shape != (112, 112, 3) or image.dtype != np.uint8:
                raise ValueError("native112x112BGR crop required")
            pixels = prep(image)
            images.append(torch.from_numpy(pixels))
            rows.append(
                dict(
                    feature_row=index,
                    crop=before[str(path)],
                    preprocessed_sha256=pixel_digest(pixels),
                )
            )
        values = capture_features(base, torch.stack(images))
        features[start : start + len(images)] = values.numpy()
        if start % (64 * 10) == 0 or start + len(images) == len(paths):
            print(
                json.dumps({"encoded_crops": start + len(images), "total": len(paths)}), flush=True
            )
    features.flush()
    del features
    if state_digest(base) != initial_state:
        raise RuntimeError("model parameters or buffers changed")
    for path in inputs:
        if file_record(path) != before[str(path)]:
            raise RuntimeError("cache source changed during encoding")
    index_path = private / "index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    summary = dict(
        cache_complete=True,
        encoded_crops=len(paths),
        shape=[len(paths), 1792, 1, 1],
        source_rows=counts,
        initial_state_sha256=initial_state,
        model_state_unchanged=True,
        test_crops_encoded=False,
        accelerator_qualified=False,
        training_complete=False,
        scientific_evaluation_complete=False,
        publication_ready=False,
    )
    result = args.out / "summary.json"
    result.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_experiment_manifest(
        args.out / "summary.manifest.json",
        experiment="facenet-private-frozen-trunk-cache",
        parameters=dict(
            device="cpu",
            threads=1,
            batch_size=64,
            stage="native avgpool output1792x1x1 before Dropout; no normalization",
            splits=["train", "val"],
            pair_files={k: str(v) for k, v in pair_paths.items()},
            no_optimizer=True,
            no_public_biometric_release=True,
            limitation="cache construction only; full cached-trainer/sampler/trajectory equivalence pending",
        ),
        metrics=summary,
        inputs=inputs,
        outputs=[result, array_path, index_path],
    )


if __name__ == "__main__":
    main()
