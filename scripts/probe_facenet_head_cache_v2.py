"""Balanced-label numerical follow-up; no trained checkpoint or qualified cache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
from scripts.probe_facenet_head_cache_v1 import ATOL, RTOL, probe
from scripts.run_oriented_campaign import verified

SIZES = (64, 48, 6, 1, 64)


def balanced_indices(labels, per_class=32):
    """Fixed source-order selection; no losses or external scores used."""
    if type(per_class) is not int or per_class < 1:
        raise ValueError("positive integer per_class required")
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError("binary integer labels required")
    groups = [
        [i for i, label in enumerate(labels) if label == target][:per_class] for target in (1, 0)
    ]
    if any(len(group) != per_class for group in groups):
        raise ValueError("insufficient rows in a class")
    return [index for pair in zip(*groups, strict=True) for index in pair]


def class_counts(labels, sizes=SIZES):
    if not sizes or any(type(size) is not int or size < 1 or size > len(labels) for size in sizes):
        raise ValueError("invalid batch schedule")
    return [
        {"batch_size": size, "positive": sum(labels[:size]), "negative": size - sum(labels[:size])}
        for size in sizes
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    torch.set_num_threads(1)
    started = time.monotonic()
    preflight = PROJECT_ROOT / "metrics/oriented_training_preflight_20261003/summary.manifest.json"
    first_probe = PROJECT_ROOT / "metrics/facenet_head_cache_probe_20261004/summary.manifest.json"
    native = verified(preflight, "oriented-uniform-training-preflight")
    prior = verified(first_probe, "facenet-head-cache-numerical-probe")
    if prior["metrics"].get("bounded_tolerance_passed") is not True:
        raise ValueError("completed v1 bounded comparison required")
    pairs = (
        PROJECT_ROOT / "metrics/oriented_exposure_matching_20261003/private/low_candidate_arm.jsonl"
    )
    base = FaceNetBackbone()
    _set_trainable(base, "head")
    base.train()
    _apply_batchnorm_policy(base, "frozen_all")
    ds = ImagePairDataset("train", pairs_file=str(pairs), preprocess=_bb_prep(base))
    if len(ds) != 1200:
        raise ValueError("expected current oriented LOW training rows")
    selected = balanced_indices(ds.labels)
    selected_labels = [ds.labels[i] for i in selected]
    composition = class_counts(selected_labels)
    if len(set(selected)) != 64:
        raise ValueError("selection must contain64distinct source rows")
    paths = sorted({path for i in selected for path in ds._items[i][:2]})
    source_paths = [
        PROJECT_ROOT / record["path"]
        for record in native["inputs"]
        if not record["path"].startswith("data/interim/faces/")
    ]
    inputs = list(
        dict.fromkeys(
            [
                preflight,
                first_probe,
                pairs,
                Path(__file__),
                PROJECT_ROOT / "scripts/probe_facenet_head_cache_v1.py",
                *source_paths,
                *paths,
            ]
        )
    )
    before = {str(path): file_record(path) for path in inputs}
    for record in [*native["inputs"], *prior["inputs"]]:
        path = PROJECT_ROOT / record["path"]
        if str(path) in before and before[str(path)] != record:
            raise RuntimeError("bound input changed before probe")
    batch = [torch.stack(list(values)) for values in zip(*(ds[i] for i in selected), strict=True)]
    rows = []
    for seed in (42, 1, 2):
        current = probe(base, batch, seed=seed, sizes=SIZES)
        for row, counts in zip(current, composition, strict=True):
            row["recorded_labels"] = counts
        rows.extend(current)
        print(
            json.dumps(
                {"seed": seed, "within_tolerance": all(r["within_tolerance"] for r in current)}
            ),
            flush=True,
        )
    for path in inputs:
        if file_record(path) != before[str(path)]:
            raise RuntimeError("probe input changed")
    summary = dict(
        diagnostic_complete=True,
        bounded_tolerance_passed=all(r["within_tolerance"] for r in rows),
        accelerator_qualified=False,
        training_complete=False,
        scientific_evaluation_complete=False,
        publication_ready=False,
        elapsed_seconds=time.monotonic() - started,
        selected_rows=64,
        steps=rows,
        class_counts=composition,
    )
    args.out.mkdir(parents=True)
    result = args.out / "summary.json"
    result.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_experiment_manifest(
        args.out / "summary.manifest.json",
        experiment="facenet-balanced-head-cache-numerical-probe",
        parameters=dict(
            device="cpu",
            threads=1,
            scope="head",
            batchnorm_policy="frozen_all",
            seeds=[42, 1, 2],
            batches=list(SIZES),
            rtol=RTOL,
            atol=ATOL,
            rows="first32positive/first32negative usable LOWtrain rows, interleaved; not campaign sampler",
            limitation="fifteen shadow Adam steps on train rows; six-row size is a shape diagnostic, no heldout labels trained; not full trajectory or cache-bank qualification",
        ),
        metrics=summary,
        inputs=inputs,
        outputs=[result],
    )


if __name__ == "__main__":
    main()
