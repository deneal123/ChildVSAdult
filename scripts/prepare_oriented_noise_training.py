"""Bind oriented permutation42 to the decoded clean frozen-BN training preflight.

Preparation only: does not train, infer, or reuse partial checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.run_oriented_campaign import PROTOCOL, paths_from, verified
from scripts.validate_clean_noise_control import validate_control


def require_preflight(native):
    if (
        native["metrics"].get("preflight_complete") is not True
        or native["parameters"] != PROTOCOL
        or native["metrics"].get("protocol") != PROTOCOL
        or native["metrics"].get("crop_audit", {}).get("all_decodable") is not True
        or native["metrics"].get("crop_audit", {}).get("missing_rows_dropped") != 0
    ):
        raise ValueError("completed unchanged decoded uniform frozen-BN preflight required")


def require_preparation(native, clean, noisy):
    if (
        native["parameters"].get("seed") != 42
        or file_record(clean) not in native["inputs"]
        or file_record(noisy) not in native["outputs"]
    ):
        raise ValueError("oriented clean arm and permutation42 output linkage required")


def require_crop_coverage(native, face_ids, resolver):
    declared = {r["path"]: r for r in native["inputs"] if r["path"].endswith(".jpg")}
    if not declared:
        raise ValueError("decoded crop records required")
    for face in sorted(face_ids):
        record = file_record(resolver(face))
        if declared.get(record["path"]) != record:
            raise ValueError("noise crop outside unchanged decoded preflight")
    return len(face_ids)


def prepare(preflight, noise_dir, out):
    if out.exists():
        raise FileExistsError("fresh output directory required")
    native = verified(preflight, "oriented-uniform-training-preflight")
    require_preflight(native)
    prepared_path = noise_dir / "summary.manifest.json"
    prepared = verified(prepared_path, "partial-exact-age-supervision-noise-control")
    clean = (
        PROJECT_ROOT
        / "metrics/oriented_exposure_matching_20261003/private/cross_candidate_arm.jsonl"
    )
    noisy = noise_dir / "private/partial_noise_arm.jsonl"
    require_preparation(prepared, clean, noisy)
    if file_record(clean) not in native["inputs"]:
        raise ValueError("clean arm not bound by decoded preflight")
    from age_gap.training.finetune import ImagePairDataset, _crop_path
    from scripts.build_partial_noise_control import build_control

    clusters = PROJECT_ROOT / "data/processed/person_clusters.jsonl"
    canonical = PROJECT_ROOT / "data/processed/pairs.jsonl"
    groups = {}
    for row in read_jsonl(clusters):
        group, person = str(row["identity_group_id"]), str(row["person_id"])
        if group in groups and groups[group] != person:
            raise ValueError("conflicting recorded-person mapping")
        groups[group] = person
    known = {
        tuple(sorted((r["face_a"], r["face_b"])))
        for r in read_jsonl(canonical)
        if r.get("label") == 1
    }
    input_paths = list(
        dict.fromkeys(
            p.resolve()
            for p in [
                *paths_from(native),
                preflight,
                *paths_from(prepared),
                prepared_path,
                clusters,
                canonical,
                Path(__file__),
                PROJECT_ROOT / "scripts/validate_clean_noise_control.py",
            ]
        )
    )
    before = [file_record(p) for p in input_paths]
    clean_rows, noisy_rows = list(read_jsonl(clean)), list(read_jsonl(noisy))
    audit = validate_control(clean_rows, noisy_rows, groups, known_genuine_edges=known)
    regenerated, _ = build_control(
        clean_rows,
        groups,
        seed=42,
        attempts=prepared["parameters"]["attempts"],
        known_positive_pairs=known,
    )
    if regenerated != noisy_rows:
        raise ValueError("noise arm not reproducible from declared intervention")
    faces = {r[f"face_{s}"] for r in noisy_rows for s in ("a", "b")}
    covered = require_crop_coverage(native, faces, _crop_path)
    counts = {}
    for split in ("train", "val", "test"):
        expected = sum(r["split"] == split for r in noisy_rows)
        counts[split] = len(ImagePairDataset(split, str(noisy), gap_weight=0.0))
        if counts[split] != expected:
            raise ValueError("noise loader dropped rows")
    if counts != native["metrics"]["loader_counts"]["CROSS"]:
        raise ValueError("clean/noise loader budgets differ")
    if before != [file_record(p) for p in input_paths]:
        raise ValueError("inputs changed during noise preparation")
    result = dict(
        preflight_complete=True,
        training_complete=False,
        evaluation_complete=False,
        publication_ready=False,
        protocol=PROTOCOL,
        permutation_seed=42,
        training_seeds=[42, 1, 2],
        loader_counts=counts,
        unique_crops_covered=covered,
        clean_noise_contract=audit,
        scope="supervision-noise sensitivity; fixed permutation not a training seed; not source/co-occurrence evidence",
    )
    out.mkdir(parents=True)
    summary = out / "summary.json"
    summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="oriented-uniform-noise-training-preflight",
        parameters=PROTOCOL | {"permutation_seed": 42},
        metrics=result,
        inputs=input_paths,
        outputs=[summary],
    )
    if before != json.loads(target.read_text(encoding="utf-8"))["inputs"]:
        target.unlink()
        raise ValueError("inputs changed during noise manifest write")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--noise", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.preflight, args.noise, args.out)


if __name__ == "__main__":
    main()
