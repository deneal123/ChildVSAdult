"""Resample LOW/CROSS negatives strictly within selected positive-image pools.

Existing positives and heldout rows are preserved; original arms are never
overwritten. This removes one budget confound, not unmatched age/quality or
unknown-identity risks. No image processing or GPU work is performed.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import roc_auc_score

from age_gap.common.io import PROJECT_ROOT, read_jsonl, write_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.audit_matched_arm_image_budget import counts
from scripts.build_matched_agegap_arms import _endpoint_pool, _sample_impostors


def restrict_arms(arms, groups, *, seed=42, bin_years=10, known_positive_pairs=()):
    if set(arms) != {"LOW", "CROSS"} or bin_years < 1:
        raise ValueError("LOW/CROSS arms and positive bin width required")
    output, report, people, faces = {}, {}, {}, {}
    heldouts = []
    face_people = {}

    def person(row, side):
        group = str(row[f"identity_group_{side}"])
        if group not in groups or not groups[group]:
            raise ValueError("missing recorded-person mapping")
        identity = str(groups[group])
        face = str(row[f"face_{side}"])
        if face in face_people and face_people[face] != identity:
            raise ValueError("conflicting face-person mapping")
        face_people[face] = identity
        return identity

    for name in ("LOW", "CROSS"):
        rows = arms[name]
        if len({r["pair_id"] for r in rows}) != len(rows):
            raise ValueError("unique pair IDs required")
        if any(r.get("split") not in {"train", "val", "test"}
               or type(r.get("label")) is not int or r["label"] not in (0, 1) for r in rows):
            raise ValueError("valid binary labels and splits required")
        before_counts = counts(rows)
        positive = [r for r in rows if r["split"] == "train" and r["label"] == 1]
        heldout = [r for r in rows if r["split"] != "train"]
        if {r["split"] for r in heldout} != {"val", "test"}:
            raise ValueError("nonempty val and test required")
        heldouts.append(heldout)
        for row in rows:
            pa, pb = person(row, "a"), person(row, "b")
            if row["split"] == "train" and (
                row["face_a"] == row["face_b"] or (pa == pb) != (row["label"] == 1)
            ):
                raise ValueError("self image or recorded identities contradict train labels")
        selected = [{**r, "_person_id": person(r, "a")} for r in positive]
        people[name] = {r["_person_id"] for r in selected}
        faces[name] = {str(r[f"face_{s}"]) for r in positive for s in ("a", "b")}
        if people[name] & {person(r, s) for r in heldout for s in ("a", "b")}:
            raise ValueError("train-heldout recorded-person overlap")
        if faces[name] & {str(r[f"face_{s}"]) for r in heldout for s in ("a", "b")}:
            raise ValueError("train-heldout image overlap")
        observations = {}
        for row in positive:
            if any(type(row.get(f"age_{s}")) is not int or row[f"age_{s}"] < 0 for s in ("a", "b")):
                raise ValueError("nonnegative integer positive ages required")
            if row.get("age_gap") != abs(row["age_a"] - row["age_b"]):
                raise ValueError("inconsistent positive age gap")
            gap = row["age_gap"]
            if not (1 <= gap <= 2 if name == "LOW" else gap >= 25):
                raise ValueError("positive outside declared age-gap arm")
            for side in ("a", "b"):
                face, age = str(row[f"face_{side}"]), row[f"age_{side}"]
                if face in observations and observations[face] != age:
                    raise ValueError("conflicting selected-face ages; no majority resolution")
                observations[face] = age
        edges = [tuple(sorted((r["face_a"], r["face_b"]))) for r in positive]
        if len(edges) != len(set(edges)):
            raise ValueError("duplicate positive edges")
        pool, conflicts = _endpoint_pool(selected, groups)
        assert not conflicts and {e["face_id"] for e in pool} == faces[name]
        negatives, diagnostic = _sample_impostors(
            selected, pool, arm=name, seed=seed + (10000 if name == "LOW" else 20000),
            bin_years=bin_years, max_age_error=1,
        )
        forbidden = set(known_positive_pairs) | set(edges)
        if any(tuple(sorted((r["face_a"], r["face_b"]))) in forbidden for r in negatives):
            raise ValueError("generated negative collides with known genuine edge")
        for row in negatives:
            row["pair_type"] = "negative_restricted_positive_image_impostor"
            row["pair_id"] = "restricted_" + row["pair_id"]
        combined = [*copy.deepcopy(positive), *negatives, *copy.deepcopy(heldout)]
        if len({r["pair_id"] for r in combined}) != len(combined):
            raise ValueError("generated pair ID collision")
        after = counts(combined)
        if after["negative_only_images"] != 0 or after["all_train_images"] != after["positive_images"]:
            raise AssertionError("restricted image budget violated")
        labels = [r["label"] for r in combined if r["split"] == "train"]
        train = [r for r in combined if r["split"] == "train"]
        report[name] = {**after, "recorded_people": len(people[name]),
                        "old_all_train_images": before_counts["all_train_images"],
                        "negative_diagnostics": diagnostic,
                        "train_age_only_class_auc": {
                            field: float(roc_auc_score(labels, [r[field] for r in train]))
                            for field in ("age_a", "age_b", "age_gap")},
                        "negative_endpoint_b_age_error_counts": dict(Counter(
                            abs(n["age_b"] - p["age_b"]) for n, p in zip(negatives, positive, strict=True)))}
        output[name] = combined
    if heldouts[0] != heldouts[1]:
        raise ValueError("canonical heldout rows differ across arms")
    if people["LOW"] & people["CROSS"] or faces["LOW"] & faces["CROSS"]:
        raise ValueError("cross-arm recorded-person/image overlap")
    for field in ("positive_pairs", "negative_pairs", "positive_images", "all_train_images", "recorded_people"):
        if report["LOW"][field] != report["CROSS"][field]:
            raise ValueError("unequal matched full budgets")
    return output, {"arms": report, "full_train_image_budgets_equal": True,
                    "positive_and_heldout_rows_preserved": True, "publication_ready": False,
                    "training_completed": False, "training_identity_independence": "unverified",
                    "scope": "restricted negative-pool preparation, not causal-source evidence",
                    "unmatched": ["age distributions between arms", "quality/pose/context", "generic-source control"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", type=Path, default=PROJECT_ROOT / "data/interim/matched_agegap_arms")
    p.add_argument("--clusters", type=Path, default=PROJECT_ROOT / "data/processed/person_clusters.jsonl")
    p.add_argument("--canonical", type=Path, default=PROJECT_ROOT / "data/processed/pairs.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    inputs = [args.arms / "low_arm.jsonl", args.arms / "cross_arm.jsonl", args.clusters, args.canonical,
              Path(__file__), PROJECT_ROOT / "scripts/build_matched_agegap_arms.py",
              PROJECT_ROOT / "scripts/audit_matched_arm_image_budget.py",
              PROJECT_ROOT / "src/age_gap/common/io.py", PROJECT_ROOT / "src/age_gap/common/manifest.py"]
    before = [file_record(path) for path in inputs]
    groups = {}
    for row in read_jsonl(args.clusters):
        key, value = str(row["identity_group_id"]), str(row["person_id"])
        if key in groups and groups[key] != value:
            raise ValueError("contradictory cluster mapping")
        groups[key] = value
    known = {tuple(sorted((r["face_a"], r["face_b"]))) for r in read_jsonl(args.canonical) if r.get("label") == 1}
    arms, summary = restrict_arms({name: list(read_jsonl(path)) for name, path in
                                zip(("LOW", "CROSS"), inputs[:2], strict=True)}, groups,
                                seed=args.seed, known_positive_pairs=known)
    if not args.execute:
        print(json.dumps({"status": "planned", **summary}))
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("new output directory required; original arms preserved")
    if before != [file_record(path) for path in inputs]:
        raise ValueError("inputs changed during construction")
    args.out.mkdir(parents=True, exist_ok=True)
    private = args.out / "private"
    private.mkdir()
    outputs = [private / "low_arm.jsonl", private / "cross_arm.jsonl", args.out / "summary.json"]
    for path, name in zip(outputs[:2], ("LOW", "CROSS"), strict=True):
        write_jsonl(path, arms[name])
    outputs[2].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = write_experiment_manifest(args.out / "summary.manifest.json",
        experiment="restricted-positive-image-low-cross-arms", inputs=inputs, outputs=outputs,
        parameters={"seed": args.seed, "bin_years": 10, "max_age_error": 1,
                    "pool": "selected positive images only", "original_positive_and_heldout_rows_unchanged": True},
        metrics=summary)
    if before != [file_record(path) for path in inputs] or json.loads(manifest.read_text())["inputs"] != before:
        manifest.unlink()
        raise ValueError("inputs changed while writing completed manifest")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
