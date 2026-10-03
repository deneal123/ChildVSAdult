"""Exact-age partial shuffled supervision control; no GPU or image processing.

Positive B endpoint tokens are permuted within exact-age strata. A token can
stay in its original row, or move only to a different recorded person. Fixed
negatives and heldout rows are unchanged. Retained genuine rows are explicit:
this is a PARTIAL label-noise intervention, not full random co-occurrence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from age_gap.common.io import PROJECT_ROOT, read_jsonl, write_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest


def digest_rows(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_control(rows, groups, *, seed=42, attempts=64, require_full=False, known_positive_pairs=None):
    if attempts < 1:
        raise ValueError("positive finite attempt budget required")
    original = copy.deepcopy(rows)
    positive = [r for r in rows if r.get("split") == "train" and r.get("label") == 1]
    negative = [r for r in rows if r.get("split") == "train" and r.get("label") == 0]
    if not positive or len(positive) != len(negative):
        raise ValueError("nonempty balanced train required")
    if len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("unique pair IDs required")
    def person(row, side):
        group = str(row[f"identity_group_{side}"])
        if group not in groups:
            raise ValueError("missing recorded-person mapping")
        return groups[group]
    face_people = {}
    # Unknown identities cannot be silently treated as distinct people.
    for row in rows:
        if row.get("split") not in {"train", "val", "test"} or row.get("label") not in (0, 1):
            raise ValueError("valid split/label required")
        for side in ("a", "b"):
            identity = person(row, side)
            face = row[f"face_{side}"]
            if face in face_people and face_people[face] != identity:
                raise ValueError("same face has conflicting recorded persons")
            face_people[face] = identity
        if row.get("split") == "train":
            if row["face_a"] == row["face_b"]:
                raise ValueError("self-image pair")
            if (person(row, "a") == person(row, "b")) != (row["label"] == 1):
                raise ValueError("recorded identities contradict original train labels")
    train_people = {person(r, s) for r in (*positive, *negative) for s in ("a", "b")}
    heldout = [r for r in rows if r["split"] in {"val", "test"}]
    if any(not any(r["split"] == split for r in heldout) for split in ("val", "test")):
        raise ValueError("nonempty val and test required")
    if train_people & {person(r, s) for r in heldout for s in ("a", "b")}:
        raise ValueError("train-heldout recorded-person overlap")
    if ({r[f"face_{s}"] for r in (*positive, *negative) for s in ("a", "b")}
            & {r[f"face_{s}"] for r in heldout for s in ("a", "b")}):
        raise ValueError("train-heldout image overlap")
    forbidden = {tuple(sorted((r["face_a"], r["face_b"]))) for r in negative}
    original_pairs = [tuple(sorted((r["face_a"], r["face_b"]))) for r in positive]
    if len(set(original_pairs)) != len(original_pairs) or set(original_pairs) & forbidden:
        raise ValueError("duplicate/contradictory original positive pairs")
    known_positive_pairs = set(original_pairs) if known_positive_pairs is None else set(known_positive_pairs) | set(original_pairs)
    strata = defaultdict(list)
    for i, row in enumerate(positive):
        if any(not isinstance(row[f"age_{s}"], int) for s in ("a", "b")):
            raise ValueError("integer endpoint ages required")
        if row["age_gap"] != abs(row["age_a"] - row["age_b"]):
            raise ValueError("inconsistent original age gap")
        strata[row["age_b"]].append(i)
    rng = np.random.default_rng(seed)
    permutation, reports = np.arange(len(positive)), []
    for age, indices in sorted(strata.items()):
        n = len(indices)
        cost = np.full((n, n), np.inf)
        for i, target in enumerate(indices):
            for j, donor in enumerate(indices):
                left, right = positive[target], positive[donor]
                if (left["face_a"] == right["face_b"]
                        or tuple(sorted((left["face_a"], right["face_b"]))) in forbidden):
                    continue
                if i == j:
                    cost[i, j] = 1  # retain original genuine pair, explicitly counted
                elif (person(left, "a") != person(right, "b")
                      and tuple(sorted((left["face_a"], right["face_b"]))) not in known_positive_pairs):
                    cost[i, j] = 0
        for _ in range(attempts):
            # Total perturbation <1: number of retained genuine pairs is minimized first.
            ii, jj = linear_sum_assignment(cost + rng.random((n, n)) / (2 * n))
            retained = int(cost[ii, jj].sum())
            edges = [tuple(sorted((positive[indices[i]]["face_a"], positive[indices[j]]["face_b"])))
                     for i, j in zip(ii, jj, strict=True)]
            if len(set(edges)) == len(edges):
                for i, j in zip(ii, jj, strict=True):
                    permutation[indices[i]] = indices[j]
                reports.append({"age_b": age, "pairs": n, "retained_genuine": retained})
                break
        else:
            raise ValueError("finite assignment attempts exhausted; no age relaxation or budget shrink")
    shuffled = []
    for i, donor in enumerate(permutation):
        row = copy.deepcopy(positive[i])
        row["pair_type"] = "partial_shuffled_noise_control"
        row["noise_control_original_pair_id"] = row["pair_id"]
        row["pair_id"] = f"partial_noise_{seed}_{i:04d}"
        row["face_b"] = positive[donor]["face_b"]
        row["identity_group_b"] = positive[donor]["identity_group_b"]
        row["synthetic_label_noise"] = bool(i != donor)
        shuffled.append(row)
    count = sum(r["synthetic_label_noise"] for r in shuffled)
    if require_full and count != len(positive):
        raise ValueError("full shuffle infeasible under declared constraints; partial is not full")
    output = [*shuffled, *copy.deepcopy(negative), *copy.deepcopy(heldout)]
    for side in ("a", "b"):
        if Counter(r[f"face_{side}"] for r in positive) != Counter(r[f"face_{side}"] for r in shuffled):
            raise AssertionError("positive endpoint image multiplicity changed")
    if digest_rows(rows) != digest_rows(original):
        raise AssertionError("input mutation")
    edges = [tuple(sorted((r["face_a"], r["face_b"]))) for r in shuffled]
    if len(set(edges)) != len(edges) or set(edges) & forbidden:
        raise ValueError("duplicate shuffled positive or fixed-negative overlap")
    summary = {"control": "PARTIAL exact-age B permutation, positive labels intentionally retained",
               "positive_pairs": len(positive), "negative_pairs": len(negative),
               "positive_distinct_person_noise_pairs": count,
               "retained_genuine_positive_pairs": len(positive) - count,
               "noise_fraction": count / len(positive), "full_shuffle": count == len(positive),
               "positive_images": len({r[f"face_{s}"] for r in positive for s in ("a", "b")}),
               "train_images_including_fixed_negatives": len({r[f"face_{s}"] for r in (*positive, *negative) for s in ("a", "b")}),
               "known_positive_unordered_pairs_checked": len(known_positive_pairs),
               "recorded_train_persons": len(train_people), "age_strata": reports,
               "heldout_sha256": digest_rows(heldout), "fixed_negative_sha256": digest_rows(negative),
               "exact_endpoint_ages_and_gaps_preserved_per_positive_row": True,
               "endpoint_image_multiplicities_preserved": True,
               "scope": "partial supervision perturbation on same source/images; not generic-source or random-cooccurrence evidence",
               "quality_pose_covariates_not_matched_per_pair": True,
               "recorded_person_partition_not_human_verified": True,
               "training_evaluation_completed": False, "publication_ready": False}
    return output, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", type=Path, default=PROJECT_ROOT / "data/interim/matched_agegap_arms/cross_arm.jsonl")
    p.add_argument("--groups", type=Path, default=PROJECT_ROOT / "data/processed/person_clusters.jsonl")
    p.add_argument("--canonical", type=Path, default=PROJECT_ROOT / "data/processed/pairs.jsonl")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/interim/partial_noise_control_20261002")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attempts", type=int, default=64)
    p.add_argument("--require-full", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    inputs = [args.arm, args.groups, args.canonical, Path(__file__)]
    before = [file_record(path) for path in inputs]
    groups = {str(r["identity_group_id"]): str(r["person_id"]) for r in read_jsonl(args.groups)}
    known = {tuple(sorted((r["face_a"], r["face_b"]))) for r in read_jsonl(args.canonical)
             if r.get("label") == 1}
    control, summary = build_control(list(read_jsonl(args.arm)), groups, seed=args.seed,
                                     attempts=args.attempts, require_full=args.require_full,
                                     known_positive_pairs=known)
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
    target, result = private / "partial_noise_arm.jsonl", args.out / "summary.json"
    write_jsonl(target, control)
    result.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = write_experiment_manifest(result.with_suffix(".manifest.json"),
        experiment="partial-exact-age-supervision-noise-control", inputs=inputs,
        outputs=[target, result], parameters={"seed": args.seed, "attempts": args.attempts,
                                            "scipy_version": version("scipy"),
                                            "numpy_version": version("numpy"),
                                            "require_full": args.require_full}, metrics=summary)
    if json.loads(manifest.read_text(encoding="utf-8"))["inputs"] != before:
        manifest.unlink()
        raise ValueError("inputs changed while writing completed manifest")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
