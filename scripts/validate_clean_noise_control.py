"""Strict shared-person clean/partial-noise contract, not LOW/CROSS splitting.

Only explicitly flagged training positives may contradict recorded identities;
benchmark and heldout labels are never interpreted as intentionally noisy.
"""
from __future__ import annotations

from collections import Counter


def validate_control(clean, noisy, groups, *, known_genuine_edges=()):
    def positive(rows):
        return [r for r in rows if r.get("split") == "train" and r.get("label") == 1]

    def fixed(rows):
        return [r for r in rows if r.get("split") != "train" or r.get("label") == 0]

    face_people, face_ages = {}, {}
    def person(row, side):
        key, face = row[f"identity_group_{side}"], row[f"face_{side}"]
        if key not in groups or not isinstance(groups[key], str) or not groups[key]:
            raise ValueError("missing or invalid recorded-person mapping")
        identity = groups[key]
        if face in face_people and face_people[face] != identity:
            raise ValueError("conflicting face-person mapping")
        face_people[face] = identity
        if row["split"] == "train":
            age = row.get(f"age_{side}")
            if type(age) is not int or age < 0:
                raise ValueError("nonnegative integer train ages required")
            if face in face_ages and face_ages[face] != age:
                raise ValueError("conflicting train-face ages")
            face_ages[face] = age
        return identity

    for rows in (clean, noisy):
        if not rows or len({r["pair_id"] for r in rows}) != len(rows):
            raise ValueError("nonempty rows and unique pair IDs required")
        for r in rows:
            if (r.get("split") not in {"train", "val", "test"} or type(r.get("label")) is not int
                    or r["label"] not in (0, 1)):
                raise ValueError("binary labels and valid splits required")
            person(r, "a")
            person(r, "b")
            if r["split"] == "train":
                if r["face_a"] == r["face_b"]:
                    raise ValueError("self-image train pair")
                if r.get("age_gap") != abs(r["age_a"] - r["age_b"]):
                    raise ValueError("inconsistent train age gap")
        if {r["split"] for r in rows if r["split"] != "train"} != {"val", "test"}:
            raise ValueError("nonempty val and test required")
    cp, np = positive(clean), positive(noisy)
    negatives = [r for r in clean if r["split"] == "train" and r["label"] == 0]
    if not cp or len(cp) != len(negatives) or len(np) != len(cp) or fixed(clean) != fixed(noisy):
        raise ValueError("balanced matched positive budget and identical fixed negatives/heldout required")
    for r in (*cp, *negatives):
        if (person(r, "a") == person(r, "b")) != (r["label"] == 1):
            raise ValueError("clean train labels contradict recorded persons")
        if r.get("synthetic_label_noise"):
            raise ValueError("clean input must not contain intentional noise")
    positive_edges = {tuple(sorted((r["face_a"], r["face_b"]))) for r in cp}
    negative_edges = {tuple(sorted((r["face_a"], r["face_b"]))) for r in negatives}
    if len(positive_edges) != len(cp) or len(negative_edges) != len(negatives) or positive_edges & negative_edges:
        raise ValueError("duplicate or contradictory clean edges")
    forbidden = set(known_genuine_edges) | positive_edges
    count = 0
    for original, row in zip(cp, np, strict=True):
        if (row.get("noise_control_original_pair_id") != original["pair_id"]
                or row.get("pair_type") != "partial_shuffled_noise_control"
                or type(row.get("synthetic_label_noise")) is not bool):
            raise ValueError("explicit row-linked partial-noise metadata required")
        allowed = {"pair_id", "pair_type", "face_b", "identity_group_b",
                   "noise_control_original_pair_id", "synthetic_label_noise"}
        if {k: v for k, v in original.items() if k not in allowed} != {k: v for k, v in row.items() if k not in allowed}:
            raise ValueError("noise changed non-intervention row fields")
        changed = row["face_b"] != original["face_b"]
        distinct = person(row, "a") != person(row, "b")
        if row["synthetic_label_noise"] != changed or changed != distinct:
            raise ValueError("noise flag must match changed-B different-person positive")
        if not changed and row["identity_group_b"] != original["identity_group_b"]:
            raise ValueError("retained genuine group changed")
        if changed and tuple(sorted((row["face_a"], row["face_b"]))) in forbidden:
            raise ValueError("noisy positive collides with known genuine edge")
        count += int(changed)
    noise_edges = [tuple(sorted((r["face_a"], r["face_b"]))) for r in np]
    if len(set(noise_edges)) != len(np) or set(noise_edges) & negative_edges:
        raise ValueError("duplicate noise edges or fixed-negative overlap")
    for side in ("a", "b"):
        if Counter(r[f"face_{side}"] for r in cp) != Counter(r[f"face_{side}"] for r in np):
            raise ValueError("positive endpoint image multiplicities changed")
    stats = []
    for rows in (clean, noisy):
        train = [r for r in rows if r["split"] == "train"]
        heldout = [r for r in rows if r["split"] != "train"]
        people = {person(r, s) for r in train for s in ("a", "b")}
        faces = {r[f"face_{s}"] for r in train for s in ("a", "b")}
        if people & {person(r, s) for r in heldout for s in ("a", "b")}:
            raise ValueError("train-heldout recorded-person overlap")
        if faces & {r[f"face_{s}"] for r in heldout for s in ("a", "b")}:
            raise ValueError("train-heldout image overlap")
        stats.append((people, faces))
    if stats[0] != stats[1]:
        raise ValueError("clean/noise full image or recorded-person budgets differ")
    if not 0 < count < len(cp):
        raise ValueError("partial, not zero or full, noise required")
    return {"positive_pairs": len(cp), "negative_pairs": len(negatives),
            "noisy_positive_pairs": count, "retained_genuine_positive_pairs": len(cp) - count,
            "noise_fraction_among_positive_labels": count / len(cp),
            "noise_fraction_among_all_train_labels": count / (2 * len(cp)),
            "all_train_images": len(stats[0][1]), "recorded_train_people": len(stats[0][0]),
            "fixed_negatives_and_heldout_unchanged": True, "endpoint_image_multiplicities_unchanged": True,
            "training_completed": False, "publication_ready": False,
            "scope": "shared-source supervision perturbation, not full co-occurrence or causal source evidence"}
