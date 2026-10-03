"""Read-only descriptive age/quality balance of restricted LOW/CROSS training arms."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest

QUALITY_FIELDS = ("image_width", "image_height", "face_width", "face_height", "blur_var",
                  "det_score", "face_quality_score", "pose_yaw_proxy_abs", "pose_roll_rad_abs")
AGE_FIELDS = ("age_a", "age_b", "age_gap", "age_younger", "age_older")


def value(row, field):
    raw = row.get(field.removesuffix("_abs"))
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        raise ValueError("finite numeric metadata or explicit null required")
    number = float(raw)
    if field.endswith("_abs"):
        return abs(number)
    if number < 0:
        raise ValueError("nonnegative quality metadata required")
    return number


def distribution(values):
    observed = np.asarray([v for v in values if v is not None], dtype=float)
    return {"n_total": len(values), "n_observed": len(observed), "n_missing": len(values) - len(observed),
            "minimum": float(observed.min()) if len(observed) else None,
            "maximum": float(observed.max()) if len(observed) else None,
            "mean": float(observed.mean()) if len(observed) else None,
            "population_sd": float(observed.std(ddof=0)) if len(observed) else None,
            "p05_p25_median_p75_p95": np.percentile(observed, [5, 25, 50, 75, 95]).tolist() if len(observed) else None}


def compare(low, cross):
    a = np.asarray([v for v in low if v is not None], dtype=float)
    b = np.asarray([v for v in cross if v is not None], dtype=float)
    result = {"LOW": distribution(low), "CROSS": distribution(cross),
              "cross_minus_low_mean": None, "smd_cross_minus_low": None,
              "smd_status": "missing arm observations", "arm_auc_cross_higher": None,
              "ecdf_max_distance": None, "observed_range_overlap": None,
              "LOW_fraction_within_CROSS_observed_range": None,
              "CROSS_fraction_within_LOW_observed_range": None}
    if not len(a) or not len(b):
        return result
    difference = float(b.mean() - a.mean())
    pooled = math.sqrt(float((a.var(ddof=0) + b.var(ddof=0)) / 2))
    result.update(cross_minus_low_mean=difference,
                  smd_cross_minus_low=difference / pooled if pooled else (0.0 if difference == 0 else None),
                  smd_status="finite" if pooled else ("both constants equal" if difference == 0 else "different constants; zero pooled SD"),
                  arm_auc_cross_higher=float(roc_auc_score([0] * len(a) + [1] * len(b), np.r_[a, b])))
    points = np.unique(np.r_[a, b])
    result["ecdf_max_distance"] = float(np.max(np.abs(
        np.searchsorted(np.sort(a), points, side="right") / len(a)
        - np.searchsorted(np.sort(b), points, side="right") / len(b))))
    lo, hi = float(max(a.min(), b.min())), float(min(a.max(), b.max()))
    result["observed_range_overlap"] = [lo, hi] if lo <= hi else None
    result["LOW_fraction_within_CROSS_observed_range"] = float(((a >= b.min()) & (a <= b.max())).mean())
    result["CROSS_fraction_within_LOW_observed_range"] = float(((b >= a.min()) & (b <= a.max())).mean())
    return result


def class_auc(rows, score):
    observed = [(r["label"], score(r)) for r in rows]
    usable = [(label, val) for label, val in observed if val is not None]
    labels = [label for label, _ in usable]
    return {"n_total": len(rows), "n_observed": len(usable), "n_missing": len(rows) - len(usable),
            "observed_positive": labels.count(1), "observed_negative": labels.count(0),
            "auc_positive_higher": float(roc_auc_score(labels, [val for _, val in usable])) if set(labels) == {0, 1} else None}


def balance(arms, groups, quality):
    if set(arms) != {"LOW", "CROSS"}:
        raise ValueError("exactly LOW/CROSS required")
    positive, train, images, face_ages, faces_people, arm_people = {}, {}, {}, {}, {}, {}
    heldouts = []
    for name in ("LOW", "CROSS"):
        rows = arms[name]
        if len({r["pair_id"] for r in rows}) != len(rows):
            raise ValueError("unique pair IDs required within each arm")
        if any(type(r.get("label")) is not int or r["label"] not in (0, 1)
               or r.get("split") not in {"train", "val", "test"} for r in rows):
            raise ValueError("valid labels/splits required")
        train[name] = [r for r in rows if r["split"] == "train"]
        positive[name] = [r for r in train[name] if r["label"] == 1]
        if not positive[name] or len(train[name]) != 2 * len(positive[name]):
            raise ValueError("nonempty balanced training required")
        heldout = [r for r in rows if r["split"] != "train"]
        if {r["split"] for r in heldout} != {"val", "test"}:
            raise ValueError("both heldout splits required")
        heldouts.append(heldout)
        face_ages[name], faces_people[name] = {}, {}
        for row in train[name]:
            if any(type(row.get(f"age_{s}")) is not int or row[f"age_{s}"] < 0 for s in ("a", "b")):
                raise ValueError("nonnegative integer caption ages required")
            gap = abs(row["age_a"] - row["age_b"])
            if row.get("age_gap") != gap:
                raise ValueError("age gap differs from actual endpoint ages")
            if row["label"] == 1 and not (1 <= gap <= 2 if name == "LOW" else gap >= 25):
                raise ValueError("positive outside intervention-defining gap range")
            people = []
            for side in ("a", "b"):
                group = row[f"identity_group_{side}"]
                if group not in groups or not groups[group]:
                    raise ValueError("missing recorded-person mapping")
                face, age, person = row[f"face_{side}"], row[f"age_{side}"], groups[group]
                if face in face_ages[name] and face_ages[name][face] != age:
                    raise ValueError("conflicting ages for same training image")
                if face in faces_people[name] and faces_people[name][face] != person:
                    raise ValueError("conflicting recorded person for same image")
                if face in quality and quality[face].get("is_usable") is not True:
                    raise ValueError("selected training image marked unusable by quality sidecar")
                face_ages[name][face], faces_people[name][face] = age, person
                people.append(person)
            if row["face_a"] == row["face_b"] or (people[0] == people[1]) != (row["label"] == 1):
                raise ValueError("self pair or recorded identities contradict label")
        images[name] = sorted(face_ages[name])
        arm_people[name] = set(faces_people[name].values())
        positive_images = {r[f"face_{s}"] for r in positive[name] for s in ("a", "b")}
        if set(images[name]) != positive_images:
            raise ValueError("negative-only training images in restricted arm")
        if any(r[f"identity_group_{s}"] not in groups for r in heldout for s in ("a", "b")):
            raise ValueError("missing heldout recorded-person mapping")
        heldout_people = {groups[r[f"identity_group_{s}"]] for r in heldout for s in ("a", "b")}
        if arm_people[name] & heldout_people:
            raise ValueError("train-heldout recorded-person overlap")
        if set(images[name]) & {r[f"face_{s}"] for r in heldout for s in ("a", "b")}:
            raise ValueError("train-heldout image overlap")
    if heldouts[0] != heldouts[1]:
        raise ValueError("heldout differs across arms")
    if set(images["LOW"]) & set(images["CROSS"]) or arm_people["LOW"] & arm_people["CROSS"]:
        raise ValueError("cross-arm image/recorded-person overlap")
    if any(len(obj["LOW"]) != len(obj["CROSS"]) for obj in (images, arm_people, train)):
        raise ValueError("unequal image/recorded-person/pair budgets")
    summaries = {}
    for name in ("LOW", "CROSS"):
        selected = images[name]
        metadata = [quality.get(face, {}) for face in selected]
        reuse = Counter(r[f"face_{s}"] for r in train[name] for s in ("a", "b"))
        summaries[name] = {"train_positive_pairs": len(positive[name]), "train_negative_pairs": len(positive[name]),
            "unique_images": len(selected), "recorded_people": len(arm_people[name]),
            "unique_photos_with_metadata": len({r["photo_id"] for r in metadata if r.get("photo_id")}),
            "images_missing_quality_rows": sum(face not in quality for face in selected),
            "image_reuse_minimum": min(reuse.values()), "image_reuse_maximum": max(reuse.values()),
            "image_reuse_mean": float(np.mean(list(reuse.values()))),
            "label_age_auc": {field: class_auc(train[name], lambda r, f=field:
                min(r["age_a"], r["age_b"]) if f == "age_younger" else
                max(r["age_a"], r["age_b"]) if f == "age_older" else r[f]) for field in AGE_FIELDS},
            "label_quality_pair_min_auc": {field: class_auc(train[name], lambda r, f=field:
                pair_min_quality(r, quality, f)) for field in QUALITY_FIELDS}}
    unique = {"caption_age": compare([face_ages["LOW"][face] for face in images["LOW"]],
                                      [face_ages["CROSS"][face] for face in images["CROSS"]])}
    for field in QUALITY_FIELDS:
        unique[field] = compare(*([value(quality.get(face, {}), field) for face in images[name]] for name in ("LOW", "CROSS")))
    exposure = {field: compare(*([value(quality.get(r[f"face_{side}"], {}), field)
                                  for r in train[name] for side in ("a", "b")] for name in ("LOW", "CROSS")))
                for field in QUALITY_FIELDS}
    exposure["caption_age"] = compare(*([r[f"age_{side}"] for r in train[name] for side in ("a", "b")]
                                         for name in ("LOW", "CROSS")))
    positive_endpoints = {field: compare(*([min(r["age_a"], r["age_b"]) if field == "age_younger" else
                 max(r["age_a"], r["age_b"]) if field == "age_older" else r[field]
                 for r in positive[name]] for name in ("LOW", "CROSS"))) for field in AGE_FIELDS}
    return {"arms": summaries, "unique_image_view": unique, "all_train_endpoint_exposure_view": exposure,
            "positive_pair_endpoint_view": positive_endpoints,
            "shared_heldout_rows": len(heldouts[0]), "budgets_equal": True,
            "quality_balanced": "not asserted; descriptive diagnostics only", "causal_source_superiority": False,
            "training_identity_independence": "unverified", "publication_ready": False,
            "inference": "no p-values or confidence intervals; reused images and recorded persons are dependent",
            "estimand": "different positive age-gap training regimes under fixed budgets; not a causal comparison of sources",
            "age_balance_constraint": "joint endpoint ages determine age gap; exact joint-age matching cannot preserve disjoint LOW/CROSS gap ranges",
            "measurement_scope": "caption ages and post-hoc quality sidecar; yaw is a five-landmark proxy, not calibrated 3D pose",
            "unmeasured": ["verified identity purity", "capture context", "scan/digital type", "collage origin", "generic-source control"],
            "scope": "read-only metadata diagnostics; no new processing of images, resampling, rebalancing or retraining"}


def pair_min_quality(row, quality, field):
    a, b = (value(quality.get(row[f"face_{s}"], {}), field) for s in ("a", "b"))
    return min(a, b) if a is not None and b is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=Path, default=PROJECT_ROOT / "data/interim/restricted_matched_arms_20261003")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    paths = [args.arms / "private/low_arm.jsonl", args.arms / "private/cross_arm.jsonl",
             args.arms / "summary.manifest.json", PROJECT_ROOT / "data/processed/person_clusters.jsonl",
             PROJECT_ROOT / "data/interim/face_quality_audit.jsonl", PROJECT_ROOT / "metrics/data_funnel.manifest.json",
             Path(__file__), PROJECT_ROOT / "src/age_gap/common/io.py", PROJECT_ROOT / "src/age_gap/common/manifest.py"]
    before = [file_record(path) for path in paths]
    arm_manifest = json.loads(paths[2].read_text(encoding="utf-8"))
    reference = json.loads(paths[5].read_text(encoding="utf-8"))
    if any(record not in arm_manifest["outputs"] for record in before[:2]) or before[4] not in reference["inputs"]:
        raise ValueError("arms or quality-sidecar reference binding mismatch")
    groups = {}
    for row in read_jsonl(paths[3]):
        group, person = row["identity_group_id"], row["person_id"]
        if group in groups and groups[group] != person:
            raise ValueError("contradictory group-person mapping")
        groups[group] = person
    arms = {name: list(read_jsonl(path)) for name, path in zip(("LOW", "CROSS"), paths[:2], strict=True)}
    selected = {r[f"face_{s}"] for rows in arms.values() for r in rows if r["split"] == "train" for s in ("a", "b")}
    quality = {}
    for row in read_jsonl(paths[4]):
        face = row["face_id"]
        if face in selected:
            if face in quality:
                raise ValueError("duplicate selected-image quality record")
            quality[face] = {field: row.get(field) for field in ("is_usable", "photo_id",
                            *(field.removesuffix("_abs") for field in QUALITY_FIELDS))}
    result = balance(arms, groups, quality)
    if before != [file_record(path) for path in paths]:
        raise ValueError("audit inputs changed")
    args.out.mkdir(parents=True)
    output = args.out / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = output.with_suffix(".manifest.json")
    write_experiment_manifest(target, experiment="restricted-matched-arm-age-quality-balance",
        parameters={"seed": None, "mode": "read-only", "smd_variance": "unweighted mean of population variances",
                    "sampling_views": ["unique train images", "all reused train endpoints", "positive pairs"],
                    "inference": "descriptive only"}, metrics=result, inputs=paths, outputs=[output])
    if json.loads(target.read_text(encoding="utf-8"))["inputs"] != before:
        target.unlink()
        raise ValueError("inputs changed while writing manifest; completed marker withdrawn")
    print(json.dumps({"arms": result["arms"], "publication_ready": False}, indent=2))


if __name__ == "__main__":
    main()
