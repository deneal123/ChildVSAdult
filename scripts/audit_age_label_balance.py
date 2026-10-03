"""Metadata-only empirical label-conditioned age balance; not predictive validation."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from age_gap.evaluation.fgnet import _match_negatives


def diagnostic(positives, negatives):
    if not positives or len(positives) != len(negatives):
        raise ValueError("nonempty equal class sizes required")
    for row in [*positives, *negatives]:
        if len(row) != 2 or any(type(v) is not int or v < 0 for v in row):
            raise ValueError("nonnegative integer endpoint ages required")
    features = {"age_a": lambda r: r[0], "age_b": lambda r: r[1],
                "age_gap": lambda r: abs(r[0] - r[1]),
                "ordered_endpoint_ages": lambda r: tuple(r),
                "unordered_endpoint_ages": lambda r: tuple(sorted(r))}
    result = {}
    for name, feature in features.items():
        p, n = [feature(r) for r in positives], [feature(r) for r in negatives]
        pc, nc = Counter(p), Counter(n)
        keys = pc.keys() | nc.keys()
        tv = sum(abs(pc[k] - nc[k]) for k in keys) / (2 * len(p))
        result[name] = {"positive_support_size": len(pc), "negative_support_size": len(nc),
                        "pooled_support_size": len(keys), "empirical_total_variation": tv,
                        "equal_prior_in_sample_oracle_accuracy": (1 + tv) / 2,
                        "rank_auc_positive_higher": float(roc_auc_score([1] * len(p) + [0] * len(n), p + n))
                        if name in {"age_a", "age_b", "age_gap"} else None}
    errors = [max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(positives, negatives, strict=True)]
    return {"positive_pairs": len(positives), "negative_pairs": len(negatives),
            "matched_block_max_endpoint_error_histogram": dict(sorted(Counter(errors).items())),
            "features": result}


def row_ages(row):
    ages = (row.get("age_a"), row.get("age_b"))
    if any(type(v) is not int or v < 0 for v in ages) or row.get("age_gap") != abs(ages[0] - ages[1]):
        raise ValueError("invalid endpoint ages or inconsistent observed gap")
    return ages


def paired_rows(rows, *, explicit_targets):
    if any(type(r.get("label")) is not int or r["label"] not in {0, 1} for r in rows):
        raise ValueError("binary integer labels required")
    positives = [r for r in rows if r["label"] == 1]
    negatives = [r for r in rows if r["label"] == 0]
    if not positives or len(positives) != len(negatives):
        raise ValueError("nonempty balanced rows required")
    ids = [r["pair_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate pair ID")
    if explicit_targets:
        mapping = {r["pair_id"]: r for r in positives}
        targets = [r.get("matched_target_pair_id") for r in negatives]
        if len(set(targets)) != len(targets) or set(targets) != set(mapping):
            raise ValueError("one-to-one positive target binding required")
        positives = [mapping[target] for target in targets]
    elif rows != positives + negatives:
        raise ValueError("ordered positive and aligned negative blocks required")
    for p, n in zip(positives, negatives, strict=True):
        if p["face_a"] != n["face_a"] or p["age_a"] != n["age_a"]:
            raise ValueError("fixed endpoint anchor mismatch")
        if p.get("identity_group_a") != p.get("identity_group_b") or n.get("identity_group_a") == n.get("identity_group_b"):
            raise ValueError("recorded group relation contradicts label")
        if p.get("identity_group_a") is None or n.get("identity_group_a") != p["identity_group_a"] or n.get("identity_group_b") is None:
            raise ValueError("recorded anchor group binding mismatch")
        if p["face_a"] == p["face_b"] or n["face_a"] == n["face_b"]:
            raise ValueError("self pair")
    return diagnostic([row_ages(r) for r in positives], [row_ages(r) for r in negatives])


def fgnet_metadata(cache):
    # NPZ member access is lazy: do not read/decompress crops or call load_pairs.
    with np.load(cache, allow_pickle=False) as data:
        subjects, ages = data["subjects"], data["ages"]
    if subjects.ndim != 1 or ages.shape != subjects.shape or not len(ages) or ages.dtype.kind not in "iu" or subjects.dtype.kind not in "iu" or (ages < 0).any():
        raise ValueError("integer FG-NET metadata required")
    groups = {}
    for index, subject in enumerate(subjects.tolist()):
        groups.setdefault(subject, []).append(index)
    positives = [(a, b, abs(int(ages[a]) - int(ages[b]))) for indices in groups.values()
                 for offset, a in enumerate(indices) for b in indices[offset + 1:]]
    matched = _match_negatives(subjects, ages, positives, tolerance=2, seed=42)
    result = {}
    for name, minimum in (("overall", 0), ("source_positive_gap_25plus", 25)):
        p, n = [], []
        for pos_index, left, right, gap, error, source_gap in matched:
            a, b, positive_gap = positives[pos_index]
            if source_gap != positive_gap or gap != source_gap or subjects[left] == subjects[right]:
                raise ValueError("FG-NET matcher metadata mismatch")
            if left != a and right != b or error > 2:
                raise ValueError("FG-NET endpoint anchor or tolerance mismatch")
            if source_gap >= minimum:
                p.append((int(ages[a]), int(ages[b])))
                n.append((int(ages[left]), int(ages[right])))
        result[name] = diagnostic(p, n)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    root = PROJECT_ROOT
    paths = [root / "data/interim/restricted_matched_arms_20261003/private/low_arm.jsonl",
             root / "data/interim/restricted_matched_arms_20261003/private/cross_arm.jsonl",
             root / "data/processed/experiments/pairs_internal_endpoint_age_matched.jsonl",
             root / "data/external/fgnet_crops.npz",
             root / "data/interim/restricted_matched_arms_20261003/summary.manifest.json",
             root / "metrics/internal_endpoint_age_matched.manifest.json",
             root / "metrics/fgnet_endpoint_subject_stats.manifest.json", Path(__file__),
             root / "src/age_gap/evaluation/fgnet.py", root / "src/age_gap/common/io.py",
             root / "src/age_gap/common/manifest.py"]
    before = [file_record(path) for path in paths]
    references = [json.loads(path.read_text(encoding="utf-8")) for path in paths[4:7]]
    for index, manifest, field in ((0, references[0], "outputs"), (1, references[0], "outputs"),
                                   (2, references[1], "outputs"), (3, references[2], "inputs")):
        if before[index] not in manifest[field]:
            raise ValueError("reference manifest binding mismatch")
    result = {name: paired_rows([r for r in read_jsonl(path) if r["split"] == "train"], explicit_targets=True)
              for name, path in zip(("LOW_train", "CROSS_train"), paths[:2], strict=True)}
    result["internal_test"] = paired_rows(list(read_jsonl(paths[2])), explicit_targets=False)
    result["FGNET"] = fgnet_metadata(paths[3])
    result["scope"] = "finite recorded metadata only; no image processing or protocol changes"
    result["inference"] = "TV and oracle are in-sample empirical descriptors, not held-out prediction, population estimates or causal effects; sparse/dependent tuples can overfit"
    result["age_shortcut_absence_proven"] = False
    result["publication_ready"] = False
    if before != [file_record(path) for path in paths]:
        raise ValueError("inputs changed during audit")
    args.out.mkdir(parents=True)
    output = args.out / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = output.with_suffix(".manifest.json")
    write_experiment_manifest(target, experiment="empirical-age-label-balance", parameters={"seed": None,
        "FGNET_protocol_seed": 42, "FGNET_tolerance": 2, "ci": "not estimated; descriptive finite sample"},
        metrics=result, inputs=paths, outputs=[output])
    if json.loads(target.read_text(encoding="utf-8"))["inputs"] != before:
        target.unlink()
        raise ValueError("manifest inputs changed; completed marker withdrawn")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
