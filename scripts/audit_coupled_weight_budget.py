"""Audit candidate positive/negative-block weights; never modify or train arms."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from age_gap.common.io import PROJECT_ROOT, read_jsonl, write_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.audit_nuisance_overlap import VIEWS, effective_size


def blocks(rows, groups):
    if len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("unique pair IDs required")
    if any(type(r["label"]) is not int or r["label"] not in {0, 1} for r in rows):
        raise ValueError("binary integer labels required")
    positives = [r for r in rows if r["label"] == 1]
    negative = {r.get("matched_target_pair_id"): r for r in rows if r["label"] == 0}
    if (
        not positives
        or len(rows) != 2 * len(positives)
        or set(negative) != {r["pair_id"] for r in positives}
    ):
        raise ValueError("one-to-one explicit positive/negative targets required")
    output = []
    face_ages = {}
    for positive in positives:
        n = negative[positive["pair_id"]]
        for row in (positive, n):
            if row.get("split") != "train":
                raise ValueError("train blocks only")
            if any(
                type(row.get(f"age_{s}")) is not int or row[f"age_{s}"] < 0 for s in ("a", "b")
            ) or row.get("age_gap") != abs(row["age_a"] - row["age_b"]):
                raise ValueError("genuine endpoint ages and actual gap required")
            if row["face_a"] == row["face_b"]:
                raise ValueError("self image pair")
            for side in ("a", "b"):
                face, age = row[f"face_{side}"], row[f"age_{side}"]
                if face in face_ages and face_ages[face] != age:
                    raise ValueError("conflicting ages for the same image")
                face_ages[face] = age
            pa, pb = (groups[row[f"identity_group_{s}"]] for s in ("a", "b"))
            if not pa or not pb or (pa == pb) != bool(row["label"]):
                raise ValueError("recorded person relation contradicts label")
        if (
            positive["face_a"] != n["face_a"]
            or positive["age_a"] != n["age_a"]
            or positive["identity_group_a"] != n["identity_group_a"]
            or abs(positive["age_b"] - n["age_b"]) > 1
        ):
            raise ValueError("explicit target anchor/age tolerance mismatch")
        output.append((positive, n))
    return output


def weighted_age_balance(pairs, weights):
    w = np.asarray(weights, float)
    if w.shape != (len(pairs),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("nonnegative finite block weights with positive total required")
    functions = {
        "age_a": lambda r: r["age_a"],
        "age_b": lambda r: r["age_b"],
        "age_gap": lambda r: r["age_gap"],
        "ordered_endpoint_ages": lambda r: (r["age_a"], r["age_b"]),
        "unordered_endpoint_ages": lambda r: tuple(sorted((r["age_a"], r["age_b"]))),
    }
    result = {}
    for name, feature in functions.items():
        masses = []
        observations = []
        for side in (0, 1):
            mass = Counter()
            values = [feature(pair[side]) for pair in pairs]
            for key, weight in zip(values, w, strict=True):
                mass[key] += float(weight)
            masses.append(mass)
            observations.append(values)
        tv = float(
            sum(abs(masses[0][key] - masses[1][key]) for key in masses[0].keys() | masses[1].keys())
            / (2 * w.sum())
        )
        result[name] = {
            "empirical_weighted_total_variation": tv,
            "equal_prior_weighted_in_sample_oracle_accuracy": (1 + tv) / 2,
            "weighted_rank_auc_positive_higher": float(
                roc_auc_score(
                    [1] * len(w) + [0] * len(w),
                    observations[0] + observations[1],
                    sample_weight=np.r_[w, w],
                )
            )
            if name in {"age_a", "age_b", "age_gap"}
            else None,
        }
    return result


def exposure_mass(pairs, weights, groups):
    images, people = Counter(), Counter()
    image_person = {}
    for (positive, negative), weight in zip(pairs, weights, strict=True):
        for row in (positive, negative):
            for side in ("a", "b"):
                face, person = row[f"face_{side}"], groups[row[f"identity_group_{side}"]]
                if face in image_person and image_person[face] != person:
                    raise ValueError("conflicting face-person mapping")
                image_person[face] = person
                images[face] += float(weight)
                people[person] += float(weight)

    def summarize(masses):
        values = np.array(list(masses.values()))
        return {
            "distinct": len(values),
            "positive_mass_units": int((values > 0).sum()),
            "total_loss_weighted_endpoint_mass": float(values.sum()),
            "mass_effective_size": effective_size(values),
            "maximum_endpoint_mass_fraction": float(values.max() / values.sum()),
        }

    return {
        "images": summarize(images),
        "recorded_people": summarize(people),
        "scope": "hypothetical loss-weighted endpoint mass, not actual image presentations or independent sample size",
    }


def audit(arms, groups, diagnostic_weights):
    train = {arm: [r for r in arms[arm] if r["split"] == "train"] for arm in ("LOW", "CROSS")}
    heldout = [[r for r in arms[arm] if r["split"] != "train"] for arm in ("LOW", "CROSS")]
    if heldout[0] != heldout[1]:
        raise ValueError("shared heldout changed")
    paired = {arm: blocks(train[arm], groups) for arm in train}
    source = {p["pair_id"]: (arm, p) for arm in paired for p, _ in paired[arm]}
    if len(source) != sum(map(len, paired.values())):
        raise ValueError("cross-arm positive ID collision")
    weights_by_view = {view: {} for view in VIEWS}
    for row in diagnostic_weights:
        view, pair_id = row["view"], row["pair_id"]
        if view not in VIEWS or pair_id not in source or pair_id in weights_by_view[view]:
            raise ValueError("unknown/duplicate diagnostic weight key")
        arm, positive = source[pair_id]
        probability, raw, weight = (
            row[k] for k in ("p_cross_oof", "raw_overlap_weight", "normalized_diagnostic_weight")
        )
        if (
            any(
                type(v) not in (int, float) or not np.isfinite(v)
                for v in (probability, raw, weight)
            )
            or not 0 <= probability <= 1
            or raw < 0
            or weight < 0
            or not np.isclose(
                raw, probability if arm == "LOW" else 1 - probability, rtol=1e-12, atol=0
            )
            or row["recorded_person"] != groups[positive["identity_group_a"]]
        ):
            raise ValueError("diagnostic weight/person/raw-probability mismatch")
        weights_by_view[view][pair_id] = (float(weight), float(raw))
    result, private = {}, []
    for view, mapping in weights_by_view.items():
        if set(mapping) != set(source):
            raise ValueError("complete positive weight mapping required for every view")
        reports = {}
        for arm in paired:
            pairs = paired[arm]
            weights = np.array([mapping[p["pair_id"]][0] for p, _ in pairs])
            raw = np.array([mapping[p["pair_id"]][1] for p, _ in pairs])
            if raw.sum() <= 0 or not np.allclose(weights, raw / raw.mean(), rtol=1e-12, atol=0):
                raise ValueError("within-arm normalization mismatch")
            reports[arm] = {
                "positive_pairs": len(pairs),
                "negative_pairs": len(pairs),
                "class_loss_mass": {
                    "positive": float(weights.sum()),
                    "negative": float(weights.sum()),
                },
                "uniform_age_balance": weighted_age_balance(pairs, np.ones(len(pairs))),
                "coupled_candidate_age_balance": weighted_age_balance(pairs, weights),
                "uniform_endpoint_mass": exposure_mass(pairs, np.ones(len(pairs)), groups),
                "coupled_candidate_endpoint_mass": exposure_mass(pairs, weights, groups),
            }
            for (positive, negative), weight in zip(pairs, weights, strict=True):
                for row in (positive, negative):
                    private.append(
                        {
                            "view": view,
                            "arm": arm,
                            "pair_id": row["pair_id"],
                            "target_positive_pair_id": positive["pair_id"],
                            "label": row["label"],
                            "candidate_loss_weight": float(weight),
                        }
                    )
        result[view] = reports
    return {
        "views": result,
        "execution_complete": True,
        "training_weights_ready": False,
        "publication_ready": False,
        "recognition_outcomes_used": False,
        "candidate_objective": "L=(1/(2N))*sum_i w_i*(ell(positive_i)+ell(negative_i)); mean(w)=1 within each arm",
        "batch_rule": "uniform block sampling; fixed globally normalized weights, never re-normalize by batch weight sum",
        "scope": "audit of a candidate paired loss objective; not balanced presentations or completed training",
        "uncertainty": "empirical weighted age TV/oracle only; no held-out shortcut accuracy or confidence interval",
        "remaining": [
            "full positive/negative joint quality balance",
            "age-error sensitivity after weighting",
            "BatchNorm and actual presentation policy",
            "fresh source-bound three-seed training",
        ],
    }, private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    root = PROJECT_ROOT
    prerequisite = root / "metrics/nuisance_overlap_20261003/summary.manifest.json"
    native = json.loads(prerequisite.read_text(encoding="utf-8"))
    if native["experiment"] != "recorded-person-crossfit-nuisance-overlap":
        raise ValueError("unexpected prerequisite experiment")
    records = native["inputs"] + native["outputs"]
    paths = [
        Path(r["path"]) if Path(r["path"]).is_absolute() else root / r["path"] for r in records
    ]
    if [file_record(p) for p in paths] != records:
        raise ValueError("prerequisite direct inputs/outputs changed")
    required = [
        root / "data/interim/restricted_matched_arms_20261003/private" / f"{arm.lower()}_arm.jsonl"
        for arm in ("LOW", "CROSS")
    ]
    required.extend(
        [
            root / "data/processed/person_clusters.jsonl",
            root / "metrics/nuisance_overlap_20261003/private/diagnostic_weights.jsonl",
        ]
    )
    if any(file_record(path) not in records for path in required):
        raise ValueError("actual consumed source/weights not bound by prerequisite")
    inputs = [*paths, prerequisite, Path(__file__)]
    before = [file_record(p) for p in inputs]
    arms = {
        arm: list(
            read_jsonl(
                root
                / "data/interim/restricted_matched_arms_20261003/private"
                / f"{arm.lower()}_arm.jsonl"
            )
        )
        for arm in ("LOW", "CROSS")
    }
    groups = {}
    for row in read_jsonl(root / "data/processed/person_clusters.jsonl"):
        key, person = row["identity_group_id"], row["person_id"]
        if key in groups and groups[key] != person:
            raise ValueError("conflicting group-person mapping")
        groups[key] = person
    result, private = audit(
        arms,
        groups,
        list(
            read_jsonl(root / "metrics/nuisance_overlap_20261003/private/diagnostic_weights.jsonl")
        ),
    )
    if before != [file_record(p) for p in inputs]:
        raise ValueError("inputs changed during candidate objective audit")
    (args.out / "private").mkdir(parents=True)
    output, ledger = args.out / "summary.json", args.out / "private/candidate_pair_weights.jsonl"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_jsonl(ledger, private)
    target = args.out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="coupled-pair-weight-objective-budget-audit",
        parameters={"seed": 42, "weight_views": list(VIEWS), "training_executed": False},
        metrics=result,
        inputs=inputs,
        outputs=[output, ledger],
    )
    if before != json.loads(target.read_text(encoding="utf-8"))["inputs"]:
        target.unlink()
        raise ValueError("inputs changed during manifest write; marker withdrawn")
    print(
        json.dumps(
            {
                view: {
                    arm: {
                        "maximum_age_tv": max(
                            v["empirical_weighted_total_variation"]
                            for v in row["coupled_candidate_age_balance"].values()
                        ),
                        "recorded_person_mass_ess": row["coupled_candidate_endpoint_mass"][
                            "recorded_people"
                        ]["mass_effective_size"],
                    }
                    for arm, row in report.items()
                }
                for view, report in result["views"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
