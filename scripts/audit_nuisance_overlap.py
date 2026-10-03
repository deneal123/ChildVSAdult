"""Outcome-free, recorded-person-cross-fitted LOW/CROSS quality-overlap diagnostics.

Private positive-pair weights are diagnostic, not training-ready or causal evidence.
Existing arms, crops and model checkpoints are never modified or evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from age_gap.common.io import PROJECT_ROOT, read_jsonl, write_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.audit_matched_arm_balance import QUALITY_FIELDS, balance, value

VIEWS = ("quality_only", "younger_anchor", "older_anchor")
PROTOCOL = {
    "version": "nuisance-overlap-diagnostic-v1",
    "seed": 42,
    "folds": 5,
    "unit": "positive training pair; fold assignment by recorded person within arm",
    "quality_fields": list(QUALITY_FIELDS),
    "features": "endpoint min/max per quality field; missing endpoint count per field",
    "log1p_fields": ["image_width", "image_height", "face_width", "face_height", "blur_var"],
    "imputation": "fit-fold median, all-missing fit column uses zero with explicit missing indicators",
    "scaling": "fit-fold StandardScaler only",
    "fit_weights": "equal total weight per arm and equal total weight per recorded person within arm",
    "model": "LogisticRegression C=1, L2, lbfgs, max_iter=2000, random_state=42",
    "predictions": "out-of-fold P(CROSS); no test/validation images or recognition outcomes used",
    "overlap_weights": "LOW=e, CROSS=1-e; diagnostic normalization to unit mean within arm",
    "views": list(VIEWS),
    "age_rule": "primary excludes age; sensitivities add younger OR older age, never joint endpoints or gap",
    "balance": "weighted mean/SMD using preweight pooled population variance, weighted ECDF distance; no automatic pass cutoff",
    "scope": "measured metadata overlap only; not a causal source comparison, balanced training arms or public preregistration",
    "method_reference": "https://doi.org/10.1080/01621459.2016.1260466",
}


def person_folds(arms, people, *, n_folds=5, seed=42):
    if set(arms) != {"LOW", "CROSS"} or n_folds < 2:
        raise ValueError("two arms and at least two folds required")
    assignment = {}
    for arm in ("LOW", "CROSS"):
        ids = sorted(
            set(people[arm]),
            key=lambda person: hashlib.sha256(f"{seed}:{arm}:{person}".encode()).hexdigest(),
        )
        if len(ids) < n_folds:
            raise ValueError("at least one recorded person per arm per fold required")
        for index, person in enumerate(ids):
            if person in assignment:
                raise ValueError("recorded person occurs in both arms")
            assignment[person] = index % n_folds
    return np.array([assignment[person] for arm in ("LOW", "CROSS") for person in people[arm]])


def features(rows, quality, view):
    if view not in VIEWS:
        raise ValueError("unknown prespecified feature view")
    matrix = []
    for row in rows:
        extrema, missing = [], []
        for field in QUALITY_FIELDS:
            sides = [value(quality.get(row[f"face_{side}"], {}), field) for side in ("a", "b")]
            missing.append(sum(v is None for v in sides))
            if None in sides:
                extrema.extend([np.nan, np.nan])
            else:
                pair = [min(sides), max(sides)]
                extrema.extend(
                    np.log1p(pair).tolist() if field in PROTOCOL["log1p_fields"] else pair
                )
        if view != "quality_only":
            extrema.append((min if view == "younger_anchor" else max)(row["age_a"], row["age_b"]))
        matrix.append(extrema + missing)
    return np.asarray(matrix, dtype=float)


def crossfit(matrix, labels, people, folds):
    x, y, ids, f = (
        np.asarray(matrix, float),
        np.asarray(labels),
        np.asarray(people),
        np.asarray(folds),
    )
    if (
        x.ndim != 2
        or len(x) != len(y)
        or ids.shape != y.shape
        or f.shape != y.shape
        or y.dtype.kind not in "iu"
        or set(y) != {0, 1}
        or np.isinf(x).any()
        or set(f) != set(range(len(set(f))))
        or len(set(f)) < 2
    ):
        raise ValueError(
            "aligned finite-or-missing features, labels, people and contiguous folds required"
        )
    for person in set(ids):
        if len(set(f[ids == person])) != 1 or len(set(y[ids == person])) != 1:
            raise ValueError("person split or arm conflict")
    probabilities, diagnostics = np.empty(len(y)), []
    for fold in sorted(set(f)):
        train, test = f != fold, f == fold
        if set(y[train]) != {0, 1} or set(y[test]) != {0, 1}:
            raise ValueError("each fit and score fold must contain both arms")
        fit = x[train].copy()
        medians = np.array(
            [
                np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0.0
                for column in fit.T
            ]
        )
        fit = np.where(np.isnan(fit), medians, fit)
        predict = np.where(np.isnan(x[test]), medians, x[test])
        scaler = StandardScaler().fit(fit)
        counts = Counter(ids[train])
        weights = np.array([1 / counts[person] for person in ids[train]])
        for label in (0, 1):
            subset = y[train] == label
            weights[subset] *= len(weights) / (2 * weights[subset].sum())
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
        model.fit(scaler.transform(fit), y[train], sample_weight=weights)
        if int(model.n_iter_.max()) >= model.max_iter:
            raise ValueError("logistic model did not converge; no completed diagnostic")
        probabilities[test] = model.predict_proba(scaler.transform(predict))[:, 1]
        diagnostics.append(
            {
                "fold": int(fold),
                "fit_pairs": int(train.sum()),
                "score_pairs": int(test.sum()),
                "fit_recorded_people": len(set(ids[train])),
                "score_recorded_people": len(set(ids[test])),
                "all_missing_fit_feature_columns": int(np.isnan(x[train]).all(axis=0).sum()),
            }
        )
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("finite out-of-fold probabilities in [0,1] required")
    return probabilities, diagnostics


def effective_size(weights):
    w = np.asarray(weights, float)
    return float(w.sum() ** 2 / np.sum(w**2)) if len(w) and np.sum(w**2) else 0.0


def weighted_compare(low, cross, wl, wc):
    def observed(values, weights):
        x = np.array([np.nan if v is None else v for v in values], float)
        w = np.asarray(weights, float)
        if x.shape != w.shape or np.any(~np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("aligned nonnegative finite weights with positive total required")
        valid = np.isfinite(x)
        return (
            x[valid],
            w[valid],
            {
                "n_total": len(x),
                "n_observed": int(valid.sum()),
                "missing_weight_fraction": float(w[~valid].sum() / w.sum()),
            },
        )

    a, wa, ca = observed(low, wl)
    b, wb, cb = observed(cross, wc)
    result = {
        "LOW": ca,
        "CROSS": cb,
        "weighted_smd": None,
        "weighted_ecdf_max_distance": None,
        "weighted_mean_difference": None,
        "zero_pooled_sd": False,
    }
    if not len(a) or not len(b) or wa.sum() <= 0 or wb.sum() <= 0:
        return result
    difference = float(np.average(b, weights=wb) - np.average(a, weights=wa))
    sd = float(np.sqrt((a.var() + b.var()) / 2))
    points = np.unique(np.r_[a, b])

    def ecdf(x, w):
        order = np.argsort(x, kind="stable")
        cumulative = np.r_[0.0, np.cumsum(w[order]) / w.sum()]
        return cumulative[np.searchsorted(x[order], points, side="right")]

    result.update(
        weighted_mean_difference=difference,
        weighted_smd=difference / sd if sd else (0.0 if difference == 0 else None),
        zero_pooled_sd=sd == 0,
        weighted_ecdf_max_distance=float(np.abs(ecdf(a, wa) - ecdf(b, wb)).max()),
    )
    return result


def diagnose(arms, groups, quality, *, n_folds=5):
    # Reuse the complete invariant audit; no weaker replacement for existing arm checks.
    balance(arms, groups, quality)
    positive = {
        arm: [r for r in arms[arm] if r["split"] == "train" and r["label"] == 1]
        for arm in ("LOW", "CROSS")
    }
    people = {arm: [groups[r["identity_group_a"]] for r in positive[arm]] for arm in positive}
    folds = person_folds(positive, people, n_folds=n_folds)
    rows = positive["LOW"] + positive["CROSS"]
    ids = people["LOW"] + people["CROSS"]
    labels = np.r_[
        np.zeros(len(positive["LOW"]), dtype=int), np.ones(len(positive["CROSS"]), dtype=int)
    ]
    summaries, private = {}, []
    for view in VIEWS:
        probabilities, diagnostics = crossfit(features(rows, quality, view), labels, ids, folds)
        raw = np.where(labels == 1, 1 - probabilities, probabilities)
        weights = raw.copy()
        report = {
            "folds": diagnostics,
            "oof_arm_auc": float(roc_auc_score(labels, probabilities)),
            "arms": {},
            "pair_quality_endpoint_views": {},
            "age_anchor_views": {},
        }
        for label, arm in enumerate(("LOW", "CROSS")):
            mask = labels == label
            if weights[mask].sum() <= 0:
                raise ValueError("no diagnostic overlap mass in one arm")
            weights[mask] /= weights[mask].mean()
            person_totals = Counter()
            for person, weight in zip(np.asarray(ids)[mask], weights[mask], strict=True):
                person_totals[person] += float(weight)
            report["arms"][arm] = {
                "positive_pairs": int(mask.sum()),
                "recorded_people": len(person_totals),
                "zero_weight_pairs": int((raw[mask] == 0).sum()),
                "raw_overlap_weight_quantiles": np.percentile(
                    raw[mask], [0, 5, 50, 95, 100]
                ).tolist(),
                "normalized_weight_mean": float(weights[mask].mean()),
                "pair_effective_size": effective_size(weights[mask]),
                "recorded_person_mass_effective_size": effective_size(list(person_totals.values())),
            }
        for field in QUALITY_FIELDS:
            for kind, aggregate in (("min", min), ("max", max)):
                values = {}
                for arm in positive:
                    values[arm] = []
                    for row in positive[arm]:
                        sides = [
                            value(quality.get(row[f"face_{s}"], {}), field) for s in ("a", "b")
                        ]
                        values[arm].append(aggregate(sides) if None not in sides else None)
                report["pair_quality_endpoint_views"][f"{field}:{kind}"] = {
                    "unweighted": weighted_compare(
                        values["LOW"],
                        values["CROSS"],
                        np.ones(len(positive["LOW"])),
                        np.ones(len(positive["CROSS"])),
                    ),
                    "diagnostic_weighted": weighted_compare(
                        values["LOW"], values["CROSS"], weights[labels == 0], weights[labels == 1]
                    ),
                }
        for kind, aggregate in (("younger", min), ("older", max)):
            ages = {
                arm: [aggregate(r["age_a"], r["age_b"]) for r in positive[arm]] for arm in positive
            }
            report["age_anchor_views"][kind] = weighted_compare(
                ages["LOW"], ages["CROSS"], weights[labels == 0], weights[labels == 1]
            )
        for row, person, fold, probability, raw_weight, weight in zip(
            rows, ids, folds, probabilities, raw, weights, strict=True
        ):
            private.append(
                {
                    "view": view,
                    "pair_id": row["pair_id"],
                    "recorded_person": person,
                    "fold": int(fold),
                    "p_cross_oof": float(probability),
                    "raw_overlap_weight": float(raw_weight),
                    "normalized_diagnostic_weight": float(weight),
                }
            )
        summaries[view] = report
    return {
        "views": summaries,
        "protocol": PROTOCOL | {"folds": n_folds},
        "execution_complete": True,
        "training_weights_ready": False,
        "quality_balance_claimed": False,
        "causal_source_superiority": False,
        "publication_ready": False,
        "recognition_outcomes_used": False,
        "uncertainty": "descriptive diagnostics only; recorded-person mass ESS is not a confidence interval",
        "remaining": [
            "joint distribution balance assessment",
            "negative-pair weighting and image-exposure budgets",
            "independent train-benchmark/human identity audit",
            "fresh weighted or matched training campaign",
        ],
        "scope": "positive training-pair metadata only; no arm modification, face processing or new model inference",
    }, private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("fresh output directory required")
    root = PROJECT_ROOT
    audited = root / "metrics/matched_arm_balance_20261003/summary.manifest.json"
    native = json.loads(audited.read_text(encoding="utf-8"))
    if native["experiment"] != "restricted-matched-arm-age-quality-balance":
        raise ValueError("unexpected prerequisite audit")
    records = native["inputs"] + native["outputs"]
    paths = [
        Path(r["path"]) if Path(r["path"]).is_absolute() else root / r["path"] for r in records
    ]
    if [file_record(p) for p in paths] != records:
        raise ValueError("prerequisite direct source checksums changed")
    inputs = [*paths, audited, Path(__file__), root / "uv.lock", root / "pyproject.toml"]
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
    selected = {
        r[f"face_{s}"]
        for rows in arms.values()
        for r in rows
        if r["split"] == "train"
        for s in ("a", "b")
    }
    quality = {}
    for row in read_jsonl(root / "data/interim/face_quality_audit.jsonl"):
        face = row["face_id"]
        if face in selected:
            if face in quality:
                raise ValueError("duplicate selected quality sidecar row")
            quality[face] = row
    result, private = diagnose(arms, groups, quality)
    if before != [file_record(p) for p in inputs]:
        raise ValueError("inputs changed during metadata diagnostic")
    (out / "private").mkdir(parents=True)
    output, weights = out / "summary.json", out / "private/diagnostic_weights.jsonl"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_jsonl(weights, private)
    target = out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="recorded-person-crossfit-nuisance-overlap",
        parameters=PROTOCOL,
        metrics=result,
        inputs=inputs,
        outputs=[output, weights],
    )
    if before != json.loads(target.read_text(encoding="utf-8"))["inputs"]:
        target.unlink()
        raise ValueError("inputs changed while writing manifest; marker withdrawn")
    print(
        json.dumps(
            {
                view: {"oof_arm_auc": row["oof_arm_auc"], "arms": row["arms"]}
                for view, row in result["views"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
