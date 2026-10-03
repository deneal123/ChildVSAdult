"""Descriptive joint-quality audit of the complete candidate paired loss objective."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.audit_coupled_weight_budget import audit, blocks
from scripts.audit_matched_arm_balance import QUALITY_FIELDS, balance, value
from scripts.audit_nuisance_overlap import PROTOCOL as NUISANCE_PROTOCOL
from scripts.audit_nuisance_overlap import VIEWS, weighted_compare
from scripts.audit_repaired_nuisance import selected_quality

PROTOCOL = {
    "version": "full-paired-loss-joint-quality-v1",
    "feature_view": "ordered A/B quality fields followed by per-endpoint missing indicators; no ages",
    "fields": list(QUALITY_FIELDS),
    "log1p_fields": NUISANCE_PROTOCOL["log1p_fields"],
    "standardization": "one unweighted pooled train-only median/mean/population-SD map for all classes and weight views",
    "all_missing": "computational zero imputation; explicit missing indicators retained",
    "joint_statistic": "squared RBF distance between weighted empirical kernel means, diagonal terms included",
    "formula": "wX^T KXX wX + wY^T KYY wY - 2 wX^T KXY wY; each weight vector sums to one",
    "bandwidth_multipliers": [0.5, 1.0, 2.0],
    "bandwidth": "sigma=multiplier*sqrt(feature_dimension); all views reported, no outcome-based choice",
    "chunk_rows": 128,
    "threads": 1,
    "uncertainty": "descriptive empirical distance only; reused images/people are dependent; no iid p-value, CI or equivalence cutoff",
    "method_reference": "https://www.jmlr.org/papers/v13/gretton12a.html",
}


def ordered_features(rows, quality):
    output = []
    for row in rows:
        observed, missing = [], []
        for side in ("a", "b"):
            for field in QUALITY_FIELDS:
                v = value(quality.get(row[f"face_{side}"], {}), field)
                missing.append(int(v is None))
                observed.append(
                    np.nan if v is None else np.log1p(v) if field in PROTOCOL["log1p_fields"] else v
                )
        output.append(observed + missing)
    return np.asarray(output, dtype=float)


def standardize(matrix):
    x = np.asarray(matrix, float)
    if x.ndim != 2 or not len(x) or not x.shape[1] or np.isinf(x).any():
        raise ValueError("nonempty finite-or-missing feature matrix required")
    median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in x.T])
    filled = np.where(np.isnan(x), median, x)
    mean, sd = filled.mean(axis=0), filled.std(axis=0)
    scale = np.where(sd == 0, 1, sd)
    return (filled - mean) / scale, {
        "medians": median.tolist(),
        "means": mean.tolist(),
        "scales": scale.tolist(),
        "all_missing_columns": int(np.isnan(x).all(axis=0).sum()),
        "constant_columns_after_imputation": int((sd == 0).sum()),
        "scope": "shared descriptive train-only transform; not held-out predictive preprocessing",
    }


def weight_matrix(weights, n):
    w = np.asarray(weights, float)
    if (
        w.ndim != 2
        or w.shape[1] != n
        or not n
        or not len(w)
        or not np.isfinite(w).all()
        or np.any(w < 0)
        or np.any(w.sum(axis=1) <= 0)
    ):
        raise ValueError("aligned nonnegative finite weight rows with positive total required")
    return w / w.sum(axis=1, keepdims=True)


def joint_mmd(x, y, wx, wy, *, sigmas, chunk=128):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[1] != y.shape[1]
        or not x.shape[1]
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or type(chunk) is not int
        or chunk < 1
    ):
        raise ValueError("aligned finite matrices and positive integer chunk required")
    wx, wy = weight_matrix(wx, len(x)), weight_matrix(wy, len(y))
    if len(wx) != len(wy) or not sigmas or any(not np.isfinite(s) or s <= 0 for s in sigmas):
        raise ValueError("aligned views and positive finite bandwidths required")
    terms = []
    for a, b, wa, wb in ((x, x, wx, wx), (y, y, wy, wy), (x, y, wx, wy)):
        totals = np.zeros((len(sigmas), len(wa)))
        for start in range(0, len(a), chunk):
            end = min(start + chunk, len(a))
            distance = cdist(a[start:end], b, metric="sqeuclidean")
            for i, sigma in enumerate(sigmas):
                kernel = np.exp(-distance / (2 * sigma**2))
                totals[i] += np.sum(wa[:, start:end].T * (kernel @ wb.T), axis=0)
        terms.append(totals)
    squared = terms[0] + terms[1] - 2 * terms[2]
    if not np.isfinite(squared).all() or np.any(squared < -1e-10):
        raise ValueError("invalid empirical kernel distance")
    return np.maximum(squared, 0), int((squared < 0).sum())


def missing_pattern_tv(x, y, wx, wy):
    x, y = np.asarray(x), np.asarray(y)
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[1] != y.shape[1]
        or not np.isin(x, [0, 1]).all()
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("aligned binary missing-pattern matrices required")

    def mass(rows, weights):
        result = Counter()
        for row, w in zip(rows, weights, strict=True):
            result[tuple(row)] += float(w)
        return result

    wx, wy = weight_matrix([wx], len(x))[0], weight_matrix([wy], len(y))[0]
    a, b = mass(x, wx), mass(y, wy)
    return float(sum(abs(a[k] - b[k]) for k in a.keys() | b.keys()) / 2)


def diagnose_joint(arms, groups, quality, diagnostics, ledger):
    balance(arms, groups, quality)
    expected, checked_ledger = audit(arms, groups, diagnostics)
    if checked_ledger != ledger:
        raise ValueError("candidate ledger differs from checked coupled objective")
    paired = {
        a: blocks([r for r in arms[a] if r["split"] == "train"], groups) for a in ("LOW", "CROSS")
    }
    rows = {a: [r for pair in paired[a] for r in pair] for a in paired}
    if any(
        (p["age_a"], p["age_b"]) != (n["age_a"], n["age_b"])
        for pairs in paired.values()
        for p, n in pairs
    ):
        raise ValueError("exact-age coupled candidates required")
    maps = {v: {} for v in VIEWS}
    for row in checked_ledger:
        maps[row["view"]][row["pair_id"]] = row["candidate_loss_weight"]
    views = ["uniform", *VIEWS]
    weights = {
        a: np.array(
            [[1.0] * len(rows[a])] + [[maps[v][r["pair_id"]] for r in rows[a]] for v in VIEWS]
        )
        for a in rows
    }
    raw = {a: ordered_features(rows[a], quality) for a in rows}
    transformed, transform = standardize(np.concatenate([raw["LOW"], raw["CROSS"]]))
    x = {"LOW": transformed[: len(rows["LOW"])], "CROSS": transformed[len(rows["LOW"]) :]}
    dim = transformed.shape[1]
    sigmas = [float(m * np.sqrt(dim)) for m in PROTOCOL["bandwidth_multipliers"]]
    comparisons = {}
    selections = [
        (name, "LOW", s, "CROSS", s)
        for name, s in (
            ("positive", slice(0, None, 2)),
            ("negative", slice(1, None, 2)),
            ("combined_paired_objective", slice(None)),
        )
    ]
    selections += [
        (f"{a}_positive_vs_negative", a, slice(0, None, 2), a, slice(1, None, 2)) for a in rows
    ]
    roundoff = 0
    for name, a, sa, b, sb in selections:
        mmd, clipped = joint_mmd(
            x[a][sa], x[b][sb], weights[a][:, sa], weights[b][:, sb], sigmas=sigmas
        )
        roundoff += clipped
        report = {}
        for i, view in enumerate(views):
            marginal = {}
            for j, field in enumerate(f"{side}:{f}" for side in ("a", "b") for f in QUALITY_FIELDS):
                av, bv = raw[a][sa, j], raw[b][sb, j]
                marginal[field] = weighted_compare(
                    [None if np.isnan(v) else float(v) for v in av],
                    [None if np.isnan(v) else float(v) for v in bv],
                    weights[a][i, sa],
                    weights[b][i, sb],
                )
            report[view] = {
                "n_a": len(x[a][sa]),
                "n_b": len(x[b][sb]),
                "joint_mmd_squared": {
                    str(multiplier): float(mmd[k, i])
                    for k, multiplier in enumerate(PROTOCOL["bandwidth_multipliers"])
                },
                "joint_missing_pattern_tv": missing_pattern_tv(
                    raw[a][sa, dim // 2 :],
                    raw[b][sb, dim // 2 :],
                    weights[a][i, sa],
                    weights[b][i, sb],
                ),
                "ordered_endpoint_marginals": marginal,
            }
        comparisons[name] = report
    return {
        "comparisons": comparisons,
        "protocol": PROTOCOL,
        "standardization": transform,
        "sigmas": sigmas,
        "roundoff_negative_distances_clipped": roundoff,
        "coupled_age_balance_verified": all(
            v["empirical_weighted_total_variation"] == 0
            for reports in expected["views"].values()
            for arm in reports.values()
            for v in arm["coupled_candidate_age_balance"].values()
        ),
        "execution_complete": True,
        "quality_balance_claimed": False,
        "training_weights_ready": False,
        "publication_ready": False,
        "recognition_outcomes_used": False,
        "scope": "joint measured training-pair quality only; not all possible nuisance variables or causal identification",
        "remaining": [
            "decide defensible nuisance/exposure/BN protocol",
            "source/crop-bound preflight",
            "independent human identity/benchmark audit",
            "fresh three-seed training",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("fresh destination required")
    root = PROJECT_ROOT
    prerequisite = root / "metrics/repaired_nuisance_20261003/summary.manifest.json"
    native = json.loads(prerequisite.read_text(encoding="utf-8"))
    if (
        native["experiment"] != "exact-supported-fresh-nuisance-coupled-audit"
        or native["metrics"].get("execution_complete") is not True
    ):
        raise ValueError("completed fresh nuisance prerequisite required")
    records = native["inputs"] + native["outputs"]
    paths = [
        Path(r["path"]) if Path(r["path"]).is_absolute() else root / r["path"] for r in records
    ]
    if [file_record(p) for p in paths] != records:
        raise ValueError("prerequisite records changed")
    consumed = [
        root / f"metrics/exact_supported_repair_v3_20261003/private/{a}_candidate_arm.jsonl"
        for a in ("low", "cross")
    ]
    consumed.extend(
        [
            root / "data/processed/person_clusters.jsonl",
            root / "data/interim/face_quality_audit.jsonl",
            prerequisite.parent / "private/diagnostic_weights.jsonl",
            prerequisite.parent / "private/candidate_pair_weights.jsonl",
        ]
    )
    if any(file_record(p) not in records for p in consumed):
        raise ValueError("consumed candidate/source/weight absent from prerequisite")
    inputs = list(dict.fromkeys(p.resolve() for p in [*paths, prerequisite, Path(__file__)]))
    before = [file_record(p) for p in inputs]
    groups = {}
    for row in read_jsonl(consumed[2]):
        key, person = row["identity_group_id"], row["person_id"]
        if not isinstance(person, str) or not person or (key in groups and groups[key] != person):
            raise ValueError("invalid/conflicting recorded-person mapping")
        groups[key] = person
    arms = {a: list(read_jsonl(consumed[i])) for i, a in enumerate(("LOW", "CROSS"))}
    selected = {
        r[f"face_{s}"]
        for rows in arms.values()
        for r in rows
        if r["split"] == "train"
        for s in ("a", "b")
    }
    quality = selected_quality(read_jsonl(consumed[3]), selected)
    with threadpool_limits(limits=1):
        result = diagnose_joint(
            arms, groups, quality, list(read_jsonl(consumed[4])), list(read_jsonl(consumed[5]))
        )
    if before != [file_record(p) for p in inputs]:
        raise ValueError("inputs changed during joint audit")
    out.mkdir(parents=True)
    output = out / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="exact-supported-full-paired-joint-quality",
        parameters=PROTOCOL,
        metrics=result,
        inputs=inputs,
        outputs=[output],
    )
    if before != json.loads(target.read_text(encoding="utf-8"))["inputs"]:
        target.unlink()
        raise ValueError("inputs changed during manifest write; marker withdrawn")
    print(
        json.dumps(
            {
                name: {v: x["joint_mmd_squared"] for v, x in row.items()}
                for name, row in result["comparisons"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
