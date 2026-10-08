"""Fresh oriented-arm weights, joint diagnostics and shared-frame comparison."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from age_gap.common.io import PROJECT_ROOT, read_jsonl, write_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.audit_coupled_weight_budget import audit, blocks
from scripts.audit_joint_pair_quality import diagnose_joint
from scripts.audit_nuisance_overlap import VIEWS
from scripts.audit_repaired_nuisance import refit, selected_quality
from scripts.compare_quality_matching import compare
from scripts.match_exact_negative_quality import positive_image_map


def endpoint_image_tv(paired, weights, side):
    w = np.asarray(weights, float)
    if (
        side not in ("a", "b")
        or w.shape != (len(paired),)
        or not np.isfinite(w).all()
        or np.any(w < 0)
        or w.sum() <= 0
    ):
        raise ValueError("aligned nonnegative weights and valid side required")
    masses = []
    for label in (0, 1):
        mass = Counter()
        for pair, weight in zip(paired, w, strict=True):
            mass[pair[label][f"face_{side}"]] += float(weight / w.sum())
        masses.append(mass)
    p, n = masses
    return float(sum(abs(p[k] - n[k]) for k in p.keys() | n.keys()) / 2)


def exposure_views(arms, groups, ledger):
    maps = {v: {} for v in VIEWS}
    for row in ledger:
        if row["pair_id"] in maps[row["view"]]:
            raise ValueError("duplicate ledger weight")
        maps[row["view"]][row["pair_id"]] = row["candidate_loss_weight"]
    output = {}
    for arm, rows in arms.items():
        paired = blocks([r for r in rows if r["split"] == "train"], groups)
        output[arm] = {}
        for view in ("uniform", *VIEWS):
            weights = (
                np.ones(len(paired))
                if view == "uniform"
                else np.asarray([maps[view][p["pair_id"]] for p, _ in paired])
            )
            output[arm][view] = {
                side: endpoint_image_tv(paired, weights, side) for side in ("a", "b")
            }
    return output


def verified(path, root, experiment):
    native = json.loads(path.read_text(encoding="utf-8"))
    if (
        native["experiment"] != experiment
        or native["metrics"].get("execution_complete") is not True
    ):
        raise ValueError("completed correctly typed prerequisite required")
    if json.loads((path.parent / "summary.json").read_text(encoding="utf-8")) != native["metrics"]:
        raise ValueError("summary/native metrics mismatch")
    records = native["inputs"] + native["outputs"]
    paths = [
        Path(r["path"]) if Path(r["path"]).is_absolute() else root / r["path"] for r in records
    ]
    if [file_record(p) for p in paths] != records:
        raise ValueError("prerequisite records changed")
    return native, paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("fresh destination required")
    root = PROJECT_ROOT
    specs = [
        (
            "previous",
            "metrics/repaired_nuisance_20261003",
            "exact-supported-fresh-nuisance-coupled-audit",
        ),
        (
            "quality_cost",
            "metrics/quality_matched_exact_20261003",
            "exact-age-negative-quality-cost-preparation",
        ),
        (
            "oriented",
            "metrics/oriented_exposure_matching_20261003",
            "joint-oriented-negative-exposure-feasibility",
        ),
    ]
    inherited, directories = [], {}
    for version, directory, experiment in specs:
        path = root / directory / "summary.manifest.json"
        native, paths = verified(path, root, experiment)
        if version == "oriented" and native["metrics"].get("both_arms_feasible") is not True:
            raise ValueError("both oriented arms required")
        inherited.extend([*paths, path])
        directories[version] = path.parent
    deps = [
        root / f"scripts/{name}.py"
        for name in (
            "audit_coupled_weight_budget",
            "audit_joint_pair_quality",
            "audit_nuisance_overlap",
            "audit_repaired_nuisance",
            "audit_matched_arm_balance",
            "compare_quality_matching",
            "match_exact_negative_quality",
        )
    ]
    inputs = list(dict.fromkeys(p.resolve() for p in [*inherited, Path(__file__), *deps]))
    before = [file_record(p) for p in inputs]
    groups = {}
    for row in read_jsonl(root / "data/processed/person_clusters.jsonl"):
        key, person = row["identity_group_id"], row["person_id"]
        if not isinstance(person, str) or not person or (key in groups and groups[key] != person):
            raise ValueError("invalid person mapping")
        groups[key] = person
    arm_dirs = {v: d / "private" for v, d in directories.items()}
    arm_dirs["previous"] = root / "metrics/exact_supported_repair_v3_20261003/private"
    versions = {
        v: {a: list(read_jsonl(d / f"{a.lower()}_candidate_arm.jsonl")) for a in ("LOW", "CROSS")}
        for v, d in arm_dirs.items()
    }
    selected = {
        r[f"face_{s}"]
        for rows in versions["previous"].values()
        for r in rows
        if r["split"] == "train"
        for s in ("a", "b")
    }
    quality = selected_quality(read_jsonl(root / "data/interim/face_quality_audit.jsonl"), selected)
    ledgers = {
        v: list(read_jsonl(directories[v] / "private/candidate_pair_weights.jsonl"))
        for v in ("previous", "quality_cost")
    }
    with threadpool_limits(limits=1):
        fresh, diagnostics, ledger = refit(versions["oriented"], groups, quality)
        ledgers["oriented"] = ledger
        for v in ("previous", "quality_cost"):
            _, expected = audit(
                versions[v],
                groups,
                list(read_jsonl(directories[v] / "private/diagnostic_weights.jsonl")),
            )
            if expected != ledgers[v]:
                raise ValueError("historical ledger differs from checked objective")
        features, fitted = positive_image_map(versions["previous"], groups, quality)
        result = dict(
            fresh_nuisance=fresh,
            joint_quality=diagnose_joint(
                versions["oriented"], groups, quality, diagnostics, ledger
            ),
            same_frame_comparisons=compare(versions, groups, features, ledgers),
            same_frame_standardization=fitted,
            sigmas=[3.0, 6.0, 12.0],
            class_endpoint_image_tv=exposure_views(versions["oriented"], groups, ledger),
            execution_complete=True,
            training_ready=False,
            publication_ready=False,
            recognition_outcomes_used=False,
            scope="descriptive measured quality only; new ordered positives/negatives, fresh weights; no automatic balance acceptance",
        )
    if before != [file_record(p) for p in inputs]:
        raise ValueError("inputs changed during audit")
    (out / "private").mkdir(parents=True)
    output = out / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    outputs = [output]
    for name, rows in (("diagnostic_weights", diagnostics), ("candidate_pair_weights", ledger)):
        path = out / f"private/{name}.jsonl"
        write_jsonl(path, rows)
        outputs.append(path)
    target = out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="oriented-fresh-nuisance-joint-audit",
        parameters={
            "n_folds": 5,
            "seed": 42,
            "training_executed": False,
            "fixed_sigmas": [3.0, 6.0, 12.0],
        },
        metrics=result,
        inputs=inputs,
        outputs=outputs,
    )
    if before != json.loads(target.read_text(encoding="utf-8"))["inputs"]:
        target.unlink()
        raise ValueError("inputs changed during manifest write")
    print(
        json.dumps(
            {
                "comparisons": result["same_frame_comparisons"],
                "endpoint_tv": result["class_endpoint_image_tv"],
                "training_ready": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
