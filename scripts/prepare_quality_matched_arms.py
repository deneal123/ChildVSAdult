"""Prepare exact-age quality-cost candidates, refit weights, and audit full paired quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import scipy
from threadpoolctl import threadpool_limits

from age_gap.common.io import PROJECT_ROOT, read_jsonl, write_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.audit_coupled_weight_budget import blocks
from scripts.audit_joint_pair_quality import diagnose_joint
from scripts.audit_repaired_nuisance import refit, selected_quality, verified_prerequisite
from scripts.match_exact_negative_quality import PROTOCOL, positive_image_map, quality_arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("fresh output directory required")
    root = PROJECT_ROOT
    prerequisite = root / "metrics/exact_supported_repair_v3_20261003/summary.manifest.json"
    inherited = verified_prerequisite(prerequisite, root)
    dependencies = [
        root / f"scripts/{name}.py"
        for name in (
            "match_exact_negative_quality",
            "audit_exact_negative_matching",
            "audit_joint_pair_quality",
            "audit_coupled_weight_budget",
            "audit_repaired_nuisance",
            "audit_nuisance_overlap",
            "audit_matched_arm_balance",
        )
    ]
    canonical = root / "data/processed/pairs.jsonl"
    inputs = list(
        dict.fromkeys(
            p.resolve()
            for p in [
                *inherited,
                prerequisite,
                canonical,
                Path(__file__),
                *dependencies,
                root / "uv.lock",
                root / "pyproject.toml",
            ]
        )
    )
    before = [file_record(p) for p in inputs]
    groups = {}
    for row in read_jsonl(root / "data/processed/person_clusters.jsonl"):
        key, person = row["identity_group_id"], row["person_id"]
        if not isinstance(person, str) or not person or (key in groups and groups[key] != person):
            raise ValueError("invalid/conflicting person mapping")
        groups[key] = person
    arms = {
        a: list(read_jsonl(prerequisite.parent / f"private/{a.lower()}_candidate_arm.jsonl"))
        for a in ("LOW", "CROSS")
    }
    selected = {
        r[f"face_{s}"]
        for rows in arms.values()
        for r in rows
        if r["split"] == "train"
        for s in ("a", "b")
    }
    quality = selected_quality(read_jsonl(root / "data/interim/face_quality_audit.jsonl"), selected)
    forbidden = {
        tuple(sorted((r["face_a"], r["face_b"]))) for r in read_jsonl(canonical) if r["label"] == 1
    }
    reports, candidates = {}, {}
    with threadpool_limits(limits=1):
        features, fitted = positive_image_map(arms, groups, quality)
        for arm in arms:
            candidates[arm], reports[arm] = quality_arm(
                arms[arm], groups, features, arm=arm, forbidden=forbidden
            )
            old_cost = 0.0
            for p, n in blocks([r for r in arms[arm] if r["split"] == "train"], groups):
                diff = features[p["face_b"]] - features[n["face_b"]]
                old_cost += float(diff @ diff)
            reports[arm]["previous_sampler_total_cost_same_map"] = old_cost
            if candidates[arm] is not None:
                if reports[arm]["total_unshifted_cost"] > old_cost + 1e-8:
                    raise ValueError("minimum cost exceeds feasible previous assignment")
                if [r for r in candidates[arm] if r["label"] == 1 and r["split"] == "train"] != [
                    r for r in arms[arm] if r["label"] == 1 and r["split"] == "train"
                ]:
                    raise ValueError("positive rows changed")
                if [r for r in candidates[arm] if r["split"] != "train"] != [
                    r for r in arms[arm] if r["split"] != "train"
                ]:
                    raise ValueError("heldout rows changed")
        feasible = all(v is not None for v in candidates.values())
        result = dict(
            protocol=PROTOCOL,
            scipy_version=scipy.__version__,
            cost_standardization=fitted,
            arms=reports,
            full_matching_feasible=feasible,
            execution_complete=True,
            training_ready=False,
            publication_ready=False,
            recognition_outcomes_used=False,
        )
        weights, ledger = [], []
        if feasible:
            nuisance, weights, ledger = refit(candidates, groups, quality)
            result["fresh_nuisance"] = nuisance
            result["joint_quality"] = diagnose_joint(candidates, groups, quality, weights, ledger)
    if before != [file_record(p) for p in inputs]:
        raise ValueError("inputs changed during candidate preparation")
    private = out / "private"
    private.mkdir(parents=True)
    output = out / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    outputs = [output]
    if feasible:
        for arm, rows in candidates.items():
            path = private / f"{arm.lower()}_candidate_arm.jsonl"
            write_jsonl(path, rows)
            outputs.append(path)
        for name, rows in (("diagnostic_weights", weights), ("candidate_pair_weights", ledger)):
            path = private / f"{name}.jsonl"
            write_jsonl(path, rows)
            outputs.append(path)
    target = out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="exact-age-negative-quality-cost-preparation",
        parameters={"protocol": PROTOCOL, "training_executed": False},
        metrics=result,
        inputs=inputs,
        outputs=outputs,
    )
    if before != json.loads(target.read_text(encoding="utf-8"))["inputs"]:
        target.unlink()
        raise ValueError("inputs changed during manifest write")
    print(json.dumps({"feasible": feasible, "arms": reports, "training_ready": False}, indent=2))


if __name__ == "__main__":
    main()
