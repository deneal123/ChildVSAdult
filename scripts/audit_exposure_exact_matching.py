"""Bounded feasibility check: exact ages, unique edges and exact positive-B exposure.

No recognition outcomes, training, row dropping or constraint relaxation.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from age_gap.common.io import PROJECT_ROOT, read_jsonl, write_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.audit_coupled_weight_budget import blocks, weighted_age_balance
from scripts.audit_repaired_nuisance import verified_prerequisite

PROTOCOL = {
    "version": "exact-negative-B-exposure-feasibility-v1",
    "objective": "zero: feasibility only, not quality optimum",
    "constraints": "one B per target; exact B-image counts from positive B; unique unordered negative edges",
    "identity": "different recorded people; exclude canonical known-genuine edges",
    "time_limit_per_arm_seconds": 30,
    "node_limit_per_arm": 1000,
    "integrality_tolerance": 1e-7,
    "scope": "fixed E88 positive pools/heldout; no training or global corpus impossibility claim",
}


def constrained_assignment(anchors, options, counts, *, seconds=30, node_limit=1000):
    """Each option is a B-image ID, not an edge with ambiguous orientation."""
    if (
        not anchors
        or len(anchors) != len(options)
        or sum(counts.values()) != len(anchors)
        or any(type(v) is not int or v <= 0 for v in counts.values())
        or not np.isfinite(seconds)
        or seconds <= 0
        or type(node_limit) is not int
        or node_limit <= 0
    ):
        raise ValueError(
            "aligned nonempty targets, positive counts and finite solver limits required"
        )
    columns = [(i, b) for i, candidates in enumerate(options) for b in sorted(set(candidates))]
    if any(b not in counts or anchors[i] == b for i, b in columns):
        raise ValueError("candidate B must have positive capacity and differ from A")
    empty = [i for i, candidates in enumerate(options) if not candidates]
    report = {
        "targets": len(anchors),
        "candidate_assignments": len(columns),
        "targets_without_candidate": len(empty),
        "training_ready": False,
    }
    if empty:
        return None, dict(report, status="structural_infeasible", solver_called=False)
    faces = sorted(counts)
    edges = sorted({tuple(sorted((anchors[i], b))) for i, b in columns})
    face_row = {b: len(anchors) + j for j, b in enumerate(faces)}
    edge_row = {e: len(anchors) + len(faces) + j for j, e in enumerate(edges)}
    ri, ci = [], []
    for j, (i, b) in enumerate(columns):
        ri.extend((i, face_row[b], edge_row[tuple(sorted((anchors[i], b)))]))
        ci.extend((j, j, j))
    matrix = coo_array(
        (np.ones(len(ri)), (np.asarray(ri, np.int32), np.asarray(ci, np.int32))),
        shape=(len(anchors) + len(faces) + len(edges), len(columns)),
    ).tocsc()
    lower = np.r_[np.ones(len(anchors)), [counts[b] for b in faces], np.zeros(len(edges))]
    upper = np.r_[np.ones(len(anchors)), [counts[b] for b in faces], np.ones(len(edges))]
    result = milp(
        np.zeros(len(columns)),
        integrality=np.ones(len(columns), np.int32),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": float(seconds), "node_limit": node_limit, "mip_rel_gap": 0.0},
    )
    report.update(
        solver_called=True,
        solver_status=int(result.status),
        solver_message=str(result.message),
        constraint_rows=matrix.shape[0],
        unique_candidate_edges=len(edges),
    )
    # Independently check any incumbent, including one returned at a time limit.
    solution = getattr(result, "x", None)
    if solution is not None:
        x = np.asarray(solution, float)
        if x.shape != (len(columns),) or not np.isfinite(x).all():
            raise ValueError("invalid solver incumbent")
        rounded = np.rint(x)
        if (
            np.any(abs(x - rounded) > PROTOCOL["integrality_tolerance"])
            or not np.isin(rounded, [0, 1]).all()
        ):
            raise ValueError("nonbinary solver incumbent")
        mass = matrix @ rounded
        if np.any(mass < lower - 1e-7) or np.any(mass > upper + 1e-7):
            raise ValueError("solver incumbent violates constraints")
        selected = [columns[j] for j in np.flatnonzero(rounded)]
        chosen = {i: b for i, b in selected}
        if len(selected) != len(anchors) or len(chosen) != len(anchors):
            raise ValueError("not one B per target")
        output = [chosen[i] for i in range(len(anchors))]
        if Counter(output) != counts or len(
            {tuple(sorted((a, b))) for a, b in zip(anchors, output, strict=True)}
        ) != len(anchors):
            raise ValueError("exposure/edge invariant violated")
        return output, dict(report, status="feasible", checked_incumbent=True)
    status = {1: "limit_without_incumbent", 2: "solver_infeasible"}.get(
        int(result.status), "solver_failure"
    )
    return None, dict(report, status=status, checked_incumbent=False)


def exposure_arm(rows, groups, *, arm, forbidden):
    positives = [p for p, _ in blocks([r for r in rows if r["split"] == "train"], groups)]
    faces, age_pool = {}, defaultdict(list)
    for p in positives:
        for side in ("a", "b"):
            f, age, g = p[f"face_{side}"], p[f"age_{side}"], p[f"identity_group_{side}"]
            person = groups[g]
            if f in faces and faces[f][:2] != (age, person):
                raise ValueError("conflicting image metadata")
            faces[f] = (age, person, min(g, faces[f][2]) if f in faces else g)
    counts = Counter(p["face_b"] for p in positives)
    for b in sorted(counts):
        age_pool[faces[b][0]].append(b)
    anchors = [p["face_a"] for p in positives]
    options = [
        {
            b
            for b in age_pool[p["age_b"]]
            if b != p["face_a"]
            and faces[b][1] != faces[p["face_a"]][1]
            and tuple(sorted((p["face_a"], b))) not in forbidden
        }
        for p in positives
    ]
    chosen, report = constrained_assignment(
        anchors,
        options,
        counts,
        seconds=PROTOCOL["time_limit_per_arm_seconds"],
        node_limit=PROTOCOL["node_limit_per_arm"],
    )
    report["positive_B_images"] = len(counts)
    if chosen is None:
        return None, report
    negatives = []
    for i, (p, b) in enumerate(zip(positives, chosen, strict=True)):
        negatives.append(
            dict(
                pair_id=f"exposureneg_{arm.lower()}_{i:04d}",
                face_a=p["face_a"],
                face_b=b,
                label=0,
                split="train",
                pair_type="negative_exact_age_B_exposure_unique_edge",
                identity_group_a=p["identity_group_a"],
                identity_group_b=faces[b][2],
                age_a=p["age_a"],
                age_b=p["age_b"],
                age_gap=p["age_gap"],
                matched_target_pair_id=p["pair_id"],
                status="ok",
                hardness="exact_age_B_exposure",
            )
        )
    output = (
        copy.deepcopy(positives)
        + negatives
        + copy.deepcopy([r for r in rows if r["split"] != "train"])
    )
    paired = blocks([r for r in output if r["split"] == "train"], groups)
    report["age_balance"] = weighted_age_balance(paired, np.ones(len(paired)))
    if any(v["empirical_weighted_total_variation"] != 0 for v in report["age_balance"].values()):
        raise ValueError("nonzero class age TV")
    positive_exposure = Counter(p[f"face_{s}"] for p in positives for s in ("a", "b"))
    negative_exposure = Counter(n[f"face_{s}"] for n in negatives for s in ("a", "b"))
    if positive_exposure != negative_exposure:
        raise ValueError("class image exposure differs")
    report.update(
        class_image_exposure_exact=True,
        images=len(positive_exposure),
        recorded_people=len({faces[f][1] for f in positive_exposure}),
        max_image_whole_pass_count=2 * max(positive_exposure.values()),
    )
    return output, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("fresh directory required")
    root = PROJECT_ROOT
    prerequisite = root / "metrics/exact_supported_repair_v3_20261003/summary.manifest.json"
    inherited = verified_prerequisite(prerequisite, root)
    canonical = root / "data/processed/pairs.jsonl"
    dependencies = [
        root / f"scripts/{n}.py"
        for n in (
            "audit_repaired_nuisance",
            "audit_coupled_weight_budget",
            "audit_nuisance_overlap",
            "audit_matched_arm_balance",
        )
    ]
    inputs = list(
        dict.fromkeys(
            x.resolve()
            for x in [*inherited, prerequisite, canonical, Path(__file__), *dependencies]
        )
    )
    before = [file_record(x) for x in inputs]
    groups = {}
    for row in read_jsonl(root / "data/processed/person_clusters.jsonl"):
        g, person = row["identity_group_id"], row["person_id"]
        if not isinstance(person, str) or not person or (g in groups and groups[g] != person):
            raise ValueError("invalid mapping")
        groups[g] = person
    forbidden = {
        tuple(sorted((r["face_a"], r["face_b"]))) for r in read_jsonl(canonical) if r["label"] == 1
    }
    reports, candidates = {}, {}
    for arm in ("LOW", "CROSS"):
        rows = list(read_jsonl(prerequisite.parent / f"private/{arm.lower()}_candidate_arm.jsonl"))
        candidates[arm], reports[arm] = exposure_arm(rows, groups, arm=arm, forbidden=forbidden)
    result = dict(
        arms=reports,
        protocol=PROTOCOL,
        scipy_version=scipy.__version__,
        execution_complete=True,
        both_arms_feasible=all(x is not None for x in candidates.values()),
        training_ready=False,
        publication_ready=False,
        recognition_outcomes_used=False,
    )
    if before != [file_record(x) for x in inputs]:
        raise ValueError("inputs changed")
    (out / "private").mkdir(parents=True)
    output = out / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    outputs = [output]
    for arm, candidate in candidates.items():
        if candidate is not None:
            path = out / f"private/{arm.lower()}_candidate_arm.jsonl"
            write_jsonl(path, candidate)
            outputs.append(path)
    target = out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="exact-negative-B-exposure-feasibility",
        parameters=PROTOCOL,
        metrics=result,
        inputs=inputs,
        outputs=outputs,
    )
    if before != json.loads(target.read_text(encoding="utf-8"))["inputs"]:
        target.unlink()
        raise ValueError("inputs changed during manifest write")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
