"""Joint positive orientation/negative selection with exact unweighted B exposure."""

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
    "version": "joint-orientation-exact-B-exposure-v1",
    "objective": "minimize number of reversed positives; accept checked feasible incumbent at limit without optimality claim",
    "time_limit_seconds_per_arm": 30,
    "node_limit_per_arm": 1000,
    "scope": "same unordered positive pairs, images, people, gaps and heldout; ordered positive rows may change",
    "exposure": "negative B count minus chosen positive B count equals zero for every image",
    "edges": "unique unordered negatives; exclude canonical genuine edges",
    "training_ready": False,
}


def joint_assignment(pairs, options, *, seconds=30, node_limit=1000):
    """options[i][orientation] lists eligible B images for target i."""
    if not pairs or len(options) != len(pairs) or any(len(o) != 2 for o in options):
        raise ValueError("nonempty aligned pairs with two orientations required")
    if (
        any(len(p) != 2 or p[0] == p[1] for p in pairs)
        or not np.isfinite(seconds)
        or seconds <= 0
        or type(node_limit) is not int
        or node_limit <= 0
    ):
        raise ValueError("nonself pairs and finite positive solver limits required")
    faces = sorted({f for p in pairs for f in p})
    columns = [
        (i, o, b)
        for i, choices in enumerate(options)
        for o in (0, 1)
        for b in sorted(set(choices[o]))
    ]
    if any(b not in faces or b in pairs[i] for i, _, b in columns):
        raise ValueError("candidate must be another selected image outside target pair")
    report = {"targets": len(pairs), "candidate_assignments": len(columns), "training_ready": False}
    if any(not (set(o[0]) | set(o[1])) for o in options):
        return None, dict(report, status="structural_infeasible", solver_called=False)
    edges = sorted({tuple(sorted((pairs[i][o], b))) for i, o, b in columns})
    fr = {f: len(pairs) + j for j, f in enumerate(faces)}
    er = {e: len(pairs) + len(faces) + j for j, e in enumerate(edges)}
    rows, cols, data = [], [], []
    for j, (i, o, b) in enumerate(columns):
        a, pb = pairs[i][o], pairs[i][1 - o]
        rows.extend((i, fr[b], fr[pb], er[tuple(sorted((a, b)))]))
        cols.extend((j, j, j, j))
        data.extend((1.0, 1.0, -1.0, 1.0))
    matrix = coo_array(
        (np.asarray(data), (np.asarray(rows, np.int32), np.asarray(cols, np.int32))),
        shape=(len(pairs) + len(faces) + len(edges), len(columns)),
    ).tocsc()
    low = np.r_[np.ones(len(pairs)), np.zeros(len(faces) + len(edges))]
    high = np.r_[np.ones(len(pairs)), np.zeros(len(faces)), np.ones(len(edges))]
    result = milp(
        np.asarray([o for _, o, _ in columns], float),
        integrality=np.ones(len(columns), np.int32),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, low, high),
        options={"time_limit": float(seconds), "node_limit": node_limit, "mip_rel_gap": 0.0},
    )
    report.update(
        solver_called=True, solver_status=int(result.status), solver_message=str(result.message)
    )
    if getattr(result, "x", None) is None:
        return None, dict(
            report,
            status={1: "limit_without_incumbent", 2: "solver_infeasible"}.get(
                int(result.status), "solver_failure"
            ),
        )
    x = np.asarray(result.x, float)
    rounded = np.rint(x)
    if (
        x.shape != (len(columns),)
        or not np.isfinite(x).all()
        or np.any(abs(x - rounded) > 1e-7)
        or not np.isin(rounded, [0, 1]).all()
    ):
        raise ValueError("invalid binary incumbent")
    mass = matrix @ rounded
    if np.any(mass < low - 1e-7) or np.any(mass > high + 1e-7):
        raise ValueError("incumbent constraint violation")
    chosen = [columns[j] for j in np.flatnonzero(rounded)]
    if len(chosen) != len(pairs) or len({i for i, _, _ in chosen}) != len(pairs):
        raise ValueError("incumbent missing/duplicate targets")
    chosen.sort()
    if Counter(b for _, _, b in chosen) != Counter(pairs[i][1 - o] for i, o, _ in chosen):
        raise ValueError("incumbent B exposures differ")
    if len({tuple(sorted((pairs[i][o], b))) for i, o, b in chosen}) != len(pairs):
        raise ValueError("incumbent reuses an unordered edge")
    report.update(
        status="feasible",
        checked_incumbent=True,
        flipped_positives=sum(o for _, o, _ in chosen),
        minimum_flips_proven=int(result.status) == 0,
    )
    return chosen, report


def oriented_arm(rows, groups, *, arm, forbidden):
    positives = [p for p, _ in blocks([r for r in rows if r["split"] == "train"], groups)]
    faces, age_pool = {}, defaultdict(list)
    for p in positives:
        for side in ("a", "b"):
            f, age, g = p[f"face_{side}"], p[f"age_{side}"], p[f"identity_group_{side}"]
            meta = (age, groups[g])
            if f in faces and faces[f][:2] != meta:
                raise ValueError("conflicting image metadata")
            faces[f] = (*meta, min(g, faces[f][2]) if f in faces else g)
    for f in sorted(faces):
        age_pool[faces[f][0]].append(f)
    pairs = [(p["face_a"], p["face_b"]) for p in positives]
    options = [
        [
            {
                b
                for b in age_pool[faces[pair[1 - o]][0]]
                if b not in pair
                and faces[b][1] != faces[pair[o]][1]
                and tuple(sorted((pair[o], b))) not in forbidden
            }
            for o in (0, 1)
        ]
        for pair in pairs
    ]
    chosen, report = joint_assignment(
        pairs,
        options,
        seconds=PROTOCOL["time_limit_seconds_per_arm"],
        node_limit=PROTOCOL["node_limit_per_arm"],
    )
    if chosen is None:
        return None, report
    oriented, negatives = [], []
    for i, o, b in chosen:
        p = copy.deepcopy(positives[i])
        if o:
            for field in ("face", "age", "identity_group"):
                p[f"{field}_a"], p[f"{field}_b"] = p[f"{field}_b"], p[f"{field}_a"]
        oriented.append(p)
        negatives.append(
            dict(
                pair_id=f"orientneg_{arm.lower()}_{i:04d}",
                face_a=p["face_a"],
                face_b=b,
                label=0,
                split="train",
                pair_type="negative_oriented_exact_B_exposure_unique_edge",
                identity_group_a=p["identity_group_a"],
                identity_group_b=faces[b][2],
                age_a=p["age_a"],
                age_b=p["age_b"],
                age_gap=p["age_gap"],
                matched_target_pair_id=p["pair_id"],
                status="ok",
                hardness="oriented_exact_B_exposure",
            )
        )
    for old, new in zip(positives, oriented, strict=True):
        if (
            set((old["face_a"], old["face_b"])) != set((new["face_a"], new["face_b"]))
            or old["age_gap"] != new["age_gap"]
        ):
            raise ValueError("unordered positive intervention changed")
    output = oriented + negatives + copy.deepcopy([r for r in rows if r["split"] != "train"])
    checked = blocks([r for r in output if r["split"] == "train"], groups)
    report["age_balance"] = weighted_age_balance(checked, np.ones(len(checked)))
    if any(v["empirical_weighted_total_variation"] != 0 for v in report["age_balance"].values()):
        raise ValueError("class ages differ")
    exposures = [
        Counter(r[f"face_{s}"] for r in rs for s in ("a", "b")) for rs in (oriented, negatives)
    ]
    if exposures[0] != exposures[1]:
        raise ValueError("class image exposures differ")
    report.update(
        class_image_exposure_exact=True,
        images=len(exposures[0]),
        recorded_people=len({faces[f][1] for f in exposures[0]}),
        max_image_whole_pass_count=2 * max(exposures[0].values()),
    )
    return output, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("fresh destination required")
    root = PROJECT_ROOT
    prerequisite = root / "metrics/exact_supported_repair_v3_20261003/summary.manifest.json"
    inherited = verified_prerequisite(prerequisite, root)
    canonical = root / "data/processed/pairs.jsonl"
    deps = [
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
            x.resolve() for x in [*inherited, prerequisite, canonical, Path(__file__), *deps]
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
        candidates[arm], reports[arm] = oriented_arm(rows, groups, arm=arm, forbidden=forbidden)
    result = dict(
        arms=reports,
        protocol=PROTOCOL,
        scipy_version=scipy.__version__,
        execution_complete=True,
        both_arms_feasible=all(x is not None for x in candidates.values()),
        training_ready=False,
        publication_ready=False,
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
        experiment="joint-oriented-negative-exposure-feasibility",
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
