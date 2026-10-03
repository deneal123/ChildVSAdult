"""Global exact-B negative feasibility under fixed positive-image pools and unique edges."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import csr_array
from scipy.sparse.csgraph import maximum_bipartite_matching

from age_gap.common.io import PROJECT_ROOT, read_jsonl, write_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.audit_coupled_weight_budget import blocks, weighted_age_balance


def match_graph(candidates):
    """Rows are targets; columns are unordered face edges, never just B images."""
    edges = sorted({edge for options in candidates for edge in options})
    lookup = {edge: i for i, edge in enumerate(edges)}
    indices, indptr = [], [0]
    for options in candidates:
        indices.extend(sorted({lookup[edge] for edge in options}))
        indptr.append(len(indices))
    graph = csr_array(
        (
            np.ones(len(indices), dtype=np.int8),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(len(candidates), len(edges)),
    )
    assignments = maximum_bipartite_matching(graph, perm_type="column")
    if assignments.shape != (len(candidates),):
        raise ValueError("unexpected target-indexed matching shape")
    chosen = [edges[int(index)] if index >= 0 else None for index in assignments]
    if len([edge for edge in chosen if edge is not None]) != len(
        {edge for edge in chosen if edge is not None}
    ):
        raise ValueError("matching reused a negative edge")
    if any(
        edge is not None and edge not in options
        for edge, options in zip(chosen, candidates, strict=True)
    ):
        raise ValueError("matching returned an unlisted edge")
    return chosen, len(edges)


def exact_arm(rows, groups, *, arm, forbidden):
    old = blocks([r for r in rows if r["split"] == "train"], groups)
    positives = [p for p, _ in old]
    faces, by_age = {}, defaultdict(list)
    for positive in positives:
        for side in ("a", "b"):
            face, age, group = (
                positive[f"face_{side}"],
                positive[f"age_{side}"],
                positive[f"identity_group_{side}"],
            )
            person = groups[group]
            if face in faces and faces[face][:2] != (age, person):
                raise ValueError("conflicting positive-image age/person")
            faces[face] = (age, person, min(group, faces[face][2]) if face in faces else group)
    for face in sorted(faces):
        by_age[faces[face][0]].append(face)
    options = []
    for positive in positives:
        a = positive["face_a"]
        person = groups[positive["identity_group_a"]]
        candidates = {
            tuple(sorted((a, b)))
            for b in by_age[positive["age_b"]]
            if b != a and faces[b][1] != person and tuple(sorted((a, b))) not in forbidden
        }
        options.append(candidates)
    chosen, edge_count = match_graph(options)
    assignment, negatives = [], []
    for i, (positive, edge) in enumerate(zip(positives, chosen, strict=True)):
        row = {
            "target_positive_pair_id": positive["pair_id"],
            "exact_candidate_edges": len(options[i]),
            "matched": edge is not None,
        }
        if edge is not None:
            a = positive["face_a"]
            b = edge[1] if edge[0] == a else edge[0]
            row["negative_face_a"], row["negative_face_b"] = a, b
            negatives.append(
                {
                    "pair_id": f"exactneg_{arm.lower()}_{i:04d}",
                    "face_a": a,
                    "face_b": b,
                    "label": 0,
                    "split": "train",
                    "pair_type": "negative_exact_age_unique_edge_impostor",
                    "identity_group_a": positive["identity_group_a"],
                    "identity_group_b": faces[b][2],
                    "age_a": positive["age_a"],
                    "age_b": positive["age_b"],
                    "age_gap": positive["age_gap"],
                    "matched_target_pair_id": positive["pair_id"],
                    "status": "ok",
                    "hardness": "exact_age",
                }
            )
        assignment.append(row)
    complete = len(negatives) == len(positives)
    report = {
        "positive_targets": len(positives),
        "maximum_matched_targets": len(negatives),
        "unmatched_targets": len(positives) - len(negatives),
        "targets_without_exact_candidate": sum(not c for c in options),
        "candidate_unique_negative_edges": edge_count,
        "candidate_count_histogram": dict(sorted(Counter(map(len, options)).items())),
        "full_fixed_positive_protocol_feasible": complete,
        "positive_images": len(faces),
        "positive_recorded_people": len({person for _, person, _ in faces.values()}),
        "known_genuine_edges_excluded": True,
        "unique_unordered_negative_edges": True,
    }
    candidate = None
    if complete:
        candidate = (
            copy.deepcopy(positives)
            + negatives
            + copy.deepcopy([r for r in rows if r["split"] != "train"])
        )
        new_blocks = blocks([r for r in candidate if r["split"] == "train"], groups)
        if any(p["age_b"] != n["age_b"] for p, n in new_blocks):
            raise ValueError("exact matching changed target B age")
        report["uniform_age_balance"] = weighted_age_balance(new_blocks, np.ones(len(new_blocks)))
        if any(
            row["empirical_weighted_total_variation"] != 0
            for row in report["uniform_age_balance"].values()
        ):
            raise ValueError("unexpected nonzero exact-block class-age TV")
    return report, assignment, candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    root = PROJECT_ROOT
    prerequisite = root / "metrics/coupled_weight_budget_20261003/summary.manifest.json"
    native = json.loads(prerequisite.read_text(encoding="utf-8"))
    if (
        native["experiment"] != "coupled-pair-weight-objective-budget-audit"
        or native["metrics"]["execution_complete"] is not True
    ):
        raise ValueError("completed coupled-weight prerequisite required")
    records = native["inputs"] + native["outputs"]
    paths = [
        Path(r["path"]) if Path(r["path"]).is_absolute() else root / r["path"] for r in records
    ]
    if [file_record(p) for p in paths] != records:
        raise ValueError("prerequisite direct source records changed")
    arm_paths = [
        root / "data/interim/restricted_matched_arms_20261003/private" / f"{arm.lower()}_arm.jsonl"
        for arm in ("LOW", "CROSS")
    ]
    clusters = root / "data/processed/person_clusters.jsonl"
    if any(file_record(p) not in records for p in [*arm_paths, clusters]):
        raise ValueError("consumed arms/mapping not bound by prerequisite")
    canonical = root / "data/processed/pairs.jsonl"
    inputs = [*paths, prerequisite, canonical, Path(__file__)]
    before = [file_record(p) for p in inputs]
    groups = {}
    for row in read_jsonl(clusters):
        key, person = row["identity_group_id"], row["person_id"]
        if key in groups and groups[key] != person:
            raise ValueError("conflicting group-person mapping")
        groups[key] = person
    forbidden = set()
    for row in read_jsonl(canonical):
        if row["label"] == 1:
            forbidden.add(tuple(sorted((row["face_a"], row["face_b"]))))
    reports, assignments, candidates = {}, {}, {}
    for arm, path in zip(("LOW", "CROSS"), arm_paths, strict=True):
        reports[arm], assignments[arm], candidates[arm] = exact_arm(
            list(read_jsonl(path)), groups, arm=arm, forbidden=forbidden
        )
    complete = all(row["full_fixed_positive_protocol_feasible"] for row in reports.values())
    result = {
        "arms": reports,
        "execution_complete": True,
        "both_arms_full_exact_feasible": complete,
        "training_ready": False,
        "publication_ready": False,
        "scipy_version": scipy.__version__,
        "algorithm": "maximum bipartite cardinality matching, target->unordered edge",
        "solution_stability": "cardinality maximal; assignment may differ across SciPy/Python versions",
        "scope": "fixed selected positive images/people/heldout; no larger pool, row dropping or training",
        "identity_purity": "recorded person metadata only; human identity independence unverified",
        "remaining": [
            "if infeasible: redesign positive selection under exact support and equal budgets",
            "negative quality/exposure policy, source/crop-bound preflight and separate training",
        ],
    }
    if before != [file_record(p) for p in inputs]:
        raise ValueError("inputs changed during exact matching")
    private = args.out / "private"
    private.mkdir(parents=True)
    output = args.out / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    outputs = [output]
    for arm in reports:
        assignment_path = private / f"{arm.lower()}_assignment.jsonl"
        write_jsonl(assignment_path, assignments[arm])
        outputs.append(assignment_path)
        if complete:
            candidate_path = private / f"{arm.lower()}_candidate_arm.jsonl"
            write_jsonl(candidate_path, candidates[arm])
            outputs.append(candidate_path)
    target = args.out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="fixed-pool-exact-negative-global-matching",
        parameters={
            "max_age_error": 0,
            "scipy_version": scipy.__version__,
            "negative_edges_unique": True,
            "training_executed": False,
        },
        metrics=result,
        inputs=inputs,
        outputs=outputs,
    )
    if before != json.loads(target.read_text(encoding="utf-8"))["inputs"]:
        target.unlink()
        raise ValueError("inputs changed during manifest write; marker withdrawn")
    print(
        json.dumps(
            {
                "arms": {
                    arm: {
                        k: row[k]
                        for k in (
                            "positive_targets",
                            "maximum_matched_targets",
                            "targets_without_exact_candidate",
                            "full_fixed_positive_protocol_feasible",
                        )
                    }
                    for arm, row in reports.items()
                },
                "training_ready": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
