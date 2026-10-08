"""Outcome-free minimum-quality-cost assignment with exact ages and unique edges.

This is candidate preparation, not a training or identity-purity certificate.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from scripts.audit_coupled_weight_budget import blocks, weighted_age_balance
from scripts.audit_exact_negative_matching import match_graph
from scripts.audit_joint_pair_quality import ordered_features, standardize

PROTOCOL = {
    "version": "exact-negative-quality-cost-v1",
    "feature_map": "shared unique positive train images; nine quality fields plus nine missing indicators",
    "cost": "squared standardized B quality distance; target A is unchanged",
    "offset": 1.0,
    "capacity": "one target per unordered negative edge; B image reuse permitted and reported",
    "selection": "full cardinality then minimum total cost; no outcome-based tuning",
    "identity_scope": "recorded person metadata, not verified true identity",
    "training_ready": False,
}


def minimum_cost_edges(candidates):
    """Map edge->cost dictionaries to a full assignment, without dropping targets."""
    if not candidates:
        raise ValueError("nonempty targets required")
    for options in candidates:
        for edge, cost in options.items():
            if (
                not isinstance(edge, tuple)
                or len(edge) != 2
                or edge[0] >= edge[1]
                or not np.isfinite(cost)
                or cost < 0
            ):
                raise ValueError("sorted nonself edges and finite nonnegative costs required")
    cardinal, edge_count = match_graph(candidates)
    matched = sum(edge is not None for edge in cardinal)
    if matched != len(candidates):
        return None, {
            "matched_targets": matched,
            "targets": len(candidates),
            "unique_edges": edge_count,
        }
    edges = sorted({edge for options in candidates for edge in options})
    lookup = {edge: i for i, edge in enumerate(edges)}
    indices, data, indptr = [], [], [0]
    for options in candidates:
        for edge in sorted(options):
            indices.append(lookup[edge])
            # Sparse matching treats zero as absent. The same offset is added
            # to every admissible edge; full assignments contain the same N.
            cost = float(options[edge]) + PROTOCOL["offset"]
            if not np.isfinite(cost) or cost <= 0:
                raise ValueError("invalid offset cost")
            data.append(cost)
        indptr.append(len(indices))
    graph = csr_array(
        (np.asarray(data), np.asarray(indices, np.int32), np.asarray(indptr, np.int32)),
        shape=(len(candidates), len(edges)),
    )
    rows, columns = min_weight_full_bipartite_matching(graph)
    if not np.array_equal(rows, np.arange(len(candidates))):
        raise ValueError("solver did not return every target")
    chosen = [edges[int(c)] for c in columns]
    if len(set(chosen)) != len(chosen) or any(
        edge not in options for edge, options in zip(chosen, candidates, strict=True)
    ):
        raise ValueError("invalid solver assignment")
    return chosen, {
        "matched_targets": matched,
        "targets": len(candidates),
        "unique_edges": edge_count,
        "total_unshifted_cost": float(sum(c[e] for c, e in zip(candidates, chosen, strict=True))),
        "tie_policy": "sorted rows/columns; assignment may vary with solver version when costs tie",
    }


def positive_image_map(arms, groups, quality):
    """Fit only once on unique selected positive images, not labels/test scores."""
    faces = {}
    for rows in arms.values():
        for p, _ in blocks([r for r in rows if r["split"] == "train"], groups):
            for side in ("a", "b"):
                face = p[f"face_{side}"]
                meta = (p[f"age_{side}"], groups[p[f"identity_group_{side}"]])
                if face in faces and faces[face] != meta:
                    raise ValueError("conflicting positive image age/person")
                faces[face] = meta
    if not set(faces) <= set(quality):
        raise ValueError("unique quality record required for every selected image")
    ids = sorted(faces)
    dummy = [{"face_a": f, "face_b": f} for f in ids]
    ordered = ordered_features(dummy, quality)
    # A/B copies are identical; retain A values and A missing indicators.
    matrix = np.concatenate((ordered[:, :9], ordered[:, 18:27]), axis=1)
    transformed, fitted = standardize(matrix)
    fitted["scope"] = (
        "unique selected positive train images pooled across arms; frozen for all candidate costs"
    )
    return dict(zip(ids, transformed, strict=True)), fitted


def quality_arm(rows, groups, features, *, arm, forbidden):
    positives = [p for p, _ in blocks([r for r in rows if r["split"] == "train"], groups)]
    faces, by_age = {}, defaultdict(list)
    for p in positives:
        for side in ("a", "b"):
            face, age, group = (p[f"face_{side}"], p[f"age_{side}"], p[f"identity_group_{side}"])
            person = groups[group]
            if face in faces and faces[face][:2] != (age, person):
                raise ValueError("conflicting image age/person")
            faces[face] = (age, person, min(group, faces[face][2]) if face in faces else group)
    if not set(faces) <= set(features):
        raise ValueError("missing selected image features")
    vectors = np.asarray([features[f] for f in sorted(faces)], float)
    if vectors.ndim != 2 or vectors.shape[1] != 18 or not np.isfinite(vectors).all():
        raise ValueError("aligned finite 18-dimensional image features required")
    for face in sorted(faces):
        by_age[faces[face][0]].append(face)
    candidates = []
    for p in positives:
        a, target = p["face_a"], p["face_b"]
        options = {}
        for b in by_age[p["age_b"]]:
            edge = tuple(sorted((a, b)))
            if b != a and faces[b][1] != faces[a][1] and edge not in forbidden:
                diff = np.asarray(features[target]) - features[b]
                options[edge] = float(diff @ diff)
        candidates.append(options)
    chosen, report = minimum_cost_edges(candidates)
    report["training_ready"] = False
    if chosen is None:
        return None, report
    negatives = []
    for i, (p, edge) in enumerate(zip(positives, chosen, strict=True)):
        a = p["face_a"]
        b = edge[1] if edge[0] == a else edge[0]
        negatives.append(
            {
                "pair_id": f"qualityneg_{arm.lower()}_{i:04d}",
                "face_a": a,
                "face_b": b,
                "label": 0,
                "split": "train",
                "pair_type": "negative_exact_age_quality_cost_unique_edge",
                "identity_group_a": p["identity_group_a"],
                "identity_group_b": faces[b][2],
                "age_a": p["age_a"],
                "age_b": p["age_b"],
                "age_gap": p["age_gap"],
                "matched_target_pair_id": p["pair_id"],
                "status": "ok",
                "hardness": "exact_age_quality",
            }
        )
    output = (
        copy.deepcopy(positives)
        + negatives
        + copy.deepcopy([r for r in rows if r["split"] != "train"])
    )
    paired = blocks([r for r in output if r["split"] == "train"], groups)
    report["age_balance"] = weighted_age_balance(paired, np.ones(len(paired)))
    if any(v["empirical_weighted_total_variation"] != 0 for v in report["age_balance"].values()):
        raise ValueError("exact class-age invariant violated")
    exposure = Counter(
        r[f"face_{side}"] for r in output if r["split"] == "train" for side in ("a", "b")
    )
    person_mass = Counter()
    for f, n in exposure.items():
        person_mass[faces[f][1]] += n
    report["endpoint_presentations"] = {
        "total": sum(exposure.values()),
        "images": len(exposure),
        "recorded_people": len(person_mass),
        "max_image_count": max(exposure.values()),
        "max_person_count": max(person_mass.values()),
        "image_count_histogram": dict(sorted(Counter(exposure.values()).items())),
        "person_count_histogram": dict(sorted(Counter(person_mass.values()).items())),
        "scope": "one complete unweighted pass, not minibatch BN policy or independent sample size",
    }
    return output, report
