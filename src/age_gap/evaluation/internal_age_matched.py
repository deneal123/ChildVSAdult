"""Endpoint-age-matched internal verification protocol.

Each eligible positive is retained and paired with one different-identity
impostor whose age matches the positive's second endpoint. This keeps the
anchor fixed and matches both endpoints instead of matching negatives only
within a coarse age bucket.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from age_gap.common.schemas import IdentityGroup, Pair
from age_gap.evaluation.metrics import roc_auc, verification_metrics


def build_endpoint_age_matched_pairs(
    positives: list[Pair],
    groups: list[IdentityGroup],
    *,
    group_splits: dict[str, str],
    seed: int = 42,
    tolerance_years: int = 2,
    min_positive_gap: int = 25,
    split: str = "test",
) -> tuple[list[Pair], dict[str, Any]]:
    """Return eligible positives plus deterministic, endpoint-matched negatives.

    One negative is attempted per eligible positive. Sampling is without
    replacement over negative face-pairs. If no candidate is available within
    the age tolerance, the positive remains in the coverage denominator but is
    omitted from the balanced evaluation set.
    """
    if tolerance_years < 0:
        raise ValueError("tolerance_years must be non-negative")

    age_by_face: dict[str, int] = {}
    group_faces: dict[str, list[str]] = {}
    for group in groups:
        if group.status not in {"auto", "mixed", "manual_verified"}:
            continue
        group_faces[group.identity_group_id] = list(group.faces)
        for label in group.age_labels:
            if label.face_id in group.faces and label.age is not None:
                age_by_face[label.face_id] = label.age

    eligible = [
        pair
        for pair in positives
        if pair.label == 1
        and pair.split == split
        and pair.age_a is not None
        and pair.age_b is not None
        and pair.age_gap is not None
        and pair.age_gap >= min_positive_gap
        and pair.identity_group_a is not None
        and pair.identity_group_a == pair.identity_group_b
    ]
    eligible.sort(key=lambda pair: pair.pair_id)
    rng = random.Random(seed)
    used_negative_pairs: set[tuple[str, str]] = set()
    matched_negatives: list[Pair] = []
    matched_positive_ids: set[str] = set()
    candidate_counts: list[int] = []
    endpoint_age_errors: list[int] = []

    for positive in eligible:
        anchor = positive.face_a
        anchor_group = positive.identity_group_a or ""
        target_age = positive.age_b
        candidates: list[tuple[str, str, int]] = []
        for group_id, face_ids in group_faces.items():
            if group_id == anchor_group or group_splits.get(group_id) != split:
                continue
            for face_id in face_ids:
                candidate_age = age_by_face.get(face_id)
                if candidate_age is None or target_age is None:
                    continue
                error = abs(candidate_age - target_age)
                key = tuple(sorted((anchor, face_id)))
                if error <= tolerance_years and key not in used_negative_pairs:
                    candidates.append((face_id, group_id, error))
        candidate_counts.append(len(candidates))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[2], item[0], item[1]))
        best_error = candidates[0][2]
        best_candidates = [item for item in candidates if item[2] == best_error]
        impostor, impostor_group, age_error = rng.choice(best_candidates)
        key = tuple(sorted((anchor, impostor)))
        used_negative_pairs.add(key)
        endpoint_age_errors.append(age_error)
        matched_negatives.append(
            Pair(
                pair_id=f"endpoint_age_neg_{positive.pair_id}_{impostor}",
                face_a=anchor,
                face_b=impostor,
                label=0,
                pair_type="negative_endpoint_age_matched",
                identity_group_a=anchor_group,
                identity_group_b=impostor_group,
                age_a=positive.age_a,
                age_b=age_by_face[impostor],
                age_gap=abs(positive.age_a - age_by_face[impostor]),
                hardness="medium",
                status="ok",
                split=split,
            )
        )
        matched_positive_ids.add(positive.pair_id)

    eval_positives = [p for p in eligible if p.pair_id in matched_positive_ids]
    evaluation_pairs = [*eval_positives, *matched_negatives]
    positive_gaps = [int(p.age_gap) for p in eval_positives if p.age_gap is not None]
    negative_gaps = [int(p.age_gap) for p in matched_negatives if p.age_gap is not None]
    positive_endpoint_ages = [int(p.age_b) for p in eval_positives if p.age_b is not None]
    negative_endpoint_ages = [int(p.age_b) for p in matched_negatives if p.age_b is not None]
    age_cue_auc = (
        roc_auc(
            np.asarray([*positive_gaps, *negative_gaps], dtype=float),
            np.asarray([1] * len(positive_gaps) + [0] * len(negative_gaps)),
        )
        if positive_gaps and negative_gaps
        else float("nan")
    )
    coverage = len(matched_negatives) / len(eligible) if eligible else 0.0
    diagnostics = {
        "protocol": "fixed-anchor, counterpart-endpoint age match; nearest age then seeded tie-break",
        "seed": seed,
        "split": split,
        "min_positive_gap_years": min_positive_gap,
        "endpoint_age_tolerance_years": tolerance_years,
        "eligible_positive_count": len(eligible),
        "matched_positive_count": len(eval_positives),
        "unmatched_positive_count": len(eligible) - len(eval_positives),
        "coverage": coverage,
        "candidate_count_per_positive": {
            "min": min(candidate_counts, default=0),
            "median": float(np.median(candidate_counts)) if candidate_counts else 0.0,
            "max": max(candidate_counts, default=0),
            "with_candidates": sum(count > 0 for count in candidate_counts),
        },
        "matched_endpoint_age_error_years": {
            "mean": float(np.mean(endpoint_age_errors)) if endpoint_age_errors else None,
            "max": max(endpoint_age_errors, default=None),
        },
        "positive_age_gap_years": _distribution(positive_gaps),
        "negative_age_gap_years": _distribution(negative_gaps),
        "positive_counterpart_age_years": _distribution(positive_endpoint_ages),
        "negative_impostor_age_years": _distribution(negative_endpoint_ages),
        "age_gap_only_roc_auc": age_cue_auc,
        "age_gap_only_predictive_auc": (
            max(age_cue_auc, 1.0 - age_cue_auc) if np.isfinite(age_cue_auc) else float("nan")
        ),
        "age_gap_class_mean_difference_years": (
            float(np.mean(positive_gaps) - np.mean(negative_gaps))
            if positive_gaps and negative_gaps
            else None
        ),
        "unmatched_reason": "no_other_identity_face_within_tolerance",
    }
    return evaluation_pairs, diagnostics


def evaluate_endpoint_age_matched(
    pairs: list[Pair], embeddings: dict[str, np.ndarray]
) -> tuple[dict[str, float], int]:
    scores: list[float] = []
    labels: list[int] = []
    skipped = 0
    for pair in pairs:
        left, right = embeddings.get(pair.face_a), embeddings.get(pair.face_b)
        if left is None or right is None:
            skipped += 1
            continue
        scores.append(float(np.dot(left, right)))
        labels.append(pair.label)
    return verification_metrics(np.asarray(scores), np.asarray(labels)), skipped


def split_matched_blocks(pairs: list[Pair]) -> tuple[list[Pair], list[Pair]]:
    """Decode the generator's ordered positive block and aligned negative block."""
    if len(pairs) % 2:
        raise ValueError("matched pair file must contain equal positive and negative blocks")
    half = len(pairs) // 2
    positives, negatives = pairs[:half], pairs[half:]
    if any(pair.label != 1 for pair in positives) or any(pair.label != 0 for pair in negatives):
        raise ValueError("matched pair file must store positives followed by aligned negatives")
    if len(positives) != len(negatives):
        raise ValueError("matched pair file is not class balanced")
    return positives, negatives


def filter_matched_blocks_by_faces(
    positives: list[Pair], negatives: list[Pair], available_faces: set[str]
) -> tuple[list[Pair], list[Pair], int]:
    """Drop whole matched observations so missing crops cannot unbalance classes."""
    if len(positives) != len(negatives):
        raise ValueError("positive and negative matched blocks must have equal length")
    kept_pos: list[Pair] = []
    kept_neg: list[Pair] = []
    dropped = 0
    for positive, negative in zip(positives, negatives, strict=True):
        if positive.identity_group_a != negative.identity_group_a:
            raise ValueError("positive and negative at the same index are not a matched observation")
        needed = {positive.face_a, positive.face_b, negative.face_a, negative.face_b}
        if needed <= available_faces:
            kept_pos.append(positive)
            kept_neg.append(negative)
        else:
            dropped += 1
    return kept_pos, kept_neg, dropped


def paired_subject_bootstrap(
    positives: list[Pair],
    negatives: list[Pair],
    positive_scores: dict[str, dict[str, float]],
    negative_scores: dict[str, dict[str, float]],
    *,
    reference: str = "facenet_frozen",
    comparison: str = "facenet_tuned",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Cluster bootstrap matched comparisons by identity, returning paired AUC CIs.

    Each matched observation depends on the positive identity and the sampled
    impostor identity, so its bootstrap weight is the product of their subject
    multiplicities. That same weight is applied to its positive and negative
    score, preserving both pairing and class balance.
    """
    if len(positives) != len(negatives) or not positives:
        raise ValueError("paired bootstrap requires non-empty balanced matched blocks")
    subjects = sorted(
        {
            pair.identity_group_a
            for pair in positives
            if pair.identity_group_a is not None
        }
        | {
            pair.identity_group_b
            for pair in negatives
            if pair.identity_group_b is not None
        }
    )
    if not subjects:
        raise ValueError("matched observations have no subject identifiers")
    if any(pair.identity_group_a is None for pair in positives) or any(
        pair.identity_group_b is None for pair in negatives
    ):
        raise ValueError("matched observations require both subject identifiers")

    labels = np.r_[np.ones(len(positives), dtype=int), np.zeros(len(negatives), dtype=int)]
    point_auc: dict[str, float] = {}
    for tag in (reference, comparison):
        point_scores = [positive_scores[tag][p.pair_id] for p in positives] + [
            negative_scores[tag][p.pair_id] for p in negatives
        ]
        point_auc[tag] = roc_auc(np.asarray(point_scores), labels)
    rng = np.random.default_rng(seed)
    bootstrap_auc = {reference: [], comparison: [], "delta": []}
    subject_index = {subject: i for i, subject in enumerate(subjects)}
    positive_subjects = [subject_index[p.identity_group_a] for p in positives]
    impostor_subjects = [subject_index[n.identity_group_b] for n in negatives]
    for _ in range(n_boot):
        sampled = rng.integers(0, len(subjects), size=len(subjects))
        multiplicities = np.bincount(sampled, minlength=len(subjects))
        weights = multiplicities[positive_subjects] * multiplicities[impostor_subjects]
        indices = np.repeat(np.arange(len(positives)), weights)
        if len(indices) == 0:
            continue
        # Keep identical sampled observations for both models and both classes.
        for tag in (reference, comparison):
            pos = np.asarray([positive_scores[tag][pair.pair_id] for pair in positives])[indices]
            neg = np.asarray([negative_scores[tag][pair.pair_id] for pair in negatives])[indices]
            labels_boot = np.r_[np.ones(len(indices), dtype=int), np.zeros(len(indices), dtype=int)]
            bootstrap_auc[tag].append(roc_auc(np.r_[pos, neg], labels_boot))
        bootstrap_auc["delta"].append(bootstrap_auc[comparison][-1] - bootstrap_auc[reference][-1])

    def ci(values: list[float]) -> list[float | None]:
        if not values:
            return [None, None]
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    return {
        "method": "paired identity-cluster bootstrap; dyad multiplicity product; percentile 95% CI",
        "seed": seed,
        "requested_replicates": n_boot,
        "valid_replicates": len(bootstrap_auc["delta"]),
        "subject_count": len(subjects),
        "matched_observation_count": len(positives),
        "point_estimates": {
            reference: point_auc[reference],
            comparison: point_auc[comparison],
            "delta_comparison_minus_reference": point_auc[comparison] - point_auc[reference],
        },
        "ci95": {
            reference: ci(bootstrap_auc[reference]),
            comparison: ci(bootstrap_auc[comparison]),
            "delta_comparison_minus_reference": ci(bootstrap_auc["delta"]),
        },
    }


def leave_one_subject_out(
    positives: list[Pair],
    negatives: list[Pair],
    positive_scores: dict[str, dict[str, float]],
    negative_scores: dict[str, dict[str, float]],
    *,
    reference: str = "facenet_frozen",
    comparison: str = "facenet_tuned",
) -> dict[str, Any]:
    """Recompute paired AUC deltas after removing each subject and linked pairs."""
    if len(positives) != len(negatives):
        raise ValueError("leave-one-subject-out requires balanced matched blocks")
    subjects = sorted(
        {
            p.identity_group_a for p in positives if p.identity_group_a is not None
        }
        | {
            n.identity_group_b for n in negatives if n.identity_group_b is not None
        }
    )
    rows: list[float] = []
    for subject in subjects:
        keep = [
            i
            for i, (positive, negative) in enumerate(zip(positives, negatives, strict=True))
            if positive.identity_group_a != subject and negative.identity_group_b != subject
        ]
        if len(keep) < 2:
            continue
        labels = np.r_[np.ones(len(keep), dtype=int), np.zeros(len(keep), dtype=int)]
        auc_by_model: dict[str, float] = {}
        for tag in (reference, comparison):
            pos = [positive_scores[tag][positives[i].pair_id] for i in keep]
            neg = [negative_scores[tag][negatives[i].pair_id] for i in keep]
            auc_by_model[tag] = roc_auc(np.r_[pos, neg], labels)
        rows.append(auc_by_model[comparison] - auc_by_model[reference])
    if not rows:
        return {"method": "leave one subject out; remove linked matched observations", "n": 0}
    return {
        "method": "leave one subject out; remove linked matched observations",
        "n": len(rows),
        "delta_comparison_minus_reference": {
            "mean": float(np.mean(rows)),
            "median": float(np.median(rows)),
            "min": float(np.min(rows)),
            "max": float(np.max(rows)),
            "p05": float(np.percentile(rows, 5)),
            "p95": float(np.percentile(rows, 95)),
            "positive_fraction": float(np.mean(np.asarray(rows) > 0)),
        },
    }


def _distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": min(values),
        "max": max(values),
    }
