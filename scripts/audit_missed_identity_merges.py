"""Read-only audit for possible missed identity merges across data splits.

The audit joins post-level identity groups to the production person map, assigns
split membership from canonical pairs, and searches cached independent FaceNet
group-centroid vectors for cross-split candidates. Candidate identifiers and
scores are written only to a private interim file; the manifest contains counts
and file hashes, not candidate rows or biometric vectors.

Retrieval is CPU-only blocked exhaustive cosine top-k over group centroids. It
does not guarantee recall of face-level matches that are diluted by a centroid,
fall below the threshold, or rank below top-k. Candidates require manual review.
No canonical data are changed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from age_gap.common.io import data_path, read_jsonl, write_jsonl
from age_gap.common.manifest import write_experiment_manifest

VALID_SPLITS = {"train", "val", "test"}


def split_membership(pairs: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Return post-group split sets, using both endpoints of every canonical pair."""
    memberships: dict[str, set[str]] = defaultdict(set)
    for row in pairs:
        split = row.get("split")
        if split not in VALID_SPLITS:
            continue
        for key in ("identity_group_a", "identity_group_b"):
            group_id = row.get(key)
            if group_id:
                memberships[str(group_id)].add(str(split))
    return dict(memberships)


def group_centroids(
    groups: list[dict[str, Any]],
    person_by_group: dict[str, str],
    splits_by_group: dict[str, set[str]],
    embedding_ids: list[str],
    embeddings: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build normalized independent-embedding centroids with explicit join counts."""
    if embeddings.ndim != 2 or len(embedding_ids) != embeddings.shape[0]:
        raise ValueError("embedding ids and matrix shape do not agree")
    vector_by_face = {face_id: embeddings[index] for index, face_id in enumerate(embedding_ids)}
    rows: list[dict[str, Any]] = []
    mapped_group_ids = {str(row["identity_group_id"]) for row in groups}
    missing_person_map = mapped_group_ids - set(person_by_group)
    for group in groups:
        group_id = str(group["identity_group_id"])
        person_id = person_by_group.get(group_id)
        split_set = splits_by_group.get(group_id, set())
        if person_id is None or len(split_set) != 1:
            continue
        vectors = [vector_by_face[str(face)] for face in group.get("faces", []) if str(face) in vector_by_face]
        if not vectors:
            continue
        centroid = np.mean(np.stack(vectors).astype(np.float32, copy=False), axis=0)
        norm = float(np.linalg.norm(centroid))
        if not np.isfinite(norm) or norm == 0:
            continue
        rows.append(
            {
                "group_id": group_id,
                "person_id": person_id,
                "split": next(iter(split_set)),
                "face_count_with_embedding": len(vectors),
                "centroid": centroid / norm,
            }
        )
    stats = {
        "post_snapshot_groups": len(groups),
        "post_snapshot_groups_joined_to_production_person_map": len(mapped_group_ids - missing_person_map),
        "post_snapshot_groups_missing_production_person_map": len(missing_person_map),
        "groups_assigned_to_exactly_one_split": sum(len(splits_by_group.get(g, set())) == 1 for g in mapped_group_ids),
        "groups_with_split_conflict": sum(len(splits_by_group.get(g, set())) > 1 for g in mapped_group_ids),
        "groups_without_pair_split_assignment": sum(g not in splits_by_group for g in mapped_group_ids),
        "groups_with_embedding_centroid": len(rows),
        "faces_with_embedding": len(set(embedding_ids)),
    }
    return rows, stats


def audit_candidates(
    groups: list[dict[str, Any]],
    person_by_group: dict[str, str],
    splits_by_group: dict[str, set[str]],
    embedding_ids: list[str],
    embeddings: np.ndarray,
    *,
    threshold: float = 0.65,
    top_k: int = 5,
    block_size: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search distinct-person groups in other splits using bounded-memory exact top-k."""
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [-1, 1]")
    if top_k <= 0 or block_size <= 0:
        raise ValueError("top_k and block_size must be positive")
    rows, joins = group_centroids(groups, person_by_group, splits_by_group, embedding_ids, embeddings)
    if not rows:
        raise ValueError("no group centroids remain after verified metadata joins")
    matrix = np.stack([row["centroid"] for row in rows]).astype(np.float32, copy=False)
    split_labels = [row["split"] for row in rows]
    person_ids = [row["person_id"] for row in rows]
    group_ids = [row["group_id"] for row in rows]
    split_code_lookup = {value: index for index, value in enumerate(sorted(set(split_labels)))}
    split_codes = np.asarray([split_code_lookup[value] for value in split_labels], dtype=np.int8)
    person_code_lookup = {value: index for index, value in enumerate(sorted(set(person_ids)))}
    person_codes = np.asarray([person_code_lookup[value] for value in person_ids], dtype=np.int32)
    chosen: dict[tuple[str, str], float] = {}

    for start in range(0, len(rows), block_size):
        stop = min(start + block_size, len(rows))
        # Bounded B x N temporary; the complete N x N similarity matrix is never materialized.
        block = matrix[start:stop] @ matrix.T
        for offset, scores in enumerate(block):
            i = start + offset
            eligible = (split_codes != split_codes[i]) & (person_codes != person_codes[i])
            eligible[i] = False
            scores = scores.copy()
            scores[~eligible] = -np.inf
            finite_count = int(np.isfinite(scores).sum())
            if not finite_count:
                continue
            take = min(top_k, finite_count)
            indices = np.argpartition(scores, -take)[-take:]
            indices = sorted(indices.tolist(), key=lambda j: (-float(scores[j]), group_ids[j]))
            for j in indices:
                similarity = float(scores[j])
                if similarity < threshold:
                    continue
                pair = tuple(sorted((group_ids[i], group_ids[j])))
                chosen[pair] = max(similarity, chosen.get(pair, -1.0))

    private_candidates = [
        {
            "group_id_a": left,
            "person_id_a": person_by_group[left],
            "split_a": next(iter(splits_by_group[left])),
            "group_id_b": right,
            "person_id_b": person_by_group[right],
            "split_b": next(iter(splits_by_group[right])),
            "group_centroid_cosine": round(score, 6),
            "review_status": "unreviewed_candidate",
        }
        for (left, right), score in sorted(chosen.items(), key=lambda item: (-item[1], item[0]))
    ]
    by_split_pair: Counter[str] = Counter()
    for row in private_candidates:
        key = "-".join(sorted((row["split_a"], row["split_b"])))
        by_split_pair[key] += 1
    person_splits: dict[str, set[str]] = defaultdict(set)
    for group_id, split_set in splits_by_group.items():
        person_id = person_by_group.get(group_id)
        if person_id is not None:
            person_splits[person_id].update(split_set)
    summary = {
        **joins,
        "canonical_pair_split_assigned_groups": len(splits_by_group),
        "canonical_pair_groups_joined_to_production_person_map": sum(
            group_id in person_by_group for group_id in splits_by_group
        ),
        "production_person_ids_checked_for_split_conflict": len(person_splits),
        "production_person_ids_assigned_to_multiple_splits": sum(
            len(split_set) > 1 for split_set in person_splits.values()
        ),
        "candidate_pairs_above_centroid_cosine_threshold": len(private_candidates),
        "candidate_pairs_by_centroid_threshold": {
            f"{cutoff:.2f}": sum(row["group_centroid_cosine"] >= cutoff for row in private_candidates)
            for cutoff in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
        },
        "candidate_pairs_by_split_pair": dict(sorted(by_split_pair.items())),
        "threshold": threshold,
        "top_k_per_group": top_k,
        "retrieval": "CPU blocked exhaustive cosine over normalized group centroids; per-query top-k",
        "peak_similarity_block_shape": [min(block_size, len(rows)), len(rows)],
        "face_level_recall_guaranteed": False,
        "manual_review_required": True,
    }
    return private_candidates, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=data_path("embeddings_cache_dir", "independent_facenet.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_path("data_dir", "interim", "missed_identity_merge_audit"),
    )
    args = parser.parse_args()

    group_path = data_path("data_dir", "processed", "identity_groups.jsonl.post_bak")
    person_path = data_path("data_dir", "processed", "person_clusters.jsonl")
    pairs_path = data_path("data_dir", "processed", "pairs.jsonl")
    groups = list(read_jsonl(group_path))
    person_by_group = {
        str(row["identity_group_id"]): str(row["person_id"])
        for row in read_jsonl(person_path)
    }
    pairs = list(read_jsonl(pairs_path))
    splits_by_group = split_membership(pairs)
    archive = np.load(args.embeddings, allow_pickle=True)
    embedding_ids = [str(value) for value in archive["face_ids"]]
    embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    candidates, summary = audit_candidates(
        groups,
        person_by_group,
        splits_by_group,
        embedding_ids,
        embeddings,
        threshold=args.threshold,
        top_k=args.top_k,
        block_size=args.block_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "private_candidates.jsonl"
    summary_path = args.output_dir / "summary.json"
    manifest_path = args.output_dir / "manifest.json"
    write_jsonl(candidate_path, candidates)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        manifest_path,
        experiment="read-only-missed-identity-merge-split-audit",
        parameters={
            "threshold": args.threshold,
            "top_k_per_group": args.top_k,
            "block_size": args.block_size,
            "embedding_model": "FaceNet InceptionResnetV1 CASIA-WebFace (existing cache)",
            "identity_join": "post-level snapshot group_id -> production person_clusters group_id; unmatched groups excluded and counted",
            "split_source": "canonical pairs.jsonl endpoint group IDs",
            "public_output_policy": "summary/manifest are aggregate-only; candidate IDs and scores are private interim artifacts",
            "mutation_policy": "read-only; no merge, resplit, image access, GPU, or network API",
        },
        metrics=summary,
        inputs=[group_path, person_path, pairs_path, args.embeddings],
        outputs=[summary_path, candidate_path],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
