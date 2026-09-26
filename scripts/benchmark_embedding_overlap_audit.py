"""Retrieve mined-training/benchmark image pairs with cached FaceNet embeddings.

This is a candidate screen only. It uses the existing independent CASIA-WebFace
FaceNet cache for mined crops and locally re-embeds benchmark crops on CPU. All
row-level outputs are private ignored artifacts; no identity matches are
confirmed by this script.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from age_gap.common.io import PROJECT_ROOT, data_path
from age_gap.common.manifest import sha256_file, write_experiment_manifest
from age_gap.models.facenet import FaceNetBackbone, preprocess_bgr

if __package__:
    from scripts.benchmark_image_overlap_audit import (
        DEFAULT_OUTPUT,
        _benchmark_images,
        _training_images,
    )
else:  # direct ``python scripts/...`` invocation
    from benchmark_image_overlap_audit import DEFAULT_OUTPUT, _benchmark_images, _training_images

DEFAULT_EMBEDDINGS = data_path("embeddings_cache_dir", "independent_facenet.npz")
DEFAULT_CHECKPOINT = Path(torch.hub.get_dir()).parent / "checkpoints" / "20180408-102900-casia-webface.pt"
SIMILARITY_THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
CANDIDATE_SIMILARITY_THRESHOLD = 0.7
TOP_K = 5


def retrieve_candidates(
    train_embeddings: np.ndarray,
    benchmark_embeddings: np.ndarray,
    *,
    thresholds: tuple[float, ...] = SIMILARITY_THRESHOLDS,
    candidate_threshold: float = CANDIDATE_SIMILARITY_THRESHOLD,
    top_k: int = TOP_K,
    query_batch_size: int = 128,
) -> tuple[list[dict[str, int | float | str]], dict[str, int]]:
    """Return union of per-query top-k and cosine-threshold candidate pairs."""
    train = np.asarray(train_embeddings, dtype=np.float32)
    query = np.asarray(benchmark_embeddings, dtype=np.float32)
    if train.ndim != 2 or query.ndim != 2 or train.shape[1] != query.shape[1]:
        raise ValueError("embedding matrices must be 2D with the same feature dimension")
    if not len(train) or not len(query):
        raise ValueError("embedding matrices must not be empty")
    if top_k < 1 or query_batch_size < 1:
        raise ValueError("top_k and query_batch_size must be positive")
    if candidate_threshold not in thresholds:
        raise ValueError("candidate_threshold must be present in the reported thresholds")

    train = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    rows: dict[tuple[int, int], dict[str, int | float | str]] = {}
    counts = {f"cosine_ge_{threshold:.2f}": 0 for threshold in thresholds}
    for start in range(0, len(query), query_batch_size):
        similarities = query[start : start + query_batch_size] @ train.T
        for local_index, scores in enumerate(similarities):
            benchmark_index = start + local_index
            k = min(top_k, len(train))
            nearest = np.argpartition(scores, -k)[-k:]
            nearest = nearest[np.argsort(scores[nearest])[::-1]]
            for rank, train_index in enumerate(nearest, start=1):
                key = (int(train_index), benchmark_index)
                if key not in rows:
                    rows[key] = {
                        "train_index": key[0],
                        "benchmark_index": benchmark_index,
                        "cosine_similarity": float(scores[train_index]),
                        "rank": rank,
                        "match_type": "top_k",
                    }
                else:
                    row = rows[key]
                    row["match_type"] = "top_k_and_threshold"
                if rank < int(rows[key]["rank"]):
                    row = rows[key]
                    row["rank"] = rank
            for threshold in thresholds:
                hits = np.flatnonzero(scores >= threshold)
                counts[f"cosine_ge_{threshold:.2f}"] += int(len(hits))
                if threshold < candidate_threshold:
                    continue
                for train_index in hits:
                    key = (int(train_index), benchmark_index)
                    if key not in rows:
                        rows[key] = {
                            "train_index": key[0],
                            "benchmark_index": benchmark_index,
                            "cosine_similarity": float(scores[train_index]),
                            "rank": -1,
                            "match_type": "cosine_threshold",
                        }
                    elif int(rows[key]["rank"]) > 0:
                        rows[key]["match_type"] = "top_k_and_threshold"
    return sorted(rows.values(), key=lambda row: (int(row["benchmark_index"]), int(row["rank"]) or 10**9)), counts


def _load_training_embeddings(path: Path, training_ids: list[str]) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        cached_ids = [str(value) for value in archive["face_ids"]]
        matrix = np.asarray(archive["embeddings"], dtype=np.float32)
    by_id = {face_id: index for index, face_id in enumerate(cached_ids)}
    retained_ids = [face_id for face_id in training_ids if face_id in by_id]
    if len(retained_ids) != len(training_ids):
        raise RuntimeError(f"independent cache misses {len(training_ids) - len(retained_ids)} training crops")
    return retained_ids, matrix[[by_id[face_id] for face_id in retained_ids]]


def _embed_benchmarks(benchmarks: dict[str, list[np.ndarray]], checkpoint: Path, batch_size: int) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Local FaceNet checkpoint not found; network loading is disabled: {checkpoint}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = FaceNetBackbone(pretrained=None).cpu().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.net.load_state_dict(state, strict=False)
    if incompatible.missing_keys or set(incompatible.unexpected_keys) != {"logits.weight", "logits.bias"}:
        raise RuntimeError(
            "local FaceNet checkpoint does not match the expected CASIA-WebFace architecture: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    outputs: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    with torch.inference_mode():
        for name, images in benchmarks.items():
            print(f"Embedding {name}: {len(images)} benchmark crops on CPU", flush=True)
            parts: list[np.ndarray] = []
            for start in range(0, len(images), batch_size):
                batch = np.stack([preprocess_bgr(image) for image in images[start : start + batch_size]])
                parts.append(model(torch.from_numpy(batch)).cpu().numpy().astype(np.float32))
            outputs[name] = np.concatenate(parts) if parts else np.empty((0, 512), dtype=np.float32)
            counts[name] = len(images)
    return outputs, counts


def run_audit(
    *,
    output_dir: Path = PROJECT_ROOT / DEFAULT_OUTPUT,
    embeddings_path: Path = DEFAULT_EMBEDDINGS,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    batch_size: int = 32,
    query_batch_size: int = 128,
    reuse_benchmark_cache: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    training = _training_images()
    train_ids, train_matrix = _load_training_embeddings(embeddings_path, [face_id for face_id, _ in training])
    external = Path(str(data_path("data_dir", "external")))
    benchmark_cache = output_dir / "private_benchmark_facenet_embeddings.npz"
    if reuse_benchmark_cache and benchmark_cache.is_file():
        benchmark_paths = {
            "FG-NET": external / "fgnet_crops.npz",
            "AgeDB-30": external / "agedb_30.bin",
            "CALFW": external / "calfw.bin",
            "LFW": external / "lfw_aligned.npz",
            "CACD-VS": external / "cacd_vs_aligned.npz",
        }
        with np.load(benchmark_cache, allow_pickle=False) as archive:
            benchmark_matrix = {name: np.asarray(archive[name], dtype=np.float32) for name in benchmark_paths}
        benchmark_counts = {name: len(matrix) for name, matrix in benchmark_matrix.items()}
    else:
        benchmarks, benchmark_paths = _benchmark_images(external)
        benchmark_matrix, benchmark_counts = _embed_benchmarks(benchmarks, checkpoint, batch_size)
        np.savez_compressed(benchmark_cache, **benchmark_matrix)

    inventory = output_dir / "private_embedding_training_inventory.jsonl"
    with inventory.open("w", encoding="utf-8") as handle:
        for index, face_id in enumerate(train_ids):
            handle.write(json.dumps({"train_index": index, "face_id": face_id}, ensure_ascii=False) + "\n")

    candidate_path = output_dir / "private_embedding_candidates.jsonl"
    summary: dict[str, Any] = {
        "status": "embedding_candidate_search_complete_manual_review_required",
        "confirmed_identity_overlaps": None,
        "verified_zero_overlap": False,
        "embedding_model": "FaceNet InceptionResnetV1 CASIA-WebFace; local checkpoint; CPU inference",
        "training_embedding_cache": str(embeddings_path),
        "top_k": TOP_K,
        "candidate_similarity_threshold": CANDIDATE_SIMILARITY_THRESHOLD,
        "benchmarks": {},
        "private_candidate_file": candidate_path.name,
        "privacy": "Candidate rows contain private image indices; keep in ignored interim storage.",
        "interpretation": "Cosine retrieval yields possible image/identity candidates only; manual blinded adjudication is still required.",
    }
    with candidate_path.open("w", encoding="utf-8") as handle:
        for name, query_matrix in benchmark_matrix.items():
            rows, threshold_counts = retrieve_candidates(
                train_matrix,
                query_matrix,
                query_batch_size=query_batch_size,
            )
            for row in rows:
                handle.write(json.dumps({"benchmark": name, **row}, ensure_ascii=False) + "\n")
            summary["benchmarks"][name] = {
                "benchmark_image_count": benchmark_counts[name],
                "top_k_candidate_pairs": min(TOP_K, len(train_matrix)) * benchmark_counts[name],
                "candidate_pairs_union": len(rows),
                "threshold_candidate_pair_counts_may_overlap": threshold_counts,
                "manual_review_status": "pending",
                "confirmed_matches": None,
                "excluded_images": None,
                "unresolved_candidates": None,
            }

    report_path = output_dir / "embedding_summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = output_dir / "embedding_manifest.json"
    write_experiment_manifest(
        manifest_path,
        experiment="mined-training-benchmark-independent-facenet-candidates",
        parameters={
            "embedding_model": summary["embedding_model"],
            "checkpoint_sha256": sha256_file(checkpoint),
            "similarity_metric": "cosine similarity after L2 normalization",
            "thresholds": list(SIMILARITY_THRESHOLDS),
            "candidate_similarity_threshold": CANDIDATE_SIMILARITY_THRESHOLD,
            "top_k": TOP_K,
            "batch_size": batch_size,
            "query_batch_size": query_batch_size,
            "device": "cpu",
            "network_calls": False,
            "manual_identity_confirmation": False,
            "private_outputs": True,
        },
        metrics={
            "runtime_seconds": round(time.perf_counter() - started_at, 3),
            "training_images": len(train_ids),
            "benchmark_images": benchmark_counts,
            "candidate_pairs_union": {name: row["candidate_pairs_union"] for name, row in summary["benchmarks"].items()},
            "confirmed_identity_overlaps": None,
            "verified_zero_overlap": False,
        },
        inputs=[embeddings_path, checkpoint, *benchmark_paths.values(), inventory, *([benchmark_cache] if reuse_benchmark_cache else [])],
        outputs=[report_path, candidate_path, *([] if reuse_benchmark_cache else [benchmark_cache])],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / DEFAULT_OUTPUT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--reuse-benchmark-cache", action="store_true")
    args = parser.parse_args()
    result = run_audit(
        output_dir=args.output_dir,
        embeddings_path=args.embeddings,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        query_batch_size=args.query_batch_size,
        reuse_benchmark_cache=args.reuse_benchmark_cache,
    )
    print(json.dumps(result["benchmarks"], indent=2))
    print(f"Private embedding candidates: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
