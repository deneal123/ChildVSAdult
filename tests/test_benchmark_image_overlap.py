from __future__ import annotations

import numpy as np

from scripts.benchmark_embedding_overlap_audit import retrieve_candidates
from scripts.benchmark_image_overlap_audit import (
    _candidate_pairs,
    hamming_distance,
    perceptual_hash,
    pixel_sha256,
)


def test_decoded_pixel_hash_is_stable_and_shape_sensitive() -> None:
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    assert pixel_sha256(image) == pixel_sha256(image.copy())
    assert pixel_sha256(image) != pixel_sha256(image.reshape(8, 32, 3))


def test_perceptual_hash_is_stable_for_identical_face_pixels() -> None:
    image = np.random.default_rng(42).integers(0, 256, (112, 112, 3), dtype=np.uint8)
    assert perceptual_hash(image) == perceptual_hash(image.copy())
    assert hamming_distance(perceptual_hash(image), perceptual_hash(image)) == 0


def test_block_index_returns_all_candidates_within_threshold() -> None:
    train = [0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF, 0x5555555555555555]
    query = train[0] ^ 0b10101
    pairs = _candidate_pairs(train, [query], max_distance=4)
    assert (0, 0, 3) in pairs
    assert all(distance <= 4 for _, _, distance in pairs)
    assert (1, 0, hamming_distance(train[1], query)) not in pairs


def test_embedding_retrieval_returns_topk_and_threshold_candidates() -> None:
    train = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    rows, counts = retrieve_candidates(train, query, thresholds=(0.7, 0.95), top_k=1)
    by_pair = {(row["train_index"], row["benchmark_index"]): row for row in rows}

    assert by_pair[(0, 0)]["rank"] == 1
    assert by_pair[(2, 1)]["rank"] == 1
    assert by_pair[(2, 1)]["match_type"] == "top_k_and_threshold"
    assert by_pair[(1, 0)]["match_type"] == "cosine_threshold"
    assert counts == {"cosine_ge_0.70": 3, "cosine_ge_0.95": 2}
