"""Synthetic tests for the FG-NET oldest-gallery retrieval protocol.

These tests are CPU-only, offline, and never load images, torch, or checkpoints.
They exercise the pure protocol/metrics code in ``fgnet_retrieval_study`` with
hand-built arrays, including adversarial similarity ties.

Run from the repository root with the project interpreter::

    .venv/Scripts/python.exe -m pytest .work/pi-workers/20261002-4a91fc33/retrieval/test_fgnet_retrieval_study.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import fgnet_retrieval_study as study


# ── Fixtures ────────────────────────────────────────────────────────────────
def _toy_metadata():
    """3 identities, 7 images with distinct ages, one oldest-age tie for subject 1.

    original index: 0    1    2    3    4    5    6
    subject:        10   10   10   20   20   30   30
    age:            2    30   31   5    40   7    7
    gallery:        idx2 (31)  idx4 (40)  idx5 (7, lowest index tie-break)
    """
    subjects = np.array([10, 10, 10, 20, 20, 30, 30], dtype=np.int64)
    ages = np.array([2, 30, 31, 5, 40, 7, 7], dtype=np.int64)
    return subjects, ages


def _basis_pairs() -> tuple[np.ndarray, np.ndarray]:
    """Three identities with two images each, plus well-separated reference rows."""
    return (
        np.array(
            [
                [1.0, 0.0],  # 0: A
                [0.99, 0.01],  # 1: A
                [0.0, 1.0],  # 2: B
                [0.01, 0.99],  # 3: B
                [0.7, 0.7],  # 4: C
                [0.7, 0.71],  # 5: C
            ],
            dtype=np.float32,
        ),
        np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
    )


# ── Protocol: oldest-gallery selection, tie handling, image disjointness ────
def test_gallery_is_oldest_image_per_identity():
    subjects, ages = _toy_metadata()
    protocol = study.build_protocol(subjects, ages, seed=7)
    by_subject = dict(zip(protocol.gallery_subject.tolist(), protocol.gallery_index.tolist(), strict=True))
    assert by_subject == {10: 2, 20: 4, 30: 5}
    assert dict(
        zip(protocol.gallery_subject.tolist(), protocol.gallery_age.tolist(), strict=True)
    ) == {10: 31, 20: 40, 30: 7}


def test_oldest_age_tie_breaks_to_lowest_original_index_and_is_counted():
    subjects, ages = _toy_metadata()
    protocol = study.build_protocol(subjects, ages, seed=7)
    # Subject 30 has two age-7 images (indices 5 and 6); index 5 wins, index 6 is excluded.
    assert int(protocol.gallery_index[protocol.gallery_subject == 30][0]) == 5
    assert protocol.n_excluded_max_age_ties == 1
    assert 6 not in set(protocol.query_index.tolist())  # not a query either: not strictly younger


def test_gallery_and_query_images_are_disjoint_but_identities_shared():
    subjects, ages = _toy_metadata()
    protocol = study.build_protocol(subjects, ages, seed=7)
    study.verify_protocol(protocol)
    assert np.intersect1d(protocol.gallery_index, protocol.query_index).size == 0
    # Positive retrieval requires shared identities.
    assert set(protocol.query_subject.tolist()) <= set(protocol.gallery_subject.tolist())
    assert np.unique(protocol.gallery_subject).size == protocol.gallery_subject.size


def test_queries_are_strictly_younger_than_gallery():
    subjects, ages = _toy_metadata()
    protocol = study.build_protocol(subjects, ages, seed=7)
    assert np.all(protocol.age_gap() >= 1)
    # Subject 20: ages 5 and 40 -> query only index 3.
    q20 = protocol.query_index[protocol.query_subject == 20]
    assert q20.tolist() == [3]


def test_identity_with_single_distinct_age_contributes_no_queries():
    subjects = np.array([1, 1, 2], dtype=np.int64)
    ages = np.array([5, 5, 9], dtype=np.int64)
    protocol = study.build_protocol(subjects, ages, seed=1)
    assert protocol.gallery_size == 2
    assert protocol.n_queries == 0  # subject 1's two images share the same age
    assert protocol.n_excluded_max_age_ties == 1


def test_dev_test_split_is_seed_dependent_but_disjoint_and_complete():
    rng = np.random.default_rng(4)
    subjects = np.repeat(np.arange(40), 3)
    ages = rng.integers(0, 60, size=subjects.size)
    partitions = set()
    for seed in range(6):
        protocol = study.build_protocol(subjects, ages, seed=seed, dev_fraction=0.5)
        assert set(protocol.query_split.tolist()) <= {"dev", "test"}
        dev = set(protocol.query_subject[protocol.split_mask("dev")].tolist())
        test = set(protocol.query_subject[protocol.split_mask("test")].tolist())
        assert not (dev & test)
        assert dev | test == set(protocol.gallery_subject.tolist())
        partitions.add(tuple(protocol.query_split.tolist()))
    # Different seeds must be able to produce different partitions.
    assert len(partitions) > 1


def test_split_assigns_a_subject_entirely_to_one_side():
    rng = np.random.default_rng(0)
    subjects = np.repeat(np.arange(40), 3)
    ages = rng.integers(0, 60, size=subjects.size)
    protocol = study.build_protocol(subjects, ages, seed=3, dev_fraction=0.5)
    for subject in np.unique(protocol.query_subject).tolist():
        splits = set(protocol.query_split[protocol.query_subject == subject].tolist())
        assert len(splits) == 1, f"subject {subject} spans multiple splits"


def test_protocol_hash_is_deterministic_and_seed_sensitive():
    subjects, ages = _toy_metadata()
    first = study.build_protocol(subjects, ages, seed=5, source_sha256="deadbeef")
    second = study.build_protocol(subjects, ages, seed=5, source_sha256="deadbeef")
    other_seed = study.build_protocol(subjects, ages, seed=6, source_sha256="deadbeef")
    other_source = study.build_protocol(subjects, ages, seed=5, source_sha256="cafebabe")
    assert first.protocol_hash == second.protocol_hash
    assert first.protocol_hash != other_seed.protocol_hash
    assert first.protocol_hash != other_source.protocol_hash
    assert len(first.protocol_hash) == 64


def test_verify_protocol_detects_tampering():
    import dataclasses

    subjects, ages = _toy_metadata()
    protocol = study.build_protocol(subjects, ages, seed=5)
    # Mutating the selected gallery image must break the bound hash.
    tampered = dataclasses.replace(protocol, gallery_index=protocol.gallery_index + 100)
    with pytest.raises(AssertionError):
        study.verify_protocol(tampered)
    # Mutating the split labels must also be detected.
    flipped = dataclasses.replace(protocol, query_split=np.where(protocol.query_split == "dev", "test", "dev"))
    with pytest.raises(AssertionError):
        study.verify_protocol(flipped)


# ── Ranking: exact ranks and deterministic tie handling ────────────────────
def test_perfect_ranking_gives_rank_one():
    embeddings, identities = _basis_pairs()
    protocol, row_map = _protocol_from_embeddings(embeddings, identities)
    sims = study.similarity_matrix(embeddings, protocol, row_map)
    ranks = study.per_query_ranks(sims, protocol.query_subject, protocol.gallery_subject)
    metrics = study.metrics_from_ranks(ranks)
    assert metrics["n_queries"] == 3
    assert metrics["recall@1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["median_rank"] == 1.0


def test_similarity_tie_breaks_by_gallery_position_deterministically():
    # Query 0 belongs to gallery position 0. Its similarity to positions 0 and 1 is identical.
    sims = np.array([[0.5, 0.5, 0.1]], dtype=np.float64)
    query_subject = np.array([7], dtype=np.int64)
    gallery_subject = np.array([7, 9, 11], dtype=np.int64)
    ranks = study.per_query_ranks(sims, query_subject, gallery_subject)
    # Relevant gallery is at position 0; the tie at position 1 must not count before it.
    assert ranks.tolist() == [1.0]

    # Reverse the tie order: relevant gallery now at position 1, so one tied item precedes it.
    sims_reversed = np.array([[0.5, 0.5, 0.1]], dtype=np.float64)
    ranks_reversed = study.per_query_ranks(sims_reversed, np.array([9]), gallery_subject)
    assert ranks_reversed.tolist() == [2.0]


def test_rank_two_when_a_distractor_is_closer():
    sims = np.array([[0.4, 0.9, 0.2]], dtype=np.float64)
    ranks = study.per_query_ranks(sims, np.array([7]), np.array([7, 9, 11]))
    assert ranks.tolist() == [2.0]
    assert study.metrics_from_ranks(ranks, ks=(1, 2))["recall@1"] == 0.0
    assert study.metrics_from_ranks(ranks, ks=(1, 2))["recall@2"] == 1.0


def test_query_without_relevant_gallery_item_is_nan_and_unscored():
    sims = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    query_subject = np.array([7, 99], dtype=np.int64)
    gallery_subject = np.array([7, 9], dtype=np.int64)
    ranks = study.per_query_ranks(sims, query_subject, gallery_subject)
    assert np.isfinite(ranks[0]) and np.isnan(ranks[1])
    metrics = study.metrics_from_ranks(ranks)
    assert metrics["n_queries"] == 1 and metrics["n_unscored"] == 1


def test_recall_equals_cmc_because_one_gallery_item_per_identity():
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(20, 5)).astype(np.float32)
    identities = np.repeat(np.arange(4), 5)[:20]
    protocol, row_map = _protocol_from_embeddings(embeddings, identities)
    sims = study.similarity_matrix(embeddings, protocol, row_map)
    ranks = study.per_query_ranks(sims, protocol.query_subject, protocol.gallery_subject)
    curve = study.cmc_curve(ranks, max_rank=10)
    metrics = study.metrics_from_ranks(ranks, ks=(1, 5, 10))
    assert curve[0] == metrics["recall@1"]
    assert curve[4] == metrics["recall@5"]
    assert curve[9] == metrics["recall@10"]


# ── Strata ──────────────────────────────────────────────────────────────────
def test_age_gap_strata_partition_queries_without_overlap():
    subjects, ages = _toy_metadata()
    protocol = study.build_protocol(subjects, ages, seed=7, dev_fraction=0.0)
    masks = study.stratum_masks(protocol, "test")
    gap_labels = [label for _, _, label in study.AGE_GAP_BINS]
    counts = {label: int(masks[label].sum()) for label in gap_labels}
    assert sum(counts.values()) == int(masks["overall"].sum())
    for i, left in enumerate(gap_labels):
        for right in gap_labels[i + 1 :]:
            assert not np.any(masks[left] & masks[right])


def test_child_to_adult_stratum_uses_query_and_gallery_age_bounds():
    subjects = np.array([1, 1, 2, 2], dtype=np.int64)
    ages = np.array([10, 40, 10, 20], dtype=np.int64)
    protocol = study.build_protocol(subjects, ages, seed=0, dev_fraction=0.0)
    masks = study.stratum_masks(protocol, "test")
    # Subject 1: query age 10 (<13) vs gallery 40 (>25) -> in stratum.
    # Subject 2: query age 10 vs gallery 20 -> not adult.
    assert masks[study.CHILD_TO_ADULT_LABEL].sum() == 1


# ── Paired query-subject bootstrap ──────────────────────────────────────────
def test_bootstrap_is_paired_and_deterministic():
    rng = np.random.default_rng(0)
    n = 60
    query_subject = np.repeat(np.arange(12), 5)
    ranks_frozen = rng.integers(1, 6, size=n).astype(np.float64)
    ranks_tuned = np.minimum(ranks_frozen, rng.integers(1, 4, size=n)).astype(np.float64)
    masks = {"overall": np.ones(n, dtype=bool)}
    first = study.paired_subject_bootstrap(ranks_frozen, ranks_tuned, query_subject, masks, n_boot=200, seed=3)
    second = study.paired_subject_bootstrap(ranks_frozen, ranks_tuned, query_subject, masks, n_boot=200, seed=3)
    assert first["overall"].as_dict() == second["overall"].as_dict()
    assert first["overall"].n_subjects == 12
    assert first["overall"].n_queries == n
    # Tuned dominates frozen by construction: delta must be >= 0 and CIs ordered.
    assert first["overall"].delta["recall@1"] >= 0
    lo, hi = first["overall"].delta_ci95["recall@1"]
    assert lo <= hi


def test_bootstrap_gallery_is_fixed_so_intervals_are_conditional():
    # Perfect and imperfect rank vectors over the same query subjects.
    query_subject = np.repeat(np.arange(8), 4)
    perfect = np.ones(query_subject.size, dtype=np.float64)
    imperfect = np.full(query_subject.size, 3.0, dtype=np.float64)
    masks = {"overall": np.ones(query_subject.size, dtype=bool)}
    result = study.paired_subject_bootstrap(
        perfect, imperfect, query_subject, masks, n_boot=100, seed=1
    )["overall"]
    # Resampling query subjects cannot change a constant rank vector's metric.
    assert result.frozen["recall@1"] == 1.0
    assert result.frozen_ci95["recall@1"] == (1.0, 1.0)
    assert result.tuned["recall@1"] == 0.0
    assert result.tuned_ci95["recall@1"] == (0.0, 0.0)
    assert result.delta["recall@1"] == -1.0


def test_bootstrap_uses_only_queries_inside_the_stratum():
    query_subject = np.array([1, 1, 2, 2], dtype=np.int64)
    ranks_frozen = np.array([1.0, 2.0, 3.0, 4.0])
    ranks_tuned = np.array([1.0, 1.0, 1.0, 1.0])
    masks = {"overall": np.ones(4, dtype=bool), "first_two": np.array([True, True, False, False])}
    result = study.paired_subject_bootstrap(ranks_frozen, ranks_tuned, query_subject, masks, n_boot=50, seed=0)
    assert result["first_two"].n_queries == 2
    assert result["first_two"].n_subjects == 1  # only subject 1 appears
    assert result["first_two"].frozen["recall@1"] == 0.5


def test_bootstrap_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        study.paired_subject_bootstrap(
            np.array([1.0, 2.0]),
            np.array([1.0]),
            np.array([1, 2]),
            {"overall": np.ones(2, dtype=bool)},
            n_boot=10,
        )


# ── Public aggregate privacy guard ─────────────────────────────────────────
def test_public_output_guard_rejects_row_level_keys():
    with pytest.raises(AssertionError):
        study.assert_public_output_has_no_ids({"results": {"subject": 12}})
    with pytest.raises(AssertionError):
        study.assert_public_output_has_no_ids({"results": {"crop_index": [1, 2, 3]}})


def test_public_output_guard_rejects_integer_vectors_and_huge_strings():
    with pytest.raises(AssertionError):
        study.assert_public_output_has_no_ids({"x": list(range(50))})
    with pytest.raises(AssertionError):
        study.assert_public_output_has_no_ids({"x": "A" * 600})


def test_public_output_guard_accepts_aggregate_payload():
    subjects, ages = _toy_metadata()
    protocol = study.build_protocol(subjects, ages, seed=7)
    public = study.build_public_output(
        protocol=protocol,
        source={"npz_sha256": "0" * 64, "n_images": 7, "n_identities": 3, "age_min": 2, "age_max": 40},
        models={"frozen": {"weights_sha256": "1" * 64}, "tuned": {}},
        results={"frozen_reference": {"test": {"overall": {"recall@1": 1.0, "n_queries": 3.0}}}},
    )
    study.assert_public_output_has_no_ids(public)
    assert public["protocol_hash"] == protocol.protocol_hash
    # Disclosures must include the required non-claims.
    joined = " ".join(public["disclosures"])
    assert "NOT identity-disjoint" in joined
    assert "unverified training-identity independence" in joined
    assert "never resampled" in joined


# ── Embedding/ranking integration (no images, no torch) ────────────────────
def test_embed_unique_crops_reuses_cache_and_embeds_once(tmp_path, monkeypatch):
    crops = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    indices = np.array([0, 2, 3], dtype=np.int64)
    calls = []

    def fake_embed(crops_arg, indices_arg, model, *, batch_size=32, threads=2):
        calls.append(np.asarray(indices_arg).tolist())
        return np.arange(np.asarray(indices_arg).size, dtype=np.float32).reshape(-1, 1)

    monkeypatch.setattr(study, "embed_indices", fake_embed)
    cache = tmp_path / "cache.npz"
    kwargs = dict(
        role="frozen",
        weights_sha256="a" * 64,
        source_sha256="b" * 64,
        cache_path=cache,
    )
    first, hit_first = study.embed_unique_crops(crops, indices, object(), **kwargs)
    assert hit_first is False
    assert calls == [[0, 2, 3]]
    second, hit_second = study.embed_unique_crops(crops, indices, object(), **kwargs)
    assert hit_second is True
    assert calls == [[0, 2, 3]]  # no second inference pass
    assert np.array_equal(first, second)

    # A different weight hash must invalidate the cache.
    third, hit_third = study.embed_unique_crops(
        crops, indices, object(), **{**kwargs, "weights_sha256": "c" * 64}
    )
    assert hit_third is False
    assert len(calls) == 2


def test_embedding_cache_key_binds_weights_source_and_indices():
    indices = np.array([3, 1, 2], dtype=np.int64)
    base = dict(role="frozen", weights_sha256="a" * 64, source_sha256="b" * 64, indices=indices)
    key = study.embedding_cache_key(**base)
    assert key == study.embedding_cache_key(**base)
    assert key != study.embedding_cache_key(**{**base, "weights_sha256": "c" * 64})
    assert key != study.embedding_cache_key(**{**base, "indices": np.array([3, 1, 2, 9], dtype=np.int64)})


def test_projection_slices_unique_embeddings_in_protocol_order():
    subjects, ages = _toy_metadata()
    protocol = study.build_protocol(subjects, ages, seed=7)
    n_crops = int(subjects.size)
    unique = protocol.unique_indices()
    embeddings = np.arange(unique.size * 2, dtype=np.float32).reshape(unique.size, 2)
    row_map = protocol.embedding_row_map(n_crops=n_crops)
    gallery, queries = study.project_embeddings(embeddings, protocol, row_map)
    position = {int(index): row for row, index in enumerate(unique.tolist())}
    for col, index in enumerate(protocol.gallery_index.tolist()):
        assert np.array_equal(gallery[col], embeddings[position[index]])
    for col, index in enumerate(protocol.query_index.tolist()):
        assert np.array_equal(queries[col], embeddings[position[index]])


# ── Regression: non-contiguous indices need an explicit index->row map ──────
def test_embedding_row_map_handles_omitted_interior_index():
    # Subject 10: ages [40, 40, 10] at indices [0, 1, 2]. The excluded max-age tie
    # is index 1 (interior); subject 20's gallery is index 3. The omitted index 1
    # therefore precedes a needed index, so raw-index slicing misaligns/crashes.
    subjects = np.array([10, 10, 10, 20], dtype=np.int64)
    ages = np.array([40, 40, 10, 30], dtype=np.int64)
    protocol = study.build_protocol(subjects, ages, seed=0)
    assert protocol.gallery_index.tolist() == [0, 3]
    assert protocol.query_index.tolist() == [2]
    assert protocol.n_excluded_max_age_ties == 1
    unique = protocol.unique_indices()
    assert unique.tolist() == [0, 2, 3]  # index 1 omitted -> rows are not original indices
    row_map = protocol.embedding_row_map(n_crops=4)
    assert row_map.tolist() == [0, -1, 1, 2]

    embeddings = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)  # rows: idx 0,2,3
    gallery, queries = study.project_embeddings(embeddings, protocol, row_map)
    assert np.array_equal(gallery, embeddings[[0, 2]])
    assert np.array_equal(queries, embeddings[[1]])
    # The naive raw-index slice is out of bounds and silently wrong under padding.
    with pytest.raises(IndexError):
        _ = embeddings[protocol.gallery_index]


def test_project_embeddings_rejects_uncomputed_rows_instead_of_padding():
    subjects = np.array([10, 10, 10, 20], dtype=np.int64)
    ages = np.array([40, 40, 10, 30], dtype=np.int64)
    protocol = study.build_protocol(subjects, ages, seed=0)
    row_map = protocol.embedding_row_map(n_crops=4)
    # Only one embedding row supplied although the gallery needs rows 0 and 2.
    truncated = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        study.project_embeddings(truncated, protocol, row_map)
    # A row map that omits an actually-needed crop index must also fail.
    bad_map = row_map.copy()
    bad_map[3] = -1
    with pytest.raises(ValueError):
        study.project_embeddings(np.zeros((3, 2), dtype=np.float32), protocol, bad_map)


def test_embedding_row_map_is_identity_for_contiguous_complete_indices():
    row_map = study.embedding_row_map(np.array([0, 1, 2], dtype=np.int64), n_crops=3)
    assert row_map.tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        study.embedding_row_map(np.array([0, 0, 1], dtype=np.int64))


# ── Empty strata must be null/NaN, not misleading 0.0 ─────────────────────
def test_empty_metrics_are_nan_and_flagged_empty():
    metrics = study.metrics_from_ranks(np.array([], dtype=np.float64), ks=(1, 5))
    assert metrics["n_queries"] == 0.0
    assert metrics["empty"] == 1.0
    assert np.isnan(metrics["recall@1"])
    assert np.isnan(metrics["recall@5"])
    assert np.isnan(metrics["mrr"])
    assert np.isnan(metrics["median_rank"])


def test_empty_cmc_curve_is_nan_not_zero():
    curve = study.cmc_curve(np.array([], dtype=np.float64), max_rank=4)
    assert len(curve) == 4
    assert all(np.isnan(value) for value in curve)


def test_empty_bootstrap_stratum_emits_nan_metrics_and_zero_size():
    result = study.paired_subject_bootstrap(
        np.array([1.0, 2.0]),
        np.array([1.0, 1.0]),
        np.array([1, 2]),
        {"empty": np.array([False, False])},
        n_boot=10,
        seed=0,
    )["empty"]
    assert result.n_queries == 0 and result.n_subjects == 0
    assert result.n_valid_resamples == 0
    assert np.isnan(result.frozen["recall@1"])
    assert np.isnan(result.delta["recall@1"])
    assert all(np.isnan(v) for pair in result.delta_ci95.values() for v in pair)


def test_nonfinite_sanitizer_converts_nan_to_null():
    payload = {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": np.float32("nan")}}
    cleaned = study.sanitize_nonfinite(payload)
    assert cleaned["a"] is None
    assert cleaned["b"][1] is None
    assert cleaned["c"]["d"] is None
    # The cleaned payload must be strict-JSON serializable.
    json.dumps(cleaned, allow_nan=False)


# ── Input validation bounds ────────────────────────────────────────────────
def test_input_validation_bounds():
    with pytest.raises(ValueError):
        study.configure_torch_threads(0)
    with pytest.raises(ValueError):
        study.configure_torch_threads(3)
    with pytest.raises(ValueError):
        study.paired_subject_bootstrap(
            np.array([1.0]), np.array([1.0]), np.array([1]), {"o": np.array([True])}, n_boot=0
        )


def test_run_study_rejects_invalid_bounds(tmp_path):
    subjects = np.repeat([1, 2], 2)
    ages = np.array([5, 1, 6, 2], dtype=np.int64)
    crops = np.zeros((4, 4, 4, 3), dtype=np.uint8)
    npz = tmp_path / "s.npz"
    np.savez_compressed(npz, crops=crops, subjects=subjects, ages=ages)
    base = tmp_path / "b.pt"
    base.write_bytes(b"x")
    for kwargs in ({"n_boot": 0}, {"batch_size": 0}, {"threads": 3}, {"threads": 0}):
        with pytest.raises(ValueError):
            study.run_study(
                npz_path=npz,
                base_weights=base,
                output_dir=tmp_path / "o",
                tuned_checkpoints={42: base},
                plan_only=True,
                **kwargs,
            )
def test_run_study_end_to_end_is_pure_and_protocol_bound(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_subjects, per_subject = 12, 4
    subjects = np.repeat(np.arange(1, n_subjects + 1), per_subject)
    # First image per subject is the oldest, then strictly decreasing ages. Subject 1
    # also gets a duplicate max-age image appended, so an excluded max-age tie leaves
    # a *non-contiguous* omitted index in the middle of the crop index range.
    subjects = np.append(subjects, 1)
    ages = np.concatenate([np.tile(np.arange(per_subject, 0, -1), n_subjects).astype(np.int64), [per_subject]])
    crops = rng.integers(0, 255, size=(subjects.size, 8, 8, 3), dtype=np.uint8)
    npz = tmp_path / "synthetic.npz"
    np.savez_compressed(npz, crops=crops, subjects=subjects, ages=ages)
    base = tmp_path / "base.pt"
    base.write_bytes(b"base")
    tuned = {}
    for seed in (42, 1, 2):
        path = tmp_path / f"tuned_seed{seed}.pt"
        path.write_bytes(f"tuned-{seed}".encode())
        tuned[seed] = path
    inventory = tmp_path / "model_inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "models": {
                    "facenet_casia": {
                        "source": "facenet-pytorch InceptionResnetV1 pretrained=casia-webface",
                        "role": "weak trainable backbone",
                        "declared_training_data": "CASIA-WebFace",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text('{"pair_id": "x"}\n', encoding="utf-8")

    # Deterministic one-hot embeddings: each identity gets a distinct direction.
    def fake_embed(crops_arg, indices_arg, model, *, role, weights_sha256, source_sha256, cache_path, batch_size=32, threads=2, reuse_cache=True):
        dim = int(subjects.max()) + 1
        out = np.zeros((len(indices_arg), dim), dtype=np.float32)
        for row, index in enumerate(np.asarray(indices_arg).tolist()):
            out[row, int(subjects[index])] = 1.0
        return out, False

    monkeypatch.setattr(study, "load_frozen_model", lambda *a, **k: object())
    monkeypatch.setattr(study, "load_tuned_model", lambda *a, **k: object())
    monkeypatch.setattr(study, "embed_unique_crops", fake_embed)

    output_dir = tmp_path / "out"
    public = study.run_study(
        npz_path=npz,
        base_weights=base,
        output_dir=output_dir,
        tuned_checkpoints=tuned,
        protocol_seed=42,
        n_boot=50,
        batch_size=4,
        model_inventory_path=inventory,
        training_pairs=pairs,
    )
    study.assert_public_output_has_no_ids(public)
    assert public["protocol_hash"]
    assert public["n_excluded_max_age_ties"] == 1
    assert public["training_identity_independence"] == "unverified"
    assert public["provenance"]["model_inventory_read"] is True
    assert public["provenance"]["tuned_training_pairs_sha256"]
    assert public["provenance"]["tuned_checkpoints"]["42"]["declared_training_data"] == "CASIA-WebFace"
    # Doctored embeddings make each identity nearest its own gallery item -> perfect recall.
    for seed in (42, 1, 2):
        overall = public["results"][f"tuned_seed{seed}"]["test"]["frozen"]["overall"]
        assert overall["recall@1"] == 1.0
        assert overall["mrr"] == 1.0
        results = public["results"][f"tuned_seed{seed}"]["test"]
        # Every stratum must either be empty (null/nan) or a genuine value.
        for label, block in results["frozen"].items():
            if block["n_queries"] == 0:
                assert block["recall@1"] is None
                assert results["paired_bootstrap"][label]["n_queries"] == 0
                assert results["paired_bootstrap"][label]["frozen"]["recall@1"] is None
    assert (output_dir / "fgnet_retrieval_study.json").is_file()
    assert (output_dir / "fgnet_retrieval_study.manifest.json").is_file()
    assert (output_dir / "private" / "retrieval_rows.jsonl").is_file()
    raw = (output_dir / "fgnet_retrieval_study.json").read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)))
    manifest = json.loads((output_dir / "fgnet_retrieval_study.manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameters"]["protocol_hash"] == public["protocol_hash"]
    assert manifest["parameters"]["training_identity_independence"] == "unverified"
    assert manifest["parameters"]["gallery_resampling"] is False
    assert manifest["parameters"]["provenance"]["model_inventory_read"] is True
    assert manifest["metrics"]["embedding_row_map_size"] == int(subjects.size)


def test_run_study_rejects_wrong_expected_protocol_hash(tmp_path):
    subjects = np.repeat([1, 2], 2)
    ages = np.array([5, 1, 6, 2], dtype=np.int64)
    crops = np.zeros((4, 4, 4, 3), dtype=np.uint8)
    npz = tmp_path / "s.npz"
    np.savez_compressed(npz, crops=crops, subjects=subjects, ages=ages)
    base = tmp_path / "b.pt"
    base.write_bytes(b"x")
    with pytest.raises(ValueError):
        study.run_study(
            npz_path=npz,
            base_weights=base,
            output_dir=tmp_path / "o",
            tuned_checkpoints={42: base},
            plan_only=True,
            expect_protocol_hash="0" * 64,
        )


def _protocol_from_embeddings(embeddings: np.ndarray, identities: np.ndarray):
    """Build a protocol whose gallery/query layout matches a tiny embedding matrix.

    One image per identity is designated gallery (the first occurrence); the rest
    are queries. Ages are synthetic and strictly decreasing after the gallery item.
    Returns ``(protocol, row_map)`` where ``row_map`` maps original indices to rows.
    """
    n = len(identities)
    ages = np.zeros(n, dtype=np.int64)
    for subject in np.unique(identities).tolist():
        positions = np.flatnonzero(identities == subject)
        ages[positions] = np.arange(len(positions), 0, -1)  # first is oldest
    protocol = study.build_protocol(identities, ages, seed=0, dev_fraction=0.0)
    return protocol, protocol.embedding_row_map(n_crops=n)
