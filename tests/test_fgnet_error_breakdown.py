"""Synthetic, CPU-only tests for the candidate FG-NET error-breakdown script.

These tests never load torch, cv2, model weights, raw images or the network, and never call
``run_breakdown``. They exercise the pure calibration/strata/bootstrap/cache-reuse code with
hand-built arrays and a monkeypatched embedder.

Run from the repository root::

    .venv/Scripts/python.exe -m pytest \
        .work/pi-workers/20261002-4a91fc33/error-analysis/tests/test_fgnet_error_breakdown.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import fgnet_error_breakdown as eb


# ── Fixtures ────────────────────────────────────────────────────────────────
def _toy_metadata():
    """Six identities, each with ages 5, 20, 40 (one gallery candidate at 40)."""
    subjects, ages = [], []
    for subject in range(1, 7):
        for age in (5, 20, 40):
            subjects.append(subject)
            ages.append(age)
    return np.asarray(subjects, dtype=np.int64), np.asarray(ages, dtype=np.int64)


def _toy_protocol():
    subjects, ages = _toy_metadata()
    return subjects, ages, eb.build_protocol(subjects, ages, seed=eb.SEED, dev_fraction=eb.DEV_FRACTION)


def _fake_embeddings(n_crops: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n_crops, dim)).astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


# ── Calibration: dev negatives only, strict >, resolution note ──────────────
def test_calibration_uses_dev_negatives_and_hits_target_fmr():
    neg = np.arange(1000, dtype=float)  # 0..999
    op = eb.calibrate_operating_point(neg, np.asarray([500.0]), 0.01)
    assert op.resolvable is True
    assert op.max_dev_accepts == 10
    assert op.dev_false_accepts == 10
    assert op.dev_fmr <= 0.01 + 1e-12
    # strictly-greater convention: the threshold itself is rejected
    assert op.threshold == 989.0


def test_calibration_threshold_independent_of_positives():
    neg = np.arange(100, dtype=float)
    a = eb.calibrate_operating_point(neg, np.asarray([0.0, 1.0]), 0.01)
    b = eb.calibrate_operating_point(neg, np.asarray([100.0]), 0.01)
    assert a.threshold == b.threshold
    assert a.dev_fnmr != b.dev_fnmr  # only the reported FNMR changes


def test_calibration_ties_never_exceed_target_fmr():
    neg = np.concatenate([np.full(50, 5.0), np.full(50, 1.0)])
    op = eb.calibrate_operating_point(neg, np.asarray([2.0]), 0.01)
    # floor(0.01*100)=1 -> threshold = 2nd largest = 5.0; all 50 fives are NOT > 5.0
    assert op.dev_false_accepts == 0
    assert op.dev_fmr <= 0.01
    assert op.tie_values_at_threshold == 50


def test_low_far_resolution_is_flagged_not_claimed():
    neg = np.arange(100, dtype=float)
    op = eb.calibrate_operating_point(neg, np.asarray([0.5]), 0.001)
    assert op.resolvable is False
    assert "NOT resolvable" in op.resolution_note
    assert "1000" in op.resolution_note  # ceil(1/0.001)


def test_counts_at_threshold_strict_convention():
    pos = np.asarray([0.9, 0.5, 0.5])
    neg = np.asarray([0.5, 0.4, 0.1])
    out = eb.counts_at_threshold(pos, neg, 0.5)
    assert out["false_accepts"] == 0  # negative equal to threshold is rejected
    assert out["false_rejects"] == 2  # positives <= threshold are rejected
    assert out["n_pos"] == 3 and out["n_neg"] == 3


# ── Split-local, endpoint-age-matched pair construction ─────────────────────
def test_pairs_are_split_local_balanced_and_age_matched():
    subjects, ages, protocol = _toy_protocol()
    dev_pairs, dev_diag = eb.build_split_pairs(subjects=subjects, ages=ages, protocol=protocol, split="dev")
    test_pairs, test_diag = eb.build_split_pairs(subjects=subjects, ages=ages, protocol=protocol, split="test")

    dev_subjects = set(protocol.query_subject[protocol.split_mask("dev")].tolist())
    test_subjects = set(protocol.query_subject[protocol.split_mask("test")].tolist())
    assert dev_subjects and test_subjects and not (dev_subjects & test_subjects)

    for pairs, own, other in ((dev_pairs, dev_subjects, test_subjects), (test_pairs, test_subjects, dev_subjects)):
        pos = [p for p in pairs if p.label == 1]
        neg = [p for p in pairs if p.label == 0]
        assert len(pos) == len(neg) > 0
        assert dev_diag["candidate_pool_is_split_local"] is True
        for pair in pairs:
            assert pair.left_subject in own and pair.right_subject in own
            assert pair.left_subject not in other and pair.right_subject not in other
        for pair in pos:
            assert pair.left_subject == pair.right_subject
            assert pair.left_age < pair.right_age  # younger query on the left
        for pair in neg:
            assert pair.left_subject != pair.right_subject
            assert abs(pair.left_age - pair.right_age) == pair.age_gap
            assert pair.left_age < pair.right_age
    # paired positives/negatives share the exact age gap
    assert [p.age_gap for p in dev_pairs if p.label == 1] == [p.age_gap for p in dev_pairs if p.label == 0]
    assert dev_diag["match_endpoint_error_max"] <= eb.ENDPOINT_AGE_TOLERANCE
    assert test_diag["candidate_pool_is_split_local"] is True


def test_pairs_do_not_reference_other_split_subjects_when_matching():
    # One split only has a single subject that has an eligible positive; with no impostor
    # available in-split, coverage must drop rather than borrow the other split's identity.
    subjects = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int64)
    ages = np.asarray([5, 20, 40, 6, 21, 41], dtype=np.int64)
    protocol = eb.build_protocol(subjects, ages, seed=1, dev_fraction=0.5)
    for split in ("dev", "test"):
        pairs, diag = eb.build_split_pairs(subjects=subjects, ages=ages, protocol=protocol, split=split)
        own = set(protocol.query_subject[protocol.split_mask(split)].tolist())
        for pair in pairs:
            assert pair.left_subject in own and pair.right_subject in own


# ── Strata: exact bins, child->adult, blur ──────────────────────────────────
def test_split_strata_partition_gap_and_define_child_to_adult():
    pos_gap = np.asarray([1, 9, 10, 19, 20, 24, 25, 40])
    pos_left = np.asarray([1, 9, 10, 12, 20, 24, 25, 5])
    pos_right = np.asarray([12, 18, 30, 31, 44, 48, 50, 60])
    neg_gap = pos_gap.copy()
    neg_left = pos_left.copy()
    neg_right = pos_right.copy()
    zeros = np.zeros_like(pos_gap)
    masks = eb.split_strata(
        pos_gap=pos_gap,
        neg_gap=neg_gap,
        pos_left_age=pos_left,
        neg_left_age=neg_left,
        pos_right_age=pos_right,
        neg_right_age=neg_right,
        pos_blur_bin=zeros,
        neg_blur_bin=zeros,
        n_blur_bins=eb.BLUR_BINS,
    )
    labels = [label for _l, _h, label in eb.GAP_BINS]
    coverage = np.zeros(pos_gap.size, dtype=int)
    for label in labels:
        pm, _nm = masks[label]
        coverage += pm.astype(int)
    assert np.all(coverage == 1)  # exact, non-overlapping partition
    pm, nm = masks[eb.CHILD_TO_ADULT_LABEL]
    # child (<13) at indices 0,1,2,3,7; adult (>25) at indices 2,3,4,5,6,7 -> intersection 2,3,7
    assert pm.tolist() == [False, False, True, True, False, False, False, True]
    assert pm.shape == nm.shape
    assert "blur_bin_0" in masks


def test_blur_cutpoints_are_dev_values_and_labels_are_monotone():
    dev = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    cuts = eb.blur_cutpoints(dev, n_bins=3)
    assert len(cuts) == 2 and cuts == sorted(cuts)
    labels = eb.blur_bin_labels(np.asarray([-1.0, 25.0, 999.0]), cuts)
    assert labels[0] <= labels[1] <= labels[2]
    assert eb.blur_cutpoints(np.asarray([])) == []


def test_pair_blur_is_min_of_endpoints():
    from types import SimpleNamespace

    pairs = [SimpleNamespace(left_crop=0, right_crop=1), SimpleNamespace(left_crop=2, right_crop=3)]
    values = np.asarray([9.0, 3.0, 5.0, 8.0])
    assert eb.pair_blur(values, pairs).tolist() == [3.0, 5.0]


# ── Modalities are unknown, never inferred ──────────────────────────────────
def test_modalities_default_to_unknown_and_require_explicit_values():
    axes = eb.modality_axes()
    for axis in eb.UNSUPPORTED_MODALITY_AXES:
        assert axes[axis] == "unknown"
    assert "never inferred" in axes["inference"]
    overridden = eb.modality_axes({"collage": "yes_reviewed"})
    assert overridden["collage"] == "yes_reviewed"
    assert overridden["pose"] == "unknown"  # missing axes stay unknown


# ── Bootstrap: both endpoints, fixed threshold ──────────────────────────────
def test_bootstrap_uses_both_endpoints_and_returns_bounded_cis():
    rng = np.random.default_rng(0)
    n = 40
    pos_subject = rng.integers(0, 10, size=n)
    neg_left = rng.integers(0, 10, size=n)
    neg_right = rng.integers(0, 10, size=n)
    pos_reject = rng.integers(0, 2, size=n).astype(float)
    neg_accept = rng.integers(0, 2, size=n).astype(float)
    out = eb.bootstrap_fmr_fnmr(
        pos_subject=pos_subject,
        pos_reject=pos_reject,
        neg_left_subject=neg_left,
        neg_right_subject=neg_right,
        neg_accept=neg_accept,
        n_boot=200,
        seed=0,
    )
    assert out["valid"] == 200
    for ci in (out["fmr_ci95"], out["fnmr_ci95"]):
        assert ci is not None and 0.0 <= ci[0] <= ci[1] <= 1.0


def test_bootstrap_requires_positive_n_boot():
    with pytest.raises(ValueError):
        eb.bootstrap_fmr_fnmr(
            pos_subject=np.asarray([1]),
            pos_reject=np.asarray([0.0]),
            neg_left_subject=np.asarray([1]),
            neg_right_subject=np.asarray([2]),
            neg_accept=np.asarray([0.0]),
            n_boot=0,
        )


# ── Embedding cache reuse by model/dataset/preproc key ──────────────────────
def _save_cache(path: Path, *, role: str, weights_sha256: str, source_sha256: str, indices, embeddings):
    path.parent.mkdir(parents=True, exist_ok=True)
    key = eb.embedding_cache_key(
        role=role, weights_sha256=weights_sha256, source_sha256=source_sha256, indices=np.asarray(indices)
    )
    np.savez_compressed(path, key=np.asarray(key), indices=np.asarray(indices), embeddings=np.asarray(embeddings, dtype=np.float32))


def test_resolve_reuses_verified_cache_and_embeds_only_missing(tmp_path, monkeypatch):
    role, weights, source = "tuned_facenet_seed1", "w" * 64, "s" * 64
    cache = tmp_path / "private_embeddings_tuned_seed1.npz"
    _save_cache(cache, role=role, weights_sha256=weights, source_sha256=source, indices=[0, 1], embeddings=[[1, 0], [0, 1]])

    calls = {"n": 0}

    def fake_embed(crops, indices, model, *, batch_size=32, threads=2):
        calls["n"] += 1
        return np.ones((len(indices), 2), dtype=np.float32) * 0.5

    monkeypatch.setattr(eb, "configure_torch_threads", lambda threads=2: None)
    monkeypatch.setattr(eb, "embed_indices", fake_embed)
    write_path = tmp_path / "out" / "private_embeddings_tuned_seed1.npz"
    matrix, meta = eb.resolve_embeddings(
        role=role,
        weights_sha256=weights,
        source_sha256=source,
        required_indices=np.asarray([0, 1, 2], dtype=np.int64),
        crops=np.zeros((3, 4, 4, 3), dtype=np.uint8),
        cache_candidates=[cache],
        write_path=write_path,
        load_model=lambda: object(),
        allow_embed=True,
    )
    assert calls["n"] == 1  # embedded only the missing crop
    assert meta["n_reused"] == 2 and meta["n_embedded"] == 1
    assert matrix.shape == (3, 2)
    assert write_path.is_file()


def test_resolve_rejects_cache_with_mismatched_key(tmp_path):
    cache = tmp_path / "private_embeddings_frozen.npz"
    _save_cache(cache, role="frozen_casia_webface", weights_sha256="a" * 64, source_sha256="s" * 64, indices=[0], embeddings=[[1, 0]])
    with pytest.raises(RuntimeError):
        eb.resolve_embeddings(
            role="frozen_casia_webface",
            weights_sha256="b" * 64,  # different weights -> cache must not be reused
            source_sha256="s" * 64,
            required_indices=np.asarray([0], dtype=np.int64),
            crops=None,
            cache_candidates=[cache],
            write_path=tmp_path / "out.npz",
            load_model=lambda: object(),
            allow_embed=False,
        )


def test_resolve_refuses_to_embed_without_execute(tmp_path):
    with pytest.raises(RuntimeError):
        eb.resolve_embeddings(
            role="frozen_casia_webface",
            weights_sha256="a" * 64,
            source_sha256="s" * 64,
            required_indices=np.asarray([0], dtype=np.int64),
            crops=None,
            cache_candidates=[tmp_path / "absent.npz"],
            write_path=tmp_path / "out.npz",
            load_model=lambda: object(),
            allow_embed=False,
        )


def test_resolve_enforces_thread_budget(tmp_path):
    with pytest.raises(ValueError):
        eb.resolve_embeddings(
            role="x",
            weights_sha256="a" * 64,
            source_sha256="s" * 64,
            required_indices=np.asarray([0], dtype=np.int64),
            crops=None,
            cache_candidates=[],
            write_path=tmp_path / "o.npz",
            load_model=lambda: object(),
            threads=4,
            allow_embed=False,
        )


# ── End-to-end synthetic evaluation and strict JSON output ──────────────────
def test_end_to_end_evaluation_produces_strict_public_json():
    subjects, ages, protocol = _toy_protocol()
    n_crops = int(subjects.size)
    row_map = eb.embedding_row_map(np.arange(n_crops), n_crops=n_crops)
    embeddings = _fake_embeddings(n_crops, dim=8, seed=3)

    dev_pairs, _ = eb.build_split_pairs(subjects=subjects, ages=ages, protocol=protocol, split="dev")
    test_pairs, _ = eb.build_split_pairs(subjects=subjects, ages=ages, protocol=protocol, split="test")
    dev_scores, dev_labels = eb.scores_for_pairs(dev_pairs, embeddings, row_map)
    ops = {far: eb.calibrate_operating_point(dev_scores[dev_labels == 0], dev_scores[dev_labels == 1], far) for far in eb.FAR_TARGETS}
    crop_blur = np.linspace(0.0, 100.0, n_crops)
    dev_blur = eb.pair_blur(crop_blur, dev_pairs)
    cutpoints = eb.blur_cutpoints(dev_blur, n_bins=eb.BLUR_BINS)
    test_scores, _ = eb.scores_for_pairs(test_pairs, embeddings, row_map)
    summary = eb.evaluate_split(
        split="test",
        pairs=test_pairs,
        scores=test_scores,
        crop_blur=crop_blur,
        operating_points=ops,
        n_boot=100,
        seed=0,
        blur_cutpoints_dev=cutpoints,
    )
    payload = eb.build_public_output(
        protocol_hash=protocol.protocol_hash,
        source_sha256="0" * 64,
        split_summaries={"test": summary},
        pair_diagnostics={"test": {"candidate_pool_is_split_local": True}},
        model_meta={},
        blur_cutpoints=cutpoints,
        modality=eb.modality_axes(),
        n_boot=100,
    )
    eb.assert_public_output_has_no_ids(payload)
    text = json.dumps(eb.sanitize_nonfinite(payload), allow_nan=False)
    assert "NaN" not in text
    reloaded = json.loads(text)
    overall = reloaded["splits"]["test"]["operating_points"]["0.01"]["strata"]["overall"]
    assert overall["n_pos"] == overall["n_neg"] > 0
    assert 0.0 <= overall["fmr"] <= 1.0 and 0.0 <= overall["fnmr"] <= 1.0


def test_public_output_guard_rejects_subject_identifiers():
    bad = {"task_id": "x", "results": {"subject": 1}}
    with pytest.raises(AssertionError):
        eb.assert_public_output_has_no_ids(bad)


def test_many_exact_gap_strata_are_aggregate_counts_not_integer_row_vectors():
    pairs = []
    for gap in range(1, 31):
        pairs.extend([
            eb.VerificationPair(f"p{gap}", 1, "test", 0, 1, gap, gap, 1, gap + 1, gap, "q"),
            eb.VerificationPair(f"n{gap}", 0, "test", 0, 1, gap, gap + 100, 1, gap + 1, gap, "n"),
        ])
    result = eb.evaluate_split(
        split="test", pairs=pairs, scores=np.full(len(pairs), .5), crop_blur=np.ones(2),
        operating_points={}, n_boot=5,
    )
    eb.assert_public_output_has_no_ids(result)
    assert len(result["age_gap_distribution"]["gap_year_counts"]) == 30
    assert result["age_gap_distribution"]["gap_year_counts"]["gap_year_20"] == {"positive": 1, "negative": 1}


# ── Follow-up regressions ───────────────────────────────────────────────────
def test_gap_year_strata_are_per_integer_gap_in_addition_to_buckets():
    gaps = np.asarray([1, 1, 3, 12, 40])
    zeros = np.zeros_like(gaps)
    masks = eb.split_strata(
        pos_gap=gaps, neg_gap=gaps, pos_left_age=gaps, neg_left_age=gaps,
        pos_right_age=2 * gaps, neg_right_age=2 * gaps,
        pos_blur_bin=zeros, neg_blur_bin=zeros, n_blur_bins=eb.BLUR_BINS,
    )
    for year in (1, 3, 12, 40):
        pm, nm = masks[f"gap_year_{year:02d}"]
        assert pm.sum() == (gaps == year).sum() and nm.sum() == pm.sum()
    assert "gap_01_09" in masks and "gap_25_plus" in masks  # buckets retained
    # per-year masks partition the same pairs the buckets do
    assert sum(masks[f"gap_year_{y:02d}"][0].sum() for y in (1, 3, 12, 40)) == gaps.size


def test_zero_event_stratum_is_flagged_not_proof_of_zero_risk():
    # 50 negatives all well below threshold, 50 positives well above -> zero FA, zero FR
    neg = np.full(50, 0.1)
    pos = np.full(50, 0.9)
    pos_subject = np.repeat(np.arange(10), 5)
    summary = eb.evaluate_stratum(
        "overall", pos_mask=np.ones(50, bool), neg_mask=np.ones(50, bool),
        pos_scores=pos, neg_scores=neg,
        pos_subject=pos_subject, neg_left_subject=np.repeat(np.arange(10, 20), 5),
        neg_right_subject=np.repeat(np.arange(20, 30), 5),
        threshold=0.5, far_target=0.01, n_boot=200, seed=0,
    )
    assert summary["false_accepts"] == 0 and summary["false_rejects"] == 0
    assert summary["fmr_event_flag"] == "no_events_observed"
    assert summary["fnmr_event_flag"] == "no_events_observed"
    assert "NOT proof of zero risk" in summary["fmr_zero_event_note"]
    assert "rule of three" in summary["fmr_zero_event_note"]
    assert "not applicable" in summary["fmr_zero_event_note"]
    assert summary["fmr_ci95"] is None and summary["fnmr_ci95"] is None
    assert summary["fmr_empirical_bootstrap_ci95"] == [0.0, 0.0]


def test_stratum_reports_subjects_and_separate_valid_bootstrap_counts():
    neg = np.linspace(0.0, 1.0, 60)
    pos = np.linspace(0.0, 1.0, 60)
    summary = eb.evaluate_stratum(
        "overall", pos_mask=np.ones(60, bool), neg_mask=np.ones(60, bool),
        pos_scores=pos, neg_scores=neg,
        pos_subject=np.repeat(np.arange(6), 10),
        neg_left_subject=np.repeat(np.arange(6, 12), 10),
        neg_right_subject=np.repeat(np.arange(12, 18), 10),
        threshold=0.5, far_target=0.01, n_boot=200, seed=1,
    )
    assert summary["n_subjects"] > 0
    # Some resamples can draw an empty class, so valid counts are <= n_boot but reported separately.
    assert 0 < summary["bootstrap_valid_fmr"] <= 200
    assert 0 < summary["bootstrap_valid_fnmr"] <= 200
    assert summary["bootstrap_valid"] == min(summary["bootstrap_valid_fmr"], summary["bootstrap_valid_fnmr"])
    assert summary["conditional_on_dev_threshold"] is True
    assert summary["threshold_selection_variance_included"] is False
    assert "test SCORES never" in " ".join(eb.DISCLOSURES)
    assert "test is never used for matching" not in " ".join(eb.DISCLOSURES)


def test_threshold_is_m_plus_1th_largest_and_documented_as_such():
    neg = np.asarray([100.0, 90.0, 80.0, 70.0, 60.0])
    op = eb.calibrate_operating_point(neg, np.asarray([50.0]), 0.2)  # m = floor(0.2*5) = 1
    assert op.max_dev_accepts == 1
    assert op.threshold == 90.0  # 2nd largest = (m+1)-th largest
    assert op.dev_false_accepts == 1  # only 100.0 is strictly greater
    assert "(m+1)-th largest" in eb.calibrate_operating_point.__doc__


def test_validate_configuration_rejects_bad_values_before_inference():
    for kwargs in (
        {"n_boot": 0, "batch_size": 32, "threads": 2},
        {"n_boot": 10, "batch_size": 0, "threads": 2},
        {"n_boot": 10, "batch_size": 32, "threads": 3},
        {"n_boot": 10, "batch_size": 32, "threads": 2, "far_targets": (0.0,)},
        {"n_boot": 10, "batch_size": 32, "threads": 2, "far_targets": (1.0,)},
        {"n_boot": 10, "batch_size": 32, "threads": 2, "seeds": (1, 1)},
    ):
        with pytest.raises(ValueError):
            eb.validate_configuration(**kwargs)
    eb.validate_configuration(n_boot=1, batch_size=1, threads=1, seeds=(1, 2, 3))


def test_nonfinite_scores_are_rejected_before_rates():
    with pytest.raises(ValueError):
        eb.calibrate_operating_point(np.asarray([1.0, np.nan]), np.asarray([0.5]), 0.01)
    with pytest.raises(ValueError):
        eb.counts_at_threshold(np.asarray([np.inf]), np.asarray([0.0]), 0.5)
    with pytest.raises(ValueError):
        eb.assert_finite_scores(np.asarray([1.0, np.nan]))
    with pytest.raises(ValueError):
        eb.calibrate_operating_point(np.asarray([1.0]), np.asarray([0.5]), 0.0)  # far must be in (0,1)
    with pytest.raises(ValueError):
        eb.bootstrap_fmr_fnmr(
            pos_subject=np.asarray([1]), pos_reject=np.asarray([np.nan]),
            neg_left_subject=np.asarray([1]), neg_right_subject=np.asarray([2]),
            neg_accept=np.asarray([0.0]), n_boot=5,
        )


def test_scored_embeddings_output_does_not_clobber_keyed_cache(tmp_path):
    private = tmp_path / "private"
    eb.write_private_artifacts(
        private, pairs_by_split={"test": []}, row_map=np.arange(3),
        embeddings=np.ones((3, 2), dtype=np.float32), model_name="tuned_seed1",
    )
    assert (private / "private_scored_embeddings_tuned_seed1.npz").is_file()
    assert not (private / "private_embeddings_tuned_seed1.npz").exists()


def test_evaluate_split_rejects_gap_mismatch_and_class_imbalance():
    pairs = [
        eb.VerificationPair("p", 1, "test", 0, 1, 1, 1, 5, 12, 7, "q"),
        eb.VerificationPair("n", 0, "test", 2, 3, 2, 3, 5, 12, 8, "e"),  # wrong gap
    ]
    with pytest.raises(ValueError):
        eb.evaluate_split(
            split="test", pairs=pairs, scores=np.asarray([0.5, 0.5]),
            crop_blur=np.zeros(4), operating_points={}, n_boot=10,
        )
