from __future__ import annotations

import numpy as np
import pytest

from age_gap.evaluation.fgnet import _age_gap_diagnostics, _match_negatives, load_pairs
from age_gap.evaluation.metrics import roc_auc


def _synthetic_fgnet_cache(path) -> None:
    subjects = np.repeat(np.arange(4), 3)
    ages = np.tile(np.asarray([10, 30, 50]), 4)
    crops = np.arange(len(subjects), dtype=np.uint8)[:, None, None, None]
    crops = np.repeat(np.repeat(crops, 2, axis=1), 2, axis=2)
    np.savez(path, crops=crops, subjects=subjects, ages=ages)


def test_matches_one_unique_negative_per_positive_and_preserves_age_gap() -> None:
    subjects = np.repeat(np.arange(4), 3)
    ages = np.tile(np.asarray([10, 30, 50]), 4)
    positives = [
        (left, right, abs(int(ages[left]) - int(ages[right])))
        for subject in range(4)
        for left in range(subject * 3, subject * 3 + 3)
        for right in range(left + 1, subject * 3 + 3)
    ]

    matches = _match_negatives(subjects, ages, positives, tolerance=2, seed=19)

    assert len(matches) == len(positives)
    negative_pairs: set[tuple[int, int]] = set()
    for positive_index, left, right, observed_gap, endpoint_error, source_gap in matches:
        assert subjects[left] != subjects[right]
        assert endpoint_error <= 2
        assert source_gap == positives[positive_index][2] == observed_gap
        negative_pairs.add((min(left, right), max(left, right)))
    assert len(negative_pairs) == len(matches)


def test_matching_is_deterministic_and_reports_unmatched_positives() -> None:
    subjects = np.asarray([0, 0, 1, 1, 2, 2])
    ages = np.asarray([10, 30, 10, 30, 90, 130])
    positives = [(0, 1, 20), (2, 3, 20), (4, 5, 40)]

    first = _match_negatives(subjects, ages, positives, tolerance=0, seed=7)
    second = _match_negatives(subjects, ages, positives, tolerance=0, seed=7)
    assert first == second
    assert len(first) == 2
    assert {match[0] for match in first} == {0, 1}
    assert len({(min(match[1], match[2]), max(match[1], match[2])) for match in first}) == 2


def test_load_pairs_has_balanced_gap_strata_and_explicit_legacy_protocol(tmp_path) -> None:
    cache = tmp_path / "fgnet.npz"
    _synthetic_fgnet_cache(cache)
    first = load_pairs(
        cache,
        protocol="endpoint_age_matched",
        endpoint_age_tolerance=0,
        return_metadata=True,
    )
    second = load_pairs(
        cache,
        protocol="endpoint_age_matched",
        endpoint_age_tolerance=0,
        return_metadata=True,
    )
    images_a, images_b, labels, gaps, metadata = first
    assert np.array_equal(metadata["subject_a"], second[4]["subject_a"])
    assert np.array_equal(metadata["subject_b"], second[4]["subject_b"])
    assert np.array_equal(metadata["age_a"], second[4]["age_a"])
    assert metadata["protocol"] == "endpoint_age_matched"
    assert len(images_a) == len(images_b) == len(labels) == len(gaps)
    negative = labels == 0
    assert int(labels.sum()) == int(negative.sum())
    assert np.all(metadata["subject_a"][negative] != metadata["subject_b"][negative])
    assert np.all(gaps[negative] >= 0)
    assert np.all(metadata["negative_endpoint_match_error"] == 0)
    assert metadata["n_positive_unmatched"] == 0
    assert metadata["positive_coverage"] == 1.0
    assert np.array_equal(
        np.sort(metadata["stratum_age_gap"][labels == 1]),
        np.sort(metadata["stratum_age_gap"][negative]),
    )
    diagnostics = _age_gap_diagnostics(labels, metadata["observed_age_gap"])
    assert diagnostics["age_gap_ks_statistic"] == 0.0
    assert roc_auc(metadata["observed_age_gap"], labels) == 0.5

    legacy = load_pairs(cache, protocol="legacy_random", return_metadata=True)
    assert legacy[4]["protocol"] == "legacy_random"
    assert np.all(legacy[3][legacy[2] == 0] == -1)


def test_matched_protocol_rejects_multiple_negatives_per_positive(tmp_path) -> None:
    cache = tmp_path / "fgnet.npz"
    _synthetic_fgnet_cache(cache)
    with pytest.raises(ValueError, match="requires neg_per_pos=1.0"):
        load_pairs(cache, neg_per_pos=2.0, protocol="endpoint_age_matched")


def test_default_protocol_remains_explicitly_labeled_legacy(tmp_path) -> None:
    cache = tmp_path / "fgnet.npz"
    _synthetic_fgnet_cache(cache)
    default = load_pairs(cache, return_metadata=True)
    assert default[4]["protocol"] == "legacy_random"
    assert np.all(default[3][default[2] == 0] == -1)
