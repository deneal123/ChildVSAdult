import pytest

from scripts.audit_cacon_batch_identities_v1 import analyse_batches


def test_same_person_cross_source_views_are_counted_separately_from_own_positives():
    result = analyse_batches([0, 0, 1], [[0, 1, 2]])
    assert result["same_recorded_person_source_pairs"] == 1
    assert result["directed_same_recorded_person_view_negatives"] == 18
    assert result["directed_other_source_view_negatives"] == 54
    assert result["recorded_identity_collision_fraction"] == 1 / 3
    assert result["anchors_with_another_same_recorded_person"] == 6


def test_singleton_tail_and_noncolliding_batches():
    result = analyse_batches([0, 0, 1], [[0, 2], [1]])
    assert result["recorded_identity_collision_fraction"] == 0
    assert result["batches_with_identity_collision"] == 0
    assert result["directed_other_source_view_negatives"] == 18
    assert analyse_batches([0], [[0]])["recorded_identity_collision_fraction"] is None


@pytest.mark.parametrize("batches", [[[0, 0]], [[1]], [[False, 1]], [[0, 1], []], [[0, 2]]])
def test_missing_duplicate_or_invalid_source_coverage_refused(batches):
    with pytest.raises(ValueError):
        analyse_batches([0, 1], batches)


def test_recorded_labels_not_truth_or_human_responses():
    with pytest.raises(ValueError):
        analyse_batches([True], [[0]])
    result = analyse_batches([0, 0], [[0, 1]])
    assert result["recorded_identity_collision_fraction"] == 1
