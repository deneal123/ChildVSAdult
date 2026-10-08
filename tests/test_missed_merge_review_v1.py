import pytest

from scripts.build_missed_merge_review_v1 import sample_candidates


def candidate(i, score=0.72, splits=("train", "test")):
    return dict(group_id_a=f"left{i}", group_id_b=f"right{i}", person_id_a=f"a{i}", person_id_b=f"b{i}",
                split_a=splits[0], split_b=splits[1], group_centroid_cosine=score)


def test_full_high_risk_and_weighted_lower_strata():
    rows = [candidate(i) for i in range(100)] + [candidate(100 + i, 0.85) for i in range(30)]
    selected, strata = sample_candidates(rows)
    assert len(selected) == 55
    assert strata["test-train:070_075"] == dict(population=100, selected=25, inclusion_probability=0.25)
    assert strata["test-train:ge085"]["inclusion_probability"] == 1
    assert sample_candidates(rows) == sample_candidates(list(reversed(rows)))


@pytest.mark.parametrize("field,value", [
    ("group_centroid_cosine", float("nan")), ("group_centroid_cosine", 0.64),
    ("group_centroid_cosine", True), ("split_a", "test"), ("split_b", "unknown"),
    ("person_id_a", "b0"), ("group_id_a", "right0"),
])
def test_invalid_candidates_refused(field, value):
    row = candidate(0)
    row[field] = value
    with pytest.raises(ValueError):
        sample_candidates([row])


def test_duplicate_pair_refused():
    row = candidate(0)
    with pytest.raises(ValueError):
        sample_candidates([row, row.copy()])


def test_split_strata_are_separate_and_small_population_fully_kept():
    selected, strata = sample_candidates([candidate(0, splits=("train", "val")), candidate(1)])
    assert len(selected) == 2 and len(strata) == 2
    assert all(value["inclusion_probability"] == 1 for value in strata.values())
