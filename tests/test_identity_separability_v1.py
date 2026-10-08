import numpy as np
import pytest

from scripts.identity_separability_v1 import infer, separation


def data():
    return (np.array([.8, .7, .9, .6, .1, .3, .2, .4]),
            np.array([1, 1, 1, 1, 0, 0, 0, 0]),
            np.array([0, 1, 2, 3, 0, 1, 2, 3]),
            np.array([0, 1, 2, 3, 1, 2, 3, 0]))


def test_standardized_difference_and_affine_invariance():
    scores, labels, _, _ = data()
    row = separation(scores, labels)
    assert row["standardized_mean_difference"] > 4
    assert separation(scores * 3 + 2, labels)["standardized_mean_difference"] == pytest.approx(
        row["standardized_mean_difference"])


def test_shared_draws_identical_models_have_zero_delta():
    scores, labels, a, b = data()
    row = infer(dict(frozen=scores, tuned=scores), labels, a, b, resamples=50)
    assert row["status"] == "ok"
    assert row["models"]["tuned"]["delta_ci95"] == [0, 0]
    assert row["positive_weight"] == "owner multiplicity once"


def test_person_metadata_conflict_rejected():
    scores, labels, a, b = data()
    b[0] = 1
    with pytest.raises(ValueError, match="contradict"):
        infer(dict(frozen=scores), labels, a, b)


def test_degenerate_variance_is_unavailable_not_infinity():
    scores, labels, a, b = data()
    row = infer(dict(frozen=np.ones_like(scores)), labels, a, b)
    assert row["status"].startswith("unavailable")


def test_common_pair_mapping_requires_exact_image_endpoints():
    from scripts.run_common_mechanism_v1 import validate_pairs

    people = np.array([0, 0, 1])
    left, right, labels = np.array([0, 0]), np.array([1, 2]), np.array([1, 0])
    validate_pairs(left, right, labels, people[left], people[right], people)
    with pytest.raises(ValueError, match="endpoint metadata"):
        validate_pairs(left, right, labels, people[left], np.array([1, 0]), people)
    with pytest.raises(ValueError, match="contradict"):
        validate_pairs(left, right, 1 - labels, people[left], people[right], people)
