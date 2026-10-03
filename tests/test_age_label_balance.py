import numpy as np
import pytest

from scripts.audit_age_label_balance import diagnostic, fgnet_metadata, paired_rows


def test_identical_distributions():
    result = diagnostic([(1, 30), (2, 31)], [(2, 31), (1, 30)])
    assert all(v["empirical_total_variation"] == 0 for v in result["features"].values())


def test_chance_rank_auc_can_hide_perfect_nonlinear_sample_cue():
    result = diagnostic([(0, 10), (2, 10)], [(1, 10), (1, 10)])
    age = result["features"]["age_a"]
    assert age["rank_auc_positive_higher"] == .5
    assert age["empirical_total_variation"] == 1
    assert age["equal_prior_in_sample_oracle_accuracy"] == 1


def test_joint_difference_despite_identical_marginals():
    result = diagnostic([(0, 0), (1, 1)], [(0, 1), (1, 0)])
    assert result["features"]["age_a"]["empirical_total_variation"] == 0
    assert result["features"]["age_b"]["empirical_total_variation"] == 0
    assert result["features"]["ordered_endpoint_ages"]["empirical_total_variation"] == 1


@pytest.mark.parametrize("p,n", [([], []), ([(1, 2)], []), ([(True, 2)], [(1, 2)]),
                                ([(1.0, 2)], [(1, 2)]), ([(-1, 2)], [(1, 2)])])
def test_invalid_metadata(p, n):
    with pytest.raises(ValueError):
        diagnostic(p, n)


def rows():
    return [{"pair_id": "p", "label": 1, "face_a": "a", "face_b": "b",
             "identity_group_a": "x", "identity_group_b": "x", "age_a": 1, "age_b": 30, "age_gap": 29},
            {"pair_id": "n", "label": 0, "face_a": "a", "face_b": "c", "matched_target_pair_id": "p",
             "identity_group_a": "x", "identity_group_b": "y", "age_a": 1, "age_b": 31, "age_gap": 30}]


def test_paired_target_or_blocks():
    assert paired_rows(rows(), explicit_targets=True) == paired_rows(rows(), explicit_targets=False)


@pytest.mark.parametrize("field,value", [("matched_target_pair_id", "unknown"), ("face_a", "different"),
                                        ("age_gap", 29), ("identity_group_b", "x"), ("label", True)])
def test_broken_negative_binding(field, value):
    sample = rows()
    sample[1][field] = value
    with pytest.raises(ValueError):
        paired_rows(sample, explicit_targets=True)


def test_fgnet_metadata_does_not_read_object_crops(tmp_path):
    path = tmp_path / "cache.npz"
    np.savez(path, subjects=np.array([1, 1, 2, 2]), ages=np.array([1, 30, 1, 30]),
             crops=np.array([{"must_not_read": True}], dtype=object))
    result = fgnet_metadata(path)
    assert result["source_positive_gap_25plus"]["positive_pairs"] == 2
    assert all(v["empirical_total_variation"] == 0 for v in result["overall"]["features"].values())
