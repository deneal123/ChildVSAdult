import copy
import json

import numpy as np
import pytest

from scripts import audit_nuisance_overlap as audit


def fixture():
    groups = {"V": "VP", "T": "TP"}
    heldout = [
        dict(
            pair_id=split,
            label=1,
            split=split,
            face_a=f"{split}a",
            face_b=f"{split}b",
            identity_group_a=group,
            identity_group_b=group,
        )
        for split, group in (("val", "V"), ("test", "T"))
    ]
    arms, quality = {}, {}
    for arm, prefix, ages in (("LOW", "L", (10, 11, 12)), ("CROSS", "C", (10, 40, 41))):
        rows = []
        for i in range(5):
            group = f"{prefix}{i}"
            groups[group] = f"{prefix}P{i}"
            for side in "abc":
                face = f"{group}{side}"
                quality[face] = dict(
                    is_usable=True,
                    photo_id=f"photo-{face}",
                    face_width=100 + i * 20 + (arm == "CROSS") * 15,
                    face_height=120 + i * 15,
                    blur_var=None if i == 0 else 30 + i * 20,
                    det_score=0.9,
                    pose_yaw_proxy=-0.1 * i,
                    pose_roll_rad=0.05 * i,
                )
            sides = [(0, 1), (1, 2)] if arm == "LOW" else [(0, 1), (0, 2)]
            for j, (a, b) in enumerate(sides):
                positive = dict(
                    pair_id=f"{group}p{j}",
                    label=1,
                    split="train",
                    face_a=f"{group}{'abc'[a]}",
                    face_b=f"{group}{'abc'[b]}",
                    identity_group_a=group,
                    identity_group_b=group,
                    age_a=ages[a],
                    age_b=ages[b],
                    age_gap=ages[b] - ages[a],
                )
                negative = {
                    **positive,
                    "pair_id": f"{group}n{j}",
                    "label": 0,
                    "face_b": f"{prefix}{(i + 1) % 5}{'abc'[b]}",
                    "identity_group_b": f"{prefix}{(i + 1) % 5}",
                }
                rows.extend([positive, negative])
        arms[arm] = rows + copy.deepcopy(heldout)
    return arms, groups, quality


def test_person_fold_assignment_is_deterministic_and_cluster_disjoint():
    arms = {"LOW": [], "CROSS": []}
    people = {"LOW": ["L0", "L0", "L1", "L2"], "CROSS": ["C0", "C1", "C1", "C2"]}
    a = audit.person_folds(arms, people, n_folds=3)
    np.testing.assert_array_equal(a, audit.person_folds(arms, people, n_folds=3))
    assert a[0] == a[1] and a[5] == a[6]
    assert set(a[:4]) == set(a[4:]) == {0, 1, 2}
    with pytest.raises(ValueError, match="both arms"):
        audit.person_folds(arms, {"LOW": ["A", "B"], "CROSS": ["A", "C"]}, n_folds=2)


def test_quality_features_symmetric_and_primary_does_not_include_ages():
    row = dict(face_a="a", face_b="b", age_a=10, age_b=40)
    quality = {
        "a": dict(face_width=10, pose_yaw_proxy=-0.3),
        "b": dict(face_width=30, pose_yaw_proxy=0.1),
    }
    x = audit.features([row], quality, "quality_only")
    swapped = {**row, "face_a": "b", "face_b": "a", "age_a": 80, "age_b": 1}
    np.testing.assert_array_equal(x, audit.features([swapped], quality, "quality_only"))
    assert x.shape == (1, 27)
    younger = audit.features([row], quality, "younger_anchor")
    older = audit.features([row], quality, "older_anchor")
    assert younger.shape == older.shape == (1, 28)
    assert younger[0, 18] == 10 and older[0, 18] == 40
    assert np.isnan(x[0, 0]) and x[0, 18] == 2  # Missing image width is not zero metadata.


def test_fit_fold_imputation_scaling_does_not_use_score_fold(monkeypatch):
    captured = []
    real = audit.StandardScaler

    def scaler():
        result = real()
        original = result.fit

        def fit(data):
            captured.append(data.copy())
            return original(data)

        result.fit = fit
        return result

    monkeypatch.setattr(audit, "StandardScaler", scaler)
    audit.crossfit([[np.nan], [20], [1], [3]], [0, 1, 0, 1], ["A", "B", "C", "D"], [0, 0, 1, 1])
    np.testing.assert_array_equal(captured[0], [[1], [3]])
    np.testing.assert_array_equal(captured[1], [[20], [20]])


@pytest.mark.parametrize(
    "kind", ["person_split", "person_arm", "inf", "bool_label", "empty_arm_fold"]
)
def test_crossfit_invalid_input_refused(kind):
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    y, people, folds = [0, 1, 0, 1], ["A", "B", "C", "D"], [0, 0, 1, 1]
    if kind == "person_split":
        people[2] = "A"
    elif kind == "person_arm":
        people[1] = "A"
    elif kind == "inf":
        x[0] = np.inf
    elif kind == "bool_label":
        y = [False, True, False, True]
    else:
        folds = [0, 1, 0, 1]
    with pytest.raises(ValueError):
        audit.crossfit(x, y, people, folds)


def test_weighted_diagnostics_exact_ties_missing_and_constant_cases():
    result = audit.weighted_compare([1, 3], [2, 4], [3, 1], [1, 3])
    assert result["weighted_mean_difference"] == 2
    assert result["weighted_smd"] == 2
    assert result["weighted_ecdf_max_distance"] == 0.75
    missing = audit.weighted_compare([1, None, 3], [2, 4], [1, 8, 1], [1, 3])
    assert missing["LOW"]["missing_weight_fraction"] == 0.8
    assert audit.weighted_compare([None], [1], [1], [1])["weighted_smd"] is None
    assert audit.weighted_compare([1, 1], [2, 2], [1, 1], [1, 1])["weighted_smd"] is None
    assert audit.effective_size([1, 1, 1]) == 3
    assert audit.effective_size([1, 0, 0]) == 1
    zero = audit.weighted_compare([1, 3], [2, 4], [0, 1], [1, 0])
    assert zero["weighted_mean_difference"] == -1
    assert zero["weighted_ecdf_max_distance"] == 1
    no_observed_mass = audit.weighted_compare([None, 3], [2, 4], [1, 0], [1, 1])
    assert no_observed_mass["weighted_smd"] is None


@pytest.mark.parametrize("weights", [[-1, 2], [0, 0], [np.nan, 1]])
def test_invalid_diagnostic_weights_refused(weights):
    with pytest.raises(ValueError):
        audit.weighted_compare([1, 3], [2, 4], weights, [1, 1])


def test_actual_diagnostic_preserves_arms_and_separates_private_ids():
    arms, groups, quality = fixture()
    before = copy.deepcopy((arms, groups, quality))
    result, private = audit.diagnose(arms, groups, quality)
    assert (arms, groups, quality) == before
    assert len(private) == 60  # 20 positive rows times three prespecified views.
    for report in result["views"].values():
        assert len(report["folds"]) == 5
        assert len(report["pair_quality_endpoint_views"]) == 18
        for arm in ("LOW", "CROSS"):
            row = report["arms"][arm]
            assert row["normalized_weight_mean"] == pytest.approx(1)
            assert 1 <= row["pair_effective_size"] <= 10 + 1e-12
            assert 1 <= row["recorded_person_mass_effective_size"] <= 5 + 1e-12
    assert result["training_weights_ready"] is False
    assert result["quality_balance_claimed"] is False
    assert result["recognition_outcomes_used"] is False
    assert result["causal_source_superiority"] is False
    text = json.dumps(result, allow_nan=False)
    for identifier in ("L0a", "LP0", "photo-L0a", "L0p0"):
        assert identifier not in text
    assert any(r["recorded_person"] == "LP0" for r in private)
