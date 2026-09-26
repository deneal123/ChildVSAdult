from __future__ import annotations

from age_gap.common.schemas import AgeLabel, IdentityGroup, Pair
from age_gap.evaluation.internal_age_matched import (
    build_endpoint_age_matched_pairs,
    filter_matched_blocks_by_faces,
    leave_one_subject_out,
    paired_subject_bootstrap,
    split_matched_blocks,
)


def _group(group_id: str, faces: list[str], ages: list[int]) -> IdentityGroup:
    return IdentityGroup(
        identity_group_id=group_id,
        source_post_id=group_id,
        faces=faces,
        age_labels=[AgeLabel(face_id=face, age=age) for face, age in zip(faces, ages, strict=True)],
    )


def test_endpoint_age_matching_fixes_anchor_and_matches_positive_counterpart():
    groups = [
        _group("person_a", ["a_child", "a_adult"], [10, 40]),
        _group("person_b", ["b_adult"], [39]),
        _group("person_c", ["c_adult"], [41]),
        _group("train_person", ["train_adult"], [40]),
    ]
    positive = Pair(
        pair_id="positive_a_child_to_a_adult",
        face_a="a_child",
        face_b="a_adult",
        label=1,
        pair_type="positive_same_post",
        identity_group_a="person_a",
        identity_group_b="person_a",
        age_a=10,
        age_b=40,
        age_gap=30,
        split="test",
    )

    pairs, diagnostics = build_endpoint_age_matched_pairs(
        [positive],
        groups,
        group_splits={
            "person_a": "test",
            "person_b": "test",
            "person_c": "test",
            "train_person": "train",
        },
        seed=7,
        tolerance_years=1,
    )

    assert len(pairs) == 2
    negative = next(pair for pair in pairs if pair.label == 0)
    assert negative.face_a == positive.face_a  # anchor remains unchanged
    assert negative.face_b in {"b_adult", "c_adult"}
    assert abs(negative.age_b - positive.age_b) <= 1
    assert negative.identity_group_b != positive.identity_group_a
    assert negative.age_gap == abs(positive.age_a - negative.age_b)
    assert diagnostics["eligible_positive_count"] == 1
    assert diagnostics["matched_positive_count"] == 1
    assert diagnostics["coverage"] == 1.0
    assert diagnostics["age_gap_only_predictive_auc"] == 1.0


def test_endpoint_age_matching_is_seeded_and_ignores_other_splits():
    groups = [
        _group("person_a", ["a_child", "a_adult"], [10, 40]),
        _group("test_person", ["test_adult"], [40]),
        _group("train_person", ["train_adult"], [40]),
    ]
    positive = Pair(
        pair_id="positive_a",
        face_a="a_child",
        face_b="a_adult",
        label=1,
        pair_type="positive_same_post",
        identity_group_a="person_a",
        identity_group_b="person_a",
        age_a=10,
        age_b=40,
        age_gap=30,
        split="test",
    )
    kwargs = {
        "group_splits": {"person_a": "test", "test_person": "test", "train_person": "train"},
        "seed": 123,
        "tolerance_years": 0,
    }

    first, _ = build_endpoint_age_matched_pairs([positive], groups, **kwargs)
    second, _ = build_endpoint_age_matched_pairs([positive], groups, **kwargs)
    negative = next(pair for pair in first if pair.label == 0)
    assert negative.face_b == "test_adult"
    assert [pair.to_dict() for pair in first] == [pair.to_dict() for pair in second]


def test_endpoint_age_matching_reports_zero_coverage_when_no_test_control_exists():
    groups = [_group("person_a", ["a_child", "a_adult"], [10, 40])]
    positive = Pair(
        pair_id="positive_a",
        face_a="a_child",
        face_b="a_adult",
        label=1,
        pair_type="positive_same_post",
        identity_group_a="person_a",
        identity_group_b="person_a",
        age_a=10,
        age_b=40,
        age_gap=30,
        split="test",
    )

    pairs, diagnostics = build_endpoint_age_matched_pairs(
        [positive], groups, group_splits={"person_a": "test"}
    )

    assert pairs == []
    assert diagnostics["eligible_positive_count"] == 1
    assert diagnostics["matched_positive_count"] == 0
    assert diagnostics["unmatched_positive_count"] == 1
    assert diagnostics["coverage"] == 0.0


def _matched_fixture() -> tuple[list[Pair], list[Pair]]:
    positives = []
    negatives = []
    for i in range(3):
        source = f"source_{i}"
        impostor = f"impostor_{i}"
        positives.append(
            Pair(
                pair_id=f"pos_{i}",
                face_a=f"{source}_anchor",
                face_b=f"{source}_counterpart",
                label=1,
                pair_type="positive_same_post",
                identity_group_a=source,
                identity_group_b=source,
                age_a=10,
                age_b=40,
                age_gap=30,
            )
        )
        negatives.append(
            Pair(
                pair_id=f"neg_{i}",
                face_a=f"{source}_anchor",
                face_b=f"{impostor}_face",
                label=0,
                pair_type="negative_endpoint_age_matched",
                identity_group_a=source,
                identity_group_b=impostor,
                age_a=10,
                age_b=40,
                age_gap=30,
            )
        )
    return positives, negatives


def test_missing_crop_filter_drops_whole_matched_observation_and_keeps_balance():
    positives, negatives = _matched_fixture()
    positive_block, negative_block = split_matched_blocks([*positives, *negatives])
    assert positive_block == positives and negative_block == negatives

    kept_pos, kept_neg, removed = filter_matched_blocks_by_faces(
        positives,
        negatives,
        {
            "source_0_anchor", "source_0_counterpart", "impostor_0_face",
            "source_1_anchor", "source_1_counterpart", "impostor_1_face",
            "source_2_anchor", "impostor_2_face",
        },
    )
    assert len(kept_pos) == len(kept_neg) == 2
    assert removed == 1


def test_subject_bootstrap_is_paired_deterministic_and_loso_removes_linked_pairs():
    positives, negatives = _matched_fixture()
    scores = {
        "facenet_frozen": {
            "pos_0": 0.4, "pos_1": 0.6, "pos_2": 0.5,
            "neg_0": 0.5, "neg_1": 0.5, "neg_2": 0.5,
        },
        "facenet_tuned": {
            "pos_0": 0.7, "pos_1": 0.65, "pos_2": 0.8,
            "neg_0": 0.4, "neg_1": 0.45, "neg_2": 0.3,
        },
    }
    kwargs = {
        "n_boot": 100,
        "seed": 19,
    }
    first = paired_subject_bootstrap(
        positives, negatives, scores, scores, **kwargs
    )
    second = paired_subject_bootstrap(
        positives, negatives, scores, scores, **kwargs
    )
    assert first == second
    assert first["valid_replicates"] > 0
    assert first["point_estimates"]["delta_comparison_minus_reference"] == 0.5
    low, high = first["ci95"]["delta_comparison_minus_reference"]
    assert low <= first["point_estimates"]["delta_comparison_minus_reference"] <= high

    loso = leave_one_subject_out(positives, negatives, scores, scores)
    assert loso["n"] == 6
    assert loso["delta_comparison_minus_reference"]["positive_fraction"] == 1.0
