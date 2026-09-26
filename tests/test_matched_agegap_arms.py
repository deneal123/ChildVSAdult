from __future__ import annotations

from copy import deepcopy

from scripts.build_matched_agegap_arms import build_arms


def _positive(pair_id: str, face_a: str, face_b: str, identity: str, age_a: int, age_b: int) -> dict:
    return {
        "pair_id": pair_id,
        "face_a": face_a,
        "face_b": face_b,
        "label": 1,
        "pair_type": "positive_same_post",
        "identity_group_a": identity,
        "identity_group_b": identity,
        "age_a": age_a,
        "age_b": age_b,
        "age_gap": abs(age_a - age_b),
        "status": "ok",
        "split": "train",
    }


def _fixture() -> tuple[list[dict], dict[str, str]]:
    pairs = [
        _positive("low-1", "l1a", "l1b", "g1", 20, 21),
        _positive("low-2", "l2a", "l2b", "g2", 21, 22),
        _positive("low-3", "l3a", "l3b", "g7", 20, 22),
        _positive("cross-1", "c1a", "c1b", "g3", 10, 40),
        _positive("cross-2", "c2a", "c2b", "g4", 15, 40),
        _positive("pool-1", "p1a", "p1b", "g5", 20, 21),
        _positive("pool-2", "p2a", "p2b", "g6", 37, 41),
        _positive("pool-3", "p3a", "p3b", "g8", 42, 46),
        _positive("shared-low", "s1a", "s1b", "g9", 10, 11),
        _positive("shared-cross", "s2a", "s2b", "g9", 10, 40),
        {"pair_id": "v1", "label": 1, "split": "val", "marker": "same"},
        {"pair_id": "t1", "label": 0, "split": "test", "marker": "same"},
    ]
    groups = {f"g{i}": f"person-{i}" for i in range(1, 10)}
    return pairs, groups


def test_build_arms_is_deterministic_and_keeps_heldout_rows_identical() -> None:
    pairs, groups = _fixture()
    original = deepcopy(pairs)
    arms1, summary1 = build_arms(
        pairs, groups, target_identities=2, target_per_class=2, target_faces=4, seed=7,
        min_bin_support=2,
    )
    arms2, summary2 = build_arms(
        pairs, groups, target_identities=2, target_per_class=2, target_faces=4, seed=7,
        min_bin_support=2,
    )

    assert arms1 == arms2
    assert summary1 == summary2
    assert pairs == original
    assert summary1["positive_identity_overlap_between_arms"] == 0
    assert summary1["positive_image_overlap_between_arms"] == 0
    assert summary1["cross_stratum_eligible_identity_exclusions"] == 1
    low = summary1["arms"]["LOW"]
    cross = summary1["arms"]["CROSS"]
    assert low["train_positive_count"] == cross["train_positive_count"] == 2
    assert low["positive_unique_faces"] == cross["positive_unique_faces"] == 4
    for arm in arms1.values():
        assert [row for row in arm if row["split"] in {"val", "test"}] == original[-2:]


def test_negatives_are_identity_disjoint_and_match_endpoint_b_age_bins() -> None:
    pairs, groups = _fixture()
    arms, summary = build_arms(
        pairs, groups, target_identities=2, target_per_class=2, target_faces=4, seed=11,
        bin_years=5, min_bin_support=2,
    )
    for name, arm in arms.items():
        positives = [row for row in arm if row.get("split") == "train" and row.get("label") == 1]
        negatives = [row for row in arm if row.get("split") == "train" and row.get("label") == 0]
        positive_by_id = {row["pair_id"]: row for row in positives}
        positive_people = {groups[row["identity_group_a"]] for row in positives}
        assert len(positives) == len(negatives) == 2
        assert len({tuple(sorted((row["face_a"], row["face_b"]))) for row in negatives}) == 2
        for negative in negatives:
            target = positive_by_id[negative["matched_target_pair_id"]]
            assert negative["identity_group_a"] != negative["identity_group_b"]
            assert negative["face_a"] == target["face_a"]
            assert negative["age_b"] // 10 == target["age_b"] // 10
            assert abs(negative["age_b"] - target["age_b"]) <= 1
            assert groups[negative["identity_group_b"]] in positive_people
        assert summary["arms"][name]["negative_endpoint_b_age_bin_match_fraction"] == 1.0
        assert summary["arms"][name]["negative_pairs_reused"] == 0
        assert set(summary["arms"][name]["train_age_only_class_auc"]) == {
            "age_a", "age_b", "age_gap"
        }
        assert (
            summary["arms"][name]["positive_endpoint_b_age_bins"]
            == summary["arms"][name]["negative_endpoint_b_age_bins"]
        )


def test_seed_changes_tie_breaks_and_manifest_diagnostics_surface_unmatched_covariates() -> None:
    pairs, groups = _fixture()
    results = [
        build_arms(
            pairs, groups, target_identities=2, target_per_class=2, target_faces=4, seed=seed,
            min_bin_support=2,
        )
        for seed in range(12)
    ]
    assert len({str(arms["LOW"]) for arms, _ in results}) > 1
    first = results[0][1]
    assert "endpoint A age and pairwise age gap for negatives" in first["covariates_not_matched"]
    assert "no generic source arm" in first["generic_face_source"]
