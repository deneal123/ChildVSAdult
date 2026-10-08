import itertools

import numpy as np
import pytest

from scripts.match_exact_negative_quality import minimum_cost_edges, positive_image_map, quality_arm


def test_global_optimum_not_greedy_and_zero_cost_present():
    e, f = ("a", "b"), ("a", "c")
    chosen, report = minimum_cost_edges([{e: 0.0, f: 2.0}, {e: 1.0, f: 100.0}])
    assert chosen == [f, e]
    assert report["total_unshifted_cost"] == 3.0
    assert minimum_cost_edges([{e: 0.0}])[0] == [e]


def test_hall_failure_not_only_empty_candidate():
    e = ("a", "b")
    chosen, report = minimum_cost_edges([{e: 0.0}, {e: 1.0}])
    assert chosen is None
    assert report["matched_targets"] == 1


def test_empty_candidate_no_drop():
    chosen, report = minimum_cost_edges([{}, {("a", "b"): 0.0}])
    assert chosen is None
    assert report["targets"] == 2


@pytest.mark.parametrize(
    "edge,cost",
    [
        (("a", "a"), 0),
        (("b", "a"), 1),
        (("a", "b"), -1),
        (("a", "b"), np.nan),
        (("a", "b"), np.inf),
    ],
)
def test_invalid_edges_or_cost(edge, cost):
    with pytest.raises(ValueError):
        minimum_cost_edges([{edge: cost}])


def test_small_bruteforce_cost():
    edges = [("a", "b"), ("a", "c"), ("b", "c"), ("b", "d")]
    rng = np.random.default_rng(42)
    for _ in range(20):
        costs = rng.integers(0, 20, (3, 4))
        options = [dict(zip(edges, row, strict=True)) for row in costs]
        _, report = minimum_cost_edges(options)
        optimum = min(
            sum(costs[i, j] for i, j in enumerate(js)) for js in itertools.permutations(range(4), 3)
        )
        assert report["total_unshifted_cost"] == optimum


def fixture_rows():
    groups = {"p": "P", "q": "Q"}
    positives = [
        dict(
            pair_id="p1",
            face_a="a",
            face_b="b",
            label=1,
            split="train",
            identity_group_a="p",
            identity_group_b="p",
            age_a=10,
            age_b=11,
            age_gap=1,
        ),
        dict(
            pair_id="p2",
            face_a="c",
            face_b="d",
            label=1,
            split="train",
            identity_group_a="q",
            identity_group_b="q",
            age_a=10,
            age_b=11,
            age_gap=1,
        ),
    ]
    negs = [
        dict(
            positives[0],
            pair_id="n1",
            label=0,
            face_b="d",
            identity_group_b="q",
            matched_target_pair_id="p1",
        ),
        dict(
            positives[1],
            pair_id="n2",
            label=0,
            face_b="b",
            identity_group_b="p",
            matched_target_pair_id="p2",
        ),
    ]
    return positives + negs + [dict(split="test", sentinel="unchanged")], groups


def test_age_identity_genuine_exclusion_and_heldout():
    rows, groups = fixture_rows()
    features = {f: np.zeros(18) for f in "abcd"}
    output, report = quality_arm(
        rows, groups, features, arm="LOW", forbidden={("a", "b"), ("c", "d")}
    )
    assert output[-1] == rows[-1]
    assert output[:2] == rows[:2]
    assert report["endpoint_presentations"]["total"] == 8
    assert all(v["empirical_weighted_total_variation"] == 0 for v in report["age_balance"].values())
    assert quality_arm(rows, groups, features, arm="LOW", forbidden={("a", "d")})[0] is None


def test_map_ignores_heldout_quality_and_retains_missingness():
    rows, groups = fixture_rows()
    quality = {f: {"blur_var": 2 if f == "a" else None} for f in "abcd"}
    a, fitted = positive_image_map({"LOW": rows}, groups, quality)
    b, other = positive_image_map({"LOW": rows}, groups, dict(quality, test={"blur_var": 1e50}))
    assert fitted == other
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["a"], a["b"])
    assert len(a["a"]) == 18


def test_requires_quality_for_selected_images():
    rows, groups = fixture_rows()
    with pytest.raises(ValueError, match="quality record"):
        positive_image_map({"LOW": rows}, groups, {})


def test_same_frame_comparison_identical_versions():
    from scripts.audit_nuisance_overlap import VIEWS
    from scripts.compare_quality_matching import compare

    rows, groups = fixture_rows()
    cross = [
        dict(
            r,
            pair_id="cross_" + r["pair_id"],
            **(
                {"matched_target_pair_id": "cross_" + r["matched_target_pair_id"]}
                if "matched_target_pair_id" in r
                else {}
            ),
        )
        if r["split"] == "train"
        else r
        for r in rows
    ]
    arms = {"LOW": rows, "CROSS": cross}
    ledger = [
        {"view": v, "pair_id": r["pair_id"], "candidate_loss_weight": 1.0}
        for v in VIEWS
        for arm in arms.values()
        for r in arm
        if r["split"] == "train"
    ]
    features = {f: np.arange(18, dtype=float) * i for i, f in enumerate("abcd")}
    result = compare({"old": arms, "new": arms}, groups, features, {"old": ledger, "new": ledger})
    assert result["old"] == result["new"]
    assert result["old"]["cross_arm_combined"]["uniform"] == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="duplicate ledger"):
        compare({"old": arms}, groups, features, {"old": ledger + ledger[:1]})
