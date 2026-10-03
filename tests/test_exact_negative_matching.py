import copy

from scripts.audit_exact_negative_matching import exact_arm, match_graph


def test_global_matching_finds_solution_a_greedy_choice_can_miss():
    a, b = ("A", "B"), ("A", "C")
    chosen, count = match_graph([{a, b}, {a}])
    assert chosen == [b, a]
    assert count == 2


def test_nonzero_support_is_not_full_unique_edge_feasibility():
    edge = ("A", "B")
    chosen, count = match_graph([{edge}, {edge}])
    assert sum(v is not None for v in chosen) == 1
    assert count == 1
    chosen, count = match_graph([set(), set()])
    assert chosen == [None, None] and count == 0


def rows():
    groups = {"G0": "P0", "G1": "P1"}
    output = []
    for i in (0, 1):
        p = dict(
            pair_id=f"p{i}",
            label=1,
            split="train",
            face_a=f"a{i}",
            face_b=f"b{i}",
            identity_group_a=f"G{i}",
            identity_group_b=f"G{i}",
            age_a=10,
            age_b=40,
            age_gap=30,
        )
        n = {
            **p,
            "pair_id": f"n{i}",
            "label": 0,
            "face_b": f"b{1 - i}",
            "identity_group_b": f"G{1 - i}",
            "matched_target_pair_id": p["pair_id"],
        }
        output.extend([p, n])
    return output, groups


def test_exact_arm_preserves_positives_heldout_and_pool():
    source, groups = rows()
    source.append(dict(pair_id="heldout", split="test", label=1, unrelated="kept"))
    before = copy.deepcopy(source)
    forbidden = {
        tuple(sorted((p["face_a"], p["face_b"])))
        for p in source
        if p["split"] == "train" and p["label"] == 1
    }
    report, assignment, candidate = exact_arm(source, groups, arm="CROSS", forbidden=forbidden)
    assert source == before
    assert report["full_fixed_positive_protocol_feasible"] is True
    assert report["positive_images"] == 4 and report["positive_recorded_people"] == 2
    assert [r for r in candidate if r["label"] == 1] == [r for r in source if r["label"] == 1]
    assert all(r["matched"] for r in assignment)
    assert all(
        v["empirical_weighted_total_variation"] == 0 for v in report["uniform_age_balance"].values()
    )


def test_forbidden_known_genuine_edges_can_prevent_full_matching():
    source, groups = rows()
    report, _, candidate = exact_arm(
        source, groups, arm="CROSS", forbidden={tuple(sorted(("a0", "b1")))}
    )
    assert report["maximum_matched_targets"] == 1
    assert report["targets_without_exact_candidate"] == 1
    assert candidate is None
