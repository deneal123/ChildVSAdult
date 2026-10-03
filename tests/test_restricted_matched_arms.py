from copy import deepcopy

import pytest

from scripts.build_restricted_matched_arms import restrict_arms


def fixture():
    groups = {f"g{i}": f"p{i}" for i in range(10)}
    def row(pid, fa, fb, ga, gb, aa, ab, label=1, split="train"):
        return dict(pair_id=pid, face_a=fa, face_b=fb, identity_group_a=ga,
                    identity_group_b=gb, age_a=aa, age_b=ab, age_gap=abs(aa-ab),
                    label=label, split=split, status="ok")
    heldout = [row("val", "v1", "v2", "g8", "g8", 1, 2, split="val"),
               row("test", "t1", "t2", "g9", "g9", 1, 2, split="test")]
    arms = {}
    for name, offset, aa, ab in (("LOW", 0, 20, 21), ("CROSS", 3, 10, 40)):
        positives = [row(f"{name}{i}", f"{name}a{i}", f"{name}b{i}",
                         f"g{i+offset}", f"g{i+offset}", aa, ab) for i in range(3)]
        negatives = [row(f"old{name}{i}", f"{name}a{i}", f"extra{name}{i}",
                         f"g{i+offset}", f"g{(i+1)%3+offset}", aa, ab, 0) for i in range(3)]
        arms[name] = [*positives, *negatives, *deepcopy(heldout)]
    return arms, groups


def test_deterministic_preserved_and_exact_full_budget():
    arms, groups = fixture()
    original = deepcopy(arms)
    out, summary = restrict_arms(arms, groups)
    assert (out, summary) == restrict_arms(arms, groups)
    assert arms == original
    for name in arms:
        assert out[name][:3] == arms[name][:3]
        assert out[name][-2:] == arms[name][-2:]
        stats = summary["arms"][name]
        assert stats["all_train_images"] == stats["positive_images"] == 6
        assert stats["negative_only_images"] == 0
        assert stats["recorded_people"] == 3
        assert stats["train_age_only_class_auc"] == dict(age_a=.5, age_b=.5, age_gap=.5)
        positives = {r["pair_id"]: r for r in out[name][:3]}
        edges = set()
        for r in out[name][3:6]:
            assert groups[r["identity_group_a"]] != groups[r["identity_group_b"]]
            assert r["face_a"] == positives[r["matched_target_pair_id"]]["face_a"]
            edges.add(tuple(sorted((r["face_a"], r["face_b"]))))
        assert len(edges) == 3
    assert summary["training_completed"] is False
    assert summary["publication_ready"] is False


@pytest.mark.parametrize("attack", ["mapping", "label", "gap", "age", "sameperson", "heldout",
                                  "duplicate", "differentheldout", "unequal", "crossarm", "unknownsplit"])
def test_refuses_invalid_inputs(attack):
    arms, groups = fixture()
    if attack == "mapping":
        groups.pop("g0")
    elif attack == "label":
        arms["LOW"][0]["label"] = True
    elif attack == "gap":
        arms["LOW"][0]["age_gap"] = 50
    elif attack == "age":
        arms["LOW"][0]["age_a"] = 20.0
    elif attack == "sameperson":
        arms["LOW"][3]["identity_group_b"] = "g0"
    elif attack == "heldout":
        arms["LOW"][-2]["identity_group_a"] = "g0"
    elif attack == "duplicate":
        arms["LOW"][1]["pair_id"] = arms["LOW"][0]["pair_id"]
    elif attack == "differentheldout":
        arms["LOW"][-1]["extra"] = "changed"
    elif attack == "unequal":
        arms["LOW"].pop(0)
        arms["LOW"].pop(2)
    elif attack == "crossarm":
        groups["g3"] = "p0"
    elif attack == "unknownsplit":
        arms["LOW"][0]["split"] = "other"
    with pytest.raises(ValueError):
        restrict_arms(arms, groups)


def test_no_pool_widening_when_restricted_age_support_absent():
    arms, groups = fixture()
    arms["LOW"][0].update(age_a=30, age_b=31, age_gap=1)
    # Old wider pool has an eligible distinct-person image; selected pool does not.
    arms["LOW"][3].update(age_b=31)
    with pytest.raises(ValueError, match="cannot construct"):
        restrict_arms(arms, groups)


def test_known_genuine_edges_are_not_relabelled_negative():
    arms, groups = fixture()
    out, _ = restrict_arms(arms, groups)
    negative = out["LOW"][3]
    edge = tuple(sorted((negative["face_a"], negative["face_b"])))
    with pytest.raises(ValueError, match="known genuine"):
        restrict_arms(arms, groups, known_positive_pairs={edge})


def test_conflicting_selected_face_age_is_not_majority_resolved():
    arms, groups = fixture()
    r = deepcopy(arms["LOW"][0])
    r.update(pair_id="new", face_b="newface", age_a=21, age_b=22)
    arms["LOW"].insert(0, r)
    n = deepcopy(arms["LOW"][4])
    n["pair_id"] = "newneg"
    arms["LOW"].append(n)
    with pytest.raises(ValueError, match="conflicting selected-face ages"):
        restrict_arms(arms, groups)
