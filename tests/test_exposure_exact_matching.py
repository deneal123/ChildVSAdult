from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.audit_exposure_exact_matching as module


def test_full_counts_and_unique_edges():
    chosen, report = module.constrained_assignment(
        ["a", "c"], [{"b", "d"}, {"b", "d"}], Counter(b=1, d=1)
    )
    assert report["status"] == "feasible"
    assert Counter(chosen) == Counter(b=1, d=1)


def test_structural_empty_candidate():
    chosen, report = module.constrained_assignment(["a"], [set()], Counter(b=1))
    assert chosen is None
    assert report["status"] == "structural_infeasible"
    assert not report["solver_called"]


def test_capacity_hall_infeasible():
    chosen, report = module.constrained_assignment(["a", "c"], [{"b"}, {"b"}], Counter(b=1, d=1))
    assert chosen is None
    assert report["status"] == "solver_infeasible"


def test_reverse_orientation_unique_edge_constraint():
    chosen, report = module.constrained_assignment(["a", "b"], [{"b"}, {"a"}], Counter(a=1, b=1))
    assert chosen is None
    assert report["status"] == "solver_infeasible"


def test_limit_not_infeasibility(monkeypatch):
    monkeypatch.setattr(
        module, "milp", lambda *a, **k: SimpleNamespace(status=1, message="time limit", x=None)
    )
    chosen, report = module.constrained_assignment(["a"], [{"b"}], Counter(b=1))
    assert chosen is None
    assert report["status"] == "limit_without_incumbent"


def test_checked_incumbent_at_limit(monkeypatch):
    monkeypatch.setattr(
        module,
        "milp",
        lambda *a, **k: SimpleNamespace(status=1, message="limit", x=np.array([1.0])),
    )
    chosen, report = module.constrained_assignment(["a"], [{"b"}], Counter(b=1))
    assert chosen == ["b"]
    assert report["status"] == "feasible"


@pytest.mark.parametrize("x", [[0.5], [0.0], [np.nan], [2.0]])
def test_reject_invalid_incumbent(monkeypatch, x):
    monkeypatch.setattr(
        module, "milp", lambda *a, **k: SimpleNamespace(status=0, message="ok", x=np.array(x))
    )
    with pytest.raises(ValueError):
        module.constrained_assignment(["a"], [{"b"}], Counter(b=1))


def test_bad_capacity_and_self_candidate():
    with pytest.raises(ValueError):
        module.constrained_assignment(["a"], [{"b"}], Counter(b=2))
    with pytest.raises(ValueError):
        module.constrained_assignment(["a"], [{"a"}], Counter(a=1))


def test_arm_preserves_positive_heldout_and_class_exposure():
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
    old_negatives = [
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
    heldout = dict(split="test", sentinel="unchanged")
    rows = positives + old_negatives + [heldout]
    output, report = module.exposure_arm(
        rows, groups, arm="LOW", forbidden={("a", "b"), ("c", "d")}
    )
    assert output[:2] == positives
    assert output[-1] == heldout
    assert report["class_image_exposure_exact"]
    assert report["images"] == 4
    assert report["recorded_people"] == 2
    assert report["max_image_whole_pass_count"] == 2
    assert all(v["empirical_weighted_total_variation"] == 0 for v in report["age_balance"].values())
    assert module.exposure_arm(rows, groups, arm="LOW", forbidden={("a", "d")})[0] is None
