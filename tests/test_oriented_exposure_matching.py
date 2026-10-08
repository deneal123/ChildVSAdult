from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.audit_oriented_exposure_matching as m


def test_orientation_enables_balanced_assignment():
    pairs = [("a", "b"), ("c", "d")]
    options = [[set(), {"d"}], [{"a"}, set()]]
    chosen, report = m.joint_assignment(pairs, options)
    assert chosen == [(0, 1, "d"), (1, 0, "a")]
    assert report["flipped_positives"] == 1
    assert report["minimum_flips_proven"]
    assert Counter(b for _, _, b in chosen) == Counter(pairs[i][1 - o] for i, o, _ in chosen)


def test_minimum_flips_preserves_original_if_feasible():
    chosen, report = m.joint_assignment([("a", "b"), ("c", "d")], [[{"d"}, {"c"}], [{"b"}, {"a"}]])
    assert report["flipped_positives"] == 0
    assert chosen == [(0, 0, "d"), (1, 0, "b")]


def test_infeasible_conservation_not_timeout():
    chosen, report = m.joint_assignment([("a", "b"), ("c", "d")], [[{"c"}, set()], [{"a"}, set()]])
    assert chosen is None
    assert report["status"] == "solver_infeasible"


def test_structural_infeasible():
    assert (
        m.joint_assignment([("a", "b")], [[set(), set()]])[1]["status"] == "structural_infeasible"
    )


def test_limit_without_incumbent_not_impossibility(monkeypatch):
    monkeypatch.setattr(
        m, "milp", lambda *a, **k: SimpleNamespace(status=1, message="limit", x=None)
    )
    assert (
        m.joint_assignment([("a", "b"), ("c", "d")], [[{"d"}, set()], [{"b"}, set()]])[1]["status"]
        == "limit_without_incumbent"
    )


def test_valid_limit_incumbent_not_optimality(monkeypatch):
    monkeypatch.setattr(
        m, "milp", lambda *a, **k: SimpleNamespace(status=1, message="limit", x=np.ones(2))
    )
    _, report = m.joint_assignment([("a", "b"), ("c", "d")], [[{"d"}, set()], [{"b"}, set()]])
    assert report["status"] == "feasible"
    assert not report["minimum_flips_proven"]


def test_invalid_incumbent_rejected(monkeypatch):
    monkeypatch.setattr(
        m, "milp", lambda *a, **k: SimpleNamespace(status=0, message="ok", x=np.array([1.0, 0.0]))
    )
    with pytest.raises(ValueError):
        m.joint_assignment([("a", "b"), ("c", "d")], [[{"d"}, set()], [{"b"}, set()]])


def test_target_image_not_impostor():
    with pytest.raises(ValueError):
        m.joint_assignment([("a", "b")], [[{"b"}, set()]])
