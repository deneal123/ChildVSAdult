from collections import Counter
from copy import deepcopy

import pytest

from scripts.build_partial_noise_control import build_control, digest_rows


def fixture():
    rows, groups = [], {name: name for name in "pqrsvt"}
    for i, person in enumerate("pqrs"):
        age = 40 if i < 3 else 41
        rows.append({"pair_id": f"pos{i}", "split": "train", "label": 1,
                     "face_a": person + "a", "face_b": person + "b",
                     "identity_group_a": person, "identity_group_b": person,
                     "age_a": 10, "age_b": age, "age_gap": age - 10})
    for i, person in enumerate("pqrs"):
        other = "pqr"[(i + 1) % 3]
        rows.append({"pair_id": f"neg{i}", "split": "train", "label": 0,
                     "face_a": person + "a", "face_b": other + "b",
                     "identity_group_a": person, "identity_group_b": other,
                     "age_a": 10, "age_b": 40, "age_gap": 30})
    for split, person in (("val", "v"), ("test", "t")):
        rows.append({"pair_id": split, "split": split, "label": 1,
                     "face_a": person + "a", "face_b": person + "b",
                     "identity_group_a": person, "identity_group_b": person})
    return rows, groups


def test_deterministic_partial_preserves_all_declared_budgets_and_heldout():
    rows, groups = fixture()
    original = deepcopy(rows)
    control, summary = build_control(rows, groups)
    assert (control, summary) == build_control(rows, groups)
    assert rows == original
    assert summary["positive_distinct_person_noise_pairs"] == 3
    assert summary["retained_genuine_positive_pairs"] == 1
    assert summary["full_shuffle"] is False
    assert summary["training_evaluation_completed"] is False
    assert summary["publication_ready"] is False
    positive = control[:4]
    for before, after in zip(rows[:4], positive, strict=True):
        assert before["age_a"] == after["age_a"]
        assert before["age_b"] == after["age_b"]
        assert before["age_gap"] == after["age_gap"]
        assert before["face_a"] == after["face_a"]
        assert after["label"] == 1  # deliberate synthetic noisy supervision
        assert after["synthetic_label_noise"] == (after["identity_group_a"] != after["identity_group_b"])
    for side in ("a", "b"):
        assert Counter(r[f"face_{side}"] for r in positive) == Counter(r[f"face_{side}"] for r in rows[:4])
    assert digest_rows(control[4:]) == digest_rows(rows[4:])
    noisy_edges = {tuple(sorted((r["face_a"], r["face_b"]))) for r in positive}
    neg_edges = {tuple(sorted((r["face_a"], r["face_b"]))) for r in rows[4:8]}
    assert len(noisy_edges) == 4 and not noisy_edges & neg_edges


def test_full_mode_never_silently_accepts_singleton_strata():
    rows, groups = fixture()
    with pytest.raises(ValueError, match="full shuffle infeasible"):
        build_control(rows, groups, require_full=True)


@pytest.mark.parametrize("problem", ["mapping", "leakage", "gap", "label", "duplicate", "self", "budget"])
def test_invalid_inputs_rejected(problem):
    rows, groups = fixture()
    kwargs = {}
    if problem == "mapping":
        del groups["p"]
    elif problem == "leakage":
        rows[-1]["identity_group_a"] = "p"
    elif problem == "gap":
        rows[0]["age_gap"] = 0
    elif problem == "label":
        rows[0]["identity_group_b"] = "q"
    elif problem == "duplicate":
        rows[1]["pair_id"] = rows[0]["pair_id"]
    elif problem == "self":
        rows[0]["face_b"] = rows[0]["face_a"]
    else:
        kwargs["attempts"] = 0
    with pytest.raises(ValueError):
        build_control(rows, groups, **kwargs)


def test_different_groups_mapping_to_same_person_cannot_be_noise():
    rows, groups = fixture()
    groups.update({"q": "p", "r": "p"})
    # Original fixed negatives are now same-person and invalidate the arm.
    with pytest.raises(ValueError, match="contradict"):
        build_control(rows, groups)


def test_conflicting_face_person_mapping_is_rejected():
    rows, groups = fixture()
    rows[-1]["face_a"] = rows[0]["face_a"]
    with pytest.raises(ValueError, match="conflicting recorded"):
        build_control(rows, groups)


def test_known_canonical_positive_edges_are_not_injected_as_noise():
    rows, groups = fixture()
    known = {tuple(sorted((left["face_a"], right["face_b"]))) for left in rows[:4] for right in rows[:4]}
    control, summary = build_control(rows, groups, known_positive_pairs=known)
    assert summary["positive_distinct_person_noise_pairs"] == 0
    assert summary["retained_genuine_positive_pairs"] == 4
    assert all(not row["synthetic_label_noise"] for row in control[:4])


def test_finite_attempt_exhaustion_is_not_false_infeasibility(monkeypatch):
    from scripts import build_partial_noise_control as module

    rows, groups = fixture()
    # Simulate duplicate assignments: terminal cap-hit, never a relaxed-age result.
    monkeypatch.setattr(module, "linear_sum_assignment", lambda cost: ([0, 0, 2], [0, 0, 2]))
    with pytest.raises(ValueError, match="finite assignment attempts exhausted"):
        build_control(rows, groups, attempts=2)
