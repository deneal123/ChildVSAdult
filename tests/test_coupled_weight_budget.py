import copy
import json

import numpy as np
import pytest

from scripts import audit_coupled_weight_budget as audit
from scripts.audit_nuisance_overlap import VIEWS


def fixture():
    arms, groups, diagnostic = {}, {}, []
    for arm, prefix in (("LOW", "L"), ("CROSS", "C")):
        rows = []
        for i in range(2):
            groups[f"{prefix}{i}"] = f"{prefix}P{i}"
            ages = (10, 11) if arm == "LOW" else (10, 40)
            positive = dict(
                pair_id=f"{prefix}{i}p",
                face_a=f"{prefix}{i}a",
                face_b=f"{prefix}{i}b",
                label=1,
                split="train",
                age_a=ages[0],
                age_b=ages[1],
                age_gap=ages[1] - ages[0],
                identity_group_a=f"{prefix}{i}",
                identity_group_b=f"{prefix}{i}",
            )
            negative = {
                **positive,
                "pair_id": f"{prefix}{i}n",
                "label": 0,
                "face_b": f"{prefix}{1 - i}b",
                "identity_group_b": f"{prefix}{1 - i}",
                "matched_target_pair_id": positive["pair_id"],
            }
            rows.extend([positive, negative])
            for view in VIEWS:
                raw = 0.25 if i == 0 else 0.75
                diagnostic.append(
                    dict(
                        view=view,
                        pair_id=positive["pair_id"],
                        recorded_person=f"{prefix}P{i}",
                        normalized_diagnostic_weight=2 * raw,
                        raw_overlap_weight=raw,
                        p_cross_oof=raw if arm == "LOW" else 1 - raw,
                    )
                )
        arms[arm] = rows
    return arms, groups, diagnostic


def test_coupling_preserves_class_mass_and_exact_age_balance():
    arms, groups, weights = fixture()
    before = copy.deepcopy((arms, groups, weights))
    result, private = audit.audit(arms, groups, weights)
    assert (arms, groups, weights) == before
    assert len(private) == 24
    for view in VIEWS:
        for row in result["views"][view].values():
            assert row["class_loss_mass"] == {"positive": 2, "negative": 2}
            assert row["coupled_candidate_endpoint_mass"]["images"]["distinct"] == 4
            assert (
                row["coupled_candidate_endpoint_mass"]["images"][
                    "total_loss_weighted_endpoint_mass"
                ]
                == 8
            )
            for feature in row["coupled_candidate_age_balance"].values():
                assert feature["empirical_weighted_total_variation"] == 0
                assert feature["equal_prior_weighted_in_sample_oracle_accuracy"] == 0.5
    assert result["training_weights_ready"] is False
    assert result["recognition_outcomes_used"] is False
    text = json.dumps(result, allow_nan=False)
    for identifier in ("L0a", "LP0", "L0p"):
        assert identifier not in text


def test_weighting_can_create_class_age_shortcut_from_unweighted_balance():
    arms, groups, weights = fixture()
    # Genuine B ages11/12; impostors swap them, keeping valid per-image ages.
    for row in arms["LOW"]:
        if row["pair_id"] in {"L1p", "L0n"}:
            row["age_b"] += 1
            row["age_gap"] += 1
    result, _ = audit.audit(arms, groups, weights)
    low = result["views"]["quality_only"]["LOW"]
    assert low["uniform_age_balance"]["age_b"]["empirical_weighted_total_variation"] == 0
    assert (
        low["coupled_candidate_age_balance"]["age_b"]["empirical_weighted_total_variation"] == 0.5
    )
    assert low["coupled_candidate_age_balance"]["age_a"]["empirical_weighted_total_variation"] == 0


@pytest.mark.parametrize(
    "kind", ["duplicate_pair", "target", "anchor", "age_error", "self", "person", "label"]
)
def test_block_binding_failure_refused(kind):
    arms, groups, _ = fixture()
    row = arms["LOW"][1]
    if kind == "duplicate_pair":
        row["pair_id"] = arms["LOW"][0]["pair_id"]
    elif kind == "target":
        row["matched_target_pair_id"] = "unknown"
    elif kind == "anchor":
        row["face_a"] = "different"
    elif kind == "age_error":
        row["age_b"] += 2
        row["age_gap"] += 2
    elif kind == "self":
        row["face_b"] = row["face_a"]
    elif kind == "person":
        row["identity_group_b"] = row["identity_group_a"]
    else:
        row["label"] = False
    with pytest.raises(ValueError):
        audit.blocks(arms["LOW"], groups)


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "person", "raw", "normalized", "nan", "negative"]
)
def test_diagnostic_weight_binding_failure_refused(kind):
    arms, groups, weights = fixture()
    if kind == "missing":
        weights.pop()
    elif kind == "duplicate":
        weights.append(weights[0])
    elif kind == "person":
        weights[0]["recorded_person"] = "wrong"
    elif kind == "raw":
        weights[0]["raw_overlap_weight"] = 0.3
    elif kind == "normalized":
        weights[0]["normalized_diagnostic_weight"] += 1
    elif kind == "nan":
        weights[0]["normalized_diagnostic_weight"] = np.nan
    else:
        weights[0]["normalized_diagnostic_weight"] = -1
    with pytest.raises(ValueError):
        audit.audit(arms, groups, weights)


def test_cross_person_face_alias_is_not_allowed():
    arms, groups, _ = fixture()
    pairs = audit.blocks(arms["LOW"], groups)
    pairs[1][0]["face_b"] = pairs[0][0]["face_b"]
    with pytest.raises(ValueError, match="face-person"):
        audit.exposure_mass(pairs, [1, 1], groups)
