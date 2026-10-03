import copy
import json
import math

import pytest

from scripts import audit_matched_arm_balance as audit


def fixture():
    groups = {"L1": "LP1", "L2": "LP2", "C1": "CP1", "C2": "CP2", "V1": "VP", "T1": "TP"}
    heldout = [dict(pair_id=f"{split}held", split=split, label=1, face_a=f"{split}a", face_b=f"{split}b",
                    identity_group_a=group, identity_group_b=group) for split, group in (("val", "V1"), ("test", "T1"))]
    arms = {}
    quality = {}
    for name, prefix, ages in (("LOW", "L", (10, 11)), ("CROSS", "C", (10, 40))):
        rows = []
        for i in (1, 2):
            rows.append(dict(pair_id=f"{prefix}{i}p", split="train", label=1,
                face_a=f"{prefix}{i}a", face_b=f"{prefix}{i}b", identity_group_a=f"{prefix}{i}",
                identity_group_b=f"{prefix}{i}", age_a=ages[0], age_b=ages[1], age_gap=ages[1] - ages[0]))
        for i, j in ((1, 2), (2, 1)):
            rows.append(dict(pair_id=f"{prefix}{i}n", split="train", label=0,
                face_a=f"{prefix}{i}a", face_b=f"{prefix}{j}b", identity_group_a=f"{prefix}{i}",
                identity_group_b=f"{prefix}{j}", age_a=ages[0], age_b=ages[1], age_gap=ages[1] - ages[0]))
        arms[name] = rows + copy.deepcopy(heldout)
        for row in rows:
            for side in ("a", "b"):
                face = row[f"face_{side}"]
                quality[face] = dict(is_usable=True, photo_id=f"photo{face}", face_width=100 if name == "LOW" else 200,
                                     pose_roll_rad=-0.25, blur_var=None)
    return arms, groups, quality


def test_unique_image_exposure_and_label_vs_arm_age_are_distinct():
    arms, groups, quality = fixture()
    before = copy.deepcopy((arms, groups, quality))
    result = audit.balance(arms, groups, quality)
    assert (arms, groups, quality) == before
    assert result["arms"]["LOW"]["unique_images"] == 4
    assert result["arms"]["LOW"]["label_age_auc"]["age_gap"]["auc_positive_higher"] == 0.5
    assert result["positive_pair_endpoint_view"]["age_gap"]["arm_auc_cross_higher"] == 1
    assert result["unique_image_view"]["caption_age"]["LOW"]["n_total"] == 4
    assert result["all_train_endpoint_exposure_view"]["caption_age"]["LOW"]["n_total"] == 8
    assert result["unique_image_view"]["blur_var"]["LOW"]["n_missing"] == 4
    assert result["unique_image_view"]["pose_roll_rad_abs"]["LOW"]["mean"] == 0.25
    assert result["unique_image_view"]["face_width"]["smd_status"] == "different constants; zero pooled SD"
    assert result["unique_image_view"]["face_width"]["smd_cross_minus_low"] is None
    assert result["publication_ready"] is False and result["causal_source_superiority"] is False
    assert "cannot preserve" in result["age_balance_constraint"]
    json.dumps(result, allow_nan=False)


def test_constant_and_missing_comparisons_are_not_fake_finite_smds():
    equal = audit.compare([1, 1], [1, 1])
    assert equal["smd_cross_minus_low"] == 0 and equal["arm_auc_cross_higher"] == 0.5
    assert equal["ecdf_max_distance"] == 0
    assert audit.compare([1, 1], [2, 2])["smd_cross_minus_low"] is None
    missing = audit.compare([None, None], [2])
    assert missing["LOW"]["n_missing"] == 2 and missing["arm_auc_cross_higher"] is None
    assert missing["observed_range_overlap"] is None


def test_fractional_range_overlap_and_mean_sign():
    result = audit.compare([0, 1, 2], [1, 2, 3])
    assert result["cross_minus_low_mean"] == 1
    assert result["observed_range_overlap"] == [1.0, 2.0]
    assert result["LOW_fraction_within_CROSS_observed_range"] == pytest.approx(2 / 3)
    assert result["CROSS_fraction_within_LOW_observed_range"] == pytest.approx(2 / 3)
    assert result["ecdf_max_distance"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("bad", [True, "1", float("nan"), float("inf"), -1])
def test_invalid_numeric_metadata_refused(bad):
    with pytest.raises(ValueError):
        audit.value({"blur_var": bad}, "blur_var")


def test_missing_sidecar_rows_are_unknown_not_zero():
    arms, groups, quality = fixture()
    quality.pop("L1a")
    result = audit.balance(arms, groups, quality)
    assert result["arms"]["LOW"]["images_missing_quality_rows"] == 1
    assert result["unique_image_view"]["face_width"]["LOW"]["n_observed"] == 3
    assert result["arms"]["LOW"]["label_quality_pair_min_auc"]["face_width"]["n_missing"] == 2


@pytest.mark.parametrize("change", ["duplicate", "label", "gap", "arm", "age_conflict", "person_conflict", "usable", "heldout", "budget"])
def test_protocol_or_metadata_conflicts_refused(change):
    arms, groups, quality = fixture()
    if change == "duplicate":
        arms["LOW"][1]["pair_id"] = arms["LOW"][0]["pair_id"]
    elif change == "label":
        arms["LOW"][0]["label"] = True
    elif change == "gap":
        arms["LOW"][0]["age_gap"] = 2
    elif change == "arm":
        arms["LOW"][0]["age_b"] = 40
        arms["LOW"][0]["age_gap"] = 30
    elif change == "age_conflict":
        arms["LOW"][2]["age_a"] = 9
        arms["LOW"][2]["age_gap"] = 2
    elif change == "person_conflict":
        arms["LOW"][2]["identity_group_a"] = "L2"
    elif change == "usable":
        quality["L1a"]["is_usable"] = False
    elif change == "heldout":
        arms["LOW"][-1]["pair_id"] = "different"
    else:
        arms["LOW"].pop(2)
    with pytest.raises(ValueError):
        audit.balance(arms, groups, quality)


def test_no_identifier_or_raw_quality_rows_exported():
    arms, groups, quality = fixture()
    text = json.dumps(audit.balance(arms, groups, quality))
    for identifier in ("L1a", "L1b", "C1a", "LP1", "CP1", "photoL1a"):
        assert identifier not in text
    assert "p-value" in text and "unverified" in text
    assert math.isfinite(audit.compare([1, 2], [2, 3])["smd_cross_minus_low"])
