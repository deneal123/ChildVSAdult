import copy
import json

import pytest

from scripts.audit_repaired_nuisance import refit, selected_quality, verified_prerequisite


@pytest.mark.parametrize(
    "kind,complete",
    [
        ("exact-supported-singleton-profile-repair", True),
        ("exact-supported-singleton-profile-repair-v2-row-regime", False),
    ],
)
def test_historical_or_incomplete_preparation_cannot_enter_refit(tmp_path, kind, complete):
    path = tmp_path / "summary.manifest.json"
    path.write_text(
        json.dumps({"experiment": kind, "metrics": {"complete": complete}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="completed exact-supported"):
        verified_prerequisite(path, tmp_path)


def fixture():
    groups = {"V": "VP", "T": "TP"}
    heldout = [
        dict(
            pair_id=s,
            split=s,
            label=1,
            face_a=s + "a",
            face_b=s + "b",
            identity_group_a=g,
            identity_group_b=g,
        )
        for s, g in (("val", "V"), ("test", "T"))
    ]
    arms, quality = {}, {}
    for arm, older in (("LOW", 11), ("CROSS", 40)):
        rows = []
        for i in range(5):
            key = f"{arm}{i}"
            other = f"{arm}{(i + 1) % 5}"
            groups[key] = key
            p = dict(
                pair_id=key + "p",
                split="train",
                label=1,
                face_a=key + "a",
                face_b=key + "b",
                identity_group_a=key,
                identity_group_b=key,
                age_a=10,
                age_b=older,
                age_gap=older - 10,
            )
            n = {
                **p,
                "pair_id": key + "n",
                "label": 0,
                "face_b": other + "b",
                "identity_group_b": other,
                "matched_target_pair_id": p["pair_id"],
            }
            rows.extend([p, n])
            for side in ("a", "b"):
                quality[key + side] = dict(
                    face_id=key + side,
                    is_usable=True,
                    face_width=100 + i * 10,
                    blur_var=40 + i * 15,
                    det_score=0.9,
                )
        arms[arm] = rows + copy.deepcopy(heldout)
    return arms, groups, quality


def test_fresh_fit_and_full_ledger_preserve_exact_class_age_balance():
    args = fixture()
    before = copy.deepcopy(args)
    report, weights, ledger = refit(*args)
    assert args == before
    assert report["execution_complete"] and report["all_coupled_age_views_exact"]
    assert not report["training_weights_ready"] and not report["publication_ready"]
    assert len(weights) == 30 and len(ledger) == 60
    for reports in report["coupled"]["views"].values():
        for row in reports.values():
            assert row["positive_pairs"] == row["negative_pairs"] == 5
            assert row["class_loss_mass"]["positive"] == pytest.approx(5)
            assert row["class_loss_mass"]["negative"] == pytest.approx(5)
            for age in row["coupled_candidate_age_balance"].values():
                assert age["empirical_weighted_total_variation"] == 0
                assert age["equal_prior_weighted_in_sample_oracle_accuracy"] == 0.5
    assert refit(*args) == (report, weights, ledger)


def test_approximate_endpoint_age_rejected_before_fit():
    args = fixture()
    args[0]["LOW"][1]["age_b"] += 1
    args[0]["LOW"][1]["age_gap"] += 1
    args[0]["LOW"][1]["face_b"] = "new-face-with-age12"
    args[2]["new-face-with-age12"] = {"is_usable": True}
    with pytest.raises(ValueError, match="exact endpoint"):
        refit(*args)


def test_missing_quality_rejected_before_fit():
    args = fixture()
    args[2].pop("LOW0a")
    with pytest.raises(ValueError, match="unique quality"):
        refit(*args)


def test_selected_quality_does_not_silently_overwrite_even_equal_rows():
    row = dict(face_id="chosen", is_usable=True)
    with pytest.raises(ValueError, match="duplicate"):
        selected_quality([row, copy.deepcopy(row)], {"chosen"})
    assert selected_quality(
        [row, dict(face_id="irrelevant"), dict(face_id="irrelevant")], {"chosen"}
    ) == {"chosen": row}
    with pytest.raises(ValueError, match="missing"):
        selected_quality([], {"chosen"})


def test_training_identity_relation_cannot_be_replaced_by_weight_fitting():
    args = fixture()
    args[1]["LOW1"] = "LOW0"
    with pytest.raises(ValueError, match="relation contradicts"):
        refit(*args)
