import copy

import pytest

from scripts.repair_exact_supported_arms import repair, unique_quality


def test_duplicate_quality_keys_are_excluded_not_first_or_last_selected():
    rows = [
        dict(face_id="a", is_usable=True),
        dict(face_id="b", is_usable=True),
        dict(face_id="a", is_usable=True),
        dict(face_id="b", is_usable=False),
        dict(face_id="c", is_usable=True),
    ]
    quality, audit = unique_quality(rows)
    assert quality == {"c": rows[-1]}
    assert audit["duplicate_rows"] == audit["excluded_duplicate_faces"] == 2
    assert audit["conflicting_rows"] == 1


def fixture():
    groups, arms, canonical, quality, unsupported = {}, {}, [], {}, {}
    for arm, older in (("LOW", 11), ("CROSS", 40)):
        positives = []
        for i in range(4):
            name = f"{arm}{i}"
            groups[name] = name
            row = dict(
                pair_id=name,
                split="train",
                label=1,
                face_a=name + "a",
                face_b=name + "b",
                identity_group_a=name,
                identity_group_b=name,
                age_a=10,
                age_b=older + (i == 2),
                age_gap=older + (i == 2) - 10,
            )
            if i < 3:
                positives.append(row)
            canonical.append(row)
            for side in ("a", "b"):
                quality[row[f"face_{side}"]] = {"is_usable": True}
        arms[arm] = positives
        unsupported[arm] = [f"{arm}2"]
    groups["HELD"] = "HELD"
    held = dict(
        pair_id="held",
        split="test",
        label=1,
        face_a="helda",
        face_b="heldb",
        identity_group_a="HELD",
        identity_group_b="HELD",
        age_a=10,
        age_b=40,
        age_gap=30,
    )
    for rows in arms.values():
        rows.append(copy.deepcopy(held))
    return arms, groups, canonical, quality, unsupported


def test_repair_preserves_budget_heldout_retained_pairs_and_exact_class_ages():
    args = fixture()
    before = copy.deepcopy(args)
    report, arms, replacements = repair(*args)
    assert args == before
    assert report["complete"] and not report["training_ready"] and not report["publication_ready"]
    assert len(replacements) == 2
    for arm, rows in arms.items():
        assert rows[-1] == before[0][arm][-1]
        assert rows[:2] == before[0][arm][:2]
        assert report["arms"][arm]["source_profile"] == report["arms"][arm]["repaired_profile"]
        assert report["arms"][arm]["source_profile"] == dict(
            positive_pairs=3, images=6, recorded_people=3
        )
        assert rows[2]["pair_id"] == f"{arm}3"
        positives = {r["pair_id"]: r for r in rows if r["split"] == "train" and r["label"] == 1}
        negatives = [r for r in rows if r["label"] == 0]
        assert len(negatives) == len(positives)
        forbidden = {tuple(sorted((r["face_a"], r["face_b"]))) for r in args[2]}
        for n in negatives:
            p = positives[n["matched_target_pair_id"]]
            assert (p["age_a"], p["age_b"]) == (n["age_a"], n["age_b"])
            assert tuple(sorted((n["face_a"], n["face_b"]))) not in forbidden
            assert args[1][n["identity_group_a"]] != args[1][n["identity_group_b"]]
        assert all(
            v["empirical_weighted_total_variation"] == 0
            for v in report["arms"][arm]["global_exact_matching"]["uniform_age_balance"].values()
        )
    assert repair(*args) == (report, arms, replacements)


@pytest.mark.parametrize(
    "cause",
    [
        "missing_quality",
        "unusable",
        "ambiguous_age",
        "dual_regime",
        "heldout_person",
        "heldout_image",
    ],
)
def test_invalid_candidate_is_not_silently_accepted(cause):
    arms, groups, canonical, quality, unsupported = fixture()
    candidate = next(r for r in canonical if r["pair_id"] == "LOW3")
    if cause == "missing_quality":
        quality.pop(candidate["face_a"])
    elif cause == "unusable":
        quality[candidate["face_a"]]["is_usable"] = False
    elif cause == "ambiguous_age":
        canonical.append({**candidate, "pair_id": "ambiguous", "age_a": 9, "age_gap": 2})
    elif cause == "dual_regime":
        canonical.append(
            {
                **candidate,
                "pair_id": "other_regime",
                "face_a": "othera",
                "face_b": "otherb",
                "age_b": 40,
                "age_gap": 30,
            }
        )
    elif cause == "heldout_person":
        groups["LOW3"] = "HELD"
    else:
        candidate["face_a"] = "helda"
    result, prepared, changes = repair(arms, groups, canonical, quality, unsupported)
    assert not result["complete"] and not result["publication_ready"]
    assert prepared is None and changes == []


def test_non_singleton_profile_requires_a_different_explicit_protocol():
    arms, groups, canonical, quality, unsupported = fixture()
    row = {**arms["LOW"][2], "pair_id": "extra", "face_a": "extraa"}
    arms["LOW"].insert(3, row)
    arms["CROSS"].insert(3, {**arms["CROSS"][2], "pair_id": "extrac", "face_a": "extraca"})
    with pytest.raises(ValueError, match="richer profiles"):
        repair(arms, groups, canonical, quality, unsupported)


@pytest.mark.parametrize("seed,limit", [(True, 64), (42, 0), (42, 1.5)])
def test_parameters_fail_closed(seed, limit):
    with pytest.raises(ValueError, match="integer"):
        repair(*fixture(), seed=seed, max_attempts=limit)


def test_existing_train_heldout_overlap_is_rejected():
    args = fixture()
    args[1]["HELD"] = "LOW0"
    with pytest.raises(ValueError, match="overlap"):
        repair(*args)


def test_duplicate_canonical_ids_rejected():
    args = fixture()
    args[2].append(copy.deepcopy(args[2][0]))
    with pytest.raises(ValueError, match="duplicate canonical"):
        repair(*args)


def test_person_regime_does_not_license_out_of_regime_pair():
    args = fixture()
    candidate = next(r for r in args[2] if r["pair_id"] == "LOW3")
    candidate.update(age_a=11, age_b=11, age_gap=0)
    args[2].append(
        {
            **candidate,
            "pair_id": "valid-regime-evidence",
            "face_a": "extra-age10",
            "face_b": "extra-age11",
            "age_a": 10,
            "age_gap": 1,
        }
    )
    # A legitimate LOW observation for the same person does not make gap0 eligible;
    # its own extra images lack quality records and cannot replace the singleton.
    result, prepared, changes = repair(*args)
    assert not result["complete"] and prepared is None and changes == []
