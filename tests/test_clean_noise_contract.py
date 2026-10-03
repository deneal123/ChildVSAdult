from copy import deepcopy

import pytest

from age_gap.common.schemas import Pair
from scripts.build_partial_noise_control import build_control
from scripts.validate_clean_noise_control import validate_control


def fixture():
    groups = {name: name for name in "pqrsvt"}
    rows = []
    for i, person in enumerate("pqrs"):
        age = 40 if i < 3 else 41
        rows.append(dict(pair_id=f"p{i}", split="train", label=1,
                         pair_type="positive_same_post", face_a=person+"a", face_b=person+"b",
                         identity_group_a=person, identity_group_b=person,
                         age_a=10, age_b=age, age_gap=age-10))
    for i, person in enumerate("pqrs"):
        other = "pqr"[(i+1) % 3]
        rows.append(dict(pair_id=f"n{i}", split="train", label=0,
                         pair_type="negative_restricted_positive_image_impostor",
                         face_a=person+"a", face_b=other+"b",
                         identity_group_a=person, identity_group_b=other,
                         age_a=10, age_b=40, age_gap=30))
    for split, person in (("val", "v"), ("test", "t")):
        rows.append(dict(pair_id=split, split=split, label=1, pair_type="positive_same_post",
                         face_a=person+"a", face_b=person+"b", identity_group_a=person, identity_group_b=person))
    noisy, _ = build_control(rows, groups)
    return rows, noisy, groups


def test_shared_person_contract_and_intentional_training_labels():
    clean, noisy, groups = fixture()
    original = deepcopy((clean, noisy, groups))
    audit = validate_control(clean, noisy, groups)
    assert audit["all_train_images"] == 8
    assert audit["recorded_train_people"] == 4
    assert audit["noisy_positive_pairs"] == 3
    assert audit["retained_genuine_positive_pairs"] == 1
    assert audit["noise_fraction_among_positive_labels"] == .75
    assert audit["noise_fraction_among_all_train_labels"] == .375
    assert audit["training_completed"] is False
    assert audit["publication_ready"] is False
    assert (clean, noisy, groups) == original
    # Pair drops intervention metadata; loader sees intentional label=1 unchanged.
    wrong = next(r for r in noisy[:4] if r["synthetic_label_noise"])
    pair = Pair.from_dict(wrong)
    assert pair.label == 1
    assert groups[pair.identity_group_a] != groups[pair.identity_group_b]
    assert not hasattr(pair, "synthetic_label_noise")


@pytest.mark.parametrize("attack", ["flag", "relabel", "ages", "link", "fixed", "metadata",
                                   "duplicate", "multiplicity", "mapping", "ageconflict", "cleanlabel",
                                   "self", "boollabel", "heldout"])
def test_refuses_contract_violations(attack):
    clean, noisy, groups = fixture()
    if attack == "flag":
        noisy[0]["synthetic_label_noise"] = False
    elif attack == "relabel":
        noisy[0]["label"] = 0
    elif attack == "ages":
        noisy[0]["age_a"] = 11
        noisy[0]["age_gap"] = noisy[0]["age_b"] - 11
    elif attack == "link":
        noisy[0]["noise_control_original_pair_id"] = "wrong"
    elif attack == "fixed":
        noisy[-1]["extra"] = "changed"
    elif attack == "metadata":
        noisy[0].pop("synthetic_label_noise")
    elif attack == "duplicate":
        noisy[0]["pair_id"] = noisy[1]["pair_id"]
    elif attack == "multiplicity":
        noisy[0]["face_b"] = "new"
    elif attack == "mapping":
        groups.pop("p")
    elif attack == "ageconflict":
        noisy[4]["age_b"] = 99
        noisy[4]["age_gap"] = 89
        clean[4].update(age_b=99, age_gap=89)
    elif attack == "cleanlabel":
        clean[0]["identity_group_b"] = "q"
    elif attack == "self":
        noisy[0]["face_b"] = noisy[0]["face_a"]
    elif attack == "boollabel":
        noisy[0]["label"] = True
    elif attack == "heldout":
        clean[-1]["identity_group_a"] = "p"
        noisy[-1]["identity_group_a"] = "p"
    with pytest.raises(ValueError):
        validate_control(clean, noisy, groups)


def test_known_genuine_pair_never_injected_as_noisy_positive():
    clean, noisy, groups = fixture()
    row = noisy[0]
    edge = tuple(sorted((row["face_a"], row["face_b"])))
    with pytest.raises(ValueError, match="known genuine"):
        validate_control(clean, noisy, groups, known_genuine_edges={edge})


def test_no_zero_noise_control_presented_as_partial():
    clean, _, groups = fixture()
    known = {tuple(sorted((a["face_a"], b["face_b"]))) for a in clean[:4] for b in clean[:4]}
    noisy, _ = build_control(clean, groups, known_positive_pairs=known)
    with pytest.raises(ValueError, match="partial"):
        validate_control(clean, noisy, groups)


def test_noise_metadata_never_changes_validation_labels(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    from age_gap.common.io import write_jsonl
    from age_gap.training import finetune as module

    clean, noisy, groups = fixture()
    validate_control(clean, noisy, groups)
    for face in {r[f"face_{s}"] for r in clean for s in ("a", "b")}:
        assert cv2.imwrite(str(tmp_path / f"{face}.jpg"), np.zeros((12, 12, 3), np.uint8))
    monkeypatch.setattr(module, "_crop_path", lambda face, crops_dir="faces": tmp_path / f"{face}.jpg")
    clean_path, noise_path = tmp_path / "clean.jsonl", tmp_path / "noise.jsonl"
    write_jsonl(clean_path, clean)
    write_jsonl(noise_path, noisy)
    clean_val = module.ImagePairDataset("val", pairs_file=str(clean_path))
    noise_val = module.ImagePairDataset("val", pairs_file=str(noise_path))
    train = module.ImagePairDataset("train", pairs_file=str(noise_path), gap_weight=0.0)
    assert clean_val.labels == noise_val.labels == [1]
    assert len(clean_val) == len(noise_val) == 1
    assert len(train) == 8
    assert train.labels == [1] * 4 + [0] * 4
