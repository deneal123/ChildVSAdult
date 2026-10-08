from scripts.audit_sota_face_inputs import audit


def group(identifier="g", faces=None, labels=None):
    return dict(identity_group_id=identifier, faces=faces or ["a", "b"], age_labels=labels or [])


def test_clean_recorded_contract_does_not_claim_human_or_crop_clearance():
    result = audit(
        [group(labels=[dict(face_id="a", age=0)])],
        [dict(identity_group_id="g", split="train")],
        [dict(identity_group_id="g", person_id="p")],
    )
    assert result["input_contract_clean"]
    assert not result["human_identity_audit_complete"] and not result["crops_verified"]
    assert not result["training_complete"] and not result["publication_ready"]


def test_conflicting_ages_and_orphan_label_not_silently_overwritten():
    result = audit(
        [
            group(
                labels=[
                    dict(face_id="a", age=0),
                    dict(face_id="a", age=1),
                    dict(face_id="outside", age=2),
                    dict(face_id=None, age=3),
                ]
            )
        ],
        [dict(identity_group_id="g", split="train")],
        [dict(identity_group_id="g", person_id="p")],
    )
    assert result["violations"]["faces_with_conflicting_age_values"] == 1
    assert result["violations"]["age_labels_outside_group_faces"] == 1
    assert result["counts"]["unmapped_age_labels"] == 1
    assert not result["input_contract_clean"]


def test_recorded_person_and_face_cross_split_detected():
    result = audit(
        [group("g"), group("h")],
        [dict(identity_group_id="g", split="train"), dict(identity_group_id="h", split="test")],
        [dict(identity_group_id="g", person_id="p"), dict(identity_group_id="h", person_id="p")],
    )
    assert result["violations"]["recorded_people_across_splits"] == 1
    assert result["violations"]["faces_across_splits"] == 2


def test_missing_mapping_and_conflicting_split_rows_are_failures():
    result = audit(
        [group()],
        [dict(identity_group_id="g", split="train"), dict(identity_group_id="g", split="val")],
        [],
    )
    assert result["violations"]["groups_without_recorded_person"] == 1
    assert result["violations"]["conflicting_split_rows"] == 1
    assert not result["input_contract_clean"]
