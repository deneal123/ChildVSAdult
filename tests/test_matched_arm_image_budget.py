import pytest

from scripts.audit_matched_arm_image_budget import counts


def test_negative_only_images_are_in_total_budget_but_not_positive_budget():
    rows = [{"split": "train", "label": 1, "face_a": "a", "face_b": "b"},
            {"split": "train", "label": 0, "face_a": "a", "face_b": "c"},
            {"split": "test", "label": 0, "face_a": "d", "face_b": "e"}]
    assert counts(rows) == {"positive_pairs": 1, "negative_pairs": 1,
                            "positive_images": 2, "all_train_images": 3,
                            "negative_only_images": 1}


def test_unbalanced_or_nonbinary_train_is_not_reported_as_matched():
    with pytest.raises(ValueError):
        counts([{"split": "train", "label": 1}])
    with pytest.raises(ValueError):
        counts([{"split": "train", "label": 2}])
