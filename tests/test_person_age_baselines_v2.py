import pytest

from scripts.person_age_baselines_v2 import constants


def probe():
    return dict(status="ok", row_indices=[0, 1, 2, 3],
                persons=["a", "a", "b", "c"], folds=[0, 0, 1, 1])


def test_fit_only_constants_and_explicit_denominators():
    result = constants(probe(), [0, 0, 10, 20])
    assert result["baselines"]["mean"]["predictions"] == [15, 15, 0, 0]
    assert result["baselines"]["median"]["predictions"] == [10, 10, 0, 0]
    assert result["baselines"]["median"]["mae_image"] == 12.5
    assert result["baselines"]["median"]["mae_person"] == pytest.approx(40 / 3)


def test_test_targets_do_not_change_fit_prediction():
    before = constants(probe(), [0, 0, 10, 20])
    after = constants(probe(), [90, 90, 10, 20])
    assert before["baselines"]["mean"]["predictions"][:2] == after["baselines"]["mean"]["predictions"][:2]


def test_rejects_fragmented_person_fold():
    value = probe()
    value["folds"] = [0, 1, 1, 1]
    with pytest.raises(ValueError, match="person-disjoint"):
        constants(value, [0, 0, 10, 20])


def test_rejects_duplicate_indices():
    value = probe()
    value["row_indices"] = [0, 0, 2, 3]
    with pytest.raises(ValueError):
        constants(value, [0, 0, 10, 20])
