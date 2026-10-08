import copy

import pytest

from scripts import run_oriented_noise_cuda_v1 as runner


def native():
    return dict(
        parameters=runner.PROTOCOL | dict(permutation_seed=42),
        inputs=[dict(path="pairs")],
        metrics=dict(
            preflight_complete=True,
            permutation_seed=42,
            clean_noise_contract=dict(
                positive_pairs=600,
                negative_pairs=600,
                all_train_images=1096,
                recorded_train_people=500,
                fixed_negatives_and_heldout_unchanged=True,
                endpoint_image_multiplicities_unchanged=True,
            ),
        ),
    )


def test_compatible_noise_preflight(monkeypatch):
    monkeypatch.setattr(runner, "file_record", lambda path: dict(path="pairs"))
    runner.require_noise(native(), "unused")


@pytest.mark.parametrize(
    "key,value",
    [
        ("all_train_images", 1121),
        ("fixed_negatives_and_heldout_unchanged", False),
        ("endpoint_image_multiplicities_unchanged", False),
    ],
)
def test_incompatible_budget_rejected(monkeypatch, key, value):
    monkeypatch.setattr(runner, "file_record", lambda path: dict(path="pairs"))
    changed = copy.deepcopy(native())
    changed["metrics"]["clean_noise_contract"][key] = value
    with pytest.raises(ValueError, match="exposure"):
        runner.require_noise(changed, "unused")


def test_unbound_pairs_rejected(monkeypatch):
    monkeypatch.setattr(runner, "file_record", lambda path: dict(path="different"))
    with pytest.raises(ValueError, match="not bound"):
        runner.require_noise(native(), "unused")
