import copy

import pytest

from scripts import run_oriented_noise_cuda_v2 as runner


def native():
    return dict(
        parameters=runner.PROTOCOL | dict(permutation_seed=1),
        inputs=[dict(path="pairs")],
        metrics=dict(
            preflight_complete=True,
            training_seeds=[42, 1, 2],
            permutation_seed=1,
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
    runner.require_noise(native(), "unused", 1)


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
        runner.require_noise(changed, "unused", 1)


def test_unbound_pairs_rejected(monkeypatch):
    monkeypatch.setattr(runner, "file_record", lambda path: dict(path="different"))
    with pytest.raises(ValueError, match="not bound"):
        runner.require_noise(native(), "unused", 1)



@pytest.mark.parametrize("seed", [True, False, 1., "1", None, 0, 3, 42, 2])
def test_noncanonical_or_mismatched_permutation_rejected(monkeypatch, seed):
    monkeypatch.setattr(runner, "file_record", lambda path: dict(path="pairs"))
    with pytest.raises(ValueError):
        runner.require_noise(native(), "unused", seed)


@pytest.mark.parametrize("seeds", [[True,1,2], [42,1,True], [42,2,1], [42,1], [42,1,2.]])
def test_incorrect_training_seed_frame_rejected(monkeypatch, seeds):
    monkeypatch.setattr(runner, "file_record", lambda path: dict(path="pairs"))
    changed = native()
    changed["metrics"]["training_seeds"] = seeds
    with pytest.raises(ValueError):
        runner.require_noise(changed, "unused", 1)


def test_permutation_two_is_separate_from_training_seed_frame(monkeypatch):
    monkeypatch.setattr(runner, "file_record", lambda path: dict(path="pairs"))
    changed = native()
    changed["parameters"]["permutation_seed"] = changed["metrics"]["permutation_seed"] = 2
    runner.require_noise(changed, "unused", 2)


def test_no_hardcoded_permutation42_in_any_generated_manifest_or_summary():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    fields = [node.value for node in ast.walk(tree)
              if isinstance(node, ast.keyword) and node.arg == "permutation_seed"]
    assert len(fields) == 4
    assert all(isinstance(value, ast.Name) and value.id == "permutation_seed" for value in fields)


@pytest.mark.parametrize("key,value", [("batch_size", 64.), ("seeds", [42, True, 2]), ("permutation_seed", True)])
def test_python_equal_but_type_wrong_protocol_rejected(monkeypatch, key, value):
    monkeypatch.setattr(runner, "file_record", lambda path: dict(path="pairs"))
    changed = copy.deepcopy(native())
    changed["parameters"][key] = value
    with pytest.raises(ValueError):
        runner.require_noise(changed, "unused", 1)


def test_numeric_one_is_not_a_true_exposure_flag(monkeypatch):
    monkeypatch.setattr(runner, "file_record", lambda path: dict(path="pairs"))
    changed = native()
    changed["metrics"]["clean_noise_contract"]["endpoint_image_multiplicities_unchanged"] = 1
    with pytest.raises(ValueError):
        runner.require_noise(changed, "unused", 1)

