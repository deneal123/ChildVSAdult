import pytest

from scripts.run_oriented_recovery import command, paths, require_stage_metrics


def test_new_targets_and_commands_never_reference_old_checkpoint_directory():
    targets = paths("test_recovery")
    smoke = command("smoke", targets["logs"] / "preflight.json", targets)
    train = command("train", targets["logs"] / "preflight.json", targets)
    assert "--smoke" not in smoke and "--smoke" in train
    assert str(targets["models"]) in train
    assert "--resume" not in train
    assert all("test_recovery" in str(path) for path in targets.values())


@pytest.mark.parametrize("tag", ["", "../escape", "a/b", "a\\b", "a b"])
def test_unsafe_tag_refused(tag):
    with pytest.raises(ValueError):
        paths(tag)


def test_process_exit_alone_does_not_close_training_gate():
    with pytest.raises(ValueError):
        require_stage_metrics("train", {"training_complete": True, "completed_cells": 1})
    with pytest.raises(ValueError):
        require_stage_metrics("smoke", {})
    require_stage_metrics("smoke", {"smoke_complete": True})
    require_stage_metrics("train", {"training_complete": True, "completed_cells": 6})
