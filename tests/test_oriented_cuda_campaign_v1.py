import copy

import pytest

from scripts import run_oriented_cuda_campaign_v1 as runner


def cell():
    return dict(
        parameters=runner.PROTOCOL | dict(device="cuda", arm="low", seed=42),
        metrics=dict(training_complete=True),
    )


def test_completed_cuda_contract(monkeypatch):
    native = cell()
    monkeypatch.setattr(runner, "verified", lambda *args: native)
    assert runner.check_cell("unused", "low", 42) is native


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "cpu"),
        ("seed", 1),
        ("selected_epoch", 9),
        ("batchnorm_policy", "adapt_all"),
        ("arm", "cross"),
    ],
)
def test_wrong_protocol_rejected(monkeypatch, field, value):
    native = copy.deepcopy(cell())
    native["parameters"][field] = value
    monkeypatch.setattr(runner, "verified", lambda *args: native)
    with pytest.raises(ValueError, match="protocol"):
        runner.check_cell("unused", "low", 42)


def test_incomplete_rejected(monkeypatch):
    native = cell()
    native["metrics"]["training_complete"] = False
    monkeypatch.setattr(runner, "verified", lambda *args: native)
    with pytest.raises(ValueError, match="completed"):
        runner.check_cell("unused", "low", 42)
