import pytest
import torch
from torch import nn

from scripts.build_mtlface_recognition_v2 import clean_backbone_state, strict_load


def test_strict_state_load_with_known_prefixes_and_discarded_classifier():
    module = nn.Linear(3, 2)
    state = {
        "module.backbone." + key: torch.ones_like(value)
        for key, value in module.state_dict().items()
    }
    state["module.head.weight"] = torch.zeros(5, 3)
    metadata = strict_load(module, {"state_dict": state})
    assert metadata["missing_keys"] == 0 and metadata["skipped_classifier_tensors"] == 1
    assert module.weight.eq(1).all()


@pytest.mark.parametrize("failure", ["missing", "extra", "shape", "dtype", "nan"])
def test_bad_initial_state_refused_before_any_weight_copy(failure):
    module = nn.Linear(3, 2)
    before = {key: value.clone() for key, value in module.state_dict().items()}
    payload = {key: torch.ones_like(value) for key, value in before.items()}
    if failure == "missing":
        del payload["bias"]
    elif failure == "extra":
        payload["unrecognized"] = torch.tensor(1.0)
    elif failure == "shape":
        payload["bias"] = torch.ones(3)
    elif failure == "dtype":
        payload["bias"] = payload["bias"].double()
    else:
        payload["bias"][0] = float("nan")
    with pytest.raises(ValueError):
        strict_load(module, payload)
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_prefix_collision_is_not_silently_overwritten():
    with pytest.raises(ValueError, match="collision"):
        clean_backbone_state({"weight": torch.ones(1), "module.weight": torch.zeros(1)})
