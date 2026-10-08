from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scripts.smoke_mtlface_joint_v3 import run_joint_smoke


class Recognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 3, 1)
        self.age_head = AgeHead()

    def set_batchnorm_policy(self, policy="frozen_all"):
        if policy != "frozen_all":
            raise ValueError("frozen BN required")

    def forward(self, images, *, return_shortcuts=False, return_age=False):
        maps = self.stem(images)
        if return_shortcuts:
            return (maps,) * 5 + (maps, maps)
        if return_age:
            return maps.mean((2, 3)), maps, maps
        return maps.mean((2, 3))

    def losses(self, images, identities, ages, head):
        loss = F.cross_entropy(head(self(images)), identities)
        return {"total": loss}


class AgeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.group = nn.Linear(3, 7)

    def forward(self, maps):
        means = maps.mean((2, 3))
        return means, self.group(means)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(3, 3, 1)

    def forward(self, images, *shortcuts, condition):
        return images + 0.1 * self.conv3(images)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 1)

    def forward(self, images, groups):
        return self.conv1(images)


class Dataset:
    records = (SimpleNamespace(age=0), SimpleNamespace(age=None))

    def __getitem__(self, index):
        return torch.full((3, 4, 4), float(index + 1)), index, 0 if index == 0 else -1


def test_smoke_executes_recognition_then_fas_with_age_zero_and_missing():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    result, indices = run_joint_smoke(
        Recognizer(), nn.Linear(3, 2), Dataset(), Generator(), Discriminator()
    )
    assert indices == [0, 1]
    assert result["target_group"] == 0
    assert result["recognizer_state_unchanged_during_fas"] is True
    assert result["identity_head_unchanged_during_fas"] is True
    assert result["training_complete"] is False
    assert result["fas"]["generator_updated"] is True


def test_smoke_refuses_missing_age_stratum_before_training():
    dataset = Dataset()
    dataset.records = (SimpleNamespace(age=0),)
    with pytest.raises(ValueError, match="missing age"):
        run_joint_smoke(Recognizer(), nn.Linear(3, 2), dataset, Generator(), Discriminator())
