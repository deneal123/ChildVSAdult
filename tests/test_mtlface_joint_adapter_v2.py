import pytest
import torch
from torch import nn

from scripts.mtlface_joint_adapter_v2 import CommonJointAdapter
from scripts.mtlface_recognition_v2 import SpatialRecognitionAdapter


class TinyBackbone(nn.Module):
    input_size = 16

    def __init__(self):
        super().__init__()
        self.net = nn.Module()
        self.net.conv1 = nn.Conv2d(3, 2, 1)
        self.net.bn1 = nn.BatchNorm2d(2)
        self.net.prelu = nn.PReLU(2)
        for name, before, after in (
            ("layer1", 2, 2),
            ("layer2", 2, 4),
            ("layer3", 4, 8),
            ("layer4", 8, 16),
        ):
            setattr(self.net, name, nn.Conv2d(before, after, 2, stride=2))
        self.net.bn2 = nn.BatchNorm2d(16)
        self.net.dropout = nn.Dropout()
        self.net.fc = nn.Linear(16, 5)
        self.net.features = nn.BatchNorm1d(5)

    def preprocess(self, image):
        return image


class Split(nn.Module):
    def forward(self, value):
        return value * 0.75, value * 0.25


@pytest.fixture
def adapters():
    torch.set_num_threads(1)
    args = (TinyBackbone(), Split(), nn.Identity(), nn.Identity())
    base = SpatialRecognitionAdapter(*args).eval()
    joint = CommonJointAdapter(*args, stage_channels=(2, 2, 4, 8, 16)).eval()
    return base, joint


def test_shortcuts_and_recognition_are_equivalent(adapters):
    base, joint = adapters
    images = torch.randn(2, 3, 16, 16)
    original = images.clone()
    expected = base.components(images)
    actual = joint(images, return_age=True)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(joint(images), expected[0])
    shortcuts = joint(images, return_shortcuts=True)
    assert [tuple(x.shape[1:]) for x in shortcuts] == [
        (2, 16, 16),
        (2, 8, 8),
        (4, 4, 4),
        (8, 2, 2),
        (16, 1, 1),
        (16, 1, 1),
        (16, 1, 1),
    ]
    torch.testing.assert_close(shortcuts[-1] + shortcuts[-2], shortcuts[4])
    torch.testing.assert_close(images, original)
    assert joint.full_joint_fas is False


def test_shortcuts_preserve_gradients_and_caller_no_grad(adapters):
    _, joint = adapters
    images = torch.randn(2, 3, 16, 16, requires_grad=True)
    maps = joint(images, return_shortcuts=True)
    maps[-2].square().sum().backward()
    assert images.grad is not None and images.grad.abs().sum() > 0
    assert joint.backbone.net.conv1.weight.grad is not None
    with torch.no_grad():
        detached = joint(images, return_shortcuts=True)
    assert all(not value.requires_grad for value in detached)


@pytest.mark.parametrize(
    "images", [torch.randn(2, 3, 15, 15), torch.zeros(2, 3, 16, 16, dtype=torch.int64)]
)
def test_bad_input_refused(adapters, images):
    with pytest.raises(ValueError):
        adapters[1](images)


def test_nonfinite_input_and_wrong_stage_refused(adapters):
    _, joint = adapters
    with pytest.raises(ValueError):
        joint(torch.full((2, 3, 16, 16), float("nan")))
    joint.backbone.net.layer1 = nn.Identity()
    with pytest.raises(ValueError, match="stage"):
        joint(torch.randn(2, 3, 16, 16))


def test_return_mode_refused(adapters):
    images = torch.randn(2, 3, 16, 16)
    with pytest.raises(ValueError):
        adapters[1](images, return_age=True, return_shortcuts=True)
    with pytest.raises(ValueError):
        adapters[1](images, return_age=1)
