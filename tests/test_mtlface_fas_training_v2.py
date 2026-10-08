import copy

import pytest
import torch
from torch import nn

from scripts.mtlface_fas_training_v2 import FASStepFailure, fas_step


class AgeHead(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.group = nn.Linear(channels, 7)

    def forward(self, maps):
        values = maps.mean((2, 3))
        return values, self.group(values)


class Recognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 3, 1)
        self.bn = nn.BatchNorm2d(3)
        self.age_head = AgeHead()

    def forward(self, images, *, return_shortcuts=False, return_age=False):
        maps = self.bn(self.stem(images))
        identity, age = maps * 0.75, maps * 0.25
        if return_shortcuts:
            return (maps,) * 5 + (identity, age)
        if return_age:
            return identity.mean((2, 3)), identity, age
        raise ValueError("mode required")


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.bn = nn.BatchNorm2d(3)
        self.condition = nn.Embedding(7, 3)

    def forward(self, source, *shortcuts, condition):
        return (
            source
            + 0.1 * self.bn(self.conv(source))
            + 0.01 * self.condition(condition)[:, :, None, None]
        )


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self.condition = nn.Embedding(7, 1)

    def forward(self, images, groups):
        return self.conv(images) + self.condition(groups)[:, :, None, None]


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    recognizer, generator, discriminator = Recognizer(), Generator(), Discriminator()
    g_opt = torch.optim.SGD(generator.parameters(), lr=0.001)
    d_opt = torch.optim.SGD(discriminator.parameters(), lr=0.001)
    source, target = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
    groups = torch.tensor([0, 6])
    return recognizer, generator, discriminator, g_opt, d_opt, source, target, groups


def state(module):
    return copy.deepcopy(module.state_dict())


def test_actual_g_d_updates_without_recognizer_updates_or_new_gradients(setup, monkeypatch):
    recognizer, generator, discriminator, _, d_opt, *_ = setup
    recognizer.train()
    recognizer.age_head.eval()  # restoration must retain mixed child modes
    recognizer.stem.weight.grad = torch.ones_like(recognizer.stem.weight)
    stale = recognizer.stem.weight.grad.clone()
    before = [state(module) for module in setup[:3]]
    flags = [(module, module.training) for module in recognizer.modules()]
    captured = {}
    original = d_opt.step

    def step():
        captured.update({name: p.grad.clone() for name, p in discriminator.named_parameters()})
        original()

    monkeypatch.setattr(d_opt, "step", step)
    result = fas_step(*setup, generator_bn_policy="adapt")
    for name, value in recognizer.state_dict().items():
        torch.testing.assert_close(value, before[0][name], rtol=0, atol=0)
    torch.testing.assert_close(recognizer.stem.weight.grad, stale)
    assert recognizer.age_head.group.weight.grad is None
    assert all(module.training == flag for module, flag in flags)
    assert all(p.requires_grad for p in recognizer.parameters())
    for index, module in enumerate((generator, discriminator), 1):
        assert any(not torch.equal(p, before[index][name]) for name, p in module.named_parameters())
    for name, parameter in discriminator.named_parameters():
        torch.testing.assert_close(parameter.grad, captured[name])  # no extra G-pass D gradients
    assert result["recognizer_updated"] is False and result["full_joint_fas"] is False
    assert result["generator_gradient_norm"] > 0 and result["discriminator_gradient_norm"] > 0
    assert result["generator_loss"] == pytest.approx(
        75 * result["adversarial_loss"] + 0.002 * result["identity_loss"] + 10 * result["age_loss"]
    )


def test_frozen_generator_bn_and_original_frozen_flags_preserved(setup):
    recognizer, generator, *_ = setup
    recognizer.stem.bias.requires_grad_(False)
    before_mean = generator.bn.running_mean.clone()
    fas_step(*setup, generator_bn_policy="frozen")
    torch.testing.assert_close(before_mean, generator.bn.running_mean)
    assert not recognizer.stem.bias.requires_grad


@pytest.mark.parametrize("phase", ["generator", "discriminator"])
def test_nonfinite_gradient_reports_partial_updates_and_restores_flags(setup, phase):
    recognizer, generator, discriminator, *_ = setup
    before = [state(generator), state(discriminator)]
    module = generator if phase == "generator" else discriminator
    module.conv.weight.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    with pytest.raises(FASStepFailure) as caught:
        fas_step(*setup, generator_bn_policy="adapt")
    assert caught.value.discriminator_updated == (phase == "generator")
    assert caught.value.generator_updated is False
    assert isinstance(caught.value.__cause__, FloatingPointError)
    assert all(p.requires_grad for p in recognizer.parameters())
    assert all(p.requires_grad for p in discriminator.parameters())
    assert all(p.grad is None for m in (generator, discriminator) for p in m.parameters())
    for name, p in generator.named_parameters():
        torch.testing.assert_close(p, before[0][name])
    if phase == "discriminator":
        for name, p in discriminator.named_parameters():
            torch.testing.assert_close(p, before[1][name])


@pytest.mark.parametrize(
    "groups",
    [
        torch.tensor([True, False]),
        torch.tensor([0.0, 6.0]),
        torch.tensor([-1, 6]),
        torch.tensor([0, 7]),
        torch.tensor([0]),
    ],
)
def test_invalid_target_groups_refused_before_updates(setup, groups):
    values = list(setup)
    values[-1] = groups
    with pytest.raises(ValueError, match="target groups"):
        fas_step(*values, generator_bn_policy="adapt")


def test_contaminated_optimizer_and_bad_policy_refused(setup):
    values = list(setup)
    values[3] = torch.optim.SGD([*setup[1].parameters(), setup[0].stem.weight], lr=0.001)
    with pytest.raises(ValueError, match="optimizer"):
        fas_step(*values, generator_bn_policy="adapt")
    with pytest.raises(ValueError, match="BN policy"):
        fas_step(*setup, generator_bn_policy="implicit")


def test_identity_loss_alone_has_differentiable_path_through_frozen_recognizer(setup):
    result = fas_step(
        *setup, generator_bn_policy="frozen", gan_weight=0.0, age_weight=0.0, identity_weight=1.0
    )
    assert result["identity_loss"] > 0 and result["generator_gradient_norm"] > 0
    assert result["generator_loss"] == pytest.approx(result["identity_loss"])


def test_nonfinite_updated_state_not_reported_success(setup, monkeypatch):
    _, generator, _, g_opt, *_ = setup
    original = g_opt.step

    def step():
        original()
        with torch.no_grad():
            generator.conv.weight.fill_(float("inf"))

    monkeypatch.setattr(g_opt, "step", step)
    with pytest.raises(FASStepFailure) as caught:
        fas_step(*setup, generator_bn_policy="adapt")
    assert caught.value.discriminator_updated and caught.value.generator_updated


def test_duplicate_optimizer_parameter_refused(setup):
    values = list(setup)
    with pytest.warns(UserWarning, match="duplicate"):
        values[3] = torch.optim.SGD([*setup[1].parameters(), setup[1].conv.weight], lr=0.001)
    with pytest.raises(ValueError, match="exactly once"):
        fas_step(*values, generator_bn_policy="adapt")


def test_shared_parameterless_buffer_refused_before_updates(setup):
    recognizer, generator, *_ = setup
    generator.register_buffer("foreign", recognizer.bn.running_mean)
    with pytest.raises(ValueError, match="state must be disjoint"):
        fas_step(*setup, generator_bn_policy="adapt")


def test_fas_step_integrates_actual_common_joint_adapter(setup):
    from scripts.mtlface_joint_adapter_v2 import CommonJointAdapter

    backbone = nn.Module()
    backbone.input_size = 16
    backbone.preprocess = lambda image: image
    backbone.net = nn.Module()
    backbone.net.conv1 = nn.Conv2d(3, 2, 1)
    backbone.net.bn1 = nn.BatchNorm2d(2)
    backbone.net.prelu = nn.PReLU(2)
    for name, before, after in (
        ("layer1", 2, 2),
        ("layer2", 2, 4),
        ("layer3", 4, 8),
        ("layer4", 8, 16),
    ):
        setattr(backbone.net, name, nn.Conv2d(before, after, 2, stride=2))
    backbone.net.bn2 = nn.BatchNorm2d(16)
    backbone.net.dropout = nn.Dropout()
    backbone.net.fc = nn.Linear(16, 5)
    backbone.net.features = nn.BatchNorm1d(5)

    class Split(nn.Module):
        def forward(self, value):
            return value * 0.75, value * 0.25

    recognizer = CommonJointAdapter(
        backbone, Split(), AgeHead(16), AgeHead(16), stage_channels=(2, 2, 4, 8, 16)
    )
    before = state(recognizer)
    values = list(setup)
    values[0] = recognizer
    values[5], values[6] = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)
    result = fas_step(*values, generator_bn_policy="frozen")
    assert result["generator_updated"] and result["discriminator_updated"]
    assert result["generator_gradient_norm"] > 0
    for name, value in recognizer.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
