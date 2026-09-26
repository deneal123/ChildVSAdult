from __future__ import annotations

import torch

from age_gap.training.sota_common import FeatureSeparation, age_group, cacon_nt_xent


def test_age_group_boundaries() -> None:
    assert [age_group(x) for x in [0, 9, 10, 19, 20, 59, 60, 100]] == [0, 0, 1, 1, 2, 5, 6, 6]


def test_feature_separation_shapes_and_normalization() -> None:
    identity, age = FeatureSeparation(8)(torch.randn(4, 8))
    assert identity.shape == age.shape == (4, 8)
    assert torch.linalg.vector_norm(identity, dim=1).allclose(torch.ones(4), atol=1e-5)


def test_cacon_loss_rewards_matching_three_views() -> None:
    torch.manual_seed(7)
    base = torch.randn(8, 16)
    matching = cacon_nt_xent(base, base + 0.01, base - 0.01)
    shuffled = cacon_nt_xent(base, base.roll(1, 0), base.roll(2, 0))
    assert matching < shuffled
