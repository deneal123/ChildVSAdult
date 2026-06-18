"""Тесты adapter и контрастивного лосса (требуют torch — иначе пропускаются)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from age_gap.models.adapter import MLPAdapter  # noqa: E402
from age_gap.training.losses import ContrastivePairLoss  # noqa: E402


def test_adapter_output_shape_and_norm():
    model = MLPAdapter(dim=512)
    x = torch.randn(4, 512)
    z = model(x)
    assert z.shape == (4, 512)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_contrastive_loss_positive_lower_when_aligned():
    loss_fn = ContrastivePairLoss(margin=0.3)
    a = torch.nn.functional.normalize(torch.randn(8, 512), dim=-1)
    same = a.clone()
    y_pos = torch.ones(8)
    # Идентичные позитивы -> cos=1 -> лосс позитивов ~0.
    assert float(loss_fn(a, same, y_pos)) < 1e-4


def test_contrastive_loss_negative_penalized_when_aligned():
    loss_fn = ContrastivePairLoss(margin=0.3)
    a = torch.nn.functional.normalize(torch.randn(8, 512), dim=-1)
    y_neg = torch.zeros(8)
    # Идентичные негативы -> cos=1 > margin -> положительный штраф.
    assert float(loss_fn(a, a.clone(), y_neg)) > 0.0


def test_pair_weight_emphasizes_large_gap():
    from age_gap.training.dataset import _pair_weight

    assert _pair_weight(1, 0, gap_weight=2.0) == 1.0  # позитив без разрыва
    assert _pair_weight(1, 30, gap_weight=2.0) == 3.0  # большой разрыв -> 1 + 2*1
    assert _pair_weight(1, None, gap_weight=2.0) == 1.0  # неизвестный возраст
    assert _pair_weight(0, 30, gap_weight=2.0) == 1.0  # негатив не взвешивается


def test_weighted_loss_prioritizes_weighted_pairs():
    loss_fn = ContrastivePairLoss(margin=0.3)
    # Два позитива; один хорошо выровнен (cos~1), другой плохо (cos~0).
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([1.0, 1.0])
    # Больший вес на плохой паре -> взвешенный лосс выше, чем равновесный.
    w_hi = torch.tensor([0.1, 3.0])
    w_eq = torch.tensor([1.0, 1.0])
    assert float(loss_fn(a, b, y, weights=w_hi)) > float(loss_fn(a, b, y, weights=w_eq))


def test_adapter_training_reduces_loss():
    torch.manual_seed(0)
    model = MLPAdapter(dim=32, hidden=32)
    loss_fn = ContrastivePairLoss(margin=0.3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    a = torch.nn.functional.normalize(torch.randn(16, 32), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(16, 32), dim=-1)
    y = torch.cat([torch.ones(8), torch.zeros(8)])

    with torch.no_grad():
        first = float(loss_fn(model(a), model(b), y))
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(model(a), model(b), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        last = float(loss_fn(model(a), model(b), y))
    assert last < first
