"""Тесты механики disentanglement (gradient reversal, age-голова, бакеты) — офлайн."""

from __future__ import annotations

import torch

from age_gap.training.disentangle import N_BUCKETS, _AgeHead, _bucket, _grad_reverse


def test_grad_reverse_flips_gradient_sign():
    x = torch.tensor([2.0], requires_grad=True)
    y = _grad_reverse(x, 0.5) * 3.0  # forward identity, backward умножает на -0.5
    y.backward()
    assert x.grad is not None and float(x.grad) == -0.5 * 3.0  # знак инвертирован


def test_age_head_output_shape():
    head = _AgeHead(dim=512, n_buckets=N_BUCKETS).eval()
    out = head(torch.zeros(4, 512), lambd=0.3)
    assert out.shape == (4, N_BUCKETS)


def test_bucket_unknown_is_minus_one():
    assert _bucket(None) == -1
    assert _bucket(8) >= 0  # известный возраст -> валидный бакет