"""Тесты ArcFace-margin головы (E21) — чистая математика, без данных/моделей."""

from __future__ import annotations

import math

import torch

from age_gap.training.arcface_train import ArcMarginHead


def test_arcmargin_output_shape():
    head = ArcMarginHead(in_features=8, n_classes=5)
    emb = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])
    out = head(emb, labels)
    assert out.shape == (4, 5)
    assert torch.isfinite(out).all()


def test_arcmargin_penalizes_target():
    # emb выровнен с весом целевого класса -> cosine_target ≈ 1; margin делает логит < scale.
    head = ArcMarginHead(in_features=8, n_classes=5, margin=0.5, scale=32.0)
    with torch.no_grad():
        w = torch.nn.functional.normalize(head.weight, dim=-1)
    emb = w[2:3].clone()
    out = head(emb, torch.tensor([2]))
    # целевой логит = scale * cos(0 + m); строго меньше scale * cos(0) = 32.
    assert out[0, 2].item() < 32.0
    assert abs(out[0, 2].item() - 32.0 * math.cos(0.5)) < 1e-2


def test_arcmargin_nontarget_is_plain_cosine():
    # Неприцелевой класс не получает margin: логит = scale * cosine.
    head = ArcMarginHead(in_features=8, n_classes=5, margin=0.5, scale=32.0)
    with torch.no_grad():
        w = torch.nn.functional.normalize(head.weight, dim=-1)
        cos = (w[2:3] @ w.T).squeeze(0)  # косинусы emb(класс2) ко всем классам
    out = head(w[2:3].clone(), torch.tensor([2]))
    # класс 0 (не цель): логит ≈ scale * cos(angle to class 0)
    assert abs(out[0, 0].item() - 32.0 * cos[0].item()) < 1e-2
