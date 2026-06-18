"""Лоссы для обучения adapter (TODO §10 «Losses», MVP-3).

ContrastivePairLoss работает напрямую на парах с метками: позитивы притягиваются
(косинус → 1), негативы отталкиваются ниже margin. Это самый прямой лосс для нашей
парной разметки (positive_same_post / negative_*).
"""

from __future__ import annotations

import torch
from torch import nn


class ContrastivePairLoss(nn.Module):
    """L = y * (1 - cos) + (1 - y) * relu(cos - margin).

    Позитив (y=1): штраф растёт при cos < 1. Негатив (y=0): штраф только если cos > margin.
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        za: torch.Tensor,
        zb: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cos = (za * zb).sum(dim=-1)  # za, zb уже L2-нормированы
        per_pair = labels * (1.0 - cos) + (1.0 - labels) * torch.relu(cos - self.margin)
        if weights is not None:
            # Взвешенное среднее (age-supervised: большие возрастные разрывы весят больше).
            return (per_pair * weights).sum() / weights.sum().clamp_min(1e-8)
        return per_pair.mean()
