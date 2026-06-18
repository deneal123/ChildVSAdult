"""Кросс-возрастной adapter поверх замороженного face-энкодера (TODO §10, MVP-3).

Adapter преобразует baseline-эмбеддинг (512-d ArcFace) в age-invariant эмбеддинг той же
размерности. Реализованы два варианта: простой 2-слойный MLP и residual-MLP (вход
складывается с выходом — стабильнее обучается и не теряет baseline-сигнал). Выход
L2-нормируется, поэтому косинус = скалярное произведение.
"""

from __future__ import annotations

import torch
from torch import nn


class MLPAdapter(nn.Module):
    """2-слойный MLP. residual=True добавляет вход к выходу (рекомендуется)."""

    def __init__(
        self,
        dim: int = 512,
        hidden: int = 512,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.residual:
            z = z + x
        return nn.functional.normalize(z, dim=-1)
