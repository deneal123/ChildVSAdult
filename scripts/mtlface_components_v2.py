"""Map-level MTLFace components, not the complete joint/FAS method.

Algorithm reference: Hzzone/MTLFace@03ad57942e19ee28733b03a441765481f6606460,
backbone/aifr.py AttentionModule / AgeEstimationModule and common/ops.py
age2group; Huang et al., CVPR 2021. Independently implemented topology with
tiny configurable dimensions for tests. Upstream licensing requires separate
verification before redistribution; no license for upstream code is asserted.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def age_target(value):
    if value is None:
        return -1
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("age must be an integer or None")
    if value < 0:
        raise ValueError("negative age is not a missing-age marker")
    return value


def age_group(value):
    value = age_target(value)
    if value < 0:
        raise ValueError("missing age has no group")
    return sum(value > threshold for threshold in (10, 20, 30, 40, 50, 60))


def _positive_dimensions(*values):
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
        raise ValueError("positive integer dimensions required")


def _map_shape(maps, channels):
    if maps.ndim != 4 or maps.shape[1] != channels or min(maps.shape[2:]) < 1:
        raise ValueError("nonempty spatial map with declared channel count required")


def _pyramid(maps, pooling):
    return torch.cat([pooling(maps, side).flatten(1) for side in (1, 2, 3)], dim=1)[
        :, :, None, None
    ]


class SpatialAgeIdentitySplit(nn.Module):
    def __init__(self, channels=512, reduction=16):
        super().__init__()
        _positive_dimensions(channels, reduction)
        hidden = channels * 14 // reduction
        _positive_dimensions(hidden)
        self.channels = channels
        self.channel = nn.Sequential(
            nn.Conv2d(channels * 14, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5, momentum=0.01),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01),
            nn.Sigmoid(),
        )

    def forward(self, maps):
        _map_shape(maps, self.channels)
        pooled = _pyramid(maps, F.adaptive_avg_pool2d) + _pyramid(maps, F.adaptive_max_pool2d)
        spatial = torch.cat((maps.max(dim=1, keepdim=True).values, maps.mean(1, keepdim=True)), 1)
        age = (maps * self.channel(pooled) + maps * self.spatial(spatial)) * 0.5
        return maps - age, age


class MapAgeHead(nn.Module):
    def __init__(self, channels=512, spatial_size=7, hidden=512, groups=7):
        super().__init__()
        _positive_dimensions(channels, spatial_size, hidden, groups)
        self.channels = channels
        self.spatial_size = spatial_size
        self.age_output_layer = nn.Sequential(
            # Official AgeEstimationModule uses default momentum=0.1, not
            # the attention block's explicit momentum=0.01.
            nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1),
            nn.Flatten(),
            nn.Linear(channels * spatial_size**2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 101),
        )
        self.group_output_layer = nn.Linear(101, groups)

    def forward(self, maps):
        _map_shape(maps, self.channels)
        if maps.shape[2:] != (self.spatial_size, self.spatial_size):
            raise ValueError("declared square map size required")
        years = self.age_output_layer(maps)
        return years, self.group_output_layer(years)
