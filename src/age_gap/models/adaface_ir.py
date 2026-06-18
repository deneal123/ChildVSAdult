"""IR / IR-SE backbone (AdaFace/InsightFace «Backbone») — вендоренная архитектура.

Точная реплика арх из репозитория minchul/cvlface (models/iresnet/model.py), которая, в свою
очередь, повторяет AdaFace/InsightFace `net.py`. Нужна, чтобы загрузить предобученные веса
AdaFace (cvlface, WebFace4M/12M) — это backbone с ДРУГИМ loss (адаптивная маржа, родственно
CurricularFace) и иной глубиной/данными, что даёт депт-контроль и loss-diversity в §1d/§1f.

Отличается от arcface_torch iResNet (см. iresnet.py) блочной структурой: input_layer (conv-bn-prelu),
body (последовательность BasicBlockIR с shortcut MaxPool/conv), output_layer (bn-dropout-fc-bn).
Вход 112x112; нормировка/порядок каналов — см. AdaFaceBackbone.preprocess.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    PReLU,
    ReLU,
    Sequential,
    Sigmoid,
)


class SEModule(Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BasicBlockIR(Module):
    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer: Module = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth)
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_layer(x) + self.shortcut_layer(x)


class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class BottleneckIR(Module):
    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__()
        reduction_channel = depth // 4
        if in_channel == depth:
            self.shortcut_layer: Module = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth)
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, reduction_channel, (1, 1), (1, 1), 0, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, reduction_channel, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, depth, (1, 1), stride, 0, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_layer(x) + self.shortcut_layer(x)


class BottleneckIRSE(BottleneckIR):
    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class _Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    pass


def _get_block(in_channel: int, depth: int, num_units: int, stride: int = 2) -> list[_Bottleneck]:
    return [_Bottleneck(in_channel, depth, stride)] + [
        _Bottleneck(depth, depth, 1) for _ in range(num_units - 1)
    ]


def _get_blocks(num_layers: int) -> list[list[_Bottleneck]]:
    if num_layers == 50:
        return [
            _get_block(64, 64, 3),
            _get_block(64, 128, 4),
            _get_block(128, 256, 14),
            _get_block(256, 512, 3),
        ]
    if num_layers == 100:
        return [
            _get_block(64, 64, 3),
            _get_block(64, 128, 13),
            _get_block(128, 256, 30),
            _get_block(256, 512, 3),
        ]
    raise ValueError(f"поддержаны num_layers 50/100, не {num_layers}")


class Backbone(Module):
    """IR/IR-SE Backbone (вход 112x112, эмбеддинг 512)."""

    def __init__(self, num_layers: int = 50, mode: str = "ir", output_dim: int = 512) -> None:
        super().__init__()
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        blocks = _get_blocks(num_layers)
        unit = {"ir": BasicBlockIR, "ir_se": BasicBlockIRSE}[mode]
        output_channel = 512
        self.output_layer = Sequential(
            BatchNorm2d(output_channel),
            Dropout(0.4),
            Flatten(),
            Linear(output_channel * 7 * 7, output_dim),
            BatchNorm1d(output_dim, affine=False),
        )
        modules = [unit(b.in_channel, b.depth, b.stride) for block in blocks for b in block]
        self.body = Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.body(x)
        return self.output_layer(x)


INPUT_SIZE = 112

# Реестр весов AdaFace (cvlface): (repo_id, filename, depth). AdaFace — адаптивно-маржинальный
# loss (родственно CurricularFace), даёт loss-diversity и сильные r50/r101 для депт-контроля.
_WEIGHTS = {
    "adaface_ir50": ("minchul/cvlface_adaface_ir50_webface4m", "pretrained_model/model.pt", 50),
    "adaface_ir101": (
        "minchul/cvlface_adaface_ir101_webface12m",
        "pretrained_model/model.pt",
        100,
    ),
}


def _local_weights(name: str) -> Path:
    from age_gap.common.io import data_path, resolve_path

    repo, fname, _depth = _WEIGHTS[name]
    local = Path(resolve_path(str(data_path("models_dir", f"{name}.pt"))))
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=repo, filename=fname)
    local.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy(downloaded, local)
    return local


class AdaFaceBackbone(nn.Module):
    """IR Backbone с весами AdaFace (cvlface), L2-нормированный выход.

    Вход — BGR 112x112, нормировка (x-127.5)/127.5 (канонический препроцессинг AdaFace).
    """

    input_size = INPUT_SIZE

    def __init__(self, name: str = "adaface_ir50", pretrained: bool = True) -> None:
        super().__init__()
        if name not in _WEIGHTS:
            raise ValueError(f"неизвестный AdaFace backbone: {name!r}")
        depth = _WEIGHTS[name][2]
        self.name = name
        self.net = Backbone(num_layers=depth, mode="ir")
        n_body = len(self.net.body)
        # head — финальный output_layer (bn-fc-bn); tail добавляет последний блок body.
        self.trainable_scopes = {
            "head": ("output_layer",),
            "tail": (f"body.{n_body - 1}.", "output_layer"),
        }
        if pretrained:
            self._load_pretrained(name)

    def _load_pretrained(self, name: str) -> None:
        sd = torch.load(str(_local_weights(name)), map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        sd = {k[4:]: v for k, v in sd.items() if k.startswith("net.")}
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        from age_gap.common.logging import get_logger

        get_logger(__name__).info(
            "AdaFace %s: загружены веса (missing=%d, unexpected=%d)",
            name,
            len(missing),
            len(unexpected),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=-1)

    def preprocess(self, img: np.ndarray, bgr: bool = True) -> np.ndarray:
        """uint8 HxWx3 -> CHW float32 112x112 BGR, нормировка AdaFace (x-127.5)/127.5."""
        if not bgr:  # вход RGB (например LFW) -> модели нужен BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        arr = (img.astype(np.float32) - 127.5) / 127.5
        return np.transpose(arr, (2, 0, 1))
