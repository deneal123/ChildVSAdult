"""ArcFace iResNet backbone (PyTorch) для строгого мультибэкбонового эксперимента.

Архитектура — каноничный insightface ``arcface_torch`` iResNet (BN-PReLU блоки, вход RGB
112x112, эмбеддинг 512-d). Веса берём в двух режимах силы (одна и та же арх, разные данные
обучения → изолирует «силу backbone»):

* ``arcface_r100`` — сильный (marcelohaps/arcface-torch, iResNet100, partial-fc на крупных
  данных → близок к насыщению, как наш frozen ONNX-ArcFace);
* ``arcface_r50_casia`` — слабый (JustinLeee/FaceMind_ArcFace_iResNet50_CASIA_FaceV5, обучен на
  крошечном CASIA-FaceV5 → большой запас по качеству на cross-age).

Один и тот же протокол позволяет построить кривую «выигрыш от наших пар vs сила backbone».

В отличие от ONNX-моделей insightface, веса обучаемы: frozen-инференс → дообучение на НАШИХ
парах → метрика на внешних бенчмарках (тот же честный протокол, что и для facenet).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

INPUT_SIZE = 112


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(
        self, block: type[IBasicBlock], layers: list[int], embedding_size: int = 512
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=0.0, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, embedding_size)
        self.features = nn.BatchNorm1d(embedding_size, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(
        self, block: type[IBasicBlock], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


_LAYERS = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 14, 3], 100: [3, 13, 30, 3]}

# Реестр весов: (repo_id, filename, depth, формат). safetensors не использует pickle (без гейта),
# .pth грузим weights_only=True (только тензоры).
_WEIGHTS = {
    "arcface_r100": ("marcelohaps/arcface-torch", "backbone.pth", 100, "pth"),
    "arcface_r50_casia": (
        "JustinLeee/FaceMind_ArcFace_iResNet50_CASIA_FaceV5",
        "ArcFace_iResNet50_CASIA_FaceV5.pth",
        50,
        "pth",
    ),
}


def _clean_state_dict(sd: dict) -> dict:
    """Снять обёртки (module./arcface.), отбросить margin-голову (head./logits)."""
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    out = {}
    for k, v in sd.items():
        if k.startswith(("head.", "logits.", "module.head.", "fc_head.")):
            continue
        nk = k
        for pref in ("module.", "arcface.", "backbone."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        out[nk] = v
    return out


def _local_weights(name: str) -> Path:
    from age_gap.common.io import data_path, resolve_path

    repo, fname, _depth, _fmt = _WEIGHTS[name]
    local = Path(resolve_path(str(data_path("models_dir", f"{name}{Path(fname).suffix}"))))
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=repo, filename=fname)
    local.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy(downloaded, local)
    return local


class ArcFaceBackbone(nn.Module):
    """iResNet ArcFace с L2-нормированным выходом (косинус = скалярное произведение).

    ``name`` — ключ из ``_WEIGHTS`` (например ``arcface_r50`` / ``arcface_r50_casia``).
    ``pretrained=False`` создаёт сеть со случайной инициализацией (для тестов).
    """

    # Слои для разморозки при дообучении (аналогично facenet): head мягко, tail агрессивнее.
    trainable_scopes = {
        "head": ("fc", "features"),
        "tail": ("layer4", "bn2", "fc", "features"),
    }
    input_size = INPUT_SIZE

    def __init__(self, name: str = "arcface_r100", pretrained: bool = True) -> None:
        super().__init__()
        if name not in _WEIGHTS:
            raise ValueError(f"неизвестный ArcFace backbone: {name!r}")
        depth = _WEIGHTS[name][2]
        self.name = name
        self.net = IResNet(IBasicBlock, _LAYERS[depth])
        if pretrained:
            self._load_pretrained(name)

    def _load_pretrained(self, name: str) -> None:
        fmt = _WEIGHTS[name][3]
        path = _local_weights(name)
        if fmt == "safetensors":
            from safetensors.torch import load_file

            sd = load_file(str(path))
        else:
            sd = torch.load(str(path), map_location="cpu", weights_only=True)
        missing, unexpected = self.net.load_state_dict(_clean_state_dict(sd), strict=False)
        from age_gap.common.logging import get_logger

        get_logger(__name__).info(
            "ArcFace %s: загружены веса (missing=%d, unexpected=%d)",
            name,
            len(missing),
            len(unexpected),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=-1)

    def preprocess(self, img: np.ndarray, bgr: bool = True) -> np.ndarray:
        """uint8 HxWx3 -> CHW float32 112x112 RGB, нормировка ArcFace (x-127.5)/127.5."""
        if bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        arr = (img.astype(np.float32) - 127.5) / 127.5
        return np.transpose(arr, (2, 0, 1))
