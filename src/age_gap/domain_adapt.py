"""Self-supervised доменная адаптация визуального энкодера (DINOv2) на VK-лицах.

Меток красоты у VK нет, поэтому «обучение на наших данных» = адаптация ПРЕДСТАВЛЕНИЯ под домен
VK-лиц без меток. Метод — SimSiam (косинусная близость двух аугментаций + stop-grad + предиктор;
без негативов и memory-bank -> дёшево по памяти на laptop-GPU). Дообучаются только верхние блоки
DINOv2 (нижние заморожены — чтобы не разрушить сильные признаки и экономить память).

Результат: адаптированный state_dict энкодера. Дальше на нём супервизится голова красоты по SCUT
(scripts/beauty_train.py --init-encoder), а тест — на человеческой разметке VK.
"""

from __future__ import annotations

import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from age_gap.beauty import _load_vision
from age_gap.common.io import resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def face_crop_paths(dirs: list[str]) -> list[str]:
    """Все hi-res кропы взрослых лиц из указанных data*/interim/faces_hires."""
    out: list[str] = []
    for d in dirs:
        out.extend(glob.glob(str(resolve_path(d, "interim", "faces_hires", "*.jpg"))))
    return sorted(out)


def _aug():
    from torchvision import transforms as T
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.4, 1.0), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(23, (0.1, 2.0))], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class TwoViews(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.aug = _aug()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.aug(img), self.aug(img)


class _MLP(nn.Sequential):
    def __init__(self, dims: list[int], last_bn: bool = True):
        layers: list[nn.Module] = []
        for k in range(len(dims) - 1):
            layers.append(nn.Linear(dims[k], dims[k + 1], bias=False))
            if k < len(dims) - 2 or last_bn:
                layers.append(nn.BatchNorm1d(dims[k + 1]))
            if k < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class SimSiam(nn.Module):
    """DINOv2 (верхние блоки обучаемы) + projector + predictor."""

    def __init__(self, backbone: str = "dinov2", unfreeze_top: int = 4, dim: int = 2048, pred: int = 512):
        super().__init__()
        self.backbone = backbone
        vm, layers, norm, hidden = _load_vision(backbone)
        self.vision = vm
        for p in self.vision.parameters():
            p.requires_grad_(False)
        for blk in layers[-unfreeze_top:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in norm.parameters():
            p.requires_grad_(True)
        self.projector = _MLP([hidden, dim, dim, dim], last_bn=True)
        self.predictor = _MLP([dim, pred, dim], last_bn=False)

    def _emb(self, x: torch.Tensor) -> torch.Tensor:
        return self.vision(pixel_values=x).pooler_output

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1, z2 = self.projector(self._emb(x1)), self.projector(self._emb(x2))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        d = lambda p, z: -F.cosine_similarity(p, z.detach(), dim=-1).mean()  # noqa: E731
        return 0.5 * d(p1, z2) + 0.5 * d(p2, z1)


def adapt(paths: list[str], device: torch.device, backbone: str = "dinov2", unfreeze_top: int = 4,
          epochs: int = 3, batch: int = 32, lr: float = 5e-4, seed: int = 0) -> dict:
    """Прогнать SimSiam. Возвращает адаптированный state_dict энкодера (self.vision) + мета."""
    torch.manual_seed(seed)
    model = SimSiam(backbone, unfreeze_top).to(device)
    dl = DataLoader(TwoViews(paths), batch_size=batch, shuffle=True, num_workers=0,
                    pin_memory=True, drop_last=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, len(dl)), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for ep in range(epochs):
        model.train()
        losses = []
        for x1, x2 in dl:
            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = model(x1, x2)
            scaler.scale(loss).backward()
            stepped = scaler.step(opt)
            scaler.update()
            if stepped is not None:
                sched.step()
            losses.append(float(loss))
        log.info("SimSiam epoch %d/%d  loss=%.4f (-1=идеал)", ep + 1, epochs, float(np.mean(losses)))
    return {"vision_state": {k: v.detach().cpu() for k, v in model.vision.state_dict().items()},
            "backbone": backbone, "unfreeze_top": unfreeze_top, "n_faces": len(paths), "epochs": epochs}
