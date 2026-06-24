"""E21: SOTA-objective baseline — ArcFace-margin классификация по личности на наших данных.

Современные cross-age методы (OE-CNN, MTLFace, AIM) строятся на margin-softmax-классификации по
идентичности (ArcFace/CosFace) + age-компонент. Здесь берём ЯДРО этой цели — ArcFace-margin
классификацию — и обучаем на наших label-free-намайненных личностях, на ТОМ ЖЕ слабом обучаемом
backbone, том же trainable-scope и той же оценке, что и наш pair-contrastive (E1/E17). Вопрос:
даёт ли классификационная SOTA-цель преимущество над простым контрастивом на наших sparse few-shot
личностях (2-3 фото)?

Селекция модели — по identity-AUC на нашем val (как в finetune), не по train-accuracy.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, read_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import IdentityGroup
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import (
    ImagePairDataset,
    PreprocessFn,
    _bb_prep,
    _crop_path,
    _set_trainable,
    _val_auc,
)

log = get_logger(__name__)


_DEFAULT_MARGIN = {"arcface": 0.5, "cosface": 0.35, "sphereface": 2.0}


class MarginHead(nn.Module):
    """Margin-softmax head over a normalized backbone embedding. Three objective families:

    * ``arcface`` --- additive angular margin cos(θ+m) (Deng et al., 2019).
    * ``cosface`` --- additive cosine margin cos(θ)-m (Wang et al., 2018).
    * ``sphereface`` --- multiplicative angular margin cos(mθ) (Liu et al., 2017), with the
      original piecewise-monotonic ψ. SphereFace has no λ-annealing here, so on sparse
      few-shot identities it can train slowly --- reported honestly as part of the comparison.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        loss_type: str = "arcface",
        margin: float | None = None,
        scale: float = 32.0,
        sub_centers: int = 1,
    ):
        super().__init__()
        self.n_classes = n_classes
        # sub_centers>1 => sub-center ArcFace (Deng et al., ECCV 2020): K centroids per class,
        # max-pooled at forward time so label-noisy/low-quality samples are routed to off-centers
        # instead of corrupting the dominant centroid. A noise-robust comparator for mined web labels.
        self.sub_centers = sub_centers
        self.weight = nn.Parameter(torch.empty(n_classes * sub_centers, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.loss_type = loss_type
        self.scale = scale
        self.margin = _DEFAULT_MARGIN[loss_type] if margin is None else margin
        m = self.margin
        self.cos_m = math.cos(m)  # arcface precompute
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # порог устойчивости при cos(θ+m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = nn.functional.linear(
            nn.functional.normalize(emb), nn.functional.normalize(self.weight)
        )
        if self.sub_centers > 1:  # max over the K sub-centers of each class -> (B, n_classes)
            cosine = cosine.view(-1, self.n_classes, self.sub_centers).amax(dim=2)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        if self.loss_type == "cosface":
            phi = cosine - self.margin
        elif self.loss_type == "sphereface":
            theta = torch.acos(cosine)
            k = torch.floor(self.margin * theta / math.pi)
            phi = ((-1.0) ** k) * torch.cos(self.margin * theta) - 2.0 * k  # piecewise ψ
        else:  # arcface
            sine = torch.sqrt(1.0 - cosine**2)
            phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ+m)
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # монотонность вне диапазона
        onehot = torch.zeros_like(cosine)
        onehot.scatter_(1, labels.view(-1, 1), 1.0)
        return (onehot * phi + (1.0 - onehot) * cosine) * self.scale


ArcMarginHead = MarginHead  # обратная совместимость


class FaceLabelDataset(Dataset):
    """Одиночные кропы лиц с меткой личности (для классификации); только личности с >= min_faces."""

    def __init__(
        self,
        split: str,
        preprocess: PreprocessFn,
        crops_dir: str = "faces",
        min_faces: int = 2,
        groups_file: str | None = None,
        split_map_file: str | None = None,
    ) -> None:
        groups_file = groups_file or str(
            data_path("data_dir", "processed", "identity_groups.jsonl")
        )
        split_map_file = split_map_file or str(data_path("splits_dir", "group_splits.jsonl"))
        split_map = {r["identity_group_id"]: r["split"] for r in read_jsonl(split_map_file)}
        self._prep = preprocess
        self._items: list[tuple[Path, int]] = []
        label_idx: dict[str, int] = {}
        for row in read_jsonl(groups_file):
            g = IdentityGroup.from_dict(row)
            if split_map.get(g.identity_group_id) != split:
                continue
            faces = [f for f in g.faces if _crop_path(f, crops_dir).exists()]
            if len(faces) < min_faces:  # класс с 1 фото не выучить -> исключаем (как и контрастив)
                continue
            lbl = label_idx.setdefault(g.identity_group_id, len(label_idx))
            for f in faces:
                self._items.append((_crop_path(f, crops_dir), lbl))
        self.n_classes = len(label_idx)
        log.info(
            "FaceLabelDataset(split=%s): лиц=%d, личностей(классов)=%d (>=%d фото)",
            split,
            len(self._items),
            self.n_classes,
            min_faces,
        )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self._items[idx]
        img = cv2.imread(str(path))
        assert img is not None, f"не прочитан кроп: {path}"
        return torch.from_numpy(self._prep(img)), label


def train_arcface(
    backbone_name: str = "facenet",
    loss_type: str = "arcface",
    epochs: int = 15,
    lr_backbone: float = 3e-5,
    lr_head: float = 1e-3,
    batch_size: int = 128,
    margin: float | None = None,
    scale: float = 32.0,
    trainable_scope: str = "head",
    patience: int = 4,
    seed: int = 42,
    sub_centers: int = 1,
    ckpt_out: Path | None = None,
) -> Path:
    """Обучить backbone ArcFace-классификацией по личности; отбор по identity val-AUC.

    backbone движется мягко (lr_backbone, как +pairs), ArcFace-классификатор учится быстрее (lr_head).
    Чекпойнт совместим с ``load_finetuned`` (сохраняется только backbone — голова не нужна для оценки).
    """
    _sc = f"_sc{sub_centers}" if sub_centers > 1 else ""
    ckpt_out = ckpt_out or data_path("models_dir", f"bb_{backbone_name}_{loss_type}{_sc}.pt")
    torch.manual_seed(seed)
    device = torch_device()

    backbone = make_backbone(backbone_name, pretrained=True).to(device)
    prep = _bb_prep(backbone)
    train_ds = FaceLabelDataset("train", preprocess=prep)
    val_ds = ImagePairDataset(split="val", preprocess=prep)
    if train_ds.n_classes == 0:
        raise RuntimeError("Нет классов: проверьте identity_groups.jsonl и group_splits.jsonl")

    size = int(getattr(backbone, "input_size", 160))
    backbone.eval()  # проб emb-dim батчем=1: train-режим уронил бы BatchNorm (нужно >1 примера)
    with torch.no_grad():
        emb_dim = int(backbone(torch.zeros(1, 3, size, size, device=device)).shape[-1])
    head = MarginHead(
        emb_dim, train_ds.n_classes, loss_type=loss_type, margin=margin, scale=scale,
        sub_centers=sub_centers,
    ).to(device)

    if trainable_scope != "full":
        _set_trainable(backbone, trainable_scope)
    bb_params = [p for p in backbone.parameters() if p.requires_grad]
    opt = torch.optim.Adam(
        [{"params": bb_params, "lr": lr_backbone}, {"params": head.parameters(), "lr": lr_head}]
    )
    ce = nn.CrossEntropyLoss()
    # drop_last: последний батч из 1 примера уронил бы BatchNorm в train-режиме.
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    log.info(
        "%s %s: classes=%d, device=%s, lr=(bb %.0e, head %.0e), m=%.2f s=%.0f",
        loss_type,
        backbone_name,
        train_ds.n_classes,
        device,
        lr_backbone,
        lr_head,
        head.margin,
        scale,
    )

    meta = {"backbone": backbone_name, "crops_dir": "faces"}
    best_auc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
    no_improve = 0
    for epoch in range(1, epochs + 1):
        backbone.train()
        head.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(head(backbone(x), y), y)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
        avg = total / len(train_ds)
        auc = _val_auc(backbone, val_ds, device, batch_size)
        if not np.isnan(auc) and auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
            no_improve = 0
            ckpt_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, **meta}, ckpt_out)
        else:
            no_improve += 1
        log.info("epoch %d/%d ce=%.4f val_auc=%.4f (best=%.4f)", epoch, epochs, avg, auc, best_auc)
        if not np.isnan(auc) and no_improve >= patience:
            log.info("Early stop на эпохе %d", epoch)
            break

    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, **meta}, ckpt_out)
    log.info("ArcFace %s сохранён -> %s (best_val_auc=%.4f)", backbone_name, ckpt_out, best_auc)
    return ckpt_out
