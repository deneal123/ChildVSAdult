"""Age-disentanglement: удаление возраста из identity-эмбеддинга (Фаза 9 / §5.2, MTLFace-style).

К обычному contrastive-обучению добавляем age-классификатор через слой ОБРАЩЕНИЯ ГРАДИЕНТА
(gradient reversal): голова учится предсказывать возрастной бакет, а backbone — наоборот, делать
возраст НЕпредсказуемым из эмбеддинга. Идея: identity-признаки сильны для личности и слабы для
возраста → возрастная инвариантность. Возрастные метки — наши (per-face, ~76% покрытие).

Гипотеза: явное удаление возраста улучшит identity-only cross-age (FG-NET large-gap) сверх +pairs.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, read_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import Pair
from age_gap.datasets.pair_builder import _AGE_BUCKETS, _age_bucket
from age_gap.models.backbones import make_backbone
from age_gap.training.dataset import _pair_weight
from age_gap.training.finetune import (
    ImagePairDataset,
    _bb_prep,
    _crop_path,
    _set_trainable,
    _val_auc,
)
from age_gap.training.losses import ContrastivePairLoss

log = get_logger(__name__)
N_BUCKETS = len(_AGE_BUCKETS)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # noqa: ANN001
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # noqa: ANN001, ANN205
        return -ctx.lambd * grad, None


def _grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)  # type: ignore[no-any-return]


class _AgeHead(nn.Module):
    """Классификатор возрастного бакета поверх эмбеддинга (через GRL)."""

    def __init__(self, dim: int = 512, n_buckets: int = N_BUCKETS) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(inplace=True), nn.Linear(128, n_buckets))

    def forward(self, x: torch.Tensor, lambd: float) -> torch.Tensor:
        return self.net(_grad_reverse(x, lambd))


def _bucket(age: int | None) -> int:
    b = _age_bucket(age) if age is not None else None
    return b if b is not None else -1


class _PairAgeDataset(Dataset):
    """Пары + возрастные бакеты обоих лиц (-1 если возраст неизвестен)."""

    def __init__(self, split: str, preprocess, crops_dir: str = "faces") -> None:  # noqa: ANN001
        self._prep = preprocess
        self._items: list[tuple[Path, Path, int, float, int, int]] = []
        for row in read_jsonl(data_path("data_dir", "processed", "pairs.jsonl")):
            p = Pair.from_dict(row)
            if p.split != split:
                continue
            ca, cb = _crop_path(p.face_a, crops_dir), _crop_path(p.face_b, crops_dir)
            if ca.exists() and cb.exists():
                w = _pair_weight(p.label, p.age_gap, 0.0)
                self._items.append((ca, cb, p.label, w, _bucket(p.age_a), _bucket(p.age_b)))
        log.info("PairAgeDataset(split=%s): пар=%d", split, len(self._items))

    def __len__(self) -> int:
        return len(self._items)

    @property
    def labels(self) -> list[int]:
        return [it[2] for it in self._items]

    def __getitem__(self, idx: int):
        ca, cb, y, w, ba, bb = self._items[idx]
        ima, imb = cv2.imread(str(ca)), cv2.imread(str(cb))
        assert ima is not None and imb is not None
        return (
            torch.from_numpy(self._prep(ima)),
            torch.from_numpy(self._prep(imb)),
            torch.tensor(float(y)),
            torch.tensor(float(w)),
            torch.tensor(ba),
            torch.tensor(bb),
        )


def train_disentangle(
    backbone_name: str = "facenet",
    epochs: int = 10,
    lr: float = 3e-5,
    batch_size: int = 64,
    margin: float = 0.3,
    patience: int = 3,
    lambda_age: float = 0.3,
    trainable_scope: str = "head",
    ckpt_out: Path | None = None,
    seed: int = 42,
) -> Path:
    """Дообучить backbone с age-adversarial головой; отбор по identity val-AUC. Возвращает чекпойнт."""
    ckpt_out = ckpt_out or data_path("models_dir", f"bb_{backbone_name}_disentangle.pt")
    torch.manual_seed(seed)
    device = torch_device()

    backbone = make_backbone(backbone_name, pretrained=True).to(device)
    prep = _bb_prep(backbone)
    train_ds = _PairAgeDataset("train", prep)
    # Val — обычный 4-кортеж (identity-AUC для отбора; возрастные бакеты не нужны).
    val_ds = ImagePairDataset(split="val", preprocess=prep, crops_dir="faces")
    if len(train_ds) == 0:
        raise RuntimeError("Пустой train-сплит")

    if trainable_scope != "full":
        _set_trainable(backbone, trainable_scope)
    age_head = _AgeHead().to(device)
    params = [p for p in backbone.parameters() if p.requires_grad] + list(age_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    id_loss = ContrastivePairLoss(margin=margin)
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    meta = {"backbone": backbone_name, "crops_dir": "faces"}

    best_auc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
    no_improve = 0
    for epoch in range(1, epochs + 1):
        backbone.train()
        age_head.train()
        tot_id = tot_age = 0.0
        for ta, tb, y, w, ba, bb in loader:
            ta, tb, y, w = ta.to(device), tb.to(device), y.to(device), w.to(device)
            ba, bb = ba.to(device), bb.to(device)
            opt.zero_grad()
            za, zb = backbone(ta), backbone(tb)
            l_id = id_loss(za, zb, y, weights=w)
            # Age-adversarial: GRL заставляет backbone убирать возраст из эмбеддинга.
            emb = torch.cat([za, zb], dim=0)
            buckets = torch.cat([ba, bb], dim=0)
            l_age = ce(age_head(emb, lambda_age), buckets)
            (l_id + l_age).backward()
            opt.step()
            tot_id += float(l_id.detach()) * len(y)
            tot_age += float(l_age.detach()) * len(y)
        auc = _val_auc(backbone, val_ds, device, batch_size)  # type: ignore[arg-type]
        if not np.isnan(auc) and auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
            no_improve = 0
            ckpt_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, **meta}, ckpt_out)
        else:
            no_improve += 1
        log.info(
            "epoch %d/%d L_id=%.4f L_age=%.4f val_auc=%.4f (best=%.4f)",
            epoch,
            epochs,
            tot_id / len(train_ds),
            tot_age / len(train_ds),
            auc,
            best_auc,
        )
        if not np.isnan(auc) and no_improve >= patience:
            log.info("Early stop на эпохе %d", epoch)
            break

    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, **meta}, ckpt_out)
    log.info("Disentangle %s -> %s (best_val_auc=%.4f)", backbone_name, ckpt_out, best_auc)
    return ckpt_out
