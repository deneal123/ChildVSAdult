"""Common-protocol reimplementations of MTLFace and CACon.

These implementations keep the mined training identities, person-disjoint
split, backbone, optimizer budget and downstream evaluation fixed.  They are
therefore controlled reimplementations, not claims to reproduce published
leaderboard numbers obtained with MS1M/CASIA and substantially larger compute.

MTLFace-CP preserves the three recognition-stage objectives from the official
implementation: CosFace identity classification, explicit age estimation from
an attention-separated age component, and adversarial age prediction from the
identity component.  CACon-CP implements its three-view, two-positive NT-Xent
objective; the third view is generated once by FRAN and cached on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from age_gap.common.io import data_path, read_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import IdentityGroup
from age_gap.models.backbones import make_backbone
from age_gap.training.arcface_train import MarginHead
from age_gap.training.finetune import (
    ImagePairDataset,
    PreprocessFn,
    _bb_prep,
    _crop_path,
    _set_trainable,
    _val_auc,
)

log = get_logger(__name__)


def age_group(age: int) -> int:
    """Seven groups used by the official MTLFace recognition stage."""
    return int(np.digitize(age, [10, 20, 30, 40, 50, 60], right=False))


@dataclass(frozen=True)
class FaceItem:
    path: Path
    identity: int
    age: int | None


def common_protocol_faces(split: str = "train", crops_dir: str = "faces") -> list[FaceItem]:
    """Person-disjoint face list with mined identity and optional explicit age."""
    split_map = {
        r["identity_group_id"]: r["split"]
        for r in read_jsonl(data_path("splits_dir", "group_splits.jsonl"))
    }
    items: list[FaceItem] = []
    identity = 0
    for row in read_jsonl(data_path("data_dir", "processed", "identity_groups.jsonl")):
        group = IdentityGroup.from_dict(row)
        if split_map.get(group.identity_group_id) != split:
            continue
        ages = {label.face_id: label.age for label in group.age_labels if label.face_id}
        paths = [(fid, _crop_path(fid, crops_dir)) for fid in group.faces]
        paths = [(fid, path) for fid, path in paths if path.exists()]
        if len(paths) < 2:
            continue
        items.extend(FaceItem(path, identity, ages.get(fid)) for fid, path in paths)
        identity += 1
    return items


class MTLFaceDataset(Dataset):
    def __init__(self, preprocess: PreprocessFn, split: str = "train") -> None:
        self.items = common_protocol_faces(split)
        self.preprocess = preprocess
        self.n_classes = 1 + max(item.identity for item in self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        item = self.items[index]
        image = cv2.imread(str(item.path))
        if image is None:
            raise RuntimeError(f"cannot read {item.path}")
        return torch.from_numpy(self.preprocess(image)), item.identity, item.age or -1


class FeatureSeparation(nn.Module):
    """MTLFace attention decomposition: x_age = A(x)x, x_id = x-x_age."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        hidden = max(64, dimension // 4)
        self.attention = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dimension),
            nn.Sigmoid(),
        )

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        age = embedding * self.attention(embedding)
        identity = nn.functional.normalize(embedding - age, dim=1)
        return identity, age


class MTLFaceCommonModel(nn.Module):
    def __init__(self, backbone: nn.Module, dimension: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.separation = FeatureSeparation(dimension)
        self.preprocess = getattr(backbone, "preprocess", None)
        self.input_size = getattr(backbone, "input_size", 112)

    def components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.separation(self.backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity, _ = self.components(x)
        return identity


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor]:  # noqa: ANN001
        return (-grad,)


class AgeHead(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(dimension, 256), nn.ReLU(inplace=True))
        self.year = nn.Linear(256, 101)
        self.group = nn.Linear(256, 7)
        self.register_buffer("years", torch.arange(101, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(x)
        return self.year(hidden), self.group(hidden)

    def loss(self, x: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
        valid = ages >= 0
        if not bool(valid.any()):
            return x.sum() * 0
        year_logits, group_logits = self(x[valid])
        years = ages[valid].float().clamp(0, 100)
        expected = (year_logits.softmax(dim=1) * self.years).sum(dim=1)
        groups = torch.as_tensor(
            [age_group(int(value)) for value in years.detach().cpu()], device=x.device
        )
        return nn.functional.mse_loss(expected, years) + nn.functional.cross_entropy(
            group_logits, groups
        )


def _embedding_dimension(backbone: nn.Module, device: torch.device) -> int:
    size = int(getattr(backbone, "input_size", 112))
    backbone.eval()
    with torch.no_grad():
        return int(backbone(torch.zeros(2, 3, size, size, device=device)).shape[1])


def train_mtlface_common(
    *,
    backbone_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    scope: str,
    seed: int,
    checkpoint: Path,
    device: torch.device,
) -> tuple[MTLFaceCommonModel, list[dict[str, float]]]:
    torch.manual_seed(seed)
    backbone = make_backbone(backbone_name, pretrained=True).to(device)
    dimension = _embedding_dimension(backbone, device)
    model = MTLFaceCommonModel(backbone, dimension).to(device)
    if scope != "full":
        _set_trainable(model.backbone, scope)
    dataset = MTLFaceDataset(_bb_prep(backbone))
    validation = ImagePairDataset("val", preprocess=_bb_prep(backbone))
    identity_head = MarginHead(dimension, dataset.n_classes, loss_type="cosface").to(device)
    age_head = AgeHead(dimension).to(device)
    adversary = AgeHead(dimension).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        [
            {"params": params, "lr": learning_rate},
            {"params": identity_head.parameters(), "lr": 1e-3},
            {"params": age_head.parameters(), "lr": 1e-3},
            {"params": adversary.parameters(), "lr": 1e-3},
        ]
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    history: list[dict[str, float]] = []
    best_auc = -np.inf
    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        totals = np.zeros(4, dtype=np.float64)
        seen = 0
        for images, identities, ages in loader:
            images = images.to(device)
            identities = identities.to(device)
            ages = ages.to(device)
            optimizer.zero_grad(set_to_none=True)
            id_embedding, age_embedding = model.components(images)
            identity_loss = nn.functional.cross_entropy(
                identity_head(id_embedding, identities), identities
            )
            explicit_age_loss = age_head.loss(age_embedding, ages)
            adversarial_age_loss = adversary.loss(_GradientReversal.apply(id_embedding), ages)
            loss = identity_loss + 0.001 * explicit_age_loss + 0.002 * adversarial_age_loss
            loss.backward()
            optimizer.step()
            count = len(images)
            totals += np.asarray(
                [loss.item(), identity_loss.item(), explicit_age_loss.item(), adversarial_age_loss.item()]
            ) * count
            seen += count
        auc = _val_auc(model, validation, device, batch_size)
        row = {
            "epoch": float(epoch),
            "loss": float(totals[0] / seen),
            "identity_loss": float(totals[1] / seen),
            "age_loss": float(totals[2] / seen),
            "domain_adaptation_loss": float(totals[3] / seen),
            "val_auc": float(auc),
        }
        history.append(row)
        log.info("MTLFace-CP epoch=%d loss=%.4f val_auc=%.4f", epoch, row["loss"], auc)
        if auc > best_auc:
            best_auc = auc
            best = {
                "backbone": {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()},
                "separation": {k: v.detach().cpu() for k, v in model.separation.state_dict().items()},
            }
    if best is None:
        raise RuntimeError("MTLFace-CP produced no checkpoint")
    model.backbone.load_state_dict(best["backbone"])
    model.separation.load_state_dict(best["separation"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "mtlface-common-v1",
            "backbone": backbone_name,
            "dimension": dimension,
            **best,
            "history": history,
            "seed": seed,
        },
        checkpoint,
    )
    return model.eval(), history


def cacon_nt_xent(
    first: torch.Tensor, second: torch.Tensor, aged: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """CACon three-view NT-Xent with both same-face views in the numerator."""
    if first.shape != second.shape or first.shape != aged.shape:
        raise ValueError("all CACon views must have the same shape")
    batch = first.shape[0]
    embeddings = nn.functional.normalize(torch.cat([first, second, aged]), dim=1)
    logits = embeddings @ embeddings.T / temperature
    eye = torch.eye(3 * batch, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(eye, -torch.inf)
    sample = torch.arange(batch, device=logits.device).repeat(3)
    positives = sample[:, None] == sample[None, :]
    positives &= ~eye
    numerator = torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return (denominator - numerator).mean()


def _augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = image.copy()
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    alpha = float(rng.uniform(0.8, 1.2))
    beta = float(rng.uniform(-15, 15))
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if rng.random() < 0.25:
        out = cv2.GaussianBlur(out, (3, 3), 0.6)
    return out


class CAConDataset(Dataset):
    def __init__(self, preprocess: PreprocessFn, aged_dir: Path, seed: int) -> None:
        self.items = common_protocol_faces("train")
        self.preprocess = preprocess
        self.aged_dir = aged_dir
        self.seed = seed
        missing = [item.path.stem for item in self.items if not (aged_dir / item.path.name).exists()]
        if missing:
            raise RuntimeError(
                f"CACon aged-view cache is incomplete ({len(missing)} missing); run prepare_cacon_cache"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = cv2.imread(str(self.items[index].path))
        aged = cv2.imread(str(self.aged_dir / self.items[index].path.name))
        if source is None or aged is None:
            raise RuntimeError(f"cannot read CACon item {self.items[index].path}")
        rng = np.random.default_rng(self.seed + index)
        first = torch.from_numpy(self.preprocess(_augment(source, rng)))
        second = torch.from_numpy(self.preprocess(_augment(source, rng)))
        third = torch.from_numpy(self.preprocess(_augment(aged, rng)))
        return first, second, third


class ProjectionHead(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.ReLU(inplace=True),
            nn.Linear(dimension, dimension),
            nn.ReLU(inplace=True),
            nn.Linear(dimension, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def train_cacon_common(
    *,
    backbone_name: str,
    aged_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    scope: str,
    seed: int,
    checkpoint: Path,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    torch.manual_seed(seed)
    backbone = make_backbone(backbone_name, pretrained=True).to(device)
    dimension = _embedding_dimension(backbone, device)
    if scope != "full":
        _set_trainable(backbone, scope)
    projection = ProjectionHead(dimension).to(device)
    dataset = CAConDataset(_bb_prep(backbone), aged_dir, seed)
    validation = ImagePairDataset("val", preprocess=_bb_prep(backbone))
    optimizer = torch.optim.Adam(
        [
            {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": learning_rate},
            {"params": projection.parameters(), "lr": 1e-3},
        ]
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    history: list[dict[str, float]] = []
    best_auc = -np.inf
    best = None
    for epoch in range(1, epochs + 1):
        backbone.train()
        projection.train()
        total = 0.0
        seen = 0
        for first, second, aged in loader:
            first, second, aged = first.to(device), second.to(device), aged.to(device)
            optimizer.zero_grad(set_to_none=True)
            merged = torch.cat([first, second, aged])
            projected = projection(backbone(merged))
            z1, z2, z3 = projected.chunk(3)
            loss = cacon_nt_xent(z1, z2, z3)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(first)
            seen += len(first)
        auc = _val_auc(backbone, validation, device, batch_size)
        row = {"epoch": float(epoch), "loss": total / seen, "val_auc": float(auc)}
        history.append(row)
        log.info("CACon-CP epoch=%d loss=%.4f val_auc=%.4f", epoch, row["loss"], auc)
        if auc > best_auc:
            best_auc = auc
            best = {k: v.detach().cpu() for k, v in backbone.state_dict().items()}
    if best is None:
        raise RuntimeError("CACon-CP produced no checkpoint")
    backbone.load_state_dict(best)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "cacon-common-v1",
            "backbone": backbone_name,
            "state_dict": best,
            "history": history,
            "seed": seed,
        },
        checkpoint,
    )
    return backbone.eval(), history
