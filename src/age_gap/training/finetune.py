"""Дообучение face-бэкбона (facenet) на НАШИХ кросс-возрастных парах.

Честный эксперимент: тот же слабый бэкбон (а) замораживается → метрика на бенчмарке,
(б) дообучается на наших парах → метрика на бенчмарке. Градиент идёт в саму сеть (не в
adapter поверх замороженных эмбеддингов). Обучение на наших выровненных кропах (112→160).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.common.schemas import Pair
from age_gap.evaluation.metrics import roc_auc
from age_gap.models.backbones import make_backbone
from age_gap.models.facenet import preprocess_bgr
from age_gap.training.dataset import _pair_weight
from age_gap.training.losses import ContrastivePairLoss

log = get_logger(__name__)

# Тип препроцессинга: BGR-изображение (uint8 HxWx3) -> CHW float32 под конкретный backbone.
PreprocessFn = Callable[[np.ndarray], np.ndarray]


def _facenet_prep(img_bgr: np.ndarray) -> np.ndarray:
    return preprocess_bgr(img_bgr)


def _crop_path(face_id: str, crops_dir: str = "faces") -> Path:
    return resolve_path(str(data_path("data_dir", "interim", crops_dir, f"{face_id}.jpg")))


class ImagePairDataset(Dataset):
    """Пары изображений (наши кропы) с метками для дообучения backbone.

    ``preprocess`` — препроцессинг под целевой backbone (по умолчанию facenet).
    ``crops_dir`` — подкаталог в data/interim (например ``faces_mtcnn`` для MTCNN-выравнивания).
    """

    def __init__(
        self,
        split: str | None,
        pairs_file: str | None = None,
        gap_weight: float = 0.0,
        preprocess: PreprocessFn | None = None,
        crops_dir: str = "faces",
    ) -> None:
        pairs_file = pairs_file or str(data_path("data_dir", "processed", "pairs.jsonl"))
        self._prep: PreprocessFn = preprocess or _facenet_prep
        # (crop_a, crop_b, label, age_gap, weight)
        self._items: list[tuple[Path, Path, int, int, float]] = []
        for row in read_jsonl(pairs_file):
            p = Pair.from_dict(row)
            if split is not None and p.split != split:
                continue
            ca, cb = _crop_path(p.face_a, crops_dir), _crop_path(p.face_b, crops_dir)
            if ca.exists() and cb.exists():
                gap = p.age_gap if p.age_gap is not None else -1
                w = _pair_weight(p.label, p.age_gap, gap_weight)
                self._items.append((ca, cb, p.label, gap, w))
        log.info(
            "ImagePairDataset(split=%s, crops=%s): пар=%d (gap_weight=%.1f)",
            split,
            crops_dir,
            len(self._items),
            gap_weight,
        )

    def __len__(self) -> int:
        return len(self._items)

    @property
    def labels(self) -> list[int]:
        return [it[2] for it in self._items]

    @property
    def gaps(self) -> list[int]:
        return [it[3] for it in self._items]

    def __getitem__(self, idx: int):
        ca, cb, y, _gap, w = self._items[idx]
        ima, imb = cv2.imread(str(ca)), cv2.imread(str(cb))
        assert ima is not None and imb is not None, f"не прочитан кроп: {ca} / {cb}"
        ta = torch.from_numpy(self._prep(ima))
        tb = torch.from_numpy(self._prep(imb))
        return (
            ta,
            tb,
            torch.tensor(float(y), dtype=torch.float32),
            torch.tensor(float(w), dtype=torch.float32),
        )


def _bb_prep(backbone: torch.nn.Module) -> PreprocessFn:
    """Препроцессинг конкретного backbone (BGR-кроп -> CHW); fallback — facenet."""
    prep = getattr(backbone, "preprocess", None)
    if prep is None:
        return _facenet_prep
    return lambda im: prep(im, True)


def _val_auc(
    backbone: torch.nn.Module, ds: ImagePairDataset, device: str, batch_size: int
) -> float:
    if len(ds) == 0:
        return float("nan")
    backbone.eval()
    loader = DataLoader(ds, batch_size=batch_size)
    scores: list[float] = []
    with torch.no_grad():
        for ta, tb, _y, _w in loader:
            za, zb = backbone(ta.to(device)), backbone(tb.to(device))
            scores.extend((za * zb).sum(dim=-1).cpu().tolist())
    return roc_auc(np.asarray(scores), np.asarray(ds.labels))


def _set_trainable(backbone: torch.nn.Module, scope: str) -> None:
    """Разморозить только слои выбранного scope; остальное заморожено (против забывания).

    Наборы слоёв берутся из ``backbone.trainable_scopes`` (свои у facenet и iResNet).
    """
    bb: Any = backbone
    trainable = bb.trainable_scopes[scope]
    for name, param in bb.net.named_parameters():
        param.requires_grad = any(t in name for t in trainable)


def finetune(
    epochs: int = 8,
    lr: float = 1e-4,
    batch_size: int = 64,
    margin: float = 0.3,
    patience: int = 3,
    trainable_scope: str = "tail",
    gap_weight: float = 0.0,
    backbone_name: str = "facenet",
    crops_dir: str = "faces",
    ckpt_out: Path | None = None,
    seed: int = 42,
) -> Path:
    """Дообучить backbone на наших train-парах; отбор по val-AUC (наш val). Возвращает чекпойнт.

    backbone_name: "facenet" | "arcface_r50_casia" | "arcface_r100" (см. models/backbones.py).
    trainable_scope: "head" (мягко) | "tail" (агрессивнее) | "full".
    crops_dir: подкаталог кропов (например "faces_mtcnn" для MTCNN-выравнивания facenet).
    gap_weight>0 — age-anchor regularization: позитивы с большим возрастным разрывом весят больше.
    """
    ckpt_out = ckpt_out or data_path("models_dir", "facenet_finetuned.pt")
    torch.manual_seed(seed)
    device = torch_device()

    backbone = make_backbone(backbone_name, pretrained=True).to(device)
    prep = _bb_prep(backbone)
    train_ds = ImagePairDataset(
        split="train", gap_weight=gap_weight, preprocess=prep, crops_dir=crops_dir
    )
    val_ds = ImagePairDataset(split="val", preprocess=prep, crops_dir=crops_dir)
    if len(train_ds) == 0:
        raise RuntimeError("Пустой train-сплит: сначала постройте пары и сплит")

    if trainable_scope != "full":
        _set_trainable(backbone, trainable_scope)
    params = [p for p in backbone.parameters() if p.requires_grad]
    log.info(
        "Fine-tune %s: device=%s, trainable-параметров=%d тензоров",
        backbone_name,
        device,
        len(params),
    )
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = ContrastivePairLoss(margin=margin)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    meta = {"backbone": backbone_name, "crops_dir": crops_dir}
    best_auc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
    no_improve = 0
    for epoch in range(1, epochs + 1):
        backbone.train()
        total = 0.0
        for ta, tb, y, w in loader:
            ta, tb, y, w = ta.to(device), tb.to(device), y.to(device), w.to(device)
            opt.zero_grad()
            loss = loss_fn(backbone(ta), backbone(tb), y, weights=w)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
        avg = total / len(train_ds)
        auc = _val_auc(backbone, val_ds, device, batch_size)
        if not np.isnan(auc) and auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
            no_improve = 0
            # Сохраняем лучшее сразу — длинный GPU-прогон переживёт обрыв.
            ckpt_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, **meta}, ckpt_out)
        else:
            no_improve += 1
        log.info(
            "epoch %d/%d loss=%.4f val_auc=%.4f (best=%.4f)", epoch, epochs, avg, auc, best_auc
        )
        if not np.isnan(auc) and no_improve >= patience:
            log.info("Early stop на эпохе %d", epoch)
            break

    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, **meta}, ckpt_out)
    log.info("Fine-tuned %s сохранён -> %s (best_val_auc=%.4f)", backbone_name, ckpt_out, best_auc)
    return ckpt_out


def load_finetuned(ckpt: Path | str, device: str) -> torch.nn.Module:
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    # pretrained=False не создаёт margin-голову/logits, которые есть в сохранённом state_dict
    # от обучения; для эмбеддингов они не нужны -> грузим strict=False.
    name = data.get("backbone", "facenet")
    backbone = make_backbone(name, pretrained=False)
    backbone.load_state_dict(data["state_dict"], strict=False)
    backbone.crops_dir = data.get("crops_dir", "faces")  # type: ignore[assignment]
    return backbone.to(device).eval()


def evaluate_our_split(
    backbone: torch.nn.Module,
    device: str,
    split: str = "test",
    batch_size: int = 64,
    large_gap_threshold: int = 25,
) -> dict[str, float]:
    """Метрики на НАШЕМ cross-age сплите: overall + бакет больших разрывов (25+).

    Препроцессинг и подкаталог кропов берутся из самого backbone (его выравнивание/нормировка).
    """
    crops_dir = getattr(backbone, "crops_dir", "faces")
    ds = ImagePairDataset(
        split=split, preprocess=_bb_prep(backbone), crops_dir=crops_dir
    )
    if len(ds) == 0:
        return {}
    backbone.eval()
    loader = DataLoader(ds, batch_size=batch_size)
    scores: list[float] = []
    with torch.no_grad():
        for ta, tb, _y, _w in loader:
            za, zb = backbone(ta.to(device)), backbone(tb.to(device))
            scores.extend((za * zb).sum(dim=-1).cpu().tolist())
    s = np.asarray(scores)
    y = np.asarray(ds.labels)
    gaps = np.asarray(ds.gaps)
    mask = (y == 0) | ((y == 1) & (gaps >= large_gap_threshold))
    return {
        "overall_auc": roc_auc(s, y),
        "large_gap_auc": roc_auc(s[mask], y[mask]) if mask.any() else float("nan"),
    }
