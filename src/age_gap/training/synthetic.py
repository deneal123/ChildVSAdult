"""Synthetic-ageing baseline: дообучение facenet на СИНТЕТИЧЕСКИХ парах (orig, aged(orig)).

Сравнение с обучением на РЕАЛЬНЫХ same-post парах (см. ablation): что лучше для cross-age —
реальные longitudinal пары или синтетическое старение. Позитивы — лицо и его «состаренная»
версия; негативы — лица других личностей. Отбор модели по РЕАЛЬНОМУ val (как и +pairs — честно).

Качество baseline зависит от aging-преобразования (``aging``): по умолчанию — прокси
(HeuristicAging), для публикуемого сравнения — реальная модель FRAN (``make_aging("fran")``).
См. models/aging.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, read_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import IdentityGroup
from age_gap.models.aging import AgingTransform, HeuristicAging
from age_gap.models.facenet import FaceNetBackbone, preprocess_bgr
from age_gap.training.finetune import (
    ImagePairDataset,
    _crop_path,
    _set_trainable,
    _val_auc,
)
from age_gap.training.losses import ContrastivePairLoss

log = get_logger(__name__)


def _train_faces(
    split: str = "train",
    crops_dir: str = "faces",
    face_ids: set[str] | None = None,
) -> list[tuple[Path, int]]:
    """Кропы лиц указанного сплита с кодом личности (для leakage-safe синтетики).

    ``crops_dir`` — подкаталог в data/interim (например ``faces_hires512`` для нативного
    разрешения). ``face_ids`` — необязательный allowlist (подмножество лиц).
    """
    gsplit = {
        r["identity_group_id"]: r["split"]
        for r in read_jsonl(data_path("splits_dir", "group_splits.jsonl"))
    }
    items: list[tuple[Path, int]] = []
    code = 0
    for row in read_jsonl(data_path("data_dir", "processed", "identity_groups.jsonl")):
        g = IdentityGroup.from_dict(row)
        if gsplit.get(g.identity_group_id) != split:
            continue
        for fid in g.faces:
            if face_ids is not None and fid not in face_ids:
                continue
            p = _crop_path(fid, crops_dir)
            if p.exists():
                items.append((p, code))
        code += 1
    return items


class SyntheticPairDataset(Dataset):
    """Позитивы (лицо, состаренное лицо) + кросс-личностные негативы (баланс 1:1)."""

    def __init__(
        self,
        aging: AgingTransform,
        split: str = "train",
        seed: int = 42,
        crops_dir: str = "faces",
        face_ids: set[str] | None = None,
    ) -> None:
        self.aging = aging
        faces = _train_faces(split, crops_dir, face_ids)
        rng = np.random.default_rng(seed)
        # items: (path_a, path_b|None, label). b=None -> позитив (состарить a).
        self._items: list[tuple[Path, Path | None, int]] = [(p, None, 1) for p, _ in faces]
        n = len(faces)
        for i in range(n):  # негативы: случайная пара разных личностей
            j = int(rng.integers(n))
            tries = 0
            while faces[j][1] == faces[i][1] and tries < 10:
                j = int(rng.integers(n))
                tries += 1
            if faces[j][1] != faces[i][1]:
                self._items.append((faces[i][0], faces[j][0], 0))
        self._seed = seed
        # Кэш состаренных позитивов: aging детерминирован по idx (rng=seed+idx), поэтому
        # картинка одинакова между эпохами — дорогую FRAN-генерацию делаем один раз на idx.
        self._aged_cache: dict[int, np.ndarray] = {}
        log.info("SyntheticPairDataset(split=%s): пар=%d (faces=%d)", split, len(self._items), n)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        pa, pb, y = self._items[idx]
        ima = cv2.imread(str(pa))
        assert ima is not None, f"не прочитан кроп: {pa}"
        if pb is None:  # позитив: вторая «фотография» — синтетически состаренная
            imb = self._aged_cache.get(idx)
            if imb is None:
                rng = np.random.default_rng(self._seed + idx)
                imb = self.aging(ima, rng)
                self._aged_cache[idx] = imb
        else:
            loaded = cv2.imread(str(pb))
            assert loaded is not None, f"не прочитан кроп: {pb}"
            imb = loaded
        ta = torch.from_numpy(preprocess_bgr(ima))
        tb = torch.from_numpy(preprocess_bgr(imb))
        return ta, tb, torch.tensor(float(y), dtype=torch.float32)


def train_synthetic(
    epochs: int = 10,
    lr: float = 3e-5,
    batch_size: int = 64,
    margin: float = 0.3,
    patience: int = 3,
    trainable_scope: str = "head",
    aging: AgingTransform | None = None,
    pretrained: str = "casia-webface",
    ckpt_out: Path | None = None,
    seed: int = 42,
    crops_dir: str = "faces",
    face_ids: set[str] | None = None,
) -> Path:
    """Дообучить facenet на синтетических парах; отбор по РЕАЛЬНОМУ val. Возвращает чекпойнт."""
    ckpt_out = ckpt_out or data_path("models_dir", "facenet_synthetic.pt")
    aging = aging or HeuristicAging(strength=1.0)
    torch.manual_seed(seed)
    device = torch_device()

    train_ds = SyntheticPairDataset(
        aging, split="train", seed=seed, crops_dir=crops_dir, face_ids=face_ids
    )
    val_ds = ImagePairDataset(split="val")  # отбор по реальным парам — как у +pairs
    if len(train_ds) == 0:
        raise RuntimeError("Нет лиц train-сплита для синтетики")

    backbone = FaceNetBackbone(pretrained=pretrained).to(device)
    if trainable_scope != "full":
        _set_trainable(backbone, trainable_scope)
    opt = torch.optim.Adam([p for p in backbone.parameters() if p.requires_grad], lr=lr)
    loss_fn = ContrastivePairLoss(margin=margin)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_auc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
    no_improve = 0
    for epoch in range(1, epochs + 1):
        backbone.train()
        total = 0.0
        for ta, tb, y in loader:
            ta, tb, y = ta.to(device), tb.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(backbone(ta), backbone(tb), y)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
        auc = _val_auc(backbone, val_ds, device, batch_size)
        if not np.isnan(auc) and auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
            no_improve = 0
            ckpt_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, "pretrained": pretrained}, ckpt_out)
        else:
            no_improve += 1
        log.info(
            "epoch %d/%d loss=%.4f val_auc=%.4f (best=%.4f)",
            epoch,
            epochs,
            total / len(train_ds),
            auc,
            best_auc,
        )
        if not np.isnan(auc) and no_improve >= patience:
            log.info("Early stop на эпохе %d", epoch)
            break

    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "pretrained": pretrained}, ckpt_out)
    log.info("Synthetic-baseline facenet -> %s (best_val_auc=%.4f)", ckpt_out, best_auc)
    return ckpt_out
