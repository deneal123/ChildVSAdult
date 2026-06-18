"""Обучение кросс-возрастного adapter и применение его к эмбеддингам (MVP-3).

Поток (TODO §10 Stage 2):
    baseline-эмбеддинги (кэш) -> adapter -> age-invariant эмбеддинги
Обучение идёт на парах train-сплита; при непустом val считается ROC-AUC и сохраняется
лучший чекпойнт. Затем adapter применяется ко всем эмбеддингам и сохраняется новый .npz
для повторного бенчмарка (adapter vs baseline).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from age_gap.common.device import torch_device
from age_gap.common.io import PROJECT_ROOT, data_path
from age_gap.common.logging import get_logger
from age_gap.evaluation.metrics import roc_auc
from age_gap.models.adapter import MLPAdapter
from age_gap.models.embeddings import load_embeddings
from age_gap.training.dataset import PairEmbeddingDataset
from age_gap.training.losses import ContrastivePairLoss

log = get_logger(__name__)


def _val_metrics(
    model: MLPAdapter, val_ds: PairEmbeddingDataset, device: str, large_gap_threshold: int
) -> tuple[float, float]:
    """Вернуть (overall_auc, large_gap_auc) на val.

    large_gap_auc — разделение позитивов с age_gap >= порога от ВСЕХ негативов (прямо целит
    в северную звезду: верификация на больших возрастных разрывах).
    """
    if len(val_ds) == 0:
        return float("nan"), float("nan")
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for i in range(len(val_ds)):
            a, b, y, _w = val_ds[i]
            za, zb = model(a.unsqueeze(0).to(device)), model(b.unsqueeze(0).to(device))
            scores.append(float((za * zb).sum()))
    s = np.asarray(scores)
    y_arr = np.asarray(val_ds.labels)
    gaps = np.asarray(val_ds.gaps)
    overall = roc_auc(s, y_arr)
    # Подмножество: все негативы + позитивы с большим разрывом.
    mask = (y_arr == 0) | ((y_arr == 1) & (gaps >= large_gap_threshold))
    large = roc_auc(s[mask], y_arr[mask]) if mask.any() else float("nan")
    return overall, large


def _selection_score(overall: float, large: float) -> float:
    """Метрика отбора модели: приоритет большим разрывам, но без потери overall."""
    if np.isnan(overall):
        return float("nan")
    if np.isnan(large):
        return overall
    return 0.4 * overall + 0.6 * large


def train_adapter(
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 256,
    margin: float = 0.3,
    residual: bool = True,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    patience: int = 8,
    gap_weight: float = 2.0,
    large_gap_threshold: int = 15,
    seed: int = 42,
    ckpt_out: Path | None = None,
) -> Path:
    """Age-supervised обучение adapter: age-gap-взвешенный контрастив (большие разрывы весят
    больше) + отбор/early-stop по комбинированной val-метрике (overall + большие разрывы),
    чтобы не разменивать 25+ на look-alike негативы. Регуляризация: dropout + weight_decay."""
    ckpt_out = ckpt_out or data_path("models_dir", "adapter.pt")
    torch.manual_seed(seed)
    device = torch_device()

    train_ds = PairEmbeddingDataset(split="train", gap_weight=gap_weight)
    val_ds = PairEmbeddingDataset(split="val")
    if len(train_ds) == 0:
        raise RuntimeError("Пустой train-сплит: сначала запустите embed.py и split.py")

    model = MLPAdapter(residual=residual, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = ContrastivePairLoss(margin=margin)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    log.info(
        "Age-supervised обучение: device=%s, dropout=%.2f, wd=%.0e, gap_weight=%.1f, thr=%d",
        device,
        dropout,
        weight_decay,
        gap_weight,
        large_gap_threshold,
    )

    best_score = -1.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for a, b, y, w in loader:
            a, b, y, w = a.to(device), b.to(device), y.to(device), w.to(device)
            opt.zero_grad()
            loss = loss_fn(model(a), model(b), y, weights=w)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
        avg = total / len(train_ds)

        overall, large = _val_metrics(model, val_ds, device, large_gap_threshold)
        score = _selection_score(overall, large)
        improved = not np.isnan(score) and score > best_score + 1e-4
        if improved:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch % 5 == 0 or epoch == 1:
            log.info(
                "epoch %d/%d loss=%.4f val_auc=%.4f val_largegap=%.4f (best_score=%.4f)",
                epoch,
                epochs,
                avg,
                overall,
                large,
                best_score,
            )
        if not np.isnan(score) and no_improve >= patience:
            log.info(
                "Early stop на эпохе %d (комбинированная val не растёт %d эпох)", epoch, patience
            )
            break

    state = best_state if best_score >= 0 else model.state_dict()
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "residual": residual, "dim": 512}, ckpt_out)
    log.info("Adapter сохранён -> %s (best_combined_val=%.4f)", ckpt_out, best_score)
    return ckpt_out


def load_adapter(ckpt: Path | str) -> MLPAdapter:
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = MLPAdapter(residual=bool(data.get("residual", True)))
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model


def apply_adapter(
    ckpt: Path | str | None = None,
    embeddings_file: str | None = None,
    out_file: Path | None = None,
) -> Path:
    """Применить обученный adapter ко всем baseline-эмбеддингам и сохранить новый .npz."""
    ckpt = ckpt or data_path("models_dir", "adapter.pt")
    out_file = out_file or data_path("embeddings_cache_dir", "adapter_arcface.npz")

    device = torch_device()
    model = load_adapter(ckpt).to(device)
    embeddings = load_embeddings(embeddings_file)
    if not embeddings:
        raise RuntimeError("Нет baseline-эмбеддингов: сначала запустите embed.py")

    face_ids = list(embeddings.keys())
    mat = torch.from_numpy(np.asarray([embeddings[f] for f in face_ids], dtype=np.float32)).to(
        device
    )
    with torch.no_grad():
        adapted = model(mat).cpu().numpy().astype(np.float32)

    out_file = out_file if out_file.is_absolute() else PROJECT_ROOT / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_file, face_ids=np.asarray(face_ids, dtype=object), embeddings=adapted)
    log.info("Adapter-эмбеддинги: %d лиц -> %s", len(face_ids), out_file)
    return out_file
