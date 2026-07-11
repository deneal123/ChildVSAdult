"""Обучаемый предиктор привлекательности лица по ПИКСЕЛЯМ (research-эксперимент).

Мотивация — аудит показал, что предсказывать красоту из ArcFace-identity эмбеддингов
и per-post усреднения бессмысленно (см. docs). Здесь честная постановка:

  * метка — РЕАЛЬНАЯ человеческая оценка красоты (SCUT-FBP5500: 5500 лиц, 1..5, 60 оценщиков);
  * энкодер — общий визуальный CLIP ViT-B/32 (а не identity-модель), верхние блоки дообучаются;
  * протокол — 5-fold CV, метрика Pearson/Spearman vs человеческие оценки (осмысленный потолок ~0.9).

Обученная модель затем применяется к VK-лицам (scripts/beauty_apply.py) — там вовлечённость
становится downstream-проверкой «красота -> реакция», а не шумной меткой.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoImageProcessor, CLIPModel

from age_gap.common.io import resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)

CLIP_MODEL = "openai/clip-vit-base-patch32"
SCUT_DATASET = "ljnlonoljpiljm/scut-fbp5500-v2-facial-beauty-scores"
VIS_HIDDEN = 768


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "pearson": float(pearsonr(y, p)[0]),
        "spearman": float(spearmanr(y, p).statistic),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
    }


# ---------------------------------------------------------------- данные (SCUT)


def load_scut() -> tuple[list, np.ndarray, dict[str, np.ndarray]]:
    """PIL-изображения + человеческие оценки красоты + fairface-конфаунды."""
    from datasets import load_dataset

    ds = load_dataset(SCUT_DATASET, split="train")
    scores = np.asarray(ds["score"], dtype=np.float32)
    meta = {
        "age": np.asarray(ds["fairface_predicted_age"]),
        "gender": np.asarray(ds["fairface_predicted_gender"]),  # 0=Male,1=Female
        "race": np.asarray(ds["fairface_predicted_race"]),
    }
    log.info("SCUT-FBP5500: %d лиц, score %.2f..%.2f (mean %.2f)",
             len(scores), scores.min(), scores.max(), scores.mean())
    return ds, scores, meta


def precompute_pixels(ds: Any, cache_name: str = "scut_clip_pixels.npz", batch: int = 128) -> np.ndarray:
    """CLIP pixel_values [N,3,224,224] fp16 — один раз, чтобы не гонять препроцессор каждую эпоху."""
    cache = resolve_path("data_beauty", "cache", cache_name)
    if cache.exists():
        z = np.load(cache)
        if len(z["pixels"]) == len(ds):
            log.info("pixel cache HIT: %s", cache.name)
            return z["pixels"]

    proc = AutoImageProcessor.from_pretrained(CLIP_MODEL)
    out = np.zeros((len(ds), 3, 224, 224), dtype=np.float16)
    imgs = ds["image"]
    for i in range(0, len(ds), batch):
        chunk = [im.convert("RGB") for im in imgs[i:i + batch]]
        pv = proc(images=chunk, return_tensors="np")["pixel_values"]
        out[i:i + len(chunk)] = pv.astype(np.float16)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, pixels=out)
    log.info("pixel cache: %d -> %s", len(ds), cache.name)
    return out


def precompute_pixels_aligned(ds: Any, cache_name: str = "scut_aligned112_pixels.npz",
                              device_str: str | None = "cuda", batch: int = 128) -> np.ndarray:
    """SCUT, прогнанный через НАШ пайплайн: detect -> norm_crop 112 -> CLIP 224.

    Домен-матчинг под VK: у VK на диске только 112px identity-выровненные кропы, поэтому
    beauty-модель для применения к VK надо учить на так же выровненном SCUT, а не на
    свободных 350px кропах. Лицо не найдено -> ресайз всего кадра к 112 (запасной путь).
    """
    cache = resolve_path("data_beauty", "cache", cache_name)
    if cache.exists():
        z = np.load(cache)
        if len(z["pixels"]) == len(ds):
            log.info("aligned pixel cache HIT: %s", cache.name)
            return z["pixels"]

    import cv2
    from PIL import Image

    from age_gap.preprocessing.crop_align import align_face
    from age_gap.preprocessing.detect import FaceDetector

    proc = AutoImageProcessor.from_pretrained(CLIP_MODEL)
    det = FaceDetector(device=device_str)
    crops: list[np.ndarray] = []
    n_det = 0
    for im in ds["image"]:
        bgr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = det.detect(bgr)
        if faces:
            f = max(faces, key=lambda d: d.area)
            crop = align_face(bgr, f.kps, 112)
            n_det += 1
        else:
            crop = cv2.resize(bgr, (112, 112))
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    out = np.zeros((len(crops), 3, 224, 224), dtype=np.float16)
    for i in range(0, len(crops), batch):
        chunk = [Image.fromarray(c) for c in crops[i:i + batch]]
        out[i:i + len(chunk)] = proc(images=chunk, return_tensors="np")["pixel_values"].astype(np.float16)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, pixels=out, n_detected=n_det)
    log.info("aligned cache: детекция у %d/%d -> %s", n_det, len(crops), cache.name)
    return out


def clip_pixels_from_crops(paths: list, batch: int = 256) -> np.ndarray:
    """Готовые 112px кропы (VK) -> CLIP pixel_values [N,3,224,224] fp16. None-путь -> нули."""
    from PIL import Image

    proc = AutoImageProcessor.from_pretrained(CLIP_MODEL)
    out = np.zeros((len(paths), 3, 224, 224), dtype=np.float16)
    buf_i: list[int] = []
    buf_im: list[Any] = []

    def flush() -> None:
        if not buf_i:
            return
        pv = proc(images=buf_im, return_tensors="np")["pixel_values"]
        out[buf_i] = pv.astype(np.float16)
        buf_i.clear()
        buf_im.clear()

    for i, p in enumerate(paths):
        if p is None:
            continue
        try:
            buf_im.append(Image.open(p).convert("RGB"))
            buf_i.append(i)
        except Exception:  # noqa: BLE001
            continue
        if len(buf_i) >= batch:
            flush()
    flush()
    return out


# ---------------------------------------------------------------- модель


class CLIPBeauty(nn.Module):
    """CLIP ViT-B/32 vision + регрессионная голова. Верхние ``unfreeze_top`` блоков дообучаются."""

    def __init__(self, unfreeze_top: int = 4, dropout: float = 0.3):
        super().__init__()
        clip = CLIPModel.from_pretrained(CLIP_MODEL)
        self.vision = clip.vision_model
        for p in self.vision.parameters():
            p.requires_grad_(False)
        if unfreeze_top > 0:
            for blk in self.vision.encoder.layers[-unfreeze_top:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for p in self.vision.post_layernorm.parameters():
                p.requires_grad_(True)
        self.frozen = unfreeze_top == 0
        self.head = nn.Sequential(
            nn.LayerNorm(VIS_HIDDEN), nn.Dropout(dropout), nn.Linear(VIS_HIDDEN, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 1),
        )

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            return self.vision(pixel_values=pixel_values).pooler_output

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(pixel_values)).squeeze(-1)


# ---------------------------------------------------------------- обучение / оценка


def _loader(px: np.ndarray, y: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.asarray(px, dtype=np.float32)), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True)


@torch.no_grad()
def _predict(model: nn.Module, dl: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for px, _ in dl:
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out.append(model(px.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def _train(model: nn.Module, dl_tr: DataLoader, dl_va: DataLoader, y_va: np.ndarray,
           device: torch.device, epochs: int, lr: float, lr_backbone: float, wd: float,
           patience: int = 3) -> nn.Module:
    heads, backbone = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (backbone if name.startswith("vision.") else heads).append(p)
    opt = torch.optim.AdamW([
        {"params": backbone, "lr": lr_backbone, "weight_decay": wd},
        {"params": heads, "lr": lr, "weight_decay": wd},
    ])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr_backbone, lr], total_steps=epochs * max(1, len(dl_tr)), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for px, y in dl_tr:
            px, y = px.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = nn.functional.smooth_l1_loss(model(px), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            stepped = scaler.step(opt)
            scaler.update()
            if stepped is not None:
                sched.step()
        pc = float(pearsonr(y_va, _predict(model, dl_va, device))[0])
        log.info("  epoch %d/%d  val_pearson=%.4f", ep + 1, epochs, pc)
        if pc > best:
            best, bad = pc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def kfold_eval(px: np.ndarray, y: np.ndarray, device: torch.device, n_splits: int = 5,
               unfreeze_top: int = 4, epochs: int = 8, batch: int = 32,
               lr: float = 3e-4, lr_backbone: float = 1e-5, wd: float = 0.05,
               val_frac: float = 0.15, seed: int = 0) -> dict[str, Any]:
    """Стандартный SCUT 5-fold. Возвращает per-fold метрики + OOF-предсказания."""
    from sklearn.model_selection import KFold

    torch.manual_seed(seed)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    folds: list[dict[str, float]] = []
    for fold, (tr, te) in enumerate(kf.split(px)):
        rng = np.random.default_rng(seed + fold)
        perm = rng.permutation(tr)
        n_val = max(1, int(len(tr) * val_frac))
        va, core = perm[:n_val], perm[n_val:]
        mu, sd = float(y[core].mean()), float(y[core].std()) + 1e-8
        model = CLIPBeauty(unfreeze_top=unfreeze_top).to(device)
        log.info("fold %d: core=%d val=%d test=%d", fold + 1, len(core), len(va), len(te))
        dl_tr = _loader(px[core], (y[core] - mu) / sd, batch, True)
        dl_va = _loader(px[va], (y[va] - mu) / sd, batch, False)
        _train(model, dl_tr, dl_va, (y[va] - mu) / sd, device, epochs, lr, lr_backbone, wd)
        pred = _predict(model, _loader(px[te], y[te], batch, False), device) * sd + mu
        oof[te] = pred
        folds.append(metrics(y[te], pred))
        log.info("fold %d: pearson=%.4f spearman=%.4f mae=%.3f",
                 fold + 1, folds[-1]["pearson"], folds[-1]["spearman"], folds[-1]["mae"])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    agg = {k: (float(np.mean([f[k] for f in folds])), float(np.std([f[k] for f in folds]))) for k in folds[0]}
    return {"per_fold": folds, "mean_std": {k: {"mean": m, "std": s} for k, (m, s) in agg.items()},
            "oof_overall": metrics(y, oof), "oof": oof}
