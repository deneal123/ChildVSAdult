"""Обучаемые GPU-модели «симпатии на показ»: bi-encoder и cross-encoder.

Мультимодальный скор поста из его КОНТЕНТА:
  * текст  — подпись поста (RU), энкодер ``cointegrated/rubert-tiny2`` (~29M), ДООБУЧАЕТСЯ;
  * лицо   — кроп лучшего ВЗРОСЛОГО лица, энкодер ``facebook/deit-tiny-patch16-224`` (~5.7M),
             ЗАМОРОЖЕН; его патч-токены кешируются один раз.

Две схемы слияния (в этом и весь смысл сравнения):
  * ``bi``    — две башни, ПОЗДНЕЕ слияние: текст и лицо кодируются независимо, взаимодействие
                только в голове-фьюзере (модальности «не видят» друг друга).
  * ``cross`` — РАННЕЕ слияние: [CLS] + токены текста + патчи лица в общий трансформер с
                cross-attention; модальности взаимодействуют на всех слоях.

Валидация ровно как у CPU-бустинга: та же ``GroupKFold(person_id)``, тот же таргет ``y_rate``
(остаток ставки-на-показ после охвата), те же метрики — числа прямо сопоставимы.

Гардрейлы: в vision идут ТОЛЬКО лица age>=18 (как в превью отчёта); несовершеннолетние-субъекты
уже исключены на этапе features. Кеши патчей и веса — локальные, в git не попадают.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from age_gap.common.io import data_path, read_jsonl
from age_gap.common.logging import get_logger
from age_gap.engagement.model import EvalResult, regression_metrics

log = get_logger(__name__)

TEXT_MODEL = "cointegrated/rubert-tiny2"
VISION_MODEL = "facebook/deit-tiny-patch16-224"
MAX_LEN = 96
IMG_TOKENS = 197  # deit-tiny: 1 cls + 196 патчей
VIS_DIM = 192


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- входы поста


def attach_captions(df: pd.DataFrame) -> list[str]:
    """post_id -> подпись поста из raw/posts.jsonl (в том же порядке, что строки df)."""
    cap: dict[str, str] = {}
    for r in read_jsonl(data_path("data_dir", "raw", "posts.jsonl")):
        cap[str(r["post_id"])] = (r.get("caption") or r.get("text") or "").strip()
    return [cap.get(str(p), "") for p in df["post_id"]]


@torch.no_grad()
def precompute_vision(df: pd.DataFrame, device: torch.device, batch: int = 64) -> np.ndarray:
    """Патч-токены замороженного deit-tiny по лучшему ВЗРОСЛОМУ лицу поста.

    Возвращает массив [N, 197, 192] fp16 в порядке строк df. Кешируется на диск;
    при совпадении порядка post_id читается из кеша. Пост без взрослого кропа -> нули.
    """
    # локальный импорт: тот же отбор кропа, что для превью (только лица age>=18)
    from age_gap.engagement.report import best_adult_face_paths

    ids = [str(p) for p in df["post_id"]]
    cache = data_path("data_dir", "cache", "vision_deit_tiny.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["post_ids"]) == ids:
            log.info("vision cache HIT: %s (%d постов)", cache.name, len(ids))
            return z["tokens"]

    proc = AutoImageProcessor.from_pretrained(VISION_MODEL)
    vmodel = AutoModel.from_pretrained(VISION_MODEL).to(device).eval()
    paths = best_adult_face_paths(set(ids))

    tokens = np.zeros((len(ids), IMG_TOKENS, VIS_DIM), dtype=np.float16)
    mask = np.zeros(len(ids), dtype=bool)
    buf_i: list[int] = []
    buf_img: list[Image.Image] = []

    def flush() -> None:
        if not buf_i:
            return
        inp = proc(images=buf_img, return_tensors="pt").to(device)
        out = vmodel(**inp).last_hidden_state  # [B,197,192]
        tokens[buf_i] = out.float().cpu().numpy().astype(np.float16)
        for i in buf_i:
            mask[i] = True
        buf_i.clear()
        buf_img.clear()

    for i, pid in enumerate(ids):
        p = paths.get(pid)
        if p is None:
            continue
        try:
            buf_img.append(Image.open(p).convert("RGB"))
            buf_i.append(i)
        except Exception:  # noqa: BLE001
            continue
        if len(buf_i) >= batch:
            flush()
    flush()

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, tokens=tokens, post_ids=np.array(ids), mask=mask)
    log.info("vision cache: кроп есть у %d/%d постов -> %s", int(mask.sum()), len(ids), cache.name)
    return tokens


@torch.no_grad()
def precompute_pixels(df: pd.DataFrame, batch: int = 128) -> np.ndarray:
    """Пиксели лучшего ВЗРОСЛОГО лица [N,3,224,224] fp16 для ЖИВОГО (дообучаемого) vision.

    Декодируем/нормируем один раз (image processor), чтобы не читать JPEG каждую эпоху.
    Пост без взрослого кропа -> нули.
    """
    from age_gap.engagement.report import best_adult_face_paths

    ids = [str(p) for p in df["post_id"]]
    cache = data_path("data_dir", "cache", "vision_pixels_224.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["post_ids"]) == ids:
            log.info("pixel cache HIT: %s (%d постов)", cache.name, len(ids))
            return z["pixels"]

    proc = AutoImageProcessor.from_pretrained(VISION_MODEL)
    paths = best_adult_face_paths(set(ids))
    pixels = np.zeros((len(ids), 3, 224, 224), dtype=np.float16)
    buf_i: list[int] = []
    buf_img: list[Image.Image] = []

    def flush() -> None:
        if not buf_i:
            return
        pv = proc(images=buf_img, return_tensors="np")["pixel_values"]
        pixels[buf_i] = pv.astype(np.float16)
        buf_i.clear()
        buf_img.clear()

    for i, pid in enumerate(ids):
        p = paths.get(pid)
        if p is None:
            continue
        try:
            buf_img.append(Image.open(p).convert("RGB"))
            buf_i.append(i)
        except Exception:  # noqa: BLE001
            continue
        if len(buf_i) >= batch:
            flush()
    flush()

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, pixels=pixels, post_ids=np.array(ids))
    log.info("pixel cache: %d/%d постов -> %s", len(ids), len(ids), cache.name)
    return pixels


def _tokenize(texts: list[str], tok: Any) -> tuple[np.ndarray, np.ndarray]:
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="np")
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)


# ---------------------------------------------------------------- модели


class VisionEncoder(nn.Module):
    """Лицо -> патч-токены [B,197,192].

    ``unfreeze_top == 0``: замороженный экстрактор, ``img`` уже кешированные токены — просто
    прокидываем их. ``unfreeze_top > 0``: живой deit-tiny, ``img`` = пиксели [B,3,224,224];
    нижние блоки заморожены, верхние ``unfreeze_top`` блоков и финальный LayerNorm обучаются.
    """

    def __init__(self, unfreeze_top: int = 0):
        super().__init__()
        self.live = unfreeze_top > 0
        if self.live:
            self.model = AutoModel.from_pretrained(VISION_MODEL)
            for p in self.model.parameters():
                p.requires_grad_(False)
            for blk in self.model.layers[-unfreeze_top:]:  # transformers 5.x: блоки в .layers
                for p in blk.parameters():
                    p.requires_grad_(True)
            for p in self.model.layernorm.parameters():
                p.requires_grad_(True)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.live:
            return self.model(pixel_values=img).last_hidden_state
        return img


class BiEncoder(nn.Module):
    """Две башни, позднее слияние: текст и лицо кодируются НЕЗАВИСИМО."""

    def __init__(self, d: int = 256, dropout: float = 0.2, unfreeze_vision: int = 0):
        super().__init__()
        self.text = AutoModel.from_pretrained(TEXT_MODEL)
        self.vis = VisionEncoder(unfreeze_vision)
        th = self.text.config.hidden_size
        self.tproj = nn.Sequential(nn.Linear(th, d), nn.GELU())
        self.vproj = nn.Sequential(nn.LayerNorm(VIS_DIM), nn.Linear(VIS_DIM, d), nn.GELU())
        self.head = nn.Sequential(
            nn.LayerNorm(4 * d), nn.Dropout(dropout), nn.Linear(4 * d, d), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(d, 1),
        )

    def forward(self, ids: torch.Tensor, am: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        t = self.text(input_ids=ids, attention_mask=am).last_hidden_state  # [B,L,th]
        tvec = (t * am.unsqueeze(-1)).sum(1) / am.sum(1, keepdim=True).clamp(min=1)  # mean-pool
        tvec = self.tproj(tvec)
        vtok = self.vis(img)
        vvec = self.vproj(vtok.mean(1))  # среднее по патчам — башни не пересекаются
        f = torch.cat([tvec, vvec, (tvec - vvec).abs(), tvec * vvec], dim=-1)
        return self.head(f).squeeze(-1)


class CrossEncoder(nn.Module):
    """Раннее слияние: токены текста и патчи лица в общий трансформер с cross-attention."""

    def __init__(self, d: int = 256, layers: int = 3, heads: int = 8, dropout: float = 0.1,
                 unfreeze_vision: int = 0):
        super().__init__()
        self.text = AutoModel.from_pretrained(TEXT_MODEL)
        self.vis = VisionEncoder(unfreeze_vision)
        th = self.text.config.hidden_size
        self.tproj = nn.Linear(th, d)
        self.vproj = nn.Sequential(nn.LayerNorm(VIS_DIM), nn.Linear(VIS_DIM, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.typ = nn.Parameter(torch.zeros(3, d))  # 0=cls, 1=text, 2=img
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.typ, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=4 * d, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.fuse = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, 1))

    def forward(self, ids: torch.Tensor, am: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        b = ids.size(0)
        t = self.tproj(self.text(input_ids=ids, attention_mask=am).last_hidden_state) + self.typ[1]
        v = self.vproj(self.vis(img)) + self.typ[2]
        cls = self.cls.expand(b, -1, -1) + self.typ[0]
        x = torch.cat([cls, t, v], dim=1)  # [B, 1+L+197, d]
        ones = torch.ones(b, 1, device=ids.device)
        vones = torch.ones(b, v.size(1), device=ids.device)
        pad = torch.cat([ones, am.float(), vones], dim=1)
        h = self.fuse(x, src_key_padding_mask=(pad == 0))
        return self.head(h[:, 0]).squeeze(-1)


def _build(arch: str, unfreeze_vision: int = 0) -> nn.Module:
    if arch == "bi":
        return BiEncoder(unfreeze_vision=unfreeze_vision)
    if arch == "cross":
        return CrossEncoder(unfreeze_vision=unfreeze_vision)
    raise ValueError(f"arch must be 'bi' or 'cross', got {arch!r}")


# ---------------------------------------------------------------- обучение


def _loader(ids: np.ndarray, am: np.ndarray, img: np.ndarray, y: np.ndarray,
            batch: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(ids), torch.from_numpy(am),
        torch.from_numpy(np.asarray(img, dtype=np.float32)), torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True, drop_last=False)


@torch.no_grad()
def _predict(model: nn.Module, dl: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for ids, am, img, _ in dl:
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            p = model(ids.to(device), am.to(device), img.to(device))
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def _rank_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Попарный логистический ranking-loss: пары в батче должны упорядочиться как y.

    Прямо давит на ранговые метрики (Spearman/NDCG), в отличие от чистого MSE.
    """
    dp = pred.unsqueeze(0) - pred.unsqueeze(1)
    dt = y.unsqueeze(0) - y.unsqueeze(1)
    sign = torch.sign(dt)
    mask = sign != 0
    loss = nn.functional.softplus(-sign * dp)
    denom = mask.sum().clamp(min=1)
    return (loss * mask).sum() / denom


def _param_groups(model: nn.Module, lr: float, lr_backbone: float, wd: float) -> list[dict]:
    """Предобученные энкодеры (text.*, vis.model.*) — маленький LR; новые головы — полный."""
    backbone, heads = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if name.startswith(("text.", "vis.model.")) else heads).append(p)
    return [
        {"params": backbone, "lr": lr_backbone, "weight_decay": wd},
        {"params": heads, "lr": lr, "weight_decay": wd},
    ]


def _val_persons(df: pd.DataFrame, tr: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Разбить train на core/val по ЛЮДЯМ (val не делит человека с core)."""
    persons = df.iloc[tr]["person_id"].astype(str).to_numpy()
    uniq = np.array(sorted(set(persons)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * frac)))
    vset = set(uniq[:n_val].tolist())
    in_val = np.array([p in vset for p in persons])
    return tr[~in_val], tr[in_val]


def _train_fold(model: nn.Module, dl_tr: DataLoader, dl_va: DataLoader, y_va: np.ndarray,
                device: torch.device, epochs: int, lr: float, wd: float,
                lr_backbone: float = 1e-4, rank_w: float = 1.0, patience: int = 3) -> nn.Module:
    groups = _param_groups(model, lr, lr_backbone, wd)
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=epochs * max(1, len(dl_tr)), pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_rho = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    bad = 0
    for ep in range(epochs):
        model.train()
        for ids, am, img, y in dl_tr:
            ids, am, img, y = ids.to(device), am.to(device), img.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(ids, am, img)
                loss = nn.functional.mse_loss(pred, y) + rank_w * _rank_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            stepped = scaler.step(opt)  # None, если AMP пропустил шаг из-за inf/nan
            scaler.update()
            if stepped is not None:
                sched.step()  # планировщик двигаем только за реальным шагом оптимизатора
        vp = _predict(model, dl_va, device)
        rho = spearmanr(y_va, vp).statistic
        rho = float(rho) if rho == rho else -np.inf
        log.info("  epoch %d/%d  val_spearman=%.4f", ep + 1, epochs, rho)
        if rho > best_rho:
            best_rho, bad = rho, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log.info("  ранняя остановка на epoch %d", ep + 1)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_deep(
    df: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    arch: str,
    device: torch.device,
    epochs: int = 12,
    batch: int = 32,
    lr: float = 3e-4,
    lr_backbone: float = 1e-4,
    wd: float = 0.01,
    rank_w: float = 1.0,
    unfreeze_vision: int = 0,
    val_frac: float = 0.15,
    seed: int = 0,
) -> EvalResult:
    """OOF-оценка bi/cross-энкодера на тех же сплитах, что и CPU-модель.

    ``unfreeze_vision > 0`` -> живой deit-tiny с дообучением верхних блоков (вход = пиксели);
    ``== 0`` -> замороженные кешированные патч-токены. ``rank_w`` добавляет ranking-loss к MSE.
    """
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    ids_all, am_all = _tokenize(attach_captions(df), tok)
    img_all = precompute_pixels(df) if unfreeze_vision > 0 else precompute_vision(df, device)
    tag = f"{arch}-encoder(gpu)" + (f"+ft{unfreeze_vision}" if unfreeze_vision > 0 else "")

    oof = np.full(len(df), np.nan, dtype=float)
    folds: list[dict[str, float]] = []
    for fold, (tr, te) in enumerate(splits):
        core, va = _val_persons(df, tr, val_frac, seed + fold)
        mu, sd = float(y[core].mean()), float(y[core].std()) + 1e-8  # стандартизация по core
        model = _build(arch, unfreeze_vision).to(device)
        log.info("[%s] фолд %d: core=%d val=%d test=%d", tag, fold + 1, len(core), len(va), len(te))
        dl_tr = _loader(ids_all[core], am_all[core], img_all[core], (y[core] - mu) / sd, batch, True)
        dl_va = _loader(ids_all[va], am_all[va], img_all[va], (y[va] - mu) / sd, batch, False)
        _train_fold(model, dl_tr, dl_va, y[va], device, epochs, lr, wd, lr_backbone, rank_w)

        dl_te = _loader(ids_all[te], am_all[te], img_all[te], y[te], batch, False)
        pred = _predict(model, dl_te, device) * sd + mu
        oof[te] = pred
        folds.append(regression_metrics(y[te], pred))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return EvalResult(tag, "y_rate", ["text", "face"], MAX_LEN + IMG_TOKENS, folds, oof, y)
