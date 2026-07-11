"""Contrastive/triplet метрик-лернинг ТОЛЬКО на лицах по таргету лайк/просмотр.

Идея (research-эксперимент): учим визуальный энкодер разносить эмбеддинги лиц так, чтобы
близость в эмбеддинге отражала близость по НОРМАЛИЗОВАННОМУ лайк/просмотр:
  * триплет (anchor, positive, negative): positive — лицо с близким таргетом, negative — с далёким;
  * онлайн batch-all майнинг: в каждом батче перебираются все валидные тройки (много случайных
    сравнений по всем лицам);
  * без текста, только пиксели лица (CLIP ViT-B/32, верхние блоки дообучаются).

Затем на замороженных эмбеддингах — линейный readout, и per-POST оценка Spearman vs e_rate,
с GroupKFold(person_id) и негативным контролем. Гардрейл: только лица age>=18.

ЧЕСТНО: beauty-эксперимент показал, что внешность лица почти не связана с per-view вовлечённостью
(ρ≈0), поэтому triplet-цель не создаёт новый сигнал — ждём тот же скромный потолок. Это проверка
формы обучения, а не способ «выжать» несуществующую связь.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPModel

from age_gap.beauty import CLIP_MODEL, VIS_HIDDEN
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.engagement.features import ADULT_MIN_AGE

log = get_logger(__name__)


def _z(x: np.ndarray) -> np.ndarray:
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)


def build_face_table(platform: str = "vk") -> pd.DataFrame:
    """Одна строка на ВЗРОСЛОЕ usable-лицо: crop_path, post/person, таргет лайк/просмотр."""
    oof_p = data_path("data_dir", "processed", "engagement_oof.parquet")
    oof = pd.read_parquet(oof_p) if oof_p.exists() else pd.read_csv(oof_p.with_suffix(".csv.gz"))
    oof["month"] = pd.to_datetime(oof["ts_unix"], unit="s", utc=True).dt.month
    oof["t_like_pv"] = _z(np.log((oof["likes"] + 1) / (oof["views"].clip(lower=1) + 1)).to_numpy())
    post = oof.set_index(oof["post_id"].astype(str))[
        ["person_id", "owner_id", "month", "t_like_pv", "e_rate", "y_rate"]].to_dict("index")

    photo2post: dict[str, str] = {}
    for r in read_jsonl(data_path("data_dir", "raw", "posts.jsonl")):
        pid = str(r["post_id"])
        if pid in post:
            for ph in r.get("photos", []) or []:
                photo2post[ph["photo_id"]] = pid
    ages = {r["face_id"]: r for r in read_jsonl(data_path("data_dir", "interim", "face_genderage.jsonl"))}

    rows: list[dict[str, Any]] = []
    for f in read_jsonl(data_path("data_dir", "interim", "faces.jsonl")):
        if not f.get("is_usable") or not f.get("face_crop_path"):
            continue
        pid = photo2post.get(f.get("photo_id"))
        if pid is None:
            continue
        ga = ages.get(f["face_id"])
        if ga is None or float(ga["age_est"]) < ADULT_MIN_AGE:  # guardrail: только взрослые
            continue
        p = post[pid]
        rows.append({
            "face_id": f["face_id"], "crop_path": str(resolve_path(f["face_crop_path"])),
            "post_id": pid, "person_id": str(p["person_id"]), "owner_id": str(p["owner_id"]),
            "month": int(p["month"]) if p["month"] == p["month"] else 0,
            "target": float(p["t_like_pv"]), "e_rate": float(p["e_rate"]), "y_rate": float(p["y_rate"]),
            "age": float(ga["age_est"]), "gender": int(ga["gender"]),
        })
    df = pd.DataFrame(rows)
    log.info("Лиц (adult usable): %d в %d постах, %d персон",
             len(df), df["post_id"].nunique(), df["person_id"].nunique())
    return df


# ---------------------------------------------------------------- модель / лосс


class TripletEncoder(nn.Module):
    """CLIP-vision (верхние блоки дообучаются) -> L2-нормированный эмбеддинг лица."""

    def __init__(self, unfreeze_top: int = 2, dim: int = 128):
        super().__init__()
        clip = CLIPModel.from_pretrained(CLIP_MODEL)
        self.vision = clip.vision_model
        for p in self.vision.parameters():
            p.requires_grad_(False)
        for blk in self.vision.encoder.layers[-unfreeze_top:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in self.vision.post_layernorm.parameters():
            p.requires_grad_(True)
        self.head = nn.Sequential(
            nn.LayerNorm(VIS_HIDDEN), nn.Linear(VIS_HIDDEN, 256), nn.GELU(), nn.Linear(256, dim))

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        e = self.head(self.vision(pixel_values=px).pooler_output)
        return F.normalize(e, dim=-1)


def triplet_batch_all(emb: torch.Tensor, t: torch.Tensor, margin: float = 0.2,
                      pos_thr: float = 0.5, neg_thr: float = 1.5) -> torch.Tensor:
    """Batch-all ordinal triplet: близкие по таргету -> близко, далёкие -> далеко."""
    d = torch.cdist(emb, emb)                      # [B,B]
    tt = (t[:, None] - t[None, :]).abs()           # [B,B]
    b = emb.size(0)
    eye = torch.eye(b, device=emb.device, dtype=torch.bool)
    pos = (tt < pos_thr) & ~eye
    neg = tt > neg_thr
    diff = d[:, :, None] - d[:, None, :] + margin   # [B,B,B] = D_ij - D_ik + margin
    mask = pos[:, :, None] & neg[:, None, :]
    vals = torch.relu(diff)[mask]
    return vals.mean() if vals.numel() > 0 else emb.sum() * 0.0


# ---------------------------------------------------------------- обучение / оценка


def _emb_all(model: nn.Module, px: np.ndarray, device: torch.device, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(px), batch):
            x = torch.from_numpy(np.asarray(px[i:i + batch], dtype=np.float32)).to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


def _train_encoder(model: nn.Module, px: np.ndarray, t: np.ndarray, device: torch.device,
                   epochs: int, batch: int, lr: float, lr_backbone: float) -> None:
    heads = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("vision.")]
    back = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("vision.")]
    opt = torch.optim.AdamW([{"params": back, "lr": lr_backbone}, {"params": heads, "lr": lr}], weight_decay=0.01)
    dl = DataLoader(TensorDataset(torch.arange(len(px))), batch_size=batch, shuffle=True, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_backbone, lr],
                                                total_steps=epochs * max(1, len(dl)), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    tt = torch.from_numpy(t.astype(np.float32))
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for (idx,) in dl:
            x = torch.from_numpy(np.asarray(px[idx.numpy()], dtype=np.float32)).to(device)
            y = tt[idx].to(device)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = triplet_batch_all(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            stepped = scaler.step(opt)
            scaler.update()
            if stepped is not None:
                sched.step()
            tot += float(loss)
        log.info("  epoch %d/%d  triplet_loss=%.4f", ep + 1, epochs, tot / max(1, len(dl)))


def _post_readout(emb: np.ndarray, is_tr: np.ndarray, post_code: np.ndarray,
                  post_target: np.ndarray, post_is_tr: np.ndarray) -> np.ndarray:
    """Агрегируем эмбеддинги по посту (mean), Ridge на train-постах -> предсказание для всех постов."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    p = post_target.shape[0]
    sums = np.zeros((p, emb.shape[1]), dtype=np.float64)
    cnt = np.zeros(p, dtype=np.float64)
    np.add.at(sums, post_code, emb)
    np.add.at(cnt, post_code, 1.0)
    post_emb = sums / np.maximum(cnt[:, None], 1.0)

    sc = StandardScaler().fit(post_emb[post_is_tr])
    rg = Ridge(alpha=10.0).fit(sc.transform(post_emb[post_is_tr]), post_target[post_is_tr])
    return rg.predict(sc.transform(post_emb))


def run_cv(df: pd.DataFrame, px: np.ndarray, device: torch.device, n_splits: int = 3,
           unfreeze_top: int = 2, epochs: int = 6, batch: int = 128, lr: float = 3e-4,
           lr_backbone: float = 1e-5, shuffle_target: bool = False, seed: int = 0) -> dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    torch.manual_seed(seed)
    # пост-уровневые метки (для readout и оценки)
    post_code, post_ids = pd.factorize(df["post_id"].to_numpy())
    post_e_rate = df.groupby("post_id")["e_rate"].first().reindex(post_ids).to_numpy().copy()
    post_month = df.groupby("post_id")["month"].first().reindex(post_ids).to_numpy()

    face_target = df["target"].to_numpy().copy()
    if shuffle_target:  # ПОЛНЫЙ негативный контроль: ломаем и таргет энкодера, и метку readout/оценки
        rng = np.random.default_rng(seed)
        for m in np.unique(df["month"]):
            idx = np.where(df["month"].to_numpy() == m)[0]
            face_target[idx] = rng.permutation(face_target[idx])
        for m in np.unique(post_month):  # e_rate readout+eval тоже перемешиваем внутри месяца
            idx = np.where(post_month == m)[0]
            post_e_rate[idx] = rng.permutation(post_e_rate[idx])

    groups = df["person_id"].to_numpy()
    per_fold = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(df, groups=groups)):
        model = TripletEncoder(unfreeze_top=unfreeze_top).to(device)
        log.info("fold %d: train_faces=%d test_faces=%d", fold + 1, len(tr), len(te))
        _train_encoder(model, px[tr], face_target[tr], device, epochs, batch, lr, lr_backbone)
        emb = _emb_all(model, px, device)

        is_tr = np.zeros(len(df), dtype=bool)
        is_tr[tr] = True
        post_is_tr = np.zeros(len(post_ids), dtype=bool)  # пост train, если хоть одно train-лицо
        post_is_tr[post_code[tr]] = True
        pred = _post_readout(emb, is_tr, post_code, post_e_rate, post_is_tr)

        te_posts = ~post_is_tr
        sp = float(spearmanr(post_e_rate[te_posts], pred[te_posts]).statistic)
        # разделение топ/низ по предсказанию — реальная ставка на показ (e_rate)
        order = np.argsort(-pred[te_posts])
        k = min(50, te_posts.sum() // 2)
        et = post_e_rate[te_posts]
        per_fold.append({"spearman_post_e_rate": sp, "n_test_posts": int(te_posts.sum()),
                         "e_rate_top": float(et[order[:k]].mean()), "e_rate_bot": float(et[order[-k:]].mean())})
        log.info("fold %d: Spearman(readout, e_rate) test = %.4f", fold + 1, sp)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"per_fold": per_fold, "shuffled": shuffle_target,
            "mean_spearman": float(np.mean([f["spearman_post_e_rate"] for f in per_fold])),
            "std_spearman": float(np.std([f["spearman_post_e_rate"] for f in per_fold]))}
