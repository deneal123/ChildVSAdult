"""Дизайн пользователя: обучить DINOv2 на natural с таргетом-формулой вовлечённости (0..5),
провалидировать на SCUT (человеческая красота), протестировать на парной разметке.

    ENV_FOR_DYNACONF=natural uv run python scripts/engagement_dinov2.py

Таргет = per-view композит e_rate (лайки/комменты/репосты на показ), нормированный в [0,5] по
перцентилю (совместимо со шкалой SCUT). Обучаем DINOv2 (верхние блоки) на лицах natural,
person-grouped сплит. Затем:
  * validation: скор модели на лицах SCUT vs человеческая оценка -> Spearman/Pearson;
  * test: pairwise-accuracy на твоих парах (then/now лица).
Пишет metrics/engagement_dinov2.json. Веса/кеши локальные.

Заметка: ранее показано beauty↔e_rate=0.006 -> низкая SCUT-корреляция ожидаема; здесь это
измеряется в рамках именно этого пайплайна.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

from age_gap import beauty, contrastive
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


class CropDS(Dataset):
    def __init__(self, paths, y, proc):
        self.paths, self.y, self.proc = paths, y.astype(np.float32), proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        pv = self.proc(images=im, return_tensors="np")["pixel_values"][0]
        return torch.from_numpy(pv), self.y[i]


@torch.no_grad()
def _score_pil_slices(model, ds, proc, device, batch=128):
    """Стриминговый скоринг HF-датасета по срезам — без загрузки всех пикселей в RAM."""
    model.eval()
    out = []
    for i in range(0, len(ds), batch):
        chunk = [im.convert("RGB") for im in ds[i:i + batch]["image"]]
        pv = torch.from_numpy(proc(images=chunk, return_tensors="np")["pixel_values"]).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out.append(model(pv).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--unfreeze-vision", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--pairs", default="reports/rating/pairs.jsonl")
    args = ap.parse_args()

    device = beauty.pick_device()
    proc = AutoImageProcessor.from_pretrained(beauty.BACKBONES[args.backbone])

    # ---- natural: per-face таблица + hi-res путь + таргет-формула в [0,5] --------------
    df = contrastive.build_face_table("vk").drop_duplicates("face_id").reset_index(drop=True)
    hd = data_path("data_dir", "interim", "faces_hires")
    df["path"] = [hd / f"{fid}.jpg" for fid in df["face_id"]]
    df = df[[p.exists() for p in df["path"]]].reset_index(drop=True)
    df["target"] = df["e_rate"].rank(pct=True).to_numpy() * 5.0  # 0..5, совместимо со SCUT
    log.info("natural лиц: %d, персон: %d", len(df), df["person_id"].nunique())

    # person-grouped split 85/15
    persons = df["person_id"].to_numpy()
    rng = np.random.default_rng(0)
    uniq = rng.permutation(np.unique(persons))
    val_p = set(uniq[: max(1, int(0.15 * len(uniq)))].tolist())
    is_val = np.array([p in val_p for p in persons])
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]

    model = beauty.BeautyRegressor(args.backbone, unfreeze_top=args.unfreeze_vision).to(device)
    mu, sd = float(df["target"].iloc[tr].mean()), float(df["target"].iloc[tr].std()) + 1e-8
    dl_tr = DataLoader(CropDS(df["path"].iloc[tr].tolist(), (df["target"].iloc[tr].to_numpy() - mu) / sd, proc),
                       batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    dl_va = DataLoader(CropDS(df["path"].iloc[va].tolist(), (df["target"].iloc[va].to_numpy() - mu) / sd, proc),
                       batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    heads = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("vision.")]
    back = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("vision.")]
    opt = torch.optim.AdamW([{"params": back, "lr": args.lr_backbone}, {"params": heads, "lr": args.lr}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_backbone, args.lr],
                                                total_steps=args.epochs * max(1, len(dl_tr)), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    yva = df["target"].iloc[va].to_numpy()
    best, best_state, bad = -np.inf, None, 0
    for ep in range(args.epochs):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = nn.functional.smooth_l1_loss(model(x), y)
            scaler.scale(loss).backward()
            stepped = scaler.step(opt)
            scaler.update()
            if stepped is not None:
                sched.step()
        pv = _score_pixels_dl(model, dl_va, device) * sd + mu
        rho = float(spearmanr(yva, pv).statistic)
        log.info("epoch %d/%d  val_spearman(engagement)=%.4f", ep + 1, args.epochs, rho)
        if rho > best:
            best, bad = rho, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    if best_state:
        model.load_state_dict(best_state)
    # чекпоинт, чтобы не потерять обучение
    wpath = resolve_path("data_beauty", "weights", "engagement_dinov2.pt")
    wpath.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "unfreeze_vision": args.unfreeze_vision, "backbone": args.backbone}, wpath)
    log.info("Модель сохранена: %s", wpath)

    # ---- VALIDATION на SCUT: скор модели vs человеческая красота (стриминг) -------------
    ds, scut_y, _ = beauty.load_scut()
    scut_pred = _score_pil_slices(model, ds, proc, device)
    scut_sp = float(spearmanr(scut_y, scut_pred).statistic)
    scut_pe = float(pearsonr(scut_y, scut_pred)[0])
    log.info("SCUT validation: Spearman=%.4f Pearson=%.4f", scut_sp, scut_pe)

    # ---- TEST на парах пользователя (then/now лица) ------------------------------------
    rows = [r for r in read_jsonl(resolve_path(args.pairs)) if r.get("winner")]
    ids = sorted({r["a"] for r in rows} | {r["b"] for r in rows})
    tn = resolve_path("data", "interim", "faces_hires")  # пары — из then/now
    pth = [tn / f"{i}.jpg" for i in ids]
    sc = beauty.score_paths(model, [p if p.exists() else None for p in pth], device, 0.0, 1.0, args.backbone)
    smap = dict(zip(ids, sc, strict=True))
    ok = [r for r in rows if np.isfinite(smap[r["a"]]) and np.isfinite(smap[r["b"]])]
    acc = float(np.mean([1.0 if (smap[r["a"]] > smap[r["b"]]) == (r["winner"] == r["a"]) else 0.0
                         for r in ok if smap[r["a"]] != smap[r["b"]]]))

    out = {
        "backbone": args.backbone, "target": "e_rate percentile *5 (audience-sympathy composite)",
        "n_natural_faces": int(len(df)), "n_persons": int(df["person_id"].nunique()),
        "val_spearman_engagement": round(best, 4),
        "scut_validation": {"spearman_vs_human_beauty": round(scut_sp, 4), "pearson": round(scut_pe, 4)},
        "pairs_test": {"pairwise_accuracy": round(acc, 4), "n_pairs": len(ok)},
        "baseline_scut_model_pairwise_acc": 0.688,
        "note": "модель обучена ПРЕДСКАЗЫВАТЬ вовлечённость; SCUT/pairs проверяют, совпадает ли это с красотой",
    }
    dst = data_path("metrics_dir", "engagement_dinov2.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("val_spearman_engagement", "scut_validation", "pairs_test")},
                     ensure_ascii=False, indent=2))


@torch.no_grad()
def _score_pixels_dl(model, dl, device):
    model.eval()
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out.append(model(x.to(device)).float().cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    main()
