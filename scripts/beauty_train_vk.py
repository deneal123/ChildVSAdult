"""VK-native beauty-модель: дообучение на ТВОИХ Likert-оценках (1..5) поверх SCUT-модели.

    ENV_FOR_DYNACONF=natural uv run python scripts/beauty_train_vk.py --ratings reports/rating/ratings.jsonl

Старт не с нуля, а от SCUT-beauty модели (она уже знает «красоту»), твои метки докручивают её
под твой вкус. Выдаёт:
  * zero-shot baseline: как SCUT-модель угадывает твои оценки БЕЗ дообучения;
  * 5-fold OOF после дообучения (выучила ли твой вкус);
  * SCUT-бенчмарк дообученной (разошёлся ли твой вкус с усреднённым человеческим).
Отдельно потом: scripts/beauty_eval_pairs.py на твоих 215 парах then/now (независимый тест).

Сохраняет data_beauty/weights/beauty_dinov2_vk.pt. Пишет metrics/beauty_vk_native.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from transformers import AutoImageProcessor

from age_gap.beauty import BACKBONES, BeautyRegressor, _train, load_scut, metrics, pick_device
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def _pixels(paths, proc, batch=64):
    """Небольшой набор (~1.6k) — можно материализовать пиксели целиком."""
    out = None
    for i in range(0, len(paths), batch):
        chunk = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        pv = proc(images=chunk, return_tensors="np")["pixel_values"].astype(np.float16)
        if out is None:
            out = np.zeros((len(paths), *pv.shape[1:]), dtype=np.float16)
        out[i:i + len(chunk)] = pv
    return out


@torch.no_grad()
def _score_px(model, px, device, batch=64):
    model.eval()
    o = []
    for i in range(0, len(px), batch):
        x = torch.from_numpy(np.asarray(px[i:i + batch], dtype=np.float32)).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            o.append(model(x).float().cpu().numpy())
    return np.concatenate(o)


@torch.no_grad()
def _score_scut(model, ds, proc, device, batch=64):
    model.eval()
    o = []
    for i in range(0, len(ds), batch):
        chunk = [im.convert("RGB") for im in ds[i:i + batch]["image"]]
        pv = torch.from_numpy(proc(images=chunk, return_tensors="np")["pixel_values"]).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            o.append(model(pv).float().cpu().numpy())
    return np.concatenate(o)


def _mk(backbone, unfreeze, init_sd, device):
    m = BeautyRegressor(backbone, unfreeze_top=unfreeze).to(device)
    if init_sd is not None:
        m.load_state_dict({k: v.to(device) for k, v in init_sd.items()})
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", default="reports/rating/ratings.jsonl")
    ap.add_argument("--init", default="data_beauty/weights/beauty_dinov2.pt", help="стартовые веса (SCUT-модель)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--unfreeze-vision", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=5e-6)
    args = ap.parse_args()

    device = pick_device()
    ck = torch.load(resolve_path(args.init), map_location="cpu", weights_only=False)
    backbone = ck.get("backbone", "dinov2")
    init_sd = ck["state_dict"]
    proc = AutoImageProcessor.from_pretrained(BACKBONES[backbone])

    rows = [r for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")]
    hd = data_path("data_dir", "interim", "faces_hires")
    rows = [r for r in rows if (hd / f"{r['face_id']}.jpg").exists()]
    paths = [hd / f"{r['face_id']}.jpg" for r in rows]
    y = np.array([float(r["score"]) for r in rows], dtype=np.float32)
    log.info("Твоих меток: %d | mean=%.2f std=%.2f", len(y), y.mean(), y.std())

    px = _pixels(paths, proc)

    # ---- zero-shot: SCUT-модель БЕЗ дообучения на твоих оценках -----------------------
    base = _mk(backbone, args.unfreeze_vision, init_sd, device)
    zs = _score_px(base, px, device) * float(ck.get("sd", 1.0)) + float(ck.get("mu", 0.0))
    zs_m = metrics(y, zs)
    log.info("ZERO-SHOT SCUT-модель на твоих метках: spearman=%.4f pearson=%.4f",
             zs_m["spearman"], zs_m["pearson"])
    del base
    torch.cuda.empty_cache()

    # ---- дообучение на твоих метках, 5-fold OOF ---------------------------------------
    oof = np.full(len(y), np.nan)
    for f, (tr, te) in enumerate(KFold(args.folds, shuffle=True, random_state=0).split(px)):
        rng = np.random.default_rng(f)
        perm = rng.permutation(tr)
        nv = max(1, int(0.15 * len(tr)))
        va, core = perm[:nv], perm[nv:]
        mu, sd = float(y[core].mean()), float(y[core].std()) + 1e-8
        m = _mk(backbone, args.unfreeze_vision, init_sd, device)
        from age_gap.beauty import _loader
        dl_tr = _loader(px[core], (y[core] - mu) / sd, args.batch, True)
        dl_va = _loader(px[va], (y[va] - mu) / sd, args.batch, False)
        _train(m, dl_tr, dl_va, (y[va] - mu) / sd, device, args.epochs, args.lr, args.lr_backbone, 0.05)
        oof[te] = _score_px(m, px[te], device) * sd + mu
        log.info("fold %d: spearman=%.4f", f + 1, float(spearmanr(y[te], oof[te]).statistic))
        del m
        torch.cuda.empty_cache()
    ft_m = metrics(y, oof)
    log.info("ДООБУЧЕНО (OOF): spearman=%.4f pearson=%.4f", ft_m["spearman"], ft_m["pearson"])

    # ---- финальная модель на всех твоих метках ----------------------------------------
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    nv = max(1, int(0.12 * len(y)))
    va, core = perm[:nv], perm[nv:]
    mu, sd = float(y[core].mean()), float(y[core].std()) + 1e-8
    final = _mk(backbone, args.unfreeze_vision, init_sd, device)
    from age_gap.beauty import _loader
    _train(final, _loader(px[core], (y[core] - mu) / sd, args.batch, True),
           _loader(px[va], (y[va] - mu) / sd, args.batch, False), (y[va] - mu) / sd,
           device, args.epochs, args.lr, args.lr_backbone, 0.05)
    wpath = resolve_path("data_beauty", "weights", "beauty_dinov2_vk.pt")
    wpath.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final.state_dict(), "mu": mu, "sd": sd,
                "unfreeze_vision": args.unfreeze_vision, "backbone": backbone}, wpath)
    log.info("VK-native модель сохранена: %s", wpath)

    # ---- SCUT-бенчмарк дообученной: разошёлся ли твой вкус с людьми --------------------
    ds, scut_y, _ = load_scut()
    sp = _score_scut(final, ds, proc, device) * sd + mu
    scut_sp = float(spearmanr(scut_y, sp).statistic)
    scut_pe = float(pearsonr(scut_y, sp)[0])
    log.info("SCUT-бенчмарк VK-native модели: spearman=%.4f pearson=%.4f", scut_sp, scut_pe)

    out = {
        "n_labels": int(len(y)), "label_mean": round(float(y.mean()), 3), "label_std": round(float(y.std()), 3),
        "zero_shot_scut_model_on_your_labels": {k: round(v, 4) for k, v in zs_m.items()},
        "finetuned_oof_on_your_labels": {k: round(v, 4) for k, v in ft_m.items()},
        "finetuned_scut_benchmark": {"spearman": round(scut_sp, 4), "pearson": round(scut_pe, 4)},
        "weights": wpath.name,
        "note": "независимый тест — scripts/beauty_eval_pairs.py на 215 парах then/now (baseline 0.688)",
    }
    dst = data_path("metrics_dir", "beauty_vk_native.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
