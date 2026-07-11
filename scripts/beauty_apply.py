"""Применить обученную beauty-модель к VK-лицам и проверить связь красоты с вовлечённостью.

    ENV_FOR_DYNACONF=natural uv run python scripts/beauty_apply.py

Красота предсказывается по ПИКСЕЛЯМ каждого ВЗРОСЛОГО лица (age>=18, guardrail), затем
агрегируется по посту (max/mean) и коррелируется с per-view вовлечённостью. Это downstream-
проверка «красота -> реакция», где красота получена из модели на человеческих оценках,
а не из самих лайков. Домен-матч: применяем веса, обученные на 112-выровненном SCUT.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from age_gap import beauty
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.engagement.features import ADULT_MIN_AGE

log = get_logger(__name__)


@torch.no_grad()
def _score(model, pixels: np.ndarray, device, mu: float, sd: float, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(pixels), batch):
        px = torch.from_numpy(np.asarray(pixels[i:i + batch], dtype=np.float32)).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out.append(model(px).float().cpu().numpy())
    return np.concatenate(out) * sd + mu


def _corr(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    m = np.isfinite(a) & np.isfinite(b)
    return {"spearman": float(spearmanr(a[m], b[m]).statistic),
            "pearson": float(pearsonr(a[m], b[m])[0]), "n": int(m.sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None, help="по умолчанию data_beauty/weights/clip_beauty_aligned.pt")
    args = ap.parse_args()

    device = beauty.pick_device()
    wpath = resolve_path(args.weights) if args.weights else resolve_path("data_beauty", "weights", "beauty_dinov2.pt")
    ckpt = torch.load(wpath, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "clip")
    model = beauty.BeautyRegressor(backbone, unfreeze_top=int(ckpt["unfreeze_vision"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    mu, sd = float(ckpt["mu"]), float(ckpt["sd"])
    log.info("Веса: %s (backbone=%s mu=%.3f sd=%.3f)", wpath.name, backbone, mu, sd)

    # посты с таргетом (уже guardrailed: без несовершеннолетних-субъектов)
    oof_p = data_path("data_dir", "processed", "engagement_oof.parquet")
    oof = pd.read_parquet(oof_p) if oof_p.exists() else pd.read_csv(oof_p.with_suffix(".csv.gz"))
    keep_posts = set(oof["post_id"].astype(str))

    # взрослые usable лица этих постов
    photo2post = {}
    for r in read_jsonl(data_path("data_dir", "raw", "posts.jsonl")):
        if str(r["post_id"]) in keep_posts:
            for ph in r.get("photos", []) or []:
                photo2post[ph["photo_id"]] = str(r["post_id"])
    ages = {r["face_id"]: r for r in read_jsonl(data_path("data_dir", "interim", "face_genderage.jsonl"))}

    hires_dir = data_path("data_dir", "interim", "faces_hires")
    face_post: list[str] = []
    face_path: list = []
    face_gender: list = []
    for f in read_jsonl(data_path("data_dir", "interim", "faces.jsonl")):
        if not f.get("is_usable"):
            continue
        pid = photo2post.get(f.get("photo_id"))
        if pid is None:
            continue
        ga = ages.get(f["face_id"])
        if ga is None or float(ga["age_est"]) < ADULT_MIN_AGE:  # guardrail: только взрослые
            continue
        hp = hires_dir / f"{f['face_id']}.jpg"
        if not hp.exists():  # используем hi-res кропы; без них лицо пропускаем
            continue
        face_post.append(pid)
        face_path.append(hp)
        face_gender.append(int(ga["gender"]))
    log.info("Взрослых лиц к оценке (hi-res): %d в %d постах", len(face_path), len(set(face_post)))

    pixels = beauty.clip_pixels_from_crops(face_path, backbone=backbone)
    beauty_face = _score(model, pixels, device, mu, sd)

    fdf = pd.DataFrame({"post_id": face_post, "beauty": beauty_face, "gender": face_gender})
    agg = fdf.groupby("post_id").agg(beauty_max=("beauty", "max"), beauty_mean=("beauty", "mean"),
                                     n_face=("beauty", "size")).reset_index()
    m = oof.merge(agg, on="post_id", how="inner")
    like_pm = m["likes"] / m["views"].clip(lower=1)

    corr = {
        "beauty_face_scut_range": [round(float(beauty_face.min()), 2), round(float(beauty_face.max()), 2),
                                   round(float(beauty_face.mean()), 2)],
        "beauty_max_vs_y_rate": _corr(m["beauty_max"].to_numpy(), m["y_rate"].to_numpy()),
        "beauty_mean_vs_y_rate": _corr(m["beauty_mean"].to_numpy(), m["y_rate"].to_numpy()),
        "beauty_max_vs_e_rate": _corr(m["beauty_max"].to_numpy(), m["e_rate"].to_numpy()),
        "beauty_max_vs_like_per_view": _corr(m["beauty_max"].to_numpy(), like_pm.to_numpy()),
        "beauty_max_vs_log_likes": _corr(m["beauty_max"].to_numpy(), np.log1p(m["likes"]).to_numpy()),
        "beauty_max_vs_age": _corr(m["beauty_max"].to_numpy(), m["age_est_median"].to_numpy()),
        "beauty_max_vs_share_female": _corr(m["beauty_max"].to_numpy(), m["share_female_adult"].to_numpy()),
    }

    # топ/низ по красоте: реальная ставка на показ
    ms = m.assign(like_pm=like_pm).sort_values("beauty_max", ascending=False)
    kk = min(50, len(ms) // 2)
    sep = {"k": kk,
           "like_permille_top": round(float(ms.head(kk)["like_pm"].mean() * 1000), 2),
           "like_permille_bot": round(float(ms.tail(kk)["like_pm"].mean() * 1000), 2),
           "yrate_top": round(float(ms.head(kk)["y_rate"].mean()), 3),
           "yrate_bot": round(float(ms.tail(kk)["y_rate"].mean()), 3)}

    result = {"n_posts_matched": int(len(m)), "n_faces": int(len(face_path)),
              "weights": wpath.name, "correlations": corr, "top_vs_bottom_by_beauty": sep}
    dst = data_path("metrics_dir", "beauty_vk.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
