"""Локальная галерея: топ/низ VK-лиц по предсказанной КРАСОТЕ (глазами оценить рейтинг).

    ENV_FOR_DYNACONF=natural uv run python scripts/beauty_gallery.py --n 24

ЛОКАЛЬНО, не публиковать: встроены кропы реальных лиц (base64). Только взрослые (age>=18).
Пишет reports/engagement/beauty_gallery_<dataset>.html (каталог в .gitignore).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from age_gap import beauty, contrastive
from age_gap.common.io import data_path, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


@torch.no_grad()
def _score(model, pixels, device, mu, sd, batch=256):
    model.eval()
    out = []
    for i in range(0, len(pixels), batch):
        px = torch.from_numpy(np.asarray(pixels[i:i + batch], dtype=np.float32)).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out.append(model(px).float().cpu().numpy())
    return np.concatenate(out) * sd + mu


def _b64(path):
    try:
        return base64.b64encode(resolve_path(path).read_bytes()).decode()
    except Exception:  # noqa: BLE001
        return None


def _cards(rows):
    out = []
    for _, r in rows.iterrows():
        b = _b64(r["hires_path"])
        if not b:
            continue
        out.append(f'''<div class="card"><img src="data:image/jpeg;base64,{b}">
<div class="sc">{r["beauty"]:.2f}<span>/5 красота</span></div>
<div class="m">{hashlib.sha256(str(r["post_id"]).encode()).hexdigest()[:8]} · ~{int(r["age"])} лет
· {"ж" if r["gender"] == 0 else "м"}<br>{r["like_pm"]:.1f}‰ лайков/показ · e_rate {r["e_rate"]:+.2f}</div></div>''')
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()

    device = beauty.pick_device()
    wpath = resolve_path(args.weights) if args.weights else resolve_path("data_beauty", "weights", "beauty_dinov2.pt")
    ckpt = torch.load(wpath, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "clip")
    model = beauty.BeautyRegressor(backbone, unfreeze_top=int(ckpt["unfreeze_vision"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    mu, sd = float(ckpt["mu"]), float(ckpt["sd"])

    df = contrastive.build_face_table("vk").drop_duplicates("face_id").reset_index(drop=True)
    hd = data_path("data_dir", "interim", "faces_hires")
    df["hires_path"] = [str(hd / f"{fid}.jpg") for fid in df["face_id"]]
    df = df[[Path(p).exists() for p in df["hires_path"]]].reset_index(drop=True)
    oof = pd.read_parquet(data_path("data_dir", "processed", "engagement_oof.parquet"))
    lpv = (oof["likes"] / oof["views"].clip(lower=1) * 1000)
    df = df.merge(pd.DataFrame({"post_id": oof["post_id"].astype(str), "like_pm": lpv}), on="post_id", how="left")

    px = beauty.clip_pixels_from_crops(df["hires_path"].tolist(), backbone=backbone)
    df["beauty"] = _score(model, px, device, mu, sd)
    df = df.sort_values("beauty", ascending=False).reset_index(drop=True)
    log.info("beauty на VK: min=%.2f max=%.2f mean=%.2f", df["beauty"].min(), df["beauty"].max(), df["beauty"].mean())

    top = df.head(args.n)
    bot = df.tail(args.n).iloc[::-1]
    html = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Beauty gallery</title><style>
 body{{margin:0;background:#0f1218;color:#e6e9ef;font:14px/1.5 Segoe UI,sans-serif}}
 .wrap{{max-width:1180px;margin:0 auto;padding:24px}}
 .banner{{background:#3a1113;border:1px solid #7d2b2b;color:#ffd7d3;padding:12px;border-radius:8px;margin-bottom:18px}}
 h1{{font-size:24px}} h2{{margin-top:28px;border-bottom:1px solid #2a3040;padding-bottom:6px}}
 .cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}}
 .card{{background:#171b24;border:1px solid #232838;border-radius:10px;padding:8px}}
 .card img{{width:100%;border-radius:6px;image-rendering:auto}}
 .sc{{font-size:20px;font-weight:700;color:#4f8ef7;margin-top:6px}} .sc span{{font-size:11px;color:#8a94a6;font-weight:400}}
 .m{{font-size:11px;color:#8a94a6;margin-top:4px}}</style></head><body><div class="wrap">
<div class="banner"><b>ЛОКАЛЬНО — НЕ ПУБЛИКОВАТЬ.</b> Кропы реальных лиц. Только взрослые (age≥18).
Рейтинг предсказан beauty-моделью ({backbone}, обучена на SCUT-FBP5500 vs людей),
применённой к VK по hi-res кропам. Шкала 1–5.</div>
<h1>Что модель считает красивым/некрасивым на VK</h1>
<p style="color:#8a94a6">Лиц оценено: {len(df):,}. Диапазон beauty: {df["beauty"].min():.2f}–{df["beauty"].max():.2f}.
Рядом — фактическая ставка лайков/показ, чтобы видеть, связана ли красота с реакцией (по числам — почти нет).</p>
<h2>Топ-{args.n} по красоте</h2><div class="cards">{_cards(top)}</div>
<h2>Низ-{args.n} по красоте</h2><div class="cards">{_cards(bot)}</div>
</div></body></html>'''

    out = resolve_path("reports", "engagement", f"beauty_gallery_{data_path('data_dir').name}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    log.info("Галерея: %s (%.1f MB)", out, out.stat().st_size / 1e6)
    print(f"OK: {out}")


if __name__ == "__main__":
    main()
