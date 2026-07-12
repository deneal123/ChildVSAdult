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

import pandas as pd
import torch

from age_gap import beauty, contrastive
from age_gap.common.io import data_path, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


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
    ap.add_argument("--sort", choices=["beauty", "engagement"], default="beauty",
                    help="beauty = сортировка по красоте (проверка модели); "
                         "engagement = по лайкам/показ (НЕкруговой тест связи красота->отклик)")
    args = ap.parse_args()

    device = beauty.pick_device()
    wpath = resolve_path(args.weights) if args.weights else resolve_path("data_beauty", "weights", "beauty_dinov2.pt")
    ckpt = torch.load(wpath, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "clip")
    is_vk = wpath.stem.endswith("_vk")
    mtag = "_vk" if is_vk else ("" if wpath.stem == "beauty_dinov2" else f"_{wpath.stem.split('_')[-1]}")
    model_desc = ("<b>дообучена на ТВОИХ 1619 оценках</b> (VK-native, старт от SCUT-модели)" if is_vk
                  else f"обучена на SCUT-FBP5500 vs людей ({backbone})")
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

    df["beauty"] = beauty.score_paths(model, df["hires_path"].tolist(), device, mu, sd, backbone)
    key = "beauty" if args.sort == "beauty" else "like_pm"
    df = df.dropna(subset=[key]).sort_values(key, ascending=False).reset_index(drop=True)
    log.info("beauty на VK: min=%.2f max=%.2f mean=%.2f", df["beauty"].min(), df["beauty"].max(), df["beauty"].mean())

    top = df.head(args.n)
    bot = df.tail(args.n).iloc[::-1]
    # ключевые средние: если сортируем по вовлечённости, смотрим, отличается ли КРАСОТА верха и низа
    t_b, b_b = float(top["beauty"].mean()), float(bot["beauty"].mean())
    t_l, b_l = float(top["like_pm"].mean()), float(bot["like_pm"].mean())
    log.info("sort=%s | beauty top=%.2f bot=%.2f | like/1000v top=%.1f bot=%.1f",
             args.sort, t_b, b_b, t_l, b_l)
    verdict = (f"Отсортировано по <b>вовлечённости</b> (лайки/показ). Средняя предсказанная КРАСОТА: "
               f"верх <b>{t_b:.2f}</b> против низ <b>{b_b:.2f}</b> (лайки/1000показов: {t_l:.1f} vs {b_l:.1f}). "
               f"Если связь была бы прямой — верх был бы заметно красивее."
               if args.sort == "engagement" else
               f"Отсортировано по <b>предсказанной красоте</b> — разница верх/низ здесь гарантирована "
               f"построением (сортируем по ней же). Средняя ставка лайков/1000показов: верх {t_l:.1f} "
               f"против низ {b_l:.1f} — вот это уже информативно.")
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
Рейтинг предсказан beauty-моделью: {model_desc}, применена к VK по hi-res кропам. Шкала 1–5.</div>
<h1>{"Топ/низ по ВОВЛЕЧЁННОСТИ — красивее ли верх?" if args.sort == "engagement" else "Что модель считает красивым/некрасивым на VK"}</h1>
<div style="background:#171b24;border-left:3px solid #4f8ef7;padding:12px 16px;border-radius:8px;margin:12px 0;color:#cfd6e4">
{verdict}</div>
<p style="color:#8a94a6">Лиц оценено: {len(df):,}. Диапазон beauty: {df["beauty"].min():.2f}–{df["beauty"].max():.2f}.</p>
<h2>Топ-{args.n} по {"вовлечённости" if args.sort == "engagement" else "красоте"}</h2><div class="cards">{_cards(top)}</div>
<h2>Низ-{args.n} по {"вовлечённости" if args.sort == "engagement" else "красоте"}</h2><div class="cards">{_cards(bot)}</div>
</div></body></html>'''

    sfx = "_by_engagement" if args.sort == "engagement" else ""
    out = resolve_path("reports", "engagement", f"beauty_gallery_{data_path('data_dir').name}{sfx}{mtag}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    log.info("Галерея: %s (%.1f MB)", out, out.stat().st_size / 1e6)
    print(f"OK: {out}")


if __name__ == "__main__":
    main()
