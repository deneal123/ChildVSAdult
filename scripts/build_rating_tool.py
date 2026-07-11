"""Собрать локальный инструмент парных human-оценок привлекательности VK-лиц.

    ENV_FOR_DYNACONF=natural uv run python scripts/build_rating_tool.py --n 500 --pairs 300

Скорит взрослые лица текущей beauty-моделью (для стратифицированного сэмпла по всему диапазону),
встраивает ~n кропов base64 в самодостаточный reports/rating/rate_<dataset>.html. Там JS показывает
случайные пары «кто привлекательнее», копит выбор и экспортирует pairs.jsonl. Полностью локально.

Гардрейл: только взрослые (age>=18). Файл не публиковать (реальные лица).
"""

from __future__ import annotations

import argparse
import base64
import json

import numpy as np
import torch

from age_gap import beauty, contrastive
from age_gap.common.io import data_path, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def _hires_paths(df):
    d = data_path("data_dir", "interim", "faces_hires")
    paths = [d / f"{fid}.jpg" for fid in df["face_id"]]
    ok = [p.exists() for p in paths]
    return paths, np.array(ok)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500, help="размер пула лиц")
    ap.add_argument("--pairs", type=int, default=300, help="рекомендуемое число пар для оценки")
    ap.add_argument("--max-candidates", type=int, default=4000, help="сколько лиц скорить для сэмпла (память)")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    device = beauty.pick_device()
    wpath = resolve_path(args.weights) if args.weights else resolve_path("data_beauty", "weights", "beauty_dinov2.pt")
    ckpt = torch.load(wpath, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "clip")
    model = beauty.BeautyRegressor(backbone, unfreeze_top=int(ckpt["unfreeze_vision"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    mu, sd = float(ckpt["mu"]), float(ckpt["sd"])

    df = contrastive.build_face_table("vk").drop_duplicates("face_id").reset_index(drop=True)
    paths, ok = _hires_paths(df)
    df, paths = df[ok].reset_index(drop=True), [p for p, k in zip(paths, ok, strict=True) if k]
    log.info("Взрослых лиц с hi-res кропом: %d", len(df))

    # для пула нужно всего ~n лиц; скорить все десятки тысяч не нужно и тяжело по памяти —
    # берём случайную подвыборку кандидатов и скорим только её (стриминг, малый расход).
    if len(df) > args.max_candidates:
        keep = np.random.default_rng(0).permutation(len(df))[: args.max_candidates]
        df = df.iloc[keep].reset_index(drop=True)
        paths = [paths[i] for i in keep]
        log.info("Подвыборка кандидатов до %d для скоринга", len(df))

    df["beauty"] = beauty.score_paths(model, paths, device, mu, sd, backbone, batch=64)

    # стратифицированный сэмпл по децилям предсказанной красоты (пары информативнее по всему диапазону)
    df["dec"] = np.clip((df["beauty"].rank(pct=True) * 10).astype(int), 0, 9)
    per = max(1, args.n // 10)
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.permutation(df.index[df["dec"] == d].to_numpy())[:per] for d in range(10)])
    pool = df.loc[pick].reset_index(drop=True)

    faces = []
    for _, r in pool.iterrows():
        p = data_path("data_dir", "interim", "faces_hires", f"{r['face_id']}.jpg")
        b = base64.b64encode(p.read_bytes()).decode()
        faces.append({"id": r["face_id"], "b": b, "s": round(float(r["beauty"]), 3)})
    log.info("В пул отобрано лиц: %d", len(faces))

    html = _TEMPLATE.replace("__FACES__", json.dumps(faces, ensure_ascii=False)).replace("__TARGET__", str(args.pairs))
    out = resolve_path("reports", "rating", f"rate_{data_path('data_dir').name}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"OK: {out} ({out.stat().st_size / 1e6:.1f} MB, пул {len(faces)} лиц)")


_TEMPLATE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Оценка привлекательности</title>
<style>
 body{margin:0;background:#0f1218;color:#e6e9ef;font:15px/1.5 Segoe UI,sans-serif;text-align:center}
 .wrap{max-width:900px;margin:0 auto;padding:20px}
 .banner{background:#3a1113;border:1px solid #7d2b2b;color:#ffd7d3;padding:10px;border-radius:8px;margin-bottom:14px;font-size:13px}
 .pair{display:flex;gap:24px;justify-content:center;margin:18px 0}
 .face{cursor:pointer;border:3px solid #232838;border-radius:14px;padding:8px;transition:.1s;background:#171b24}
 .face:hover{border-color:#4f8ef7;transform:translateY(-3px)}
 .face img{width:300px;height:300px;object-fit:cover;border-radius:10px;display:block}
 .bar{height:8px;background:#232838;border-radius:4px;overflow:hidden;margin:12px auto;max-width:500px}
 .bar>div{height:100%;background:#4f8ef7;width:0}
 button{background:#222839;color:#e6e9ef;border:1px solid #2a3040;border-radius:8px;padding:9px 16px;font-size:14px;cursor:pointer;margin:4px}
 button:hover{border-color:#4f8ef7}
 .muted{color:#8a94a6;font-size:13px}
 kbd{background:#222839;border-radius:4px;padding:1px 6px;font-family:monospace}
</style></head><body><div class="wrap">
<div class="banner"><b>ЛОКАЛЬНО — не публиковать.</b> Кропы реальных взрослых лиц. Оценки нужны для валидации модели.</div>
<h2>Кто привлекательнее?</h2>
<p class="muted">Клик по фото или <kbd>←</kbd>/<kbd>→</kbd>. Равны — <kbd>Пробел</kbd>. Цель: <b><span id="tgt">__TARGET__</span></b> пар.</p>
<div class="pair"><div class="face" id="fL" onclick="pick('L')"><img id="iL"></div>
<div class="face" id="fR" onclick="pick('R')"><img id="iR"></div></div>
<div><button onclick="pick('T')">≈ Равны / пропустить (Пробел)</button></div>
<div class="bar"><div id="prog"></div></div>
<p class="muted">Оценено: <b id="cnt">0</b>. <button onclick="save()">💾 Скачать pairs.jsonl</button></p>
<script>
const FACES=__FACES__; let a,b,done=0, out=[];
function rnd(){return Math.floor(Math.random()*FACES.length)}
function next(){a=rnd();b=rnd();while(b===a)b=rnd();
 document.getElementById('iL').src='data:image/jpeg;base64,'+FACES[a].b;
 document.getElementById('iR').src='data:image/jpeg;base64,'+FACES[b].b;}
function pick(w){let winner=w==='L'?FACES[a].id:w==='R'?FACES[b].id:null;
 out.push({a:FACES[a].id,b:FACES[b].id,winner:winner,sa:FACES[a].s,sb:FACES[b].s});
 done++;document.getElementById('cnt').textContent=done;
 document.getElementById('prog').style.width=Math.min(100,100*done/__TARGET__)+'%';next();}
function save(){let s=out.map(o=>JSON.stringify(o)).join('\n');
 let bl=new Blob([s],{type:'application/x-ndjson'});let u=URL.createObjectURL(bl);
 let el=document.createElement('a');el.href=u;el.download='pairs.jsonl';el.click();}
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')pick('L');
 else if(e.key==='ArrowRight')pick('R');else if(e.key===' '){e.preventDefault();pick('T')}});
next();
</script></div></body></html>"""


if __name__ == "__main__":
    main()
