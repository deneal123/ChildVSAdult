"""Инструмент разметки красоты по шкале 1–5 (Likert) на ВЕСЬ датасет natural.

    ENV_FOR_DYNACONF=natural uv run python scripts/build_rating_likert.py

Парная разметка не масштабируется (18k лиц -> ~250k сравнений), поэтому здесь по одному лицу
за клик, шкала 1..5 — та же, что у SCUT-FBP5500, поэтому метки втыкаются прямо в beauty_train
и дают VK-native модель на ТВОЁМ вкусе.

Картинки грузятся по ОТНОСИТЕЛЬНЫМ путям (без base64 — иначе HTML был бы сотни МБ).
Прогресс автосохраняется в localStorage браузера: можно закрыть и продолжить с места.
Экспорт -> ratings.jsonl {face_id, score}.

ЛОКАЛЬНО, не публиковать. Только взрослые (age>=18).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from age_gap import contrastive
from age_gap.common.io import data_path, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help=">0: ограничить число лиц (0 = все)")
    ap.add_argument("--retest", default=None,
                    help="путь к ratings.jsonl: показать ТОЛЬКО уже оценённые лица (замер "
                         "test-retest: насколько ты согласен сам с собой)")
    args = ap.parse_args()

    df = contrastive.build_face_table("vk").drop_duplicates("face_id").reset_index(drop=True)
    hd = data_path("data_dir", "interim", "faces_hires")
    df["path"] = [hd / f"{fid}.jpg" for fid in df["face_id"]]
    df = df[[p.exists() for p in df["path"]]].reset_index(drop=True)

    if args.retest:
        from age_gap.common.io import read_jsonl as _rj
        rated = {r["face_id"] for r in _rj(resolve_path(args.retest)) if r.get("score")}
        df = df[df["face_id"].isin(rated)].reset_index(drop=True)
        log.info("RETEST-режим: только уже оценённые лица (%d)", len(df))

    # случайный порядок -> частичная разметка остаётся несмещённой выборкой
    df = df.iloc[np.random.default_rng(0).permutation(len(df))].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    log.info("Лиц к разметке: %d", len(df))

    sfx = "_retest" if args.retest else ""
    out = resolve_path("reports", "rating", f"likert_{data_path('data_dir').name}{sfx}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    # относительный путь от html к кропам
    faces = [{"id": r["face_id"],
              "p": os.path.relpath(str(r["path"]), str(out.parent)).replace("\\", "/")}
             for _, r in df.iterrows()]

    html = (_TPL.replace("__FACES__", json.dumps(faces, ensure_ascii=False))
                .replace("__KEY__", "likert_retest_v1" if args.retest else "likert_ratings_v1"))
    out.write_text(html, encoding="utf-8")
    print(f"OK: {out} ({out.stat().st_size / 1e6:.1f} MB, лиц {len(faces):,})")


_TPL = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Оценка красоты 1–5</title>
<style>
 body{margin:0;background:#0f1218;color:#e6e9ef;font:15px/1.5 Segoe UI,sans-serif;text-align:center}
 .wrap{max-width:760px;margin:0 auto;padding:18px}
 .banner{background:#3a1113;border:1px solid #7d2b2b;color:#ffd7d3;padding:9px;border-radius:8px;margin-bottom:12px;font-size:13px}
 img{width:340px;height:340px;object-fit:cover;border-radius:14px;border:3px solid #232838;background:#171b24}
 .btns{margin:14px 0}
 button{background:#222839;color:#e6e9ef;border:1px solid #2a3040;border-radius:10px;
   padding:12px 20px;font-size:18px;cursor:pointer;margin:3px;min-width:56px}
 button:hover{border-color:#4f8ef7;background:#2a3040}
 .sk{font-size:14px;padding:8px 14px;min-width:auto}
 .bar{height:8px;background:#232838;border-radius:4px;overflow:hidden;margin:12px auto;max-width:520px}
 .bar>div{height:100%;background:#4f8ef7;width:0}
 .mut{color:#8a94a6;font-size:13px} kbd{background:#222839;border-radius:4px;padding:1px 6px;font-family:monospace}
 .exp{background:#1d3a2a;border-color:#2fbf71}
</style></head><body><div class="wrap">
<div class="banner"><b>ЛОКАЛЬНО — не публиковать.</b> Реальные лица (age≥18). Прогресс сохраняется в браузере.</div>
<h2 style="margin:6px">Насколько привлекательно лицо?</h2>
<p class="mut">Клавиши <kbd>1</kbd>–<kbd>5</kbd> (1 = совсем нет, 5 = очень). <kbd>Пробел</kbd> — пропустить.
Можно закрыть и вернуться — продолжит с места.</p>
<img id="im" alt="">
<div class="btns">
 <button onclick="rate(1)">1</button><button onclick="rate(2)">2</button><button onclick="rate(3)">3</button>
 <button onclick="rate(4)">4</button><button onclick="rate(5)">5</button>
 <button class="sk" onclick="rate(0)">пропустить</button>
</div>
<div class="bar"><div id="prog"></div></div>
<p class="mut">Размечено: <b id="cnt">0</b> из <span id="tot">0</span> ·
 <button class="sk exp" onclick="save()">💾 Скачать ratings.jsonl</button>
 <button class="sk" onclick="reset()">сбросить</button></p>
<script>
const FACES=__FACES__; const KEY='__KEY__';
let done = JSON.parse(localStorage.getItem(KEY) || '{}');
let i = 0;
document.getElementById('tot').textContent = FACES.length.toLocaleString();
function nextIdx(){ while(i < FACES.length && done[FACES[i].id] !== undefined) i++; return i; }
function show(){
  if(nextIdx() >= FACES.length){ document.getElementById('im').removeAttribute('src');
    alert('Всё размечено! Не забудь скачать ratings.jsonl'); return; }
  document.getElementById('im').src = FACES[i].p;
  if(FACES[i+1]) { const pre=new Image(); pre.src=FACES[i+1].p; }  // предзагрузка
  const n = Object.keys(done).length;
  document.getElementById('cnt').textContent = n.toLocaleString();
  document.getElementById('prog').style.width = (100*n/FACES.length)+'%';
}
function rate(s){
  if(i >= FACES.length) return;
  if(s > 0) done[FACES[i].id] = s; else done[FACES[i].id] = null;  // null = пропуск
  localStorage.setItem(KEY, JSON.stringify(done));
  i++; show();
}
function save(){
  const lines = Object.entries(done).filter(([,v]) => v !== null)
    .map(([id,s]) => JSON.stringify({face_id:id, score:s}));
  const bl = new Blob([lines.join('\n')], {type:'application/x-ndjson'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(bl);
  a.download = 'ratings.jsonl'; a.click();
}
function reset(){ if(confirm('Стереть весь прогресс?')){ done={}; localStorage.removeItem(KEY); i=0; show(); } }
document.addEventListener('keydown', e => {
  if(e.key >= '1' && e.key <= '5') rate(parseInt(e.key));
  else if(e.key === ' '){ e.preventDefault(); rate(0); }
});
show();
</script></div></body></html>"""


if __name__ == "__main__":
    main()
