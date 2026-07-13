"""Итоговая галерея ФИНАЛЬНОГО пайплайна: рекомендация АНКЕТЫ, а не кадра.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_gallery.py --n 12 --size 448

Что изменилось против старой beauty_gallery.py: единица показа — ЧЕЛОВЕК (анкета), а не лицо.
Скор анкеты = residual-голова поверх популяционного приора на эмбеддинге, УСРЕДНЁННОМ по всем
её фото (+0.062 против одного кадра — крупнейший прирост ветки). Внутри анкеты фото отсортированы
покадровым скором — это фича «какое фото ставить главным» (точность 0.66), и её видно глазами.

ВСЕ СКОРЫ OUT-OF-FOLD (GroupKFold по человеку): анкета скорится головой, которая её НЕ ВИДЕЛА.
Иначе галерея круговая — сортируй чем угодно, топ будет выглядеть убедительно. На этих граблях
ветка уже стояла (engagement-галерея), второй раз не наступаем.

Секции:
  ТОП / НИЗ анкет по предсказанию — работает ли модель;
  КРУПНЕЙШИЕ ОШИБКИ (|предсказание − твоя оценка|) — где она врёт.

ВАЖНО про секцию ошибок: это НЕ слепое пятно модели, а УСАДКА К СРЕДНЕМУ. Предсказания живут в
0.39..3.29 против твоих 1.0..5.0 — разброс сжат в 1.9 раза (модель завышает низ на +0.59 и
занижает верх на −1.53). Ridge минимизирует квадрат ошибки, а 33% дисперсии меток — шум, поэтому
оптимальный предсказатель ОБЯЗАН сжимать. На ранжирование это не влияет (Spearman инвариантен к
монотонным преобразованиям), но абсолютную «4.5 из 5» модель не выдаст никогда.

Кропы перегенерируются из ОРИГИНАЛОВ в высоком разрешении (--size) только для показанных лиц.
ЛОКАЛЬНО, не публиковать: внутри base64 реальных лиц. Только взрослые (гардрейл в faces_hires).
Пишет reports/beauty/recsys_gallery.html (каталог в .gitignore).
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from age_gap import contrastive
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.common.schemas import RawPost
from age_gap.preprocessing.crop_align import margin_crop

log = get_logger(__name__)

ENC = ["dino_b", "clip_b32", "clip_L14", "siglip_b"]
GRID = [(k, a) for k in [40, 80, 160] for a in [50, 200, 800]]


def _fit(X, y, pr, tr, te, k, a):
    sc = StandardScaler().fit(X[tr])
    pca = PCA(min(k, len(tr) - 1, X.shape[1]), random_state=0, whiten=True).fit(sc.transform(X[tr]))
    c, d = np.polyfit(pr[tr], y[tr], 1)
    rg = Ridge(alpha=a).fit(pca.transform(sc.transform(X[tr])), y[tr] - (c * pr[tr] + d))
    return (c * pr[te] + d) + rg.predict(pca.transform(sc.transform(X[te])))


def _hq(fid, meta, photos, size):
    """Кроп высокого разрешения из ОРИГИНАЛА (для показа)."""
    m = meta.get(fid)
    if not m or not m.get("bbox"):
        return None
    lp = photos.get(m.get("photo_id"))
    if not lp:
        return None
    img = cv2.imread(str(resolve_path(lp)))
    if img is None:
        return None
    crop = margin_crop(img, m["bbox"], margin=0.4, size=size)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12, help="анкет в каждой секции")
    ap.add_argument("--size", type=int, default=448, help="сторона кропа для показа")
    ap.add_argument("--max-photos", type=int, default=6)
    args = ap.parse_args()

    Z = dict(np.load(resolve_path("data_beauty", "cache", "embeds_person.npz"), allow_pickle=True))
    fids = list(Z["fids"])
    EMB = np.hstack([Z[k] for k in ENC])
    PRI = Z["prior"]
    idx = {f: i for i, f in enumerate(fids)}

    ft = contrastive.build_face_table("vk").drop_duplicates("face_id").set_index("face_id")
    pers = np.array([ft.loc[f, "post_id"] for f in fids])
    lab = {r["face_id"]: float(r["score"])
           for r in read_jsonl(resolve_path("reports/rating/ratings.jsonl")) if r.get("score")}
    lab = {f: s for f, s in lab.items() if f in idx}

    by_p = collections.defaultdict(list)
    for i, p in enumerate(pers):
        by_p[p].append(i)
    lbp = collections.defaultdict(list)
    for f, s in lab.items():
        lbp[pers[idx[f]]].append((idx[f], s))
    pk = sorted(lbp)
    yb = np.array([np.mean([s for _, s in lbp[p]]) for p in pk])
    prb = np.array([np.mean([PRI[i] for i, _ in lbp[p]]) for p in pk])
    Emean = np.array([EMB[by_p[p]].mean(0) for p in pk])          # <- усреднение по ВСЕМ фото анкеты

    # --- OOF-скор АНКЕТЫ (голова не видела этого человека) ---
    grp = np.arange(len(pk))                                       # один человек = одна строка
    sp = np.full(len(pk), np.nan)
    for tr, te in GroupKFold(5).split(Emean, yb, grp):
        folds = [(tr[i1], tr[i2]) for i1, i2 in GroupKFold(3).split(Emean[tr], yb[tr], grp[tr])]
        best, bs = None, -np.inf
        for k, a in GRID:
            s = np.mean([spearmanr(yb[b], _fit(Emean, yb, prb, a_, b, k, a)).statistic
                         for a_, b in folds])
            if s > bs:
                bs, best = s, (k, a)
        sp[te] = _fit(Emean, yb, prb, tr, te, *best)
    log.info("OOF-скор анкеты: Spearman=%.4f (n=%d)", spearmanr(yb, sp).statistic, len(pk))

    # --- OOF-скор КАДРА (для порядка фото внутри анкеты) ---
    li = np.array([idx[f] for f in lab])
    yf = np.array([lab[f] for f in lab])
    gf = pers[li]
    sf = np.full(len(fids), np.nan)
    for tr, te in GroupKFold(5).split(EMB[li], yf, gf):
        tgt = sorted({i for p in set(gf[te]) for i in by_p[p]})    # ВСЕ фото тестовых людей
        sc = StandardScaler().fit(EMB[li][tr])
        pca = PCA(80, random_state=0, whiten=True).fit(sc.transform(EMB[li][tr]))
        c, d = np.polyfit(PRI[li][tr], yf[tr], 1)
        rg = Ridge(alpha=200).fit(pca.transform(sc.transform(EMB[li][tr])),
                                  yf[tr] - (c * PRI[li][tr] + d))
        sf[tgt] = (c * PRI[tgt] + d) + rg.predict(pca.transform(sc.transform(EMB[tgt])))

    # --- секции ---
    order = np.argsort(-sp)
    err = np.argsort(-np.abs(sp - yb))
    SEC = [("Топ анкет по предсказанию", order[:args.n], "win"),
           ("Низ анкет по предсказанию", order[-args.n:][::-1], "dead"),
           ("Крупнейшие ОШИБКИ модели", err[:args.n], "err")]

    meta = {f["face_id"]: f for f in read_jsonl(data_path("data_dir", "interim", "faces.jsonl"))}
    photos = {}
    for row in read_jsonl(data_path("data_dir", "raw", "posts.jsonl")):
        for ph in RawPost.from_dict(row).photos:
            if ph.local_path:
                photos[ph.photo_id] = ph.local_path

    parts = []
    for title, ids, cls in SEC:
        cards = []
        for j in ids:
            p = pk[j]
            ph = sorted(by_p[p], key=lambda i: -sf[i] if np.isfinite(sf[i]) else 0)[:args.max_photos]
            imgs = []
            for rank, i in enumerate(ph):
                b = _hq(fids[i], meta, photos, args.size)
                if not b:
                    continue
                mine = "★ главное" if rank == 0 else f"#{rank + 1}"
                you = f' · <b>ты: {lab[fids[i]]:.0f}</b>' if fids[i] in lab else ""
                imgs.append(f'<figure{" class=best" if rank == 0 else ""}>'
                            f'<img src="data:image/jpeg;base64,{b}">'
                            f'<figcaption>{mine} · модель {sf[i]:.2f}{you}</figcaption></figure>')
            if not imgs:
                continue
            d = sp[j] - yb[j]
            cards.append(
                f'<div class="prof {cls}"><div class="hd">'
                f'<span class="id">{hashlib.sha256(str(p).encode()).hexdigest()[:8]}</span>'
                f'<span class="sc">модель <b>{sp[j]:.2f}</b></span>'
                f'<span class="tr">ты <b>{yb[j]:.2f}</b></span>'
                f'<span class="pr">приор {prb[j]:.2f}</span>'
                f'<span class="df {"neg" if abs(d) > 0.8 else ""}">Δ {d:+.2f}</span>'
                f'</div><div class="ph">{"".join(imgs)}</div></div>')
        parts.append(f'<h2>{title}</h2><div class="grid">{"".join(cards)}</div>')

    rho = spearmanr(yb, sp).statistic
    html = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Финальный пайплайн — галерея анкет</title><style>
 body{{margin:0;background:#0f1218;color:#e6e9ef;font:14px/1.6 Segoe UI,sans-serif}}
 .wrap{{max-width:1500px;margin:0 auto;padding:26px}}
 h1{{font-size:26px}} h2{{font-size:19px;margin:34px 0 8px;border-bottom:1px solid #2a3040;padding-bottom:6px}}
 .note{{background:#171b24;border-left:3px solid #4f8ef7;padding:12px 16px;border-radius:8px;margin:12px 0;color:#cfd6e4}}
 .grid{{display:flex;flex-direction:column;gap:12px}}
 .prof{{background:#171b24;border:1px solid #232838;border-left:3px solid #4f8ef7;border-radius:10px;padding:10px 12px}}
 .prof.win{{border-left-color:#2fbf71}} .prof.dead{{border-left-color:#e0564a}} .prof.err{{border-left-color:#e0a23a}}
 .hd{{display:flex;gap:16px;align-items:center;margin-bottom:8px;color:#8a94a6;font-size:13px}}
 .hd b{{color:#e6e9ef;font-size:15px}} .id{{font-family:monospace}}
 .df.neg b,.df.neg{{color:#e0a23a}}
 .ph{{display:flex;gap:8px;overflow-x:auto}}
 figure{{margin:0;flex:0 0 auto}} figure img{{height:190px;border-radius:8px;display:block}}
 figure.best img{{outline:2px solid #2fbf71;outline-offset:1px}}
 figcaption{{font-size:11px;color:#8a94a6;text-align:center;margin-top:3px}}
 .mut{{color:#8a94a6}}
</style></head><body><div class="wrap">
<h1>Финальный пайплайн — галерея анкет</h1>
<div class="note"><b>Единица показа — АНКЕТА, не кадр.</b> Скор анкеты = residual-голова поверх
популяционного приора на эмбеддинге, усреднённом по всем её фото. Фото внутри анкеты отсортированы
покадровым скором: <b>★ = какое ставить главным</b>.<br>
<b>Все скоры out-of-fold</b> (GroupKFold по человеку) — голову, которая скорит анкету, эта анкета
никогда не видела. Иначе галерея была бы круговой и «убедительной» при любой сортировке.<br>
Качество OOF на этой выборке: <b>Spearman {rho:.3f}</b> ({len(pk)} анкет). Потолок 0.82.</div>
<div class="note"><b>Как читать секцию ошибок.</b> Это НЕ слепое пятно, а <b>усадка к среднему</b>:
предсказания лежат в 0.39–3.29 против твоих 1.0–5.0, разброс сжат в <b>1.9 раза</b> (низ завышается
на +0.59, верх занижается на −1.53). Ridge минимизирует квадрат ошибки, а треть дисперсии твоих
меток — шум, поэтому оптимальный предсказатель <b>обязан</b> сжимать. Ранжирование от этого не
страдает (Spearman инвариантен к монотонным преобразованиям), но абсолютную «4.5 из 5» модель не
выдаст никогда — учитывать, если показывать балл пользователю.</div>
<p class="mut">Δ = предсказание − твоя оценка. Локально, не публиковать.</p>
{"".join(parts)}
</div></body></html>"""
    out = resolve_path("reports", "beauty", "recsys_gallery.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    log.info("готово: %s (%.1f МБ)", out, out.stat().st_size / 1e6)
    print(f"OK: {out}")


if __name__ == "__main__":
    main()
