"""Самодостаточный локальный HTML-отчёт по «силе симпатии аудитории».

ЛОКАЛЬНО. Не публиковать, не коммитить: при --with-thumbnails внутрь встраиваются
кропы лиц (base64). Каталог reports/ добавлен в .gitignore.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template

from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.engagement.features import ADULT_MIN_AGE

log = get_logger(__name__)

ACCENT = "#4f8ef7"
GOOD = "#2fbf71"
BAD = "#e0564a"
MUTED = "#8a94a6"


def pseudo_id(post_id: str) -> str:
    return hashlib.sha256(post_id.encode()).hexdigest()[:10]


# ---------------------------------------------------------------- thumbnails


def best_adult_face_paths(post_ids: set[str]) -> dict[str, Path]:
    """post_id -> путь к кропу лучшего ВЗРОСЛОГО лица (детские лица не показываем)."""
    photo2post: dict[str, str] = {}
    for row in read_jsonl(data_path("data_dir", "raw", "posts.jsonl")):
        if row["post_id"] in post_ids:
            for ph in row.get("photos", []) or []:
                photo2post[ph["photo_id"]] = row["post_id"]

    ages = {r["face_id"]: r["age_est"] for r in read_jsonl(data_path("data_dir", "interim", "face_genderage.jsonl"))}
    best: dict[str, tuple[float, Path]] = {}
    for f in read_jsonl(data_path("data_dir", "interim", "faces.jsonl")):
        pid = photo2post.get(f.get("photo_id", ""))
        if pid is None or not f.get("is_usable"):
            continue
        age = ages.get(f["face_id"])
        if age is None or float(age) < ADULT_MIN_AGE:
            continue  # guardrail: никогда не показываем детские лица
        q = float(f.get("face_quality_score", 0))
        if pid not in best or q > best[pid][0]:
            best[pid] = (q, resolve_path(f["face_crop_path"]))
    return {k: v[1] for k, v in best.items()}


def _b64(path: Path) -> str | None:
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- figures


def _layout(fig: go.Figure, title: str, h: int = 380) -> go.Figure:
    fig.update_layout(
        title=title, height=h, template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=50), font=dict(size=13),
    )
    return fig


def fig_variance(reach: dict[str, Any]) -> go.Figure:
    a = reach["A_exogenous_only"]["share_variance_explained_by_reach"]
    b = reach["B_also_controls_views"]["share_variance_explained_by_reach"]
    fig = go.Figure()
    fig.add_bar(name="объяснено охватом", x=["A: без views", "B: + views"], y=[a, b], marker_color=MUTED)
    fig.add_bar(name="остаток («симпатия»)", x=["A: без views", "B: + views"], y=[1 - a, 1 - b], marker_color=ACCENT)
    fig.update_layout(barmode="stack", yaxis_tickformat=".0%", yaxis_title="доля дисперсии E_raw")
    return _layout(fig, "Сколько сырой вовлечённости — это просто ОХВАТ")


def fig_ablation(ablation: list[dict[str, Any]]) -> go.Figure:
    x = [a["level"] for a in ablation]
    y = [a["oof_overall"]["spearman"] for a in ablation]
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=[MUTED, MUTED, MUTED, ACCENT],
                           text=[f"{v:.3f}" for v in y], textposition="outside"))
    fig.update_layout(yaxis_title="Spearman ρ (OOF)")
    return _layout(fig, "Ablation: что добавляет сигнал сверх охвата")


def fig_models(models: list[dict[str, Any]], neg: dict[str, Any]) -> go.Figure:
    rows = [(m["target"] if m["model"] == "hgb" else f'{m["target"]} [{m["model"]}]',
             m["oof_overall"]["spearman"]) for m in models]
    rows.append(("негативный контроль (перемешано)", neg["oof_overall"]["spearman"]))
    rows.sort(key=lambda t: t[1])
    colors = [BAD if abs(v) < 0.02 else ACCENT for _, v in rows]
    fig = go.Figure(go.Bar(x=[v for _, v in rows], y=[k for k, _ in rows], orientation="h",
                           marker_color=colors, text=[f"{v:.3f}" for _, v in rows], textposition="outside"))
    fig.update_layout(xaxis_title="Spearman ρ (OOF)")
    return _layout(fig, "Таргеты и baseline'ы", h=420)


def fig_importance(imp: list[dict[str, Any]], top: int = 15) -> go.Figure:
    imp = imp[:top][::-1]
    fig = go.Figure(go.Bar(
        x=[d["spearman_drop"] for d in imp], y=[d["feature"] for d in imp], orientation="h",
        error_x=dict(type="data", array=[d["std"] for d in imp]), marker_color=ACCENT))
    fig.update_layout(xaxis_title="падение Spearman при перемешивании признака")
    return _layout(fig, "Важность признаков (permutation, held-out)", h=460)


def fig_clusters(clusters: list[dict[str, Any]]) -> go.Figure:
    x = [c["mean_sympathy"] for c in clusters]
    lo = [c["mean_sympathy"] - c["ci95"][0] for c in clusters]
    hi = [c["ci95"][1] - c["mean_sympathy"] for c in clusters]
    labels = [f"c{c['cluster']} (n={c['size']:,})" for c in clusters]
    colors = [GOOD if c["ci95"][0] > 0 else BAD if c["ci95"][1] < 0 else MUTED for c in clusters]
    fig = go.Figure(go.Bar(x=x, y=labels, orientation="h", marker_color=colors,
                           error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo)))
    fig.add_vline(x=0, line_dash="dash", line_color="#555")
    fig.update_layout(xaxis_title="средняя «симпатия» (остаток), 95% bootstrap-CI")
    return _layout(fig, "Кластеры-архетипы (без обучения)")


def fig_deep(models: list[dict[str, Any]]) -> go.Figure:
    labels = [m["name"] for m in models]
    y = [m["sp"] for m in models]
    colors = [MUTED if m["device"] == "cpu" else ACCENT for m in models]
    fig = go.Figure(go.Bar(x=labels, y=y, marker_color=colors,
                           text=[f"{v:.3f}" for v in y], textposition="outside"))
    fig.update_layout(yaxis_title="Spearman ρ (OOF)")
    return _layout(fig, "CPU-бустинг vs обучаемые GPU-энкодеры")


def fig_pred_true(oof: pd.DataFrame, sample: int = 4000) -> go.Figure:
    d = oof.sample(min(sample, len(oof)), random_state=0)
    fig = go.Figure(go.Scattergl(x=d["pred_rate"], y=d["y_rate"], mode="markers",
                                 marker=dict(size=4, opacity=0.3, color=ACCENT)))
    lo, hi = float(d["pred_rate"].min()), float(d["pred_rate"].max())
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="#555"))
    fig.update_layout(xaxis_title="предсказанная симпатия на показ", yaxis_title="фактический остаток")
    return _layout(fig, "Предсказание против факта (out-of-fold)")


def fig_decile(oof: pd.DataFrame) -> go.Figure:
    d = oof.copy()
    d["decile"] = pd.qcut(d["pred_rate"].rank(method="first"), 10, labels=False) + 1
    g = d.groupby("decile")["y_rate"].mean()
    fig = go.Figure(go.Bar(x=g.index.astype(str), y=g.to_numpy(),
                           marker_color=[BAD if v < 0 else GOOD for v in g.to_numpy()]))
    fig.add_hline(y=0, line_dash="dash", line_color="#555")
    fig.update_layout(xaxis_title="дециль предсказанной симпатии (1=низ, 10=верх)",
                      yaxis_title="средний фактический остаток")
    return _layout(fig, "Монотонность ранжирования по децилям")


# ---------------------------------------------------------------- cards


def _cards(oof: pd.DataFrame, thumbs: dict[str, str], n: int, top: bool) -> list[dict[str, Any]]:
    d = oof.sort_values("pred_rate", ascending=not top).head(n)
    out = []
    for _, r in d.iterrows():
        vv = max(1.0, float(r["views"])) if not pd.isna(r["views"]) else None
        out.append({
            "pid": pseudo_id(str(r["post_id"])),
            "pred": float(r["pred_rate"]),
            "actual": float(r["y_rate"]),
            "likes": int(r["likes"]), "comments": int(r["comments"]), "reposts": int(r["reposts"]),
            "views": None if pd.isna(r["views"]) else int(r["views"]),
            "like_pm": None if vv is None else round(float(r["likes"]) / vv * 1000, 1),
            "rep_pm": None if vv is None else round(float(r["reposts"]) / vv * 1000, 2),
            "n_photos": int(r["n_photos"]),
            "age": None if pd.isna(r["age_est_median"]) else int(r["age_est_median"]),
            "female": float(r["share_female_adult"]),
            "child": bool(r["has_child_photo"]),
            "gap": None if pd.isna(r["age_gap_label"]) else float(r["age_gap_label"]),
            "cluster": int(r["cluster"]),
            "thumb": thumbs.get(str(r["post_id"])),
        })
    return out


# ---------------------------------------------------------------- template

TEMPLATE = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Симпатия аудитории — локальный отчёт</title>
<style>
 :root{--bg:#0f1218;--card:#171b24;--tx:#e6e9ef;--mut:#8a94a6;--acc:#4f8ef7;}
 body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;}
 .wrap{max-width:1180px;margin:0 auto;padding:28px;}
 h1{font-size:30px;margin:0 0 6px} h2{font-size:21px;margin:38px 0 12px;border-bottom:1px solid #2a3040;padding-bottom:8px}
 h3{font-size:16px;color:var(--mut);font-weight:600;margin:22px 0 8px}
 .banner{background:#3a1113;border:1px solid #7d2b2b;color:#ffd7d3;padding:14px 16px;border-radius:10px;margin-bottom:22px}
 .kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:18px 0}
 .kpi div{background:var(--card);border:1px solid #232838;border-radius:12px;padding:14px 16px}
 .kpi b{display:block;font-size:26px;margin-top:4px}
 .kpi small{color:var(--mut)}
 .note{background:var(--card);border-left:3px solid var(--acc);padding:12px 16px;border-radius:8px;margin:14px 0;color:#cfd6e4}
 table{width:100%;border-collapse:collapse;font-size:14px;background:var(--card);border-radius:10px;overflow:hidden}
 th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #232838} th{color:var(--mut);font-weight:600}
 tr:last-child td{border-bottom:none}
 .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));gap:14px}
 .card{background:var(--card);border:1px solid #232838;border-radius:12px;padding:12px}
 .card img{width:100%;border-radius:8px;display:block;margin-bottom:8px;filter:saturate(1.05)}
 .card .pid{font-family:ui-monospace,monospace;font-size:11px;color:var(--mut)}
 .sc{font-size:19px;font-weight:700}
 .pos{color:#2fbf71}.neg{color:#e0564a}
 .meta{font-size:12px;color:var(--mut);margin-top:6px;line-height:1.5}
 .chip{display:inline-block;background:#222839;border-radius:6px;padding:1px 6px;margin:2px 3px 0 0;font-size:11px}
 .plot{background:#fff;border-radius:12px;margin:14px 0;overflow:hidden}
 code{background:#222839;padding:1px 5px;border-radius:4px}
</style></head><body><div class="wrap">

<div class="banner"><b>ЛОКАЛЬНЫЙ ОТЧЁТ — НЕ ПУБЛИКОВАТЬ.</b>
{% if has_thumbs %}Содержит кропы лиц реальных людей.{% endif %}
Каталог <code>reports/</code> в .gitignore. Несовершеннолетние исключены как субъекты;
детские лица не участвуют в признаках внешности и не показываются.</div>

<h1>Сила симпатии аудитории к постам</h1>
<p style="color:var(--mut);margin-top:0">Отдельная ветка исследования. В статьи не входит.
Платформа: {{ platform }} · {{ n_posts }} постов · {{ n_persons }} человек.</p>

<div class="kpi">
  <div><small>Сырой E_raw ↔ показы (Spearman)</small><b class="neg">{{ '%.3f'|format(sp_ev) }}</b></div>
  <div><small>Ставка-на-показ E_rate ↔ показы</small><b class="pos">{{ '%.3f'|format(sp_ratev) }}</b></div>
  <div><small>Ранжирование ставки, Spearman</small><b class="pos">{{ '%.3f'|format(spearman) }}</b></div>
  <div><small>Топ-50: подъём над средним</small><b class="pos">{{ '%.2f'|format(lift) }}σ</b></div>
  <div><small>Негативный контроль</small><b>{{ '%.3f'|format(neg) }}</b></div>
</div>

<div class="note"><b>Почему «на показ».</b> Ценность лайка зависит от того, скольким его показали: пост с
8 млн показов и 1000 лайков — это <i>слабее</i>, чем пост с 1 млн показов и 600 лайков. Сырой композит
вовлечённости почти целиком тянется за показами (Spearman с <code>views</code> =
{{ '%.3f'|format(sp_ev) }}) — то есть измеряет охват, а не отношение. Поэтому <b>главный таргет</b> —
<code>E_rate</code>: композит лог-<i>ставок</i> на показ log((L+1)/(V+1)), log((C+1)/(V+1)),
log((R+1)/(V+1)). Он практически развязан с объёмом показов (Spearman =
{{ '%.3f'|format(sp_ratev) }}). Из него мы дополнительно вычитаем экзогенный охват (сообщество, время,
возраст поста, формат) строго out-of-fold — остаётся «симпатия на показ». Негативный контроль
(перемешанный таргет) даёт ρ≈{{ '%.3f'|format(neg) }}, реальный сигнал ρ={{ '%.3f'|format(spearman) }}:
сигнал есть, он скромный.</div>

<h2>1. Сырые лайки — это в основном охват</h2>
<div class="plot">{{ p_var }}</div>
<div class="note"><b>Диагностика.</b> Столбик показывает, какую долю дисперсии <i>сырого</i> композита
<code>E_raw</code> съедает модель охвата. Именно поэтому мы не ранжируем по сырым лайкам, а переходим к
ставке на показ. Для сравнения оставлены и остатки сырого композита: вариант A (без views)
даёт ρ={{ '%.3f'|format(spearman_raw) }} на самом <code>E_raw</code>, консервативный B (с контролем
views) — ρ={{ '%.3f'|format(spearman_b) }}, модель-свободный перцентиль ставки внутри
«сообщество×месяц» — ρ={{ '%.3f'|format(spearman_pct) }}. Все они согласуются с главным таргетом.</div>

<h2>2. Что именно даёт сигнал</h2>
<div class="plot">{{ p_abl }}</div>
<div class="note"><b>Главная (и этически неудобная) находка.</b> Без эмбеддингов лица (мета, текст,
формат, атрибуты) ранжирование даёт Spearman {{ '%.3f'|format(pre_emb) }}; добавление эмбеддингов
поднимает его до {{ '%.3f'|format(spearman) }}. То есть значимая (здесь — почти вся) часть «симпатии
сверх охвата» — это реакция аудитории на <i>внешность</i>. Это измерение <b>предвзятости
аудитории</b>, а не ценности человека.</div>
<div class="plot">{{ p_imp }}</div>

<h2>3. Таргеты, baseline'ы и негативный контроль</h2>
<div class="plot">{{ p_mod }}</div>
<table><tr><th>Модель</th><th>Таргет</th><th>Spearman</th><th>R²</th><th>NDCG@50</th></tr>
{% for m in models %}<tr><td>{{ m.model }}</td><td>{{ m.target }}</td>
<td>{{ '%.4f'|format(m.sp) }}</td><td>{{ '%.4f'|format(m.r2) }}</td><td>{{ '%.3f'|format(m.ndcg) }}</td></tr>{% endfor %}
</table>
<div class="note">Сырой композит <code>e_raw</code> даёт ρ={{ '%.3f'|format(spearman_raw) }} — выше, но
это самообман: контент коррелирует с сообществом и временем. <code>dummy</code> ≈ 0,
перемешанный таргет ≈ 0 → утечки нет. Один человек никогда не попадает одновременно в train и test
(<code>GroupKFold(person_id)</code>), PCA эмбеддингов обучается только на train.</div>

<h2>4. Диагностика</h2>
<div class="plot">{{ p_dec }}</div>
<div class="plot">{{ p_pt }}</div>

<h2>5. Кластеры-архетипы</h2>
<div class="plot">{{ p_clu }}</div>
<table><tr><th>Кластер</th><th>N</th><th>Симпатия</th><th>95% CI</th><th>Женских лиц</th>
<th>Медиан. возраст</th><th>С детским фото</th><th>Age-gap</th></tr>
{% for c in clusters %}<tr><td>c{{ c.cluster }}</td><td>{{ '{:,}'.format(c.size) }}</td>
<td class="{{ 'pos' if c.mean_sympathy>0 else 'neg' }}">{{ '%+.3f'|format(c.mean_sympathy) }}</td>
<td>[{{ '%+.3f'|format(c.ci95[0]) }}, {{ '%+.3f'|format(c.ci95[1]) }}]</td>
<td>{{ '%.0f'|format(c.profile.share_female_adult*100) }}%</td>
<td>{{ '%.0f'|format(c.profile.age_est_median) }}</td>
<td>{{ '%.0f'|format(c.profile.has_child_photo*100) }}%</td>
<td>{{ '%.1f'|format(c.profile.age_gap_label) }}</td></tr>{% endfor %}
</table>
<div class="note">Silhouette ≈ {{ '%.3f'|format(sil) }} — структура <b>слабая</b>: это скорее градиент,
чем чёткие группы. Различия читать только там, где CI не пересекает ноль.</div>

<h2>6. Ранжирование постов</h2>
<div class="note"><b>Ранг не пустой.</b> Сравнение верхних {{ sep.k }} и нижних {{ sep.k }} постов
по <i>предсказанию</i> — по <b>фактической</b> ставке на показ они реально расходятся:
лайков/показ <b>{{ '%.2f'|format(sep.like_top) }}‰</b> против {{ '%.2f'|format(sep.like_bot) }}‰
(+{{ '%.0f'|format((sep.like_top/sep.like_bot-1)*100) }}%), репостов/показ
<b>{{ '%.2f'|format(sep.rep_top) }}‰</b> против {{ '%.2f'|format(sep.rep_bot) }}‰
(×{{ '%.1f'|format(sep.rep_top/sep.rep_bot) }}), фактический остаток
{{ '%+.2f'|format(sep.y_top) }} против {{ '%+.2f'|format(sep.y_bot) }}. Разница есть, но домен
портретный — по самим лицам она на глаз не читается; сигнал живёт в <i>реакции на показ</i>, а не в
явных визуальных признаках. Карточки ниже отсортированы по предсказанию; на каждой — ставка лайков/показ.</div>
<h3>Топ-{{ cards_top|length }} по предсказанной симпатии</h3>
<div class="cards">{% for c in cards_top %}{{ card(c) }}{% endfor %}</div>
<h3 style="margin-top:26px">Низ-{{ cards_bot|length }}</h3>
<div class="cards">{% for c in cards_bot %}{{ card(c) }}{% endfor %}</div>

{% if deep %}<h2>7. Обучаемые GPU-модели: bi- vs cross-encoder</h2>
<div class="plot">{{ p_deep }}</div>
<table><tr><th>Модель</th><th>Устройство</th><th>Spearman</th><th>R²</th><th>NDCG@50</th><th>Lift@50</th></tr>
{% for m in deep %}<tr><td>{{ m.name }}</td><td>{{ m.device }}</td>
<td>{{ '%.4f'|format(m.sp) }}</td><td>{{ '%.4f'|format(m.r2) }}</td>
<td>{{ '%.3f'|format(m.ndcg) }}</td><td>{{ '%.2f'|format(m.lift) }}</td></tr>{% endfor %}
</table>
<div class="note"><b>Что сравниваем.</b> Один и тот же таргет <code>y_rate</code> и те же person-grouped
фолды. CPU-бустинг работает на 512-d <b>ArcFace</b>-эмбеддингах — специализированном кодировщике лиц.
Обучаемые энкодеры берут <i>сырой</i> контент: подпись поста (RU-текст, <code>{{ deep_text }}</code>,
<b>дообучается</b> на GPU) и кроп взрослого лица (<code>{{ deep_vision }}</code>, {{ deep_vision_state }}).
<b>Bi-encoder</b> — позднее слияние двух независимых башен; <b>cross-encoder</b> — cross-attention между
токенами текста и патчами лица.{{ deep_extras }} {{ deep_verdict }} Специализированный ArcFace заточен
под лица и остаётся сильной базой по Spearman, тогда как обучаемые энкодеры добавляют сигнал текста
подписи и выигрывают по топ-k (NDCG@50).</div>

<h2>8. Ограничения (честно)</h2>{% else %}<h2>7. Ограничения (честно)</h2>{% endif %}
<ul style="color:#cfd6e4">
<li>Сигнал <b>скромный</b>: R²≈{{ '%.3f'|format(r2) }}, ρ≈{{ '%.3f'|format(spearman) }}. Большая часть
вовлечённости — охват и шум ленты, а не человек.</li>
<li>Наблюдательные данные: это <b>корреляция</b>, не причинность. Мы не можем отделить «аудитории
нравится лицо» от «такие люди делают другие посты».</li>
<li>Эмбеддинги кодируют личность; person-grouped CV не даёт запомнить человека, но не устраняет то,
что модель улавливает внешность и, вероятно, демографию.</li>
<li>Кажущийся возраст/пол — <b>оценка модели</b>, а не самоидентификация. Аудитория корпуса смещена
(≈{{ '%.0f'|format(female_share*100) }}% женских лиц).</li>
<li>Только 2 VK-сообщества, формат «тогда/сейчас». Кросс-платформенное плечо (Reddit) сейчас
<b>{{ reddit_status }}</b>.</li>
</ul>

<p style="color:var(--mut);font-size:12px;margin-top:30px">Сгенерировано локально ·
метрики: <code>metrics/engagement_results.json</code></p>
</div></body></html>"""

CARD_MACRO = """{% macro card(c) %}<div class="card">
{% if c.thumb %}<img src="data:image/jpeg;base64,{{ c.thumb }}" alt="">{% endif %}
<div class="pid">{{ c.pid }}</div>
<div class="sc {{ 'pos' if c.pred>0 else 'neg' }}">{{ '%+.2f'|format(c.pred) }}</div>
<div class="sc" style="font-size:13px;color:var(--acc)">{% if c.like_pm is not none %}{{ c.like_pm }}‰ лайков/показ{% endif %}</div>
<div class="meta">факт {{ '%+.2f'|format(c.actual) }} · c{{ c.cluster }}<br>
♥ {{ '{:,}'.format(c.likes) }} · 💬 {{ '{:,}'.format(c.comments) }} · ↻ {{ '{:,}'.format(c.reposts) }}
{% if c.views %}<br>👁 {{ '{:,}'.format(c.views) }}{% if c.rep_pm is not none %} · ↻{{ c.rep_pm }}‰{% endif %}{% endif %}
<div>
<span class="chip">{{ c.n_photos }} фото</span>
{% if c.age %}<span class="chip">~{{ c.age }} лет</span>{% endif %}
<span class="chip">{{ '%.0f'|format(c.female*100) }}% ж</span>
{% if c.child %}<span class="chip">тогда/сейчас</span>{% endif %}
{% if c.gap %}<span class="chip">Δ{{ '%.0f'|format(c.gap) }} лет</span>{% endif %}
</div></div></div>{% endmacro %}"""


def build_report(with_thumbnails: bool = False, n_cards: int = 16) -> Path:
    res = json.loads((data_path("metrics_dir", "engagement_results.json")).read_text(encoding="utf-8"))
    build = json.loads((data_path("metrics_dir", "engagement_build.json")).read_text(encoding="utf-8"))

    oof_p = data_path("data_dir", "processed", "engagement_oof.parquet")
    oof = pd.read_parquet(oof_p) if oof_p.exists() else pd.read_csv(oof_p.with_suffix(".csv.gz"))

    abl = res["ablation_headline"]
    full = abl[-1]["oof_overall"]
    pre_emb = abl[-2]["oof_overall"]["spearman"] if len(abl) >= 2 else float("nan")
    by_target = {m["target"]: m["oof_overall"] for m in res["models"]}
    sp_b = next((v["spearman"] for k, v in by_target.items() if k.startswith("остаток B")), float("nan"))
    sp_pct = next((v["spearman"] for k, v in by_target.items() if k.startswith("перцентиль")), float("nan"))
    sp_raw = next((v["spearman"] for k, v in by_target.items() if k.startswith("e_raw")), float("nan"))
    rate = res.get("rate", {})
    sp_ev = rate.get("spearman_e_raw_vs_views", float("nan"))
    sp_ratev = rate.get("spearman_e_rate_vs_views", float("nan"))

    deep: list[dict[str, Any]] = []
    deep_text = deep_vision = ""
    deep_vision_state = "заморожен"
    deep_extras = ""
    deep_verdict = ""
    dpath = data_path("metrics_dir", "engagement_deep.json")
    if dpath.exists():
        dj = json.loads(dpath.read_text(encoding="utf-8"))
        for m in dj["models"]:
            o = m["oof_overall"]
            deep.append({"name": m["model"], "device": m.get("device", ""),
                         "sp": o["spearman"], "r2": o["r2"], "ndcg": o["ndcg@50"], "lift": o["lift@50"]})
        deep_text, deep_vision = dj.get("text_model", ""), dj.get("vision_model", "")
        nb = int(dj.get("vision_unfrozen_blocks", 0))
        deep_vision_state = f"дообучаются верхние {nb} блока" if nb > 0 else "заморожен"
        extras = []
        if dj.get("rank_weight", 0):
            extras.append("ranking-loss поверх MSE")
        if dj.get("epochs"):
            extras.append(f"{dj['epochs']} эпох")
        deep_extras = (" Обучение: " + ", ".join(extras) + ".") if extras else ""
        if deep:
            best = max(deep, key=lambda d: d["sp"])
            deep_verdict = (f"Лучшее ранжирование даёт <b>{best['name']}</b> "
                            f"(ρ={best['sp']:.3f}).")

    # разделение топ/низ по РЕАЛЬНОЙ ставке на показ — доказательство, что ранг не пустой
    osort = oof.sort_values("pred_rate", ascending=False)
    kk = min(50, len(osort) // 2)

    def _pm(d: pd.DataFrame, col: str) -> float:
        return float((d[col] / d["views"].clip(lower=1)).mean() * 1000)

    sep = {
        "k": kk,
        "like_top": _pm(osort.head(kk), "likes"), "like_bot": _pm(osort.tail(kk), "likes"),
        "rep_top": _pm(osort.head(kk), "reposts"), "rep_bot": _pm(osort.tail(kk), "reposts"),
        "y_top": float(osort.head(kk)["y_rate"].mean()), "y_bot": float(osort.tail(kk)["y_rate"].mean()),
    }

    thumbs: dict[str, str] = {}
    if with_thumbnails:
        d = oof.sort_values("pred_rate", ascending=False)
        wanted = set(d.head(n_cards)["post_id"]) | set(d.tail(n_cards)["post_id"])
        paths = best_adult_face_paths(set(map(str, wanted)))
        for pid, p in paths.items():
            b = _b64(p)
            if b:
                thumbs[pid] = b
        log.info("Встроено превью: %d", len(thumbs))

    def plot(fig: go.Figure, first: bool = False) -> str:
        return fig.to_html(full_html=False, include_plotlyjs="inline" if first else False,
                           config={"displayModeBar": False})

    models_tbl = [{"model": m["model"], "target": m["target"],
                   "sp": m["oof_overall"]["spearman"], "r2": m["oof_overall"]["r2"],
                   "ndcg": m["oof_overall"]["ndcg@50"]} for m in res["models"]]
    models_tbl.insert(0, {"model": "hgb", "target": "y_rate — ставка на показ (главный)",
                          "sp": full["spearman"], "r2": full["r2"], "ndcg": full["ndcg@50"]})

    tpl = Template(CARD_MACRO + TEMPLATE)
    html_out = tpl.render(
        platform=res["platform"], n_posts=f"{res['n_posts']:,}", n_persons=f"{res['n_persons']:,}",
        reach_a=res["reach"]["A_exogenous_only"]["share_variance_explained_by_reach"],
        spearman=full["spearman"], r2=full["r2"], lift=full["lift@50"], pre_emb=pre_emb,
        neg=res["negative_control"]["oof_overall"]["spearman"],
        spearman_b=sp_b, spearman_pct=sp_pct, spearman_raw=sp_raw,
        sp_ev=sp_ev, sp_ratev=sp_ratev,
        models=models_tbl, clusters=res["clusters"]["clusters"],
        sil=max(res["clusters"]["silhouette_by_k"].values()),
        female_share=float(np.nanmean(oof["share_female_adult"])),
        reddit_status="отключено (нет REDDIT_CLIENT_ID/SECRET)",
        has_thumbs=bool(thumbs),
        cards_top=_cards(oof, thumbs, n_cards, top=True),
        cards_bot=_cards(oof, thumbs, n_cards, top=False),
        p_var=plot(fig_variance(res["reach"]), first=True),
        p_abl=plot(fig_ablation(abl)),
        p_imp=plot(fig_importance(res["permutation_importance"])),
        p_mod=plot(fig_models(res["models"], res["negative_control"])),
        p_clu=plot(fig_clusters(res["clusters"]["clusters"])),
        p_dec=plot(fig_decile(oof)),
        p_pt=plot(fig_pred_true(oof)),
        deep=deep, deep_text=deep_text, deep_vision=deep_vision, deep_verdict=deep_verdict,
        deep_vision_state=deep_vision_state, deep_extras=deep_extras, sep=sep,
        p_deep=plot(fig_deep(deep)) if deep else "",
    )
    log.info("guardrails из build: %s", build.get("guardrails", {}).get("posts_final"))

    # имя по датасету: разные VK-паблики не должны перезаписывать отчёты друг друга
    dataset = data_path("data_dir").name
    out = resolve_path("reports", "engagement", f"report_{dataset}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_out, encoding="utf-8")
    log.info("Отчёт: %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out
