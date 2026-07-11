"""Сводный локальный HTML-отчёт по beauty-ветке (модель красоты + научный анализ).

    uv run python scripts/beauty_report.py

Собирает метрики из metrics/ и metrics_natural/ (что найдёт) в один отчёт:
бенчмарк SCUT, согласие с твоей разметкой, «вовлечённость ≠ красота», beauty→вовлечённость
по доменам (net-of-age/пол), ссылки на галереи. Пишет reports/beauty/summary.html (gitignored).
"""

from __future__ import annotations

import json

from age_gap.common.io import resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def _load(*parts):
    p = resolve_path(*parts)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _sp(d, *keys, default="—"):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _fmt(v, f="%.3f"):
    try:
        return f % float(v)
    except Exception:  # noqa: BLE001
        return "—"


def main() -> None:
    scut_d = _load("metrics", "beauty_scut_dinov2.json")
    scut_c = _load("metrics", "beauty_scut.json")
    rating = _load("metrics", "beauty_rating.json")
    rerank = _load("metrics", "beauty_rerank.json")
    engd = _load("metrics_natural", "engagement_dinov2.json")
    vk_nat = _load("metrics_natural", "beauty_vk.json")
    vk_tn = _load("metrics", "beauty_vk.json")

    def corr_rows(vk, name):
        if not vk:
            return f"<tr><td>{name}</td><td colspan=5 class=mut>нет данных</td></tr>"
        c = vk["correlations"]
        return (f"<tr><td>{name}</td>"
                f"<td>{vk.get('n_posts_matched', '—')}</td>"
                f"<td>{_fmt(_sp(c, 'beauty_max_vs_e_rate', 'spearman'))}</td>"
                f"<td>{_fmt(_sp(c, 'beauty_vs_e_rate__net_age_gender', 'spearman'))}</td>"
                f"<td>{_fmt(_sp(c, 'beauty_max_vs_age', 'spearman'))}</td>"
                f"<td>{_fmt(_sp(c, 'beauty_max_vs_like_per_view', 'spearman'))}</td></tr>")

    html = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Beauty — сводка</title><style>
 body{{margin:0;background:#0f1218;color:#e6e9ef;font:15px/1.6 Segoe UI,sans-serif}}
 .wrap{{max-width:1000px;margin:0 auto;padding:28px}}
 h1{{font-size:28px}} h2{{font-size:20px;margin-top:34px;border-bottom:1px solid #2a3040;padding-bottom:6px}}
 table{{width:100%;border-collapse:collapse;background:#171b24;border-radius:10px;overflow:hidden;margin:12px 0}}
 th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid #232838}} th{{color:#8a94a6}}
 .kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0}}
 .kpi div{{background:#171b24;border:1px solid #232838;border-radius:12px;padding:14px}}
 .kpi b{{display:block;font-size:24px;margin-top:4px}} .mut{{color:#8a94a6}}
 .note{{background:#171b24;border-left:3px solid #4f8ef7;padding:12px 16px;border-radius:8px;margin:12px 0;color:#cfd6e4}}
 a{{color:#4f8ef7}} .pos{{color:#2fbf71}} .neg{{color:#e0564a}}</style></head><body><div class="wrap">
<h1>Модель привлекательности лица — сводка</h1>
<p class="mut">Отдельная research-ветка. Пивот с «вовлечённости» на моделирование красоты по пикселям.
Всё локально, в статьи не идёт.</p>

<h2>1. Beauty-модель (бенчмарк SCUT-FBP5500 — человеческие оценки)</h2>
<div class="kpi">
  <div><small>DINOv2, Pearson</small><b class="pos">{_fmt(_sp(scut_d, 'cv_mean_std', 'pearson', 'mean'))}</b></div>
  <div><small>DINOv2, Spearman</small><b>{_fmt(_sp(scut_d, 'cv_mean_std', 'spearman', 'mean'))}</b></div>
  <div><small>CLIP-B, Pearson</small><b>{_fmt(_sp(scut_c, 'cv_mean_std', 'pearson', 'mean'))}</b></div>
</div>
<div class="note">Красоту по пикселям лица можно предсказывать на уровне SOTA — но метка обязана быть
человеческой (SCUT). Это внешний бенчмарк валидности.</div>

<h2>2. Перенос на VK — согласие с твоей разметкой (тест)</h2>
<div class="kpi">
  <div><small>SCUT-модель vs твои пары</small><b class="pos">{_fmt(_sp(rating, 'pairwise_accuracy'))}</b></div>
  <div><small>Re-ranker на 215 парах</small><b>{_fmt(_sp(rerank, 'reranker_heldout_acc'))}</b></div>
  <div><small>Пар размечено</small><b>{_sp(rating, 'n_decisive')}</b></div>
</div>
<div class="note">SCUT-модель угадывает твоё предпочтение в {_fmt(_sp(rating, 'pairwise_accuracy'), '%.0f%%').replace('0.','')}
пар (0.5=случай). Дообучение на 215 парах baseline не побило ({_fmt(_sp(rerank, 'reranker_heldout_acc'))}) —
меток мало, они лучше как тест.</div>

<h2>3. Вовлечённость ≠ красота (твой дизайн, измерено)</h2>
<table><tr><th>Метрика</th><th>Значение</th></tr>
<tr><td>DINOv2 учит композит вовлечённости (held-out persons)</td><td>{_fmt(_sp(engd, 'val_spearman_engagement'))}</td></tr>
<tr><td>Валидация на SCUT (vs человеческая красота)</td><td class="neg">{_fmt(_sp(engd, 'scut_validation', 'spearman_vs_human_beauty'))}</td></tr>
<tr><td>Тест на твоих парах (pairwise acc)</td><td class="neg">{_fmt(_sp(engd, 'pairs_test', 'pairwise_accuracy'))}</td></tr>
<tr><td>— для сравнения: SCUT-beauty модель на тех же парах</td><td class="pos">{_fmt(_sp(rating, 'pairwise_accuracy'))}</td></tr>
</table>
<div class="note">Композит лайков/просмотров отлично учится с лица, но это <b>отклик аудитории</b>, а не
красота: с человеческой красотой SCUT совпадает лишь на {_fmt(_sp(engd, 'scut_validation', 'spearman_vs_human_beauty'))},
а с твоими оценками — на {_fmt(_sp(engd, 'pairs_test', 'pairwise_accuracy'))} (≈случай). Твоя же разметка это подтверждает.</div>

<h2>4. Красота → вовлечённость по доменам (net-of-age/пол)</h2>
<table><tr><th>Домен</th><th>Постов</th><th>beauty↔e_rate</th><th>…при контроле age+пол</th><th>beauty↔age</th><th>beauty↔лайк/показ</th></tr>
{corr_rows(vk_tn, "then/now (широкий разброс)")}
{corr_rows(vk_nat, "natural (курир., сжатый)")}
</table>
<div class="note">Если связь красота→вовлечённость близка к нулю в обоих доменах и не растёт на широком
then/now — эффект не объясняется range-restriction'ом; красота просто слабо гонит per-view реакцию.
Контроль возраста/пола показывает чистый вклад внешности.</div>

<h2>5. Галереи (глазами)</h2>
<ul>
<li><a href="../engagement/beauty_gallery_data_natural.html">natural: топ/низ по красоте</a></li>
<li><a href="../engagement/beauty_gallery_data.html">then/now: топ/низ по красоте</a></li>
<li><a href="../rating/rate_data.html">инструмент разметки пар</a></li>
</ul>

<h2>6. Ограничения (честно)</h2>
<ul class="mut">
<li>Beauty-супервизия — SCUT (студийные, азиато-смещён); домен VK иной. Перенос ~0.69 на твой вкус.</li>
<li>Твоя разметка — один оценщик, субъективна (это и есть цель — твой вкус).</li>
<li>Вовлечённость на VK — не метка красоты (доказано); красота ≠ реакция аудитории.</li>
<li>Кропы hi-res из оригиналов; несовершеннолетние исключены (age≥18); всё локально.</li>
</ul>
<p class="mut" style="font-size:12px">Сгенерировано локально из metrics/ и metrics_natural/.</p>
</div></body></html>"""

    out = resolve_path("reports", "beauty", "summary.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"OK: {out}")


if __name__ == "__main__":
    main()
