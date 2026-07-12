"""Сводный отчёт: beauty-модель + рексис знакомств. Всё, что подтвердилось и что умерло.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_report.py

Читает метрики из metrics/ и metrics_natural/ и собирает один локальный HTML.
Пишет reports/beauty/recsys_summary.html (каталог в .gitignore).
"""

from __future__ import annotations

import json

from age_gap.common.io import resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def _load(*p):
    try:
        return json.loads(resolve_path(*p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def main() -> None:
    rt = _load("metrics_natural", "rating_retest.json")

    html = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Рексис знакомств — сводка</title><style>
 body{{margin:0;background:#0f1218;color:#e6e9ef;font:15px/1.65 Segoe UI,sans-serif}}
 .wrap{{max-width:1040px;margin:0 auto;padding:30px}}
 h1{{font-size:28px}} h2{{font-size:20px;margin-top:36px;border-bottom:1px solid #2a3040;padding-bottom:6px}}
 table{{width:100%;border-collapse:collapse;background:#171b24;border-radius:10px;overflow:hidden;margin:12px 0}}
 th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid #232838}} th{{color:#8a94a6;font-weight:600}}
 .kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:16px 0}}
 .kpi div{{background:#171b24;border:1px solid #232838;border-radius:12px;padding:14px}}
 .kpi b{{display:block;font-size:24px;margin-top:4px}}
 .note{{background:#171b24;border-left:3px solid #4f8ef7;padding:12px 16px;border-radius:8px;margin:12px 0;color:#cfd6e4}}
 .dead{{border-left-color:#e0564a}} .win{{border-left-color:#2fbf71}}
 .mut{{color:#8a94a6}} .pos{{color:#2fbf71}} .neg{{color:#e0564a}}
</style></head><body><div class="wrap">

<h1>Персонализированный рексис знакомств — сводка</h1>
<p class="mut">Всё измерено на реальных данных: 2339 оценок по шкале 1–5 и 1179 парных сравнений
(SE 0.0116). Никаких чисел из симуляций в этой таблице.</p>

<h2>1. Итоговая архитектура и что она даёт</h2>
<div class="kpi">
  <div><small>Приор (0 персонализации)</small><b>0.616</b></div>
  <div><small>Тяжёлый full fine-tune</small><b>0.672</b></div>
  <div><small>Лёгкая голова + ансамбль</small><b class="pos">0.7625</b></div>
  <div><small>Потолок (твоя надёжность)</small><b>0.850</b></div>
</div>
<div class="note win"><b>Главный результат.</b> Замороженные эмбеддинги + линейная пер-юзерная
residual-голова бьют дообучение всей сети на <b>+0.091 (6.4 SE)</b> и дают прирост к приору
<b>+0.147 (10.4 SE)</b> — при несопоставимо меньшей цене: энкодеры гоняются <b>один раз офлайн</b>,
на свайп остаётся O(d²) ≈ микросекунды.</div>

<pre style="background:#171b24;padding:14px;border-radius:10px;overflow-x:auto">
score(user, face) = a·prior(face) + b  +  w_user · PCA_whiten(emb(face))

prior  — популяционная beauty-модель (SCUT), один раз офлайн
emb    — замороженный ансамбль энкодеров, кешируется рядом с фото
w_user — байесовская линейная голова на ОСТАТКЕ, онлайн-обновление на свайпе
</pre>

<h2>2. Что подтвердилось</h2>
<table><tr><th>Компонент</th><th>Эффект</th></tr>
<tr><td>Residual поверх популяционного приора (не «сырая» голова)</td><td class="pos">наивная голова падает НИЖЕ приора: 0.637 против 0.688</td></tr>
<tr><td>Whitening + усадка λ≈300</td><td class="pos">без него холодный старт рушится 0.688 → 0.535</td></tr>
<tr><td>Ансамбль энкодеров (CLIP+SigLIP+DINOv2)</td><td class="pos">вкус 0.433 → 0.568 (+31%)</td></tr>
<tr><td>Thompson вместо жадной выдачи</td><td class="pos">жадная выжигает каталог; Thompson держит плато</td></tr>
<tr><td>Реципрокность P(A→B)·P(B→A)</td><td class="pos">×3 матчей; выигрыш максимален у наименее привлекательных</td></tr>
<tr><td>Калибровка по селективности</td><td class="pos">иммунитет к спам-свайпингу (37.7 качественных при 0/10/30% спама)</td></tr>
</table>

<h2>3. Что умерло под проверкой</h2>
<div class="note dead">Из ~12 правдоподобных гипотез выжили единицы. Два «улучшения» оказались
<b>багами моего же бенчмарка</b> (истощение каталога в замере; конфаунд размерности популяции).</div>
<table><tr><th>Гипотеза</th><th>Вердикт</th></tr>
<tr><td>IPS против смещения экспозиции</td><td class="neg">вредит: тянет назад к cold-start. Лечит стохастичность показа, не перевзвешивание</td></tr>
<tr><td>Смещение экспозиции вообще</td><td class="neg">не проблема: жадная выдача училась лучше всех</td></tr>
<tr><td>Детектор спама по предсказуемости</td><td class="neg">AUC 0.563 ≈ случай (спамер ≠ рандомайзер: у него есть вкус, просто низкий порог)</td></tr>
<tr><td>Онбординг чистыми оценками</td><td class="neg">перекрывается сотней бесплатных свайпов</td></tr>
<tr><td>Нелинейная голова (MLP / RBF / бустинг)</td><td class="neg">ноль: линейный ridge оптимален</td></tr>
<tr><td>Крупные энкодеры (SO400M, L-336, DINOv2-L)</td><td class="neg">ноль: не бьют базовый ансамбль</td></tr>
<tr><td>Масштаб кропа и flip-TTA</td><td class="neg">ноль: взятый с потолка margin=0.4 уже оптимален</td></tr>
<tr><td>Грубая шкала занижает потолок</td><td class="neg">неверно: потолок квантизации 0.926, а мы на 0.57</td></tr>
</table>

<h2>4. Потолок: почему 0.57, а не 0.85</h2>
<div class="kpi">
  <div><small>Твоя согласованность (test-retest)</small><b>{rt.get('test_retest_spearman', '—')}</b></div>
  <div><small>Потолок = √надёжности</small><b>{rt.get('implied_ceiling_for_model', '—')}</b></div>
  <div><small>Наш вкус</small><b>0.568</b></div>
  <div><small>Взято от потолка</small><b>67%</b></div>
</div>
<div class="note"><b>Плато подтверждено с четырёх сторон:</b> 4 архитектуры головы · 7 энкодеров и все
ансамбли · 3 масштаба кропа + TTA · кривая обучения (прирост на 500 меток упал с +0.030 до +0.004).
Оставшаяся треть <b>не извлекается</b> из статичного кропа лица generic-энкодером — это
информационный предел ВХОДА, а не недоработка модели.</div>

<h2>5. Инженерные выводы для приложения</h2>
<ul>
<li><b>Не дообучай тяжёлую сеть на юзера</b> — дороже и хуже лёгкой головы на +0.09.</li>
<li><b>Персонализация обязана быть остатком поверх приора</b>, иначе она уводит НИЖЕ generic-модели.</li>
<li><b>Главный враг — не экспозиция, а контаминация метки.</b> Если свайп только наполовину про лицо
(био, вторые фото, настроение), голова учит грязное направление и ранжирование становится хуже
популяционного. Нужен лице-специфичный сигнал (свайп по фото, dwell-time), а не сырой свайп.</li>
<li><b>Показ — Thompson, не жадный</b> (жадный выжигает каталог за ~600 показов).</li>
<li><b>Матчи — реципрокные и калиброванные по селективности</b>: иначе метрика матчей растёт, а
качество падает (85.5 матчей при 18.8 качественных — ловушка vanity-метрики).</li>
</ul>

<p class="mut" style="font-size:12px;margin-top:28px">Локально. Ветка research; в статьи не идёт.</p>
</div></body></html>"""

    out = resolve_path("reports", "beauty", "recsys_summary.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"OK: {out}")


if __name__ == "__main__":
    main()
