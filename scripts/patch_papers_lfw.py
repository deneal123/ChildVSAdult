"""Разовая правка всех статей: выровненный LFW + снятие ложной оговорки + трудные негативы.

    uv run python scripts/patch_papers_lfw.py [--dry-run]

ЧТО ИСПРАВЛЯЕМ И ПОЧЕМУ.

1. LFW считался через sklearn ``fetch_lfw_pairs``: изображения БЕЗ 5-точечного выравнивания и,
   хуже, обрезанные до 125x94 вместо кадра 250x250 (параметр slice_). ArcFace r100 давал на нём
   0.8298 при 0.9835 на более сложном AgeDB-30 — невозможное сочетание, которое и выдало баг.
   Пересчитано на выровненных кропах (RetinaFace 5 точек -> norm_crop 112px, промахи 0.36%).

2. Следствие: заголовочная оговорка статьи оказалась артефактом замера.
       было:  LFW 0.9688 -> 0.9503 (-0.0185), TAR@0.1% 0.858 -> 0.575 («существенная деградация»)
       стало: LFW 0.9640 -> 0.9679 (+0.0039 по трём сидам, std 0.0002), TAR@0.1% 0.814 -> 0.816
   Забывания нет ни по одной метрике LFW. AgeDB-30 (-0.008) и CALFW (без изменений) не трогались —
   они всегда читались из выровненного .bin. Статья НЕДООЦЕНИВАЛА собственный результат.

3. Кривая headroom интерпретировалась как «у сильных backbone нет запаса». Это верно только для
   СЛУЧАЙНЫХ импостеров: с похожими негативами IR-101 падает с 0.956 до 0.705 (rank-10) и 0.474
   (rank-1). Значит запас есть, просто НАШИ данные его не берут — другое утверждение.

Сопоставление устойчиво к переносам строк (пробелы и \n трактуются одинаково), потому что
конференционные версии свёрстаны с жёсткими переносами. Скрипт идемпотентен.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUPPLEMENTS = [
    "latex/papers/journal-1-tbiom/en/supplement.tex",
    "latex/papers/journal-1-tnnls/en/supplement.tex",
    "latex/conference/aaai-27/en/supplement.tex",
    "latex/conference/aaai-27/ru/supplement_ru.tex",
]

PAPERS = [
    "latex/papers/journal-1-tbiom/en/main.tex",
    "latex/papers/journal-1-tbiom/ru/main_ru.tex",
    "latex/papers/journal-1-tnnls/en/main.tex",
    "latex/papers/journal-1-tnnls/ru/main_ru.tex",
    "latex/conference/aaai-27/en/main.tex",
    "latex/conference/aaai-27/ru/main_ru.tex",
]


def sub(text: str, old: str, new: str) -> tuple[str, bool]:
    """Замена, нечувствительная к тому, как текст разбит на строки."""
    if new.strip() and re.sub(r"\s+", " ", new) in re.sub(r"\s+", " ", text):
        return text, True                      # уже применено (идемпотентность)
    pat = r"\s+".join(re.escape(t) for t in old.split())
    new_text, n = re.subn(pat, lambda _m: new, text, count=1)
    return new_text, n > 0


# ---------------------------------------------------------------- строки таблиц (общие)
TABLES = [
    (r"LFW accuracy & 0.9688 & 0.9503 $\pm$ 0.0031 & $-$0.0185 & $-$0.0283 \\",
     r"LFW accuracy & 0.9640 & 0.9679 $\pm$ 0.0002 & $+$0.0039 & n/a \\"),
    (r"LFW (AUC) & 0.994 [0.99, 1.00] & 0.987 [0.98, 0.99] & 0.032$\to$0.052 & 0.940$\to$0.869 "
     r"& 0.858$\to$0.575 \\",
     r"LFW (AUC) & 0.983 [0.98, 0.99] & 0.986 [0.98, 0.99] & 0.038$\to$0.037 & 0.939$\to$0.940 "
     r"& 0.814$\to$0.816 \\"),
]

# ---------------------------------------------------------------- английский текст
EN = [
    (r"at the cost of $-$0.019 LFW accuracy and \emph{substantial} low-FAR degradation---a "
     r"targeted retrieval gain, not a general-verifier upgrade.",
     r"at no measurable cost on easy benchmarks (LFW accuracy $+$0.004, LFW TAR@FAR$=$0.1\% "
     r"$+$0.002; AgeDB-30 $-$0.008)---a gain concentrated on the large-gap regime rather than a "
     r"trade-off."),
    (r"while easy LFW falls only $-$0.019 and AgeDB-30/CALFW move negligibly",
     r"while easy LFW does not degrade at all ($+$0.004) and AgeDB-30/CALFW move negligibly"),
    # журнальный вариант обсуждения
    (r"Crucially, the improvement is a \emph{targeted} large-gap gain, not a drop-in upgrade to a "
     r"general verifier: low-FAR operating points on easy benchmarks degrade (LFW TAR@FAR$=$0.1\% "
     r"falls $0.858 \rightarrow 0.575$; Table~\ref{tab:stats}), so deployment should be scoped to "
     r"the large-gap cross-age regime rather than substituted for a general-purpose model.",
     r"The improvement is \emph{concentrated} on the large-gap regime rather than uniform, but it "
     r"is not a trade-off: on correctly aligned LFW every operating point is flat or better "
     r"(TAR@FAR$=$0.1\% $0.814 \rightarrow 0.816$, EER $0.038 \rightarrow 0.037$; "
     r"Table~\ref{tab:stats}), and CALFW improves at low FAR ($+$0.106 at TAR@FAR$=$0.1\%). Only "
     r"AgeDB-30 shows a small significant decrease ($-$0.008). An earlier version of this analysis "
     r"reported substantial low-FAR degradation; that was an artifact of an unaligned LFW "
     r"evaluation path (Sec.~\ref{sec:method}) and does not survive correction."),
    # конференционный (короче)
    (r"Crucially, the improvement is a \emph{targeted} large-gap gain, not a drop-in upgrade to a "
     r"general verifier: low-FAR operating points on easy benchmarks degrade (LFW TAR@FAR$=$0.1\% "
     r"falls $0.858 \rightarrow 0.575$), so deployment should be scoped to the large-gap "
     r"cross-age regime.",
     r"The improvement is \emph{concentrated} on the large-gap regime rather than uniform, but it "
     r"is not a trade-off: on correctly aligned LFW every operating point is flat or better "
     r"(TAR@FAR$=$0.1\% $0.814 \rightarrow 0.816$), and CALFW improves at low FAR. An earlier "
     r"version of this analysis reported substantial low-FAR degradation; that was an artifact of "
     r"an unaligned LFW evaluation path and does not survive correction."),
    (r"Operating points (Table~\ref{tab:stats}) make the trade-off concrete: large-gap AUC and "
     r"TAR@1\% rise sharply, while easy-benchmark low-FAR points (LFW TAR@0.1\%) degrade.",
     r"Operating points (Table~\ref{tab:stats}) make the effect concrete: large-gap AUC and "
     r"TAR@1\% rise sharply, while easy-benchmark operating points stay flat."),
    (r"(a)~the effect is \emph{not} a uniform verification gain---low-FAR operating points on easy "
     r"benchmarks degrade (Table~\ref{tab:stats});",
     r"(a)~the effect is \emph{not} a uniform verification gain---it is concentrated on the "
     r"large-gap subset, with easy benchmarks flat (Table~\ref{tab:stats});"),
    (r"the effect is not a uniform verification gain (low-FAR operating points degrade);",
     r"the effect is not a uniform verification gain (it is concentrated on the large-gap subset);"),
    (r"(with moderate easy-benchmark accuracy loss but \emph{substantial} low-FAR degradation, "
     r"which scopes the method to targeted large-gap retrieval rather than general verification)",
     r"(without measurable degradation on easy benchmarks, so the gain is concentrated rather than "
     r"traded off)"),
    (r"\textbf{Backbone dependence:} the gain concentrates in weak/medium backbones; strong "
     r"saturated models do not improve and slightly forget (gentle lr mitigates but does not "
     r"reverse this).",
     r"\textbf{Backbone dependence:} the gain concentrates in weak/medium backbones; strong models "
     r"do not improve and slightly forget (gentle lr mitigates but does not reverse this). They "
     r"are saturated only against random impostors---with look-alike negatives they are far from "
     r"ceiling (Table~\ref{tab:lookalike}), so this bounds \emph{our} supervision, not the task."),
    (r"\textbf{Backbone dependence:} the gain concentrates in weak/medium backbones; strong "
     r"saturated models do not improve and slightly forget.",
     r"\textbf{Backbone dependence:} the gain concentrates in weak/medium backbones; strong models "
     r"do not improve and slightly forget. They are saturated only against random impostors---with "
     r"look-alike negatives they are far from ceiling, so this bounds \emph{our} supervision, not "
     r"the task."),
]

# ---------------------------------------------------------------- русский текст
RU = [
    (r"LFW точность & 0.9688 & 0.9503 $\pm$ 0.0031 & $-$0.0185 & $-$0.0283 \\",
     r"LFW точность & 0.9640 & 0.9679 $\pm$ 0.0002 & $+$0.0039 & n/a \\"),
    (r"ценой $-$0.019 точности на LFW и \emph{существенной} low-FAR-деградации --- это целевой "
     r"выигрыш для поиска, а не апгрейд общего верификатора.",
     r"без измеримых потерь на лёгких бенчмарках (точность LFW $+$0.004, LFW TAR@FAR$=$0.1\% "
     r"$+$0.002; AgeDB-30 $-$0.008) --- прирост сконцентрирован на large-gap-режиме, а не "
     r"разменян на потери."),
    (r"ценой $-$0.019 точности на LFW и \emph{существенной} деградации при низком FAR — это "
     r"целевой выигрыш для поиска, а не апгрейд универсального верификатора.",
     r"без измеримых потерь на лёгких бенчмарках (точность LFW $+$0.004, LFW TAR@FAR$=$0.1\% "
     r"$+$0.002; AgeDB-30 $-$0.008) — прирост сконцентрирован на режиме большого разрыва, а не "
     r"разменян на потери."),
    (r"тогда как точность на лёгком LFW падает лишь на $-$0.019, а AgeDB-30/CALFW почти не "
     r"двигаются",
     r"тогда как точность на лёгком LFW вовсе не падает ($+$0.004), а AgeDB-30/CALFW почти не "
     r"двигаются"),
    (r"при этом лёгкая LFW падает лишь на $-$0.019, а AgeDB-30/CALFW почти не меняются",
     r"при этом лёгкая LFW вовсе не падает ($+$0.004), а AgeDB-30/CALFW почти не меняются"),
    (r"низко-FAR рабочие точки на лёгких бенчмарках деградируют (LFW TAR@FAR$=$0.1\% падает "
     r"$0.858 \rightarrow 0.575$; Таблица~\ref{tab:stats}), поэтому развёртывание следует "
     r"ограничивать large-gap кросс-возрастным режимом, а не подменять им модель общего назначения.",
     r"улучшение \emph{сконцентрировано} на large-gap-режиме, а не равномерно, но это не размен: "
     r"на корректно выровненном LFW все рабочие точки не хуже прежних (TAR@FAR$=$0.1\% "
     r"$0.814 \rightarrow 0.816$, EER $0.038 \rightarrow 0.037$; Таблица~\ref{tab:stats}), а CALFW "
     r"на низком FAR улучшается ($+$0.106 по TAR@FAR$=$0.1\%). Небольшое значимое снижение даёт "
     r"только AgeDB-30 ($-$0.008). В более ранней версии этого анализа сообщалось о существенной "
     r"low-FAR-деградации; это был артефакт невыровненного пути оценки LFW "
     r"(Раздел~\ref{sec:method}), который не переживает исправления."),
    (r"точки низкого FAR на лёгких бенчмарках деградируют (LFW TAR@FAR$=$0.1\% падает "
     r"$0.858\rightarrow0.575$), поэтому развёртывание следует ограничивать режимом большого "
     r"кросс-возрастного разрыва.",
     r"улучшение \emph{сконцентрировано} на режиме большого разрыва, а не равномерно, но это не "
     r"размен: на корректно выровненном LFW все рабочие точки не хуже прежних (TAR@FAR$=$0.1\% "
     r"$0.814\rightarrow0.816$), а CALFW на низком FAR улучшается. В более ранней версии этого "
     r"анализа сообщалось о существенной low-FAR-деградации; это был артефакт невыровненного пути "
     r"оценки LFW, который не переживает исправления."),
    (r"(a)~эффект \emph{не} является равномерным приростом верификации --- низко-FAR рабочие точки "
     r"на лёгких бенчмарках деградируют (Таблица~\ref{tab:stats});",
     r"(a)~эффект \emph{не} является равномерным приростом верификации --- он сконцентрирован на "
     r"large-gap-подмножестве, а лёгкие бенчмарки не меняются (Таблица~\ref{tab:stats});"),
    (r"это не равномерный прирост верификации (точки низкого FAR деградируют);",
     r"это не равномерный прирост верификации (он сконцентрирован на подмножестве большого "
     r"разрыва);"),
    (r"(с умеренной потерей точности на лёгких бенчмарках, но \emph{существенной} "
     r"low-FAR-деградацией, что ограничивает метод целевым large-gap-поиском, а не общей "
     r"верификацией)",
     r"(без измеримой деградации на лёгких бенчмарках --- прирост сконцентрирован, а не разменян)"),
    (r"\textbf{Зависимость от backbone:} прирост сосредоточен у слабых/средних backbone; сильные "
     r"насыщенные не растут и слегка забывают (мягкий lr смягчает, но не обращает).",
     r"\textbf{Зависимость от backbone:} прирост сосредоточен у слабых/средних backbone; сильные "
     r"модели не растут и слегка забывают (мягкий lr смягчает, но не обращает). Они насыщены "
     r"только против случайных импостеров --- с похожими негативами они далеки от потолка "
     r"(Таблица~\ref{tab:lookalike}), так что это ограничение \emph{нашей} супервизии, а не задачи."),
    (r"\textbf{Зависимость от backbone:} прирост сосредоточен в слабых/средних backbone; сильные "
     r"насыщенные не улучшаются и слегка забывают.",
     r"\textbf{Зависимость от backbone:} прирост сосредоточен в слабых/средних backbone; сильные "
     r"модели не улучшаются и слегка забывают. Они насыщены только против случайных импостеров — "
     r"с похожими негативами они далеки от потолка, так что это ограничение \emph{нашей} "
     r"супервизии, а не задачи."),
]

# ---------------------------------------------------------------- методика LFW (дописывается к якорю)
LFW_METHOD_EN = (
    r" LFW is evaluated on \emph{aligned} crops: the pair images are re-detected with RetinaFace "
    r"and warped by the same 5-point \texttt{norm\_crop} used for our own crops (0.36\% detector "
    r"misses). This matters: the commonly used \texttt{sklearn} loader returns unaligned images "
    r"cropped to 125$\times$94, on which margin-trained recognizers collapse (frozen ArcFace r100 "
    r"scores 0.83 there while scoring 0.98 on the harder AgeDB-30). Absolute LFW values under our "
    r"alignment sit slightly below published ones and should not be compared to leaderboards; "
    r"frozen and fine-tuned models are measured identically, so the reported differences are valid."
)
LFW_METHOD_RU = (
    r" LFW считается на \emph{выровненных} кропах: изображения пар заново детектируются RetinaFace "
    r"и приводятся тем же 5-точечным \texttt{norm\_crop}, что и наши кропы (промахи детектора "
    r"0.36\%). Это существенно: широко используемый загрузчик \texttt{sklearn} отдаёт "
    r"невыровненные изображения, обрезанные до 125$\times$94, на которых распознаватели с "
    r"margin-обучением рушатся (замороженный ArcFace r100 даёт там 0.83 при 0.98 на более сложном "
    r"AgeDB-30). Абсолютные значения LFW при нашем выравнивании немного ниже опубликованных и не "
    r"должны сравниваться с лидербордами; frozen и дообученные модели меряются одинаково, поэтому "
    r"приведённые разности корректны."
)
METHOD_ANCHORS = {
    "en": [r"We distinguish \emph{threshold-free} metrics (ROC-AUC) from \emph{threshold-based} "
           r"ones (accuracy, TAR@FAR).",
           r"distinguishing \emph{threshold-free} metrics (ROC-AUC) from \emph{threshold-based} "
           r"ones (accuracy, TAR@FAR)."],
    "ru": [r"Различаем \emph{threshold-free} метрики (ROC-AUC) и \emph{threshold-based} "
           r"(accuracy, TAR@FAR).",
           r"различая \emph{беспороговые} метрики (ROC-AUC) и \emph{пороговые} (точность, TAR@FAR)."],
}

# ---------------------------------------------------------------- трудные негативы
HARDNEG_EN = r"""

\textbf{The headroom curve is measured against \emph{random} impostors.} Our test negatives are
random cross-identity pairs (plus a small age-controlled subset), which modern backbones separate
trivially: frozen AdaFace IR-101 reaches \texttt{our.25+} $=$ 0.968 and FG-NET large-gap 0.957,
above anything our pipeline attains. This also validates by measurement, rather than merely
asserts, the attribution argument of Sec.~\ref{sec:method}: on this task strong embeddings really
are near-perfect, so an adapter on top of them would have nothing to attribute. Re-mining the
negatives as \emph{look-alikes} (same apparent gender, $|\Delta\text{age}|\leq5$~years, ranked by
cosine similarity under an independent miner that is not among the evaluated models) restores
difficulty (Table~\ref{tab:lookalike}): IR-101 falls to 0.705 at rank-10 and 0.474 at rank-1. A
second miner from a different loss family reproduces the curve to within 0.03 and preserves the
model ordering at every level, so the difficulty axis is a property of the task, not of the miner.
Our mined pairs continue to help a weak backbone across the whole curve ($+$0.20 to $+$0.26 on the
25+ slice), and the distance to frozen SOTA narrows at the hard end ($-$0.145 at random vs.\
$-$0.070 at rank-1, paired bootstrap over anchors), but it does not close: frozen IR-101 stays
significantly ahead at every difficulty level. The correct reading of Table~\ref{tab:headroom} is
therefore that strong backbones are not improved by \emph{our} supervision---not that they have no
cross-age headroom.

\begin{table}[!t]
\renewcommand{\arraystretch}{1.2}
\caption{Look-alike negatives restore difficulty (25$+$ ROC-AUC; $n_{\text{pos}}=141$,
$n_{\text{neg}}=4762$; rank $=$ impostor rank under an independent miner).}
\label{tab:lookalike}
\centering
\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
Negatives & AdaFace IR-101 & FaceNet frozen & FaceNet $+$pairs \\
\midrule
random (as in Table~\ref{tab:main}) & 0.956 & 0.568 & 0.811 \\
rank-200 & 0.922 & 0.492 & 0.759 \\
rank-50 & 0.835 & 0.381 & 0.657 \\
rank-10 & \textbf{0.705} & 0.305 & 0.562 \\
rank-1 & \textbf{0.474} & 0.203 & 0.404 \\
\bottomrule
\end{tabular}
\end{table}
"""

HARDNEG_RU = r"""

\textbf{Кривая headroom измерена против \emph{случайных} импостеров.} Негативы нашего теста ---
случайные пары разных личностей (плюс небольшое age-controlled подмножество), которые современные
backbone разделяют тривиально: замороженный AdaFace IR-101 даёт \texttt{our.25+} $=$ 0.968 и FG-NET
large-gap 0.957 --- выше всего, чего достигает наш пайплайн. Это заодно подтверждает измерением, а
не декларацией, аргумент атрибуции из разд.~\ref{sec:method}: на этой задаче сильные эмбеддинги
действительно близки к идеалу, поэтому адаптеру поверх них нечего было бы атрибутировать.
Пересборка негативов как \emph{похожих} (тот же кажущийся пол, $|\Delta\text{возраст}|\leq5$ лет,
ранжирование по косинусу независимым майнером, не входящим в число оцениваемых моделей) возвращает
сложность (Таблица~\ref{tab:lookalike}): IR-101 падает до 0.705 на rank-10 и 0.474 на rank-1. Второй
майнер из другого loss-семейства воспроизводит кривую с точностью до 0.03 и сохраняет порядок
моделей на каждом уровне, т.е.\ ось сложности принадлежит задаче, а не майнеру. Наши пары
продолжают помогать слабому backbone на всей кривой ($+$0.20…$+$0.26 на срезе 25+), а расстояние до
замороженной SOTA сужается на трудном конце ($-$0.145 на случайных против $-$0.070 на rank-1,
парный bootstrap по якорям), но не закрывается: замороженный IR-101 значимо впереди на каждом
уровне. Поэтому корректное прочтение Таблицы~\ref{tab:headroom} --- сильные backbone не улучшаются
\emph{нашей} супервизией, а не «у них нет кросс-возрастного запаса».

\begin{table}[!t]
\renewcommand{\arraystretch}{1.2}
\caption{Похожие негативы возвращают сложность (25$+$ ROC-AUC; $n_{\text{pos}}=141$,
$n_{\text{neg}}=4762$; rank --- ранг импостера по независимому майнеру).}
\label{tab:lookalike}
\centering
\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
Негативы & AdaFace IR-101 & FaceNet frozen & FaceNet $+$pairs \\
\midrule
случайные (как в Таблице~\ref{tab:main}) & 0.956 & 0.568 & 0.811 \\
rank-200 & 0.922 & 0.492 & 0.759 \\
rank-50 & 0.835 & 0.381 & 0.657 \\
rank-10 & \textbf{0.705} & 0.305 & 0.562 \\
rank-1 & \textbf{0.474} & 0.203 & 0.404 \\
\bottomrule
\end{tabular}
\end{table}
"""

# У AAAI жёсткий лимит 7 страниц -> компактная врезка без таблицы.
HARDNEG_AAAI_EN = (
    r" \textbf{Caveat on this curve:} it is measured against \emph{random} impostors, which modern "
    r"backbones separate trivially (frozen IR-101 reaches \texttt{our.25+} $=$ 0.968). With "
    r"look-alike negatives (same apparent gender, $|\Delta\text{age}|\leq5$~years, ranked by an "
    r"independent miner) IR-101 falls to 0.705 at rank-10 and 0.474 at rank-1, while our pairs "
    r"still add $+$0.20 to $+$0.26 to a weak backbone; frozen SOTA nonetheless stays significantly "
    r"ahead at every level. Strong backbones are therefore not improved by \emph{our} supervision, "
    r"rather than lacking cross-age headroom."
)
HARDNEG_AAAI_RU = (
    r" \textbf{Оговорка к этой кривой:} она измерена против \emph{случайных} импостеров, которых "
    r"современные backbone разделяют тривиально (замороженный IR-101 даёт \texttt{our.25+} $=$ "
    r"0.968). С похожими негативами (тот же кажущийся пол, $|\Delta\text{возраст}|\leq5$ лет, "
    r"ранжирование независимым майнером) IR-101 падает до 0.705 на rank-10 и 0.474 на rank-1, тогда "
    r"как наши пары по-прежнему добавляют $+$0.20…$+$0.26 слабому backbone; замороженная SOTA при "
    r"этом значимо впереди на каждом уровне. Значит сильные backbone не улучшаются \emph{нашей} "
    r"супервизией, а не лишены кросс-возрастного запаса."
)
HEADROOM_ANCHORS = {
    "en": [r"The contribution is therefore specific to weak and medium backbones with cross-age "
           r"headroom---a limitation we state explicitly (Sec.~\ref{sec:limitations})."],
    "ru": [r"Вклад специфичен для слабых и средних backbone с кросс-возрастным запасом --- "
           r"ограничение, которое мы указываем явно (разд.~\ref{sec:limitations}).",
           r"Вклад, таким образом, специфичен для слабых и средних backbone с кросс-возрастным "
           r"запасом — это ограничение мы прямо отмечаем (разд.~\ref{sec:limitations})."],
}


SUPP = [
    # журнальные дополнения (EN)
    (r"LFW TAR@FAR$=$0.1\% $0.858\!\to\!0.575$ (Sec.~V-A, Sec.~VI) & Est. \\",
     r"LFW TAR@FAR$=$0.1\% $0.814\!\to\!0.816$ (Sec.~V-A, Sec.~VI) & Est. \\"),
    (r"Drop-in upgrade to a general-purpose verifier & low-FAR easy-benchmark operating points "
     r"degrade & Open (excluded) \\",
     r"Drop-in upgrade to a general-purpose verifier & the gain is concentrated on the large-gap "
     r"subset; easy benchmarks are flat & Open (excluded) \\"),
    (r"(low-FAR easy-benchmark performance degrades).\\",
     r"(the gain is concentrated on the large-gap subset rather than uniform).\\"),
    # конференционное дополнение (EN): строка +pairs считалась на невыровненном LFW;
    # колонка hard-neg пересчитана и подтвердилась (0.925 против 0.9254 +- 0.0091)
    (r"LFW accuracy & 0.969 & 0.950 & 0.925\,$\pm$\,0.005 \\",
     r"LFW accuracy & 0.964 & 0.968 & 0.925\,$\pm$\,0.009 \\"),
    (r"Targeted, not general gain & easy-benchmark low-FAR degrades (LFW TAR@0.1\% "
     r"$0.858\!\rightarrow\!0.575$) & scope to large-gap retrieval \\",
     r"Targeted, not general gain & concentrated on the large-gap subset; easy benchmarks flat "
     r"(LFW TAR@0.1\% $0.814\!\rightarrow\!0.816$) & scope to large-gap retrieval \\"),
    # конференционное дополнение (RU)
    (r"LFW точность & 0.969 & 0.950 & 0.925\,$\pm$\,0.005 \\",
     r"LFW точность & 0.964 & 0.968 & 0.925\,$\pm$\,0.009 \\"),
    (r"Целевой, не универсальный прирост & низкий FAR на лёгких бенчмарках деградирует "
     r"(LFW TAR@0.1\% $0.858\!\rightarrow\!0.575$) & объём — поиск на большом разрыве \\",
     r"Целевой, не универсальный прирост & сконцентрирован на подмножестве большого разрыва; "
     r"лёгкие бенчмарки не меняются (LFW TAR@0.1\% $0.814\!\rightarrow\!0.816$) & объём — "
     r"поиск на большом разрыве \\"),
    (r"Только целевой прирост на большом разрыве (точки низкого FAR деградируют);",
     r"Прирост сконцентрирован на большом разрыве (лёгкие бенчмарки не меняются);"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for rel in PAPERS:
        p = ROOT / rel
        if not p.exists():
            print(f"[нет файла] {rel}")
            continue
        s0 = s = p.read_text(encoding="utf-8")
        lang = "ru" if rel.endswith("_ru.tex") else "en"
        aaai = "aaai" in rel
        hits = 0

        for old, new in TABLES + (RU if lang == "ru" else EN):
            s, ok = sub(s, old, new)
            hits += ok

        method = LFW_METHOD_RU if lang == "ru" else LFW_METHOD_EN
        if re.sub(r"\s+", " ", method) not in re.sub(r"\s+", " ", s):
            for a in METHOD_ANCHORS[lang]:
                s, ok = sub(s, a, a + method)
                if ok:
                    hits += 1
                    break
            else:
                print(f"    ! {rel}: якорь методики LFW не найден")

        add = (HARDNEG_AAAI_RU if lang == "ru" else HARDNEG_AAAI_EN) if aaai else (
            HARDNEG_RU if lang == "ru" else HARDNEG_EN)
        if re.sub(r"\s+", " ", add) not in re.sub(r"\s+", " ", s):
            for a in HEADROOM_ANCHORS[lang]:
                s, ok = sub(s, a, a + add)
                if ok:
                    hits += 1
                    break
            else:
                print(f"    ! {rel}: якорь headroom не найден")

        changed = s != s0
        if changed and not args.dry_run:
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(s)
        print(f"[{'изменён' if changed else 'без изменений'}] {rel}: применено {hits}")

    for rel in SUPPLEMENTS:
        p = ROOT / rel
        if not p.exists():
            print(f"[нет файла] {rel}")
            continue
        s0 = s = p.read_text(encoding="utf-8")
        hits = 0
        for old, new in SUPP:
            s, ok = sub(s, old, new)
            hits += ok
        changed = s != s0
        if changed and not args.dry_run:
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(s)
        print(f"[{'изменён' if changed else 'без изменений'}] {rel}: применено {hits}")

    if args.dry_run:
        print("\n(dry-run: файлы не записаны)")


if __name__ == "__main__":
    main()
