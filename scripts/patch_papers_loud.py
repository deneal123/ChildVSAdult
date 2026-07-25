"""Вторая правка статей: полный набор 25+ (n=1208) и инверсия сходства как главный тезис.

    uv run python scripts/patch_papers_loud.py [--dry-run]

ЗАЧЕМ. Первый патч (scripts/patch_papers_lfw.py) внёс кривую трудных негативов на ТЕСТОВОМ срезе
(n_pos=141) — интервалы там широкие, и «ниже случайного» держалось на грани. Замороженные модели
не видели НИ ОДНОГО нашего примера, поэтому для них бенчмарк корректно считать по всем сплитам:
n_pos 141 -> 1208, n_neg 4762 -> 31590, интервалы втрое уже, и результат становится решающим
(IR-101 rank-1: 0.376 [0.356, 0.394], верхняя граница далеко ниже 0.5).

ЧТО ДОБАВЛЯЕМ СОДЕРЖАТЕЛЬНО. Механизм провала измерен: медианный ArcFace-косинус позитивов 25+
равен 0.232, а самого похожего ровесника-незнакомца — 0.289. То есть при разрыве в 25 лет человек
похож на себя МЕНЬШЕ, чем на самого похожего чужого. Одна эта инверсия объясняет весь обвал и
переводит работу из «мы улучшили слабый backbone» в «задача не решена».

ЧЕГО НЕ ДЕЛАЕМ. Не заявляем, что наш метод обходит SOTA: замороженный IR-101 значимо впереди нас
на каждом уровне сложности (это было проверено парным bootstrap и отозвано в d514d5f).

ВАЖНО ПРО ТАБЛИЦУ. Числа на 1208 и на 141 паре НЕСРАВНИМЫ между собой: сложность rank-k растёт с
размером пула импостеров (при 6.6x большем пуле десятый по похожести заметно похожее). Поэтому
таблица разбита на два явно подписанных блока с указанием n, а не слита в один.

Пишет правки в 6 .tex (EN+RU x T-BIOM/TNNLS/AAAI). Идемпотентно.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAPERS = [
    "latex/papers/journal-1-tbiom/en/main.tex",
    "latex/papers/journal-1-tbiom/ru/main_ru.tex",
    "latex/papers/journal-1-tnnls/en/main.tex",
    "latex/papers/journal-1-tnnls/ru/main_ru.tex",
    "latex/conference/aaai-27/en/main.tex",
    "latex/conference/aaai-27/ru/main_ru.tex",
]


def sub(text: str, old: str, new: str) -> tuple[str, bool]:
    if new.strip() and re.sub(r"\s+", " ", new) in re.sub(r"\s+", " ", text):
        return text, True
    pat = r"\s+".join(re.escape(t) for t in old.split())
    out, n = re.subn(pat, lambda _m: new, text, count=1)
    return out, n > 0


# ------------------------------------------------------------------ журнальный абзац + таблица
OLD_PARA_EN = (
    r"\textbf{The headroom curve is measured against \emph{random} impostors.} Our test negatives "
    r"are random cross-identity pairs (plus a small age-controlled subset), which modern backbones "
    r"separate trivially: frozen AdaFace IR-101 reaches \texttt{our.25+} $=$ 0.968 and FG-NET "
    r"large-gap 0.957, above anything our pipeline attains. This also validates by measurement, "
    r"rather than merely asserts, the attribution argument of Sec.~\ref{sec:method}: on this task "
    r"strong embeddings really are near-perfect, so an adapter on top of them would have nothing "
    r"to attribute. Re-mining the negatives as \emph{look-alikes} (same apparent gender, "
    r"$|\Delta\text{age}|\leq5$~years, ranked by cosine similarity under an independent miner that "
    r"is not among the evaluated models) restores difficulty (Table~\ref{tab:lookalike}): IR-101 "
    r"falls to 0.705 at rank-10 and 0.474 at rank-1. A second miner from a different loss family "
    r"reproduces the curve to within 0.03 and preserves the model ordering at every level, so the "
    r"difficulty axis is a property of the task, not of the miner. Our mined pairs continue to help "
    r"a weak backbone across the whole curve ($+$0.20 to $+$0.26 on the 25+ slice), and the "
    r"distance to frozen SOTA narrows at the hard end ($-$0.145 at random vs.\ $-$0.070 at rank-1, "
    r"paired bootstrap over anchors), but it does not close: frozen IR-101 stays significantly "
    r"ahead at every difficulty level. The correct reading of Table~\ref{tab:headroom} is therefore "
    r"that strong backbones are not improved by \emph{our} supervision---not that they have no "
    r"cross-age headroom."
)

NEW_PARA_EN = (
    r"\textbf{The random-impostor protocol understates the task.} Our test negatives are random "
    r"cross-identity pairs, which modern backbones separate trivially: frozen AdaFace IR-101 "
    r"reaches \texttt{our.25+} $=$ 0.968. That validates by measurement, rather than merely "
    r"asserts, the attribution argument of Sec.~\ref{sec:method}: under this protocol strong "
    r"embeddings really are near-perfect, so an adapter on top of them would have nothing to "
    r"attribute. The reason is visible in the raw similarities. Median ArcFace cosine is 0.456 for "
    r"positives below 10 years, 0.307 at 10--25 years and 0.232 at 25$+$, against 0.012 for random "
    r"negatives---but \textbf{0.289} for the most similar same-gender, same-apparent-age impostor. "
    r"\emph{At a 25-year gap a person is less similar to themselves than to the most similar "
    r"same-age stranger.} Re-mining the negatives accordingly (same apparent gender, "
    r"$|\Delta\text{age}|\leq5$~years, ranked by an independent miner outside the evaluated set) "
    r"collapses every modern recognizer we test (Table~\ref{tab:lookalike}): frozen IR-101 falls "
    r"from 0.961 to \textbf{0.376}~[0.356,\,0.394] at rank-1, \emph{significantly below chance}. "
    r"This is not a labeling artifact: the mined impostors are 0.00\% same-person under the "
    r"identity clustering, and the benchmark carries zero identity leakage across splits, zero "
    r"same-person negatives (0/31\,865), 99.1\% LLM-verified single-person groups among "
    r"those surviving into positives (92.8\% before curation removes multi-person groups) "
    r"and 0.00\% "
    r"near-duplicates (Sec.~\ref{sec:data-curation}). A second miner from a different loss family "
    r"reproduces the curve to within 0.03 and preserves the model ordering at every level, so the "
    r"difficulty axis belongs to the task, not to the miner. Our mined pairs still help a weak "
    r"backbone across the whole curve ($+$0.20 to $+$0.28 on the 25$+$ slice) and the distance to "
    r"frozen SOTA narrows at the hard end ($-$0.145 at random vs.\ $-$0.070 at rank-1, paired "
    r"bootstrap), but it does not close. The correct reading of Table~\ref{tab:headroom} is "
    r"therefore that strong backbones are not improved by \emph{our} supervision---not that they "
    r"lack cross-age headroom, and not that the task is solved."
)

OLD_TABLE_EN = r"""\caption{Look-alike negatives restore difficulty (25$+$ ROC-AUC; $n_{\text{pos}}=141$,
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
\end{tabular}"""

NEW_TABLE_EN = r"""\caption{Look-alike impostors break modern face recognition (25$+$ ROC-AUC; rank $=$ impostor
rank under an independent miner). \emph{Top:} frozen recognizers on the \emph{full} 25$+$ set
($n_{\text{pos}}\!=\!1208$, $n_{\text{neg}}\!=\!31\,590$), valid because no frozen model has seen
any of our data. \emph{Bottom:} the effect of our supervision, necessarily test-only
($n_{\text{pos}}\!=\!141$, $n_{\text{neg}}\!=\!4762$). The blocks use different impostor pools and
are \emph{not} comparable across blocks: rank-$k$ difficulty grows with pool size.}
\label{tab:lookalike}
\centering
\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
\multicolumn{4}{@{}l}{\emph{Frozen recognizers, full 25$+$ set}} \\
Negatives & AdaFace IR-101 & ArcFace r100 & AdaFace IR-50 \\
\midrule
random & 0.961 & 0.941 & 0.908 \\
rank-50 & 0.727 & 0.677 & 0.560 \\
rank-10 & 0.603 & 0.565 & 0.437 \\
rank-1 & \textbf{0.376} & \textbf{0.362} & \textbf{0.258} \\
\midrule
\multicolumn{4}{@{}l}{\emph{Our supervision, test split only}} \\
Negatives & FaceNet frozen & FaceNet $+$pairs & $\Delta$ \\
\midrule
random & 0.568 & 0.811 & $+$0.243 \\
rank-50 & 0.381 & 0.657 & $+$0.276 \\
rank-10 & 0.305 & 0.562 & $+$0.257 \\
rank-1 & 0.203 & 0.404 & $+$0.201 \\
\bottomrule
\end{tabular}"""

OLD_PARA_RU = (
    r"\textbf{Кривая headroom измерена против \emph{случайных} импостеров.} Негативы нашего теста "
    r"--- случайные пары разных личностей (плюс небольшое age-controlled подмножество), которые "
    r"современные backbone разделяют тривиально: замороженный AdaFace IR-101 даёт "
    r"\texttt{our.25+} $=$ 0.968 и FG-NET large-gap 0.957 --- выше всего, чего достигает наш "
    r"пайплайн. Это заодно подтверждает измерением, а не декларацией, аргумент атрибуции из "
    r"разд.~\ref{sec:method}: на этой задаче сильные эмбеддинги действительно близки к идеалу, "
    r"поэтому адаптеру поверх них нечего было бы атрибутировать. Пересборка негативов как "
    r"\emph{похожих} (тот же кажущийся пол, $|\Delta\text{возраст}|\leq5$ лет, ранжирование по "
    r"косинусу независимым майнером, не входящим в число оцениваемых моделей) возвращает сложность "
    r"(Таблица~\ref{tab:lookalike}): IR-101 падает до 0.705 на rank-10 и 0.474 на rank-1. Второй "
    r"майнер из другого loss-семейства воспроизводит кривую с точностью до 0.03 и сохраняет "
    r"порядок моделей на каждом уровне, т.е.\ ось сложности принадлежит задаче, а не майнеру. Наши "
    r"пары продолжают помогать слабому backbone на всей кривой ($+$0.20…$+$0.26 на срезе 25+), а "
    r"расстояние до замороженной SOTA сужается на трудном конце ($-$0.145 на случайных против "
    r"$-$0.070 на rank-1, парный bootstrap по якорям), но не закрывается: замороженный IR-101 "
    r"значимо впереди на каждом уровне. Поэтому корректное прочтение Таблицы~\ref{tab:headroom} "
    r"--- сильные backbone не улучшаются \emph{нашей} супервизией, а не «у них нет "
    r"кросс-возрастного запаса»."
)

NEW_PARA_RU = (
    r"\textbf{Протокол со случайными импостерами занижает сложность задачи.} Негативы нашего теста "
    r"--- случайные пары разных личностей, которые современные backbone разделяют тривиально: "
    r"замороженный AdaFace IR-101 даёт \texttt{our.25+} $=$ 0.968. Это подтверждает измерением, а "
    r"не декларацией, аргумент атрибуции из разд.~\ref{sec:method}: при таком протоколе сильные "
    r"эмбеддинги действительно близки к идеалу, поэтому адаптеру поверх них нечего было бы "
    r"атрибутировать. Причина видна в сырых сходствах. Медианный ArcFace-косинус позитивов равен "
    r"0.456 при разрыве до 10 лет, 0.307 при 10--25 и 0.232 при 25$+$ против 0.012 у случайных "
    r"негативов --- но \textbf{0.289} у самого похожего импостера того же пола и кажущегося "
    r"возраста. \emph{При разрыве в 25 лет человек похож на себя меньше, чем на самого похожего "
    r"ровесника-незнакомца.} Пересборка негативов соответствующим образом (тот же кажущийся пол, "
    r"$|\Delta\text{возраст}|\leq5$ лет, ранжирование независимым майнером вне числа оцениваемых "
    r"моделей) обрушивает каждый проверенный современный распознаватель "
    r"(Таблица~\ref{tab:lookalike}): замороженный IR-101 падает с 0.961 до "
    r"\textbf{0.376}~[0.356,\,0.394] на rank-1 --- \emph{значимо ниже случайного угадывания}. Это "
    r"не артефакт разметки: намайненные импостеры на 0.00\% совпадают по личности с якорем, а сам "
    r"бенчмарк не имеет утечки личностей между сплитами, не содержит негативов той же личности "
    r"(0/31\,865), даёт 99.1\% позитивов из групп, подтверждённых LLM как «один человек» "
    r"(92.8\% до того, как курирование убирает многолюдные группы), и 0.00\% "
    r"почти-дубликатов (разд.~\ref{sec:data-curation}). Второй майнер из другого loss-семейства "
    r"воспроизводит кривую с точностью до 0.03 и сохраняет порядок моделей на каждом уровне, т.е.\ "
    r"ось сложности принадлежит задаче, а не майнеру. Наши пары по-прежнему помогают слабому "
    r"backbone на всей кривой ($+$0.20…$+$0.28 на срезе 25$+$), а расстояние до замороженной SOTA "
    r"сужается на трудном конце ($-$0.145 на случайных против $-$0.070 на rank-1, парный "
    r"bootstrap), но не закрывается. Поэтому корректное прочтение Таблицы~\ref{tab:headroom} --- "
    r"сильные backbone не улучшаются \emph{нашей} супервизией, а не «у них нет кросс-возрастного "
    r"запаса» и не «задача решена»."
)

OLD_TABLE_RU = r"""\caption{Похожие негативы возвращают сложность (25$+$ ROC-AUC; $n_{\text{pos}}=141$,
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
\end{tabular}"""

NEW_TABLE_RU = r"""\caption{Похожие импостеры ломают современное распознавание лиц (25$+$ ROC-AUC; rank --- ранг
импостера по независимому майнеру). \emph{Сверху:} замороженные распознаватели на \emph{полном}
наборе 25$+$ ($n_{\text{pos}}\!=\!1208$, $n_{\text{neg}}\!=\!31\,590$) --- это корректно, т.к.\ ни
одна замороженная модель не видела наших данных. \emph{Снизу:} эффект нашей супервизии, неизбежно
только на тесте ($n_{\text{pos}}\!=\!141$, $n_{\text{neg}}\!=\!4762$). Блоки используют разные пулы
импостеров и \emph{не} сравнимы между собой: сложность rank-$k$ растёт с размером пула.}
\label{tab:lookalike}
\centering
\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
\multicolumn{4}{@{}l}{\emph{Замороженные распознаватели, полный набор 25$+$}} \\
Негативы & AdaFace IR-101 & ArcFace r100 & AdaFace IR-50 \\
\midrule
случайные & 0.961 & 0.941 & 0.908 \\
rank-50 & 0.727 & 0.677 & 0.560 \\
rank-10 & 0.603 & 0.565 & 0.437 \\
rank-1 & \textbf{0.376} & \textbf{0.362} & \textbf{0.258} \\
\midrule
\multicolumn{4}{@{}l}{\emph{Наша супервизия, только тестовый сплит}} \\
Негативы & FaceNet frozen & FaceNet $+$pairs & $\Delta$ \\
\midrule
случайные & 0.568 & 0.811 & $+$0.243 \\
rank-50 & 0.381 & 0.657 & $+$0.276 \\
rank-10 & 0.305 & 0.562 & $+$0.257 \\
rank-1 & 0.203 & 0.404 & $+$0.201 \\
\bottomrule
\end{tabular}"""

# ------------------------------------------------------------------ конференционная врезка
OLD_AAAI_EN = (
    r" \textbf{Caveat on this curve:} it is measured against \emph{random} impostors, which modern "
    r"backbones separate trivially (frozen IR-101 reaches \texttt{our.25+} $=$ 0.968). With "
    r"look-alike negatives (same apparent gender, $|\Delta\text{age}|\leq5$~years, ranked by an "
    r"independent miner) IR-101 falls to 0.705 at rank-10 and 0.474 at rank-1, while our pairs "
    r"still add $+$0.20 to $+$0.26 to a weak backbone; frozen SOTA nonetheless stays significantly "
    r"ahead at every level. Strong backbones are therefore not improved by \emph{our} supervision, "
    r"rather than lacking cross-age headroom."
)
NEW_AAAI_EN = (
    r" \textbf{The random-impostor protocol understates the task.} Median ArcFace cosine at a "
    r"25-year gap is 0.232, but \textbf{0.289} for the most similar same-gender, same-apparent-age "
    r"impostor: a person is less similar to themselves than to the most similar same-age stranger. "
    r"Re-mining negatives as look-alikes (ranked by an independent miner) collapses every modern "
    r"recognizer on the full 25$+$ set ($n_{\text{pos}}\!=\!1208$; frozen models never saw our "
    r"data): AdaFace IR-101 falls from 0.961 to \textbf{0.376}~[0.356,\,0.394], \emph{below "
    r"chance}, ArcFace r100 to 0.362, AdaFace IR-50 to 0.258. It is not a labeling artifact --- the "
    r"impostors are 0.00\% same-person, and the benchmark has zero cross-split identity leakage and "
    r"zero same-person negatives. Our pairs still add $+$0.20 to $+$0.28 to a weak backbone "
    r"(test-only), yet frozen SOTA stays significantly ahead at every level: strong backbones are "
    r"not improved by \emph{our} supervision, rather than lacking cross-age headroom."
)
OLD_AAAI_RU = (
    r" \textbf{Оговорка к этой кривой:} она измерена против \emph{случайных} импостеров, которых "
    r"современные backbone разделяют тривиально (замороженный IR-101 даёт \texttt{our.25+} $=$ "
    r"0.968). С похожими негативами (тот же кажущийся пол, $|\Delta\text{возраст}|\leq5$ лет, "
    r"ранжирование независимым майнером) IR-101 падает до 0.705 на rank-10 и 0.474 на rank-1, тогда "
    r"как наши пары по-прежнему добавляют $+$0.20…$+$0.26 слабому backbone; замороженная SOTA при "
    r"этом значимо впереди на каждом уровне. Значит сильные backbone не улучшаются \emph{нашей} "
    r"супервизией, а не лишены кросс-возрастного запаса."
)
NEW_AAAI_RU = (
    r" \textbf{Протокол со случайными импостерами занижает сложность.} Медианный ArcFace-косинус "
    r"при разрыве 25 лет равен 0.232, а у самого похожего импостера того же пола и кажущегося "
    r"возраста --- \textbf{0.289}: человек похож на себя меньше, чем на самого похожего "
    r"ровесника-незнакомца. Пересборка негативов как похожих (ранжирование независимым майнером) "
    r"обрушивает каждый современный распознаватель на полном наборе 25$+$ "
    r"($n_{\text{pos}}\!=\!1208$; замороженные модели наших данных не видели): AdaFace IR-101"
    r" падает с 0.961 до \textbf{0.376}~[0.356,\,0.394] --- \emph{ниже случайного}, ArcFace r100 "
    r"до 0.362, AdaFace IR-50 до 0.258. Это не артефакт разметки: импостеры на 0.00\% совпадают по "
    r"личности, а бенчмарк не имеет утечки личностей между сплитами и негативов той же личности. "
    r"Наши пары по-прежнему добавляют $+$0.20…$+$0.28 слабому backbone (только на тесте), но "
    r"замороженная SOTA значимо впереди на каждом уровне: сильные backbone не улучшаются "
    r"\emph{нашей} супервизией, а не лишены кросс-возрастного запаса."
)

# ------------------------------------------------------------------ аннотация
ABS_ANCHOR_EN = (r"a gain concentrated on the large-gap regime rather than a trade-off.")
ABS_ADD_EN = (
    r" We further show that the standard \emph{random-impostor} protocol understates this task: at "
    r"a 25-year gap a person is less similar to themselves than to the most similar same-age "
    r"stranger (median ArcFace cosine 0.232 vs.\ 0.289), and a frozen state-of-the-art recognizer "
    r"(AdaFace IR-101) falls from 0.961 to 0.376~[0.356,\,0.394] ROC-AUC---\emph{below chance}---"
    r"once impostors are look-alikes rather than random."
)
ABS_ANCHOR_RU = (r"прирост сконцентрирован на large-gap-режиме, а не разменян на потери.")
ABS_ANCHOR_RU_AAAI = (r"прирост сконцентрирован на режиме большого разрыва, а не разменян на потери.")
ABS_ADD_RU = (
    r" Мы также показываем, что стандартный протокол со \emph{случайными} импостерами занижает эту "
    r"задачу: при разрыве в 25 лет человек похож на себя меньше, чем на самого похожего "
    r"ровесника-незнакомца (медианный ArcFace-косинус 0.232 против 0.289), и замороженный "
    r"современный распознаватель (AdaFace IR-101) падает с 0.961 до 0.376~[0.356,\,0.394] ROC-AUC "
    r"--- \emph{ниже случайного} --- как только импостеры становятся похожими, а не случайными."
)


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
        ru = rel.endswith("_ru.tex")
        aaai = "aaai" in rel
        hits, miss = 0, []

        if aaai:
            pairs_ = [(OLD_AAAI_RU, NEW_AAAI_RU)] if ru else [(OLD_AAAI_EN, NEW_AAAI_EN)]
        elif ru:
            pairs_ = [(OLD_PARA_RU, NEW_PARA_RU), (OLD_TABLE_RU, NEW_TABLE_RU)]
        else:
            pairs_ = [(OLD_PARA_EN, NEW_PARA_EN), (OLD_TABLE_EN, NEW_TABLE_EN)]
        for old, new in pairs_:
            s, ok = sub(s, old, new)
            hits += ok
            if not ok:
                miss.append(old.split(".")[0][:50])

        add = ABS_ADD_RU if ru else ABS_ADD_EN
        if re.sub(r"\s+", " ", add) not in re.sub(r"\s+", " ", s):
            anchors = ([ABS_ANCHOR_RU_AAAI, ABS_ANCHOR_RU] if ru else [ABS_ANCHOR_EN])
            for a in anchors:
                s, ok = sub(s, a, a + add)
                if ok:
                    hits += 1
                    break
            else:
                miss.append("аннотация")

        changed = s != s0
        if changed and not args.dry_run:
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(s)
        print(f"[{'изменён' if changed else 'без изменений'}] {rel}: применено {hits}"
              + (f" | НЕ НАЙДЕНО: {miss}" if miss else ""))

    if args.dry_run:
        print("\n(dry-run: файлы не записаны)")


if __name__ == "__main__":
    main()
