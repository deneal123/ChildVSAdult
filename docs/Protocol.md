# Protocol — журнал работ

Хронология «что и как делали». Числа — в [Results.md](Results.md), находки/баги — в
[Observations.md](Observations.md), задачи — в [TODO.md](TODO.md).

Формат: дата — действие (как / результат / артефакт).

---

## Этап основания (ранее, до 2026-06-17)

Конвейер данных и базовые эксперименты (MVP-1…MVP-4 + честный протокол):

- Скелет репозитория, settings (dynaconf + .env), схемы данных, JSONL-IO, выбор устройства CPU/GPU.
- **Ingestion** 1-й стены VK (-77072632) через `wall.get` (сервисный токен): RawPost + provenance,
  параллельная загрузка фото (16 потоков). Масштабирование 200 → 1000 → 3000 → **8046 постов**.
- **Препроцессинг** InsightFace (RetinaFace + выравнивание ArcFace 112), quality-score, резюмируемо,
  GPU(cuDNN-стабильно)/CPU → **9 750 usable-лиц**.
- **Группы личностей** (1 пост = кандидат) + извлечение возраста: regex (RU) + GigaChat-LLM, привязка
  по позиции фото → **5 504 группы**, 69% позитивов с age_gap.
- **Пары** (позитивы C(n,2), простые/age-controlled/hard негативы, uncertain-фильтр) + **leakage-safe
  сплит** по identity_group_id → **12 330 пар**.
- **Baseline** frozen ArcFace (`w600k_r50`): ROC-AUC/EER/TAR@FAR по age-gap; retrieval (Rank-1 0.70).
- **Adapter** поверх замороженного ArcFace → признан методологически некорректным
  ([Observations](Observations.md)); разворот на честный протокол.
- **Честный протокол** (слабый обучаемый facenet casia-webface, frozen vs дообученный) + **ablation**
  (frozen | +pairs | +pairs+age-reg) на LFW/AgeDB-30/CALFW/FG-NET: реальные пары — почти весь выигрыш
  ([Results E1](Results.md)).
- **Synthetic-ageing baseline (proxy)**: реальные пары ≫ синтетика.

## 2026-06-17 — реальная aging-модель, мультибэкбон, выравнивание

- **Реальная aging-модель FRAN** (Face Re-Aging Network, U-Net; веса `timroelofs123/face_re-aging`,
  MIT) подключена за интерфейсом `AgingTransform` (`models/aging.py`, `make_aging("proxy"|"fran")`).
  Веса грузятся `weights_only=True`. Smoke-тест: target=75 → морщины, target=8 → детское лицо.
  Прогон `synthetic_baseline.py --aging both`: даже настоящая FRAN ≈ frozen на cross-age →
  «реальные пары ≫ синтетика» усилено ([Results E3](Results.md)).
- **Несколько backbone** (`backbones_benchmark.py`): сделан backbone-агностичный eval-стек (каждый
  backbone несёт свой `preprocess` + `trainable_scopes`); добавлены ArcFace iResNet50/CASIA-FaceV5
  (слабый) и iResNet100 (сильный) — `models/iresnet.py`. Выигрыш воспроизводится на всех
  ([Results E4](Results.md)).
- **MTCNN-выравнивание facenet** (`preprocessing/mtcnn_align.py`, `align_mtcnn.py`): 9 348 кропов из
  оригиналов (матчинг по IoU). Родное выравнивание даёт facenet +0.043 ([Results E5](Results.md)).
- **Депт-контроль + loss-diversity:** вендорена IR/IR-SE арх AdaFace (`models/adaface_ir.py`),
  подключены AdaFace IR-50 (WebFace4M) и IR-101 (WebFace12M) из cvlface. Чистый контроль глубины →
  выигрыш ~ запас backbone ([Results E6](Results.md)). AdaFace закрывает интент CurricularFace.

## 2026-06-18 — расширение датасета, LLM-аудит возраста, документация

- **Вторая стена VK** «Запах минувших дней» (-134190297). Найден и исправлен **баг пагинации**
  `get_wall` (брал 399 из 36 563 — прерывался на короткой странице). Добавлен `ingest --append`
  (дедуп по post_id, инкрементальная дозапись чанками). Итог: **44 609 постов** (8046 + 36 563),
  44 204 с фото.
- **Слэш/дефис-формат возраста** («6/18/21», «5-26») добавлен в regex-экстрактор (позиционная
  привязка, срабатывает при совпадении числа возрастов и фото). На новой стене: 92% мультифото,
  51% с чистой по-фотографной привязкой.
- **LLM-аудит возраста** (`audit_ages.py`): конкурентный (10 потоков GigaChat) проход по всем
  подписям, кэш (резюмируемо), отчёт regex-vs-LLM, `--apply` (пересборка групп через
  `CachedAgeExtractor`). Улучшен системный промпт LLM (слэш ≠ дата). Полный аудит 33 697 подписей:
  LLM расходится с regex на **27.2%** — в основном правки в пользу LLM ([Results D2](Results.md),
  [Observations](Observations.md)).
- **Препроцессинг расширенного датасета** запущен (на снапшоте, чтобы перекрыть детекцию с
  загрузкой) → к этому моменту 16 827+ usable-кропов; догон на полном `posts.jsonl` — в работе.
- **Расширенный датасет пересобран на LLM-возрастах** (`audit_ages.py --apply` → `build_pairs` →
  `split`): 44 609 постов → 84 147 фото → **49 604 usable-лица → 27 488 групп → 64 966 пар**
  (5×); сплит train/val/test 38 952/5 524/5 446; покрытие возраста 76% (97% привязок из LLM).
  Перезапуск экспериментов на расширенном датасете ([Results D1](Results.md)).
- **Фикс баланса негативов в сплите** (`splits.run` генерирует негативы ПОсплитово,
  балансируя к позитивам; `pair_builder.build_negative_pairs(split=...)`): test стал 4753/4753
  (было 4753/693) — `our.*` снова честные. Корень — глобальные негативы выкидывались как
  кросс-сплитовые при 27k групп ([Observations](Observations.md)).
- **Мультибэкбон на расширенном + сбалансированном датасете** ([Results E9](Results.md)): слабые
  backbone уверенно растут (FG-NET large-gap +0.12/+0.04), сильные — слегка ПРОСЕДАЮТ на внешнем
  FG-NET (−0.03…−0.055) даже после корректного отбора → реальное мягкое забывание; кривая headroom
  заострена. Качество: ruff/mypy ✓, pytest ✓ (92).
- **Документация реорганизована:** TODO.md сжат (867→298 строк, дизайн-спека убрана — она в коде),
  затем разнесён на README (общее + установка/запуск), Protocol.md (этот журнал), Results.md
  (таблицы), Observations.md (находки), TODO.md (только список задач). Добавлен научный бэклог
  гипотез ([TODO §5](TODO.md)).

## 2026-06-19 — TODO по порядку

- **Item 1 (забывание сильных backbone):** дообучение сильных backbone с мягким lr 1e-5 (вместо
  3e-5). Прогон прерывался (ноут уснул, exit 4 на eval), но чекпойнты сохранились → возобновлено
  через новый флаг `backbones_benchmark.py --eval-only` (оценка готовых чекпойнтов без
  переобучения). Итог: мягкий lr ~вдвое-вчетверо уменьшает просадку на FG-NET и улучшает внутренний
  выигрыш ([Results E10](Results.md)) — забывание lr-driven, митигируется.
- **Item 2 (hard-негативы, переориентировано на backbone):** `mine()` переписан на память-safe
  chunked top-k (полная 49k² матрица не влезает); re-embed 49k ArcFace → майнинг (7 410 hard в
  train) → дообучение. Помогают facenet, переусердствуют на пластичном arcface_r50_casia
  ([Results E11](Results.md)).
- **Ручная проверка high-cosine пар (по подсказке пользователя):** при cos 0.55–0.8 это СМЕСЬ
  look-alike и same-person; надёжно «тот же» только ≥0.85; ~1.0 — дубликаты. Вскрыло **утечку
  личности** (сплит по посту, а не по человеку) ([Observations](Observations.md)).
- **Leakage-safe сплит ПО ЧЕЛОВЕКУ:** `person_clusters.py` (union-find по cos≥0.85) слил 27 487
  групп → 21 847 личностей (~20% повторов); `--apply` → person-level группы → пары 131 794
  (+cross-post позитивы) → сплит по человеку. Перезапуск на честном сплите: **выигрыш от пар
  устойчив** (facenet our.25+ +0.169, FG-NET large-gap +0.119); `our.*` даже чуть выше —
  per-post сплит занижался ложными негативами ([Results E12](Results.md)).
- **§5.1 age-matched negatives** (`build_negative_pairs(age_matched=True)` + `split --age-matched-neg`):
  негативы из одного возрастного бакета → identity-only. Доказали возрастной шорткат: frozen
  our.25+ 0.705→0.592 под age-control, +pairs → 0.865 ([Results E13](Results.md)). Ценность —
  честный протокол оценки; тренировочный эффект маргинален.
- **§5.3 age-gap калибровка** (`calibration.py`, `calibrate.py`): P(same | cos, age_gap) vs
  Platt(cos). Сырой косинус плохо калиброван на gap 15–25 (ECE 0.088), age-gap-aware чинит (0.054),
  Brier 0.041→0.034 ([Results E14](Results.md)). Отвечает на вопрос «насколько вероятно при большом разрыве».
- **Фаза 9 / §5.2 disentanglement** (`disentangle.py`: GRL + age-голова): малый устойчивый прирост
  cross-age (+0.008 FG-NET large-gap, +0.014 our.25+) без вреда easy-бенчмаркам ([Results E15](Results.md)).
  Согласуется с E1 — драйвер данные, не явная age-супервизия. (Баг val-кортежа поправлен на лету.)
- **§5.5 cross-wall генерализация** (`split.py --by-wall B`): train на стене B → test на held-out
  стене A. Прирост переносится: внешний FG-NET large-gap +0.123 (≈within-source), стена A our.25+
  +0.102 ([Results E16](Results.md)) — source-agnostic возрастная инвариантность.

## 2026-06-20 — Q1-подготовка

> Цель: довести доказательную базу до уровня Q1-публикации. Порядок:
> multi-seed → scaling-law → демографический аудит → консолидация Report → финализация.

- **Статистическая значимость (multi-seed, E17):** `multiseed.py` — 3 сида (42/1/2) дообучения
  facenet, mean±std ключевых метрик (frozen детерминирован). Прирост значим: FG-NET large-gap
  +0.122±0.003, our.25+ +0.168±0.001 (std≤0.003) ([Results E17](Results.md)). Прогон пережил краш
  на печати `Δ` (cp1251) — восстановлен флагом `--reuse` (ре-оценка готовых чекпойнтов) + ASCII-вывод.
- **Data scaling-law (`scaling_law.py`):** прирост vs доля train-личностей (0.1/0.25/0.5/1.0; val/test
  фиксированы). Мутирует `pairs.jsonl` → идёт ОДИН, в конце восстанавливает полный сплит. Первый
  прогон пользователь остановил на frac 0.5; `pairs.jsonl` вернул полным пересплитом вручную.
  Скрипт улучшен: **печатает результаты каждой доли сразу** (инкрементально), чтобы обрыв не обнулял
  посчитанное. **Итог:** кривая насыщается рано — ~87% прироста уже на 10% личностей (~1.5k),
  от 25% плато ([Results E18](Results.md)). Данные-эффективно; нужен не объём, а трудность данных.
- **Демографический аудит (`evaluation/fairness.py`, `scripts/fairness_audit.py`):** insightface
  `genderage` (CPU — обходит cuDNN-конфликт с torch-CUDA) даёт apparent пол+возраст на каждое лицо;
  стратификация по атрибутам якоря `face_a` (есть у поз. и нег. → AUC определён в каждой страте);
  frozen vs +pairs по стратам пол/возраст + прирост. Резюмируемый кэш `face_genderage.jsonl`
  (7521 лицо за ~17 с, CPU). **Итог:** улучшаются ВСЕ страты, сильнее всего младшие 0–17
  (frozen 0.763 → 0.908, +0.145 — там frozen слабее всего) → метод сужает разрыв; 45+ разрежена,
  источник перекошен в F (76%) ([Results E19](Results.md)). overall совпал с E17/E18 — скоринг верен.
- **Консолидация Report:** §1.1 «Связанные работы и позиционирование» (3 семейства cross-age,
  новизна = label-free источник, бенчмарки общие — #2); §3.1 заострена self-supervised ablation
  (same-post без меток даёт почти весь прирост, age-anchor +0.009 сверху — #4); Ограничения
  уточнены (SOTA позиционирован, открыт только прямой train внешнего метода на наших парах).
- **Фигуры (`make_figures.py`, оформление):** 5 публикационных фигур (PNG 300dpi + векторный PDF в
  `docs/figures/`) из готовых E-таблиц, воспроизводимы без перезапуска: headroom (E9), scaling (E18),
  age-shortcut (E13), демо-страты (E19), сложность-vs-разрыв (E7). Встроены в Report §3.10 с
  подписями. Добавлена зависимость `matplotlib` в `ml`-extra (надёжнее plotly+kaleido на Windows).
- **A — age-leakage probe (E20, `age_leakage.py`):** linear-probe декодируемости apparent-возраста
  из эмбеддинга. Возраст остаётся декодируемым у всех моделей (frozen 0.681 → +pairs 0.679 →
  disentangle 0.674 bal-acc, chance 0.25), хотя identity-AUC растёт 0.856→0.932. ⇒ +pairs учит
  инвариантную МЕТРИКУ, не стирает возраст из представления (representational-нюанс к E13).
- **C — bootstrap-CI прироста (`metrics.bootstrap_auc_ci`, парный в fairness):** E19 дополнен 95%-CI.
  Прирост значим для всех страт, КРОМЕ 45+ (CI [−0.007,+0.049] включает 0 — мало данных); 0–17
  значим и не пересекается со взрослыми; F значимо больше M.
- **B — SOTA-objective (E21, `arcface_train.py`):** обучили ArcFace-margin классификацию по личности
  (10 043 класса, 29 324 лица из identity_groups+group_splits) на том же facenet/scope/протоколе, что
  контрастив. Ранний баг (BatchNorm на батче=1) → фикс: eval-проб emb-dim + `drop_last=True`. Итог:
  на FG-NET large-gap ArcFace ≈ контрастив (0.8548 vs 0.8546) — **драйвер данные, не метод**; ArcFace
  забывает меньше на easy (LFW 0.953 vs 0.942). Закрыт пробел «нет SOTA на общем протоколе»; Report
  §3.10 + §1.1/Ограничения/Заключение/аннотация обновлены.

## 2026-06-21 — Чистка и качество датасета

> Вектор: обогащение метаданных (пол) + LLM-валидация возраста/групп + дедуп. План: Part 1 пол →
> Part 2 LLM-валидация (возраст готов, группы) → Part 3 дедуп/слияния.

- **Part 1 — gender в метаданные (`gender_meta.py`, `enrich_gender.py`):** genderage на всех 49 606
  usable-лицах (CPU, резюмируемо) → агрегация per-group (majority-пол + consistency + медианный
  apparent-возраст) в `IdentityGroup.apparent_gender/age/consistency` (+ post-level сводка
  `post_gender.jsonl`). Покрытие 100% (21 848/21 848 групп). Датасет: **76.7% F / 23.3% M**
  (подтверждает перекос с теста на всём датасете), apparent-age медиана 23. Находка:
  **gender-consistency — бесплатный детектор шума** (2919 мультифото-групп с согласием <0.6 =
  вероятный мульти-человек).
- **Part 2b — LLM-валидация целостности групп (`group_validator.py`, `validate_groups.py`):**
  GigaChat классифицирует каждый мультифото-пост: single | multi_person | collage | meme | unknown
  (async concurrency 10, резюмируемый кэш, как audit_ages). Баг парсинга (хвостовая «}» в ответе
  GigaChat) → фикс балансировщиком скобок. Полный проход (33 697 постов): **92.8% single, 6.0%
  multi_person, 0.2% meme/collage**; `--apply` проставил `identity_review` (matched 21 373, noisy 421).
- **Part 2a — возраст:** LLM-аудит уже полон (33 697/33 697 в кэше); группы на LLM-возрастах.
- **Финал — prune + пересборка (`clean.py`):** выброшены 421 шумная группа + 8190 near-dup лиц →
  группы 21 848→21 427, лица 49 606→40 586, **позитивы 65 897→31 865 (−52%!)** ([Results D3](Results.md)).
  Крупнейший эффект — near-dup (квадратичный C(n,2)): сырой счёт пар был раздут 2× дубликатами. Честная
  де-инфляция: frozen our.overall 0.856→0.836, our.25+ 0.705→0.640. Шумных групп мало (421) — person-
  кластеризация уже развела мульти-человек. ⚠️ внешние бенчмарки не затронуты; ключевые экспы на
  чистых данных — follow-up. Бэкапы: `.pre_gender_bak`/`.pre_validate_bak`/`.pre_prune_bak`.

## 2026-06-22 — Перепрогоны ключевых экспериментов на ЧИСТЫХ данных

> После чистки (D3) `our.*` и обучающий набор изменились. Перепрогон подтверждает, что центральные
> тезисы держатся на чистом, более честном бенчмарке. Старые чекпойнты (на грязных данных) перезаписаны.

- **E17 (multi-seed, headline):** facenet 3 сида на чистых парах. Прирост держится и значим
  (std≤0.003): FG-NET large-gap **+0.112** (был +0.122), our.25+ **+0.195** (был +0.168 — ВЫРОС, т.к.
  frozen упал до 0.640 на честном тесте, +pairs → 0.835). Таблица обновлена ([Results E17](Results.md)).
- **E21 (SOTA ArcFace):** переобучен на чистых личностях (9 204 класса). Objective-invariance держится:
  FG-NET large-gap контрастив 0.8476 ≈ ArcFace 0.8506 (Δ0.003); ArcFace забывает меньше на easy
  (LFW 0.952 vs 0.946). Выводы идентичны грязным ([Results E21](Results.md)).
- **Вывод:** оба headline-тезиса (прирост от данных значим; данные > метод) **робастны к чистке** —
  это усиливает статью (честный труднее бенчмарк, тот же вывод). Внешние FG-NET-числа почти не
  сдвинулись; внутренние `our.*` честно ниже.

## 2026-06-22 — Data governance / обезличивание (по итогам ресёрча правовых рисков)

> Ресёрч (152-ФЗ/519-ФЗ/ст.11 биометрия; GDPR ст.9/89; прецеденты MS-Celeb-1M, Clearview AI):
> «официальный API + публичный пост» НЕ даёт права на биометрическую обработку; явного письменного
> согласия нет → нужен пакет мер, иначе desk-reject.

- **Создан `docs/DATA_GOVERNANCE.md`:** datasheet (Gebru), правовой gap-анализ (152-ФЗ+GDPR), политика
  обезличивания (НЕ релизить сырые лица; хеш-id; страйп PII; controlled access/DUA; только производные
  эмбеддинги/агрегаты), DPIA (таблица риск→мера), раздел про несовершеннолетних (изъятие страты 0–17 из
  релиза), takedown/права субъектов, чек-лист перед сабмитом.
- **§8 обоих черновиков переписан** (`PAPER_DRAFT_EN/RU`): 8.1 legal basis + честный пробел, 8.2
  de-identification/release, 8.3 DPIA/минимизация, 8.4 minors, 8.5 subject rights.
- **Реализован `anonymize_release.py`** (+ `datasets/anonymize.py`, тесты): HMAC-хеш id (соль
  `data/.anon_salt`, gitignored, не публикуется), страйп PII, исключение несовершеннолетних (apparent<18),
  экспорт только производного в `release/`, safety-scan (regex `-\d{5,}`). Прогон: групп 20 405 / пар
  45 302 / minor исключено 8 201; 0 сырых VK-id в выводе.
- **TODO (вне кода, на вас):** этическая экспертиза/IRB организации + юрист по 152-ФЗ (биометрия/дети).

## 2026-06-23 — Ответ рецензенту: правки статьи + 4 future-work эксперимента

> Внешняя рецензия («publishable after major revision»). Пользователь выбрал 4 пакета правок и
> попросил реально выполнить пункты future-work. Лимит страниц снят (приоритет — полнота).

**Пакеты правок (в LaTeX EN+RU, оба компилируются чисто):**

- П1 — формулировки/язык: «данные > метод» → within-protocol large-gap; headline = targeted
  large-gap robustness; добавлен primary endpoint + H0 (FG-NET ≥25 лет) в §4. Коммит 0053d83.
- П2 — статистика в тексте: bootstrap-CI заголовочных AUC, EER, TAR@FAR, identity-level bootstrap
  (Таблица операционных точек). FG-NET large-gap CI не пересекаются (frozen [0.70,0.78] vs
  +pairs [0.82,0.88]). Коммит c3f6788.
- П3 — sensitivity дедупа (см. [Results D3](Results.md)). Коммит 1b57542.
- П4 — CI на фигуре fairness (gain-CI усы + n по стратам). Коммит 1924d8c.

**Future-work эксперименты:**

- #1 child↔adult / независимый не-VK — **ГОТОВО** (`eval_child.py` → metrics/child_eval.json,
  [Results E22](Results.md)). FG-NET child↔adult (<13 ↔ >25; 195 поз): frozen 0.684 → +pairs 0.813
  (+0.129, CI не пересекаются) ≈ внутренний 0–17 (+0.130). В §5.7. Коммит 472011d.
- #3 dedup-threshold sensitivity — **ГОТОВО** (`sensitivity_dedup.py`). Первый перепрогон был
  ВЫРОЖДЕН (re-dedup уже-курированных групп → 0 near-dup; все пороги бит-в-бит совпали). Валидно —
  только на PRE-prune группах (`identity_groups.jsonl.pre_prune_bak`): −52% устойчиво на 0.93–0.97,
  плавно к 0.99 (−41%). В §3.4. Коммит 1b57542.
- #2 SOTA-objectives (CosFace/SphereFace) — **В РАБОТЕ** (`sota_arcface.py`; `MarginHead` обобщён на
  arcface/cosface/sphereface). Полные MTLFace/OE-CNN конвейеры вне scope (портирование 2 чужих статей).
- #4 aging на нативном res — **В РАБОТЕ** (`aging_hires_subset.py`). FRAN внутри апскейлит 112→512;
  проверяем на 512-перекропах из raw, не артефакт ли разрешения находка §5.2.
- #3-CI scaling per-point multi-seed — **В ОЧЕРЕДИ** (`scaling_multiseed.py`, 12 facenet, ~10ч) →
  CI-полосы на fig_scaling.

> GPU сериализован (один тренинг за раз). Чистый сплит проверяется/восстанавливается между прогонами
> (train 22179/22179, val 4579, test 5107). Инфра #2/#4 — коммит ed34113.

## 2026-06-23 — Второй источник супервизии: Reddit r/PastAndPresentPics (не-VK)

> По рецензии главный Q1-апгрейд (после этики) — независимая не-VK внешняя валидация. Добавлен
> Reddit-ингест мультифото «then/now»-постов, зеркальный VK: тот же RawPost/Photo формат →
> downstream (preprocess → LLM-возраст → группы → пары → сплит) работает без изменений.

- **`parsing/reddit_client.py`** — OAuth2 (`oauth.reddit.com`) с client_id/secret из `.env`;
  публичный `.json` как fallback. Reddit с 2023 отдаёт **403 Blocked** на неаутентифицированный
  скрейпинг (проверено: 403 даже с браузерным UA и на old.reddit.com → блок по IP/анти-бот в
  dev-среде). **OAuth НЕОБЯЗАТЕЛЕН:** публичный режим усилен (браузерные заголовки + фоллбэк
  хостов www→old + бэкофф) и обычно работает с резидентного IP; OAuth — фоллбэк для
  заблокированных дата-центровых IP. Листинг (top/new/hot, дедуп по id, ~1000/sort), извлечение фото
  из gallery (`media_metadata`/`gallery_data`) и одиночных image-постов.
- **`parsing/reddit_ingest.py`** + **`scripts/ingest_reddit.py`** — нормализация в RawPost
  (post_id `reddit_<id>`, source `reddit_public`, caption=заголовок → возраст той же LLM),
  параллельная загрузка фото → стандартный `posts.jsonl` + `images/reddit_<id>/` под активным data_dir.
- Парсинг провалидирован офлайн (gallery→N фото по порядку, single→1, permalink-чистка, to_dict).
  Живой fetch из dev-среды блокируется по IP → запускать на резидентном IP с OAuth-креденшелами.
- **Отдельный data-root через dynaconf-окружение `[reddit]`** (settings.toml): `ENV_FOR_DYNACONF=reddit`
  перенаправляет `data_dir → data_reddit/`, `splits_dir → data_reddit/splits` (merge), а `models_dir`/
  `metrics_dir` наследуются (общий VK-чекпойнт). Так весь штатный пайплайн гоняется на Reddit, не трогая VK.
- **`scripts/eval_cross_platform.py`** — frozen + VK-чекпойнт на произвольном held-out pairs-наборе:
  overall + large-gap (≥25) ROC-AUC c 95% bootstrap-CI, EER, TAR@FAR → `metrics/cross_platform_<tag>.json`.
- **Рантбук (на машине с доступом, ВСЁ под `ENV_FOR_DYNACONF=reddit`):**
  1. script-app на reddit.com/prefs/apps → `.env` (REDDIT_CLIENT_ID/SECRET[/USERNAME/PASSWORD/USER_AGENT]);
  2. `ingest_reddit.py` → `data_reddit/raw/posts.jsonl`; `preprocess.py` → faces; `build_groups.py`;
     (опц.) `audit_ages.py --apply` (возрасты нужны для 25+ среза); `cluster_persons.py --apply`; dedup;
     `build_pairs.py`; `split.py`;
  3. `eval_cross_platform.py --ckpts models/bb_facenet_seed42.pt --tag vk2reddit`;
  4. вписать как независимую не-VK валидацию в §5 (снимает ключевую научную претензию рецензента).

---

> Качество кодовой базы поддерживается зелёным на каждом шаге: ruff ✓, mypy ✓, pytest ✓
> (на 2026-06-22 — 146 тестов).
