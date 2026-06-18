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

---

> Качество кодовой базы поддерживается зелёным на каждом шаге: ruff ✓, mypy ✓, pytest ✓
> (на 2026-06-18 — 92 теста).
