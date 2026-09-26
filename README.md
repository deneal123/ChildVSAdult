# age-gap — Age-Invariant Identity Matching

Исследовательский проект по **кросс-возрастному сопоставлению личности**: оценка того, изображён
ли на двух фотографиях, снятых с разрывом в годы, один и тот же человек.

**Ключевая идея.** Мультифото-посты, где один человек показан в разном возрасте, дают «естественно
заякоренные» позитивные пары личности без ручной разметки. Если подпись содержит возраст — пост
также даёт age-anchor для обучения/оценки age-invariant эмбеддингов.

```text
мультифото-пост с одним человеком в разном возрасте
  → естественные позитивные пары (один человек, разный возраст)
  → возраст из подписи (regex + LLM) → age_gap
  → верификация и метрики в разбивке по возрастному разрыву
```

**Главный результат (кратко).** Дообучение слабого FaceNet на извлечённых VK-парах улучшило
FG-NET large-gap ROC-AUC с 0.736 до 0.848 при старом протоколе случайных импостеров.
После сопоставления возраста *обоих концов* негативной пары результат на FG-NET 25+ составил
0.802 → 0.848 (парный subject-level прирост +0.046; 95% CI [0.006, 0.091]). Внутренний
endpoint-age-matched тест тоже показывает рост ROC-AUC, но TAR при FAR=1% ухудшается; на
AgeDB-30 есть небольшое статистически значимое снижение ROC-AUC. Сравнение с равноценным
контрольным источником и аудит пересечений train–benchmark ещё не завершены, поэтому эти
результаты не доказывают причинное преимущество именно longitudinal-источника. Подробности —
в статье (`latex/papers/journal-1-tbiom/`); внутренний журнал результатов не публикуется.

## Документация

Рабочие журналы (`docs/`) не публикуются. Всё, что нужно для понимания и воспроизведения,
находится в статье и её дополнении: `latex/papers/journal-1-tbiom/` (протокол, статистика,
этика и governance, воспроизводимость) и в машиночитаемых метриках `metrics*/`.

---

## Назначение и безопасность

> **Controlled, consent-based, human-reviewed** кросс-возрастное сопоставление для архивов,
> семейных фото, исследовательских бенчмарков, восстановления доступа к аккаунту или авторизованных
> гуманитарных задач (поиск пропавших с согласия партнёра).

Система возвращает **кандидатов для проверки человеком**, а не автоматическое решение о личности.
**Запрещено:** массовая идентификация, деанонимизация в соцсетях, видеонаблюдение, real-time
биометрия, трекинг между платформами, social scoring, обработка данных несовершеннолетних без
правового основания и мер защиты.

---

## Конвейер

```text
VK wall.get ─► RawPost (+provenance) ─► загрузка фото (параллельно)
   └─► InsightFace: детекция → выбор лица → выравнивание (ArcFace 112) → quality score
         └─► usable-кропы
   └─► группы личностей (1 пост = кандидат-личность)
         └─► возраст из подписи: regex + GigaChat-LLM → привязка к фото по позиции
   └─► пары: позитивы C(n,2) | простые/age-controlled негативы | hard-негативы (mining)
         └─► leakage-safe сплит по identity_group_id (train/val/test)
   └─► эмбеддинги ─► бенчмарк (ROC-AUC/EER/TAR@FAR по age-gap) | retrieval
         └─► честный протокол: слабый backbone frozen vs дообученный на наших парах
```

---

## Установка

Менеджер пакетов — [uv](https://docs.astral.sh/uv/). Python 3.12.

**CPU** (портативный вариант, конвейер данных):

```bash
uv sync --extra cpu
```

**GPU** (NVIDIA) — ускорение детекции/эмбеддингов и обучение:

```bash
uv sync --extra gpu --extra ml
uv pip install --reinstall onnxruntime-gpu==1.26.0  # версия из uv.lock; вытеснить CPU wheel
# после этого: uv run --no-sync ...  или напрямую .venv/Scripts/python.exe ...
```

> `onnxruntime` и `onnxruntime-gpu` делят один Python-модуль и не уживаются вместе; insightface
> жёстко зависит от CPU-`onnxruntime`, поэтому после sync нужен `--reinstall onnxruntime-gpu`.
> Обычный `uv run` может синхронизировать транзитивный CPU wheel обратно; для GPU-команд используйте
> `uv run --no-sync` после указанной переустановки. При обновлении lock замените `1.26.0` на
> зафиксированную там версию `onnxruntime-gpu`.
> Устройство выбирается автоматически (`[default.compute].device = "auto"`, см.
> [common/device.py](src/age_gap/common/device.py)); можно задать `"cpu"`/`"cuda"`.

**Секреты** — `src/age_gap/settings/.env`:

- `VK_TOKEN=...` — токен VK для парсинга (`wall.get` работает с сервисным; `wall.getById` /
  `wall.getComments` — только не-сервисный);
- `GIGACHAT_SECRET=...` — доступ к GigaChat для LLM-извлечения возраста. Параметры модели/эндпоинтов
  — в `settings.toml` (`[default.gigachat]`), сертификат — `settings/certs/`.

---

## Использование

### Конвейер данных

```bash
# 1) Парсинг стены сообщества (wall.get, сервисный токен). --append для нескольких стен (дедуп):
uv run python scripts/ingest.py --owner -77072632 --count 1000
uv run python scripts/ingest.py --owner -134190297 --count 40000 --append

# 2) Детекция/выравнивание/кроп лиц (GPU авто; резюмируемо — можно перезапускать):
uv run python scripts/preprocess.py            # при сбое GPU: --device cpu

# 3) Baseline ArcFace-эмбеддинги usable-лиц:
uv run python scripts/embed.py

# 4) Группы личностей + извлечение/привязка возраста (резюмируемо):
uv run python scripts/build_groups.py --age-extractor combined   # regex | llm | combined

# 5) Генерация пар и leakage-safe сплит:
uv run python scripts/build_pairs.py --neg-per-pos 1
uv run python scripts/split.py

# (опц.) Hard-negative mining → пересплит:
uv run python scripts/mine_hard_negatives.py --top-k 3 --max 6000
uv run python scripts/split.py
```

### LLM-аудит возраста (GigaChat)

```bash
# Проход по всем подписям LLM-ом (10 потоков), отчёт regex-vs-LLM; --apply пересобирает группы:
uv run python scripts/audit_ages.py --concurrency 10
uv run python scripts/audit_ages.py --apply        # группы на LLM-возрастах (CachedAgeExtractor)
```

### Оценка и эксперименты

```bash
# Baseline-бенчмарк (ROC-AUC/EER/TAR@FAR по age-gap) и retrieval:
uv run python scripts/benchmark.py --split test
uv run python scripts/retrieval.py --split test

# Честный протокол (слабый facenet: frozen vs дообученный) + внешний FG-NET:
uv run python scripts/finetune_facenet.py --epochs 10 --lr 3e-5 --trainable head
uv run python scripts/eval_fgnet.py

# Сводная ablation (frozen | +pairs | +pairs+age-reg) на LFW/AgeDB-30/CALFW/FG-NET:
#   положить agedb_30.bin / calfw.bin в data/external/ (loader понимает .bin).
uv run python scripts/ablation.py --epochs 10

# Реальные пары vs синтетическое старение (proxy + реальная FRAN):
uv run python scripts/synthetic_baseline.py --epochs 10 --aging both

# Несколько backbone (facenet / ArcFace / AdaFace) одним протоколом:
uv run python scripts/backbones_benchmark.py --epochs 10 --lr 3e-5
uv run python scripts/align_mtcnn.py                                  # MTCNN-кропы для facenet
uv run python scripts/backbones_benchmark.py --backbones facenet --crops faces_mtcnn

# Ревизионный пакет T-BIOM: CACD-VS, blinded audit, SOTA и strong-backbone matrix.
# После GPU-установки используйте --no-sync, чтобы uv не заменил onnxruntime-gpu CPU-wheel'ом.
uv run --no-sync python scripts/build_cacd_vs.py
uv run --no-sync python scripts/eval_cacd_vs.py --backbone adaface_ir101
uv run --no-sync python scripts/eval_cacd_vs.py --checkpoints models/bb_facenet_seed42.pt models/bb_facenet_seed1.pt models/bb_facenet_seed2.pt
uv run --no-sync python scripts/build_blinded_audit.py --n 400
uv run --no-sync python scripts/build_age_audit_baselines.py
# После двух независимых ответов и adjudication:
uv run --no-sync python scripts/score_blinded_audit.py
uv run --no-sync python scripts/sota_common_protocol.py --methods mtlface cacon --epochs 8
uv run --no-sync python scripts/strong_backbone_study.py --epochs 8
uv run --no-sync python scripts/independent_embedding_audit.py
uv run --no-sync python scripts/model_inventory.py
uv run --no-sync python scripts/pretraining_overlap_audit.py
$env:ENV_FOR_DYNACONF='reddit'; uv run --no-sync python scripts/eval_cross_platform.py --ckpts models/bb_facenet_seed42.pt models/bb_facenet_seed1.pt models/bb_facenet_seed2.pt --tag vk2reddit
Remove-Item Env:ENV_FOR_DYNACONF; uv run --no-sync python scripts/cross_source.py --ckpts models/bb_reddit_src_s42.pt models/bb_reddit_src_s1.pt models/bb_reddit_src_s2.pt --tag reddit2vk

# (опц.) Adapter поверх замороженного ArcFace; apparent-age из комментариев (нужен не-сервисный токен):
uv run python scripts/train_adapter.py --epochs 60 --gap-weight 2.0
uv run python scripts/fetch_comments.py --limit 200
```

> Артефакты: `data/...`, `cache/embeddings/*.npz`, `models/*.pt` (гитигнорятся),
> `data/external/{FGNET.zip,fgnet_crops.npz,agedb_30.bin,calfw.bin}`, `metrics/`.

---

## Структура проекта

```text
src/age_gap/
  settings/        конфиг (dynaconf + .env), сертификаты
  common/          схемы данных, JSONL-IO, логирование, выбор устройства (CPU/GPU)
  parsing/         VK-клиент (wall.get/getById), ingest + параллельная загрузка
  preprocessing/   детекция, выравнивание, quality, оркестрация; MTCNN-выравнивание (mtcnn_align)
  datasets/        группы личностей, age-anchors (regex + LLM + аудит), пары, hard-негативы, сплит
  models/          ArcFace-эмбеддер (frozen), adapter, facenet, iresnet (ArcFace), adaface_ir, aging (FRAN)
  training/        датасет пар, contrastive/age-supervised лосс, fine-tune backbone, synthetic
  evaluation/      метрики, age-gap бенчмарк, retrieval, внешние (LFW/AgeDB/CALFW/.bin/FG-NET), suite
  infrastructure/  GigaChat-клиент (chat/embeddings, retry, async)
scripts/           CLI этапов конвейера, аудита возраста, оценки и экспериментов
tests/             юнит-тесты (regex/слэш, привязка, пары, сплит, метрики, backbone, aging, LLM-кэш)
```

---

## Разработка

```bash
uv run ruff check . && uv run mypy && uv run pytest -q
```

Линт/типы — в [pyproject.toml](pyproject.toml); хуки — [.pre-commit-config.yaml](.pre-commit-config.yaml).

> Результаты получены преимущественно на сообществах VK и носят исследовательский характер.
> Кросс-платформенный перенос подтверждён на независимом **не-VK** источнике (Reddit
> r/PastAndPresentPics: overall ROC-AUC frozen 0.718 → +pairs 0.749); HTML-скрейпер old.reddit —
> `scripts/ingest_reddit.py --mode html` (англ. подписи + восстановление коллажей; см.
> статью). Дальнейшая внешняя валидация полезна.
