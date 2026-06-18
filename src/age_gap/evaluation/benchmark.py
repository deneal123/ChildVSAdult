"""Baseline-бенчмарк верификации (MVP-2, TODO §11).

Считает косинусную близость для каждой пары по ArcFace-эмбеддингам и метрики верификации
суммарно и в разбивке по возрастному разрыву (age-gap). Эмбеддинги L2-нормированы, поэтому
косинус = скалярное произведение.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from age_gap.common.io import data_path, read_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import Pair
from age_gap.evaluation.metrics import verification_metrics
from age_gap.models.embeddings import load_embeddings

log = get_logger(__name__)

# Границы возрастных бакетов (SKILL §15). Пары без возраста идут в "unknown".
_GAP_BUCKETS = [(0, 5), (5, 10), (10, 15), (15, 25), (25, 200)]


def _gap_bucket(gap: int | None) -> str:
    if gap is None:
        return "unknown"
    for lo, hi in _GAP_BUCKETS:
        if lo <= gap < hi:
            return f"{lo}-{hi}"
    return "25+"


def run(
    pairs_file: str | None = None,
    embeddings_file: str | None = None,
    metrics_out: Path | None = None,
    split: str | None = None,
) -> dict[str, dict[str, float]]:
    pairs_file = pairs_file or str(data_path("data_dir", "processed", "pairs.jsonl"))
    metrics_out = metrics_out or data_path("metrics_dir", "baseline_arcface.json")

    embeddings = load_embeddings(embeddings_file)

    scores: list[float] = []
    labels: list[int] = []
    buckets: list[str] = []
    skipped = 0

    for row in read_jsonl(pairs_file):
        p = Pair.from_dict(row)
        if split is not None and p.split != split:
            continue
        ea = embeddings.get(p.face_a)
        eb = embeddings.get(p.face_b)
        if ea is None or eb is None:
            skipped += 1
            continue
        scores.append(float(np.dot(ea, eb)))
        labels.append(p.label)
        buckets.append(_gap_bucket(p.age_gap))

    if not scores:
        log.warning("Нет пар с доступными эмбеддингами (skipped=%d)", skipped)
        return {}

    scores_arr = np.asarray(scores)
    labels_arr = np.asarray(labels)

    results: dict[str, dict[str, float]] = {"overall": verification_metrics(scores_arr, labels_arr)}

    # Разбивка по age-gap (только бакеты, где есть и позитивы, и негативы — иначе метрики NaN).
    for b in sorted(set(buckets)):
        mask = np.asarray([bk == b for bk in buckets])
        results[f"age_gap:{b}"] = verification_metrics(scores_arr[mask], labels_arr[mask])

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(
        "Бенчмарк: пар=%d (skipped=%d), ROC-AUC=%.4f, EER=%.4f -> %s",
        len(scores),
        skipped,
        results["overall"]["roc_auc"],
        results["overall"]["eer"],
        metrics_out,
    )
    return results
