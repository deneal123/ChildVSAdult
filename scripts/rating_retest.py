"""Test-retest: насколько оценщик согласен САМ С СОБОЙ. Даёт ПОТОЛОК для любой модели.

    uv run python scripts/rating_retest.py

Показываем те же лица второй раз (build_rating_likert.py --retest) и сравниваем с первой
разметкой. По классической теории надёжности: если test-retest корреляция = r, то корреляция
наблюдаемой оценки с ИСТИННЫМ латентным вкусом ≈ sqrt(r) — это и есть потолок, который дала бы
ИДЕАЛЬНАЯ модель. Без этого числа непонятно, упёрлись мы в модель или в шум разметки.

Пишет metrics/rating_retest.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import pearsonr, spearmanr

from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def _load(p):
    return {r["face_id"]: float(r["score"]) for r in read_jsonl(resolve_path(p)) if r.get("score")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", default="reports/rating/ratings.jsonl")
    ap.add_argument("--retest", default="reports/rating/ratings_retest.jsonl")
    ap.add_argument("--model-taste", type=float, default=0.500, help="OOF Spearman нашей модели")
    args = ap.parse_args()

    a_map, b_map = _load(args.first), _load(args.retest)
    common = sorted(set(a_map) & set(b_map))
    a = np.array([a_map[f] for f in common])
    b = np.array([b_map[f] for f in common])
    d = np.abs(a - b)

    r = float(spearmanr(a, b).statistic)
    ceiling = float(np.sqrt(max(r, 0.0)))          # корреляция наблюдаемой оценки с латентом
    out = {
        "n_retested": len(common),
        "test_retest_spearman": round(r, 4),
        "test_retest_pearson": round(float(pearsonr(a, b)[0]), 4),
        "exact_agreement": round(float((d == 0).mean()), 4),
        "within_1_point": round(float((d <= 1).mean()), 4),
        "mean_abs_shift": round(float(d.mean()), 3),
        "mean_signed_drift": round(float((b - a).mean()), 3),
        "implied_ceiling_for_model": round(ceiling, 4),
        "our_model_taste": args.model_taste,
        "fraction_of_ceiling_reached": round(args.model_taste / ceiling, 4) if ceiling > 0 else None,
        "note": "потолок = sqrt(test-retest): столько дала бы ИДЕАЛЬНАЯ модель против шумных меток",
    }
    log.info("test-retest=%.3f -> потолок=%.3f | модель=%.3f (%.0f%% потолка)",
             r, ceiling, args.model_taste, 100 * args.model_taste / ceiling)
    dst = data_path("metrics_dir", "rating_retest.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
