"""Оценить перенос beauty-модели на VK по human-парам: agreement + Bradley-Terry.

    ENV_FOR_DYNACONF=natural uv run python scripts/rating_fit.py --pairs data_beauty/ratings/pairs.jsonl

Метрики:
  * pairwise_accuracy — доля пар, где порядок модели совпал с выбором человека (главное число переноса);
  * spearman(model, BT) — согласие ранга модели с латентной привлекательностью Bradley-Terry.
Пишет metrics/beauty_rating.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def bradley_terry(faces: list[str], wins: dict, games: dict, iters: int = 200) -> dict[str, float]:
    """MM-алгоритм (Zermelo). wins[i]=победы, games[(i,j)]=число игр. -> сила p_i (лог-шкала)."""
    idx = {f: k for k, f in enumerate(faces)}
    n = len(faces)
    p = np.ones(n)
    w = np.zeros(n)
    for f, c in wins.items():
        w[idx[f]] = c
    pair_n = np.zeros((n, n))
    for (i, j), c in games.items():
        pair_n[idx[i], idx[j]] += c
        pair_n[idx[j], idx[i]] += c
    for _ in range(iters):
        denom = np.zeros(n)
        for i in range(n):
            with np.errstate(divide="ignore", invalid="ignore"):
                denom[i] = np.sum(pair_n[i] / (p[i] + p))
        newp = w / np.maximum(denom, 1e-12)
        newp = np.where(newp > 0, newp, p)
        newp /= newp.mean()
        if np.allclose(newp, p, rtol=1e-6):
            p = newp
            break
        p = newp
    return {f: float(np.log(p[idx[f]] + 1e-12)) for f in faces}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=None, help="по умолчанию data_beauty/ratings/pairs.jsonl")
    ap.add_argument("--min-games", type=int, default=2, help="мин. игр у лица для Spearman")
    args = ap.parse_args()

    path = resolve_path(args.pairs) if args.pairs else resolve_path("data_beauty", "ratings", "pairs.jsonl")
    rows = list(read_jsonl(path))
    dec = [r for r in rows if r.get("winner")]  # решительные (не «равны»)
    log.info("Пар всего %d, решительных %d", len(rows), len(dec))

    # модельные скоры каждого лица (из sa/sb, записанных инструментом)
    mscore: dict[str, float] = {}
    for r in rows:
        mscore[r["a"]] = float(r["sa"])
        mscore[r["b"]] = float(r["sb"])

    # pairwise accuracy: совпал ли порядок модели с человеком
    agree = [1.0 if (r["sa"] > r["sb"]) == (r["winner"] == r["a"]) else 0.0
             for r in dec if r["sa"] != r["sb"]]
    pacc = float(np.mean(agree)) if agree else float("nan")

    # Bradley-Terry
    wins: dict[str, int] = {}
    games: dict[tuple, int] = {}
    ngames: dict[str, int] = {}
    for r in dec:
        a, b, win = r["a"], r["b"], r["winner"]
        wins[win] = wins.get(win, 0) + 1
        key = (a, b) if a < b else (b, a)
        games[key] = games.get(key, 0) + 1
        ngames[a] = ngames.get(a, 0) + 1
        ngames[b] = ngames.get(b, 0) + 1
    faces = sorted({f for pair in games for f in pair})
    bt = bradley_terry(faces, wins, games) if faces else {}

    keep = [f for f in faces if ngames.get(f, 0) >= args.min_games]
    sp = float(spearmanr([mscore[f] for f in keep], [bt[f] for f in keep]).statistic) if len(keep) > 3 else float("nan")

    result = {
        "n_pairs_total": len(rows), "n_decisive": len(dec), "n_faces_rated": len(faces),
        "pairwise_accuracy": round(pacc, 4), "n_pairs_used_for_acc": len(agree),
        "spearman_model_vs_bt": round(sp, 4), "n_faces_ge_min_games": len(keep),
        "note": "pairwise_accuracy — доля пар, где модель согласна с человеком (0.5 = случайно)",
    }
    dst = data_path("metrics_dir", "beauty_rating.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
