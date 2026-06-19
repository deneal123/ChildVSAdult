"""Калибровка вероятности «тот же человек» с учётом возрастного разрыва (§5.3).

Сырой косинус не калиброван: один и тот же порог означает разную P(same) при gap 2 и gap 30
(на больших разрывах модель переуверена). Здесь обучаем калибратор P(same | cosine, age_gap)
и сравниваем с обычным Platt (только cosine) по ECE / Brier, в т.ч. в разбивке по age_gap.

Чистая логика (ece, fit) тестируется офлайн; данные берутся из эмбеддингов + pairs.jsonl.
"""

from __future__ import annotations

import numpy as np

from age_gap.common.io import data_path, read_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import Pair
from age_gap.models.embeddings import load_embeddings

log = get_logger(__name__)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE: средневзвешенное |уверенность − точность| по бинам вероятности."""
    probs = np.clip(probs, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if not m.any():
            continue
        conf = float(probs[m].mean())
        acc = float(labels[m].mean())
        ece += (m.sum() / n) * abs(conf - acc)
    return ece


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.clip(probs, 0, 1) - labels) ** 2))


def pair_features(
    pairs_file: str | None = None,
    embeddings_file: str | None = None,
) -> dict[str, np.ndarray]:
    """(cos, age_gap, label, split) по парам, у которых известен возраст ОБОИХ лиц."""
    embeddings = load_embeddings(embeddings_file)
    cos: list[float] = []
    gap: list[float] = []
    lab: list[int] = []
    spl: list[str] = []
    pf = pairs_file or str(data_path("data_dir", "processed", "pairs.jsonl"))
    for row in read_jsonl(pf):
        p = Pair.from_dict(row)
        if p.age_a is None or p.age_b is None:
            continue
        ea, eb = embeddings.get(p.face_a), embeddings.get(p.face_b)
        if ea is None or eb is None:
            continue
        cos.append(float(np.dot(ea, eb)))
        gap.append(float(abs(p.age_a - p.age_b)))
        lab.append(int(p.label))
        spl.append(p.split or "none")
    return {
        "cos": np.asarray(cos, np.float64),
        "gap": np.asarray(gap, np.float64),
        "label": np.asarray(lab, np.int64),
        "split": np.asarray(spl, dtype=object),
    }


def _fit_logistic(x: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000, C=10.0).fit(x, y)


def evaluate_calibration(
    pairs_file: str | None = None, embeddings_file: str | None = None
) -> dict[str, object]:
    """Сравнить Platt(cos) и age-gap-aware(cos,gap) калибраторы. Обучение на val, оценка на test."""
    d = pair_features(pairs_file, embeddings_file)
    cos, gap, y, spl = d["cos"], d["gap"], d["label"], d["split"]
    tr = spl == "train"
    te = spl == "test"
    if tr.sum() < 50 or te.sum() < 50:  # fallback: фит на train, иначе на всём
        tr = spl != "test"
    if te.sum() < 50:
        te = np.ones(len(y), bool)

    # Baseline: Platt по одному косинусу.
    platt = _fit_logistic(cos[tr][:, None], y[tr])
    p_platt = platt.predict_proba(cos[te][:, None])[:, 1]

    # Age-gap-aware: cos, gap, cos*gap.
    feats = lambda c, g: np.column_stack([c, g, c * g])  # noqa: E731
    agecal = _fit_logistic(feats(cos[tr], gap[tr]), y[tr])
    p_age = agecal.predict_proba(feats(cos[te], gap[te]))[:, 1]

    yte = y[te]
    out: dict[str, object] = {
        "n_test": int(te.sum()),
        "ece_platt": expected_calibration_error(p_platt, yte),
        "ece_agegap": expected_calibration_error(p_age, yte),
        "brier_platt": brier(p_platt, yte),
        "brier_agegap": brier(p_age, yte),
    }
    # Разбивка ECE по возрастному разрыву (где переуверенность сильнее всего).
    bins = [(0, 5), (5, 15), (15, 25), (25, 200)]
    per_gap = {}
    gte = gap[te]
    for lo, hi in bins:
        m = (gte >= lo) & (gte < hi)
        if m.sum() < 20:
            continue
        per_gap[f"{lo}-{hi}"] = {
            "n": int(m.sum()),
            "ece_platt": round(expected_calibration_error(p_platt[m], yte[m]), 4),
            "ece_agegap": round(expected_calibration_error(p_age[m], yte[m]), 4),
        }
    out["per_gap"] = per_gap
    return out
