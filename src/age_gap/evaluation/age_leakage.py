"""E20: линейный probe age-leakage — декодируется ли apparent-возраст из identity-эмбеддинга.

Поведенческий E13 показал, что frozen-модель опирается на возраст (шорткат). Здесь —
representational-подтверждение: обучаем линейный классификатор предсказывать возрастной бакет ИЗ
эмбеддинга. Высокая точность = возраст «течёт» в эмбеддинг. Ожидание: frozen (высоко) → +pairs
(ниже: учились различать по личности) → +disentangle (ещё ниже: GRL явно убирает возраст). Параллельно
проверяем identity-верификацию (overall AUC) — снятие возраста не должно вредить личности.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from age_gap.common.logging import get_logger
from age_gap.common.schemas import Pair
from age_gap.evaluation.metrics import roc_auc

log = get_logger(__name__)


def age_probe(
    embeddings: dict[str, np.ndarray],
    face_bucket: dict[str, str],
    seeds: tuple[int, ...] = (42, 1, 2),
    train_frac: float = 0.7,
) -> dict[str, float]:
    """Балансированная точность линейного probe возраст-бакета из эмбеддингов (mean±std по сидам).

    Балансированная точность (macro-recall) робастна к перекосу классов; chance = 1/#классов.
    Один и тот же train/test-сплит лиц должен применяться к моделям для сопоставимости (фикс. сиды).
    """
    ids = sorted(set(embeddings) & set(face_bucket))
    x = np.stack([embeddings[f] for f in ids])
    y = np.asarray([face_bucket[f] for f in ids])
    n = len(ids)
    accs: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        cut = int(train_frac * n)
        tr, te = idx[:cut], idx[cut:]
        scaler = StandardScaler().fit(x[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(scaler.transform(x[tr]), y[tr])
        pred = clf.predict(scaler.transform(x[te]))
        accs.append(float(balanced_accuracy_score(y[te], pred)))
    return {
        "bal_acc": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "chance": 1.0 / len(set(y)),
        "n": float(n),
    }


def identity_auc(embeddings: dict[str, np.ndarray], pairs: list[Pair]) -> float:
    """Overall ROC-AUC верификации на парах по этим эмбеддингам (raw dot, как в основном протоколе)."""
    scores: list[float] = []
    labels: list[int] = []
    for p in pairs:
        if p.face_a in embeddings and p.face_b in embeddings:
            scores.append(float(embeddings[p.face_a] @ embeddings[p.face_b]))
            labels.append(p.label)
    return roc_auc(np.asarray(scores), np.asarray(labels))


def summarize(
    name: str, embeddings: dict[str, np.ndarray], face_bucket: dict[str, str], pairs: list[Pair]
) -> dict[str, Any]:
    probe = age_probe(embeddings, face_bucket)
    auc = identity_auc(embeddings, pairs)
    log.info(
        "%s: age-probe bal-acc=%.3f±%.3f (chance %.3f), identity overall AUC=%.4f",
        name,
        probe["bal_acc"],
        probe["std"],
        probe["chance"],
        auc,
    )
    return {"model": name, **probe, "identity_auc": auc}


def print_leakage(rows: list[dict[str, Any]]) -> None:
    print(f"\n{'model':<14}{'age-probe bal-acc':>20}{'chance':>9}{'identity AUC':>14}")
    for r in rows:
        print(
            f"{r['model']:<14}{r['bal_acc']:>13.3f}±{r['std']:.3f}"
            f"{r['chance']:>9.3f}{r['identity_auc']:>14.4f}"
        )
