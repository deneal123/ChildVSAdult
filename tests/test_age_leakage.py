"""Тесты age-leakage probe (E20) — чистая логика на синтетике, без моделей."""

from __future__ import annotations

import numpy as np

from age_gap.common.schemas import Pair
from age_gap.evaluation.age_leakage import age_probe, identity_auc


def _make(face_id: str, label: int, a: str, b: str) -> Pair:
    return Pair(pair_id=face_id, face_a=a, face_b=b, label=label, pair_type="t")


def test_age_probe_recovers_structure():
    # Эмбеддинг кодирует бакет (one-hot + шум) -> bal-acc >> chance.
    rng = np.random.default_rng(0)
    buckets = ["0-17", "18-29", "30-44", "45+"]
    emb: dict[str, np.ndarray] = {}
    fb: dict[str, str] = {}
    for ci, b in enumerate(buckets):
        for j in range(40):
            fid = f"{b}_{j}"
            v = np.zeros(4, dtype=np.float32)
            v[ci] = 1.0
            emb[fid] = v + rng.normal(0, 0.05, 4).astype(np.float32)
            fb[fid] = b
    out = age_probe(emb, fb, seeds=(0, 1))
    assert out["chance"] == 0.25
    assert out["bal_acc"] > 0.8  # возраст легко декодируется


def test_age_probe_random_near_chance():
    rng = np.random.default_rng(1)
    buckets = ["0-17", "18-29", "30-44", "45+"]
    emb: dict[str, np.ndarray] = {}
    fb: dict[str, str] = {}
    for b in buckets:
        for j in range(40):
            fid = f"{b}_{j}"
            emb[fid] = rng.normal(0, 1, 8).astype(np.float32)  # не зависит от бакета
            fb[fid] = b
    out = age_probe(emb, fb, seeds=(0, 1))
    assert out["bal_acc"] < 0.45  # около случайности (0.25)


def test_identity_auc_perfect():
    emb = {
        "a": np.array([1.0, 0.0]),
        "a2": np.array([0.98, 0.0]),  # тот же человек
        "b": np.array([0.0, 1.0]),
    }
    pairs = [_make("p1", 1, "a", "a2"), _make("p2", 0, "a", "b")]
    assert identity_auc(emb, pairs) == 1.0
