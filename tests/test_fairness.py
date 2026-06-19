"""Тесты логики демографического аудита (страты/AUC) — чистые функции, без моделей."""

from __future__ import annotations

import math

from age_gap.evaluation.fairness import _GENDER, _age_band, _stratum_auc


def test_age_band_boundaries():
    assert _age_band(0) == "0-17"
    assert _age_band(17) == "0-17"
    assert _age_band(18) == "18-29"
    assert _age_band(29) == "18-29"
    assert _age_band(30) == "30-44"
    assert _age_band(44) == "30-44"
    assert _age_band(45) == "45+"
    assert _age_band(90) == "45+"


def test_gender_mapping():
    assert _GENDER[0] == "F"
    assert _GENDER[1] == "M"


def test_stratum_auc_empty():
    auc, n_pos, n_neg = _stratum_auc([])
    assert math.isnan(auc) and n_pos == 0 and n_neg == 0


def test_stratum_auc_too_few_returns_nan_but_counts():
    rows = [(0.9, 1)] * 5 + [(0.1, 0)] * 5  # ниже min_pos/min_neg=20
    auc, n_pos, n_neg = _stratum_auc(rows)
    assert math.isnan(auc) and n_pos == 5 and n_neg == 5


def test_stratum_auc_perfect_separation():
    rows = [(0.9, 1)] * 25 + [(0.1, 0)] * 25
    auc, n_pos, n_neg = _stratum_auc(rows)
    assert n_pos == 25 and n_neg == 25
    assert auc == 1.0
