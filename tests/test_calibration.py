"""Тесты метрик калибровки (ECE/Brier) — чистые функции, офлайн."""

from __future__ import annotations

import numpy as np

from age_gap.evaluation.calibration import brier, expected_calibration_error


def test_ece_zero_when_perfectly_calibrated():
    # Вероятности равны фактической доле в своём бине -> ECE 0.
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert expected_calibration_error(probs, labels) == 0.0


def test_ece_high_when_overconfident():
    # Уверенность 0.99, а точность 0 -> большой ECE.
    probs = np.full(100, 0.99)
    labels = np.zeros(100, dtype=int)
    assert expected_calibration_error(probs, labels) > 0.9


def test_brier_bounds():
    probs = np.array([1.0, 0.0])
    assert brier(probs, np.array([1, 0])) == 0.0
    assert brier(probs, np.array([0, 1])) == 1.0
