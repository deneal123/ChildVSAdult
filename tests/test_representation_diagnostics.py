from __future__ import annotations

import numpy as np

from age_gap.evaluation.representation_diagnostics import linear_cka


def test_linear_cka_identical_representations() -> None:
    values = np.arange(30, dtype=np.float64).reshape(10, 3)
    assert np.isclose(linear_cka(values, values), 1.0)


def test_linear_cka_requires_matching_rows() -> None:
    left = np.ones((3, 2))
    right = np.ones((4, 2))
    assert np.isnan(linear_cka(left, right))
