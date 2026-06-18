"""Тесты прокси-старения (форма/тип сохраняются, пиксели меняются, детерминизм по seed)."""

from __future__ import annotations

import numpy as np
import pytest

from age_gap.models.aging import AgingTransform, HeuristicAging, make_aging


def _img() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)


def test_shape_and_dtype_preserved():
    img = _img()
    out = HeuristicAging(1.0)(img, np.random.default_rng(1))
    assert out.shape == img.shape and out.dtype == np.uint8


def test_changes_pixels():
    img = _img()
    out = HeuristicAging(1.0)(img, np.random.default_rng(1))
    assert not np.array_equal(out, img)


def test_deterministic_with_same_rng_seed():
    img = _img()
    a = HeuristicAging(1.0)(img, np.random.default_rng(7))
    b = HeuristicAging(1.0)(img, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_make_aging_proxy_satisfies_protocol():
    aging = make_aging("proxy")
    assert isinstance(aging, AgingTransform)
    out = aging(_img(), np.random.default_rng(3))
    assert out.shape == (112, 112, 3) and out.dtype == np.uint8


def test_make_aging_unknown_raises():
    with pytest.raises(ValueError, match="неизвестный aging"):
        make_aging("nope")
