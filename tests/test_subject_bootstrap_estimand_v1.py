"""Bootstrap multiplicities resample subjects; they do not imply equal-person estimands."""

import numpy as np
import pytest

from scripts.identity_separability_v1 import infer, separation


def samples():
    return (np.array([.8, .7, .9, .6, .1, .3, .2, .4]),
            np.array([1, 1, 1, 1, 0, 0, 0, 0]),
            np.array([0, 1, 2, 3, 0, 1, 2, 3]),
            np.array([0, 1, 2, 3, 1, 2, 3, 0]))


def install_draw(monkeypatch, draw):
    class FixedDraw:
        def integers(self, high, size):
            assert high == size == 4
            return np.array(draw)
    monkeypatch.setattr(np.random, "default_rng", lambda seed: FixedDraw())


def test_all_ones_subject_counts_recover_pair_point(monkeypatch):
    scores, labels, a, b = samples()
    install_draw(monkeypatch, [0, 1, 2, 3])
    result = infer(dict(frozen=scores), labels, a, b, resamples=3)
    point = separation(scores, labels)["standardized_mean_difference"]
    assert result["models"]["frozen"]["ci95"] == pytest.approx([point, point])


def test_genuine_owner_once_impostor_both_endpoint_product(monkeypatch):
    scores, labels, a, b = samples()
    draw = [0, 0, 1, 2]
    install_draw(monkeypatch, draw)
    result = infer(dict(frozen=scores), labels, a, b, resamples=3)
    counts = np.bincount(draw, minlength=4)
    weights = counts[a] * np.where(labels == 1, 1, counts[b])
    expected = separation(scores, labels, weights)["standardized_mean_difference"]
    assert result["models"]["frozen"]["ci95"] == pytest.approx([expected, expected])
    assert result["models"]["frozen"]["standardized_mean_difference"] == pytest.approx(
        separation(scores, labels)["standardized_mean_difference"])
