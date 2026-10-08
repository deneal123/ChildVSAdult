import cv2
import numpy as np
import pytest

from scripts.cacon_crop_v2 import random_resized_crop


def image():
    return np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)


def test_exact_full_crop_preserves_source_without_mutation():
    source = image()
    before = source.copy()
    output, meta = random_resized_crop(
        source, np.random.default_rng(0), scale=(1, 1), ratio=(4 / 3, 4 / 3)
    )
    np.testing.assert_array_equal(output, source)
    np.testing.assert_array_equal(source, before)
    assert (meta["top"], meta["left"], meta["height"], meta["width"]) == (0, 0, 24, 32)
    assert not meta["fallback"]


def test_sampled_coordinates_and_resize_agree_with_independent_slice():
    source = image()
    output, meta = random_resized_crop(
        source, np.random.default_rng(4), scale=(0.25, 0.25), ratio=(1, 1)
    )
    top, left, height, width = (meta[k] for k in ("top", "left", "height", "width"))
    assert 0 <= top <= 24 - height and 0 <= left <= 32 - width
    expected = cv2.resize(
        source[top : top + height, left : left + width], (32, 24), interpolation=cv2.INTER_LINEAR
    )
    np.testing.assert_array_equal(output, expected)
    assert output.shape == source.shape and output.dtype == source.dtype


def test_reproducible_but_advancing_rng_stream():
    first = np.random.default_rng(7)
    second = np.random.default_rng(7)
    metadata = []
    for _ in range(6):
        a, ma = random_resized_crop(image(), first)
        b, mb = random_resized_crop(image(), second)
        np.testing.assert_array_equal(a, b)
        assert ma == mb
        metadata.append((ma["top"], ma["left"], ma["height"], ma["width"]))
    assert len(set(metadata)) > 1


def test_bounded_attempts_have_nonempty_center_fallback():
    source = image()
    output, meta = random_resized_crop(
        source, np.random.default_rng(0), scale=(1, 1), ratio=(0.1, 0.1), attempts=1
    )
    assert meta["fallback"] and meta["width"] > 0 and meta["height"] > 0
    assert output.shape == source.shape


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": (0, 1)},
        {"ratio": (2, 1)},
        {"scale": (1, float("nan"))},
        {"attempts": True},
        {"attempts": 0},
    ],
)
def test_invalid_policy_refused(kwargs):
    with pytest.raises(ValueError):
        random_resized_crop(image(), np.random.default_rng(0), **kwargs)


def test_invalid_image_and_implicit_rng_refused():
    with pytest.raises(ValueError):
        random_resized_crop(np.zeros((0, 32, 3), np.uint8), np.random.default_rng(0))
    with pytest.raises(ValueError):
        random_resized_crop(image(), None)
