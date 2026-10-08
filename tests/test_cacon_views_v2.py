import copy

import cv2
import numpy as np
import pytest

from scripts.cacon_crop_v2 import random_resized_crop
from scripts.cacon_views_v2 import ViewPolicy, augment_view, three_views


def image():
    return np.random.default_rng(7).integers(0, 256, (31, 23, 3), dtype=np.uint8)


def test_seed_reproduces_composition_without_mutating_inputs():
    original = image()
    before = original.copy()
    policy = ViewPolicy(color_probability=1, blur_probability=1)
    first, metadata = augment_view(original, np.random.default_rng(42), policy=policy)
    repeated, repeated_metadata = augment_view(original, np.random.default_rng(42), policy=policy)
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(original, before)
    assert metadata == repeated_metadata and metadata["color"] is not None
    assert metadata["blur_sigma"] is not None and first.shape == original.shape
    assert first.dtype == np.uint8


def test_continuing_rng_advances_across_calls():
    rng = np.random.default_rng(42)
    before = copy.deepcopy(rng.bit_generator.state)
    first, first_metadata = augment_view(image(), rng)
    second, second_metadata = augment_view(image(), rng)
    assert before != rng.bit_generator.state
    assert first_metadata != second_metadata and not np.array_equal(first, second)


def test_disabled_color_and_blur_match_independent_crop_component():
    policy = ViewPolicy(color_probability=0, blur_probability=0)
    augmented, metadata = augment_view(image(), np.random.default_rng(42), policy=policy)
    expected, _ = random_resized_crop(image(), np.random.default_rng(42))
    np.testing.assert_array_equal(augmented, expected)
    assert metadata["color"] is None and metadata["blur_sigma"] is None


def test_blur_matches_declared_sigma_and_kernel():
    policy = ViewPolicy(color_probability=0, blur_probability=1, sigma_min=0.6, sigma_max=0.6)
    augmented, metadata = augment_view(image(), np.random.default_rng(42), policy=policy)
    cropped, _ = random_resized_crop(image(), np.random.default_rng(42))
    np.testing.assert_array_equal(augmented, cv2.GaussianBlur(cropped, (3, 3), 0.6))
    assert metadata["blur_sigma"] == 0.6


def test_three_views_preserve_supplied_synthesis_and_do_not_claim_generator():
    source, synthesized = image(), np.full_like(image(), 23)
    views, metadata = three_views(source, synthesized, np.random.default_rng(42))
    assert all(view.shape == source.shape for view in views)
    assert not np.array_equal(views[0], views[1])
    np.testing.assert_array_equal(views[2], synthesized)
    views[2][0, 0] = 0
    assert synthesized[0, 0].tolist() == [23, 23, 23]
    assert metadata["generator_implemented"] is False


@pytest.mark.parametrize(
    "options",
    [
        {"color_probability": -1},
        {"brightness": float("nan")},
        {"hue": 0.6},
        {"sigma_min": 0},
        {"sigma_max": float("inf")},
        {"blur_kernel": 2},
        {"blur_kernel": True},
    ],
)
def test_invalid_policy_refused(options):
    with pytest.raises(ValueError):
        ViewPolicy(**options)


def test_mismatched_synthesis_and_non_uint8_refused():
    with pytest.raises(ValueError):
        three_views(image(), np.zeros((2, 2, 3), dtype=np.uint8), np.random.default_rng(42))
    with pytest.raises(ValueError):
        augment_view(image().astype(np.float32), np.random.default_rng(42))


@pytest.mark.parametrize("options", [{"hue": False}, {"sigma_min": True}, {"sigma_max": None}])
def test_nonnumeric_hue_sigma_policy_refused(options):
    with pytest.raises(ValueError):
        ViewPolicy(**options)


def test_negative_contrast_values_clip_to_black_without_absolute_reflection(monkeypatch):
    source = np.zeros((3, 4, 3), dtype=np.uint8)
    source[:, 2:] = 255
    monkeypatch.setattr(
        "scripts.cacon_views_v2.random_resized_crop", lambda image, rng: (image.copy(), {})
    )
    policy = ViewPolicy(
        color_probability=1, brightness=0, contrast=1, saturation=0, hue=0, blur_probability=0
    )
    output, metadata = augment_view(source, np.random.default_rng(42), policy=policy)
    assert metadata["color"]["contrast"] > 1
    assert output[:, :2].max() == 0
    assert output[:, 2:].min() == 255
