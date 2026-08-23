from __future__ import annotations

import numpy as np

from prom_service.model_init import write_projection


def test_bootstrap_projection_is_deterministic_and_has_vision_shape(tmp_path):
    digest = "ab" * 32
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    assert write_projection(first, digest, input_dimensions=768) == write_projection(
        second, digest, input_dimensions=768
    )
    with np.load(first, allow_pickle=False) as projection:
        assert projection["mean"].shape == (768,)
        assert projection["components"].shape == (64, 768)
        assert projection["scale"].shape == (768,)
        assert np.allclose(projection["components"] @ projection["components"].T, np.eye(64), atol=1e-5)
