import numpy as np
import pytest
import torch

from age_gap.evaluation.benchmark_external import pair_scores


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []
        self.channel_flags = []

    def preprocess(self, image, bgr=True):
        self.channel_flags.append(bgr)
        return image.astype(np.float32)

    def forward(self, batch):
        self.batch_sizes.append(len(batch))
        return torch.nn.functional.normalize(batch, dim=1)


@pytest.mark.parametrize("rgb", [True, False])
def test_streamed_scores_match_full_array_with_bounded_batches(rgb):
    generator = np.random.default_rng(42)
    first = generator.normal(size=(257, 4)).astype(np.float32)
    second = generator.normal(size=(257, 4)).astype(np.float32)
    model = _Model()
    scores = pair_scores(model, list(first), list(second), "cpu", rgb)
    expected = ((first / np.linalg.norm(first, axis=1, keepdims=True)) *
                (second / np.linalg.norm(second, axis=1, keepdims=True))).sum(axis=1)
    np.testing.assert_allclose(scores, expected, atol=1e-7)
    assert max(model.batch_sizes) == 128
    assert sum(model.batch_sizes) == 514
    assert set(model.channel_flags) == {not rgb}


def test_empty_and_mismatched_pairs():
    model = _Model()
    assert pair_scores(model, [], [], "cpu", True).size == 0
    with pytest.raises(ValueError, match="same length"):
        pair_scores(model, [np.ones(4)], [], "cpu", False)
