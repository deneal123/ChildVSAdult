import numpy as np
import pytest
import torch

from age_gap.training.losses import ContrastivePairLoss
from scripts.validation_pair_loss_v1 import measure


def test_matches_actual_training_formula_and_global_weighting():
    scores = np.array([.8, .7, -.1, .4])
    labels = np.array([1, 0, 0, 1])
    weights = np.array([1., 2., 3., 4.])
    za = torch.tensor(np.tile([1., 0.], (4, 1)))
    zb = torch.tensor(np.column_stack([scores, np.sqrt(1 - scores ** 2)]))
    expected = ContrastivePairLoss(.3)(za, zb, torch.tensor(labels), torch.tensor(weights)).item()
    result = measure(scores, labels, weights=weights)
    assert result["validation_loss"] == pytest.approx(expected)
    parts = [measure(scores[:1], labels[:1], weights=weights[:1]),
             measure(scores[1:], labels[1:], weights=weights[1:])]
    aggregate = sum(p["numerator"] for p in parts) / sum(p["denominator"] for p in parts)
    assert aggregate == pytest.approx(expected)
    assert np.mean([p["validation_loss"] for p in parts]) != pytest.approx(expected)


@pytest.mark.parametrize("scores,labels,weights", [([], [], None), ([2], [1], None),
                         ([float("nan")], [1], None), ([.2], [2], None),
                         ([.2], [1], [0]), ([.2], [1], [-1]), ([.2], [], None)])
def test_rejects_invalid_measurement(scores, labels, weights):
    with pytest.raises(ValueError):
        measure(scores, labels, weights=weights)
