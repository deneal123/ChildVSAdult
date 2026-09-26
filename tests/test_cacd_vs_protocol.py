from __future__ import annotations

import numpy as np

from age_gap.evaluation.benchmark_external import accuracy_10fold


def test_accuracy_uses_supplied_protocol_folds() -> None:
    scores = np.asarray([0.9, 0.8, 0.2, 0.1] * 10)
    labels = np.asarray([1, 1, 0, 0] * 10)
    fold_ids = np.repeat(np.arange(10), 4)
    assert accuracy_10fold(scores, labels, fold_ids=fold_ids) == 1.0


def test_accuracy_rejects_wrong_fold_vector_length() -> None:
    with np.testing.assert_raises(ValueError):
        accuracy_10fold(np.ones(10), np.ones(10), fold_ids=np.ones(9))
