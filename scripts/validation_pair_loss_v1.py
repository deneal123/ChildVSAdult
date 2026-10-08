"""Explicit validation contrastive loss from fixed normalized-embedding scores.

Aggregate all validation rows at once (or sum numerators/denominators), never average
batch means. This measurement does not change checkpoint selection or recover missing
historical epoch scores. It requires a new versioned producer for real trajectories.
"""

import numpy as np


def measure(scores, labels, *, margin=.3, weights=None):
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    if (scores.ndim != 1 or not len(scores) or labels.shape != scores.shape
            or not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all()
            or np.any(np.abs(scores) > 1 + 1e-5)
            or not np.isfinite(margin) or not -1 <= margin <= 1):
        raise ValueError("finite cosine scores, binary aligned labels and valid margin required")
    # No silent clipping: retain the same finite-precision score as the trainer.
    loss = labels * (1 - scores) + (1 - labels) * np.maximum(scores - margin, 0)
    mass = np.ones(len(scores)) if weights is None else np.asarray(weights, float)
    if (mass.shape != scores.shape or not np.isfinite(mass).all()
            or np.any(mass < 0) or mass.sum() <= 0):
        raise ValueError("aligned finite nonnegative weights with positive total required")
    numerator, denominator = float(np.sum(loss * mass)), float(mass.sum())
    return dict(version="validation-pair-loss-v1", n_pairs=len(scores),
                margin=float(margin), numerator=numerator, denominator=denominator,
                validation_loss=numerator / denominator,
                aggregation="global pair-weighted mean; not mean of batch means",
                limitation="measurement only; no historical recovery or checkpoint reselection")
