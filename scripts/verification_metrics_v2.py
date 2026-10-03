"""Explicit empirical ROC metrics, separate from frozen legacy training code.

Scores are similarities (higher means genuine). TAR uses attainable deterministic
thresholds accepting all tied scores together. Interpolated EER is a separate
ROC-segment quantity, not necessarily an attainable deterministic operating point.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

VERSION = "empirical-roc-v2"


def validate(scores, labels, weights=None):
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    if scores.ndim != 1 or not len(scores) or labels.shape != scores.shape:
        raise ValueError("nonempty equal-length score and label vectors required")
    if not np.isfinite(scores).all() or labels.dtype.kind not in "iu" or set(np.unique(labels)) != {0, 1}:
        raise ValueError("finite similarities and genuine integer binary labels required")
    weights = np.ones(len(scores)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != scores.shape or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("nonnegative finite equal-length weights required")
    if any(weights[labels == label].sum() <= 0 for label in (0, 1)):
        raise ValueError("both classes need positive weight")
    return scores, labels, weights


def eer_from_roc(far, tar):
    """Return explicit interpolated crossing and discrete minimax definitions."""
    far, tar = np.asarray(far, float), np.asarray(tar, float)
    if (far.ndim != 1 or far.shape != tar.shape or len(far) < 2
            or not np.isfinite(far).all() or not np.isfinite(tar).all()
            or (np.diff(far) < 0).any() or (np.diff(tar) < 0).any()
            or ((far < 0) | (far > 1) | (tar < 0) | (tar > 1)).any()
            or far[0] != 0 or tar[0] != 0 or far[-1] != 1 or tar[-1] != 1):
        raise ValueError("complete monotone ROC from (0,0) to (1,1) required")
    frr = 1 - tar
    difference = far - frr
    right = int(np.flatnonzero(difference >= 0)[0])
    if difference[right] == 0:
        crossing = float(far[right])
    else:
        left = right - 1
        fraction = -difference[left] / (difference[right] - difference[left])
        crossing = float(far[left] + fraction * (far[right] - far[left]))
    return {"eer_interpolated": crossing, "eer_discrete_minimax": float(np.maximum(far, frr).min())}


def empirical_metrics(scores, labels, *, far_targets=(.01, .001), weights=None):
    scores, labels, weights = validate(scores, labels, weights)
    targets = list(far_targets)
    if not targets or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float))
                          or not math.isfinite(v) or not 0 <= v <= 1 for v in targets):
        raise ValueError("finite FAR targets in [0,1] required")
    if len(set(targets)) != len(targets):
        raise ValueError("duplicate FAR targets")
    far, tar, thresholds = roc_curve(labels, scores, sample_weight=weights, drop_intermediate=False)
    operating = {}
    for target in targets:
        eligible = np.flatnonzero(far <= target)
        # First maximal-TPR point chooses lowest attained FAR; no splitting ties.
        selected = int(eligible[np.argmax(tar[eligible])])
        threshold = float(thresholds[selected])
        operating[f"{target:g}"] = {"tar": float(tar[selected]), "far_achieved": float(far[selected]),
            "similarity_threshold": threshold if math.isfinite(threshold) else None,
            "reject_all": selected == 0}
    return {"metric_version": VERSION, "n_pairs": len(scores),
            "n_positive": int((labels == 1).sum()), "n_negative": int((labels == 0).sum()),
            "positive_weight": float(weights[labels == 1].sum()), "negative_weight": float(weights[labels == 0].sum()),
            "roc_auc": float(roc_auc_score(labels, scores, sample_weight=weights)),
            **eer_from_roc(far, tar), "operating_points": operating,
            "scope": "test-derived empirical ROC; thresholds not development-calibrated; no CI",
            "eer_scope": "interpolated crossing may require randomized ROC-segment mixing; discrete minimax is separately named"}
