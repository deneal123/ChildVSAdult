"""Mechanistic diagnostics for representation changes after fine-tuning."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from age_gap.training.finetune import ImagePairDataset


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear centered-kernel alignment for two representation matrices."""
    if left.shape[0] != right.shape[0] or left.shape[0] < 2:
        return float("nan")
    x = left - left.mean(axis=0, keepdims=True)
    y = right - right.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denom = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / denom) if denom > 0 else float("nan")


def _ridge_probe_mae(features: np.ndarray, target: np.ndarray, folds: int = 5) -> float:
    """Deterministic out-of-fold ridge MAE without an additional ML dependency."""
    if len(target) < folds * 2:
        return float("nan")
    indices = np.arange(len(target))
    errors: list[float] = []
    for fold in range(folds):
        test = indices[indices % folds == fold]
        train = indices[indices % folds != fold]
        x_train, x_test = features[train], features[test]
        mean = x_train.mean(axis=0, keepdims=True)
        scale = x_train.std(axis=0, keepdims=True) + 1e-6
        x_train = (x_train - mean) / scale
        x_test = (x_test - mean) / scale
        # Dual ridge is substantially cheaper than inverting a 512x512 matrix
        # when the diagnostic sample is small.
        regularization = 1.0
        alpha = np.linalg.solve(
            x_train @ x_train.T + regularization * np.eye(len(train)), target[train]
        )
        prediction = x_test @ x_train.T @ alpha
        errors.extend(np.abs(prediction - target[test]).tolist())
    return float(np.mean(errors))


def compare_representations(
    frozen: torch.nn.Module,
    tuned: torch.nn.Module,
    dataset: ImagePairDataset,
    device: str,
    *,
    batch_size: int = 32,
    max_pairs: int = 512,
) -> dict[str, float]:
    """Measure CKA, sample-wise drift, separability, and an age-gap probe."""
    labels_all = np.asarray(dataset.labels)
    positive_indices = np.flatnonzero(labels_all == 1)
    negative_indices = np.flatnonzero(labels_all == 0)
    per_class = max_pairs // 2
    selected = np.concatenate([positive_indices[:per_class], negative_indices[:per_class]])
    if len(selected) == 0:
        return {}
    subset = Subset(dataset, selected.tolist())
    loader = DataLoader(subset, batch_size=batch_size)
    before_first: list[np.ndarray] = []
    before_second: list[np.ndarray] = []
    after_first: list[np.ndarray] = []
    after_second: list[np.ndarray] = []
    before_scores: list[np.ndarray] = []
    after_scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    frozen.eval()
    tuned.eval()
    with torch.no_grad():
        for first, second, label, _weight in loader:
            first, second = first.to(device), second.to(device)
            b_first, b_second = frozen(first), frozen(second)
            a_first, a_second = tuned(first), tuned(second)
            before_first.append(b_first.cpu().numpy())
            before_second.append(b_second.cpu().numpy())
            after_first.append(a_first.cpu().numpy())
            after_second.append(a_second.cpu().numpy())
            before_scores.append((b_first * b_second).sum(dim=-1).cpu().numpy())
            after_scores.append((a_first * a_second).sum(dim=-1).cpu().numpy())
            labels.append(label.numpy())

    b_first = np.concatenate(before_first)
    b_second = np.concatenate(before_second)
    a_first = np.concatenate(after_first)
    a_second = np.concatenate(after_second)
    before = np.concatenate([b_first, b_second])
    after = np.concatenate([a_first, a_second])
    b_scores = np.concatenate(before_scores)
    a_scores = np.concatenate(after_scores)
    y = np.concatenate(labels).astype(int)
    drift = 1.0 - np.sum(before * after, axis=1)

    pair_count = len(selected)
    gaps = np.asarray(dataset.gaps)[selected]
    positive = (y == 1) & (gaps >= 0)
    # Pair-difference features quantify how readily a linear model can recover
    # age gap; lower MAE means more age information remains in the embedding.
    b_pair = np.abs(b_first - b_second)
    a_pair = np.abs(a_first - a_second)

    def separability(scores: np.ndarray) -> float:
        return float(scores[y == 1].mean() - scores[y == 0].mean())

    return {
        "linear_cka": linear_cka(before, after),
        "mean_cosine_drift": float(drift.mean()),
        "frozen_identity_separability": separability(b_scores),
        "tuned_identity_separability": separability(a_scores),
        "frozen_age_gap_probe_mae": _ridge_probe_mae(b_pair[positive], gaps[positive]),
        "tuned_age_gap_probe_mae": _ridge_probe_mae(a_pair[positive], gaps[positive]),
        "diagnostic_pairs": float(pair_count),
    }
