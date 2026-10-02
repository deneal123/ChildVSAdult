"""Exploratory representation diagnostics with identity-disjoint age-gap probe folds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def resolve_identity_groups(
    dataset: Any, group_ids: Sequence[str | None] | None
) -> tuple[list[str | None] | None, str]:
    """Return ``(groups, status)`` for a dataset, without guessing."""
    if group_ids is not None:
        groups = list(group_ids)
        if len(groups) != len(dataset):
            return None, "unavailable: group_ids length does not match dataset"
        return groups, "ok"
    accessor = getattr(dataset, "identity_groups", None)
    if accessor is not None:
        groups = list(accessor)
        if len(groups) != len(dataset):
            return None, "unavailable: dataset.identity_groups length mismatch"
        return groups, "ok"
    return None, "unavailable: missing identity-group metadata"


# --------------------------------------------------------------------------------------
# Linear CKA (unchanged: this metric was already correct)
# --------------------------------------------------------------------------------------
def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear centered-kernel alignment for two representation matrices."""
    if left.shape[0] != right.shape[0] or left.shape[0] < 2:
        return float("nan")
    x = left - left.mean(axis=0, keepdims=True)
    y = right - right.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denom = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / denom) if denom > 0 else float("nan")


# --------------------------------------------------------------------------------------
# Identity-group-disjoint folds
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GroupFolds:
    """Fold assignment over row indices, with the identity accounting needed to defend it."""

    fold_of_row: np.ndarray  # int64, -1 for rows whose identity is unknown
    n_identities: int
    fold_sizes: tuple[int, ...]
    identity_counts: tuple[int, ...]

    @property
    def usable(self) -> bool:
        return self.n_identities > 0 and all(size > 0 for size in self.fold_sizes)


def identity_group_folds(
    groups: Sequence[str | None], folds: int, *, seed: int = 42
) -> GroupFolds:
    """Assign whole identity groups to folds. Never falls back to row-level folds."""
    if folds < 2:
        raise ValueError("folds must be >= 2")
    rows = np.asarray([g is not None for g in groups], dtype=bool)
    known = np.flatnonzero(rows)
    fold_of_row = np.full(len(groups), -1, dtype=np.int64)
    if known.size == 0:
        return GroupFolds(fold_of_row, 0, tuple(0 for _ in range(folds)), ())

    unique = sorted({str(groups[i]) for i in known})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    assignment = {unique[int(i)]: int(k) % folds for k, i in enumerate(order)}
    for i in known:
        fold_of_row[i] = assignment[str(groups[i])]
    identity_counts = tuple(
        sum(1 for g in unique if assignment[g] == f) for f in range(folds)
    )
    fold_sizes = tuple(int((fold_of_row == f).sum()) for f in range(folds))
    return GroupFolds(fold_of_row, len(unique), fold_sizes, identity_counts)


# --------------------------------------------------------------------------------------
# Ridge probe with training-mean intercept + per-fold constant baselines
# --------------------------------------------------------------------------------------
def _ridge_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, regularization: float
) -> np.ndarray:
    """Dual ridge with an intercept realized by centering on the training fold."""
    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = float(y_train.mean())
    xc = x_train - x_mean
    alpha = np.linalg.solve(
        xc @ xc.T + regularization * np.eye(len(xc)), y_train - y_mean
    )
    return (x_test - x_mean) @ xc.T @ alpha + y_mean


def ridge_probe_with_baselines(
    features: np.ndarray,
    target: np.ndarray,
    groups: Sequence[str | None],
    *,
    folds: int = 5,
    seed: int = 42,
    regularization: float = 1.0,
) -> dict[str, Any]:
    """Out-of-fold ridge MAE with identity-group-disjoint folds and constant baselines.

    Returns ``status`` explaining any explicit null. Probe values are ``NaN`` when the split
    is not usable; callers must not interpret NaN as a result.
    """
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    out: dict[str, Any] = {
        "status": "ok",
        "mae": float("nan"),
        "baseline_mean_mae": float("nan"),
        "baseline_median_mae": float("nan"),
        "skill_over_mean": float("nan"),
        "skill_over_best_constant": float("nan"),
        "n_rows": int(len(target)),
        "n_identities": 0,
        "n_folds_used": 0,
        "fold_mae": [],
    }
    if (features.ndim != 2 or target.ndim != 1
            or features.shape[0] != target.shape[0] or len(groups) != len(target)):
        out["status"] = "unavailable: features/target shape mismatch"
        return out
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        out["status"] = "unavailable: nonfinite features or targets"
        return out
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    if len(target) < folds * 2:
        out["status"] = "unavailable: too few rows for the requested folds"
        return out

    gf = identity_group_folds(groups, folds, seed=seed)
    if gf.n_identities == 0:
        out["status"] = "unavailable: no identity-group metadata on rows"
        return out
    if gf.n_identities < folds:
        out["status"] = "unavailable: fewer distinct identities than folds"
        return out
    if not all(size > 0 for size in gf.fold_sizes):
        out["status"] = "unavailable: a fold would be empty"
        return out
    if gf.identity_counts and min(gf.identity_counts) < 2:
        out["status"] = "unavailable: a fold would carry a single identity"
        return out

    errors: list[float] = []
    mean_errors: list[float] = []
    median_errors: list[float] = []
    fold_mae: list[float] = []
    for fold in range(folds):
        test = np.flatnonzero(gf.fold_of_row == fold)
        train = np.flatnonzero((gf.fold_of_row >= 0) & (gf.fold_of_row != fold))
        if train.size == 0 or test.size == 0:
            out["status"] = "unavailable: empty train or test fold"
            return out
        prediction = _ridge_predict(features[train], target[train], features[test], regularization)
        fold_err = np.abs(prediction - target[test])
        errors.extend(fold_err.tolist())
        mean_errors.extend(np.abs(float(target[train].mean()) - target[test]).tolist())
        median_errors.extend(np.abs(float(np.median(target[train])) - target[test]).tolist())
        fold_mae.append(float(fold_err.mean()))

    mae = float(np.mean(errors))
    base_mean = float(np.mean(mean_errors))
    base_median = float(np.mean(median_errors))
    best_constant = min(base_mean, base_median)
    out.update(
        {
            "n_rows": int(len(errors)),
            "n_unknown_group_rows": int(np.sum(gf.fold_of_row < 0)),
            "mae": mae,
            "baseline_mean_mae": base_mean,
            "baseline_median_mae": base_median,
            "skill_over_mean": float(1.0 - mae / base_mean) if base_mean > 0 else float("nan"),
            "skill_over_best_constant": (
                float(1.0 - mae / best_constant) if best_constant > 0 else float("nan")
            ),
            "n_identities": int(gf.n_identities),
            "n_folds_used": int(folds),
            "identity_counts_per_fold": list(gf.identity_counts),
            "fold_mae": fold_mae,
        }
    )
    return out


# --------------------------------------------------------------------------------------
# Representation comparison
# --------------------------------------------------------------------------------------
def stratified_sample(
    labels: np.ndarray, max_pairs: int, *, seed: int = 42
) -> np.ndarray:
    """Seeded stratified sample: half positives, half negatives, no prefix slicing."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    per_class = max_pairs // 2
    pos_pick = rng.choice(pos, size=min(per_class, len(pos)), replace=False)
    neg_pick = rng.choice(neg, size=min(per_class, len(neg)), replace=False)
    return np.sort(np.concatenate([pos_pick, neg_pick])) if pos_pick.size or neg_pick.size else np.array([], dtype=np.int64)


def compare_representations(
    frozen: torch.nn.Module,
    tuned: torch.nn.Module,
    dataset: Any,
    device: str,
    *,
    batch_size: int = 32,
    max_pairs: int = 512,
    seed: int = 42,
    folds: int = 5,
    group_ids: Sequence[str | None] | None = None,
    split_label: str | None = None,
) -> dict[str, Any]:
    """CKA, drift, separability and an identity-group-disjoint age-gap probe."""
    if batch_size < 1 or max_pairs < 2:
        raise ValueError("positive batch size and max_pairs >= 2 required")
    labels_all = np.asarray(dataset.labels)
    gaps_all = np.asarray(dataset.gaps)
    groups, group_status = resolve_identity_groups(dataset, group_ids)
    selected = stratified_sample(labels_all, max_pairs, seed=seed)
    if selected.size == 0:
        return {"probe_status": "unavailable: empty diagnostic sample"}

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

    def separability(scores: np.ndarray) -> float:
        return float(scores[y == 1].mean() - scores[y == 0].mean())

    b_pair = np.abs(b_first - b_second)
    a_pair = np.abs(a_first - a_second)
    positive = (y == 1) & (gaps_all[selected] >= 0)
    pos_idx = np.flatnonzero(positive)
    pos_groups = [groups[i] for i in selected[pos_idx]] if groups is not None else None
    frozen_probe = ridge_probe_with_baselines(
        b_pair[pos_idx], gaps_all[selected][pos_idx], pos_groups or [], folds=folds, seed=seed
    )
    tuned_probe = ridge_probe_with_baselines(
        a_pair[pos_idx], gaps_all[selected][pos_idx], pos_groups or [], folds=folds, seed=seed
    )

    return {
        "linear_cka": linear_cka(before, after),
        "mean_cosine_drift": float(drift.mean()),
        "frozen_identity_separability": separability(b_scores),
        "tuned_identity_separability": separability(a_scores),
        "frozen_age_gap_probe_mae": frozen_probe["mae"],
        "tuned_age_gap_probe_mae": tuned_probe["mae"],
        "frozen_age_gap_probe": frozen_probe,
        "tuned_age_gap_probe": tuned_probe,
        "probe_status": tuned_probe["status"],
        "group_metadata_status": group_status,
        "diagnostic_pairs": float(selected.size),
        "diagnostic_scope": {
            "split": split_label,
            "selection_split_reused": True if split_label is not None else None,
            "independence_assumption": "exploratory: this split is also the checkpoint-selection split",
            "sample": f"seeded stratified half-positive/half-negative, max_pairs={max_pairs}",
            "positive_rows_in_probe": int(pos_idx.size),
            "seed": seed,
        },
    }
