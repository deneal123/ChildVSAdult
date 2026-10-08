"""Fit-only chronological-age constants on a verified probe's fixed person folds."""

import numpy as np


def constants(probe, all_ages):
    if probe.get("status") != "ok":
        raise ValueError("successful fixed-fold probe required")
    rows = np.asarray(probe["row_indices"])
    persons, folds = np.asarray(probe["persons"]), np.asarray(probe["folds"])
    ages = np.asarray(all_ages, float)
    if (rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows)
            or np.any(rows < 0) or np.any(rows >= len(ages))
            or len(np.unique(rows)) != len(rows)
            or persons.shape != rows.shape or folds.shape != rows.shape
            or folds.dtype.kind not in "iu" or np.any(folds < 0)
            or not np.isfinite(ages[rows]).all() or np.any(ages[rows] < 0)):
        raise ValueError("valid aligned recorded probe coverage required")
    people = sorted(set(persons.tolist()))
    if any(len(set(folds[persons == p])) != 1 for p in people):
        raise ValueError("person-disjoint folds required")
    targets, mean_predictions, median_predictions = ages[rows], np.zeros(len(rows)), np.zeros(len(rows))
    for fold in sorted(set(folds.tolist())):
        fit, test = folds != fold, folds == fold
        if not fit.any():
            raise ValueError("at least two occupied folds required")
        fit_people, counts = np.unique(persons[fit], return_counts=True)
        mass = dict(zip(fit_people, counts, strict=True))
        weights = np.array([1 / mass[p] for p in persons[fit]])
        fit_ages = targets[fit]
        mean_predictions[test] = np.average(fit_ages, weights=weights)
        order = np.argsort(fit_ages, kind="stable")
        index = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)
        median_predictions[test] = fit_ages[order[index]]
    output = {}
    for name, predictions in (("mean", mean_predictions), ("median", median_predictions)):
        errors = np.abs(predictions - targets)
        output[name] = dict(mae_image=float(errors.mean()),
                            mae_person=float(np.mean([errors[persons == p].mean() for p in people])),
                            predictions=predictions.tolist(), absolute_errors=errors.tolist())
    return dict(version="person-age-constant-baselines-v2", n_images=len(rows),
                n_persons=len(people), baselines=output,
                limitation="fit-only equal-person constants; fixed probe folds; private predictions")
