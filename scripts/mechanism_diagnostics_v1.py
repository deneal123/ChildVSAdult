"""CPU-only diagnostics on aligned embeddings; no model fitting or checkpoint selection.

Age regression fits a diagnostic linear probe, not a recognition backbone. All images
of a recorded person share a fold. Results are conditional on fixed embeddings/folds;
age decodability alone does not establish a recognition shortcut or disentanglement.
"""

import numpy as np

VERSION = "person-disjoint-image-age-diagnostics-v1"


def age_probe(embeddings, ages, persons, *, folds=5, seed=42, alpha=1.0):
    """OOF fixed-alpha ridge with fit-only scaling and equal person training mass.

    Missing ages/persons are excluded, never imputed into target measurements.
    Returned row predictions, errors and fold assignments are private artifacts.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2 or len(ages) != len(x) or len(persons) != len(x):
        raise ValueError("aligned image embeddings/ages/persons required")
    if folds < 2 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("positive alpha and at least two folds required")
    ids = [
        None if p is None or str(p).strip().lower() in {"", "none", "nan"}
        else str(p).strip() for p in persons
    ]
    y = np.array([np.nan if a is None else float(a) for a in ages])
    good_age = np.isfinite(y) & (y >= 0)
    good_person = np.array([p is not None for p in ids])
    valid = good_age & good_person
    if not np.isfinite(x[valid]).all():
        raise ValueError("nonfinite embedding on a usable image")
    unique = sorted({ids[i] for i in np.flatnonzero(valid)})
    result = dict(version=VERSION, target="per_image_age", alpha=alpha, seed=seed,
                  n_input=len(x), n_scored=int(valid.sum()),
                  n_missing_age=int((~good_age).sum()),
                  n_missing_person=int((~good_person).sum()), n_persons=len(unique),
                  limitation="fixed-embedding/fold diagnostic; not recognition causality")
    if len(unique) < folds:
        return {**result, "status": "unavailable: fewer persons than folds"}
    order = np.random.default_rng(seed).permutation(len(unique))
    assignment = {unique[int(i)]: k % folds for k, i in enumerate(order)}
    row_fold = np.array([assignment[p] if valid[i] else -1
                         for i, p in enumerate(ids)], dtype=int)
    pred = np.full(len(x), np.nan)
    mean_pred, median_pred = pred.copy(), pred.copy()
    ledgers = []
    for fold in range(folds):
        train = np.flatnonzero((row_fold >= 0) & (row_fold != fold))
        test = np.flatnonzero(row_fold == fold)
        train_ids, counts = np.unique([ids[i] for i in train], return_counts=True)
        mass = dict(zip(train_ids, counts, strict=True))
        weights = np.array([1.0 / mass[ids[i]] for i in train])
        weights *= len(train) / weights.sum()
        mean = np.average(x[train], axis=0, weights=weights)
        variance = np.average((x[train] - mean) ** 2, axis=0, weights=weights)
        scale = np.sqrt(variance)
        scale[scale == 0] = 1.0
        z = (x[train] - mean) / scale
        target_mean = float(np.average(y[train], weights=weights))
        root_weight = np.sqrt(weights)
        zw = z * root_weight[:, None]
        yw = (y[train] - target_mean) * root_weight
        coef = np.linalg.solve(zw.T @ zw + alpha * np.eye(x.shape[1]), zw.T @ yw)
        pred[test] = ((x[test] - mean) / scale) @ coef + target_mean
        mean_pred[test] = target_mean
        ranked = np.argsort(y[train], kind="stable")
        cumulative = np.cumsum(weights[ranked])
        median = float(y[train][ranked[np.searchsorted(cumulative, weights.sum() / 2)]])
        median_pred[test] = median
        ledgers.append(dict(fold=fold, n_fit=len(train), n_test=len(test),
                            fit_persons=sorted(train_ids.tolist()),
                            test_persons=sorted({ids[i] for i in test}),
                            fit_mean=mean.tolist(), fit_scale=scale.tolist()))
    errors = np.abs(pred[valid] - y[valid])
    person_error = [float(errors[np.array([ids[i] == p for i in np.flatnonzero(valid)])].mean())
                    for p in unique]
    return {**result, "status": "ok", "mae_image": float(errors.mean()),
            "mae_person": float(np.mean(person_error)),
            "mean_baseline_mae_image": float(np.abs(mean_pred[valid] - y[valid]).mean()),
            "median_baseline_mae_image": float(np.abs(median_pred[valid] - y[valid]).mean()),
            "row_indices": np.flatnonzero(valid).tolist(),
            "persons": [ids[i] for i in np.flatnonzero(valid)],
            "folds": row_fold[valid].tolist(), "predictions": pred[valid].tolist(),
            "absolute_errors": errors.tolist(), "fold_ledger": ledgers}


def paired_age_error(before, after, *, resamples=2000, seed=0):
    """Equal-person mean tuned-minus-frozen MAE with shared subject resampling.

    Probe fits are held fixed: this CI excludes probe fitting and training-seed
    population uncertainty. Positive delta means higher age prediction error, not proof
    of disentanglement; constant-baseline comparisons must also be examined.
    """
    for key in ("status", "row_indices", "persons", "folds"):
        if before.get(key) != after.get(key):
            raise ValueError("common probe coverage and person folds required")
    if before["status"] != "ok" or resamples < 1:
        raise ValueError("successful probes and positive resamples required")
    persons = np.asarray(before["persons"])
    delta = np.asarray(after["absolute_errors"]) - np.asarray(before["absolute_errors"])
    per_person = np.array([delta[persons == p].mean() for p in sorted(set(persons))])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(per_person), size=(resamples, len(per_person)))
    return dict(delta_mae_person=float(per_person.mean()),
                ci95=np.quantile(per_person[draws].mean(axis=1), [.025, .975]).tolist(),
                n_persons=len(per_person), resamples=resamples, seed=seed,
                limitation="conditional fixed OOF predictions; no probe refitting or seed-population CI")


def drift(before, after):
    """Descriptive image-aligned cosine drift and centered linear CKA."""
    x, y = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or len(x) < 2:
        raise ValueError("same-shaped aligned embedding matrices required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite embeddings required")
    norms = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    if np.any(norms == 0):
        raise ValueError("nonzero embeddings required")
    cosine_drift = 1 - np.sum(x * y, axis=1) / norms
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    denominator = np.linalg.norm(xc.T @ xc) * np.linalg.norm(yc.T @ yc)
    return dict(mean_cosine_drift=float(cosine_drift.mean()),
                linear_cka=float(np.linalg.norm(xc.T @ yc) ** 2 / denominator)
                if denominator > 0 else None, n_images=len(x),
                limitation="descriptive representation change; not a mechanism decision")
