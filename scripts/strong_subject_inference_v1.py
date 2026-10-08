"""ROC-v2 subject inference for actual strong checkpoints, including one seed."""

import numpy as np

from scripts.benchmark_metrics_v2 import METRICS, metric_vector
from scripts.verification_metrics_v2 import VERSION


def infer(scores, labels, subject_a, subject_b, *, n_boot=2000, seed=0):
    if "frozen" not in scores or len(scores) < 2:
        raise ValueError("frozen and at least one actual tuned checkpoint required")
    if type(n_boot) is not int or n_boot < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive bootstrap count and nonnegative seed required")
    y, a, b = np.asarray(labels), np.asarray(subject_a), np.asarray(subject_b)
    if (
        y.ndim != 1
        or y.dtype.kind not in "iu"
        or set(np.unique(y)) != {0, 1}
        or a.shape != y.shape
        or b.shape != y.shape
    ):
        raise ValueError("aligned binary labels and subject endpoints required")
    for ids in (a, b):
        if (
            ids.dtype.kind not in "iuUS"
            or any(str(v).strip().lower() in {"", "none", "nan", "unknown"} for v in ids)
            or (ids.dtype.kind in "iu" and (ids < 0).any())
        ):
            raise ValueError("known nonnegative subject IDs required")
    a, b = a.astype(str), b.astype(str)
    if (a[y == 1] != b[y == 1]).any() or (a[y == 0] == b[y == 0]).any():
        raise ValueError("subject metadata contradicts labels")
    keys = ["frozen", *sorted(key for key in scores if key != "frozen")]
    matrix = np.stack([np.asarray(scores[key], float) for key in keys])
    if matrix.shape != (len(keys), len(y)) or not np.isfinite(matrix).all():
        raise ValueError("finite aligned scores required")

    def values(mask=None, weights=None):
        mask = np.ones(len(y), bool) if mask is None else mask
        return np.stack(
            [
                metric_vector(row[mask], y[mask], None if weights is None else weights[mask])
                for row in matrix
            ]
        )

    points = values()
    people, inverse = np.unique(np.r_[a, b], return_inverse=True)
    aa, bb = inverse[: len(y)], inverse[len(y) :]
    rng, draws = np.random.default_rng(seed), []
    for _ in range(n_boot):
        multiplicity = np.bincount(
            rng.integers(len(people), size=len(people)), minlength=len(people)
        )
        weights = multiplicity[aa] * np.where(y == 1, 1, multiplicity[bb])
        if any(weights[y == label].sum() == 0 for label in (0, 1)):
            continue
        draws.append(values(weights=weights))
    if not draws:
        raise ValueError("no valid subject resamples")
    draws = np.stack(draws)
    loo = []
    for person in people:
        mask = (a != person) & (b != person)
        if set(np.unique(y[mask])) == {0, 1}:
            v = values(mask)
            loo.append(v[1:] - v[0])
    models = {}
    for i, key in enumerate(keys):
        models[key] = {
            name: dict(
                point=float(points[i, j]),
                ci95=np.percentile(draws[:, i, j], [2.5, 97.5]).tolist(),
                delta_vs_frozen=float(points[i, j] - points[0, j]),
                delta_ci95=np.percentile(draws[:, i, j] - draws[:, 0, j], [2.5, 97.5]).tolist(),
                loo_delta_min=(float(np.asarray(loo)[:, i - 1, j].min()) if loo and i else None),
                loo_delta_max=(float(np.asarray(loo)[:, i - 1, j].max()) if loo and i else None),
            )
            for j, name in enumerate(METRICS)
        }
    return dict(
        metric_version=VERSION,
        n_pairs=len(y),
        n_subjects=len(people),
        actual_tuned_checkpoint_count=len(keys) - 1,
        models=models,
        loo_valid=len(loo),
        loo_unavailable=len(people) - len(loo),
        bootstrap=dict(
            requested=n_boot,
            valid=len(draws),
            seed=seed,
            shared_draws_across_models=True,
            positive_weight="owner multiplicity once",
            negative_weight="endpoint multiplicity product",
            thresholds_reselected=True,
        ),
        scope="individual fixed-checkpoint subject intervals; no fabricated three-seed aggregate, no training-seed population inference; test ROC not deployment calibration; not multiplicity corrected",
        training_identity_independence="unverified",
        publication_ready=False,
    )
