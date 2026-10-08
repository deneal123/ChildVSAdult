"""Post-campaign paired ROC-v2 inference; no scoring of incomplete training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.benchmark_metrics_v2 import METRICS, metric_vector
from scripts.verification_metrics_v2 import VERSION

SEEDS = (42, 1, 2)


def paired_infer(low, cross, labels, subject_a, subject_b, *, frozen=None, n_boot=2000, seed=0):
    low, cross = np.asarray(low, float), np.asarray(cross, float)
    y, a, b = np.asarray(labels), np.asarray(subject_a), np.asarray(subject_b)
    if (
        y.ndim != 1
        or low.shape != (3, len(y))
        or cross.shape != low.shape
        or a.shape != y.shape
        or b.shape != y.shape
    ):
        raise ValueError("three aligned score rows per arm and subject endpoints required")
    if y.dtype.kind not in "iu" or set(np.unique(y)) != {0, 1}:
        raise ValueError("genuine integer binary labels required")
    if any(
        ids.dtype.kind not in "iuUS"
        or any(str(v).strip().lower() in {"", "none", "nan", "unknown"} for v in ids)
        for ids in (a, b)
    ):
        raise ValueError("known subject metadata required")
    if any(ids.dtype.kind in "iu" and (ids < 0).any() for ids in (a, b)):
        raise ValueError("nonnegative subject identifiers required")
    a, b = a.astype(str), b.astype(str)
    if np.any((y == 1) & (a != b)) or np.any((y == 0) & (a == b)):
        raise ValueError("subject metadata contradicts labels")
    if type(n_boot) is not int or n_boot < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive resample count and nonnegative seed required")
    scores = np.concatenate((low, cross))
    if frozen is not None:
        frozen = np.asarray(frozen, float)
        if frozen.shape != y.shape:
            raise ValueError("aligned frozen scores required")
        scores = np.concatenate((scores, frozen[None]))
    if not np.isfinite(scores).all():
        raise ValueError("finite scores required")

    def values(weights=None, mask=None):
        mask = np.ones(len(y), bool) if mask is None else mask
        return np.stack(
            [
                metric_vector(row[mask], y[mask], None if weights is None else weights[mask])
                for row in scores
            ]
        )

    points = values()
    people, inverse = np.unique(np.r_[a, b], return_inverse=True)
    aa, bb = inverse[: len(y)], inverse[len(y) :]
    rng, draws = np.random.default_rng(seed), []
    for _ in range(n_boot):
        m = np.bincount(rng.integers(len(people), size=len(people)), minlength=len(people))
        weights = m[aa] * np.where(y == 1, 1, m[bb])
        if any(weights[y == label].sum() <= 0 for label in (0, 1)):
            continue
        draws.append(values(weights))
    if not draws:
        raise ValueError("no valid subject resamples")
    draws = np.asarray(draws)
    per_seed_delta = points[3:6] - points[:3]
    delta_draws = draws[:, 3:6] - draws[:, :3]
    loo = []
    for person in people:
        mask = (a != person) & (b != person)
        if set(np.unique(y[mask])) == {0, 1}:
            point = values(mask=mask)
            loo.append((point[3:6] - point[:3]).mean(axis=0))
    loo = np.asarray(loo)
    report = {}
    for j, metric in enumerate(METRICS):
        entry = {}
        for arm, start in (("low", 0), ("cross", 3)):
            arm_points = points[start : start + 3, j]
            arm_draws = draws[:, start : start + 3, j]
            entry[arm] = dict(
                mean=float(arm_points.mean()),
                sd=float(arm_points.std(ddof=1)),
                mean_checkpoint_ci95=np.percentile(arm_draws.mean(axis=1), [2.5, 97.5]).tolist(),
                per_seed={
                    str(s): {
                        "point": float(arm_points[i]),
                        "ci95": np.percentile(arm_draws[:, i], [2.5, 97.5]).tolist(),
                    }
                    for i, s in enumerate(SEEDS)
                },
            )
            if frozen is not None:
                entry[arm]["mean_delta_vs_frozen"] = float(arm_points.mean() - points[6, j])
                entry[arm]["delta_vs_frozen_ci95"] = np.percentile(
                    arm_draws.mean(axis=1) - draws[:, 6, j], [2.5, 97.5]
                ).tolist()
        entry["cross_minus_low"] = dict(
            mean=float(per_seed_delta[:, j].mean()),
            sd=float(per_seed_delta[:, j].std(ddof=1)),
            mean_checkpoint_ci95=np.percentile(
                delta_draws[:, :, j].mean(axis=1), [2.5, 97.5]
            ).tolist(),
            per_seed={
                str(s): {
                    "point": float(per_seed_delta[i, j]),
                    "ci95": np.percentile(delta_draws[:, i, j], [2.5, 97.5]).tolist(),
                }
                for i, s in enumerate(SEEDS)
            },
            loo_min=float(loo[:, j].min()) if len(loo) else None,
            loo_max=float(loo[:, j].max()) if len(loo) else None,
        )
        if frozen is not None:
            entry["frozen"] = dict(
                point=float(points[6, j]), ci95=np.percentile(draws[:, 6, j], [2.5, 97.5]).tolist()
            )
        report[metric] = entry
    return dict(
        metrics=report,
        metric_version=VERSION,
        training_seeds=list(SEEDS),
        n_pairs=len(y),
        n_subjects=len(people),
        loo_valid=len(loo),
        loo_unavailable=len(people) - len(loo),
        bootstrap=dict(
            seed=seed,
            requested=n_boot,
            valid=len(draws),
            shared_draws_across_all_models=True,
            positive_weight="owner multiplicity once",
            negative_weight="endpoint multiplicity product",
            thresholds_reselected=True,
            conditioning="fixed checkpoints and protocol; not training-seed population or score ensemble",
        ),
        scope="raw CROSS minus LOW; smaller EER is better; test-derived ROC not deployment calibration; intervals not multiplicity corrected",
        training_identity_independence="unverified",
        causal_source_evidence=False,
        publication_ready=False,
    )


def readiness(campaign, models):
    from scripts.run_oriented_campaign import PROTOCOL, training_contract, verified

    binding = campaign / "training-bound.manifest.json"
    native = verified(binding, "oriented-uniform-serial-training")
    if (
        native["metrics"].get("training_complete") is not True
        or native["metrics"].get("completed_cells") != 6
    ):
        raise ValueError("six completed source-bound training cells required")
    if native.get("parameters") != PROTOCOL:
        raise ValueError("completed campaign protocol mismatch")
    selected = {}
    for arm in ("low", "cross"):
        pair = (
            PROJECT_ROOT
            / f"metrics/oriented_exposure_matching_20261003/private/{arm}_candidate_arm.jsonl"
        )
        for seed in SEEDS:
            checkpoint = models / f"{arm}_s{seed}.pt"
            train_manifest = checkpoint.with_suffix(".manifest.json")
            cell_path = campaign / f"{arm}_s{seed}.manifest.json"
            if any(
                file_record(p) not in native["outputs"]
                for p in (checkpoint, train_manifest, cell_path)
            ):
                raise ValueError("cell/model not bound by completed campaign")
            actual = verified(train_manifest, "pair-contrastive-backbone-finetune")
            training_contract(actual, pair, seed)
            if file_record(checkpoint) not in actual["outputs"]:
                raise ValueError("actual training manifest does not bind selected checkpoint")
            cell = verified(cell_path, "oriented-source-crop-bound-training-cell")
            if (
                cell["metrics"].get("training_complete") is not True
                or cell.get("parameters") != PROTOCOL | {"seed": seed, "arm": arm}
                or cell["inputs"] != native["inputs"]
                or any(record not in cell["inputs"] for record in actual["inputs"])
                or file_record(checkpoint) not in cell["outputs"]
                or file_record(train_manifest) not in cell["outputs"]
            ):
                raise ValueError("cell linkage mismatch")
            selected[f"{arm}_s{seed}"] = checkpoint
    return native, selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    native, checkpoints = readiness(args.campaign, args.models)
    if not args.execute:
        print("Six verified cells ready; no inference performed")
        return
    from age_gap.models.backbones import make_backbone
    from age_gap.training.finetune import load_finetuned
    from scripts.fgnet_retrieval_study import embed_indices
    from scripts.reevaluate_fgnet_metrics_v2 import protocol
    from scripts.run_oriented_campaign import cpu_setup, paths_from

    cpu_setup()
    cache = PROJECT_ROOT / "data/external/fgnet_crops.npz"
    deps = [
        PROJECT_ROOT / f"scripts/{n}.py"
        for n in (
            "run_oriented_campaign",
            "benchmark_metrics_v2",
            "verification_metrics_v2",
            "reevaluate_fgnet_metrics_v2",
            "fgnet_retrieval_study",
        )
    ]
    inputs = list(
        dict.fromkeys(
            p.resolve()
            for p in [
                *paths_from(native),
                args.campaign / "training-bound.manifest.json",
                cache,
                Path(__file__),
                *deps,
            ]
        )
    )
    before = [file_record(p) for p in inputs]
    with np.load(cache, allow_pickle=False) as data:
        crops, subjects, ages = data["crops"], data["subjects"], data["ages"]
    if crops.shape != (len(subjects), 112, 112, 3) or crops.dtype != np.uint8:
        raise ValueError("full aligned BGR crop cache required")
    left, right, y, gaps, coverage = protocol(subjects, ages)
    scores, embeddings = {}, {}
    for key in ("frozen", *checkpoints):
        model = (
            make_backbone("facenet", pretrained=True).cpu().eval()
            if key == "frozen"
            else load_finetuned(checkpoints[key], "cpu").eval()
        )
        vectors = embed_indices(crops, np.arange(len(crops)), model, batch_size=32, threads=1)
        if (
            vectors.shape != (len(crops), 512)
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        ):
            raise ValueError("full finite unit embeddings required")
        scores[key] = np.sum(vectors[left] * vectors[right], axis=1)
        embeddings[key] = vectors
        del model
        print(f"embedded full crop pool: {key}", flush=True)
    low, cross = (np.stack([scores[f"{a}_s{s}"] for s in SEEDS]) for a in ("low", "cross"))
    result = {}
    for name, mask in (("overall", np.ones(len(y), bool)), ("large_gap_25plus", gaps >= 25)):
        result[name] = paired_infer(
            low[:, mask],
            cross[:, mask],
            y[mask],
            subjects[left][mask],
            subjects[right][mask],
            frozen=scores["frozen"][mask],
        )
    result.update(
        coverage=coverage,
        protocol="full cached FG-NET endpoint_age_matched seed42/tolerance2; source-gap strata",
        strata_are_not_independent=True,
        execution_complete=True,
        publication_ready=False,
        scope="uniform oriented within-source age-gap control; cached preprocessing not replayed; no causal source claim",
    )
    if before != [file_record(p) for p in inputs]:
        raise ValueError("inputs changed during scoring/inference")
    (args.out / "private").mkdir(parents=True)
    outputs = []
    for key, vectors in embeddings.items():
        path = args.out / f"private/embeddings_{key}.npz"
        np.savez_compressed(path, indices=np.arange(len(crops)), embeddings=vectors)
        outputs.append(path)
    stored = args.out / "private/scores.npz"
    np.savez_compressed(
        stored,
        **scores,
        labels=y,
        source_gap=gaps,
        subject_a=subjects[left],
        subject_b=subjects[right],
        left=left,
        right=right,
    )
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = args.out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="oriented-controls-full-fgnet-roc-v2",
        parameters={"n_boot": 2000, "bootstrap_seed": 0, "protocol_seed": 42, "tolerance": 2},
        metrics=result,
        inputs=inputs,
        outputs=[summary, stored, *outputs],
    )
    if before != json.loads(target.read_text(encoding="utf-8"))["inputs"]:
        target.unlink()
        raise ValueError("inputs changed during manifest write")


if __name__ == "__main__":
    main()
