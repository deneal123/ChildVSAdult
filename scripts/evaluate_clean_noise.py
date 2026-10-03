"""Paired fixed-checkpoint clean/noise inference; never ensemble-score AUC.

Completed source/crop-bound campaigns are mandatory. Training-seed uncertainty
and longitudinal-source causality are not inferred from subject resampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.run_restricted_matched_campaign import require_fresh, verify_records

SEEDS = (42, 1, 2)


def paired_effect(clean, noise, labels, subject_a, subject_b, *, n_boot=2000, seed=0):
    clean, noise = np.asarray(clean, float), np.asarray(noise, float)
    labels, a, b = np.asarray(labels), np.asarray(subject_a), np.asarray(subject_b)
    if (labels.ndim != 1 or clean.shape != (3, len(labels)) or noise.shape != clean.shape
            or a.shape != labels.shape or b.shape != labels.shape):
        raise ValueError("three aligned checkpoint score rows and endpoint metadata required")
    if labels.dtype.kind not in "iu" or set(np.unique(labels)) != {0, 1}:
        raise ValueError("genuine integer binary benchmark labels required")
    if not np.isfinite(clean).all() or not np.isfinite(noise).all():
        raise ValueError("finite paired scores required")
    for ids in (a, b):
        if ids.dtype.kind not in "iuUS" or any(str(v).strip().lower() in {"", "none", "nan", "unknown"} for v in ids):
            raise ValueError("nonmissing benchmark subject metadata required")
        if ids.dtype.kind in "iu" and (ids < 0).any():
            raise ValueError("nonnegative subject identifiers required")
    a, b = a.astype(str), b.astype(str)
    if np.any((labels == 1) & (a != b)) or np.any((labels == 0) & (a == b)):
        raise ValueError("subject endpoints contradict genuine benchmark labels")
    if type(n_boot) is not int or n_boot < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive resample budget and nonnegative bootstrap seed required")
    subjects, inverse = np.unique(np.concatenate((a, b)), return_inverse=True)
    aa, bb = inverse[:len(labels)], inverse[len(labels):]
    def deltas(weights=None, mask=None):
        mask = np.ones(len(labels), bool) if mask is None else mask
        weights = None if weights is None else weights[mask]
        return np.array([roc_auc_score(labels[mask], nr[mask], sample_weight=weights)
                         - roc_auc_score(labels[mask], cr[mask], sample_weight=weights)
                         for cr, nr in zip(clean, noise, strict=True)])
    point = deltas()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sample = rng.integers(len(subjects), size=len(subjects))
        m = np.bincount(sample, minlength=len(subjects))
        weights = m[aa] * np.where(labels == 1, 1, m[bb])
        if any(not np.any(weights[labels == y]) for y in (0, 1)):
            continue
        draws.append(deltas(weights))
    if not draws:
        raise ValueError("no valid paired subject draws")
    draws = np.asarray(draws)
    mean_draws = draws.mean(axis=1)
    loo = []
    for identity in subjects:
        mask = (a != identity) & (b != identity)
        if set(np.unique(labels[mask])) == {0, 1}:
            loo.append(float(deltas(mask=mask).mean()))
    return {"direction": "noise_minus_clean", "training_seeds": list(SEEDS),
            "permutation_seed": 42, "n_pairs": len(labels), "n_subjects": len(subjects),
            "per_seed": {str(s): {"clean_auc": float(roc_auc_score(labels, clean[i])),
                                  "noise_auc": float(roc_auc_score(labels, noise[i])),
                                  "delta_auc": float(point[i]),
                                  "ci95": np.percentile(draws[:, i], [2.5, 97.5]).tolist()}
                         for i, s in enumerate(SEEDS)},
            "mean_checkpoint_delta_auc": float(point.mean()),
            "delta_sd_across_three_training_seeds": float(point.std(ddof=1)),
            "mean_checkpoint_ci95": np.percentile(mean_draws, [2.5, 97.5]).tolist(),
            "bootstrap": {"method": "paired subject-cluster percentile; positive owner once, negative dyad product",
                          "seed": seed, "n_requested": n_boot, "n_valid": len(draws),
                          "shared_draws_across_all_checkpoints": True,
                          "subject_ordering": "lexicographic canonical subject strings",
                          "estimand": "mean of three noise-minus-clean AUC deltas, not AUC of mean scores",
                          "conditioning": "fixed checkpoints, permutation42 and benchmark protocol; not training-seed population"},
            "leave_one_subject_out": {"valid": len(loo), "unavailable": len(subjects)-len(loo),
                                      "mean_delta_min": min(loo) if loo else None,
                                      "mean_delta_max": max(loo) if loo else None,
                                      "scope": "sensitivity extrema, not a confidence interval"},
            "causal_source_evidence": False, "publication_ready": False}


def load_binding(path, experiment):
    if not path.is_file():
        raise FileNotFoundError("completed source/crop-bound campaign is not available")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != experiment:
        raise ValueError("unexpected bound campaign experiment")
    verify_records(payload["inputs"] + payload["outputs"])
    return payload


def compare_common_inputs(clean, noise):
    def keyed(records):
        result = {}
        for record in records:
            if record["path"] in result and result[record["path"]] != record:
                raise ValueError("conflicting declared source records")
            result[record["path"]] = record
        return result
    c, n = keyed(clean["inputs"]), keyed(noise["inputs"])
    # Noise training inherits the verified clean preflight plus its own sources.
    for path, record in c.items():
        if path in n and n[path] != record:
            raise ValueError("clean/noise common input bytes differ")
    required = ("src/age_gap/training/finetune.py", "src/age_gap/models/facenet.py",
                "scripts/train_matched_agegap_arms.py", "data/external/fgnet_crops.npz",
                "metrics/model_inventory.json", "data/interim/restricted_matched_arms_20261003/private/cross_arm.jsonl")
    for path in required:
        if path not in c or path not in n or c[path] != n[path]:
            raise ValueError("common scientific source/cache/clean-arm binding missing")
    for path, record in c.items():
        if (path.endswith((".jpg", ".py", ".toml"))
                or path.endswith("20180408-102900-casia-webface.pt")) and n.get(path) != record:
            raise ValueError("shared crop/source/config/base weight binding missing")
    if not any(p.endswith("20180408-102900-casia-webface.pt") for p in c) or not any(p.endswith(".jpg") for p in c):
        raise ValueError("actual base weight and crop records required")


def run(clean_dir, noise_dir, clean_models, noise_models, out, *, execute=False, n_boot=2000, threads=2):
    clean_dir, noise_dir, clean_models, noise_models, out = map(Path, (clean_dir, noise_dir, clean_models, noise_models, out))
    if type(threads) is not int or threads < 1:
        raise ValueError("positive CPU thread count required")
    if type(n_boot) is not int or n_boot < 1:
        raise ValueError("positive subject resample budget required")
    require_fresh([out])
    clean_path, noise_path = clean_dir / "campaign-bound.manifest.json", noise_dir / "training-bound.manifest.json"
    clean = load_binding(clean_path, "restricted-matched-source-crop-bound-campaign")
    noise = load_binding(noise_path, "partial-noise-source-crop-bound-training")
    if clean["metrics"].get("training_completed") is not True or noise["metrics"].get("noise_training_completed") is not True:
        raise ValueError("completed clean and noise training required")
    if noise["parameters"].get("permutation_seed") != 42 or noise["parameters"].get("training_seeds") != list(SEEDS):
        raise ValueError("fixed permutation42 and three aligned training seeds required")
    compare_common_inputs(clean, noise)
    summary_path = clean_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    from scripts.run_restricted_matched_campaign import validate_completed
    validate_completed(summary)
    selected = {r["seed"]: r for r in summary["runs"] if r["arm"] == "cross"}
    checkpoints = [clean_models / f"{selected[s]['run_id']}.pt" for s in SEEDS]
    noisy_checkpoints = [noise_models / f"partial_noise_s{s}.pt" for s in SEEDS]
    for paths, binding in ((checkpoints, clean), (noisy_checkpoints, noise)):
        if any(file_record(path) not in binding["outputs"] for path in paths):
            raise ValueError("checkpoint not an output of completed bound training")
    inputs = [clean_path, noise_path, summary_path, *checkpoints, *noisy_checkpoints,
              Path(__file__), PROJECT_ROOT / "scripts/run_restricted_matched_campaign.py",
              PROJECT_ROOT / "scripts/train_matched_agegap_arms.py",
              PROJECT_ROOT / "data/external/fgnet_crops.npz"]
    before = [file_record(path) for path in inputs]
    if not execute:
        print(json.dumps({"status": "ready_for_evaluation", "training_seeds": list(SEEDS),
                          "permutation_seed": 42, "publication_ready": False}))
        return
    import torch

    from age_gap.evaluation.benchmark_external import pair_scores
    from age_gap.training.finetune import load_finetuned
    from scripts.train_matched_agegap_arms import load_matched_fgnet

    torch.set_num_threads(threads)
    a, b, labels, large, metadata = load_matched_fgnet(inputs[-1], seed=42, endpoint_age_tolerance=2)
    clean_scores, noise_scores = [], []
    for target, paths in ((clean_scores, checkpoints), (noise_scores, noisy_checkpoints)):
        for path in paths:
            model = load_finetuned(path, "cpu")
            target.append(pair_scores(model, a, b, "cpu", rgb=False))
            del model
            print(f"scored checkpoint {len(clean_scores)+len(noise_scores)}/6", flush=True)
    clean_scores, noise_scores = np.asarray(clean_scores), np.asarray(noise_scores)
    sa, sb = np.asarray(metadata["subject_a"]), np.asarray(metadata["subject_b"])
    result = {"overall": paired_effect(clean_scores, noise_scores, labels, sa, sb, n_boot=n_boot),
              "large_gap_25_plus": paired_effect(clean_scores[:, large], noise_scores[:, large],
                  labels[large], sa[large], sb[large], n_boot=n_boot),
              "training_identity_independence": "unverified", "publication_ready": False,
              "strata_share_fgnet_corpus_and_are_not_independent": True,
              "scope": "shared-source supervision-noise sensitivity; not generic/co-occurrence/causal source comparison"}
    verify_records(before + clean["inputs"] + clean["outputs"] + noise["inputs"] + noise["outputs"])
    (out / "private").mkdir(parents=True)
    private = out / "private/scores.npz"
    np.savez_compressed(private, clean_scores=clean_scores, noise_scores=noise_scores,
                        labels=labels, large=large, subject_a=sa, subject_b=sb)
    target = out / "summary.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest = write_experiment_manifest(out / "summary.manifest.json", experiment="clean-noise-paired-subject-comparison",
        parameters={"n_boot": n_boot, "bootstrap_seed": 0, "training_seeds": list(SEEDS),
                    "permutation_seed": 42, "threads": threads, "device": "cpu"},
        metrics=result, inputs=inputs, outputs=[target, private])
    try:
        verify_records(before + clean["inputs"] + clean["outputs"] + noise["inputs"] + noise["outputs"])
    except Exception:
        manifest.unlink()
        raise
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", type=Path, default=PROJECT_ROOT / "metrics/restricted_matched_fixed10_20261003")
    p.add_argument("--noise", type=Path, default=PROJECT_ROOT / "metrics/partial_noise_restricted_fixed10_20261003")
    p.add_argument("--clean-models", type=Path, default=PROJECT_ROOT / "models/restricted_matched_fixed10_20261003")
    p.add_argument("--noise-models", type=Path, default=PROJECT_ROOT / "models/partial_noise_restricted_fixed10_20261003")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    available = ((args.clean / "campaign-bound.manifest.json").is_file(),
                 (args.noise / "training-bound.manifest.json").is_file())
    if not all(available):
        print(json.dumps({"status": "pending_completed_bound_campaigns",
                          "missing_clean_binding": not available[0], "missing_noise_binding": not available[1],
                          "comparison_completed": False, "publication_ready": False}))
        if args.execute:
            raise SystemExit(2)
        return
    run(args.clean, args.noise, args.clean_models, args.noise_models, args.out,
        execute=args.execute, n_boot=args.n_boot, threads=args.threads)


if __name__ == "__main__":
    main()
