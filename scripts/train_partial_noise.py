"""Fresh source-bound partial-noise training on shared-person restricted CROSS.

Permutation seed is fixed independently of the three training seeds. Clean
comparison is deliberately pending until its separate bound campaign completes.
No change to existing trainer/evaluator sources or public APIs is required.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.run_restricted_matched_campaign import require_fresh, verify_records
from scripts.validate_clean_noise_control import validate_control

SEEDS = (42, 1, 2)


def checked_manifest(path, experiment):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != experiment:
        raise ValueError("unexpected input manifest")
    verify_records(payload["inputs"] + payload["outputs"])
    return payload


def verify_training(models, noisy):
    manifests = sorted(models.glob("*.manifest.json"))
    if len(manifests) != 3:
        raise ValueError("three training manifests required")
    for seed in SEEDS:
        path = models / f"partial_noise_s{seed}.manifest.json"
        payload = checked_manifest(path, "pair-contrastive-backbone-finetune")
        p = payload["parameters"]
        expected = {"seed": seed, "epochs_requested": 10, "epochs_executed": 10,
                    "selected_epoch": 10, "checkpoint_selection": "last_epoch",
                    "batchnorm_policy": "adapt_all", "backbone": "facenet",
                    "trainable_scope": "head", "batch_size": 64, "learning_rate": 3e-5,
                    "margin": .3, "gap_weight": 0.0, "crops_dir": "faces"}
        if any(p.get(k) != v for k, v in expected.items()):
            raise ValueError("actual noise training protocol mismatch")
        if Path(p["pairs_file"]).resolve() != noisy.resolve() or file_record(noisy) not in payload["inputs"]:
            raise ValueError("noise training input linkage mismatch")
    return manifests


def worker(noisy, models, out):
    import numpy as np
    import torch

    from age_gap.evaluation.benchmark_external import pair_scores
    from age_gap.training.finetune import finetune, load_finetuned
    from scripts.train_matched_agegap_arms import _fgnet_metrics_from_scores, load_matched_fgnet

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_available():
        raise ValueError("CPU-only worker environment required")
    models.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        checkpoint = models / f"partial_noise_s{seed}.pt"
        if checkpoint.exists() or checkpoint.with_suffix(".manifest.json").exists():
            raise FileExistsError("noise worker refuses existing checkpoints")
        finetune(epochs=10, lr=3e-5, batch_size=64, margin=.3, patience=10,
                 trainable_scope="head", gap_weight=0.0, backbone_name="facenet",
                 crops_dir="faces", ckpt_out=checkpoint, seed=seed, pairs_file=str(noisy),
                 checkpoint_selection="last_epoch", batchnorm_policy="adapt_all")
        model = load_finetuned(checkpoint, "cpu")
        cache = PROJECT_ROOT / "data/external/fgnet_crops.npz"
        a, b, labels, large, metadata = load_matched_fgnet(cache, seed=42, endpoint_age_tolerance=2)
        scores = pair_scores(model, a, b, "cpu", rgb=False)
        if scores.shape != labels.shape or not np.isfinite(scores).all():
            raise ValueError("invalid noise FG-NET scores")
        private = out / "private" / f"fgnet_scores_s{seed}.npz"
        np.savez_compressed(private, scores=scores, labels=labels, large=large,
                            subject_a=metadata["subject_a"], subject_b=metadata["subject_b"])
        result = out / f"noise_s{seed}.json"
        metrics = {"training_seed": seed, "permutation_seed": 42,
                   "fgnet": _fgnet_metrics_from_scores(scores, labels, large, metadata),
                   "clean_comparison_completed": False, "publication_ready": False}
        result.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        write_experiment_manifest(result.with_suffix(".manifest.json"),
            experiment="partial-noise-fgnet-evaluation", parameters={"training_seed": seed,
                "permutation_seed": 42, "protocol": "endpoint_age_matched", "pair_seed": 42,
                "endpoint_age_tolerance": 2}, metrics=metrics,
            inputs=[noisy, checkpoint, cache], outputs=[result, private])
        print(f"noise seed {seed}: training/evaluation complete; clean comparison pending", flush=True)


def run(clean_dir, noise_dir, clean_preflight, models, out, *, execute=False, threads=2):
    from age_gap.training.finetune import _crop_path
    from scripts.build_partial_noise_control import build_control

    if type(threads) is not int or threads < 1:
        raise ValueError("positive CPU thread count required")
    clean_dir, noise_dir, clean_preflight, models, out = map(Path, (clean_dir, noise_dir, clean_preflight, models, out))
    models, out = models.resolve(), out.resolve()
    require_fresh([models, out])
    prepared = checked_manifest(noise_dir / "summary.manifest.json", "partial-exact-age-supervision-noise-control")
    if prepared["parameters"].get("seed") != 42:
        raise ValueError("fixed permutation seed42 required; training seeds vary independently")
    clean_manifest = checked_manifest(clean_dir / "summary.manifest.json", "restricted-positive-image-low-cross-arms")
    baseline_preflight = checked_manifest(clean_preflight, "restricted-matched-campaign-preflight")
    clean, noisy = clean_dir / "private/cross_arm.jsonl", noise_dir / "private/partial_noise_arm.jsonl"
    if (file_record(clean) not in prepared["inputs"] or file_record(noisy) not in prepared["outputs"]
            or file_record(clean) not in clean_manifest["outputs"]
            or file_record(clean) not in baseline_preflight["inputs"]):
        raise ValueError("clean/noise/preflight manifest linkage missing")
    clusters = PROJECT_ROOT / "data/processed/person_clusters.jsonl"
    groups = {}
    for row in read_jsonl(clusters):
        key, value = row["identity_group_id"], row["person_id"]
        if key in groups and groups[key] != value:
            raise ValueError("contradictory cluster mapping")
        groups[key] = value
    canonical = PROJECT_ROOT / "data/processed/pairs.jsonl"
    known = {tuple(sorted((r["face_a"], r["face_b"]))) for r in read_jsonl(canonical) if r.get("label") == 1}
    clean_rows, noisy_rows = list(read_jsonl(clean)), list(read_jsonl(noisy))
    audit = validate_control(clean_rows, noisy_rows, groups, known_genuine_edges=known)
    regenerated, _ = build_control(clean_rows, groups, seed=42,
                                   attempts=prepared["parameters"]["attempts"], known_positive_pairs=known)
    if regenerated != noisy_rows:
        raise ValueError("noise rows differ from reproducible intervention")
    decoded = baseline_preflight["parameters"]["crop_audit"]
    if decoded.get("all_decodable") is not True or decoded.get("missing_rows_dropped") != 0:
        raise ValueError("successful clean crop preflight required")
    crop_records = {r["path"]: r for r in baseline_preflight["inputs"] if r["path"].endswith(".jpg")}
    for face in {r[f"face_{s}"] for r in noisy_rows for s in ("a", "b")}:
        record = file_record(_crop_path(face))
        if crop_records.get(record["path"]) != record:
            raise ValueError("noise crop not covered by unchanged decoded preflight")
    inputs = [clean, noisy, noise_dir / "summary.manifest.json", clean_dir / "summary.manifest.json",
              clean_preflight, clusters, canonical, Path(__file__),
              PROJECT_ROOT / "scripts/validate_clean_noise_control.py",
              PROJECT_ROOT / "scripts/build_partial_noise_control.py"]
    records = [file_record(path) for path in inputs] + baseline_preflight["inputs"]
    verify_records(records)
    parameters = {"contract": audit, "training_seeds": list(SEEDS), "permutation_seed": 42,
                  "permutation_sensitivity_completed": False, "device": "cpu", "threads": threads,
                  "epochs": 10, "checkpoint_selection": "last_epoch", "batchnorm_policy": "adapt_all",
                  "clean_comparison_completed": False, "publication_ready": False}
    print(json.dumps(parameters), flush=True)
    if not execute:
        return parameters
    require_fresh([models, out])
    (out / "private").mkdir(parents=True)
    input_paths = [Path(r["path"]) if Path(r["path"]).is_absolute() else PROJECT_ROOT / r["path"] for r in records]
    contract = out / "contract.json"
    contract.write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    preflight = write_experiment_manifest(out / "preflight.manifest.json", experiment="partial-noise-training-preflight",
        parameters=parameters, metrics=audit, inputs=input_paths, outputs=[contract])
    verify_records(records)
    preflight_record = file_record(preflight)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    command = [sys.executable, "-m", "scripts.train_partial_noise", "--worker", "--noise", str(noise_dir),
               "--models", str(models), "--out", str(out)]
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    if result.returncode:
        raise RuntimeError("noise worker failed; partial artifacts retained, no completed binding")
    verify_records(records + [preflight_record])
    manifests = verify_training(models, noisy)
    evaluations = [out / f"noise_s{seed}.manifest.json" for seed in SEEDS]
    for path in evaluations:
        checked_manifest(path, "partial-noise-fgnet-evaluation")
    target = out / "training-bound.manifest.json"
    outputs = [*manifests, *models.glob("*.pt"), *evaluations,
               *[out / f"noise_s{seed}.json" for seed in SEEDS], *out.joinpath("private").glob("*.npz")]
    write_experiment_manifest(target, experiment="partial-noise-source-crop-bound-training",
        parameters=parameters, metrics={"noise_training_completed": True,
            "clean_comparison_completed": False, "publication_ready": False},
        inputs=[*input_paths, preflight], outputs=outputs)
    try:
        verify_records(records + [preflight_record])
    except Exception:
        target.unlink()
        raise
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", type=Path, default=PROJECT_ROOT / "data/interim/restricted_matched_arms_20261003")
    p.add_argument("--noise", type=Path, default=PROJECT_ROOT / "data/interim/partial_noise_restricted_20261003/seed42")
    p.add_argument("--clean-preflight", type=Path, default=PROJECT_ROOT / "metrics/restricted_matched_fixed10_20261003/preflight.manifest.json")
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args.noise / "private/partial_noise_arm.jsonl", args.models, args.out)
    else:
        run(args.clean, args.noise, args.clean_preflight, args.models, args.out,
            execute=args.execute, threads=args.threads)


if __name__ == "__main__":
    main()
