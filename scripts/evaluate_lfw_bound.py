"""Evaluate source-bound LFW: official folds and paired both-endpoint subject CIs.

Run as ``python -m scripts.evaluate_lfw_bound --execute`` AFTER the new cache
producer succeeds. Legacy caches/results are never overwritten or silently reused.
All image linkage, embeddings and scores remain under private/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, sha256_file, write_experiment_manifest
from scripts.build_lfw_bound_cache import verify_binding
from scripts.export_public_evidence import scan_unsafe
from scripts.render_fgnet_evidence import verified_result

THRESHOLDS = np.linspace(-1.0, 1.0, 4001)
THRESHOLDS.setflags(write=False)
THRESHOLD_SHA256 = hashlib.sha256(THRESHOLDS.astype("<f8").tobytes()).hexdigest()
SEEDS = (42, 1, 2)
ROC_KEYS = ("roc_auc", "eer", "tar@far=0.01", "tar@far=0.001")
METRICS = (*ROC_KEYS, "accuracy_official_folds")


def validate_vectors(scores, labels, folds, weights=None):
    s, y, f = np.asarray(scores, dtype=float), np.asarray(labels), np.asarray(folds)
    if s.ndim != 1 or not len(s) or y.shape != s.shape or f.shape != s.shape:
        raise ValueError("equal-length score/label/fold vectors required")
    if not np.isfinite(s).all() or np.any(np.abs(s) > 1 + 1e-6):
        raise ValueError("finite cosine scores in [-1,1] required")
    if set(np.unique(y)) != {0, 1} or not np.issubdtype(f.dtype, np.integer):
        raise ValueError("binary labels and integer fold IDs required")
    if len(np.unique(f)) < 2:
        raise ValueError("at least two official folds required")
    if not np.array_equal(np.unique(f), np.arange(len(np.unique(f)))):
        raise ValueError("consecutive nonnegative official fold IDs required")
    w = np.ones(len(s)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != s.shape or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("nonnegative finite equal-length weights required")
    return np.clip(s, -1, 1), y.astype(int), f, w


def official_fold_accuracy(scores, labels, folds, weights=None) -> dict:
    """Train-only fixed-grid thresholds; no test-score-dependent grid/extrema.

    Weighted histograms exactly reproduce accept iff cosine >= threshold.
    Equal fold averaging follows the official accuracy estimand, including in bootstrap.
    """
    s, y, f, w = validate_vectors(scores, labels, folds, weights)
    fold_values = np.unique(f)
    bins = np.searchsorted(THRESHOLDS, s, side="right") - 1
    pos, neg = [], []
    for fold in fold_values:
        pos.append(np.bincount(bins[(f == fold) & (y == 1)],
                               weights=w[(f == fold) & (y == 1)], minlength=len(THRESHOLDS)))
        neg.append(np.bincount(bins[(f == fold) & (y == 0)],
                               weights=w[(f == fold) & (y == 0)], minlength=len(THRESHOLDS)))
    pos, neg = np.asarray(pos), np.asarray(neg)
    total_pos, total_neg = pos.sum(axis=0), neg.sum(axis=0)
    reports = []
    for i, fold in enumerate(fold_values):
        train_pos, train_neg = total_pos - pos[i], total_neg - neg[i]
        if min(train_pos.sum(), train_neg.sum(), pos[i].sum(), neg[i].sum()) <= 0:
            raise ValueError("each training/test fold must retain both weighted classes")
        train_correct = np.cumsum(train_pos[::-1])[::-1] + np.r_[0, np.cumsum(train_neg)[:-1]]
        threshold_index = int(np.argmax(train_correct))  # declared lowest-threshold tie break
        test_correct = pos[i, threshold_index:].sum() + neg[i, :threshold_index].sum()
        reports.append({"fold": int(fold), "threshold": float(THRESHOLDS[threshold_index]),
                        "accuracy": float(test_correct / (pos[i].sum() + neg[i].sum()))})
    return {"accuracy": float(np.mean([row["accuracy"] for row in reports])), "folds": reports}


def roc_points(scores, labels, weights=None) -> dict:
    """Tie-safe empirical ROC points, NOT development-calibrated deployment operating points."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels)
    if s.shape != y.shape or s.ndim != 1 or not np.isfinite(s).all() or np.any(np.abs(s) > 1 + 1e-6):
        raise ValueError("invalid ROC vectors")
    w = np.ones(len(s)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != s.shape or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("invalid ROC weights")
    if set(np.unique(y)) != {0, 1} or min(w[y == 0].sum(), w[y == 1].sum()) <= 0:
        raise ValueError("both weighted ROC classes required")
    fpr, tpr, _ = roc_curve(y, s, sample_weight=w, drop_intermediate=False)
    return {"roc_auc": float(auc(fpr, tpr)), "eer": float(np.min(np.maximum(fpr, 1 - tpr))),
            **{f"tar@far={far:g}": float(np.max(tpr[fpr <= far + 1e-12])) for far in (0.01, 0.001)}}


def subject_weights(labels, multiplicity, index_a, index_b):
    return multiplicity[index_a] * np.where(np.asarray(labels) == 1, 1, multiplicity[index_b])


def paired_statistics(score_map: dict[str, np.ndarray], labels, folds, subject_a, subject_b,
                      *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Shared subject draws for all models; accuracy reselects fold thresholds in EACH draw."""
    if n_boot < 1 or "frozen" not in score_map or len(score_map) < 2:
        raise ValueError("positive bootstrap count, frozen and tuned models required")
    y, f = np.asarray(labels), np.asarray(folds)
    a, b = np.asarray(subject_a), np.asarray(subject_b)
    if a.shape != y.shape or b.shape != y.shape:
        raise ValueError("both person arrays must match labels")
    if np.any((y == 1) & (a != b)) or np.any((y == 0) & (a == b)):
        raise ValueError("person metadata contradicts pair classes")
    if any(not isinstance(person, (str, np.str_)) or not person.strip() for person in np.r_[a, b]):
        raise ValueError("missing person metadata")
    subjects, inverse = np.unique(np.r_[a, b], return_inverse=True)
    if len(subjects) < 2:
        raise ValueError("at least two persons required")
    ia, ib = inverse[:len(y)], inverse[len(y):]
    fold_persons = {fold: set(a[f == fold]) | set(b[f == fold]) for fold in np.unique(f)}
    shared_person_counts = {int(fold): len(people & set().union(
        *(other for key, other in fold_persons.items() if key != fold)
    )) for fold, people in fold_persons.items()}
    points, fold_reports = {}, {}
    for key, scores in score_map.items():
        validate_vectors(scores, y, f)
        report = official_fold_accuracy(scores, y, f)
        points[key] = {**roc_points(scores, y), "accuracy_official_folds": report["accuracy"]}
        fold_reports[key] = report["folds"]
    draws = {key: {metric: [] for metric in METRICS} for key in score_map}
    rng = np.random.default_rng(seed)
    valid_roc = valid_accuracy = 0
    for _ in range(n_boot):
        multiplicity = np.bincount(rng.integers(len(subjects), size=len(subjects)), minlength=len(subjects))
        weights = subject_weights(y, multiplicity, ia, ib)
        if min(weights[y == 0].sum(), weights[y == 1].sum()) <= 0:
            continue
        valid_roc += 1
        for key, scores in score_map.items():
            for metric, value in roc_points(scores, y, weights).items():
                draws[key][metric].append(value)
        # Eligibility is a property of the shared draw, not the model or outcome.
        eligible = all(min(weights[(f == fold) & (y == label)].sum(),
                           weights[(f != fold) & (y == label)].sum()) > 0
                       for fold in np.unique(f) for label in (0, 1))
        if eligible:
            valid_accuracy += 1
            for key, scores in score_map.items():
                draws[key]["accuracy_official_folds"].append(
                    official_fold_accuracy(scores, y, f, weights)["accuracy"])

    def ci(values):
        if not len(values):
            return None
        return np.percentile(np.asarray(values), [2.5, 97.5]).tolist()

    models = {}
    tuned_keys = [key for key in score_map if key != "frozen"]
    for key in score_map:
        models[key] = {"metrics": points[key], "official_folds": fold_reports[key],
                       "subject_ci95": {metric: ci(draws[key][metric]) for metric in METRICS}}
        if key != "frozen":
            models[key]["paired_gain"] = {
                metric: {"delta": points[key][metric] - points["frozen"][metric],
                         "subject_ci95": ci(np.asarray(draws[key][metric]) - np.asarray(draws["frozen"][metric]))}
                for metric in METRICS}
    aggregate = {}
    for metric in METRICS:
        seed_values = [points[key][metric] for key in tuned_keys]
        boot_mean = np.mean([draws[key][metric] for key in tuned_keys], axis=0) if len(draws[tuned_keys[0]][metric]) else []
        delta_draws = np.asarray(boot_mean) - np.asarray(draws["frozen"][metric])
        aggregate[metric] = {"mean": float(np.mean(seed_values)),
                             "std": float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else None,
                             "n_seeds": len(seed_values), "delta_mean_vs_frozen": float(np.mean(seed_values) - points["frozen"][metric]),
                             "fixed_checkpoint_mean_gain_subject_ci95": ci(delta_draws)}
    return {"n_pairs": len(y), "n_subjects": len(subjects), "n_positive": int((y == 1).sum()),
            "n_negative": int((y == 0).sum()), "models": models, "seed_aggregate": aggregate,
            "bootstrap": {"sampling_unit": "person", "n_requested": n_boot, "seed": seed,
                          "n_valid_roc": valid_roc, "n_valid_accuracy": valid_accuracy,
                          "positive_weight": "one shared person multiplicity",
                          "negative_weight": "product of both endpoint multiplicities",
                          "accuracy_threshold_reselection": True,
                          "conditional_on_fixed_trained_checkpoints_and_official_folds": True,
                          "includes_training_seed_population_uncertainty": False,
                          "folds_are_subject_disjoint": all(count == 0 for count in shared_person_counts.values()),
                          "train_test_shared_person_count_per_fold": shared_person_counts,
                          "boundary_percentile_intervals_do_not_establish_zero_risk": True},
            "delta_definition": "tuned minus frozen; EER improvement is negative, other metric improvements positive",
            "roc_scope": "empirical ROC points; test-derived thresholds, not deployment calibration",
            "negative_events_at_nominal_far": {str(far): float((y == 0).sum() * far)
                                               for far in (0.01, 0.001)}}


def detection_sensitivity(score_map, labels, rows, sources):
    """Descriptive ROC subset, never relabelled canonical full-protocol accuracy."""
    y = np.asarray(labels)
    mask = np.asarray([all(sources[row[f"image_{side}"]]["detected"] is True
                          for side in ("a", "b")) for row in rows])
    endpoint_misses = sum(sources[row[f"image_{side}"]]["detected"] is not True
                          for row in rows for side in ("a", "b"))
    valid = set(y[mask]) == {0, 1}
    return {"endpoint_fallback_count": int(endpoint_misses),
            "endpoint_fallback_fraction": endpoint_misses / (2 * len(rows)),
            "pairs_with_any_fallback": int((~mask).sum()),
            "both_detected_pairs": int(mask.sum()),
            "both_detected_positive": int((y[mask] == 1).sum()),
            "both_detected_negative": int((y[mask] == 0).sum()),
            "both_detected_roc": {key: roc_points(np.asarray(scores)[mask], y[mask])
                                  for key, scores in score_map.items()} if valid else None,
            "scope": "descriptive retained-pair subset; no subset CI or canonical accuracy; selection can change composition"}


def load_bound(summary: Path, root: Path, protocol: Path, raw_root: Path):
    result = verified_result(summary, root)
    if result.get("row_binding_verified") is not True or result.get("full_transform_replay") is not True:
        raise ValueError("completed full row/source binding required before evaluation")
    cache = summary.parent / "private/lfw_bound.npz"
    replay = verify_binding(cache, protocol, raw_root)
    if replay["all_endpoints_checked"] != 12000 or replay["folds"] != 10:
        raise ValueError("full official View-2 cache required")
    rows = [json.loads(line) for line in (cache.parent / "pair_rows.jsonl").read_text(encoding="utf-8").splitlines()]
    return cache, rows, result


def unique_pool(cache: Path, rows: list[dict]):
    seen, images, indices = {}, [], []
    with np.load(cache, allow_pickle=False) as z:
        labels, folds = z["issame"], z["fold_ids"]
        for side in ("a", "b"):
            array = z[side]
            if len(array) != len(rows):
                raise ValueError("row/cache size mismatch")
            mapping = []
            for image, row in zip(array, rows, strict=True):
                digest = row[f"crop_sha256_{side}"]
                actual = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
                if actual != digest:
                    raise ValueError("unique-pool crop binding mismatch")
                if digest not in seen:
                    seen[digest] = len(images)
                    images.append(image)
                mapping.append(seen[digest])
            indices.append(np.asarray(mapping, dtype=np.int64))
    return images, indices[0], indices[1], labels, folds


def load_model(weights: Path, *, frozen: bool, expected_seed: int | None = None):
    import torch

    from age_gap.models.backbones import make_backbone

    data = torch.load(weights, map_location="cpu", weights_only=frozen)
    model = make_backbone("facenet", pretrained=False).to("cpu")  # never downloads weights
    if frozen:
        incompat = model.net.load_state_dict(data, strict=False)
        allowed = {"logits.weight", "logits.bias"}
        provenance = "local CASIA-WebFace pretrained weights"
    else:
        if data.get("backbone", "facenet") != "facenet":
            raise ValueError("checkpoint is not a FaceNet backbone")
        if "seed" in data and data["seed"] != expected_seed:
            raise ValueError("checkpoint seed differs from requested seed")
        incompat = model.load_state_dict(data["state_dict"], strict=False)
        allowed = {"net.logits.weight", "net.logits.bias"}
        provenance = "legacy checkpoint; training manifests not independently verified"
    if incompat.missing_keys or set(incompat.unexpected_keys) - allowed:
        raise ValueError("missing or unexpected embedding network parameters")
    return model.eval(), provenance


def embed_cpu(model, images, batch_size: int):
    import torch

    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            tensor = torch.from_numpy(np.stack([model.preprocess(image, bgr=True)
                                               for image in images[start:start + batch_size]]))
            output = model(tensor).cpu().numpy()
            if output.ndim != 2 or len(output) != len(tensor) or not np.isfinite(output).all():
                raise ValueError("invalid embedding output")
            if not np.allclose(np.linalg.norm(output, axis=1), 1, atol=1e-4):
                raise ValueError("embeddings must be L2-normalized")
            embeddings.append(output)
            if (start // batch_size) % 20 == 0:
                print(f"embedded {min(start + batch_size, len(images))}/{len(images)}", flush=True)
    return np.concatenate(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    home = Path.home()
    parser.add_argument("--cache-result", type=Path, default=PROJECT_ROOT / "metrics/lfw_bound_cache_20261002/lfw_bound_cache.json")
    parser.add_argument("--protocol", type=Path, default=home / "scikit_learn_data/lfw_home/pairs.txt")
    parser.add_argument("--raw-root", type=Path, default=home / "scikit_learn_data/lfw_home/lfw_funneled")
    parser.add_argument("--base-weights", type=Path, default=home / ".cache/torch/checkpoints/20180408-102900-casia-webface.pt")
    parser.add_argument("--models", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "metrics/lfw_bound_evaluation_20261002")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.n_boot) < 1:
        parser.error("positive threads, batch-size and bootstrap count required")
    checkpoints = {f"tuned_seed{seed}": args.models / f"bb_facenet_seed{seed}.pt" for seed in SEEDS}
    weights = {"frozen": args.base_weights, **checkpoints}
    if not args.execute:
        print(json.dumps({"status": "planned", "cache_manifest_exists": args.cache_result.with_suffix(".manifest.json").is_file(),
                          "local_weights_available": {key: path.is_file() for key, path in weights.items()},
                          "seeds": list(SEEDS), "device": "cpu", "n_boot": args.n_boot}))
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("refusing to overwrite evaluation; choose a new output directory")
    if any(not path.is_file() for path in weights.values()):
        raise FileNotFoundError("all local weights required; no download fallback")
    cache, rows, cache_result = load_bound(args.cache_result, PROJECT_ROOT, args.protocol, args.raw_root)
    import facenet_pytorch.models.inception_resnet_v1 as network_module
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sources = [Path(__file__), PROJECT_ROOT / "scripts/build_lfw_bound_cache.py",
               PROJECT_ROOT / "scripts/render_fgnet_evidence.py", PROJECT_ROOT / "scripts/export_public_evidence.py",
               PROJECT_ROOT / "src/age_gap/models/facenet.py", PROJECT_ROOT / "src/age_gap/models/backbones.py",
               Path(network_module.__file__)]
    manifest_path = args.cache_result.with_suffix(".manifest.json")
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bound_paths = [Path(record["path"]) if Path(record["path"]).is_absolute() else PROJECT_ROOT / record["path"]
                   for record in (*cache_manifest["inputs"], *cache_manifest["outputs"])]
    inputs = list(dict.fromkeys([manifest_path, *bound_paths, *weights.values(), *sources]))
    before = [file_record(path) for path in inputs]
    images, ia, ib, labels, folds = unique_pool(cache, rows)
    args.out.mkdir(parents=True, exist_ok=True)
    private = args.out / "private"
    private.mkdir()
    score_map, provenance, outputs = {}, {}, []
    for key, weight in weights.items():
        print(f"evaluating {key} on {len(images)} unique crops", flush=True)
        expected_seed = None if key == "frozen" else int(key.removeprefix("tuned_seed"))
        model, note = load_model(weight, frozen=key == "frozen", expected_seed=expected_seed)
        embedding = embed_cpu(model, images, args.batch_size)
        score_map[key] = np.sum(embedding[ia] * embedding[ib], axis=1)
        stored = private / f"embeddings_{key}.npz"
        np.savez_compressed(stored, embeddings=embedding, index_a=ia, index_b=ib)
        outputs.append(stored)
        provenance[key] = {"checkpoint_name": weight.name, "checkpoint_sha256": sha256_file(weight), "status": note}
        del model, embedding
    raw_scores = private / "scores.npz"
    np.savez_compressed(raw_scores, **score_map, labels=labels, folds=folds)
    outputs.append(raw_scores)
    result = paired_statistics(score_map, labels, folds, [row["subject_a"] for row in rows],
                               [row["subject_b"] for row in rows], n_boot=args.n_boot, seed=args.bootstrap_seed)
    sources_meta = json.loads((cache.parent / "source_metadata.json").read_text(encoding="utf-8"))
    package_versions = {name: version(name) for name in ("torch", "facenet-pytorch", "numpy", "scikit-learn", "Pillow")}
    result.update({"protocol": "official_lfw_view2_contiguous_folds_source_bound", "seeds": list(SEEDS),
                   "n_unique_crops_embedded": len(images), "cache_full_transform_replay": cache_result["full_transform_replay"],
                   "threshold_grid": {"minimum": -1, "maximum": 1, "count": len(THRESHOLDS),
                                      "sha256_float64_little_endian": THRESHOLD_SHA256, "tie_break": "lowest threshold"},
                   "package_versions": package_versions,
                   "detection_sensitivity": detection_sensitivity(score_map, labels, rows, sources_meta),
                   "accuracy_acceptance": "cosine >= threshold", "provenance": provenance,
                   "training_identity_independence": "unverified", "publication_ready": False,
                   "legacy_accuracy_protocol": "interleaved folds; not substituted or pooled with new results"})
    scan_unsafe(result)
    if before != [file_record(path) for path in inputs]:
        raise ValueError("evaluation inputs changed; completed artifact refused")
    target = args.out / "lfw_bound_evaluation.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    outputs.append(target)
    manifest = write_experiment_manifest(target.with_suffix(".manifest.json"), experiment="lfw-bound-3seed-subject-evaluation",
                              parameters={"seeds": list(SEEDS), "device": "cpu", "threads": args.threads,
                                          "batch_size": args.batch_size, "n_boot": args.n_boot,
                                          "bootstrap_seed": args.bootstrap_seed, "threshold_grid_count": len(THRESHOLDS),
                                          "threshold_grid_sha256_float64_little_endian": THRESHOLD_SHA256,
                                          "package_versions": package_versions,
                                          "fold_thresholds_reselected_in_bootstrap": True},
                              metrics=result, inputs=inputs, outputs=outputs,
                              command=[sys.executable, "-m", "scripts.evaluate_lfw_bound", *sys.argv[1:]])
    if json.loads(manifest.read_text(encoding="utf-8"))["inputs"] != before:
        manifest.unlink()
        raise ValueError("inputs changed while writing final manifest")
    print(f"completed: {target}", flush=True)


if __name__ == "__main__":
    main()
