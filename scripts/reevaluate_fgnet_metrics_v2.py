"""Reconstruct the full headline FG-NET protocol from bound full-pool embeddings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from age_gap.evaluation.fgnet import _match_negatives
from scripts.benchmark_metrics_v2 import KEYS, infer
from scripts.fgnet_retrieval_study import embedding_cache_key
from scripts.render_fgnet_evidence import verified_result


def full_embeddings(path, *, n_crops, role, weights_sha, source_sha):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"key", "indices", "embeddings"}:
            raise ValueError("keyed embedding cache required")
        indices, vectors = data["indices"], data["embeddings"]
        if (indices.dtype.kind not in "iu" or indices.shape != (n_crops,)
                or not np.array_equal(np.sort(indices), np.arange(n_crops))
                or vectors.ndim != 2 or len(vectors) != n_crops or vectors.dtype.kind != "f"
                or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
            raise ValueError("complete unique indexed unit embeddings required; no subset substitution")
        expected = embedding_cache_key(role=role, weights_sha256=weights_sha, source_sha256=source_sha, indices=indices)
        if str(data["key"]) != expected:
            raise ValueError("embedding model/source/preprocessing key mismatch")
    reordered = np.empty_like(vectors)
    reordered[indices] = vectors
    return reordered


def protocol(subjects, ages):
    subjects, ages = np.asarray(subjects), np.asarray(ages)
    if subjects.ndim != 1 or ages.shape != subjects.shape or any(v.dtype.kind not in "iu" for v in (subjects, ages)) or (ages < 0).any():
        raise ValueError("numeric FG-NET metadata required")
    if (ages > np.iinfo(np.int64).max).any():
        raise ValueError("ages exceed signed integer range")
    ages = ages.astype(np.int64)
    groups = {}
    for index, subject in enumerate(subjects.tolist()):
        groups.setdefault(subject, []).append(index)
    positives = [(a, b, abs(int(ages[a]) - int(ages[b]))) for ids in groups.values()
                 for j, a in enumerate(ids) for b in ids[j + 1:]]
    matched = _match_negatives(subjects, ages, positives, tolerance=2, seed=42)
    if not matched:
        raise ValueError("empty matched protocol")
    positive = [positives[p] for p, *_ in matched]
    negatives = [(a, b, gap) for _, a, b, gap, _, _ in matched]
    pairs = positive + negatives
    left, right = np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])
    labels = np.r_[np.ones(len(positive), int), np.zeros(len(positive), int)]
    source_gap = np.asarray([p[2] for p in positive] * 2)
    if not np.array_equal(np.abs(ages[left] - ages[right]), source_gap):
        raise ValueError("observed negative gap differs from source positive gap")
    return left, right, labels, source_gap, {"source_positives": len(positives), "retained_positives": len(positive),
        "source_25plus_positives": sum(gap >= 25 for _, _, gap in positives),
        "retained_25plus_positives": sum(gap >= 25 for _, _, gap in positive)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    root = PROJECT_ROOT
    source = root / "data/external/fgnet_crops.npz"
    original = root / "metrics/fgnet_error_breakdown/fgnet_error_breakdown.json"
    prior = verified_result(original, root)
    manifest_path = original.with_suffix(".manifest.json")
    native = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = Path.home() / ".cache/torch/checkpoints/20180408-102900-casia-webface.pt"
    weights = {"frozen": base, **{key: root / f"models/bb_facenet_seed{key.removeprefix('tuned_seed')}.pt" for key in KEYS[1:]}}
    embeddings = {key: original.parent / f"private/private_embeddings_{key}.npz" for key in KEYS}
    if any(file_record(p) not in native["inputs"] for p in [source, *weights.values()]) or any(file_record(p) not in native["outputs"] for p in embeddings.values()):
        raise ValueError("full-cache source/weights/embedding original binding mismatch")
    paths = [original, manifest_path, source, *weights.values(), *embeddings.values(), Path(__file__),
        root / "scripts/benchmark_metrics_v2.py", root / "scripts/verification_metrics_v2.py",
        root / "scripts/fgnet_error_breakdown.py", root / "scripts/fgnet_retrieval_study.py",
        root / "scripts/render_fgnet_evidence.py", *sorted((root / "src/age_gap").rglob("*.py")),
        *sorted((root / "src/age_gap").rglob("*.toml"))]
    before = [file_record(p) for p in paths]
    with np.load(source, allow_pickle=False) as data:
        subjects, ages = data["subjects"], data["ages"]  # do not load crops
    left, right, labels, gaps, coverage = protocol(subjects, ages)
    scores = {}
    for key in KEYS:
        role = "frozen_casia_webface" if key == "frozen" else f"tuned_facenet_seed{key.removeprefix('tuned_seed')}"
        model = full_embeddings(embeddings[key], n_crops=len(subjects), role=role,
            weights_sha=file_record(weights[key])["sha256"], source_sha=file_record(source)["sha256"])
        scores[key] = np.sum(model[left] * model[right], axis=1)
    result = {"protocol": "original full-cache endpoint_age_matched; seed42/tolerance2; source-positive-gap strata",
        "overall": infer(scores, labels, subjects[left], subjects[right]),
        "large_gap_25plus": infer({k: v[gaps >= 25] for k, v in scores.items()}, labels[gaps >= 25],
                                  subjects[left][gaps >= 25], subjects[right][gaps >= 25]),
        "coverage": coverage, "cached_images": len(subjects), "cached_subjects": len(np.unique(subjects)),
        "scope": "all cached images, not all raw FG-NET photographs; cached preprocessing not replayed",
        "reused_cache_preprocessing": native["parameters"]["preprocessing"],
        "original_error_protocol_reused": False, "image_inference_performed": False,
        "source_embedding_model_count": len(prior["models"]), "publication_ready": False}
    if before != [file_record(p) for p in paths]:
        raise ValueError("inputs changed during FG-NET metric re-evaluation")
    args.out.mkdir(parents=True)
    private = args.out / "private"
    private.mkdir()
    stored = private / "scores.npz"
    np.savez_compressed(stored, **scores, labels=labels, subject_a=subjects[left], subject_b=subjects[right], source_gap=gaps)
    output = args.out / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = output.with_suffix(".manifest.json")
    write_experiment_manifest(target, experiment="fgnet-headline-full-cache-roc-v2", parameters={"seed": 0,
        "protocol_seed": 42, "tolerance": 2, "n_boot": 2000, "image_inference": False}, metrics=result,
        inputs=paths, outputs=[output, stored])
    if json.loads(target.read_text(encoding="utf-8"))["inputs"] != before:
        target.unlink()
        raise ValueError("inputs changed while writing manifest; completed marker withdrawn")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
