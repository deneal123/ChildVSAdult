"""Bound local CPU CACD-VS scoring and ROC-v2; legacy results are never overwritten."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.benchmark_metrics_v2 import KEYS
from scripts.cacd_metrics_v2 import infer
from scripts.export_lfw_evidence import verify_manifest
from scripts.fgnet_retrieval_study import (
    PREPROCESSING,
    embed_unique_crops,
    load_frozen_model,
    load_tuned_model,
)


def canonical(labels, folds):
    y, f = np.asarray(labels), np.asarray(folds)
    expected_f = np.arange(4000) // 400
    expected_y = (np.arange(4000) % 400 < 200).astype(int)
    if y.dtype.kind not in "iu" or f.dtype.kind not in "iu" or not np.array_equal(y, expected_y) or not np.array_equal(f, expected_f):
        raise ValueError("exact canonical 4000 pair order/classes/contiguous folds required")


def checkpoint_records(weights):
    if set(weights) != set(KEYS):
        raise ValueError("exact four checkpoint roles required")
    records = {key: file_record(weights[key]) for key in KEYS}
    if len({record["sha256"] for record in records.values()}) != len(KEYS):
        raise ValueError("distinct frozen/tuned checkpoint files required")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    root = PROJECT_ROOT
    source = root / "data/external/cacd_vs_aligned.npz"
    source_manifest = source.with_suffix(".manifest.json")
    native = verify_manifest(source_manifest, root, "cacd-vs-alignment-cache")
    if file_record(source) not in native["outputs"]:
        raise ValueError("aligned source is not a bound output")
    base = Path.home() / ".cache/torch/checkpoints/20180408-102900-casia-webface.pt"
    weights = {"frozen": base, "tuned_seed42": root / "models/bb_facenet_seed42.pt",
        "tuned_seed1": root / "models/bb_facenet_seed1.pt", "tuned_seed2": root / "models/bb_facenet_seed2.pt"}
    weight_records = checkpoint_records(weights)
    paths = [source, source_manifest, *weights.values(), Path(__file__), root / "scripts/cacd_metrics_v2.py",
        root / "scripts/benchmark_metrics_v2.py", root / "scripts/verification_metrics_v2.py",
        root / "scripts/evaluate_lfw_bound.py", root / "scripts/fgnet_retrieval_study.py", root / "scripts/export_lfw_evidence.py",
        root / "uv.lock", root / "pyproject.toml",
        *sorted((root / "src/age_gap").rglob("*.py")), *sorted((root / "src/age_gap").rglob("*.toml"))]
    before = [file_record(p) for p in paths]
    with np.load(source, allow_pickle=False) as data:
        if set(data.files) != {"a", "b", "issame", "fold_ids", "miss_rate"}:
            raise ValueError("unexpected aligned source schema")
        labels, folds = data["issame"], data["fold_ids"]
        canonical(labels, folds)
        miss_rate = float(data["miss_rate"])
        if not np.isfinite(miss_rate) or not 0 <= miss_rate <= .05:
            raise ValueError("invalid alignment fallback fraction")
        if args.execute:
            a, b = data["a"], data["b"]
            if any(x.dtype != np.uint8 or x.shape != (4000, 112, 112, 3) for x in (a, b)):
                raise ValueError("canonical BGR uint8 aligned arrays required")
            crops = np.concatenate((a, b))
    plan = {"seed": 0, "n_boot": 2000, "device": "cpu", "threads": 2, "batch_size": 32,
        "n_pairs": 4000, "n_crop_rows": 8000, "preprocessing": PREPROCESSING,
        "checkpoint_records": weight_records, "source_record": before[0],
        "image_inference_authorized_by_execute": args.execute, "subject_metadata_available": False,
        "scope": "fresh local weights/scores and pair CI; original alignment replay and person independence unverified",
        "publication_ready": False}
    if before != [file_record(p) for p in paths] or checkpoint_records(weights) != weight_records:
        raise ValueError("preflight inputs changed")
    args.out.mkdir(parents=True)
    plan_path = args.out / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    plan_manifest = plan_path.with_suffix(".manifest.json")
    write_experiment_manifest(plan_manifest, experiment="cacd-vs-roc-v2-preflight", parameters=plan,
        metrics={"execution_complete": False, "publication_ready": False}, inputs=paths, outputs=[plan_path])
    if json.loads(plan_manifest.read_text(encoding="utf-8"))["inputs"] != before:
        plan_manifest.unlink()
        raise ValueError("inputs changed during preflight manifest write; marker withdrawn")
    plan_records = [file_record(plan_path), file_record(plan_manifest)]
    if not args.execute:
        print("Source/weights/protocol preflight passed; no image inference performed")
        return
    private = args.out / "private"
    private.mkdir()
    scores, outputs = {}, []
    for key in KEYS:
        print(f"CACD-VS CPU scoring {key}: 8000 crop rows", flush=True)
        model = load_frozen_model(weights[key]) if key == "frozen" else load_tuned_model(weights[key])
        cache = private / f"embeddings_{key}.npz"
        vectors, _ = embed_unique_crops(crops, np.arange(8000), model, role=f"cacd_vs_{key}",
            weights_sha256=file_record(weights[key])["sha256"], source_sha256=before[0]["sha256"],
            cache_path=cache, batch_size=32, threads=2, reuse_cache=False)
        if vectors.shape[0] != 8000 or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
            raise ValueError("invalid full indexed embeddings")
        scores[key] = np.sum(vectors[:4000] * vectors[4000:], axis=1)
        outputs.append(cache)
        del model, vectors
        gc.collect()
        print(f"CACD-VS CPU scoring {key}: complete", flush=True)
    score_path = private / "scores.npz"
    np.savez_compressed(score_path, **scores, labels=labels, folds=folds)
    print("CACD-VS paired fold/class-stratified pair bootstrap", flush=True)
    result = {"inference": infer(scores, labels, folds), "plan": plan, "alignment_fallback_fraction": miss_rate,
        "execution_complete": True,
        "legacy_results_rewritten": False, "publication_ready": False}
    if before != [file_record(p) for p in paths] or plan_records != [file_record(plan_path), file_record(plan_manifest)]:
        raise ValueError("inputs changed during execution")
    output = args.out / "summary.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    target = output.with_suffix(".manifest.json")
    write_experiment_manifest(target, experiment="cacd-vs-local-4checkpoint-roc-v2", parameters=plan, metrics=result,
        inputs=paths + [plan_path, plan_manifest], outputs=[output, score_path, *outputs])
    if json.loads(target.read_text(encoding="utf-8"))["inputs"] != before + plan_records:
        target.unlink()
        raise ValueError("inputs changed during manifest write; completed marker withdrawn")
    print("CACD-VS ROC-v2 complete; native scores/embeddings/result bound", flush=True)


if __name__ == "__main__":
    main()
