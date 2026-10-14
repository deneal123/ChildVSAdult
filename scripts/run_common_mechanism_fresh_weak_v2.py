"""Prospective fresh-weak common diagnostics; no legacy weak-history recovery.

Run only after coordinator acceptance of the fresh CUDA evaluator and E426
reader. This distinct schema is not consumed by old common renderers/auditors.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.fresh_weak_common_binding_v1 import (
    PAIR_FIELDS,
    WEAK_EXPERIMENT,
    exact_pairs,
    read_vectors,
    script_ancestry,
    strong_cell,
    weak_files,
)
from scripts.identity_separability_v1 import infer
from scripts.mechanism_diagnostics_v1 import age_probe, drift, paired_age_error
from scripts.run_oriented_campaign import paths_from, verified


def validate_pairs(left, right, labels, a, b, people):
    left, right, labels = np.asarray(left), np.asarray(right), np.asarray(labels)
    if (
        left.ndim != 1
        or right.shape != left.shape
        or labels.shape != left.shape
        or left.dtype.kind not in "iu"
        or right.dtype.kind not in "iu"
        or labels.dtype.kind not in "iu"
        or set(np.unique(labels)) != {0, 1}
        or np.any(left < 0)
        or np.any(right < 0)
        or np.any(left >= len(people))
        or np.any(right >= len(people))
    ):
        raise ValueError("valid shared image-index pairs required")
    if not np.array_equal(a, people[left]) or not np.array_equal(b, people[right]):
        raise ValueError("pair endpoint metadata not linked to image indices")
    if not np.array_equal(labels, (people[left] == people[right]).astype(int)):
        raise ValueError("pair labels contradict source people")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong", type=Path, required=True)
    parser.add_argument("--weak", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output required")
    strong = verified(args.strong, "adaface-fixed8-cuda-full-fgnet-roc-v2")
    strong_cell(strong)
    weak = verified(args.weak, WEAK_EXPERIMENT)
    fresh_weak_files = weak_files(weak, args.weak)
    if strong["metrics"].get("execution_complete") is not True:
        raise ValueError("completed strong evaluation required")
    cache = PROJECT_ROOT / "data/external/fgnet_crops.npz"
    for native in (strong, weak):
        if file_record(cache) not in native["inputs"]:
            raise ValueError("shared image metadata cache not bound")
    dependencies = [
        PROJECT_ROOT / f"scripts/{name}.py"
        for name in (
            "identity_separability_v1",
            "mechanism_diagnostics_v1",
            "run_mechanism_diagnostics_v1",
            "run_oriented_campaign",
            "run_restricted_matched_campaign",
            "evaluate_oriented_cuda_v1",
        )
    ]
    dependencies.extend(script_ancestry(Path(__file__), PROJECT_ROOT))
    dependencies.extend(
        script_ancestry(PROJECT_ROOT / "scripts/evaluate_facenet_fixed8_cuda_v3.py", PROJECT_ROOT)
    )
    dependencies.extend(sorted((PROJECT_ROOT / "src/age_gap").rglob("*.py")))
    inputs = sorted(
        set(
            [
                cache,
                Path(__file__),
                args.strong,
                args.weak,
                *dependencies,
                PROJECT_ROOT / "src/age_gap/common/manifest.py",
                *paths_from(strong),
                *paths_from(weak),
            ]
        )
    )
    before = [file_record(p) for p in inputs]
    with np.load(cache, allow_pickle=False) as data:
        people, ages = data["subjects"], data["ages"]
    pair_path = args.strong.parent / "private/scores.npz"
    if file_record(pair_path) not in strong["outputs"]:
        raise ValueError("common pair protocol not bound")
    with np.load(pair_path, allow_pickle=False) as data:
        left, right, labels = data["left"], data["right"], data["labels"]
        a, b = data["subject_a"], data["subject_b"]
    validate_pairs(left, right, labels, a, b, people)
    weak_pair_path = args.weak.parent / "private/scores.npz"
    if file_record(weak_pair_path) not in weak["outputs"]:
        raise ValueError("fresh weak pair protocol not bound")
    with np.load(pair_path, allow_pickle=False) as data:
        strong_pairs = {key: data[key] for key in PAIR_FIELDS}
    with np.load(weak_pair_path, allow_pickle=False) as data:
        weak_pairs = {key: data[key] for key in PAIR_FIELDS}
    exact_pairs(strong_pairs, weak_pairs, people, ages)
    strong_files = {"frozen": args.strong.parent / "private/frozen_embeddings.npz"}
    strong_files.update(
        {
            key: args.strong.parent / f"private/{key}_embeddings.npz"
            for key in strong["parameters"]["actual_cells"]
        }
    )
    weak_vectors = fresh_weak_files
    result, private = {}, {}
    with threadpool_limits(limits=1):
        for family, files, native in (
            ("strong", strong_files, strong),
            ("weak", weak_vectors, weak),
        ):
            for path in files.values():
                if file_record(path) not in native["outputs"]:
                    raise ValueError("full embedding cache not bound")
            vectors = {k: read_vectors(p) for k, p in files.items()}
            scores = {k: np.sum(v[left] * v[right], axis=1) for k, v in vectors.items()}
            probes = {k: age_probe(v, ages, people) for k, v in vectors.items()}
            diagnostics = {
                k: dict(
                    representation=drift(vectors["frozen"], v),
                    paired_age_error=paired_age_error(probes["frozen"], probes[k]),
                )
                for k, v in vectors.items()
                if k != "frozen"
            }
            result[family] = dict(separability=infer(scores, labels, a, b), diagnostics=diagnostics)
            private[family] = probes
            print(f"common image-index family completed: {family}", flush=True)
    if before != [file_record(p) for p in inputs]:
        raise RuntimeError("inputs changed")
    (args.out / "private").mkdir(parents=True)
    detail = args.out / "private/age_probes.json"
    detail.write_text(json.dumps(private, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metrics = dict(
        execution_complete=True,
        publication_ready=False,
        results=result,
        fresh_weak_lineage=True,
        legacy_weak_history_recovered=False,
        strong_cells=strong["parameters"]["actual_cells"],
        weak_keys=list(fresh_weak_files),
        weak_upstream_bindings=weak["metrics"]["bindings"],
        fixed_oof_bootstrap=True,
        neural_inference_replayed=False,
        mechanism_complete=False,
        n_images=len(people),
        n_pairs=len(labels),
        limitation="common fixed image pairs and fresh weak fixed8 lineage, conditional checkpoints/fixed OOF folds; independent native acceptance remains coordinator responsibility; no legacy recovery, controlled causal architecture comparison, human train-benchmark independence or completed strong matrix/mechanism",
    )
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = args.out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="common-image-index-mechanism-v2-fresh-weak",
        parameters=dict(
            device="cpu",
            blas_threads=1,
            age_alpha=1,
            age_folds=5,
            age_fold_seed=42,
            bootstrap_seed=0,
            resamples=2000,
            protocol="exact strong and fresh weak bound image-index pairs",
            weak_parent_experiment=WEAK_EXPERIMENT,
            weak_key_mapping="literal tuned_sN; no legacy renaming",
        ),
        metrics=metrics,
        inputs=inputs,
        outputs=[summary, detail],
    )
    from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs

    validate_written_inputs(target, before)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(
            json.dumps(
                dict(execution_complete=False, reason="fresh-common-refusal-no-private-details")
            )
        )
        raise SystemExit(1) from None
