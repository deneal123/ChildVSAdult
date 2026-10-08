"""Native-bound CPU representation/age diagnostics for completed strong FG-NET cells."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.mechanism_diagnostics_v1 import VERSION, age_probe, drift, paired_age_error
from scripts.run_oriented_campaign import paths_from, verified


def aligned_vectors(path, count):
    with np.load(path, allow_pickle=False) as data:
        indices, vectors = data["indices"], data["embeddings"]
    if not np.array_equal(indices, np.arange(count)):
        raise ValueError("full canonical image-index order required")
    if vectors.shape != (count, 512) or not np.isfinite(vectors).all():
        raise ValueError("complete finite 512d embeddings required")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("unit embeddings required")
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output required")
    cache = PROJECT_ROOT / "data/external/fgnet_crops.npz"
    natives = [verified(p, "adaface-fixed8-cuda-full-fgnet-roc-v2") for p in args.bindings]
    for native in natives:
        if native["metrics"].get("execution_complete") is not True:
            raise ValueError("completed external evaluation required")
        if file_record(cache) not in native["inputs"]:
            raise ValueError("metadata cache not bound by embedding producer")
    inputs = sorted(set([cache, Path(__file__),
                         PROJECT_ROOT / "scripts/mechanism_diagnostics_v1.py",
                         PROJECT_ROOT / "scripts/run_oriented_campaign.py",
                         PROJECT_ROOT / "scripts/run_restricted_matched_campaign.py",
                         PROJECT_ROOT / "src/age_gap/common/manifest.py",
                         *args.bindings, *(p for m in natives for p in paths_from(m))]))
    before = [file_record(p) for p in inputs]
    with np.load(cache, allow_pickle=False) as data:
        ages, people = data["ages"], data["subjects"]
    summary, private = {}, {}
    with threadpool_limits(limits=1):
        for binding, native in zip(args.bindings, natives, strict=True):
            frozen_path = binding.parent / "private/frozen_embeddings.npz"
            frozen = aligned_vectors(frozen_path, len(ages))
            baseline = age_probe(frozen, ages, people)
            for key in native["parameters"]["actual_cells"]:
                if key in summary:
                    raise ValueError("duplicate cell")
                tuned_path = binding.parent / f"private/{key}_embeddings.npz"
                if any(file_record(p) not in native["outputs"]
                       for p in (frozen_path, tuned_path)):
                    raise ValueError("embedding output not bound")
                tuned = aligned_vectors(tuned_path, len(ages))
                probe = age_probe(tuned, ages, people)
                private[key] = dict(frozen_age_probe=baseline, tuned_age_probe=probe)
                summary[key] = dict(representation=drift(frozen, tuned),
                                    paired_age_error=paired_age_error(baseline, probe),
                                    frozen_mae_person=baseline["mae_person"],
                                    tuned_mae_person=probe["mae_person"],
                                    frozen_mean_baseline_mae_image=baseline["mean_baseline_mae_image"],
                                    tuned_mean_baseline_mae_image=probe["mean_baseline_mae_image"])
                print(f"diagnosed fixed cell: {key}", flush=True)
    if before != [file_record(p) for p in inputs]:
        raise RuntimeError("inputs changed during diagnostics")
    (args.out / "private").mkdir(parents=True)
    private_path = args.out / "private/age_probes.json"
    private_path.write_text(json.dumps(private, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result = dict(execution_complete=True, publication_ready=False, cells=summary,
                  limitation="FG-NET descriptive fixed-checkpoint diagnostics; incomplete36-cell matrix; weak comparison and identity separability remain open; chronological ages; conditional OOF uncertainty")
    target = args.out / "summary.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_experiment_manifest(args.out / "summary.manifest.json", experiment=VERSION,
                              parameters=dict(folds=5, fold_seed=42, alpha=1,
                                              bootstrap_seed=0, resamples=2000,
                                              device="cpu", blas_threads=1,
                                              fits="equal-person mass, fit-only standardization"),
                              metrics=result, inputs=inputs, outputs=[target, private_path])
    from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs

    validate_written_inputs(args.out / "summary.manifest.json", before)


if __name__ == "__main__":
    main()
