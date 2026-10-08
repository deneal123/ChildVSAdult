"""Bind a metadata-only target stream preflight; no training/image exposures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from age_gap.models.iresnet import ArcFaceBackbone
from scripts.mtlface_epoch_v2 import load_bound_dataset
from scripts.mtlface_target_stream_v3 import TargetAgeStream


def _resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def arcface_preprocess(image):
    # Exact bound preprocessing, without allocating/loading a neural backbone.
    return ArcFaceBackbone.preprocess(None, image)


def prepare(faces, out, seeds, batches, batch_size):
    if out.exists():
        raise FileExistsError("fresh output required")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**63 for s in seeds)
        or isinstance(batches, bool)
        or not isinstance(batches, int)
        or batches <= 0
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("unique nonnegative seeds and positive batch partition required")
    dataset, native = load_bound_dataset(faces, arcface_preprocess)
    # Bind both the native manifest and every native input/output, plus local
    # transitive implementation sources. Do not accept current bytes as new
    # expectations for crops: expectations come exclusively from native records.
    bindings = {_resolve(record["path"]): record for record in native["inputs"]}
    sources = {
        _resolve(faces),
        *bindings,
        *[_resolve(record["path"]) for record in native["outputs"]],
        *PROJECT_ROOT.glob("src/age_gap/**/*.py"),
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "uv.lock",
        Path(__file__).resolve(),
        *[
            PROJECT_ROOT / ("scripts/" + name + ".py")
            for name in (
                "mtlface_target_stream_v3",
                "mtlface_epoch_v2",
                "mtlface_training_v2",
                "mtlface_components_v2",
                "prepare_sota_face_list",
            )
        ],
    }
    sources = sorted(sources, key=str)
    before = [file_record(path) for path in sources]
    cells = []
    for seed in seeds:
        stream = TargetAgeStream(dataset, torch.Generator().manual_seed(seed), bindings)
        for _ in range(batches):
            stream.draw_indices(batch_size)
        cells.append(dict(seed=seed, ledger=stream.ledger()))
    metrics = dict(
        metadata_sampling_preflight_complete=True,
        training_complete=False,
        scientific_evaluation_complete=False,
        publication_ready=False,
        retained_images=len(dataset),
        retained_recorded_identities=dataset.n_classes,
        missing_age_rows=native["metrics"]["counts"]["retained_age_missing"],
        conflict_masked_age_rows=native["metrics"]["counts"]["retained_age_conflict_masked"],
        cells=cells,
    )
    if before != [file_record(path) for path in sources]:
        raise RuntimeError("preflight inputs changed")
    out.mkdir(parents=True)
    summary = out / "summary.json"
    summary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest = out / "summary.manifest.json"
    write_experiment_manifest(
        manifest,
        experiment="mtlface-target-stream-metadata-preflight",
        parameters=dict(
            seeds=seeds,
            batches=batches,
            batch_size=batch_size,
            draws_not_training_budget=True,
            policy="uniform-group-then-uniform-row-with-replacement",
            age_boundaries=[10, 20, 30, 40, 50, 60],
            group_rule="age strictly greater than boundary",
            preprocessing="bound ArcFaceBackbone.preprocess; unused for metadata-only draws",
            rng_policy="caller-owned CPU generator; no per-item or epoch reseeding",
        ),
        metrics=metrics,
        inputs=sources,
        outputs=[summary],
    )
    if before != [file_record(path) for path in sources]:
        manifest.unlink(missing_ok=True)
        raise RuntimeError("preflight inputs changed during binding")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    metrics = prepare(args.faces, args.out, args.seeds, args.batches, args.batch_size)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
