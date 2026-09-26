"""Run three-seed MTLFace-CP and CACon-CP under one controlled protocol."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.common.manifest import write_experiment_manifest
from age_gap.evaluation.external_suite import eval_all
from age_gap.models.aging import FRANAging, fran_weights_path
from age_gap.training.sota_common import (
    common_protocol_faces,
    train_cacon_common,
    train_mtlface_common,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _aggregate_completed_runs(
    *, method: str, seeds: list[int], backbone: str, epochs: int, results_dir: Path
) -> Path | None:
    paths = [results_dir / f"{method}_{backbone}_e{epochs}_s{seed}.json" for seed in seeds]
    if not all(path.exists() for path in paths):
        return None
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    common_metrics = set.intersection(*(set(run["metrics"]) for run in runs))
    aggregate: dict[str, dict[str, float | int]] = {}
    for metric in sorted(common_metrics):
        values = np.asarray([run["metrics"][metric] for run in runs], dtype=float)
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n_seeds": len(values),
        }
    output = results_dir / f"{method}_{backbone}_e{epochs}_summary.json"
    payload = {
        "method": f"{method}-common-protocol",
        "backbone": backbone,
        "epochs": epochs,
        "seeds": seeds,
        "runs": [run["run_id"] for run in runs],
        "metrics": aggregate,
    }
    _write(output, payload)
    write_experiment_manifest(
        output.with_suffix(".manifest.json"),
        experiment=f"{method}-common-protocol-three-seed-summary",
        parameters={"backbone": backbone, "epochs": epochs, "seeds": seeds},
        metrics=aggregate,
        inputs=paths,
        outputs=[output],
    )
    return output


def _file_set_digest(paths: list[Path]) -> dict[str, int | str]:
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                total_bytes += len(chunk)
    return {"files": len(paths), "bytes": total_bytes, "sha256": digest.hexdigest()}


def prepare_cacon_cache(destination: Path, seed: int = 20260922) -> Path:
    """Generate each held-in training face's third (cross-age) view exactly once."""
    destination.mkdir(parents=True, exist_ok=True)
    items = common_protocol_faces("train")
    # Keep the RNG stream tied to the face's stable position in the complete
    # identity-safe training set.  Otherwise resuming a partial cache would
    # renumber the remaining items and generate different views.
    missing = [
        (source_index, item)
        for source_index, item in enumerate(items)
        if not (destination / item.path.name).exists()
    ]
    if missing:
        aging = FRANAging()
        for completed, (source_index, item) in enumerate(missing, start=1):
            image = cv2.imread(str(item.path))
            if image is None:
                raise RuntimeError(f"cannot read {item.path}")
            transformed = aging(image, np.random.default_rng(seed + source_index))
            if not cv2.imwrite(str(destination / item.path.name), transformed):
                raise RuntimeError(f"cannot write {destination / item.path.name}")
            if completed % 100 == 0:
                print(f"CACon cache {completed}/{len(missing)}")
        del aging
    outputs = [destination / item.path.name for item in items]
    cache_index = destination / "cache_index.json"
    cache_index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_faces": _file_set_digest([item.path for item in items]),
                "aged_views": _file_set_digest(outputs),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = destination / "manifest.json"
    write_experiment_manifest(
        manifest,
        experiment="cacon-fran-third-view-cache",
        parameters={"generator": "FRAN", "seed": seed, "faces": len(items)},
        metrics={"cached_faces": len(outputs)},
        inputs=[
            data_path("data_dir", "processed", "identity_groups.jsonl"),
            fran_weights_path(),
        ],
        outputs=[cache_index],
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=["mtlface", "cacon"], default=["mtlface", "cacon"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    parser.add_argument(
        "--backbone",
        default="arcface_r50_casia",
        help="shared backbone for both comparators (default has locally checksummed weights)",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--scope", choices=["head", "tail", "full"], default="head")
    parser.add_argument("--prepare-cacon", action="store_true")
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()

    aged_dir = data_path("data_dir", "interim", "cacon_fran_views")
    if args.prepare_cacon or "cacon" in args.methods:
        prepare_cacon_cache(aged_dir)
        if args.prepare_cacon and args.methods == ["mtlface", "cacon"]:
            return

    device = torch_device()
    results_dir = data_path("metrics_dir", "sota_common_protocol")
    models_dir = data_path("models_dir", "sota_common_protocol")
    completed = 0
    for method in args.methods:
        for seed in args.seeds:
            run_id = f"{method}_{args.backbone}_e{args.epochs}_s{seed}"
            output = results_dir / f"{run_id}.json"
            manifest = results_dir / f"{run_id}.manifest.json"
            if output.exists() and manifest.exists():
                continue
            if args.max_runs is not None and completed >= args.max_runs:
                return
            checkpoint = models_dir / f"{run_id}.pt"
            common = {
                "backbone_name": args.backbone,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "scope": args.scope,
                "seed": seed,
                "checkpoint": checkpoint,
                "device": device,
            }
            if method == "mtlface":
                model, history = train_mtlface_common(**common)
                extra_inputs: list[Path] = []
            else:
                model, history = train_cacon_common(aged_dir=aged_dir, **common)
                extra_inputs = [aged_dir / "manifest.json"]
            metrics = eval_all(model, device)
            payload = {
                "run_id": run_id,
                "method": f"{method}-common-protocol",
                "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in common.items() if key != "device"},
                "history": history,
                "metrics": metrics,
            }
            _write(output, payload)
            write_experiment_manifest(
                manifest,
                experiment=f"{method}-common-protocol",
                parameters=payload["parameters"],
                metrics=metrics,
                inputs=[data_path("data_dir", "processed", "identity_groups.jsonl"), *extra_inputs],
                outputs=[checkpoint, output],
            )
            completed += 1
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for method in args.methods:
        summary = _aggregate_completed_runs(
            method=method,
            seeds=args.seeds,
            backbone=args.backbone,
            epochs=args.epochs,
            results_dir=results_dir,
        )
        if summary is not None:
            print(f"summary: {summary}")


if __name__ == "__main__":
    main()
