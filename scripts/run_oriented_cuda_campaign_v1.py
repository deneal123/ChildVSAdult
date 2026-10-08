"""Serial remaining five CUDA cells after verified LOWseed42 completion."""

import argparse
import subprocess
import sys
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.run_oriented_campaign import PROTOCOL, paths_from, verified


def check_cell(path, arm, seed):
    native = verified(path, "oriented-native-cuda-training-cell")
    parameters = native["parameters"]
    expected = PROTOCOL | dict(device="cuda", arm=arm, seed=seed)
    for key in (
        "backbone",
        "device",
        "arm",
        "seed",
        "epochs_requested",
        "epochs_executed",
        "selected_epoch",
        "checkpoint_selection",
        "batchnorm_policy",
        "trainable_scope",
        "learning_rate",
        "batch_size",
        "margin",
        "gap_weight",
        "crops_dir",
    ):
        if parameters.get(key) != expected[key]:
            raise ValueError(f"CUDA cell protocol mismatch: {key}")
    if native["metrics"].get("training_complete") is not True:
        raise ValueError("completed CUDA cell required")
    return native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh campaign root required")
    first = check_cell(args.first, "low", 42)
    runner = PROJECT_ROOT / "scripts/run_oriented_cuda_cell_v1.py"
    source_before = file_record(runner)
    args.out.mkdir(parents=True)
    cells = [args.first]
    bindings = paths_from(first)
    for arm, seed in [("low", 1), ("low", 2), ("cross", 42), ("cross", 1), ("cross", 2)]:
        if file_record(runner) != source_before:
            raise RuntimeError("CUDA runner changed")
        cell_dir = args.out / f"{arm}_s{seed}"
        print(f"CUDA serial {arm} seed{seed}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.run_oriented_cuda_cell_v1",
                "--out",
                str(cell_dir),
                "--arm",
                arm,
                "--seed",
                str(seed),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        manifest = cell_dir / "summary.manifest.json"
        native = check_cell(manifest, arm, seed)
        if any(
            native["parameters"].get(key) != first["parameters"].get(key)
            for key in ("torch_version", "cuda_version", "cudnn_version")
        ):
            raise ValueError("CUDA environment differs between cells")
        cells.append(manifest)
        bindings.extend(paths_from(native))
    for path, (arm, seed) in zip(
        cells,
        [("low", 42), ("low", 1), ("low", 2), ("cross", 42), ("cross", 1), ("cross", 2)],
        strict=True,
    ):
        check_cell(path, arm, seed)
    write_experiment_manifest(
        args.out / "training-bound.manifest.json",
        experiment="oriented-uniform-cuda-serial-training",
        parameters=PROTOCOL | dict(device="cuda"),
        metrics=dict(
            training_complete=True,
            completed_cells=6,
            evaluation_complete=False,
            publication_ready=False,
        ),
        inputs=list(dict.fromkeys([Path(__file__), runner, *cells, *bindings])),
        outputs=cells,
    )


if __name__ == "__main__":
    main()
