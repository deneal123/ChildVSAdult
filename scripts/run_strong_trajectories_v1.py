"""Aggregate verified completed native histories; validation loss remains unavailable."""

import argparse
import json
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs
from scripts.run_oriented_campaign import paths_from, verified
from scripts.strong_training_trajectory_v1 import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh destination required")
    natives = [verified(p, "adaface-fixed8-native-cuda-training-cell") for p in args.bindings]
    inputs = sorted(set([Path(__file__), *args.bindings,
                         PROJECT_ROOT / "scripts/strong_training_trajectory_v1.py",
                         PROJECT_ROOT / "scripts/run_oriented_campaign.py",
                         PROJECT_ROOT / "scripts/run_restricted_matched_campaign.py",
                         PROJECT_ROOT / "scripts/evaluate_oriented_cuda_v1.py",
                         PROJECT_ROOT / "src/age_gap/common/manifest.py",
                         *(p for m in natives for p in paths_from(m))]))
    before = [file_record(p) for p in inputs]
    cells = {}
    for native in natives:
        p, m = native["parameters"], native["metrics"]
        if (m.get("training_complete") is not True or p.get("device") != "cuda"
                or p.get("checkpoint_selection") != "last_epoch"
                or p.get("selected_epoch") != 8 or p.get("batchnorm_policy") != "frozen_all"):
            raise ValueError("completed fixed8 CUDA/last/frozen-BN contract required")
        key = f"{p['negative']}_{p['trainable_scope']}_lr{p['learning_rate']:g}_s{p['seed']}"
        if key in cells:
            raise ValueError("duplicate native cell")
        cells[key] = summarize(m["history"])
    if before != [file_record(p) for p in inputs]:
        raise RuntimeError("inputs changed during aggregation")
    args.out.mkdir(parents=True)
    summary = args.out / "summary.json"
    metrics = dict(execution_complete=True, publication_ready=False, cells=cells,
                   limitation="recorded completed histories only; incomplete matrix; no validation loss or mechanism decision")
    summary.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = args.out / "summary.manifest.json"
    write_experiment_manifest(target, experiment="strong-native-training-trajectories-v1",
                              parameters=dict(epochs=8, checkpoint_selection="last_epoch"),
                              metrics=metrics, inputs=inputs, outputs=[summary])
    validate_written_inputs(target, before)
    print(f"completed verified trajectory aggregation: {len(cells)} cells", flush=True)


if __name__ == "__main__":
    main()
