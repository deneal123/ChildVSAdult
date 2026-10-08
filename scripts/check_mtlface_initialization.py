"""Bind a strict local initialization check; no model inference or training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.build_mtlface_recognition_v2 import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    sources = [
        args.weights,
        Path(__file__),
        *sorted((PROJECT_ROOT / "src/age_gap").rglob("*.py")),
        PROJECT_ROOT / "src/age_gap/settings/settings.toml",
        *[
            PROJECT_ROOT / "scripts" / name
            for name in (
                "build_mtlface_recognition_v2.py",
                "mtlface_components_v2.py",
                "mtlface_recognition_v2.py",
                "mtlface_training_v2.py",
            )
        ],
    ]
    before = [file_record(path) for path in sources]
    torch.set_num_threads(1)
    model, head, initialization = build_model(args.weights, args.classes, seed=42, scope="head")
    metrics = dict(
        initialization=initialization,
        identity_classes=head.classes,
        model_parameters=sum(p.numel() for p in model.parameters()),
        initialization_complete=True,
        real_crop_smoke_complete=False,
        training_complete=False,
        publication_ready=False,
    )
    if before != [file_record(path) for path in sources]:
        raise RuntimeError("initialization sources changed")
    args.out.mkdir(parents=True)
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        args.out / "summary.manifest.json",
        experiment="mtlface-common-recognition-initialization",
        parameters=dict(
            seed=42,
            scope="head",
            device="cpu",
            threads=1,
            classes=args.classes,
            full_joint_fas=False,
        ),
        metrics=metrics,
        inputs=sources,
        outputs=[summary],
    )
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
