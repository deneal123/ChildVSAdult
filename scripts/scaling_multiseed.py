"""Scaling law with per-point multi-seed CI bands.

For each train-identity fraction, materialize an immutable seed-42 subsample + val/test split, then
fine-tune FaceNet with 3 training seeds and report mean +/- std of FG-NET large-gap
and our.25+ -> confidence bands for the scaling figure. The canonical pairs file is never rewritten.

    uv run python scripts/scaling_multiseed.py

GPU, 12 retrains.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.common.manifest import write_experiment_manifest
from age_gap.datasets.splits import run as split_run
from age_gap.evaluation.external_suite import eval_all
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import finetune, load_finetuned

FRACS = [0.10, 0.25, 0.50, 1.0]
SEEDS = [42, 1, 2]
KEYS = ("fgnet.large_gap", "our.25+")


def main() -> None:
    import torch

    device = torch_device()
    models = data_path("models_dir")
    dst = data_path("metrics_dir", "scaling_multiseed.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    canonical_pairs = Path(str(data_path("data_dir", "processed", "pairs.jsonl")))
    experiment_dir = Path(str(data_path("data_dir", "processed", "experiments")))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    # Resume: keep already-computed fractions (the sweep is long and the laptop GPU can throw
    # a transient CUDA illegal-access after hours of sequential training).
    out: dict = json.loads(dst.read_text(encoding="utf-8")) if dst.exists() else {}
    out["schema_version"] = 2
    out.setdefault("fracs", {})
    if "frozen" not in out:
        frozen = eval_all(make_backbone("facenet", pretrained=True).to(device).eval(), device)
        out["frozen"] = {k: float(frozen.get(k, float("nan"))) for k in KEYS}
    fraction_protocols: list[Path] = []
    for f in FRACS:
        fraction_tag = str(f).replace(".", "p")
        fraction_pairs = experiment_dir / f"pairs_scaling_f{fraction_tag}.jsonl"
        fraction_splits = experiment_dir / f"group_splits_scaling_f{fraction_tag}.jsonl"
        fraction_protocols.extend([fraction_pairs, fraction_splits])
        cached = out["fracs"].get(f"{f}")
        expected_ckpts = [
            Path(str(models / f"bb_facenet_scale_f{fraction_tag}_s{s}.pt")) for s in SEEDS
        ]
        if (
            cached
            and cached.get("checkpoints")
            and fraction_pairs.is_file()
            and fraction_splits.is_file()
            and all(path.is_file() for path in expected_ckpts)
        ):
            print(f"frac={f}: cached, skip", flush=True)
            continue
        split_run(
            pairs_file=str(canonical_pairs),
            pairs_out=str(fraction_pairs),
            split_map_out=str(fraction_splits),
            neg_per_pos=1.0,
            seed=42,
            train_frac=f,
        )
        per: list[dict] = []
        for s, checkpoint in zip(SEEDS, expected_ckpts, strict=True):
            ck = finetune(
                epochs=10,
                lr=3e-5,
                trainable_scope="head",
                backbone_name="facenet",
                seed=s,
                ckpt_out=checkpoint,
                pairs_file=str(fraction_pairs),
            )
            per.append(
                eval_all(load_finetuned(ck, device), device, pairs_file=str(fraction_pairs))
            )
            if torch.cuda.is_available():  # release GPU state between trainings
                torch.cuda.empty_cache()
        metric_rows = {
            k: {
                "mean": float(np.mean([p[k] for p in per])),
                "std": float(np.std([p[k] for p in per])),
                "vals": [float(p[k]) for p in per],
            }
            for k in KEYS
        }
        out["fracs"][f"{f}"] = {
            "seeds": SEEDS,
            "checkpoints": [path.name for path in expected_ckpts],
            "pairs_file": fraction_pairs.name,
            "split_map": fraction_splits.name,
            **metric_rows,
        }
        dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(
            f"frac={f}: "
            + " ".join(
                f"{k}={out['fracs'][f'{f}'][k]['mean']:.4f}+/-{out['fracs'][f'{f}'][k]['std']:.4f}"
                for k in KEYS
            ),
            flush=True,
        )
    all_checkpoints = [
        Path(str(models / name))
        for row in out["fracs"].values()
        for name in row.get("checkpoints", [])
    ]
    write_experiment_manifest(
        dst.with_suffix(".manifest.json"),
        experiment="training-data-scaling-multiseed",
        parameters={
            "backbone": "facenet",
            "fractions": FRACS,
            "seeds": SEEDS,
            "epochs": 10,
            "learning_rate": 3e-5,
            "trainable_scope": "head",
            "split_seed": 42,
            "negative_per_positive": 1.0,
        },
        metrics=out,
        inputs=[canonical_pairs, Path(str(data_path("metrics_dir", "model_inventory.json")))],
        outputs=[dst, *fraction_protocols, *all_checkpoints],
    )
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
