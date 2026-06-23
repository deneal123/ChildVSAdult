"""Scaling law with per-point multi-seed CI bands.

For each train-identity fraction, fix the (seed-42) subsample + val/test split, then
fine-tune FaceNet with 3 training seeds and report mean +/- std of FG-NET large-gap
and our.25+ -> confidence bands for the scaling figure. Restores the full standard
split in a finally-block.

    uv run python scripts/scaling_multiseed.py

WARNING: rewrites data/processed/pairs.jsonl during the sweep; GPU, ~12 retrains.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.datasets.splits import run as split_run
from age_gap.evaluation.external_suite import eval_all
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import finetune, load_finetuned

DEV = torch_device()
FRACS = [0.10, 0.25, 0.50, 1.0]
SEEDS = [42, 1, 2]
KEYS = ("fgnet.large_gap", "our.25+")


def main() -> None:
    import torch

    models = data_path("models_dir")
    dst = data_path("metrics_dir", "scaling_multiseed.json")
    # Resume: keep already-computed fractions (the sweep is long and the laptop GPU can throw
    # a transient CUDA illegal-access after hours of sequential training).
    out: dict = json.loads(dst.read_text(encoding="utf-8")) if dst.exists() else {}
    out.setdefault("fracs", {})
    if "frozen" not in out:
        frozen = eval_all(make_backbone("facenet", pretrained=True).to(DEV).eval(), DEV)
        out["frozen"] = {k: float(frozen.get(k, float("nan"))) for k in KEYS}
    try:
        for f in FRACS:
            if f"{f}" in out["fracs"]:
                print(f"frac={f}: cached, skip", flush=True)
                continue
            split_run(neg_per_pos=1.0, seed=42, train_frac=f)  # fixed subsample + val/test
            per: list[dict] = []
            for s in SEEDS:
                ck = finetune(
                    epochs=10, lr=3e-5, trainable_scope="head", backbone_name="facenet",
                    seed=s, ckpt_out=Path(str(models / f"bb_facenet_scale_s{s}.pt")),
                )
                per.append(eval_all(load_finetuned(ck, DEV), DEV))
                if torch.cuda.is_available():  # release GPU state between trainings
                    torch.cuda.empty_cache()
            out["fracs"][f"{f}"] = {
                k: {
                    "mean": float(np.mean([p[k] for p in per])),
                    "std": float(np.std([p[k] for p in per])),
                    "vals": [float(p[k]) for p in per],
                }
                for k in KEYS
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
    finally:
        split_run(neg_per_pos=1.0, seed=42)  # restore the full standard split
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
