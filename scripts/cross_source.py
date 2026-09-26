"""CLI: cross-source transfer Reddit->VK (complements the VK->Reddit cross-platform test).

Evaluates a FaceNet trained on the independent Reddit ``then/now'' pairs (models/bb_reddit_src.pt,
produced by ``ENV_FOR_DYNACONF=reddit finetune_facenet.py --ckpt models/bb_reddit_src.pt``) on the
VK internal 25+ test and the external FG-NET/AgeDB/CALFW suite, against the frozen baseline. Run
under the DEFAULT (VK) env. Question: does supervision mined from an independent non-VK source
transfer back to the VK large-gap regime, mirroring the VK->Reddit direction?

    uv run python scripts/cross_source.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.common.manifest import write_experiment_manifest
from age_gap.evaluation.external_suite import eval_all, print_table
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import load_finetuned


def _aggregate(rows: dict[str, dict[str, float]]) -> dict[str, dict[str, float | int]]:
    common = set.intersection(*(set(row) for row in rows.values())) if rows else set()
    out: dict[str, dict[str, float | int]] = {}
    for metric in sorted(common):
        values = np.asarray([row[metric] for row in rows.values()], dtype=float)
        out[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n_seeds": len(values),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Reddit-trained checkpoints on VK/external data")
    parser.add_argument(
        "--ckpts",
        nargs="+",
        default=["models/bb_reddit_src_s42.pt", "models/bb_reddit_src_s1.pt", "models/bb_reddit_src_s2.pt"],
    )
    parser.add_argument("--tag", default="reddit2vk")
    args = parser.parse_args()

    device = torch_device()
    checkpoints = [Path(item) for item in args.ckpts]
    missing = [item for item in checkpoints if not item.exists()]
    if missing:
        raise SystemExit(
            f"missing checkpoints: {', '.join(map(str, missing))}; train each under "
            "ENV_FOR_DYNACONF=reddit with scripts/finetune_facenet.py --seed SEED"
        )

    frozen = eval_all(make_backbone("facenet", pretrained=True).to(device).eval(), device)
    tuned = {
        checkpoint.stem: eval_all(load_finetuned(checkpoint, device), device)
        for checkpoint in checkpoints
    }
    table = {"frozen": frozen, **tuned}
    print_table(table, list(table))
    results = {
        "direction": "reddit-to-vk",
        "frozen": frozen,
        "fine_tuned": tuned,
        "fine_tuned_aggregate": _aggregate(tuned),
    }

    dst = data_path("metrics_dir", f"cross_source_{args.tag}.json")
    dst.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    pairs = data_path("data_dir", "processed", "pairs.jsonl")
    write_experiment_manifest(
        dst.with_suffix(".manifest.json"),
        experiment="cross-source-transfer",
        parameters={"direction": "reddit-to-vk", "checkpoint_count": len(checkpoints)},
        metrics=results,
        inputs=[pairs, *checkpoints],
        outputs=[dst],
    )
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
