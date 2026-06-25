"""CLI: cross-source transfer Reddit->VK (complements the VK->Reddit cross-platform test).

Evaluates a FaceNet trained on the independent Reddit ``then/now'' pairs (models/bb_reddit_src.pt,
produced by ``ENV_FOR_DYNACONF=reddit finetune_facenet.py --ckpt models/bb_reddit_src.pt``) on the
VK internal 25+ test and the external FG-NET/AgeDB/CALFW suite, against the frozen baseline. Run
under the DEFAULT (VK) env. Question: does supervision mined from an independent non-VK source
transfer back to the VK large-gap regime, mirroring the VK->Reddit direction?

    uv run python scripts/cross_source.py
"""

from __future__ import annotations

import json
from pathlib import Path

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.evaluation.external_suite import eval_all, print_table
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import load_finetuned


def main() -> None:
    device = torch_device()
    models = data_path("models_dir")
    ck = Path(str(models / "bb_reddit_src.pt"))
    if not ck.exists():
        raise SystemExit(
            f"missing {ck}; train it first: "
            "ENV_FOR_DYNACONF=reddit uv run python scripts/finetune_facenet.py "
            "--ckpt models/bb_reddit_src.pt"
        )

    results = {
        "frozen": eval_all(make_backbone("facenet", pretrained=True).to(device).eval(), device),
        "+reddit_src": eval_all(load_finetuned(ck, device), device),
    }
    order = ["frozen", "+reddit_src"]
    print_table(results, order)

    dst = data_path("metrics_dir", "cross_source_reddit2vk.json")
    dst.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
