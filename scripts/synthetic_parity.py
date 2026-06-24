"""CLI: parity-controlled synthetic-ageing benchmark (reviewer revision).

The review notes that synthetic ageing shows at least modest gains in long-gap settings, so a
strong "real pairs > synthetic ageing" claim needs PARITY. FRAN's default ages each crop to a
fixed wide gap (8-22 -> 55-78); here we instead age the synthetic positive to the SAME age-gap
distribution as the real mined training pairs the contrastive model trains on, removing the gap
mismatch as a confound. Same identities, same pair count, same weak backbone, same trainable scope
and same external evaluation as ``+real`` and the original fixed-gap ``+syn_fran``. Question: does
matching the gap rescue synthetic ageing, or do real pairs still dominate?

    uv run python scripts/synthetic_parity.py --epochs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, read_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import Pair
from age_gap.evaluation.external_suite import eval_all, print_table
from age_gap.models.aging import FRANAging
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import load_finetuned
from age_gap.training.synthetic import train_synthetic

log = get_logger(__name__)


def _real_train_gaps() -> list[int]:
    """Age-gaps of the real positive TRAIN pairs --- the distribution +pairs trains on."""
    rows = read_jsonl(str(data_path("data_dir", "processed", "pairs.jsonl")))
    return [
        p.age_gap
        for p in (Pair.from_dict(r) for r in rows)
        if p.split == "train" and p.label == 1 and p.age_gap is not None
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parity-controlled synthetic ageing (gap-matched)")
    parser.add_argument("--backbone", default="facenet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--skip-train", action="store_true", help="reuse facenet_syn_fran_parity.pt")
    args = parser.parse_args()

    device = torch_device()
    models = data_path("models_dir")
    bb = args.backbone

    gaps = _real_train_gaps()
    if not gaps:
        raise RuntimeError("No real train-pair age-gaps found in pairs.jsonl")
    log.info(
        "real train-pair gaps: n=%d, median=%.1f, p90=%.1f, max=%d",
        len(gaps), float(np.median(gaps)), float(np.percentile(gaps, 90)), max(gaps),
    )

    ckpt = Path(str(models / "facenet_syn_fran_parity.pt"))
    if not args.skip_train:
        ckpt = train_synthetic(epochs=args.epochs, aging=FRANAging(real_gaps=gaps), ckpt_out=ckpt)

    results = {
        "frozen": eval_all(make_backbone(bb, pretrained=True).to(device).eval(), device),
        "+real": eval_all(load_finetuned(Path(str(models / f"bb_{bb}_seed42.pt")), device), device),
    }
    order = ["frozen", "+real"]
    fixed = Path(str(models / "facenet_syn_fran.pt"))
    if fixed.exists():
        results["+syn_fran_fixed"] = eval_all(load_finetuned(fixed, device), device)
        order.append("+syn_fran_fixed")
    results["+syn_fran_parity"] = eval_all(load_finetuned(ckpt, device), device)
    order.append("+syn_fran_parity")

    print_table(results, order)
    dst = data_path("metrics_dir", "synthetic_parity.json")
    payload = {
        "gap_stats": {
            "n": len(gaps),
            "median": float(np.median(gaps)),
            "p90": float(np.percentile(gaps, 90)),
            "max": int(max(gaps)),
        },
        "models": results,
    }
    dst.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
