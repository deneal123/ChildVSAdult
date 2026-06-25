"""CLI: hard-negative mining, multi-seed (removes the single-run caveat).

Mines hard negatives into pairs.jsonl, re-splits (``splits.run`` keeps them in train and balances
negatives 1:1), trains ``+pairs+hard-neg`` on seeds 42/1/2, and evaluates the EXTERNAL benchmarks
(FG-NET/LFW/AgeDB-30/CALFW -- the only ones the hard-neg table reports; they are independent of
pairs.jsonl). Reports mean +/- std. The canonical ``pairs.jsonl`` and ``group_splits.jsonl`` are
backed up and RESTORED in a finally block, so the main data is never left mutated.

    uv run python scripts/hardneg_multiseed.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.datasets.hard_negatives import run as mine_hard
from age_gap.datasets.splits import run as split_run
from age_gap.evaluation.external_suite import eval_all
from age_gap.training.finetune import finetune, load_finetuned

SEEDS = [42, 1, 2]
# External metrics only -- these are the ones tab:hardneg reports and are independent of pairs.jsonl.
KEYS = ["fgnet.large_gap", "fgnet.roc", "LFW.acc", "agedb_30.roc", "calfw.roc"]


def main() -> None:
    device = torch_device()
    models = data_path("models_dir")
    pairs = Path(str(data_path("data_dir", "processed", "pairs.jsonl")))
    splits = Path(str(data_path("splits_dir", "group_splits.jsonl")))
    bak_p = pairs.with_suffix(".jsonl.hnbak")
    bak_s = splits.with_suffix(".jsonl.hnbak")
    shutil.copy(str(pairs), str(bak_p))
    shutil.copy(str(splits), str(bak_s))

    try:
        added = mine_hard(top_k=5)
        print(f"mined hard negatives: {added}")
        split_run()  # keeps hard negatives in train, balances negatives 1:1

        per_seed: list[dict] = []
        for s in SEEDS:
            ck = finetune(seed=s, ckpt_out=Path(str(models / f"bb_facenet_hardneg_s{s}.pt")))
            res = eval_all(load_finetuned(ck, device), device)
            per_seed.append({k: res.get(k) for k in KEYS})
            print(f"seed {s}: fgnet.large_gap = {res.get('fgnet.large_gap'):.4f}")

        agg = {
            "seeds": SEEDS,
            "mean": {k: float(np.mean([r[k] for r in per_seed])) for k in KEYS},
            "std": {k: float(np.std([r[k] for r in per_seed])) for k in KEYS},
            "per_seed": per_seed,
        }
        dst = data_path("metrics_dir", "hardneg_multiseed.json")
        dst.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print(f"wrote {dst}")
        m, sd = agg["mean"]["fgnet.large_gap"], agg["std"]["fgnet.large_gap"]
        print(f"FG-NET large-gap (+pairs+hard-neg): {m:.4f} +/- {sd:.4f}  (single-run was 0.866)")
    finally:
        shutil.move(str(bak_p), str(pairs))
        shutil.move(str(bak_s), str(splits))
        print("restored canonical pairs.jsonl + group_splits.jsonl")


if __name__ == "__main__":
    main()
