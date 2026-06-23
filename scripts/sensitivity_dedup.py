"""Dedup-threshold sensitivity for the headline result.

Rebuilds positives at cos >= {0.95, 0.99} (0.97 is the reference), retrains the
+pairs FaceNet with the SAME config as the headline (head scope, lr 3e-5, 10
epochs, seed 42), and evaluates on the external + internal benchmarks. The
external metrics (FG-NET large-gap, LFW, AgeDB-30, CALFW) are split-independent
and thus directly comparable across thresholds; internal our.* uses each
threshold's own test split. Finally RESTORES the clean standard split (dedup
0.97 -> build_pairs -> split), in a finally-block so it restores even on error.

    uv run python scripts/sensitivity_dedup.py

WARNING: rewrites data/processed/pairs.jsonl during the sweep; do not run other
jobs that read the split concurrently. ~2 retrains (GPU, ~1.5h each).
"""

from __future__ import annotations

import json
import subprocess
import sys

from age_gap.common.device import torch_device
from age_gap.common.io import resolve_path
from age_gap.evaluation.external_suite import eval_all
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import finetune, load_finetuned

DEV = torch_device()
PY = sys.executable


def _cli(script: str, *args: str) -> None:
    subprocess.run([PY, str(resolve_path("scripts", script)), *map(str, args)], check=True)


def _rebuild(dedup_thr: str) -> None:
    """Re-curate at a given dedup threshold and re-split (clustering stays 0.85)."""
    _cli("dedup_faces.py", "--threshold", dedup_thr)
    _cli("build_pairs.py", "--neg-per-pos", "1", "--seed", "42")
    _cli("split.py", "--train", "0.7", "--val", "0.15", "--test", "0.15", "--seed", "42",
         "--neg-per-pos", "1.0")


def main() -> None:
    results: dict[str, dict] = {}
    # frozen baseline (constant) + the 0.97 reference from the existing headline checkpoint,
    # both evaluated BEFORE we touch the split.
    results["frozen"] = eval_all(make_backbone("facenet", pretrained=True).to(DEV).eval(), DEV)
    results["dedup_0.97"] = eval_all(
        load_finetuned(resolve_path("models", "bb_facenet_seed42.pt"), DEV), DEV
    )
    print(f"frozen fgnet.large_gap={results['frozen'].get('fgnet.large_gap'):.4f} "
          f"| 0.97 +pairs fgnet.large_gap={results['dedup_0.97'].get('fgnet.large_gap'):.4f}")

    try:
        for thr in ["0.95", "0.99"]:
            print(f"=== dedup {thr}: rebuild + retrain ===", flush=True)
            _rebuild(thr)
            ck = finetune(
                epochs=10, lr=3e-5, trainable_scope="head", backbone_name="facenet",
                seed=42, ckpt_out=resolve_path("models", f"sens_dedup_{thr}.pt"),
            )
            results[f"dedup_{thr}"] = eval_all(load_finetuned(ck, DEV), DEV)
            print(f"  dedup {thr}: +pairs fgnet.large_gap="
                  f"{results[f'dedup_{thr}'].get('fgnet.large_gap'):.4f}", flush=True)
    finally:
        print("=== RESTORE clean standard split (dedup 0.97) ===", flush=True)
        _rebuild("0.97")

    dst = resolve_path("docs", "sensitivity_dedup.json")
    dst.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {dst}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
