"""Cross-platform transfer eval: score frozen + fine-tuned backbones on a held-out
pairs set from ANOTHER platform.

Answers the reviewer's non-VK external-validation ask: train on VK (our shipped
bb_facenet_seed42.pt), test on a Reddit pairs set (--pairs pairs_reddit.jsonl) ---
or the reverse (--pairs the VK pairs, --ckpts a Reddit-trained checkpoint). Mirrors
the internal our.* eval (cosine on aligned crops) but on an arbitrary pairs file,
and reports overall + large-gap (>=25y) ROC-AUC with 95% bootstrap CI, EER, TAR@FAR.

Prereq: build the Reddit dataset under a separate data-root first (see docs/Protocol.md
2026-06-23 Reddit entry). Run every Reddit step under ``ENV_FOR_DYNACONF=reddit`` so the
standard pipeline (ingest_reddit -> preprocess -> build_groups -> cluster_persons -> dedup
-> build_pairs -> split) writes to ``data_reddit/`` without touching VK; the VK checkpoint
in the shared ``models/`` is reused. Then, also under the reddit env so crops resolve:

    ENV_FOR_DYNACONF=reddit uv run python scripts/eval_cross_platform.py \
        --ckpts models/bb_facenet_seed42.pt --tag vk2reddit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.common.manifest import write_experiment_manifest
from age_gap.evaluation.metrics import bootstrap_auc_ci, eer, roc_auc, tar_at_far
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import ImagePairDataset, _bb_prep, load_finetuned

DEV = torch_device()


def _score(backbone: torch.nn.Module, ds: ImagePairDataset, batch_size: int = 64) -> np.ndarray:
    backbone.eval()
    loader = DataLoader(ds, batch_size=batch_size)
    out: list[float] = []
    with torch.no_grad():
        for ta, tb, _y, _w in loader:
            za, zb = backbone(ta.to(DEV)), backbone(tb.to(DEV))
            out.extend((za * zb).sum(dim=-1).cpu().tolist())
    return np.asarray(out)


def _metrics(s: np.ndarray, y: np.ndarray, gaps: np.ndarray, thr: int = 25) -> dict:
    lo, hi = bootstrap_auc_ci(s, y, n_boot=2000, alpha=0.05, seed=0)
    m: dict = {
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "overall_auc": round(float(roc_auc(s, y)), 4),
        "overall_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "eer": round(float(eer(s, y)), 4),
        "tar@1e-2": round(float(tar_at_far(s, y, 0.01)), 4),
    }
    mask = (y == 0) | ((y == 1) & (gaps >= thr))
    if mask.any() and (y[mask] == 1).any() and (y[mask] == 0).any():
        slo, shi = bootstrap_auc_ci(s[mask], y[mask], n_boot=2000, alpha=0.05, seed=0)
        m["large_gap_auc"] = round(float(roc_auc(s[mask], y[mask])), 4)
        m["large_gap_ci95"] = [round(float(slo), 4), round(float(shi), 4)]
        m["large_gap_n_pos"] = int((y[mask] == 1).sum())
    return m


def _aggregate(models: dict[str, dict]) -> dict[str, dict[str, float | int]]:
    """Mean/std over fine-tuned checkpoints; per-checkpoint CIs remain authoritative."""
    rows = [value for key, value in models.items() if key != "frozen"]
    out: dict[str, dict[str, float | int]] = {}
    if not rows:
        return out
    scalar_keys = sorted(
        key
        for key in set.intersection(*(set(row) for row in rows))
        if isinstance(rows[0][key], (int, float)) and not key.startswith("n_")
    )
    for key in scalar_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n_seeds": len(values),
        }
    for key in ("overall_ci95", "large_gap_ci95"):
        widths = [float(row[key][1] - row[key][0]) for row in rows if key in row]
        if widths:
            out[f"{key}_width"] = {
                "mean": float(np.mean(widths)),
                "max": float(np.max(widths)),
                "n_seeds": len(widths),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-platform transfer eval on a held-out pairs set")
    ap.add_argument("--pairs", default=str(data_path("data_dir", "processed", "pairs.jsonl")),
                    help="held-out pairs (under reddit env -> data_reddit/processed/pairs.jsonl)")
    ap.add_argument("--split", default=None, help="None=all pairs (cross-platform); or test/val/train")
    ap.add_argument("--backbone", default="facenet")
    ap.add_argument(
        "--ckpts", nargs="*",
        default=[str(data_path("models_dir", "bb_facenet_seed42.pt"))],
        help="fine-tuned checkpoints to compare against frozen",
    )
    ap.add_argument("--tag", default="vk2reddit")
    args = ap.parse_args()

    if not Path(args.pairs).exists():
        raise SystemExit(
            f"pairs file not found: {args.pairs}\n"
            "Build the cross-platform pairs first (see this script's docstring)."
        )

    models: dict[str, torch.nn.Module] = {
        "frozen": make_backbone(args.backbone, pretrained=True).to(DEV).eval()
    }
    for c in args.ckpts:
        models[Path(c).stem] = load_finetuned(Path(c), DEV)

    results: dict[str, object] = {"pairs": args.pairs, "split": args.split, "models": {}}
    model_results: dict[str, dict] = results["models"]  # type: ignore[assignment]
    for name, bb in models.items():
        crops_dir = getattr(bb, "crops_dir", "faces")
        ds = ImagePairDataset(
            split=args.split, pairs_file=args.pairs, preprocess=_bb_prep(bb), crops_dir=crops_dir
        )
        if len(ds) == 0:
            raise SystemExit(f"{name}: pairs set empty (check crops exist for {args.pairs})")
        m = _metrics(_score(bb, ds), np.asarray(ds.labels), np.asarray(ds.gaps))
        model_results[name] = m
        print(name, m)

    results["fine_tuned_aggregate"] = _aggregate(model_results)

    dst = data_path("metrics_dir", f"cross_platform_{args.tag}.json")
    dst.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        dst.with_suffix(".manifest.json"),
        experiment="cross-platform-transfer",
        parameters={
            "tag": args.tag,
            "split": args.split,
            "backbone": args.backbone,
            "large_gap_threshold": 25,
            "bootstrap_resamples": 2000,
            "checkpoint_count": len(args.ckpts),
        },
        metrics=results,
        inputs=[Path(args.pairs), *(Path(item) for item in args.ckpts)],
        outputs=[dst],
    )
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
