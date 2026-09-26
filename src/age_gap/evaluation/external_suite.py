"""Единая оценка backbone на наборе бенчмарков (для ablation и synthetic-baseline)."""

from __future__ import annotations

from pathlib import Path

import torch

from age_gap.common.io import data_path
from age_gap.evaluation.benchmark_external import evaluate_bin, evaluate_cacd_vs, evaluate_lfw
from age_gap.evaluation.fgnet import evaluate as eval_fgnet
from age_gap.training.finetune import evaluate_our_split

METRICS = [
    "LFW.acc",
    "agedb_30.acc",
    "agedb_30.roc",
    "calfw.acc",
    "calfw.roc",
    "cacd_vs.acc",
    "cacd_vs.roc",
    "cacd_vs.eer",
    "cacd_vs.tar@far=0.01",
    "cacd_vs.tar@far=0.001",
    "fgnet.roc",
    "fgnet.large_gap",
    "our.overall",
    "our.25+",
]


def eval_all(
    backbone: torch.nn.Module, device: str, pairs_file: str | None = None
) -> dict[str, float]:
    """LFW + AgeDB-30/CALFW (если есть .bin) + FG-NET (если есть кэш) + наш test."""
    out: dict[str, float] = {}
    out["LFW.acc"] = evaluate_lfw(backbone, device)["accuracy_10fold"]
    ext = Path(str(data_path("data_dir", "external")))
    for name in ("agedb_30", "calfw"):
        p = ext / f"{name}.bin"
        if p.exists():
            r = evaluate_bin(backbone, device, p)
            out[f"{name}.acc"] = r["accuracy_10fold"]
            out[f"{name}.roc"] = r["roc_auc"]
    cacd = ext / "cacd_vs_aligned.npz"
    if cacd.exists():
        r = evaluate_cacd_vs(backbone, device, cacd)
        out["cacd_vs.acc"] = r["accuracy_10fold"]
        out["cacd_vs.roc"] = r["roc_auc"]
        out["cacd_vs.eer"] = r["eer"]
        out["cacd_vs.tar@far=0.01"] = r["tar@far=0.01"]
        out["cacd_vs.tar@far=0.001"] = r["tar@far=0.001"]
    if (ext / "fgnet_crops.npz").exists():
        fg = eval_fgnet(backbone, device)
        out["fgnet.roc"] = fg["roc_auc"]
        out["fgnet.large_gap"] = fg["large_gap_auc"]
    ours = evaluate_our_split(backbone, device, pairs_file=pairs_file)
    out["our.overall"] = ours.get("overall_auc", float("nan"))
    out["our.25+"] = ours.get("large_gap_auc", float("nan"))
    return out


def print_table(results: dict[str, dict[str, float]], labels: list[str]) -> None:
    print(f"\n{'metric':<18}" + "".join(f"{lab:>15}" for lab in labels))
    for m in METRICS:
        if not any(m in results[lab] for lab in labels):
            continue
        row = "".join(f"{results[lab].get(m, float('nan')):>15.4f}" for lab in labels)
        print(f"{m:<18}{row}")
