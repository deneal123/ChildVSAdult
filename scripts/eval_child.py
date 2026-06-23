"""Child<->adult cross-age verification on FG-NET (independent, non-VK benchmark).

A child<->adult-specific evaluation the review asked for, on data fully independent
of our VK source. Positives: within-subject FG-NET pairs where one face is a child
(age < 13) and the other an adult (age > 25). Negatives: balanced cross-subject
pairs. Reports frozen vs. +pairs FaceNet ROC-AUC (95% bootstrap CI), EER, TAR@FAR.

    uv run python scripts/eval_child.py
"""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, resolve_path
from age_gap.evaluation.benchmark_external import pair_scores
from age_gap.evaluation.metrics import bootstrap_auc_ci, eer, roc_auc, tar_at_far
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import load_finetuned

DEV = torch_device()
CHILD_MAX, ADULT_MIN = 13, 25


def build_pairs(seed: int = 42):
    d = np.load(str(data_path("data_dir", "external", "fgnet_crops.npz")))
    crops, subj, ages = d["crops"], d["subjects"], d["ages"]
    by: dict[int, list[int]] = {}
    for i, s in enumerate(subj.tolist()):
        by.setdefault(s, []).append(i)
    pos: list[tuple[int, int]] = []
    for idxs in by.values():
        for a, b in combinations(idxs, 2):
            la, lb = int(ages[a]), int(ages[b])
            if (la < CHILD_MAX and lb > ADULT_MIN) or (lb < CHILD_MAX and la > ADULT_MIN):
                pos.append((a, b))
    rng = np.random.default_rng(seed)
    n = len(subj)
    neg: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(neg) < len(pos):
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if subj[i] == subj[j]:
            continue
        k = (min(i, j), max(i, j))
        if k in seen:
            continue
        seen.add(k)
        neg.append((i, j))
    a_img = [crops[i] for i, _ in pos] + [crops[i] for i, _ in neg]
    b_img = [crops[j] for _, j in pos] + [crops[j] for _, j in neg]
    y = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    return a_img, b_img, y, len(pos)


def main() -> None:
    a_img, b_img, y, npos = build_pairs()
    print(f"FG-NET child<->adult: positives={npos}, negatives={len(y) - npos}")
    models = {
        "frozen": make_backbone("facenet", pretrained=True).to(DEV).eval(),
        "+pairs": load_finetuned(resolve_path("models", "bb_facenet_seed42.pt"), DEV),
    }
    out: dict = {"n_pos": npos, "n_neg": int(len(y) - npos)}
    for name, m in models.items():
        s = pair_scores(m, a_img, b_img, DEV, rgb=False)
        lo, hi = bootstrap_auc_ci(s, y, n_boot=2000, alpha=0.05, seed=0)
        out[name] = {
            "auc": round(float(roc_auc(s, y)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "eer": round(float(eer(s, y)), 4),
            "tar@far1e-2": round(float(tar_at_far(s, y, 0.01)), 4),
        }
        print(name, out[name])
    data_path("metrics_dir", "child_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote metrics/child_eval.json")


if __name__ == "__main__":
    main()
