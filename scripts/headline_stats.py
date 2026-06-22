"""In-text statistics for the paper: ROC-AUC + 95% bootstrap CI, EER and TAR@FAR
for frozen vs. +pairs FaceNet on the external cross-age benchmarks and the
internal test. For the internal test we report BOTH a pair-level and an
IDENTITY-LEVEL bootstrap (resampling persons, not pairs), since internal pairs
share identities and pair-level CIs are anti-conservative.

    uv run python scripts/headline_stats.py

Writes docs/headline_stats.json and prints a summary. No training; evaluates the
already-trained checkpoints (frozen + models/bb_facenet_seed42.pt).
"""

from __future__ import annotations

import json
from collections import defaultdict

import cv2
import numpy as np
import torch

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.schemas import Pair
from age_gap.evaluation import benchmark_external as bx
from age_gap.evaluation import fgnet as fg
from age_gap.evaluation.metrics import bootstrap_auc_ci, eer, roc_auc, tar_at_far
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import _bb_prep, _crop_path, load_finetuned

DEV = torch_device()
NBOOT = 2000


def _r(x: float, n: int = 4) -> float:
    return round(float(x), n)


def _pair_ci(scores: np.ndarray, labels: np.ndarray) -> list[float]:
    lo, hi = bootstrap_auc_ci(np.asarray(scores), np.asarray(labels), n_boot=NBOOT, alpha=0.05, seed=0)
    return [_r(lo), _r(hi)]


def _ext_row(scores: np.ndarray, labels: np.ndarray) -> dict:
    s, y = np.asarray(scores), np.asarray(labels)
    return {
        "auc": _r(roc_auc(s, y)),
        "ci95": _pair_ci(s, y),
        "eer": _r(eer(s, y)),
        "tar@far1e-2": _r(tar_at_far(s, y, 0.01)),
        "tar@far1e-3": _r(tar_at_far(s, y, 0.001)),
        "n": int(len(y)),
    }


def _identity_ci(scores: np.ndarray, labels: np.ndarray, persons: np.ndarray) -> list[float]:
    """Bootstrap by resampling persons (blocks keyed on person of endpoint A)."""
    rng = np.random.default_rng(0)
    by_p: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(persons.tolist()):
        by_p[p].append(i)
    uniq = np.asarray(list(by_p.keys()), dtype=object)
    blocks = {p: np.asarray(idx) for p, idx in by_p.items()}
    aucs: list[float] = []
    for _ in range(NBOOT):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([blocks[p] for p in samp])
        yb = labels[idx]
        if yb.min() == yb.max():
            continue
        aucs.append(roc_auc(scores[idx], yb))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return [_r(lo), _r(hi)]


def _score_internal(model: torch.nn.Module, pcmap: dict[str, str]):
    """Score the internal test split; return (scores, labels, gaps, person_of_A)."""
    prep = _bb_prep(model)
    crops_dir = getattr(model, "crops_dir", "faces")
    rows = []  # (crop_a, crop_b, label, gap, person_a)
    pairs_file = str(data_path("data_dir", "processed", "pairs.jsonl"))
    for r in read_jsonl(pairs_file):
        if r.get("split") != "test":
            continue
        p = Pair.from_dict(r)
        ca, cb = _crop_path(p.face_a, crops_dir), _crop_path(p.face_b, crops_dir)
        if ca.exists() and cb.exists():
            gap = p.age_gap if p.age_gap is not None else -1
            person_a = pcmap.get(p.identity_group_a, p.identity_group_a)
            rows.append((ca, cb, p.label, gap, person_a))
    scores: list[float] = []
    model.eval()
    bs = 128
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            chunk = rows[i : i + bs]
            ta = torch.from_numpy(np.stack([prep(cv2.imread(str(c[0]))) for c in chunk]))
            tb = torch.from_numpy(np.stack([prep(cv2.imread(str(c[1]))) for c in chunk]))
            za, zb = model(ta.to(DEV)), model(tb.to(DEV))
            scores.extend((za * zb).sum(dim=-1).cpu().tolist())
    return (
        np.asarray(scores),
        np.asarray([r[2] for r in rows]),
        np.asarray([r[3] for r in rows]),
        np.asarray([r[4] for r in rows], dtype=object),
    )


def main() -> None:
    print(f"device={DEV}, n_boot={NBOOT}")
    frozen = make_backbone("facenet", pretrained=True).to(DEV).eval()
    frozen.crops_dir = "faces"  # type: ignore[attr-defined]
    tuned = load_finetuned(resolve_path("models", "bb_facenet_seed42.pt"), DEV)
    models = {"frozen": frozen, "+pairs": tuned}
    out: dict = {}

    # --- FG-NET (overall + large-gap >=25) ---
    a, b, issame, gaps = fg.load_pairs()
    lg = (issame == 0) | ((issame == 1) & (gaps >= 25))
    for mname, model in models.items():
        sc = bx.pair_scores(model, a, b, DEV, rgb=False)
        out.setdefault("fgnet.overall", {})[mname] = _ext_row(sc, issame)
        out.setdefault("fgnet.large_gap", {})[mname] = _ext_row(sc[lg], issame[lg])
        print(f"  fgnet {mname}: overall={out['fgnet.overall'][mname]['auc']} "
              f"large_gap={out['fgnet.large_gap'][mname]['auc']}")

    # --- LFW / AgeDB-30 / CALFW ---
    benches = [
        ("LFW", bx.load_lfw(), True),
        ("AgeDB-30", bx.load_bin(resolve_path("data", "external", "agedb_30.bin")), False),
        ("CALFW", bx.load_bin(resolve_path("data", "external", "calfw.bin")), False),
    ]
    for bname, (aa, bb_, iss), rgb in benches:
        for mname, model in models.items():
            sc = bx.pair_scores(model, aa, bb_, DEV, rgb=rgb)
            out.setdefault(bname, {})[mname] = _ext_row(sc, iss)
        print(f"  {bname}: frozen={out[bname]['frozen']['auc']} +pairs={out[bname]['+pairs']['auc']}")

    # --- Internal test (pair-level AND identity-level CI) ---
    pcmap = {
        r["identity_group_id"]: r.get("person_id", r["identity_group_id"])
        for r in read_jsonl(str(data_path("data_dir", "processed", "person_clusters.jsonl")))
    }
    for mname, model in models.items():
        sc, y, g, persons = _score_internal(model, pcmap)
        m25 = (y == 0) | ((y == 1) & (g >= 25))
        out.setdefault("our.overall", {})[mname] = {
            "auc": _r(roc_auc(sc, y)), "pair_ci95": _pair_ci(sc, y),
            "identity_ci95": _identity_ci(sc, y, persons), "n": int(len(y)),
        }
        out.setdefault("our.25+", {})[mname] = {
            "auc": _r(roc_auc(sc[m25], y[m25])), "pair_ci95": _pair_ci(sc[m25], y[m25]),
            "identity_ci95": _identity_ci(sc[m25], y[m25], persons[m25]), "n": int(m25.sum()),
        }
        print(f"  internal {mname}: overall={out['our.overall'][mname]['auc']} "
              f"25+={out['our.25+'][mname]['auc']}")

    dst = resolve_path("docs", "headline_stats.json")
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {dst}")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
