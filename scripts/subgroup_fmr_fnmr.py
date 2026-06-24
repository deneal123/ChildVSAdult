"""CLI: subgroup FMR/FNMR + per-stratum calibration (fairness beyond AUC).

Standard fairness-at-a-fixed-operating-point analysis for the internal test split. A single
GLOBAL decision threshold is fixed at FAR=1% on the FULL test set (per model: cosine scales
differ between frozen and +pairs, so each model keeps its own global FAR=1% threshold and that
one threshold is then applied unchanged across every subgroup). For each apparent stratum
(gender F / M; age bands 0-17 / 18-29 / 30-44 / 45+ by face_a's estimated age) we report, for
frozen and +pairs (bb_<bb>_seed42.pt):

- FMR  = negatives accepted at the threshold (false-match rate),
- FNMR = positives rejected at the threshold (false-non-match rate),
- n_pos / n_neg,
- a per-stratum calibration slope+intercept (logistic fit of label ~ +pairs score).

Apparent attributes come from insightface genderage (CPU); scoring reuses the fairness module's
single-pass face encoder so the pair set is identical for both models.

    uv run python scripts/subgroup_fmr_fnmr.py --backbone facenet --tuned models/bb_facenet_seed42.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, read_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import Pair
from age_gap.evaluation.fairness import _age_band, _encode_faces, compute_face_attributes
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import load_finetuned

log = get_logger(__name__)


def _threshold_at_far(scores: np.ndarray, labels: np.ndarray, far_target: float) -> float:
    """Global score threshold whose FAR (negatives accepted with score>=thr) <= far_target.

    Mirrors evaluation.metrics.tar_at_far's thresholding: take the negatives sorted descending,
    allow floor(far_target * n_neg) of them above the threshold.
    """
    neg = scores[labels == 0]
    if len(neg) == 0:
        return float("nan")
    neg_sorted = np.sort(neg)[::-1]
    allowed = int(np.floor(far_target * len(neg)))
    if allowed <= 0:
        # Strictly above the largest negative -> nextafter so that score>=thr excludes it.
        return float(np.nextafter(neg_sorted[0], np.inf))
    return float(neg_sorted[allowed - 1])


def _fmr_fnmr(scores: np.ndarray, labels: np.ndarray, thr: float) -> tuple[float, float]:
    """(FMR, FNMR) at threshold thr; accept iff score >= thr."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    fmr = float((neg >= thr).mean()) if len(neg) else float("nan")
    fnmr = float((pos < thr).mean()) if len(pos) else float("nan")
    return fmr, fnmr


def _calib_logistic(
    scores: np.ndarray, labels: np.ndarray, iters: int = 200, lr: float = 0.5
) -> tuple[float, float]:
    """Newton-IRLS logistic fit label ~ slope*score + intercept; returns (slope, intercept).

    Score is standardized internally for conditioning, then the fitted (slope, intercept) are
    mapped back to the raw-score scale so they are directly interpretable as a calibration line.
    Returns (nan, nan) if a stratum has only one class or is too small.
    """
    y = labels.astype(float)
    if len(y) < 10 or y.min() == y.max():
        return float("nan"), float("nan")
    mu, sd = float(scores.mean()), float(scores.std())
    if sd == 0.0:
        return float("nan"), float("nan")
    z = (scores - mu) / sd
    x = np.column_stack([z, np.ones_like(z)])  # [slope, intercept] on standardized score
    beta = np.zeros(2, dtype=float)
    for _ in range(iters):
        eta = x @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        grad = x.T @ (p - y)
        hess = (x * w[:, None]).T @ x + 1e-6 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        beta -= lr * step
        if np.max(np.abs(step)) < 1e-8:
            break
    slope_z, intercept_z = float(beta[0]), float(beta[1])
    # Undo standardization: eta = slope_z*(s-mu)/sd + intercept_z = (slope_z/sd)*s + (...).
    slope = slope_z / sd
    intercept = intercept_z - slope_z * mu / sd
    return slope, intercept


def _pair_cosines(
    emb: dict[str, np.ndarray], pairs: list[Pair]
) -> tuple[np.ndarray, np.ndarray, list[Pair]]:
    """Cosine (dot of L2-normed embeddings) per pair for which both faces were encoded."""
    scores: list[float] = []
    labels: list[int] = []
    kept: list[Pair] = []
    for p in pairs:
        if p.face_a in emb and p.face_b in emb:
            scores.append(float(emb[p.face_a] @ emb[p.face_b]))
            labels.append(int(p.label))
            kept.append(p)
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=np.int64), kept


def _strata_of(p: Pair, attrs: dict[str, tuple[str, int]]) -> list[str]:
    a = attrs.get(p.face_a)
    if a is None:
        return []
    return [f"gender:{a[0]}", f"age:{_age_band(a[1])}"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Subgroup FMR/FNMR + calibration at FAR=1%")
    parser.add_argument("--backbone", default="facenet")
    parser.add_argument("--tuned", default=None, help="default models/bb_<backbone>_seed42.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--far", type=float, default=0.01)
    args = parser.parse_args()

    device = torch_device()
    models = data_path("models_dir")
    bb = args.backbone
    tuned_ckpt = Path(args.tuned) if args.tuned else Path(str(models / f"bb_{bb}_seed42.pt"))

    pairs_file = str(data_path("data_dir", "processed", "pairs.jsonl"))
    pairs = [Pair.from_dict(r) for r in read_jsonl(pairs_file) if r.get("split") == args.split]
    face_ids = sorted({p.face_a for p in pairs} | {p.face_b for p in pairs})
    log.info("split=%s: pairs=%d, unique faces=%d", args.split, len(pairs), len(face_ids))

    # Apparent gender/age (CPU onnx; resumable cache).
    attrs = compute_face_attributes(face_ids, device="cpu")

    # Encode every face once with each backbone (identical pair set for frozen/tuned).
    frozen = make_backbone(bb, pretrained=True).to(device).eval()
    frozen.crops_dir = "faces"  # type: ignore[assignment]
    tuned = load_finetuned(tuned_ckpt, device)
    emb_f = _encode_faces(frozen, device, face_ids)
    emb_t = _encode_faces(tuned, device, face_ids)

    sf, lab_f, kept_f = _pair_cosines(emb_f, pairs)
    st, lab_t, kept_t = _pair_cosines(emb_t, pairs)
    # Both backbones use the same crops dir here -> identical kept pairs; assert to be safe.
    assert [p.pair_id for p in kept_f] == [p.pair_id for p in kept_t], "pair sets differ"
    labels = lab_t
    kept = kept_t

    # ONE global threshold per model at FAR=1% on the FULL test set (report both).
    thr_frozen = _threshold_at_far(sf, labels, args.far)
    thr_tuned = _threshold_at_far(st, labels, args.far)
    g_fmr_f, g_fnmr_f = _fmr_fnmr(sf, labels, thr_frozen)
    g_fmr_t, g_fnmr_t = _fmr_fnmr(st, labels, thr_tuned)
    log.info(
        "global FAR=%.3f thresholds: frozen=%.4f (FMR=%.4f,FNMR=%.4f), tuned=%.4f (FMR=%.4f,FNMR=%.4f)",
        args.far,
        thr_frozen,
        g_fmr_f,
        g_fnmr_f,
        thr_tuned,
        g_fmr_t,
        g_fnmr_t,
    )

    # Bucket pair indices by apparent stratum of the anchor (face_a).
    strata_idx: dict[str, list[int]] = {}
    for i, p in enumerate(kept):
        for s in _strata_of(p, attrs):
            strata_idx.setdefault(s, []).append(i)
    strata_idx["overall"] = list(range(len(kept)))

    result: dict[str, dict[str, Any]] = {}
    for stratum, idx in strata_idx.items():
        ii = np.asarray(idx, dtype=int)
        y = labels[ii]
        sfi, sti = sf[ii], st[ii]
        fmr_f, fnmr_f = _fmr_fnmr(sfi, y, thr_frozen)
        fmr_t, fnmr_t = _fmr_fnmr(sti, y, thr_tuned)
        slope, intercept = _calib_logistic(sti, y)  # calibration on +pairs scores
        result[stratum] = {
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
            "fmr_frozen": fmr_f,
            "fnmr_frozen": fnmr_f,
            "fmr_tuned": fmr_t,
            "fnmr_tuned": fnmr_t,
            "calib_slope": slope,
            "calib_intercept": intercept,
        }

    payload = {
        "far_target": args.far,
        "threshold_frozen": thr_frozen,
        "threshold_tuned": thr_tuned,
        "global_frozen": {"fmr": g_fmr_f, "fnmr": g_fnmr_f},
        "global_tuned": {"fmr": g_fmr_t, "fnmr": g_fnmr_t},
        "strata": result,
    }
    dst = data_path("metrics_dir", "subgroup_fmr_fnmr.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_table(args.far, thr_frozen, thr_tuned, result)
    print(f"\nwrote {dst}")


def _print_table(
    far: float, thr_f: float, thr_t: float, result: dict[str, dict[str, Any]]
) -> None:
    order = ["overall", "gender:F", "gender:M", "age:0-17", "age:18-29", "age:30-44", "age:45+"]
    keys = [k for k in order if k in result] + [k for k in result if k not in order]
    print(
        f"\nFAR={far:.1%} global thresholds: frozen={thr_f:.4f}  tuned={thr_t:.4f}\n"
        f"{'stratum':<12}{'n_pos':>7}{'n_neg':>7}"
        f"{'fmr_fz':>9}{'fnmr_fz':>9}{'fmr_tn':>9}{'fnmr_tn':>9}"
        f"{'cal_slope':>11}{'cal_int':>10}"
    )
    for k in keys:
        r = result[k]
        print(
            f"{k:<12}{r['n_pos']:>7}{r['n_neg']:>7}"
            f"{r['fmr_frozen']:>9.4f}{r['fnmr_frozen']:>9.4f}"
            f"{r['fmr_tuned']:>9.4f}{r['fnmr_tuned']:>9.4f}"
            f"{r['calib_slope']:>11.3f}{r['calib_intercept']:>10.3f}"
        )


if __name__ == "__main__":
    main()
