"""CLI: DeLong test for correlated ROC (frozen vs +pairs) --- the paper's #1 stats gap.

For each benchmark we obtain per-pair cosine scores for BOTH models on the IDENTICAL pair set
(+ binary labels), then compare their ROC-AUCs with:

- the fast DeLong algorithm (Sun & Xu 2014, implemented here in numpy --- no new dependency):
  reports auc_frozen, auc_tuned, delta, the DeLong z, two-sided p_delong, and a 95% CI on the
  delta derived from the DeLong covariance (it accounts for the correlation between the two
  models' scores on the same pairs);
- a paired permutation test (within-pair random swap of the two models' scores) -> p_perm.

Benchmarks: FG-NET large-gap (>=25y) [PRIMARY], FG-NET overall, AgeDB-30, CALFW, internal-25+.
The frozen baseline is base FaceNet with NO fine-tune; +pairs = models/bb_facenet_seed42.pt.

    uv run python scripts/delong_test.py
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
from age_gap.common.logging import get_logger
from age_gap.evaluation.benchmark_external import load_bin, pair_scores
from age_gap.evaluation.fgnet import load_pairs as load_fgnet_pairs
from age_gap.evaluation.fgnet import prepare_crops
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import ImagePairDataset, _bb_prep, load_finetuned

log = get_logger(__name__)


# --------------------------------------------------------------------------------------------
# Fast DeLong (Sun & Xu, 2014, IEEE Signal Process. Lett.) --- pure numpy.
# --------------------------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks (ties averaged), as required by the structural-component AUC estimator."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1.0  # 1-based average rank over the tie block
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong for two correlated predictors on the SAME samples.

    Returns ``(aucs, cov)`` where ``aucs`` is shape (2,) [model A, model B] and ``cov`` is the
    2x2 covariance matrix of the AUC estimates (DeLong structural components, Sun & Xu 2014).
    Convention: positives have label 1, score should be higher for positives.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    # Predictions stacked as (k=2, n): row 0 = model A, row 1 = model B.
    predictions = np.vstack([scores_a, scores_b])
    pos = predictions[:, pos_mask]
    neg = predictions[:, neg_mask]
    m = pos.shape[1]  # positives
    n = neg.shape[1]  # negatives
    k = 2

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(np.concatenate([pos[r], neg[r]]))

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    # Structural components (placement values).
    v01 = (tz[:, :m] - tx) / n  # (k, m): over positives
    v10 = 1.0 - (tz[:, m:] - ty) / m  # (k, n): over negatives
    sx = np.cov(v01)  # 2x2
    sy = np.cov(v10)  # 2x2
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_compare(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """DeLong comparison of AUC(b) vs AUC(a) on identical samples.

    ``a`` = frozen, ``b`` = tuned. Returns auc_a, auc_b, delta=auc_b-auc_a, z, two-sided
    p_delong, and the 95% CI on the delta (lo, hi) from the DeLong variance of the difference.
    """
    aucs, cov = _fast_delong(scores_a, scores_b, labels)
    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    delta = auc_b - auc_a
    # Var(delta) = Var(b) + Var(a) - 2 Cov(a, b).
    var_delta = float(cov[1, 1] + cov[0, 0] - 2.0 * cov[0, 1])
    var_delta = max(var_delta, 0.0)
    se = float(np.sqrt(var_delta))
    if se == 0.0:
        z = 0.0 if delta == 0.0 else float(np.sign(delta) * np.inf)
        p = 1.0 if delta == 0.0 else 0.0
    else:
        z = delta / se
        p = float(2.0 * _norm_sf(abs(z)))
    half = 1.959963984540054 * se  # 95% normal CI
    return {
        "auc_frozen": auc_a,
        "auc_tuned": auc_b,
        "delta": delta,
        "z": z,
        "p_delong": p,
        "ci95_lo": delta - half,
        "ci95_hi": delta + half,
    }


def _norm_sf(x: float) -> float:
    """Upper-tail of the standard normal via erfc (no scipy)."""
    import math

    return 0.5 * math.erfc(x / math.sqrt(2.0))


# --------------------------------------------------------------------------------------------
# Paired permutation test: within-pair swap of the two models' scores.
# --------------------------------------------------------------------------------------------
def paired_permutation_pvalue(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    n_perm: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided p for delta=AUC(b)-AUC(a) under random within-sample swaps of (a,b).

    The exchangeability null is "the two models are interchangeable on every pair"; each
    permutation independently swaps model A/B scores for a random subset of pairs and recomputes
    the AUC difference. Uses the fast rank-based AUC so 10k permutations are cheap.
    """
    rng = np.random.default_rng(seed)
    obs = _auc_rank(scores_b, labels) - _auc_rank(scores_a, labels)
    n = len(labels)
    count = 0
    a = scores_a.astype(float)
    b = scores_b.astype(float)
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        pa = np.where(swap, b, a)
        pb = np.where(swap, a, b)
        diff = _auc_rank(pb, labels) - _auc_rank(pa, labels)
        if abs(diff) >= abs(obs) - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def _auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via midranks (Mann-Whitney), ties-aware --- matches the project's roc_auc."""
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _midrank(scores)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# --------------------------------------------------------------------------------------------
# Per-pair scores for both models on identical pair sets.
# --------------------------------------------------------------------------------------------
def _both_scores_bin(
    frozen: torch.nn.Module,
    tuned: torch.nn.Module,
    device: str,
    images_a: list,
    images_b: list,
    rgb: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine scores from frozen and tuned on the SAME image lists (identical pairs)."""
    sf = pair_scores(frozen, images_a, images_b, device, rgb=rgb)
    st = pair_scores(tuned, images_a, images_b, device, rgb=rgb)
    return sf, st


def _internal_scores(
    frozen: torch.nn.Module,
    tuned: torch.nn.Module,
    device: str,
    split: str = "test",
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-pair cosine for both models on our internal split (identical pairs, dataset order).

    Returns (scores_frozen, scores_tuned, labels, gaps). The crops subdir is taken from each
    backbone (frozen uses 'faces'; tuned carries crops_dir from its checkpoint) --- both default
    to 'faces' here, so the pair set is identical.
    """

    def run(bb: torch.nn.Module) -> np.ndarray:
        crops_dir = getattr(bb, "crops_dir", "faces")
        ds = ImagePairDataset(split=split, preprocess=_bb_prep(bb), crops_dir=crops_dir)
        bb.eval()
        loader = DataLoader(ds, batch_size=batch_size)
        out: list[float] = []
        with torch.no_grad():
            for ta, tb, _y, _w in loader:
                za, zb = bb(ta.to(device)), bb(tb.to(device))
                out.extend((za * zb).sum(dim=-1).cpu().tolist())
        return np.asarray(out, dtype=float)

    # Labels/gaps come from one dataset instance (deterministic order, no shuffle).
    ref = ImagePairDataset(split=split, preprocess=_bb_prep(frozen), crops_dir="faces")
    labels = np.asarray(ref.labels, dtype=np.int64)
    gaps = np.asarray(ref.gaps, dtype=np.int64)
    sf = run(frozen)
    st = run(tuned)
    return sf, st, labels, gaps


# --------------------------------------------------------------------------------------------
# Self-test on a tiny synthetic example with two known-different AUCs.
# --------------------------------------------------------------------------------------------
def _sanity_check() -> None:
    """Validate the DeLong implementation against an independent AUC + sane delta sign."""
    rng = np.random.default_rng(123)
    n = 400
    labels = np.array([1] * n + [0] * n)
    # Model A: weak separation; model B: stronger separation (same pairs, correlated noise).
    common = rng.normal(0, 1, 2 * n)
    sig = np.concatenate([np.ones(n), np.zeros(n)])
    a = 0.5 * sig + 0.7 * common + 0.7 * rng.normal(0, 1, 2 * n)
    b = 1.3 * sig + 0.7 * common + 0.7 * rng.normal(0, 1, 2 * n)
    res = delong_compare(a, b, labels)
    auc_a_ref = _auc_rank(a, labels)
    auc_b_ref = _auc_rank(b, labels)
    assert abs(res["auc_frozen"] - auc_a_ref) < 1e-9, (res["auc_frozen"], auc_a_ref)
    assert abs(res["auc_tuned"] - auc_b_ref) < 1e-9, (res["auc_tuned"], auc_b_ref)
    assert res["delta"] > 0.05, res["delta"]
    assert res["p_delong"] < 0.01, res["p_delong"]
    # Identical predictors -> delta 0, p 1.
    same = delong_compare(a, a.copy(), labels)
    assert abs(same["delta"]) < 1e-12, same["delta"]
    assert same["p_delong"] >= 0.999, same["p_delong"]
    log.info(
        "DeLong self-test OK: dAUC=%.3f (z=%.2f, p=%.2e); identical-pred p=%.3f",
        res["delta"],
        res["z"],
        res["p_delong"],
        same["p_delong"],
    )


def _record(
    scores_f: np.ndarray,
    scores_t: np.ndarray,
    labels: np.ndarray,
    n_perm: int,
) -> dict[str, float | list[float]]:
    d = delong_compare(scores_f, scores_t, labels)
    p_perm = paired_permutation_pvalue(scores_f, scores_t, labels, n_perm=n_perm)
    return {
        "auc_frozen": d["auc_frozen"],
        "auc_tuned": d["auc_tuned"],
        "delta": d["delta"],
        "z": d["z"],
        "p_delong": d["p_delong"],
        "p_perm": p_perm,
        "ci95_delta": [d["ci95_lo"], d["ci95_hi"]],
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DeLong correlated-ROC test (frozen vs +pairs)")
    parser.add_argument("--backbone", default="facenet")
    parser.add_argument("--tuned", default=None, help="default models/bb_<backbone>_seed42.pt")
    parser.add_argument("--large-gap", type=int, default=25, help="FG-NET / internal large gap")
    parser.add_argument("--perms", type=int, default=10000)
    args = parser.parse_args()

    _sanity_check()

    device = torch_device()
    models = data_path("models_dir")
    bb = args.backbone
    tuned_ckpt = Path(args.tuned) if args.tuned else Path(str(models / f"bb_{bb}_seed42.pt"))
    log.info("frozen=%s (pretrained), tuned=%s, device=%s", bb, tuned_ckpt, device)

    frozen = make_backbone(bb, pretrained=True).to(device).eval()
    frozen.crops_dir = "faces"  # type: ignore[assignment]
    tuned = load_finetuned(tuned_ckpt, device)

    results: dict[str, dict[str, float | list[float]]] = {}

    # ---- FG-NET (external cross-age): build identical pairs once, score both models. ----
    prepare_crops()
    fa, fb, fg_issame, fg_gaps = load_fgnet_pairs()
    sf, st = _both_scores_bin(frozen, tuned, device, fa, fb, rgb=False)
    large_mask = (fg_issame == 0) | ((fg_issame == 1) & (fg_gaps >= args.large_gap))
    results["fgnet_large_gap"] = _record(
        sf[large_mask], st[large_mask], fg_issame[large_mask], args.perms
    )
    results["fgnet_overall"] = _record(sf, st, fg_issame, args.perms)

    # ---- AgeDB-30 / CALFW (insightface .bin). ----
    ext = Path(str(data_path("data_dir", "external")))
    for name, key in (("agedb_30", "agedb_30"), ("calfw", "calfw")):
        p = ext / f"{name}.bin"
        if not p.exists():
            log.warning("%s missing (%s) -- skipped", name, p)
            continue
        ba, bbimg, issame = load_bin(p)
        s_f, s_t = _both_scores_bin(frozen, tuned, device, ba, bbimg, rgb=False)
        results[key] = _record(s_f, s_t, issame, args.perms)

    # ---- Internal 25+ (our test split, large-gap positives + all negatives). ----
    i_sf, i_st, i_lab, i_gap = _internal_scores(frozen, tuned, device)
    imask = (i_lab == 0) | ((i_lab == 1) & (i_gap >= args.large_gap))
    results["internal_25plus"] = _record(i_sf[imask], i_st[imask], i_lab[imask], args.perms)

    dst = data_path("metrics_dir", "delong_tests.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(results, indent=2), encoding="utf-8")

    _print_table(results)
    print(f"\nwrote {dst}")


def _print_table(results: dict[str, dict[str, float | list[float]]]) -> None:
    order = ["fgnet_large_gap", "fgnet_overall", "agedb_30", "calfw", "internal_25plus"]
    keys = [k for k in order if k in results] + [k for k in results if k not in order]
    print(
        f"\n{'benchmark':<18}{'auc_frozen':>12}{'auc_tuned':>12}{'delta':>10}"
        f"{'z':>9}{'p_delong':>12}{'p_perm':>12}{'n_pos':>8}{'n_neg':>8}"
    )

    def fmt(v: float) -> str:
        return f"{v:.3e}" if 0 < v < 1e-3 else f"{v:.4f}"

    for k in keys:
        r = results[k]
        print(
            f"{k:<18}{r['auc_frozen']:>12.4f}{r['auc_tuned']:>12.4f}{r['delta']:>+10.4f}"
            f"{r['z']:>9.2f}{fmt(float(r['p_delong'])):>12}{fmt(float(r['p_perm'])):>12}"
            f"{r['n_pos']:>8}{r['n_neg']:>8}"
        )


if __name__ == "__main__":
    main()
