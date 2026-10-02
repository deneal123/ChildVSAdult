"""Cross-age verification error breakdown with development-only calibration.

Builds on
``scripts/fgnet_retrieval_study.py`` (protocol, constants, row map, cache keys, guards):

* Endpoint-age-matched positives/negatives are built **separately inside each dev/test
  split** via the reused ``age_gap.evaluation.fgnet._match_negatives``; matching is never
  done globally and then filtered.
* Operating points are calibrated on **dev negatives only** and frozen on test, so the test
  split is never test-chosen. Acceptance is strict: ``accepted iff score > threshold``.
* Exact age-gap bins + explicit ``child(<13) -> adult(>25)`` stratum; age gap is a stratum,
  never a decision feature.
* Intrinsic blur from raw cached crops (Laplacian variance); cutpoints chosen on dev only.
* Pose / scan / digital / collage are always ``unknown`` and never inferred from pixels.
* Group dependence: subject-cluster bootstrap over BOTH pair endpoints, thresholds fixed.
  FMR/FNMR CIs, class counts and false counts per stratum; low-FAR resolution is disclosed
  (N_neg < ceil(1/FAR) cannot establish that FAR, e.g. N_neg<1000 for 0.1%).
* Embeddings reused by (model, dataset, preprocessing) key; only missing crops are embedded,
  locally, CPU, ``threads <= 2``, on root ``--execute`` (not in synthetic tests).

Training-identity independence is ``unverified`` (overlap audit is manual/pending).

Root run::

    .venv/Scripts/python.exe -m scripts.fgnet_error_breakdown --plan-only
    .venv/Scripts/python.exe -m scripts.fgnet_error_breakdown --execute \
        --output-dir metrics/fgnet_error_breakdown
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from age_gap.common.io import PROJECT_ROOT, write_jsonl
from age_gap.common.manifest import sha256_file, write_experiment_manifest
from age_gap.evaluation.fgnet import _match_negatives
from scripts.fgnet_retrieval_study import (
    ADULT_MIN_AGE,
    CHILD_MAX_AGE,
    PREPROCESSING,
    assert_public_output_has_no_ids,
    build_protocol,
    canonical_json,
    configure_torch_threads,
    embed_indices,
    embedding_cache_key,
    embedding_row_map,
    load_frozen_model,
    load_tuned_model,
    sanitize_nonfinite,
    sha256_text,
    verify_protocol,
)

# ── Fixed constants ─────────────────────────────────────────────────────────
SEED = 42
DEV_FRACTION = 0.5
ENDPOINT_AGE_TOLERANCE = 2
FAR_TARGETS: tuple[float, ...] = (0.01, 0.001)
# One expected event is the resolution floor for a target FAR.
FAR_RESOLUTION_FLOOR = {far: int(np.ceil(1.0 / far)) for far in FAR_TARGETS}
BLUR_BINS = 3
GAP_BINS: tuple[tuple[int, int | None, str], ...] = (
    (1, 9, "gap_01_09"),
    (10, 19, "gap_10_19"),
    (20, 24, "gap_20_24"),
    (25, None, "gap_25_plus"),
)
CHILD_TO_ADULT_LABEL = "child_lt13_to_adult_gt25"
UNSUPPORTED_MODALITY_AXES = ("pose", "scan", "digital", "collage")
TRAINING_INDEPENDENCE = "unverified"
DEFAULT_SOURCE_NPZ = PROJECT_ROOT / "data" / "external" / "fgnet_crops.npz"
DEFAULT_BASE_WEIGHTS = Path.home() / ".cache" / "torch" / "checkpoints" / "20180408-102900-casia-webface.pt"
DEFAULT_TUNED_SEEDS = (42, 1, 2)
DEFAULT_RETRIEVAL_CACHE_DIR = PROJECT_ROOT / "metrics" / "fgnet_retrieval_20261002" / "private"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "metrics" / "fgnet_error_breakdown"

DISCLOSURES: tuple[str, ...] = (
    "Pairs are endpoint-age-matched within each split. Test METADATA (subject/age) defines the test "
    "pairs, but test SCORES never select matching, calibrate thresholds or pick blur cutpoints.",
    "Thresholds are calibrated on development negatives only; test thresholds are frozen.",
    "Acceptance is strict (score > threshold); negatives equal to the threshold are rejected.",
    "Test FMR/FNMR are conditional on the development threshold: the dev split has its own sampling "
    "uncertainty, so the reported bootstrap CIs do not include threshold-selection variance.",
    "Subject-cluster bootstrap covers BOTH endpoints; thresholds are fixed, never re-calibrated.",
    "A percentile CI of [0, 0] from zero observed false events does NOT prove zero risk. "
    "Boundary empirical intervals are retained separately; inferential intervals are unavailable. "
    "No independent-trial binomial bound is asserted for subject-dependent pairs.",
    "Pose, scan, digital and collage are unsupported and reported as 'unknown', never inferred.",
    "Low-FAR resolution is limited: a target FAR needs >= ceil(1/FAR) negatives; N_neg < 1000 "
    "cannot establish a 0.1% FAR claim.",
    "Age gap is only a stratum, never a decision feature.",
    "Training-identity independence is unverified; the overlap audit is manual and pending.",
)

# Event-count labels attached to every per-stratum rate.
SMALL_EVENT_COUNT = 5


def validate_configuration(
    *, n_boot: int, batch_size: int, threads: int,
    far_targets: Sequence[float] = FAR_TARGETS, seeds: Sequence[int] | None = None,
) -> None:
    """Reject impossible settings *before* any costly embedding or scoring happens."""
    if int(n_boot) < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not 1 <= int(threads) <= 2:
        raise ValueError(f"threads must be in 1..2 (CPU budget), got {threads}")
    for far in far_targets:
        if not 0.0 < float(far) < 1.0:
            raise ValueError(f"every FAR target must satisfy 0 < far < 1, got {far}")
    if seeds is not None:
        seeds = list(seeds)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"seed list contains duplicates: {seeds}")


def assert_finite_scores(scores: np.ndarray, *, context: str = "scores") -> np.ndarray:
    """Fail loudly on non-finite scores instead of silently propagating NaN into rates."""
    values = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        bad = int(np.count_nonzero(~np.isfinite(values)))
        raise ValueError(f"{context} contain {bad} non-finite value(s)")
    return values


def _event_flag(events: int, n: int) -> str:
    if n == 0:
        return "no_pairs"
    if events == 0:
        return "no_events_observed"
    if events < SMALL_EVENT_COUNT:
        return "small_event_count"
    return "adequate_event_count"


def _zero_event_note(events: int, n: int, ci: list[float] | None, kind: str) -> str | None:
    """Do not apply independent-trial bounds to pairs sharing subjects."""
    if n == 0 or events > 0:
        return None
    note = (
        f"0 {kind} observed: the [0, 0] percentile CI is NOT proof of zero risk. "
        "An independent-trial rule of three is not applicable to these subject-dependent pairs; "
        "no reliable risk upper bound is available from this empirical bootstrap."
    )
    if ci is not None and (ci[0] > 0.0 or ci[1] > 0.0):
        note += f" Resampled CI was {ci}."
    return note


# ── Pair model ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VerificationPair:
    pair_id: str
    label: int  # 1 same identity, 0 different identity
    split: str
    left_crop: int
    right_crop: int
    left_subject: int
    right_subject: int
    left_age: int
    right_age: int
    age_gap: int
    anchor: str


def _split_seed(seed: int, split: str) -> int:
    return int(seed) + int.from_bytes(sha256_text(split).encode()[:4], "little") % 10_000


def _gap_bin(gap: int) -> str:
    for low, high, label in GAP_BINS:
        if gap >= low and (high is None or gap <= high):
            return label
    return "gap_out_of_range"


def build_split_pairs(
    *,
    subjects: np.ndarray,
    ages: np.ndarray,
    protocol: Any,
    split: str,
    tolerance: int = ENDPOINT_AGE_TOLERANCE,
    seed: int = SEED,
) -> tuple[list[VerificationPair], dict[str, Any]]:
    """Build balanced endpoint-age-matched pairs for one split, using only that split's
    subjects (matching is split-local, never global-then-filtered)."""
    subjects = np.asarray(subjects)
    ages = np.asarray(ages)
    gallery_by_subject = {int(s): int(i) for s, i in zip(protocol.gallery_subject, protocol.gallery_index, strict=True)}
    gallery_age = protocol.gallery_age_by_subject()
    mask = np.asarray(protocol.split_mask(split))
    split_subjects = set(np.asarray(protocol.query_subject)[mask].tolist())
    if not split_subjects:
        raise ValueError(f"split {split!r} has no queried subjects")
    pool = np.asarray([i for i, s in enumerate(subjects.tolist()) if int(s) in split_subjects], dtype=np.int64)
    local_of = {int(o): p for p, o in enumerate(pool.tolist())}
    local_positives: list[tuple[int, int, int]] = []
    anchors: list[dict[str, int]] = []
    for q_index, q_subject, q_age in zip(
        protocol.query_index[mask], protocol.query_subject[mask], protocol.query_age[mask], strict=True
    ):
        g_index = gallery_by_subject[int(q_subject)]
        if int(q_index) not in local_of or int(g_index) not in local_of:
            continue
        local_positives.append((local_of[int(q_index)], local_of[int(g_index)], int(gallery_age[int(q_subject)]) - int(q_age)))
        anchors.append({"q": int(q_index), "g": int(g_index), "q_age": int(q_age), "g_age": int(gallery_age[int(q_subject)])})

    matched = _match_negatives(subjects[pool], ages[pool], local_positives, tolerance=tolerance, seed=_split_seed(seed, split))
    pairs: list[VerificationPair] = []
    endpoint_errors: list[int] = []
    for pos_index, left, right, gap, error, _src_gap in matched:
        q_index, g_index = anchors[pos_index]["q"], anchors[pos_index]["g"]
        pairs.append(VerificationPair(
            pair_id=f"{split}_pos_{q_index}_{g_index}", label=1, split=split,
            left_crop=q_index, right_crop=g_index,
            left_subject=int(subjects[q_index]), right_subject=int(subjects[g_index]),
            left_age=anchors[pos_index]["q_age"], right_age=anchors[pos_index]["g_age"],
            age_gap=int(gap), anchor="query_younger_to_gallery",
        ))
        neg_left, neg_right = int(pool[int(left)]), int(pool[int(right)])
        if int(ages[neg_left]) > int(ages[neg_right]):  # orient younger endpoint left
            neg_left, neg_right = neg_right, neg_left
        pairs.append(VerificationPair(
            pair_id=f"{split}_neg_{neg_left}_{neg_right}", label=0, split=split,
            left_crop=neg_left, right_crop=neg_right,
            left_subject=int(subjects[neg_left]), right_subject=int(subjects[neg_right]),
            left_age=int(ages[neg_left]), right_age=int(ages[neg_right]),
            age_gap=int(gap), anchor="endpoint_age_matched_impostor",
        ))
        endpoint_errors.append(int(error))

    diagnostics = {
        "split": split,
        "n_pool_crops": int(pool.size),
        "n_split_subjects": len(split_subjects),
        "n_eligible_positives": len(local_positives),
        "n_matched_positives": len(matched),
        "n_unmatched_positives": len(local_positives) - len(matched),
        "positive_coverage": (len(matched) / len(local_positives)) if local_positives else float("nan"),
        "match_endpoint_error_mean": float(np.mean(endpoint_errors)) if endpoint_errors else None,
        "match_endpoint_error_max": int(max(endpoint_errors)) if endpoint_errors else None,
        "matching_seed": _split_seed(seed, split),
        "endpoint_age_tolerance": int(tolerance),
        "candidate_pool_is_split_local": True,
    }
    return pairs, diagnostics


def pair_arrays(pairs: list[VerificationPair]) -> dict[str, np.ndarray]:
    """Flatten a split's pairs into aligned arrays, positives then negatives."""
    pos = [p for p in pairs if p.label == 1]
    neg = [p for p in pairs if p.label == 0]
    if len(pos) != len(neg):
        raise ValueError("matched pairs must be class-balanced within a split")

    def _stack(rows: list[VerificationPair]) -> dict[str, np.ndarray]:
        return {
            key: np.asarray([getattr(p, key) for p in rows], dtype=np.int64)
            for key in ("left_crop", "right_crop", "left_subject", "right_subject", "age_gap", "left_age", "right_age")
        }

    out = {f"pos_{k}": v for k, v in _stack(pos).items()}
    out.update({f"neg_{k}": v for k, v in _stack(neg).items()})
    return out


def pair_cosine_scores(embeddings: np.ndarray, rows_a: np.ndarray, rows_b: np.ndarray) -> np.ndarray:
    """Cosine similarity of aligned embedding rows (L2-normalised defensively)."""
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm embedding row cannot be scored")
    unit = emb / norms
    scores = np.einsum("ij,ij->i", unit[np.asarray(rows_a)], unit[np.asarray(rows_b)])
    return assert_finite_scores(scores, context="pair cosine scores")


def scores_for_pairs(pairs: list[VerificationPair], embeddings: np.ndarray, row_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, labels) for a split; labels are for evaluation only."""
    labels = np.asarray([p.label for p in pairs], dtype=np.int64)
    rows_a = row_map[np.asarray([p.left_crop for p in pairs], dtype=np.int64)]
    rows_b = row_map[np.asarray([p.right_crop for p in pairs], dtype=np.int64)]
    if np.any(rows_a < 0) or np.any(rows_b < 0):
        raise ValueError("pair references a crop without an embedding row")
    return pair_cosine_scores(embeddings, rows_a, rows_b), labels


# ── Calibration (dev negatives only, strict >) ──────────────────────────────
@dataclass(frozen=True)
class OperatingPoint:
    far_target: float
    threshold: float
    n_dev_neg: int
    max_dev_accepts: int
    dev_false_accepts: int
    dev_fmr: float
    dev_false_rejects: int
    dev_n_pos: int
    dev_fnmr: float
    tie_values_at_threshold: int
    resolvable: bool
    resolution_note: str


def calibrate_operating_point(dev_neg_scores: np.ndarray, dev_pos_scores: np.ndarray, far_target: float) -> OperatingPoint:
    """Lowest threshold whose strict dev FMR stays at/below ``far_target``.

    With ``m = floor(far_target * N)`` (the maximum allowed dev accepts), ``N = len(neg)``, the
    threshold is the **(m+1)-th largest** dev negative, i.e. ``sorted(neg)[N-1-m]``. Exactly ``m``
    negatives are strictly greater than it when there are no ties, so dev FMR is at most ``m/N``;
    negatives tied at the threshold are rejected and cannot push the FMR above target."""
    if not 0.0 < float(far_target) < 1.0:
        raise ValueError(f"far_target must satisfy 0 < far < 1, got {far_target}")
    neg = assert_finite_scores(dev_neg_scores, context="dev negative scores")
    pos = assert_finite_scores(dev_pos_scores, context="dev positive scores")
    n = int(neg.size)
    floor = FAR_RESOLUTION_FLOOR.get(far_target, int(np.ceil(1.0 / far_target)))
    resolvable = n >= floor
    note = "observable: N_neg >= ceil(1/FAR)" if resolvable else f"NOT resolvable: N_neg={n} < {floor}=ceil(1/{far_target:g})"
    if n == 0:
        return OperatingPoint(far_target, float("inf"), 0, 0, 0, float("nan"), 0, int(pos.size), float("nan"), 0, False, "no dev negatives")
    m = int(np.floor(far_target * n))
    threshold = float(np.sort(neg)[n - 1 - m]) if m < n else float("-inf")
    false_accept = int(np.count_nonzero(neg > threshold))
    false_reject = int(np.count_nonzero(pos <= threshold))
    return OperatingPoint(
        far_target=float(far_target), threshold=threshold, n_dev_neg=n, max_dev_accepts=m,
        dev_false_accepts=false_accept, dev_fmr=(false_accept / n),
        dev_false_rejects=false_reject, dev_n_pos=int(pos.size),
        dev_fnmr=(false_reject / pos.size) if pos.size else float("nan"),
        tie_values_at_threshold=int(np.count_nonzero(neg == threshold)),
        resolvable=resolvable, resolution_note=note,
    )


def counts_at_threshold(pos_scores: np.ndarray, neg_scores: np.ndarray, threshold: float) -> dict[str, float]:
    """False accepts/rejects and FMR/FNMR at a frozen threshold (strict ``>``)."""
    pos = assert_finite_scores(pos_scores, context="positive scores")
    neg = assert_finite_scores(neg_scores, context="negative scores")
    fa = int(np.count_nonzero(neg > threshold))
    fr = int(np.count_nonzero(pos <= threshold))
    return {
        "n_pos": int(pos.size), "n_neg": int(neg.size), "false_accepts": fa, "false_rejects": fr,
        "fmr": (fa / neg.size) if neg.size else float("nan"), "fnmr": (fr / pos.size) if pos.size else float("nan"),
    }


# ── Uncertainty: subject-cluster bootstrap over BOTH endpoints ──────────────
def bootstrap_fmr_fnmr(
    *, pos_subject: np.ndarray, pos_reject: np.ndarray,
    neg_left_subject: np.ndarray, neg_right_subject: np.ndarray, neg_accept: np.ndarray,
    n_boot: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    """Percentile CI resampling subjects shared by both endpoints of each pair.

    Positives weigh by one identity's multiplicity, negatives by the product of the two
    impostor endpoints' multiplicities. Thresholds are fixed, so the interval reflects
    identity sampling, not re-calibration."""
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1")
    pos_subject = np.asarray(pos_subject, dtype=np.int64)
    neg_left_subject = np.asarray(neg_left_subject, dtype=np.int64)
    neg_right_subject = np.asarray(neg_right_subject, dtype=np.int64)
    pos_reject = assert_finite_scores(pos_reject, context="positive rejection indicators")
    neg_accept = assert_finite_scores(neg_accept, context="negative acceptance indicators")
    if not (pos_subject.size == pos_reject.size and neg_left_subject.size == neg_right_subject.size == neg_accept.size):
        raise ValueError("bootstrap inputs must be aligned per class")
    universe = np.unique(np.concatenate([pos_subject, neg_left_subject, neg_right_subject]))
    if universe.size == 0:
        return {"n_boot": int(n_boot), "valid": 0, "valid_fmr": 0, "valid_fnmr": 0, "fmr_ci95": None, "fnmr_ci95": None}
    index = {int(s): i for i, s in enumerate(universe.tolist())}
    pos_idx = np.asarray([index[int(s)] for s in pos_subject.tolist()], dtype=np.int64)
    left_idx = np.asarray([index[int(s)] for s in neg_left_subject.tolist()], dtype=np.int64)
    right_idx = np.asarray([index[int(s)] for s in neg_right_subject.tolist()], dtype=np.int64)
    rng = np.random.default_rng(_split_seed(seed, "bootstrap"))
    fmr_draws: list[float] = []
    fnmr_draws: list[float] = []
    for _ in range(int(n_boot)):
        mult = np.bincount(rng.integers(universe.size, size=universe.size), minlength=universe.size).astype(np.float64)
        w_pos, w_neg = mult[pos_idx], mult[left_idx] * mult[right_idx]
        if w_pos.sum() > 0:
            fnmr_draws.append(float(np.sum(w_pos * pos_reject) / w_pos.sum()))
        if w_neg.sum() > 0:
            fmr_draws.append(float(np.sum(w_neg * neg_accept) / w_neg.sum()))

    def ci(values: list[float]) -> list[float] | None:
        if not values:
            return None
        lo, hi = np.percentile(np.asarray(values), [2.5, 97.5])
        return [float(lo), float(hi)]

    return {
        "n_boot": int(n_boot), "valid": int(min(len(fmr_draws), len(fnmr_draws))),
        "valid_fmr": int(len(fmr_draws)), "valid_fnmr": int(len(fnmr_draws)),
        "n_subjects_fmr": int(np.unique(np.concatenate([neg_left_subject, neg_right_subject])).size),
        "n_subjects_fnmr": int(np.unique(pos_subject).size),
        "method": "subject-cluster bootstrap; both pair endpoints; percentile 95% CI; fixed dev threshold",
        "fmr_ci95": ci(fmr_draws), "fnmr_ci95": ci(fnmr_draws),
    }


# ── Blur (raw cached crops; dev-only cutpoints) ─────────────────────────────
def crop_blur_variances(crops: np.ndarray) -> np.ndarray:
    """Laplacian-variance sharpness for every raw cached crop (reused quality util)."""
    from age_gap.preprocessing.quality import blur_variance  # lazy: cv2 only when executing

    return np.asarray([blur_variance(np.asarray(crop)) for crop in np.asarray(crops)], dtype=np.float64)


def blur_cutpoints(dev_pair_blur: np.ndarray, n_bins: int = BLUR_BINS) -> list[float]:
    """Cutpoints from dev pair blur values only; degenerate input degrades safely."""
    values = np.asarray(dev_pair_blur, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or n_bins < 2:
        return []
    cutpoints = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return [float(c) for c in np.unique(cutpoints)]


def blur_bin_labels(pair_blur_values: np.ndarray, cutpoints: Sequence[float]) -> np.ndarray:
    """Assign each pair to a blur bin (0 = blurriest) using dev-only cutpoints."""
    return np.digitize(np.asarray(pair_blur_values, dtype=np.float64), np.asarray(list(cutpoints), dtype=np.float64)).astype(np.int64)


def pair_blur(crop_blur: np.ndarray, pairs: list[VerificationPair]) -> np.ndarray:
    """Pair sharpness = min(endpoint sharpness): a pair is as sharp as its blurriest face."""
    values = np.asarray(crop_blur, dtype=np.float64)
    return np.asarray([min(values[p.left_crop], values[p.right_crop]) for p in pairs], dtype=np.float64)


# ── Unsupported modalities (never inferred) ─────────────────────────────────
def modality_axes(explicit: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pose/scan/digital/collage status. Anything not explicitly adjudicated is ``unknown``.

    No image inspection happens here: an axis can only leave ``unknown`` when the caller
    passes an explicit value from an already-reviewed manifest."""
    explicit = explicit or {}
    axes = {axis: (str(explicit[axis]) if explicit.get(axis) is not None else "unknown") for axis in UNSUPPORTED_MODALITY_AXES}
    axes["inference"] = "none: unsupported capture modalities are never inferred from pixels"
    return axes


# ── Embedding reuse by (model, dataset, preprocessing) key ──────────────────
def _cache_components_key(*, role: str, weights_sha256: str, source_sha256: str) -> str:
    return sha256_text(canonical_json({
        "role": role, "weights_sha256": weights_sha256, "source_sha256": source_sha256, "preprocessing": PREPROCESSING,
    }))


def _read_verified_cache(path: Path, *, role: str, weights_sha256: str, source_sha256: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (indices, embeddings) only if the cache digest matches all components."""
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if "key" not in archive or "indices" not in archive or "embeddings" not in archive:
            return None
        indices = np.asarray(archive["indices"], dtype=np.int64)
        expected = embedding_cache_key(role=role, weights_sha256=weights_sha256, source_sha256=source_sha256, indices=indices)
        if str(archive["key"]) != expected:
            return None
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        if indices.ndim != 1 or embeddings.ndim != 2 or len(embeddings) != len(indices):
            raise ValueError("cache embedding rows do not match crop indices")
        if len(np.unique(indices)) != len(indices) or np.any(indices < 0):
            raise ValueError("cache crop indices must be unique and non-negative")
        if not np.all(np.isfinite(embeddings)) or np.any(np.linalg.norm(embeddings, axis=1) == 0):
            raise ValueError("invalid cached embeddings")
        return indices, embeddings


def resolve_embeddings(
    *, role: str, weights_sha256: str, source_sha256: str, required_indices: np.ndarray,
    crops: np.ndarray | None, cache_candidates: Sequence[Path], write_path: Path, load_model: Any,
    batch_size: int = 32, threads: int = 2, allow_embed: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reuse verified caches by model/dataset/preproc key and embed only missing crops.

    The 648-crop retrieval cache is recycled even though all 650 verification crops are
    required: existing rows are reused and only the 2 missing crops would be embedded,
    locally, CPU ``threads <= 2``, and only when ``allow_embed`` (root execution)."""
    if not 1 <= int(threads) <= 2:
        raise ValueError("threads must be in 1..2 (CPU budget)")
    required = np.asarray(required_indices, dtype=np.int64)
    if np.unique(required).size != required.size:
        raise ValueError("required crop indices must be unique")
    available: dict[int, np.ndarray] = {}
    reused_from: list[str] = []
    for candidate in cache_candidates:
        verified = _read_verified_cache(candidate, role=role, weights_sha256=weights_sha256, source_sha256=source_sha256)
        if verified is None:
            continue
        indices, embeddings = verified
        for i, row in zip(indices.tolist(), embeddings, strict=True):
            available.setdefault(int(i), row)
        reused_from.append(Path(candidate).name)
    missing = np.asarray([i for i in required.tolist() if i not in available], dtype=np.int64)
    embedded = False
    if missing.size:
        if not allow_embed:
            raise RuntimeError(f"cache miss for {missing.size} crop(s) for role={role!r}; re-run with --execute to embed locally")
        if crops is None:
            raise ValueError("raw crops are required to embed missing indices")
        configure_torch_threads(threads)
        new_embeddings = embed_indices(crops, missing, load_model(), batch_size=batch_size, threads=threads)
        for i, row in zip(missing.tolist(), new_embeddings, strict=True):
            available[int(i)] = row
        embedded = True
    matrix = np.stack([np.asarray(available[int(i)], dtype=np.float32) for i in required.tolist()])
    write_path = Path(write_path)
    write_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        write_path,
        key=np.asarray(embedding_cache_key(role=role, weights_sha256=weights_sha256, source_sha256=source_sha256, indices=required)),
        indices=required, embeddings=matrix,
    )
    return matrix, {
        "role": role, "cache_components_key": _cache_components_key(role=role, weights_sha256=weights_sha256, source_sha256=source_sha256),
        "n_required": int(required.size), "n_reused": int(required.size - missing.size), "n_embedded": int(missing.size),
        "embedded_missing": bool(embedded), "reused_from": reused_from, "write_path_name": write_path.name,
    }


# ── Stratum evaluation ──────────────────────────────────────────────────────
def _empty_stratum(name: str) -> dict[str, Any]:
    return {"stratum": name, "n_pos": 0, "n_neg": 0, "n_subjects": 0, "false_accepts": 0, "false_rejects": 0,
            "fmr": None, "fnmr": None, "empty": True, "fmr_ci95": None, "fnmr_ci95": None,
            "bootstrap_valid": 0, "bootstrap_valid_fmr": 0, "bootstrap_valid_fnmr": 0,
            "fmr_event_flag": "no_pairs", "fnmr_event_flag": "no_pairs",
            "fmr_zero_event_note": None, "fnmr_zero_event_note": None,
            "conditional_on_dev_threshold": True, "threshold_selection_variance_included": False}


def evaluate_stratum(
    name: str, *, pos_mask: np.ndarray, neg_mask: np.ndarray, pos_scores: np.ndarray, neg_scores: np.ndarray,
    pos_subject: np.ndarray, neg_left_subject: np.ndarray, neg_right_subject: np.ndarray,
    threshold: float, far_target: float, n_boot: int, seed: int,
) -> dict[str, Any]:
    """FMR/FNMR counts, rates and both-endpoint bootstrap CI for one stratum."""
    pos_mask = np.asarray(pos_mask, dtype=bool)
    neg_mask = np.asarray(neg_mask, dtype=bool)
    p, n = pos_scores[pos_mask], neg_scores[neg_mask]
    if p.size == 0 and n.size == 0:
        return _empty_stratum(name)
    summary = counts_at_threshold(p, n, threshold)
    ci = bootstrap_fmr_fnmr(
        pos_subject=pos_subject[pos_mask], pos_reject=(p <= threshold).astype(np.float64),
        neg_left_subject=neg_left_subject[neg_mask], neg_right_subject=neg_right_subject[neg_mask],
        neg_accept=(n > threshold).astype(np.float64), n_boot=n_boot, seed=seed,
    )
    n_subjects = int(np.unique(np.concatenate([
        np.asarray(pos_subject)[pos_mask], np.asarray(neg_left_subject)[neg_mask], np.asarray(neg_right_subject)[neg_mask],
    ])).size) if (p.size or n.size) else 0
    summary.update({
        "stratum": name, "far_target": float(far_target), "empty": False,
        "n_subjects": n_subjects,
        "fmr_ci95": ci["fmr_ci95"], "fnmr_ci95": ci["fnmr_ci95"],
        "bootstrap_n_boot": ci["n_boot"], "bootstrap_valid": ci["valid"],
        "bootstrap_valid_fmr": ci["valid_fmr"], "bootstrap_valid_fnmr": ci["valid_fnmr"],
        "bootstrap_subjects_fmr": ci["n_subjects_fmr"], "bootstrap_subjects_fnmr": ci["n_subjects_fnmr"],
        "fmr_event_flag": _event_flag(int(summary["false_accepts"]), int(summary["n_neg"])),
        "fnmr_event_flag": _event_flag(int(summary["false_rejects"]), int(summary["n_pos"])),
        "fmr_zero_event_note": _zero_event_note(int(summary["false_accepts"]), int(summary["n_neg"]), ci["fmr_ci95"], "false accepts"),
        "fnmr_zero_event_note": _zero_event_note(int(summary["false_rejects"]), int(summary["n_pos"]), ci["fnmr_ci95"], "false rejects"),
        "conditional_on_dev_threshold": True,
        "threshold_selection_variance_included": False,
        "fmr_resolution_floor_negatives": FAR_RESOLUTION_FLOOR.get(far_target),
        "fmr_resolvable": bool(n.size >= FAR_RESOLUTION_FLOOR.get(far_target, 0)),
    })
    for metric, events in (("fmr", "false_accepts"), ("fnmr", "false_rejects")):
        summary[f"{metric}_empirical_bootstrap_ci95"] = summary[f"{metric}_ci95"]
        if summary[events] == 0:
            summary[f"{metric}_ci95"] = None
            summary[f"{metric}_ci_status"] = "unavailable: zero-event empirical boundary"
        else:
            summary[f"{metric}_ci_status"] = "empirical percentile; conditional on dev threshold"
    return summary


def split_strata(
    *, pos_gap: np.ndarray, neg_gap: np.ndarray, pos_left_age: np.ndarray, neg_left_age: np.ndarray,
    pos_right_age: np.ndarray, neg_right_age: np.ndarray, pos_blur_bin: np.ndarray, neg_blur_bin: np.ndarray, n_blur_bins: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Boolean (pos_mask, neg_mask) per stratum: overall, exact gap bins, child->adult, blur."""
    masks: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "overall": (np.ones(pos_gap.size, dtype=bool), np.ones(neg_gap.size, dtype=bool))
    }
    for _low, _high, label in GAP_BINS:
        masks[label] = (
            np.asarray([_gap_bin(int(g)) == label for g in pos_gap]),
            np.asarray([_gap_bin(int(g)) == label for g in neg_gap]),
        )
    masks[CHILD_TO_ADULT_LABEL] = (
        (pos_left_age < CHILD_MAX_AGE) & (pos_right_age > ADULT_MIN_AGE),
        (neg_left_age < CHILD_MAX_AGE) & (neg_right_age > ADULT_MIN_AGE),
    )
    for b in range(int(n_blur_bins)):
        masks[f"blur_bin_{b}"] = (pos_blur_bin == b, neg_blur_bin == b)
    # Exact integer year gaps: one stratum per observed gap, so a breakdown is not limited to buckets.
    observed = sorted(set(np.asarray(pos_gap, dtype=np.int64).tolist()) | set(np.asarray(neg_gap, dtype=np.int64).tolist()))
    for year in observed:
        masks[f"gap_year_{int(year):02d}"] = (pos_gap == year, neg_gap == year)
    return masks


def evaluate_split(
    *, split: str, pairs: list[VerificationPair], scores: np.ndarray, crop_blur: np.ndarray,
    operating_points: dict[float, OperatingPoint], n_boot: int = 2000, seed: int = SEED,
    blur_cutpoints_dev: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Evaluate one split against frozen (dev-calibrated) thresholds."""
    assert_finite_scores(scores, context=f"{split} pair scores")
    arrays = pair_arrays(pairs)
    # Exact-gap invariants: observed endpoints must reproduce the gap exactly, and both
    # classes must share the same gap distribution (matched impostors mirror positives).
    for side in ("pos", "neg"):
        if not np.array_equal(arrays[f"{side}_right_age"] - arrays[f"{side}_left_age"], arrays[f"{side}_age_gap"]):
            raise ValueError(f"{split}/{side}: age_gap does not equal right_age - left_age")
    if not np.array_equal(np.sort(arrays["pos_age_gap"]), np.sort(arrays["neg_age_gap"])):
        raise ValueError(f"{split}: positive and negative age-gap distributions differ; matching is unbalanced")
    labels = np.asarray([p.label for p in pairs], dtype=np.int64)
    pos_scores, neg_scores = scores[labels == 1], scores[labels == 0]
    blur = pair_blur(crop_blur, pairs)
    edges = list(blur_cutpoints_dev or [])
    pos_bin = blur_bin_labels(blur[labels == 1], edges)
    neg_bin = blur_bin_labels(blur[labels == 0], edges)
    masks = split_strata(
        pos_gap=arrays["pos_age_gap"], neg_gap=arrays["neg_age_gap"],
        pos_left_age=arrays["pos_left_age"], neg_left_age=arrays["neg_left_age"],
        pos_right_age=arrays["pos_right_age"], neg_right_age=arrays["neg_right_age"],
        pos_blur_bin=pos_bin, neg_blur_bin=neg_bin, n_blur_bins=max(len(edges) + 1, BLUR_BINS),
    )
    out: dict[str, Any] = {"split": split, "operating_points": {}}
    for far_target, op in sorted(operating_points.items()):
        out["operating_points"][f"{far_target:g}"] = {
            "threshold": op.threshold, "dev_n_neg": op.n_dev_neg, "dev_false_accepts": op.dev_false_accepts,
            "dev_fmr": op.dev_fmr, "dev_false_rejects": op.dev_false_rejects, "dev_fnmr": op.dev_fnmr,
            "resolvable": op.resolvable, "resolution_note": op.resolution_note,
            "strata": {
                name: evaluate_stratum(
                    name, pos_mask=pm, neg_mask=nm, pos_scores=pos_scores, neg_scores=neg_scores,
                    pos_subject=arrays["pos_left_subject"], neg_left_subject=arrays["neg_left_subject"],
                    neg_right_subject=arrays["neg_right_subject"], threshold=op.threshold,
                    far_target=far_target, n_boot=n_boot, seed=seed,
                )
                for name, (pm, nm) in masks.items()
            },
        }
    out["age_gap_distribution"] = {
        "positive": _distribution(arrays["pos_age_gap"].astype(float)),
        "negative": _distribution(arrays["neg_age_gap"].astype(float)),
        "classes_share_gap_distribution": True,
        "gap_year_counts": {f"gap_year_{int(year):02d}": {
            "positive": int(np.count_nonzero(arrays["pos_age_gap"] == year)),
            "negative": int(np.count_nonzero(arrays["neg_age_gap"] == year)),
        } for year in sorted(set(arrays["pos_age_gap"].tolist()) | set(arrays["neg_age_gap"].tolist()))},
    }
    out["n_pos"] = int(labels.size - int((labels == 0).sum()))
    out["n_neg"] = int((labels == 0).sum())
    out["n_subjects"] = int(np.unique(np.concatenate([arrays["pos_left_subject"], arrays["pos_right_subject"],
                                                      arrays["neg_left_subject"], arrays["neg_right_subject"]])).size)
    out["blur_cutpoints_used"] = edges
    return out


def _distribution(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max())}


# ── Public aggregate + orchestration ────────────────────────────────────────
def build_public_output(
    *, protocol_hash: str, source_sha256: str, split_summaries: dict[str, Any], pair_diagnostics: dict[str, Any],
    model_meta: dict[str, Any], blur_cutpoints: Sequence[float], modality: dict[str, Any], n_boot: int,
) -> dict[str, Any]:
    """Aggregate-only payload; no subject identifiers, crops, scores or embeddings."""
    return {
        "task_id": "error-analysis",
        "analysis": "fgnet-endpoint-age-matched-verification-error-breakdown",
        "protocol_hash": protocol_hash,
        "source_npz_sha256": source_sha256,
        "seed": SEED, "dev_fraction": DEV_FRACTION, "endpoint_age_tolerance": ENDPOINT_AGE_TOLERANCE,
        "acceptance_convention": "strict: accepted iff cosine > threshold",
        "threshold_provenance": "development negatives only; frozen on test",
        "bootstrap": {"method": "subject-cluster bootstrap over both pair endpoints; thresholds fixed", "n_boot": int(n_boot)},
        "far_targets": list(FAR_TARGETS),
        "far_resolution_floor_negatives": {f"{k:g}": v for k, v in FAR_RESOLUTION_FLOOR.items()},
        "age_gap_bins": [label for _l, _h, label in GAP_BINS],
        "age_gap_year_strata": "one 'gap_year_NN' stratum per observed integer age gap, in addition to the 4 buckets",
        "child_to_adult": {"child_max_age_exclusive": CHILD_MAX_AGE, "adult_min_age_exclusive": ADULT_MIN_AGE},
        "zero_event_policy": "zero-event bootstrap boundaries are not valid risk bounds for "
                             "subject-dependent pairs; inferential CI unavailable, empirical CI separate",
        "blur": {"measure": "Laplacian variance (age_gap.preprocessing.quality.blur_variance)",
                 "pair_rule": "pair sharpness = min(endpoint sharpness)",
                 "cutpoints": [float(c) for c in blur_cutpoints], "cutpoint_source": "development split only"},
        "unsupported_modalities": modality,
        "pair_diagnostics": pair_diagnostics,
        "models": model_meta,
        "splits": split_summaries,
        "training_identity_independence": TRAINING_INDEPENDENCE,
        "disclosures": list(DISCLOSURES),
        "privacy": "Aggregate only: no subject identifiers, crop indices, scores or embeddings.",
    }


def write_private_artifacts(private_dir: Path, *, pairs_by_split: dict[str, list[VerificationPair]],
                            row_map: np.ndarray, embeddings: np.ndarray, model_name: str) -> list[Path]:
    """Row-level scores/metadata and scored embeddings live only in the private directory.

    The scored embedding matrix is written to a *separate* ``private_scored_embeddings_<model>.npz``
    so it can never clobber the resolver's keyed cache ``private_embeddings_<model>.npz`` (which
    carries the ``key`` digest that makes reuse safe)."""
    private_dir = Path(private_dir)
    private_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for split, pairs in pairs_by_split.items():
        split_scores, _labels = scores_for_pairs(pairs, embeddings, row_map)
        for pair, score in zip(pairs, split_scores, strict=True):
            rows.append({
                "split": split, "pair_id": pair.pair_id, "label": pair.label,
                "left_crop": pair.left_crop, "right_crop": pair.right_crop,
                "left_subject": pair.left_subject, "right_subject": pair.right_subject,
                "age_gap": pair.age_gap, "score": float(score),
            })
    scores_path = private_dir / f"private_pair_scores_{model_name}.jsonl"
    write_jsonl(scores_path, rows)
    embeddings_path = private_dir / f"private_scored_embeddings_{model_name}.npz"
    np.savez_compressed(embeddings_path, indices=np.flatnonzero(row_map >= 0), row_map=row_map,
                        embeddings=np.asarray(embeddings, dtype=np.float32))
    return [scores_path, embeddings_path]


def run_breakdown(
    *, source_npz: Path = DEFAULT_SOURCE_NPZ, base_weights: Path = DEFAULT_BASE_WEIGHTS,
    tuned_checkpoints: dict[int, Path] | None = None, output_dir: Path = DEFAULT_OUTPUT_DIR,
    retrieval_cache_dir: Path = DEFAULT_RETRIEVAL_CACHE_DIR, n_boot: int = 2000, batch_size: int = 32,
    threads: int = 2, allow_embed: bool = False, modality_explicit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Root-execution orchestrator: build split pairs, reuse/embed, calibrate on dev, report test.

    Never called by the synthetic tests. Loads raw crops and, only if a verified cache is
    incomplete, runs local CPU embedding with ``threads <= 2``."""
    from scripts.fgnet_retrieval_study import _load_crops

    validate_configuration(n_boot=n_boot, batch_size=batch_size, threads=threads)
    tuned_checkpoints = tuned_checkpoints or default_tuned_checkpoints()
    validate_configuration(n_boot=n_boot, batch_size=batch_size, threads=threads, seeds=list(tuned_checkpoints))
    output_dir = Path(output_dir)
    if (output_dir / "fgnet_error_breakdown.json").exists():
        raise FileExistsError("completed result exists; choose a new output directory")
    private_dir = output_dir / "private"
    # Bind reused biometric caches to the original experiment's full checksums,
    # not just to a metadata-derived model/dataset cache key.
    from scripts.render_fgnet_evidence import verified_result

    retrieval_result = Path(retrieval_cache_dir).parent / "fgnet_retrieval_study.json"
    verified_result(retrieval_result, PROJECT_ROOT)
    print("Retrieval inputs and caches checksum-verified; legacy training provenance remains absent.", flush=True)
    crops, subjects, ages = _load_crops(Path(source_npz))
    source_sha256 = sha256_file(source_npz)
    protocol = build_protocol(subjects, ages, seed=SEED, dev_fraction=DEV_FRACTION, source_sha256=source_sha256)
    verify_protocol(protocol)
    required = np.arange(int(crops.shape[0]), dtype=np.int64)
    row_map = embedding_row_map(required, n_crops=int(crops.shape[0]))
    models: dict[str, dict[str, Any]] = {
        "frozen": {"role": "frozen_casia_webface", "weights_sha256": sha256_file(base_weights),
                   "loader": lambda: load_frozen_model(base_weights)}
    }
    for seed, checkpoint in sorted(tuned_checkpoints.items()):
        models[f"tuned_seed{seed}"] = {"role": f"tuned_facenet_seed{seed}", "weights_sha256": sha256_file(checkpoint),
                                       "loader": (lambda c=checkpoint: load_tuned_model(c))}

    pairs_by_split: dict[str, list[VerificationPair]] = {}
    pair_diag: dict[str, Any] = {}
    for split in ("dev", "test"):
        pairs_by_split[split], pair_diag[split] = build_split_pairs(subjects=subjects, ages=ages, protocol=protocol, split=split)
    crop_blur = crop_blur_variances(crops)
    cutpoints = blur_cutpoints(pair_blur(crop_blur, pairs_by_split["dev"]))

    splits_by_model: dict[str, Any] = {}
    model_meta: dict[str, Any] = {}
    private_outputs: list[Path] = []
    reused_inputs: list[Path] = []
    for name, spec in models.items():
        print(f"Evaluating {name} with fixed dev calibration...", flush=True)
        embeddings, meta = resolve_embeddings(
            role=spec["role"], weights_sha256=spec["weights_sha256"], source_sha256=source_sha256,
            required_indices=required, crops=crops,
            cache_candidates=[Path(retrieval_cache_dir) / f"private_embeddings_{name}.npz",
                              private_dir / f"private_embeddings_{name}.npz"],
            write_path=private_dir / f"private_embeddings_{name}.npz",
            load_model=spec["loader"], batch_size=batch_size, threads=threads, allow_embed=allow_embed,
        )
        model_meta[name] = meta
        private_outputs.append(private_dir / f"private_embeddings_{name}.npz")
        reused_inputs.append(Path(retrieval_cache_dir) / f"private_embeddings_{name}.npz")
        dev_pairs = pairs_by_split["dev"]
        dev_scores, dev_labels = scores_for_pairs(dev_pairs, embeddings, row_map)
        ops = {far: calibrate_operating_point(dev_scores[dev_labels == 0], dev_scores[dev_labels == 1], far) for far in FAR_TARGETS}
        splits_by_model[name] = {
            "dev": evaluate_split(split="dev", pairs=dev_pairs, scores=dev_scores, crop_blur=crop_blur,
                                  operating_points=ops, n_boot=n_boot, seed=_split_seed(SEED, name), blur_cutpoints_dev=cutpoints),
            "test": evaluate_split(split="test", pairs=pairs_by_split["test"],
                                   scores=scores_for_pairs(pairs_by_split["test"], embeddings, row_map)[0],
                                   crop_blur=crop_blur, operating_points=ops, n_boot=n_boot,
                                   seed=_split_seed(SEED, name), blur_cutpoints_dev=cutpoints),
        }
        private_outputs.extend(write_private_artifacts(
            private_dir, pairs_by_split=pairs_by_split, row_map=row_map,
            embeddings=embeddings, model_name=name))

    payload = build_public_output(
        protocol_hash=protocol.protocol_hash, source_sha256=source_sha256, split_summaries=splits_by_model,
        pair_diagnostics=pair_diag, model_meta=model_meta, blur_cutpoints=cutpoints,
        modality=modality_axes(modality_explicit), n_boot=n_boot,
    )
    assert_public_output_has_no_ids(payload)
    payload = sanitize_nonfinite(payload)
    public_path = output_dir / "fgnet_error_breakdown.json"
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_experiment_manifest(
        output_dir / "fgnet_error_breakdown.manifest.json",
        experiment="fgnet-endpoint-age-matched-error-breakdown",
        parameters={"protocol_hash": protocol.protocol_hash, "acceptance_convention": "strict>",
                    "far_targets": list(FAR_TARGETS), "endpoint_age_tolerance": ENDPOINT_AGE_TOLERANCE,
                    "n_boot": n_boot, "bootstrap_seed": SEED, "protocol_seed": SEED,
                    "training_seeds": sorted(tuned_checkpoints), "batch_size": batch_size,
                    "preprocessing": PREPROCESSING,
                    "threads": threads, "device": "cpu", "network_calls": False,
                    "training_identity_independence": TRAINING_INDEPENDENCE},
        metrics={"n_required_crops": int(required.size), "n_models": len(models)},
        inputs=[source_npz, base_weights, *tuned_checkpoints.values(),
                retrieval_result, retrieval_result.with_suffix(".manifest.json"),
                Path(__file__), PROJECT_ROOT / "scripts/fgnet_retrieval_study.py", *reused_inputs],
        outputs=[public_path, *private_outputs],
    )
    return payload


def default_tuned_checkpoints() -> dict[int, Path]:
    return {seed: PROJECT_ROOT / "models" / f"bb_facenet_seed{seed}.pt" for seed in DEFAULT_TUNED_SEEDS}


def main() -> None:
    parser = argparse.ArgumentParser(description="FG-NET endpoint-age-matched error breakdown.")
    parser.add_argument("--source-npz", type=Path, default=DEFAULT_SOURCE_NPZ)
    parser.add_argument("--base-weights", type=Path, default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--retrieval-cache-dir", type=Path, default=DEFAULT_RETRIEVAL_CACHE_DIR)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--execute", action="store_true", help="required: embed any missing crops locally on CPU")
    parser.add_argument("--plan-only", action="store_true", help="build the protocol and print split sizes; no crops embedded")
    args = parser.parse_args()
    if args.plan_only:
        with np.load(args.source_npz, allow_pickle=False) as archive:
            subjects = np.asarray(archive["subjects"], dtype=np.int64)
            ages = np.asarray(archive["ages"], dtype=np.int64)
        protocol = build_protocol(subjects, ages, seed=SEED, dev_fraction=DEV_FRACTION,
                                  source_sha256=sha256_file(args.source_npz))
        print(f"protocol_hash={protocol.protocol_hash} gallery={protocol.gallery_size} queries={protocol.n_queries}")
        return
    if not args.execute:
        print("dry run: pass --execute or --plan-only explicitly; synthetic tests use the pure functions")
        return
    run_breakdown(
        source_npz=args.source_npz, base_weights=args.base_weights, output_dir=args.output_dir,
        retrieval_cache_dir=args.retrieval_cache_dir, n_boot=args.n_boot, batch_size=args.batch_size,
        threads=args.threads, allow_embed=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "OperatingPoint", "VerificationPair", "blur_bin_labels", "blur_cutpoints", "bootstrap_fmr_fnmr",
    "build_public_output", "build_split_pairs", "calibrate_operating_point", "counts_at_threshold",
    "crop_blur_variances", "evaluate_split", "evaluate_stratum", "modality_axes", "pair_arrays",
    "pair_blur", "pair_cosine_scores", "resolve_embeddings", "run_breakdown", "scores_for_pairs", "split_strata",
]
