"""Candidate script: fixed cross-age *retrieval* protocol on FG-NET crops.

This file is a ready-to-integrate candidate. It is intentionally NOT placed under
``scripts/`` and performs no canonical repository edits. It implements the
pre-registered retrieval protocol requested for task ``retrieval-protocol``:

* One **oldest-age gallery image per identity** (deterministic tie break: lowest
  original crop index). Every other distinct, strictly younger image of the same
  identity is a query.
* Gallery and query images are **image-disjoint** by construction, but identities
  are necessarily shared (a query's positive is its own identity's gallery image).
* Identities are partitioned into **development / test** by a fixed protocol seed
  *before* any scoring.
* Frozen (local CASIA-WebFace) and tuned (``bb_facenet_seed{42,1,2}.pt``) FaceNet
  backbones are scored with **equal preprocessing**; each unique crop is embedded
  exactly once and cached.
* Recall@1/5/10, MRR, CMC, and stratified age-gap + child->adult query metrics are
  reported with a **paired query-subject bootstrap CI conditional on the fixed
  gallery**. The gallery is never resampled.

Explicit non-claims (also emitted in the public aggregate output):

* This is a *retrieval* protocol; it must never be described as an
  "identity-disjoint gallery and query identities" protocol.
* ``unverified training-identity independence``: whether the tuned checkpoints'
  training identities overlap FG-NET identities is not adjudicated here.
* Recall@K equals CMC@K because each identity contributes exactly one gallery
  image (exactly one relevant item per query).
* There is a small gallery and **no gallery resampling**; the reported CIs are
  therefore conditional on that fixed gallery.

The heavy work (torch inference) is gated behind :func:`main` / ``__main__``. The
protocol and metric functions are pure NumPy so the synthetic test-suite runs
without GPU, network access, or image inference.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from age_gap.common.io import PROJECT_ROOT, data_path, write_jsonl
from age_gap.common.manifest import sha256_file, write_experiment_manifest

# ── Fixed protocol constants ────────────────────────────────────────────────
PROTOCOL_NAME = "fgnet_oldest_gallery_cross_age_retrieval_v1"
DEFAULT_PROTOCOL_SEED = 42
DEFAULT_DEV_FRACTION = 0.5
DEFAULT_SEEDS = (42, 1, 2)
KS = (1, 5, 10)
MAX_CMC_RANK = 50

# Age-gap strata over (gallery_age - query_age). Query age is strictly younger
# than the gallery age, so the minimum gap is 1.
AGE_GAP_BINS: tuple[tuple[int, int | None, str], ...] = (
    (1, 9, "gap_01_09"),
    (10, 19, "gap_10_19"),
    (20, 24, "gap_20_24"),
    (25, None, "gap_25_plus"),
)
CHILD_TO_ADULT_LABEL = "child_query_to_adult_gallery"
CHILD_MAX_AGE = 13
ADULT_MIN_AGE = 25

# Single, shared preprocessing description. Both frozen and tuned backbones use
# the exact same routine (``age_gap.models.facenet.preprocess_bgr``): BGR uint8
# -> RGB 160x160 CHW float32, normalization (x - 127.5) / 128.
PREPROCESSING = {
    "function": "age_gap.models.facenet.preprocess_bgr",
    "input_size": 160,
    "channel_order": "RGB",
    "normalization": "(x - 127.5) / 128",
    "source_crop_format": "BGR uint8 112x112",
    "shared_between_frozen_and_tuned": True,
}

DISCLOSURES: tuple[str, ...] = (
    "Protocol is NOT identity-disjoint: gallery and query identities are necessarily "
    "shared so that each query has a positive gallery item.",
    "Gallery and query *images* are disjoint by construction (a query is never the "
    "identity's selected gallery image).",
    "unverified training-identity independence: overlap between the tuned checkpoints' "
    "training identities and FG-NET identities is not adjudicated by this study.",
    "Recall@K equals CMC@K because every identity contributes exactly one gallery image, "
    "so there is exactly one relevant gallery item per query.",
    "The gallery is small and is never resampled; reported paired bootstrap CIs are "
    "conditional on the fixed gallery.",
    "Checkpoint training provenance is disclosed, not verified: the frozen backbone is "
    "declared pretrained on CASIA-WebFace and the tuned checkpoints are trained on "
    "data/processed/pairs.jsonl, but exact identity/image overlap with FG-NET is not "
    "adjudicated (training_identity_independence = unverified).",
    "Public aggregate output contains no subject identifiers and no images.",
)

# Local CASIA-WebFace FaceNet weights. Loading is offline; network loading is disabled.
DEFAULT_BASE_WEIGHTS = Path.home() / ".cache" / "torch" / "checkpoints" / "20180408-102900-casia-webface.pt"

# Integrated default output location (public aggregate + manifest + private/ subdir).
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "metrics" / "fgnet_retrieval"
# Declared training data / provenance inventory; used when present, never fabricated.
DEFAULT_MODEL_INVENTORY = PROJECT_ROOT / "metrics" / "model_inventory.json"
DEFAULT_TRAINING_PAIRS = PROJECT_ROOT / "data" / "processed" / "pairs.jsonl"


# ── Canonical hashing helpers ───────────────────────────────────────────────
def canonical_json(payload: Any) -> str:
    """Deterministic JSON text used for protocol hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Protocol construction (pure NumPy) ──────────────────────────────────────
@dataclass(frozen=True)
class RetrievalProtocol:
    """Frozen, hashable description of the oldest-gallery retrieval protocol."""

    name: str
    seed: int
    dev_fraction: float
    source_sha256: str | None
    gallery_index: np.ndarray  # (G,) original crop index of the selected gallery image
    gallery_subject: np.ndarray  # (G,) identity of each gallery image
    gallery_age: np.ndarray  # (G,) age of each gallery image
    query_index: np.ndarray  # (Q,) original crop index of each query image
    query_subject: np.ndarray  # (Q,) identity of each query image
    query_age: np.ndarray  # (Q,) age of each query image
    query_split: np.ndarray  # (Q,) "<U4" dev/test
    n_excluded_max_age_ties: int
    protocol_hash: str

    @property
    def gallery_size(self) -> int:
        return int(self.gallery_index.size)
    @property
    def n_queries(self) -> int:
        return int(self.query_index.size)

    def split_mask(self, split: str) -> np.ndarray:
        return self.query_split == split

    def gallery_age_by_subject(self) -> dict[int, int]:
        return {int(s): int(a) for s, a in zip(self.gallery_subject, self.gallery_age, strict=True)}

    def age_gap(self) -> np.ndarray:
        """gallery_age - query_age for every query, in protocol order."""
        mapping = self.gallery_age_by_subject()
        return np.asarray(
            [mapping[int(s)] - int(a) for s, a in zip(self.query_subject, self.query_age, strict=True)],
            dtype=np.int64,
        )

    def as_payload(self) -> dict[str, Any]:
        """Canonical private payload. Never written to the public aggregate output."""
        return {
            "protocol_name": self.name,
            "protocol_seed": int(self.seed),
            "dev_fraction": float(self.dev_fraction),
            "source_sha256": self.source_sha256,
            "gallery_rule": "max_age__lowest_original_index",
            "query_rule": "age_strictly_less_than_selected_gallery_age",
            "tie_break": "stable_ranking_by_gallery_index",
            "gallery": [
                [int(s), int(a), int(i)]
                for s, a, i in zip(self.gallery_subject, self.gallery_age, self.gallery_index, strict=True)
            ],
            "queries": [
                [int(s), int(a), int(i), str(sp)]
                for s, a, i, sp in zip(
                    self.query_subject, self.query_age, self.query_index, self.query_split, strict=True
                )
            ],
        }

    def unique_indices(self) -> np.ndarray:
        """Sorted unique original crop indices required to score both splits."""
        return np.unique(np.concatenate([self.gallery_index, self.query_index]).astype(np.int64))

    def embedding_row_map(self, *, n_crops: int | None = None) -> np.ndarray:
        """Explicit map ``original crop index -> row in the unique-embedding matrix``.

        Gallery/query indices are original ``crops`` indices and need NOT be
        contiguous (excluded max-age ties leave gaps). Never index an embedding
        matrix with a raw crop index; always go through this map. Entries for
        crops that were not embedded are ``-1``.
        """
        return embedding_row_map(self.unique_indices(), n_crops=n_crops)


def embedding_row_map(indices: np.ndarray, *, n_crops: int | None = None) -> np.ndarray:
    """Build the explicit ``original crop index -> embedding row`` lookup.

    ``indices`` are the original crop indices that were actually embedded (one row
    per entry, in the given order). The returned array has one slot per original
    crop index (``n_crops`` slots, inferred from the maximum index when omitted);
    a slot is ``-1`` when that crop index has no embedding row. This makes omitted
    interior indices an explicit error instead of a silent misalignment.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("embedding indices must be a 1D array")
    if indices.size and np.unique(indices).size != indices.size:
        raise ValueError("embedding indices must be unique; each crop is embedded exactly once")
    if indices.size and np.any(indices < 0):
        raise ValueError("embedding indices must be non-negative")
    inferred = int(indices.max()) + 1 if indices.size else 0
    size = inferred if n_crops is None else int(n_crops)
    if size < inferred:
        raise ValueError(f"n_crops={size} cannot hold embedding index {inferred - 1}")
    mapping = np.full(size, -1, dtype=np.int64)
    mapping[indices] = np.arange(indices.size, dtype=np.int64)
    return mapping


def _require_rows(crop_indices: np.ndarray, row_map: np.ndarray, *, what: str) -> np.ndarray:
    crop_indices = np.asarray(crop_indices, dtype=np.int64)
    if crop_indices.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(crop_indices < 0) or np.any(crop_indices >= row_map.size):
        raise ValueError(f"{what} references a crop index outside the embedding row map")
    rows = row_map[crop_indices]
    missing = crop_indices[rows < 0]
    if missing.size:
        raise ValueError(
            f"{what} requires embeddings that were not computed for crop indices "
            f"{missing[:8].tolist()} (no silent padding of uncomputed rows)"
        )
    return rows


def _validate_metadata(subjects: np.ndarray, ages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    subjects = np.asarray(subjects)
    ages = np.asarray(ages)
    if subjects.ndim != 1 or ages.ndim != 1 or subjects.shape != ages.shape:
        raise ValueError("subjects and ages must be equal-length 1D arrays")
    if subjects.size == 0:
        raise ValueError("at least one image is required")
    if not np.issubdtype(subjects.dtype, np.integer) or not np.issubdtype(ages.dtype, np.integer):
        raise ValueError("subjects and ages must be integer arrays")
    if np.any(ages < 0):
        raise ValueError("ages must be non-negative")
    return subjects.astype(np.int64), ages.astype(np.int64)


def build_protocol(
    subjects: np.ndarray,
    ages: np.ndarray,
    *,
    seed: int = DEFAULT_PROTOCOL_SEED,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    source_sha256: str | None = None,
    name: str = PROTOCOL_NAME,
) -> RetrievalProtocol:
    """Build the fixed oldest-gallery cross-age retrieval protocol.

    * One gallery image per identity: the image with maximum age; ties are broken
      by the lowest original crop index. Remaining max-age ties are excluded and
      counted in ``n_excluded_max_age_ties``.
    * Queries: every other image with age strictly below the selected gallery age.
      This makes gallery and query images disjoint.
    * Identities are split into dev/test by ``seed`` *before* scoring.
    """
    subjects, ages = _validate_metadata(subjects, ages)
    if not 0.0 <= dev_fraction < 1.0:
        raise ValueError("dev_fraction must be in [0, 1)")

    unique_subjects = np.unique(subjects)  # ascending, deterministic
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(unique_subjects.size)
    n_dev = int(round(unique_subjects.size * float(dev_fraction)))
    dev_subjects = set(unique_subjects[permutation[:n_dev]].tolist())

    gallery_index: list[int] = []
    gallery_subject: list[int] = []
    gallery_age: list[int] = []
    query_index: list[int] = []
    query_subject: list[int] = []
    query_age: list[int] = []
    query_split: list[str] = []
    n_excluded = 0

    for subject in unique_subjects.tolist():
        positions = np.flatnonzero(subjects == subject)  # ascending original indices
        subject_ages = ages[positions]
        max_age = int(subject_ages.max())
        max_positions = np.flatnonzero(subject_ages == max_age)  # ascending
        chosen_pos = int(max_positions[0])
        chosen_index = int(positions[chosen_pos])
        n_excluded += int(max_positions.size - 1)

        split = "dev" if subject in dev_subjects else "test"
        gallery_index.append(chosen_index)
        gallery_subject.append(int(subject))
        gallery_age.append(max_age)
        for pos in positions.tolist():
            if pos == chosen_index:
                continue
            if int(ages[pos]) < max_age:
                query_index.append(int(pos))
                query_subject.append(int(subject))
                query_age.append(int(ages[pos]))
                query_split.append(split)

    protocol = RetrievalProtocol(
        name=str(name),
        seed=int(seed),
        dev_fraction=float(dev_fraction),
        source_sha256=source_sha256,
        gallery_index=np.asarray(gallery_index, dtype=np.int64),
        gallery_subject=np.asarray(gallery_subject, dtype=np.int64),
        gallery_age=np.asarray(gallery_age, dtype=np.int64),
        query_index=np.asarray(query_index, dtype=np.int64),
        query_subject=np.asarray(query_subject, dtype=np.int64),
        query_age=np.asarray(query_age, dtype=np.int64),
        query_split=np.asarray(query_split, dtype="<U4"),
        n_excluded_max_age_ties=int(n_excluded),
        protocol_hash="",
    )
    payload = protocol.as_payload()
    return RetrievalProtocol(**{**protocol.__dict__, "protocol_hash": sha256_text(canonical_json(payload))})


def verify_protocol(protocol: RetrievalProtocol) -> None:
    """Assert the protocol's structural invariants (image-disjointness, uniqueness)."""
    if np.intersect1d(protocol.gallery_index, protocol.query_index).size:
        raise AssertionError("gallery and query images must be disjoint")
    if np.unique(protocol.gallery_subject).size != protocol.gallery_subject.size:
        raise AssertionError("exactly one gallery image per identity is required")
    if np.unique(protocol.gallery_index).size != protocol.gallery_index.size:
        raise AssertionError("gallery indices must be unique")
    if np.unique(protocol.query_index).size != protocol.query_index.size:
        raise AssertionError("query indices must be unique")
    expected = sha256_text(canonical_json(protocol.as_payload()))
    if expected != protocol.protocol_hash:
        raise AssertionError("protocol hash does not match the canonical payload")
    mapping = protocol.gallery_age_by_subject()
    gaps = protocol.age_gap()
    if np.any(gaps < 1):
        raise AssertionError("every query must be strictly younger than its gallery image")
    for subject in np.unique(protocol.query_subject).tolist():
        if subject not in mapping:
            raise AssertionError(f"query subject {subject} has no gallery image")


def project_embeddings(
    embeddings: np.ndarray,
    protocol: RetrievalProtocol,
    row_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice unique-crop embeddings into gallery/query matrices using explicit rows.

    ``row_map`` must come from :meth:`RetrievalProtocol.embedding_row_map` (or
    :func:`embedding_row_map`) and maps original crop indices to embedding rows.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    row_map = np.asarray(row_map, dtype=np.int64)
    if row_map.ndim != 1:
        raise ValueError("row_map must be a 1D array")
    gallery_rows = _require_rows(protocol.gallery_index, row_map, what="gallery")
    query_rows = _require_rows(protocol.query_index, row_map, what="queries")
    if gallery_rows.size and int(gallery_rows.max()) >= embeddings.shape[0]:
        raise ValueError("embedding matrix has fewer rows than the gallery mapping requires")
    if query_rows.size and int(query_rows.max()) >= embeddings.shape[0]:
        raise ValueError("embedding matrix has fewer rows than the query mapping requires")
    return embeddings[gallery_rows], embeddings[query_rows]


def similarity_matrix(
    unique_embeddings: np.ndarray,
    protocol: RetrievalProtocol,
    row_map: np.ndarray,
) -> np.ndarray:
    gallery, queries = project_embeddings(unique_embeddings, protocol, row_map)
    return queries @ gallery.T


# ── Ranking and metrics (pure NumPy) ────────────────────────────────────────
def per_query_ranks(
    similarities: np.ndarray,
    query_subject: np.ndarray,
    gallery_subject: np.ndarray,
) -> np.ndarray:
    """Rank of the single relevant (same-identity) gallery item for each query.

    Deterministic tie handling: gallery items are ordered by descending similarity
    and, on exact ties, by ascending gallery position. ``rank = 1`` is best.
    Queries without a relevant gallery item yield ``nan``.
    """
    sims = np.asarray(similarities, dtype=np.float64)
    gallery_subject = np.asarray(gallery_subject)
    query_subject = np.asarray(query_subject)
    if sims.ndim != 2 or sims.shape[0] != query_subject.size or sims.shape[1] != gallery_subject.size:
        raise ValueError("similarity matrix shape does not match query/gallery subjects")
    gallery_position = {int(s): pos for pos, s in enumerate(gallery_subject.tolist())}
    if len(gallery_position) != gallery_subject.size:
        raise ValueError("gallery subjects must be unique (one gallery image per identity)")
    positions = np.arange(gallery_subject.size)
    ranks = np.full(query_subject.size, np.nan, dtype=np.float64)
    for row, subject in enumerate(query_subject.tolist()):
        relevant = gallery_position.get(int(subject))
        if relevant is None:
            continue
        relevant_similarity = sims[row, relevant]
        strictly_greater = int(np.count_nonzero(sims[row] > relevant_similarity))
        equal_before = int(np.count_nonzero((sims[row] == relevant_similarity) & (positions < relevant)))
        ranks[row] = strictly_greater + equal_before + 1
    return ranks


def metrics_from_ranks(ranks: np.ndarray, ks: Sequence[int] = KS) -> dict[str, float]:
    """Recall@K, MRR and median rank from a per-query rank vector.

    Empty strata return ``nan`` (not 0.0) so an unpopulated stratum is never
    mistaken for a genuine zero-recall result; ``n_queries`` is 0 and
    ``empty`` is True.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    valid = ranks[np.isfinite(ranks)]
    out: dict[str, float] = {
        "n_queries": float(valid.size),
        "n_unscored": float(ranks.size - valid.size),
        "empty": 1.0 if valid.size == 0 else 0.0,
    }
    if valid.size == 0:
        for k in ks:
            out[f"recall@{k}"] = float("nan")
        out["mrr"] = float("nan")
        out["median_rank"] = float("nan")
        return out
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(valid <= k))
    out["mrr"] = float(np.mean(1.0 / valid))
    out["median_rank"] = float(np.median(valid))
    return out


def cmc_curve(ranks: np.ndarray, max_rank: int = MAX_CMC_RANK) -> list[float]:
    """CMC values at ranks 1..max_rank (equals Recall@k for this protocol).

    The curve always has exactly ``max_rank`` entries; values saturate at 1.0 once
    every query's relevant item has been retrieved.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    valid = ranks[np.isfinite(ranks)]
    if valid.size == 0:
        return [float("nan")] * int(max_rank)
    return [float(np.mean(valid <= k)) for k in range(1, int(max_rank) + 1)]


def stratum_masks(protocol: RetrievalProtocol, split: str) -> dict[str, np.ndarray]:
    """Boolean masks over all queries for a split's overall, age-gap and child->adult strata."""
    base = protocol.split_mask(split)
    gap = protocol.age_gap()
    gallery_age_by_subject = protocol.gallery_age_by_subject()
    gallery_age = np.asarray([gallery_age_by_subject[int(s)] for s in protocol.query_subject.tolist()], dtype=np.int64)
    masks: dict[str, np.ndarray] = {"overall": base}
    for low, high, label in AGE_GAP_BINS:
        upper = np.inf if high is None else float(high)
        masks[label] = base & (gap >= low) & (gap <= upper)
    masks[CHILD_TO_ADULT_LABEL] = base & (protocol.query_age < CHILD_MAX_AGE) & (gallery_age > ADULT_MIN_AGE)
    return masks


def metrics_by_stratum(ranks: np.ndarray, masks: dict[str, np.ndarray], ks: Sequence[int] = KS) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label, mask in masks.items():
        selected = np.asarray(ranks)[mask]
        out[label] = metrics_from_ranks(selected, ks)
    return out


# ── Paired query-subject bootstrap (gallery fixed) ──────────────────────────
def _stratum_rng_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def _weighted_metrics(ranks: np.ndarray, weights: np.ndarray, ks: Sequence[int]) -> dict[str, float]:
    total = float(weights.sum())
    out: dict[str, float] = {"n_queries": total}
    for k in ks:
        out[f"recall@{k}"] = float(np.sum(weights * (ranks <= k)) / total)
    out["mrr"] = float(np.sum(weights / ranks) / total)
    return out


@dataclass(frozen=True)
class PairedBootstrapResult:
    stratum: str
    n_queries: int
    n_subjects: int
    n_valid_resamples: int
    n_requested_resamples: int
    frozen: dict[str, float]
    tuned: dict[str, float]
    delta: dict[str, float]
    frozen_ci95: dict[str, tuple[float, float]]
    tuned_ci95: dict[str, tuple[float, float]]
    delta_ci95: dict[str, tuple[float, float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_queries": int(self.n_queries),
            "n_subjects": int(self.n_subjects),
            "n_valid_resamples": int(self.n_valid_resamples),
            "n_requested_resamples": int(self.n_requested_resamples),
            "frozen": {k: float(v) for k, v in self.frozen.items()},
            "tuned": {k: float(v) for k, v in self.tuned.items()},
            "delta": {k: float(v) for k, v in self.delta.items()},
            "frozen_ci95": {k: [float(a), float(b)] for k, (a, b) in self.frozen_ci95.items()},
            "tuned_ci95": {k: [float(a), float(b)] for k, (a, b) in self.tuned_ci95.items()},
            "delta_ci95": {k: [float(a), float(b)] for k, (a, b) in self.delta_ci95.items()},
        }


def paired_subject_bootstrap(
    ranks_frozen: np.ndarray,
    ranks_tuned: np.ndarray,
    query_subject: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ks: Sequence[int] = KS,
) -> dict[str, PairedBootstrapResult]:
    """Cluster bootstrap over query subjects with the gallery held fixed.

    Each replicate resamples the *query subjects* present in a stratum with
    replacement and reuses the same multiplicities for the frozen and tuned rank
    vectors (paired). The gallery is never resampled, so the intervals are
    conditional on the fixed gallery. Subjects with no query in a stratum are absent.
    """
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1")
    ranks_frozen = np.asarray(ranks_frozen, dtype=np.float64)
    ranks_tuned = np.asarray(ranks_tuned, dtype=np.float64)
    query_subject = np.asarray(query_subject)
    if ranks_frozen.shape != ranks_tuned.shape or ranks_frozen.shape != query_subject.shape:
        raise ValueError("rank vectors and query subjects must be equal-length")
    quantiles = np.asarray([2.5, 97.5])

    results: dict[str, PairedBootstrapResult] = {}
    for label, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        rf = ranks_frozen[mask]
        rt = ranks_tuned[mask]
        subjects = query_subject[mask]
        finite = np.isfinite(rf) & np.isfinite(rt)
        rf, rt, subjects = rf[finite], rt[finite], subjects[finite]
        if rf.size == 0:
            # Empty stratum: emit explicit null metrics, never a misleading 0.0.
            nan_metrics = {**{f"recall@{k}": float("nan") for k in ks}, "mrr": float("nan")}
            nan_ci = {metric: (float("nan"), float("nan")) for metric in nan_metrics}
            results[label] = PairedBootstrapResult(
                stratum=label,
                n_queries=0,
                n_subjects=0,
                n_valid_resamples=0,
                n_requested_resamples=int(n_boot),
                frozen=dict(nan_metrics),
                tuned=dict(nan_metrics),
                delta={metric: float("nan") for metric in nan_metrics},
                frozen_ci95=dict(nan_ci),
                tuned_ci95=dict(nan_ci),
                delta_ci95=dict(nan_ci),
            )
            continue
        _, inverse = np.unique(subjects, return_inverse=True)
        n_subjects = int(inverse.max()) + 1
        point_frozen = _weighted_metrics(rf, np.ones_like(rf), ks)
        point_tuned = _weighted_metrics(rt, np.ones_like(rt), ks)
        rng = np.random.default_rng(_stratum_rng_seed(seed, label))
        point_frozen.pop("n_queries", None)
        point_tuned.pop("n_queries", None)
        sampled_frozen: list[dict[str, float]] = []
        sampled_tuned: list[dict[str, float]] = []
        for _ in range(int(n_boot)):
            draw = rng.integers(n_subjects, size=n_subjects)
            multiplicity = np.bincount(draw, minlength=n_subjects).astype(np.float64)
            weights = multiplicity[inverse]
            if weights.sum() == 0:
                continue
            sampled_frozen.append(_weighted_metrics(rf, weights, ks))
            sampled_tuned.append(_weighted_metrics(rt, weights, ks))
        if not sampled_frozen:
            raise ValueError(f"no valid bootstrap resamples for stratum {label!r}")

        def interval(values: np.ndarray) -> tuple[float, float]:
            low, high = np.percentile(values, quantiles)
            return float(low), float(high)

        frozen_ci: dict[str, tuple[float, float]] = {}
        tuned_ci: dict[str, tuple[float, float]] = {}
        delta_ci: dict[str, tuple[float, float]] = {}
        delta_point: dict[str, float] = {}
        for metric in point_frozen:
            frozen_values = np.asarray([sample[metric] for sample in sampled_frozen])
            tuned_values = np.asarray([sample[metric] for sample in sampled_tuned])
            frozen_ci[metric] = interval(frozen_values)
            tuned_ci[metric] = interval(tuned_values)
            delta_ci[metric] = interval(tuned_values - frozen_values)
            delta_point[metric] = float(point_tuned[metric] - point_frozen[metric])
        results[label] = PairedBootstrapResult(
            stratum=label,
            n_queries=int(rf.size),
            n_subjects=n_subjects,
            n_valid_resamples=len(sampled_frozen),
            n_requested_resamples=int(n_boot),
            frozen={k: float(v) for k, v in point_frozen.items()},
            tuned={k: float(v) for k, v in point_tuned.items()},
            delta=delta_point,
            frozen_ci95=frozen_ci,
            tuned_ci95=tuned_ci,
            delta_ci95=delta_ci,
        )
    return results


# ── Public aggregate output (no identifiers, no images) ─────────────────────
def build_public_output(
    *,
    protocol: RetrievalProtocol,
    source: dict[str, Any],
    models: dict[str, Any],
    results: dict[str, Any],
    primary_split: str = "test",
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the publishable aggregate. Must contain no subject IDs or images."""
    return {
        "task_id": "retrieval-protocol",
        "protocol_name": protocol.name,
        "protocol_hash": protocol.protocol_hash,
        "protocol_seed": int(protocol.seed),
        "dev_fraction": float(protocol.dev_fraction),
        "source": source,
        "preprocessing": PREPROCESSING,
        "cpu_threads": 2,
        "primary_split": primary_split,
        "models": models,
        "provenance": provenance
        if provenance is not None
        else {
            "checkpoint_training_manifests": "absent for legacy FaceNet checkpoints; current pair file is a declared source, not independently bound training provenance",
        "training_identity_independence": "unverified",
            "note": "Declared provenance not supplied by caller; overlap with FG-NET is "
            "not adjudicated.",
        },
        "training_identity_independence": "unverified",
        "gallery_resampling": "none (fixed gallery)",
        "disclosures": list(DISCLOSURES),
        "n_excluded_max_age_ties": int(protocol.n_excluded_max_age_ties),
        "results": results,
        "privacy": "Public aggregate only: no subject identifiers and no images. "
        "Per-query rows remain in a gitignored private artifact.",
    }


# Keys that would indicate row-level identity/image data if they appeared anywhere.
FORBIDDEN_PUBLIC_KEYS: frozenset[str] = frozenset(
    {
        "subject",
        "subjects",
        "subject_id",
        "subject_ids",
        "identity",
        "identities",
        "identity_id",
        "crop_index",
        "crop_indices",
        "image",
        "images",
        "image_index",
        "face_id",
        "gallery",
        "queries",
        "query_index",
        "retrieval_rows",
        "path",
    }
)
# Longest integer run tolerated in aggregate output (CMC lists are floats; labels are strings).
MAX_INT_RUN = 8
# A single string this long would indicate an embedded image or base64 blob.
MAX_STRING_LENGTH = 500


def assert_public_output_has_no_ids(payload: dict[str, Any]) -> None:
    """Structurally fail if the aggregate could carry subject identifiers or images.

    The public payload is assembled from an explicit aggregate-only whitelist; this
    guard catches accidental regressions that would add per-subject/per-image data.
    """

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in FORBIDDEN_PUBLIC_KEYS:
                    raise AssertionError(f"public output contains row-level key {path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            int_run = 0
            for item in node:
                if isinstance(item, bool) or not isinstance(item, int):
                    int_run = 0
                else:
                    int_run += 1
                    if int_run > MAX_INT_RUN:
                        raise AssertionError(f"public output contains an integer vector at {path}")

                walk(item, path)
        elif isinstance(node, str) and len(node) > MAX_STRING_LENGTH:
            raise AssertionError(f"public output contains an over-long string at {path}")

    walk(payload, "$")


# ── Model loading / embedding (torch, lazy; offline, CPU) ───────────────────
def load_frozen_model(base_weights: Path | str):
    """Load the frozen CASIA-WebFace FaceNet from a local checkpoint. No network."""
    import torch

    from age_gap.models.facenet import FaceNetBackbone

    base_weights = Path(base_weights)
    if not base_weights.is_file():
        raise FileNotFoundError(f"local base weights not found (network disabled): {base_weights}")
    model = FaceNetBackbone(pretrained=None)
    state = torch.load(base_weights, map_location="cpu", weights_only=True)
    incompatible = model.net.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"base weights missing keys: {incompatible.missing_keys[:8]}")
    if any("logits" not in key for key in incompatible.unexpected_keys):
        raise RuntimeError(f"unexpected base-weight keys: {incompatible.unexpected_keys[:8]}")
    return model.cpu().eval()


def load_tuned_model(checkpoint: Path | str):
    """Load a fine-tuned FaceNet checkpoint (``net.``-prefixed state dict). No network."""
    import torch

    from age_gap.models.facenet import FaceNetBackbone

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"tuned checkpoint not found: {checkpoint}")
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data
    model = FaceNetBackbone(pretrained=None)  # facenet-pytorch requires None, not False
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"tuned checkpoint missing keys: {incompatible.missing_keys[:8]}")
    if any("logits" not in key for key in incompatible.unexpected_keys):
        raise RuntimeError(f"unexpected tuned-weight keys: {incompatible.unexpected_keys[:8]}")
    return model.cpu().eval()


def embedding_cache_key(
    *,
    role: str,
    weights_sha256: str,
    source_sha256: str,
    indices: np.ndarray,
) -> str:
    """Deterministic cache key binding weights, source data, preprocessing and crop indices."""
    payload = {
        "role": role,
        "weights_sha256": weights_sha256,
        "source_sha256": source_sha256,
        "preprocessing": PREPROCESSING,
        "indices": [int(i) for i in np.asarray(indices).tolist()],
    }
    return sha256_text(canonical_json(payload))


_THREADS_CONFIGURED = False


def configure_torch_threads(threads: int = 2) -> None:
    """Pin CPU threads once (torch forbids changing interop threads after parallel work)."""
    global _THREADS_CONFIGURED
    threads = int(threads)
    if not 1 <= threads <= 2:
        raise ValueError("threads must be in 1..2 (CPU budget)")
    if _THREADS_CONFIGURED:
        return
    import torch

    torch.set_num_threads(int(threads))  # CPU threads <= 2 as required
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)  # already started; intra-op threads are already constrained
    _THREADS_CONFIGURED = True


def embed_indices(
    crops: np.ndarray,
    indices: np.ndarray,
    model: Any,
    *,
    batch_size: int = 32,
    threads: int = 2,
) -> np.ndarray:
    """Embed each unique crop index exactly once with the shared preprocessing."""
    import torch

    from age_gap.models.facenet import preprocess_bgr

    configure_torch_threads(threads)
    crops = np.asarray(crops)
    indices = np.asarray(indices, dtype=np.int64)
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    if indices.ndim != 1:
        raise ValueError("indices must be a 1D array")
    if np.unique(indices).size != indices.size:
        raise ValueError("indices must be unique; each crop is embedded exactly once")
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, indices.size, int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            batch = np.stack([preprocess_bgr(crops[int(i)]) for i in batch_indices.tolist()])
            latent = model(torch.from_numpy(batch))
            parts.append(np.asarray(latent.detach().cpu().numpy(), dtype=np.float32))
    return np.concatenate(parts) if parts else np.empty((0, 0), dtype=np.float32)


def embed_unique_crops(
    crops: np.ndarray,
    indices: np.ndarray,
    model: Any,
    *,
    role: str,
    weights_sha256: str,
    source_sha256: str,
    cache_path: Path,
    batch_size: int = 32,
    threads: int = 2,
    reuse_cache: bool = True,
) -> tuple[np.ndarray, bool]:
    """Return embeddings for ``indices``, reusing a matching cache when present.

    Returns ``(embeddings, cache_hit)``.
    """
    indices = np.asarray(indices, dtype=np.int64)
    key = embedding_cache_key(role=role, weights_sha256=weights_sha256, source_sha256=source_sha256, indices=indices)
    if reuse_cache and Path(cache_path).is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            cached_key = str(archive["key"]) if "key" in archive else ""
            cached_indices = np.asarray(archive["indices"], dtype=np.int64)
            if cached_key == key and np.array_equal(cached_indices, indices):
                return np.asarray(archive["embeddings"], dtype=np.float32), True
    embeddings = embed_indices(crops, indices, model, batch_size=batch_size, threads=threads)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, key=np.asarray(key), indices=indices, embeddings=embeddings)
    return embeddings, False


# ── Orchestration ───────────────────────────────────────────────────────────
def _load_crops(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as archive:
        crops = np.asarray(archive["crops"])
        subjects = np.asarray(archive["subjects"], dtype=np.int64)
        ages = np.asarray(archive["ages"], dtype=np.int64)
    return crops, subjects, ages


def _source_summary(npz_path: Path, subjects: np.ndarray, ages: np.ndarray) -> dict[str, Any]:
    return {
        "npz_sha256": sha256_file(npz_path),
        "n_images": int(subjects.size),
        "n_identities": int(np.unique(subjects).size),
        "age_min": int(ages.min()),
        "age_max": int(ages.max()),
    }


def sanitize_nonfinite(payload: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` so JSON stays valid.

    Empty strata emit ``nan`` internally; they must serialize as JSON ``null``
    rather than the non-standard ``NaN`` token.
    """
    if isinstance(payload, dict):
        return {key: sanitize_nonfinite(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [sanitize_nonfinite(item) for item in payload]
    if isinstance(payload, (float, np.floating)):
        value = float(payload)
        return value if np.isfinite(value) else None
    if isinstance(payload, (np.integer,)):
        return int(payload)
    if isinstance(payload, np.bool_):
        return bool(payload)
    return payload


def _write_private_plan(protocol: RetrievalProtocol, private_dir: Path) -> tuple[Path, Path]:
    private_dir.mkdir(parents=True, exist_ok=True)
    plan_path = private_dir / "protocol_plan.json"
    plan_path.write_text(canonical_json(protocol.as_payload()) + "\n", encoding="utf-8")
    layout_path = private_dir / "retrieval_plan.jsonl"
    gallery_age_by_subject = protocol.gallery_age_by_subject()
    rows = [
        {
            "split": str(split),
            "subject": int(subject),
            "crop_index": int(index),
            "age": int(age),
            "gallery_age": int(gallery_age_by_subject[int(subject)]),
        }
        for subject, index, age, split in zip(
            protocol.query_subject, protocol.query_index, protocol.query_age, protocol.query_split, strict=True
        )
    ]
    write_jsonl(layout_path, rows)
    return plan_path, layout_path


def _provenance_summary(
    *,
    base_weights: Path,
    tuned_checkpoints: dict[int, Path],
    model_inventory_path: Path | None,
    training_pairs: Path | None,
    source_sha256: str,
) -> dict[str, Any]:
    """Declared (not verified) training provenance for every scored model.

    Values are read from files that exist; missing inventory is reported as
    unavailable rather than fabricated. ``training_identity_independence`` stays
    ``unverified`` because exact identity/image overlap with FG-NET is not
    adjudicated here.
    """
    inventory: dict[str, Any] | None = None
    if model_inventory_path is not None and Path(model_inventory_path).is_file():
        with open(model_inventory_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        inventory = payload.get("models", payload) if isinstance(payload, dict) else None

    def _declared(name: str) -> dict[str, Any]:
        if not isinstance(inventory, dict) or name not in inventory:
            return {"declared_training_data": "not found in model inventory"}
        entry = inventory[name]
        return {
            "source": entry.get("source"),
            "role": entry.get("role"),
            "declared_training_data": entry.get("declared_training_data"),
        }

    tuned = {
        str(seed): _declared("facenet_casia")
        for seed in sorted(tuned_checkpoints)
    }
    return {
        "base_weights": {
            "path_name": base_weights.name,
            "declared_training_data": _declared("facenet_casia").get("declared_training_data"),
            "source": _declared("facenet_casia").get("source"),
        },
        "tuned_checkpoints": tuned,
        "tuned_training_pairs": (
            Path(training_pairs).name if training_pairs is not None else "not provided"
        ),
        "tuned_training_pairs_sha256": (
            sha256_file(training_pairs) if training_pairs is not None and Path(training_pairs).is_file() else None
        ),
        "model_inventory": Path(model_inventory_path).name if model_inventory_path is not None else "not provided",
        "model_inventory_read": inventory is not None,
        "checkpoint_training_manifest_status": "legacy training manifests absent; current pairs are a declared source, not independently bound training provenance",
        "benchmark_source_sha256": source_sha256,
        "training_identity_independence": "unverified",
        "note": "Declared provenance is reported as-is; identity/image overlap with FG-NET "
        "is not adjudicated.",
    }


def _validate_run_inputs(
    *,
    n_boot: int,
    batch_size: int,
    threads: int,
    dev_fraction: float,
    tuned_checkpoints: dict[int, Path],
) -> None:
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    if not 1 <= int(threads) <= 2:
        raise ValueError("threads must be in 1..2 (CPU budget)")
    if not 0.0 <= float(dev_fraction) < 1.0:
        raise ValueError("dev_fraction must be in [0, 1)")
    if not tuned_checkpoints:
        raise ValueError("at least one tuned checkpoint is required")
    for seed, checkpoint in tuned_checkpoints.items():
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"tuned checkpoint for seed {seed} not found: {checkpoint}")


def run_study(
    *,
    npz_path: Path,
    base_weights: Path,
    output_dir: Path,
    tuned_checkpoints: dict[int, Path],
    protocol_seed: int = DEFAULT_PROTOCOL_SEED,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    n_boot: int = 2000,
    batch_size: int = 32,
    threads: int = 2,
    plan_only: bool = False,
    reuse_cache: bool = True,
    expect_protocol_hash: str | None = None,
    model_inventory_path: Path | None = DEFAULT_MODEL_INVENTORY,
    training_pairs: Path | None = DEFAULT_TRAINING_PAIRS,
) -> dict[str, Any]:
    """Run the full retrieval study (or only build the protocol when ``plan_only``)."""
    started_at = time.perf_counter()
    _validate_run_inputs(
        n_boot=n_boot,
        batch_size=batch_size,
        threads=threads,
        dev_fraction=dev_fraction,
        tuned_checkpoints=tuned_checkpoints,
    )
    output_dir = Path(output_dir)
    private_dir = output_dir / "private"
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    crops = None
    subjects = ages = None
    if not plan_only:
        crops, subjects, ages = _load_crops(npz_path)
    else:
        with np.load(npz_path, allow_pickle=False) as archive:
            subjects = np.asarray(archive["subjects"], dtype=np.int64)
            ages = np.asarray(archive["ages"], dtype=np.int64)

    source_sha256 = sha256_file(npz_path)
    protocol = build_protocol(subjects, ages, seed=protocol_seed, dev_fraction=dev_fraction, source_sha256=source_sha256)
    verify_protocol(protocol)
    if expect_protocol_hash is not None and protocol.protocol_hash != expect_protocol_hash:
        raise ValueError(
            f"protocol hash mismatch: expected {expect_protocol_hash}, built {protocol.protocol_hash}"
        )

    plan_path, layout_path = _write_private_plan(protocol, private_dir)
    print(f"protocol_hash={protocol.protocol_hash}")
    print(
        f"gallery={protocol.gallery_size} queries={protocol.n_queries} "
        f"excluded_max_age_ties={protocol.n_excluded_max_age_ties}"
    )
    if plan_only:
        (private_dir / "protocol_hash.txt").write_text(protocol.protocol_hash + "\n", encoding="utf-8")
        return {
            "status": "plan_only",
            "protocol_hash": protocol.protocol_hash,
            "gallery_size": protocol.gallery_size,
            "n_queries": protocol.n_queries,
            "private_plan": str(plan_path),
            "private_layout": str(layout_path),
        }

    assert crops is not None
    unique_indices = protocol.unique_indices()
    assert int(crops.shape[0]) == int(subjects.size), "crops and metadata must align by original index"
    row_map = protocol.embedding_row_map(n_crops=int(crops.shape[0]))
    source = _source_summary(npz_path, subjects, ages)  # type: ignore[arg-type]
    provenance = _provenance_summary(
        base_weights=Path(base_weights),
        tuned_checkpoints=tuned_checkpoints,
        model_inventory_path=model_inventory_path,
        training_pairs=training_pairs,
        source_sha256=source_sha256,
    )

    # Frozen model.
    frozen_sha = sha256_file(base_weights)
    frozen_model = load_frozen_model(base_weights)
    frozen_embeddings, frozen_hit = embed_unique_crops(
        crops,
        unique_indices,
        frozen_model,
        role="frozen_casia_webface",
        weights_sha256=frozen_sha,
        source_sha256=source_sha256,
        cache_path=private_dir / "private_embeddings_frozen.npz",
        batch_size=batch_size,
        threads=threads,
        reuse_cache=reuse_cache,
    )
    frozen_sims = similarity_matrix(frozen_embeddings, protocol, row_map)
    frozen_ranks = per_query_ranks(frozen_sims, protocol.query_subject, protocol.gallery_subject)

    models_meta: dict[str, Any] = {
        "frozen": {
            "weights_sha256": frozen_sha,
            "n_images": int(unique_indices.size),
            "cache": "private/private_embeddings_frozen.npz",
            "cache_hit": bool(frozen_hit),
        },
        "tuned": {},
    }
    results: dict[str, Any] = {}
    age_gaps = protocol.age_gap()
    gallery_age_by_subject = protocol.gallery_age_by_subject()
    base_rows = [
        {
            "split": str(split),
            "subject": int(subject),
            "crop_index": int(index),
            "age": int(age),
            "gallery_age": int(gallery_age_by_subject[int(subject)]),
            "age_gap": int(age_gaps[row]),
            "frozen_rank": float(frozen_ranks[row]),
        }
        for row, (subject, index, age, split) in enumerate(
            zip(
                protocol.query_subject,
                protocol.query_index,
                protocol.query_age,
                protocol.query_split,
                strict=True,
            )
        )
    ]

    for seed, checkpoint in sorted(tuned_checkpoints.items()):
        checkpoint = Path(checkpoint)
        tuned_sha = sha256_file(checkpoint)
        tuned_model = load_tuned_model(checkpoint)
        tuned_embeddings, cache_hit = embed_unique_crops(
            crops,
            unique_indices,
            tuned_model,
            role=f"tuned_facenet_seed{seed}",
            weights_sha256=tuned_sha,
            source_sha256=source_sha256,
            cache_path=private_dir / f"private_embeddings_tuned_seed{seed}.npz",
            batch_size=batch_size,
            threads=threads,
            reuse_cache=reuse_cache,
        )
        tuned_sims = similarity_matrix(tuned_embeddings, protocol, row_map)
        tuned_ranks = per_query_ranks(tuned_sims, protocol.query_subject, protocol.gallery_subject)
        models_meta["tuned"][str(seed)] = {
            "checkpoint": checkpoint.name,
            "weights_sha256": tuned_sha,
            "cache": f"private/private_embeddings_tuned_seed{seed}.npz",
            "cache_hit": bool(cache_hit),
        }
        results[f"tuned_seed{seed}"] = {
            split: {
                "frozen": metrics_by_stratum(frozen_ranks, stratum_masks(protocol, split)),
                "tuned": metrics_by_stratum(tuned_ranks, stratum_masks(protocol, split)),
                "paired_bootstrap": {
                    label: item.as_dict()
                    for label, item in paired_subject_bootstrap(
                        frozen_ranks,
                        tuned_ranks,
                        protocol.query_subject,
                        stratum_masks(protocol, split),
                        n_boot=n_boot,
                        seed=protocol_seed,
                    ).items()
                },
            }
            for split in ("test", "dev")
        }
        results[f"tuned_seed{seed}"]["cmc_test_frozen"] = cmc_curve(frozen_ranks[protocol.split_mask("test")])
        results[f"tuned_seed{seed}"]["cmc_test_tuned"] = cmc_curve(tuned_ranks[protocol.split_mask("test")])
        for row, rank in enumerate(tuned_ranks.tolist()):
            base_rows[row][f"tuned_rank_seed{seed}"] = float(rank)

    write_jsonl(private_dir / "retrieval_rows.jsonl", base_rows)

    # Frozen-only reference result (seed independent), reported once.
    results["frozen_reference"] = {
        split: metrics_by_stratum(frozen_ranks, stratum_masks(protocol, split)) for split in ("test", "dev")
    }
    results["frozen_reference"]["cmc_test"] = cmc_curve(frozen_ranks[protocol.split_mask("test")])

    public = build_public_output(
        protocol=protocol,
        source=source,
        models=models_meta,
        results=results,
        primary_split="test",
        provenance=provenance,
    )
    assert_public_output_has_no_ids(public)
    public = sanitize_nonfinite(public)
    public_path = output_dir / "fgnet_retrieval_study.json"
    public_path.write_text(json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    inputs: list[Path | str] = [npz_path, base_weights, *[Path(p) for p in sorted(tuned_checkpoints.values())], plan_path, layout_path]
    for extra in (model_inventory_path, training_pairs):
        if extra is not None and Path(extra).is_file():
            inputs.append(Path(extra))
    inputs.extend(sorted(private_dir.glob("private_embeddings_*.npz")))
    write_experiment_manifest(
        output_dir / "fgnet_retrieval_study.manifest.json",
        experiment="fgnet-oldest-gallery-cross-age-retrieval",
        parameters={
            "protocol_name": protocol.name,            "protocol_hash": protocol.protocol_hash,
            "protocol_seed": protocol_seed,
            "dev_fraction": dev_fraction,
            "gallery_rule": "max_age__lowest_original_index",
            "query_rule": "age_strictly_less_than_selected_gallery_age",
            "tie_break": "stable_ranking_by_gallery_index",
            "preprocessing": PREPROCESSING,
            "n_boot": n_boot,
            "batch_size": batch_size,
            "cpu_threads": threads,
            "device": "cpu",
            "network_calls": False,
            "gallery_resampling": False,
            "training_identity_independence": "unverified",
            "provenance": provenance,
        },
        metrics={
            "runtime_seconds": round(time.perf_counter() - started_at, 3),
            "gallery_size": protocol.gallery_size,
            "n_queries": protocol.n_queries,
            "n_excluded_max_age_ties": protocol.n_excluded_max_age_ties,
            "unique_crops_embedded": int(unique_indices.size),
            "embedding_row_map_size": int(row_map.size),
        },
        inputs=inputs,
        outputs=[public_path],
    )
    return public


def _parse_seeds(raw: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        seed_text, _, path_text = token.partition("=")
        if not path_text:
            raise argparse.ArgumentTypeError("expected SEED=PATH entries")
        out[int(seed_text)] = Path(path_text)
    if not out:
        raise argparse.ArgumentTypeError("at least one SEED=PATH entry is required")
    return out


def default_tuned_checkpoints() -> dict[int, Path]:
    return {seed: PROJECT_ROOT / "models" / f"bb_facenet_seed{seed}.pt" for seed in DEFAULT_SEEDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=data_path("data_dir", "external", "fgnet_crops.npz"))
    parser.add_argument("--base-weights", type=Path, default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument(
        "--tuned",
        type=str,
        default=",".join(f"{seed}={path}" for seed, path in default_tuned_checkpoints().items()),
        help="comma-separated SEED=PATH entries for the tuned checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="public aggregate + manifest + private/ subdir (integrated default: metrics/fgnet_retrieval)",
    )
    parser.add_argument("--model-inventory", type=Path, default=DEFAULT_MODEL_INVENTORY)
    parser.add_argument("--training-pairs", type=Path, default=DEFAULT_TRAINING_PAIRS)
    parser.add_argument("--protocol-seed", type=int, default=DEFAULT_PROTOCOL_SEED)
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true", help="build the protocol only; no image inference")
    parser.add_argument("--no-reuse-cache", action="store_true")
    parser.add_argument("--expect-protocol-hash", type=str, default=None)
    args = parser.parse_args()
    result = run_study(
        npz_path=args.npz,
        base_weights=args.base_weights,
        output_dir=args.output_dir,
        tuned_checkpoints=_parse_seeds(args.tuned),
        protocol_seed=args.protocol_seed,
        dev_fraction=args.dev_fraction,
        n_boot=args.n_boot,
        batch_size=args.batch_size,
        threads=args.threads,
        plan_only=args.plan_only,
        reuse_cache=not args.no_reuse_cache,
        expect_protocol_hash=args.expect_protocol_hash,
        model_inventory_path=args.model_inventory,
        training_pairs=args.training_pairs,
    )
    print(json.dumps({k: v for k, v in result.items() if k not in {"results"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
