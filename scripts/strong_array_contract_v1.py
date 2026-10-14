"""Pure supplemental strict-array validator for the strong explicit-statistics cell.

This module is deliberately NOT a CLI and NOT a native reader.  It performs no
file I/O, no ``np.load``, no byte hashing and no neural/statistical execution.
It accepts already-decoded NumPy arrays and enforces the *strict dtype, shape,
domain and algebraic* contract that ``check_strong_explicit_statistics_e419_v2.py``
only checked by shape/value with dtype-insensitive ``np.array_equal`` /
``np.allclose``.

Why this cannot accept native output on its own
-----------------------------------------------
A validator that only sees in-memory arrays cannot prove that those arrays came
from ORIGINAL, hash-verified bytes.  ``np.load`` is happy to hand back arrays
reconstructed from a tampered file, and ``np.array_equal`` ignores dtypes.
Acceptance of the real native artifacts therefore requires an outer wrapper
that:

1. decodes the ORIGINAL bytes itself (read the manifest-bound file bytes and
   only then ``np.load(io.BytesIO(raw), allow_pickle=False)``), never trusting a
   cached/decoded copy;
2. source-pins this helper (and every other helper) by exact-byte ``sha256`` +
   length before and after use;
3. executes two stable byte phases: ``phase-1`` re-resolves every alias and
   re-hashes every target BEFORE any semantic check, and ``phase-2`` repeats the
   re-guard/re-resolve/re-hash AFTER the semantic checks, so a mutation in the
   window between the phases is caught.

Strictness rules
----------------
* No dtype coercion to rescue wrong inputs.  A float metadata array whose values
  are integral is rejected: the contract is integer-*kind*, not integral-value.
* Integer fields use ``dtype.kind in "iu"``; floating fields use
  ``dtype.kind == "f"``.  This matches the sibling house checkers
  (``check_common_*``) and the canonical producers, which emit int64 for
  metadata/pairs/indices and float32 for embeddings/scores.
* Exact float cosines are compared with ``np.array_equal`` on the recomputed sum
  in the embedding dtype (no tolerance), after kind and finiteness are checked.
"""

from __future__ import annotations

import re

import numpy as np

# --------------------------------------------------------------------------
# Canonical schema constants.
#   Locators:
#     scripts/verify_strong_native_cell_v1.py  (NEGATIVES/SCOPES/LEARNING_RATES/
#                                               SEEDS/STRATA/cell_name)
#     src/age_gap/evaluation/fgnet.py          (subjects/ages -> int64)
#     scripts/evaluate_strong_cuda_fgnet_v1.py (indices/embeddings/scores)
#     scripts/reevaluate_fgnet_metrics_v2.py   (protocol -> left/right/labels/gap)
# --------------------------------------------------------------------------
NEGATIVES = ("lookalike", "random")
SCOPES = ("full", "head", "tail")
LEARNING_RATES = (1e-6, 1e-5)
SEEDS = (1, 2, 42)
FROZEN = "frozen"
N_IMAGES = 650
N_UNIQUE_SUBJECTS = 82
N_PAIRS = 5308
EMBEDDING_DIM = 512
STRATA = (("overall", N_PAIRS, 82), ("large_gap_25plus", 430, 75))

# Safe canonical cell label, identical to check_common_fresh_e461.canonical_cell.
CELL_PATTERN = re.compile(r"(random|lookalike)_(head|tail|full)_lr(1e-06|1e-05)_s(42|1|2)")


class TypedArrayError(ValueError):
    """A strict array-contract violation (dtype, shape, domain or algebra)."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise TypedArrayError(message)


def is_integer(array) -> bool:
    return isinstance(array, np.ndarray) and array.dtype.kind in "iu"


def is_floating(array) -> bool:
    return isinstance(array, np.ndarray) and array.dtype.kind == "f"


def canonical_cell(value) -> str:
    """Return the canonical cell label or raise; never coerce a non-string."""
    _need(type(value) is str, "cell-not-string")
    _need(CELL_PATTERN.fullmatch(value) is not None, "cell-not-canonical")
    return value


def check_metadata(subjects, ages) -> None:
    """metadata ``subjects``/``ages``: integer, length 650, 82 unique subjects."""
    _need(is_integer(subjects), "subjects-not-integer")
    _need(is_integer(ages), "ages-not-integer")
    _need(subjects.shape == (N_IMAGES,), "subjects-shape")
    _need(ages.shape == (N_IMAGES,), "ages-shape")
    _need((subjects >= 0).all(), "subjects-negative")
    _need((ages >= 0).all(), "ages-negative")
    _need(int(ages.max()) <= np.iinfo(np.int64).max, "ages-outside-int64-domain")
    _need(int(np.unique(subjects).size) == N_UNIQUE_SUBJECTS, "subjects-unique-82")


def check_pairs(pair: dict, subjects, ages) -> None:
    """pair integer arrays length 5308 with exact chronological labels/gaps."""
    fields = ("left", "right", "labels", "subject_a", "subject_b", "source_gap")
    _need(type(pair) is dict and set(pair) == set(fields), "pair-fields")
    for name in fields:
        array = pair[name]
        _need(is_integer(array), f"{name}-not-integer")
        _need(array.shape == (N_PAIRS,), f"{name}-shape")
    left, right = pair["left"], pair["right"]
    _need((left >= 0).all() and (left < N_IMAGES).all(), "left-range")
    _need((right >= 0).all() and (right < N_IMAGES).all(), "right-range")
    _need(set(np.unique(pair["labels"]).tolist()) == {0, 1}, "labels-domain")
    # Exact chronological identity: subjects are the pool rows the pair points at.
    _need(np.array_equal(pair["subject_a"], subjects[left]), "subject_a-mismatch")
    _need(np.array_equal(pair["subject_b"], subjects[right]), "subject_b-mismatch")
    _need(
        np.array_equal(pair["labels"], (subjects[left] == subjects[right])),
        "label-definition-mismatch",
    )
    expected_gap = np.abs(ages[left].astype(np.int64) - ages[right].astype(np.int64))
    _need(np.array_equal(pair["source_gap"], expected_gap), "source_gap-mismatch")


def check_embedding(indices, embeddings) -> None:
    """embedding ``indices`` integer arange(650); ``embeddings`` finite unit 650x512."""
    _need(is_integer(indices), "indices-not-integer")
    _need(indices.shape == (N_IMAGES,), "indices-shape")
    _need(np.array_equal(indices, np.arange(N_IMAGES)), "indices-not-arange")
    _need(is_floating(embeddings), "embeddings-not-floating")
    _need(embeddings.shape == (N_IMAGES, EMBEDDING_DIM), "embeddings-shape")
    _need(bool(np.isfinite(embeddings).all()), "embeddings-nonfinite")
    norms = np.linalg.norm(embeddings, axis=1)
    _need(bool(np.allclose(norms, 1, atol=1e-5)), "embeddings-not-unit")


def check_scores(scores: dict, cell: str, pair: dict, embeddings: dict) -> None:
    """score keys ``frozen`` + literal canonical ``cell``; floating, finite, exact cosines."""
    canonical_cell(cell)
    _need(set(scores) == {FROZEN, cell}, "score-fields")
    left, right = pair["left"], pair["right"]
    for model in (FROZEN, cell):
        values = scores[model]
        _need(is_floating(values), f"{model}-scores-not-floating")
        _need(values.shape == (N_PAIRS,), f"{model}-scores-shape")
        _need(bool(np.isfinite(values).all()), f"{model}-scores-nonfinite")
        vectors = embeddings[model]
        expected = np.sum(vectors[left] * vectors[right], axis=1)
        _need(np.array_equal(values, expected), f"{model}-cosine-mismatch")


def validate_arrays(metadata, pair, embedding, scores, cell: str) -> dict:
    """Full strict validation of one already-decoded strong cell.

    ``metadata``  -> {"subjects": int[650], "ages": int[650]}
    ``pair``      -> {"left","right","labels","subject_a","subject_b","source_gap"}
    ``embedding`` -> {"frozen": {"indices","embeddings"}, cell: {...}}
    ``scores``    -> {"frozen": float[5308], cell: float[5308]}
    """
    canonical_cell(cell)
    _need(type(metadata) is dict and set(metadata) == {"subjects", "ages"}, "metadata-fields")
    check_metadata(metadata["subjects"], metadata["ages"])
    check_pairs(pair, metadata["subjects"], metadata["ages"])
    _need(set(embedding) == {FROZEN, cell}, "embedding-fields")
    vectors = {}
    for model in (FROZEN, cell):
        rows = embedding[model]
        _need(set(rows) == {"indices", "embeddings"}, f"{model}-embedding-fields")
        check_embedding(rows["indices"], rows["embeddings"])
        vectors[model] = rows["embeddings"]
    check_scores(scores, cell, pair, vectors)
    return {
        "cell": cell,
        "n_images": N_IMAGES,
        "unique_subjects": N_UNIQUE_SUBJECTS,
        "pairs": N_PAIRS,
        "embedding_dim": EMBEDDING_DIM,
        "dtype_contract": "int-kind metadata/pairs/indices; float-kind embeddings/scores",
        "coerced": False,
    }
