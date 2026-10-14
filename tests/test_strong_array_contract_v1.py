"""Generated strict-dtype tests for typed_array_validator_e473.

Synthetic arrays only: no native data, corpus, checkpoints or model calls.
Run with pytest basetemp/cache kept inside the assigned artifact dir, e.g.

    .venv/Scripts/python.exe -m pytest \
        .work/pi-workers/20261009-third-14b6fd1b/strong-statistics-typed-v3-e473 \
        -p no:cacheprovider -q

Each test builds a tiny *canonically valid* cell by construction, then mutates
exactly one dtype/shape/domain/identity property and asserts rejection.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts import strong_array_contract_v1 as V

CELL = "random_full_lr1e-06_s42"
FROZEN = "frozen"
N, P, D = V.N_IMAGES, V.N_PAIRS, V.EMBEDDING_DIM


def fixture():
    """A small self-consistent 650-subject / 5308-pair synthetic cell.

    Subjects/ages are int32 (int-kind but not the producer's int64) to prove the
    contract is kind-level; embeddings/scores are float32.
    """
    rng = np.random.default_rng(0)
    subjects = np.repeat(np.arange(V.N_UNIQUE_SUBJECTS, dtype=np.int32), 8)[:N]
    subjects = subjects.astype(np.int32)
    ages = (rng.integers(0, 60, size=N)).astype(np.int32)

    # Build every available same-subject positive pair, then cross-subject
    # negatives, up to exactly P=5308 total.  The validator does not fix the
    # positive/negative split (the real cell happens to have 2654/2654); it only
    # requires at least one of each label and exact subject/gap identities.
    groups = {s: np.where(subjects == s)[0] for s in np.unique(subjects)}
    order = list(np.unique(subjects))
    positives = [
        (int(a), int(b)) for s in order for i, a in enumerate(groups[s]) for b in groups[s][i + 1 :]
    ]
    assert len(positives) >= 1
    left = np.empty(P, np.int32)
    right = np.empty(P, np.int32)
    count = min(len(positives), P - 1)  # keep room for >= 1 negative
    for i, (a, b) in enumerate(positives[:count]):
        left[i], right[i] = a, b
    n_idx = count
    g = 0
    while n_idx < P:
        s1 = order[g % len(order)]
        s2 = order[(g * 7 + 3) % len(order)]
        if s1 == s2:
            g += 1
            continue
        left[n_idx] = groups[s1][g % len(groups[s1])]
        right[n_idx] = groups[s2][(g * 3) % len(groups[s2])]
        n_idx += 1
        g += 1

    labels = (subjects[left] == subjects[right]).astype(np.int32)
    # Guarantee at least one of each label for domain checks.
    assert labels.min() == 0 and labels.max() == 1
    source_gap = np.abs(ages[left] - ages[right]).astype(np.int32)
    pair = dict(
        left=left,
        right=right,
        labels=labels,
        subject_a=subjects[left].copy(),
        subject_b=subjects[right].copy(),
        source_gap=source_gap,
    )
    metadata = dict(subjects=subjects, ages=ages)

    embedding = {}
    for model in (FROZEN, CELL):
        vec = rng.standard_normal((N, D)).astype(np.float32)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        embedding[model] = dict(
            indices=np.arange(N, dtype=np.int64), embeddings=vec.astype(np.float32)
        )
    scores = {}
    for model in (FROZEN, CELL):
        v = embedding[model]["embeddings"]
        scores[model] = np.sum(v[left] * v[right], axis=1).astype(np.float32)
    return metadata, pair, embedding, scores


def clone(obj):
    if isinstance(obj, dict):
        return {k: clone(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.copy()
    return obj


def run(metadata, pair, embedding, scores, cell=CELL):
    return V.validate_arrays(metadata, pair, embedding, scores, cell)


# ---------------------------------------------------------------- happy path
def test_valid_cell_accepted():
    result = run(*fixture())
    assert result["cell"] == CELL
    assert result["coerced"] is False
    assert result["pairs"] == P


def test_dtype_contract_is_kind_level_not_exact_width():
    """int32 metadata is accepted: the contract is integer-kind, not int64."""
    metadata, pair, embedding, scores = fixture()
    assert metadata["subjects"].dtype == np.int32
    assert run(metadata, pair, embedding, scores)["unique_subjects"] == V.N_UNIQUE_SUBJECTS


# ------------------------------------------------- strict dtype enforcement
@pytest.mark.parametrize("field", ["subjects", "ages"])
def test_float_metadata_rejected(field):
    metadata, pair, embedding, scores = fixture()
    metadata[field] = metadata[field].astype(np.float64)  # integral values, wrong kind
    with pytest.raises(V.TypedArrayError, match="not-integer"):
        run(metadata, pair, embedding, scores)


@pytest.mark.parametrize(
    "field", ["left", "right", "labels", "subject_a", "subject_b", "source_gap"]
)
def test_float_pair_field_rejected(field):
    metadata, pair, embedding, scores = fixture()
    pair[field] = pair[field].astype(np.float64)
    with pytest.raises(V.TypedArrayError, match="not-integer"):
        run(metadata, pair, embedding, scores)


def test_float_indices_rejected():
    metadata, pair, embedding, scores = fixture()
    embedding[FROZEN]["indices"] = embedding[FROZEN]["indices"].astype(np.float64)
    with pytest.raises(V.TypedArrayError, match="indices-not-integer"):
        run(metadata, pair, embedding, scores)


def test_integer_embeddings_rejected():
    metadata, pair, embedding, scores = fixture()
    embedding[FROZEN]["embeddings"] = embedding[FROZEN]["embeddings"].astype(np.int32)
    with pytest.raises(V.TypedArrayError, match="embeddings-not-floating"):
        run(metadata, pair, embedding, scores)


@pytest.mark.parametrize("model", [FROZEN, CELL])
def test_integer_scores_rejected(model):
    metadata, pair, embedding, scores = fixture()
    scores[model] = scores[model].astype(np.int32)
    with pytest.raises(V.TypedArrayError, match="scores-not-floating"):
        run(metadata, pair, embedding, scores)


def test_float16_is_still_floating_kind():
    """A narrower float is still floating-kind; only *kind* is contracted."""
    metadata, pair, embedding, scores = fixture()
    embedding[CELL]["embeddings"] = embedding[CELL]["embeddings"].astype(np.float16)
    v = embedding[CELL]["embeddings"]
    scores[CELL] = np.sum(v[pair["left"]] * v[pair["right"]], axis=1).astype(np.float16)
    run(metadata, pair, embedding, scores)


# ------------------------------------------------------- shape enforcement
def test_wrong_metadata_length_rejected():
    metadata, pair, embedding, scores = fixture()
    metadata["ages"] = metadata["ages"][:-1]
    with pytest.raises(V.TypedArrayError, match="ages-shape"):
        run(metadata, pair, embedding, scores)


def test_wrong_pair_length_rejected():
    metadata, pair, embedding, scores = fixture()
    pair["left"] = pair["left"][:-1]
    with pytest.raises(V.TypedArrayError, match="left-shape"):
        run(metadata, pair, embedding, scores)


def test_wrong_embedding_shape_rejected():
    metadata, pair, embedding, scores = fixture()
    embedding[FROZEN]["embeddings"] = embedding[FROZEN]["embeddings"][:, :256]
    with pytest.raises(V.TypedArrayError, match="embeddings-shape"):
        run(metadata, pair, embedding, scores)


def test_indices_must_be_exact_arange():
    metadata, pair, embedding, scores = fixture()
    embedding[FROZEN]["indices"] = embedding[FROZEN]["indices"][::-1].copy()
    with pytest.raises(V.TypedArrayError, match="indices-not-arange"):
        run(metadata, pair, embedding, scores)


def test_nonfinite_embedding_rejected():
    metadata, pair, embedding, scores = fixture()
    embedding[FROZEN]["embeddings"][0, 0] = np.nan
    with pytest.raises(V.TypedArrayError, match="embeddings-nonfinite"):
        run(metadata, pair, embedding, scores)


def test_nonunit_embedding_rejected():
    metadata, pair, embedding, scores = fixture()
    embedding[FROZEN]["embeddings"][0] *= 2.0
    with pytest.raises(V.TypedArrayError, match="embeddings-not-unit"):
        run(metadata, pair, embedding, scores)


def test_nonfinite_score_rejected():
    metadata, pair, embedding, scores = fixture()
    scores[CELL][0] = np.inf
    with pytest.raises(V.TypedArrayError, match="scores-nonfinite"):
        run(metadata, pair, embedding, scores)


# ------------------------------------------------ identity / algebra checks
def test_subject_identity_must_match_pool():
    metadata, pair, embedding, scores = fixture()
    pair["subject_a"][0] = pair["subject_a"][0] + 1
    with pytest.raises(V.TypedArrayError, match="subject_a-mismatch"):
        run(metadata, pair, embedding, scores)


def test_label_definition_must_hold():
    metadata, pair, embedding, scores = fixture()
    pair["labels"][0] = 1 - pair["labels"][0]
    with pytest.raises(V.TypedArrayError, match="label-definition-mismatch"):
        run(metadata, pair, embedding, scores)


def test_source_gap_must_be_chronological():
    metadata, pair, embedding, scores = fixture()
    pair["source_gap"][0] = pair["source_gap"][0] + 1
    with pytest.raises(V.TypedArrayError, match="source_gap-mismatch"):
        run(metadata, pair, embedding, scores)


def test_unique_subject_count_enforced():
    metadata, pair, embedding, scores = fixture()
    metadata["subjects"][:] = 0  # 1 unique subject, not 82
    with pytest.raises(V.TypedArrayError, match="subjects-unique-82"):
        run(metadata, pair, embedding, scores)


def test_cosine_score_exactness():
    metadata, pair, embedding, scores = fixture()
    scores[FROZEN] = scores[FROZEN].copy()
    scores[FROZEN][5] = np.float32(scores[FROZEN][5] + np.float32(1e-6))
    with pytest.raises(V.TypedArrayError, match="cosine-mismatch"):
        run(metadata, pair, embedding, scores)


# ---------------------------------------------------- canonical cell regex
@pytest.mark.parametrize(
    "bad",
    [
        "random_full_lr1e-06_s42 ",  # trailing space
        "random_full_lr1e-6_s42",  # non-canonical lr spelling
        "RANDOM_full_lr1e-06_s42",  # case
        "random_full_lr1e-06_s3",  # undeclared seed
        "random_full_lr1e-06_s42\n",  # trailing newline
        "frozen",  # the reserved frozen key
        "",  # empty
        "random_full_lr1e-06_s42;drop",  # injection tail
    ],
)
def test_bad_cell_labels_rejected(bad):
    metadata, pair, embedding, scores = fixture()
    with pytest.raises(V.TypedArrayError, match="cell-not-canonical|cell-not-string"):
        run(metadata, pair, embedding, scores, cell=bad)


@pytest.mark.parametrize(
    "bad_type", [None, 42, b"random_full_lr1e-06_s42", ["random_full_lr1e-06_s42"]]
)
def test_non_string_cell_rejected(bad_type):
    metadata, pair, embedding, scores = fixture()
    with pytest.raises(V.TypedArrayError, match="cell-not-string"):
        run(metadata, pair, embedding, scores, cell=bad_type)


@pytest.mark.parametrize(
    "cell",
    [
        "random_full_lr1e-06_s42",
        "lookalike_head_lr1e-05_s1",
        "random_tail_lr1e-05_s2",
        "lookalike_full_lr1e-06_s42",
    ],
)
def test_canonical_cells_accepted(cell):
    metadata, pair, embedding, scores = fixture()
    scores = {FROZEN: scores[FROZEN], cell: scores[CELL]}
    embedding = {FROZEN: embedding[FROZEN], cell: embedding[CELL]}
    V.validate_arrays(metadata, pair, embedding, scores, cell)


def test_extra_score_key_rejected():
    metadata, pair, embedding, scores = fixture()
    scores["extra"] = scores[FROZEN]
    with pytest.raises(V.TypedArrayError, match="score-fields"):
        run(metadata, pair, embedding, scores)


def test_missing_cell_score_key_rejected():
    metadata, pair, embedding, scores = fixture()
    del scores[CELL]
    with pytest.raises(V.TypedArrayError, match="score-fields"):
        run(metadata, pair, embedding, scores)


def test_wrapped_uint8_indices_rejected():
    metadata, pair, embedding, scores = fixture()
    embedding[FROZEN]["indices"] = np.arange(N).astype(np.uint8)
    with pytest.raises(V.TypedArrayError, match="indices-not-arange"):
        run(metadata, pair, embedding, scores)


def test_unsigned_age_gap_uses_signed_arithmetic():
    metadata, pair, embedding, scores = fixture()
    metadata["ages"] = metadata["ages"].astype(np.uint8)
    run(metadata, pair, embedding, scores)


def test_unsigned_age_overflow_rejected():
    metadata, pair, embedding, scores = fixture()
    metadata["ages"] = metadata["ages"].astype(np.uint64)
    metadata["ages"][0] = np.iinfo(np.uint64).max
    with pytest.raises(V.TypedArrayError, match="int64-domain"):
        run(metadata, pair, embedding, scores)
