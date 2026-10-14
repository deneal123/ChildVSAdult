"""Fresh-family presentation schema, separate from legacy common renderers.

Pure dictionary validation only. This is neither original-byte acceptance nor
numerical replay; a caller must independently verify the native manifest and
bind a completed independent-reader receipt before publishing any table.
"""

import copy

from scripts.render_common_mechanism_v2 import (
    STRONG_CELL,
    checked_delta,
    checked_diagnostics,
    checked_frozen,
    checked_sep_model,
    checked_separability,
    count,
)

EXPERIMENT = "common-image-index-mechanism-v2-fresh-weak"
WEAK_KEYS = ("frozen", "tuned_s1", "tuned_s2", "tuned_s42")
PARAMETERS = dict(
    device="cpu",
    blas_threads=1,
    age_alpha=1,
    age_folds=5,
    age_fold_seed=42,
    bootstrap_seed=0,
    resamples=2000,
    protocol="exact strong and fresh weak bound image-index pairs",
    weak_parent_experiment="facenet-fixed8-validation-full-fgnet-cuda-v3",
    weak_key_mapping="literal tuned_sN; no legacy renaming",
)
LIMITATION = (
    "per-checkpoint descriptive diagnostics; fresh weak fixed8/head/b16 family, "
    "not legacy headline recovery; fixed OOF predictions, no probe refit within "
    "bootstrap or multiplicity correction; no training-seed population inference, "
    "controlled causal architecture comparison, human identity clearance or "
    "completed strong-matrix/mechanism claim"
)


def exact(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return set(a) == set(b) and all(exact(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def collect(native):
    """Validate exactly one new-schema strong cell and three literal weak seeds."""
    if native.get("experiment") != EXPERIMENT or not exact(native.get("parameters"), PARAMETERS):
        raise ValueError("fresh common schema/protocol required")
    metrics = native["metrics"]
    flags = dict(
        execution_complete=True,
        publication_ready=False,
        fresh_weak_lineage=True,
        legacy_weak_history_recovered=False,
        fixed_oof_bootstrap=True,
        neural_inference_replayed=False,
        mechanism_complete=False,
    )
    if not all(exact(metrics.get(k), v) for k, v in flags.items()):
        raise ValueError("fresh lineage and non-mechanism flags required")
    count(metrics.get("n_images"), 650)
    count(metrics.get("n_pairs"), 5308)
    names = metrics.get("strong_cells")
    if (
        type(names) is not list
        or len(names) != 1
        or type(names[0]) is not str
        or not STRONG_CELL.fullmatch(names[0])
    ):
        raise ValueError("one canonical strong checkpoint required")
    if not exact(metrics.get("weak_keys"), list(WEAK_KEYS)):
        raise ValueError("literal fresh weak key mapping required")
    bindings = metrics.get("weak_upstream_bindings")
    if (
        type(bindings) is not list
        or len(bindings) != 3
        or not all(type(v) is str and v.strip() for v in bindings)
        or len(set(bindings)) != 3
    ):
        raise ValueError("three unique fresh native bindings required")
    results = metrics["results"]
    if set(results) != {"strong", "weak"}:
        raise ValueError("exact two model families required")
    for family, keys in (("strong", ("frozen", names[0])), ("weak", WEAK_KEYS)):
        entry = results[family]
        sep = checked_separability(entry["separability"], keys)
        checked_frozen(sep["models"]["frozen"])
        if set(entry["diagnostics"]) != set(keys[1:]):
            raise ValueError("exact per-checkpoint diagnostics required")
        for key in keys[1:]:
            checked_sep_model(sep["models"][key])
            checked_delta(sep["models"][key], sep["models"]["frozen"])
            checked_diagnostics(entry["diagnostics"][key])
    return dict(
        results=copy.deepcopy(results),
        strong_cells=list(names),
        weak_keys=list(WEAK_KEYS),
        weak_upstream_bindings=list(bindings),
        n_images=650,
        n_pairs=5308,
        n_persons=82,
        observed_strong_cells=1,
        expected_strong_matrix_cells=36,
        mechanism_complete=False,
        publication_ready=False,
        legacy_weak_history_recovered=False,
        limitation=LIMITATION,
    )
