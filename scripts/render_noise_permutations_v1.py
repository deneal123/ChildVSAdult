"""Compact LaTeX projection of the fixed-permutation supervision-noise summary.

Proposal only.  Renders the aggregate ``summary.json`` records of exactly three
permutation arms (42, 1, 2) across exactly two strata (``overall``,
``large_gap_25plus``) into a supplement-ready ``table*`` tabular.

This module performs *no* inference, no score/embedding access and no native
acceptance.  It consumes only the public aggregate ``summary.json`` objects and
checks their declared scope before printing numbers.  All findings remain
conditional on one fixed artificial permutation per arm and must not be read as
an empirical identity-error rate, a training-seed population claim, an
ensemble-score AUC, or causal source superiority.

Definitions preserved from ``scripts/benchmark_metrics_v2.py``
-------------------------------------------------------------
``roc_auc``                  ROC-AUC.
``eer_interpolated``         Interpolated EER (smaller is better).
``eer_discrete_minimax``     Discrete minimax EER (smaller is better).
``tar@far=0.01``             TAR at tested operating threshold FAR = 1%.
``tar@far=0.001``            TAR at tested operating threshold FAR = 0.1%.

Intervals are nominal paired subject-bootstrap 95% intervals and are *not*
multiplicity-adjusted.  Thresholds are tested operating points, not deployment
calibration.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Closed vocabulary of the contract.
# ---------------------------------------------------------------------------

PERMUTATION_ARMS: tuple[int, ...] = (42, 1, 2)
TRAINING_SEEDS: tuple[int, ...] = (42, 1, 2)
STRATA: tuple[str, ...] = ("overall", "large_gap_25plus")

METRICS: tuple[str, ...] = (
    "roc_auc",
    "eer_interpolated",
    "eer_discrete_minimax",
    "tar@far=0.01",
    "tar@far=0.001",
)

METRIC_LABELS: dict[str, str] = {
    "roc_auc": "ROC-AUC",
    "eer_interpolated": "Interpolated EER",
    "eer_discrete_minimax": "Discrete minimax EER",
    "tar@far=0.01": r"TAR@FAR=1\%",
    "tar@far=0.001": r"TAR@FAR=0.1\%",
}

STRATUM_LABELS: dict[str, str] = {
    "overall": "Overall",
    "large_gap_25plus": r"Large gap ($\geq 25$y)",
}

# Exactly the metric stack the evaluator writes; an ``ensemble`` or
# ``population`` key would break set equality and must never be added.
METRIC_STACK_KEYS: frozenset[str] = frozenset({"low", "cross", "cross_minus_low", "frozen"})

# Labels that positively assert a construct this presentation is not allowed to
# claim.  They are rejected wherever they appear as a *key*, except for the one
# legitimate key whose declared value is the negative scope flag.
FORBIDDEN_KEY_TOKENS: tuple[str, ...] = ("ensemble", "population", "causal")
FORBIDDEN_KEY_ALLOWLIST: frozenset[str] = frozenset({"causal_source_evidence"})

# Disclaimers the accepted scope text must still contain.
REQUIRED_SCOPE_MARKERS: tuple[str, ...] = ("not deployment", "not multiplicity")
REQUIRED_CONDITIONING_MARKERS: tuple[str, ...] = (
    "not training-seed population",
    "score ensemble",
)

# Numeric self-consistency tolerance (float64 round-trip of an arithmetic mean).
CONSISTENCY_TOL = 1e-9


class SummaryContractError(ValueError):
    """A summary object violates the frozen presentation contract."""


# ---------------------------------------------------------------------------
# Exact-native-type helpers.  ``bool`` is a subclass of ``int``: ``type(x) is
# int`` therefore rejects ``True``/``False`` for integer fields, and
# ``type(x) is float`` rejects integers for float fields.
# ---------------------------------------------------------------------------

def _int(value, where: str) -> int:
    if type(value) is not int:
        raise SummaryContractError(f"{where}: exact int required, got {type(value).__name__}")
    return value


def _bool(value, where: str) -> bool:
    if type(value) is not bool:
        raise SummaryContractError(f"{where}: exact bool required, got {type(value).__name__}")
    return value


def _flt(value, where: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise SummaryContractError(f"{where}: finite native float required")
    return value


def _exact_int_list(value, expected: tuple[int, ...], where: str) -> list[int]:
    if type(value) is not list or len(value) != len(expected):
        raise SummaryContractError(f"{where}: exact list of {len(expected)} ints required")
    for item in value:
        _int(item, where)
    if tuple(value) != expected:
        raise SummaryContractError(f"{where}: expected {list(expected)}, got {value!r}")
    return value


def _ci95(pair, where: str) -> tuple[float, float]:
    if type(pair) is not list or len(pair) != 2:
        raise SummaryContractError(f"{where}: two-element numeric CI required")
    lo, hi = _flt(pair[0], f"{where}[0]"), _flt(pair[1], f"{where}[1]")
    if lo > hi:
        raise SummaryContractError(f"{where}: CI bounds must be ordered lo <= hi")
    return lo, hi


def _reject_forbidden_keys(node, where: str) -> None:
    if isinstance(node, dict):
        for key, child in node.items():
            lowered = str(key).lower()
            if lowered not in FORBIDDEN_KEY_ALLOWLIST:
                for token in FORBIDDEN_KEY_TOKENS:
                    if token in lowered:
                        raise SummaryContractError(f"{where}: forbidden label {key!r} ({token})")
            _reject_forbidden_keys(child, f"{where}.{key}")
    elif isinstance(node, list):
        for index, child in enumerate(node):
            _reject_forbidden_keys(child, f"{where}[{index}]")


def _require_text_markers(text, markers: tuple[str, ...], where: str) -> str:
    if type(text) is not str or not text.strip():
        raise SummaryContractError(f"{where}: nonempty scope text required")
    for marker in markers:
        if marker not in text:
            raise SummaryContractError(f"{where}: missing required disclaimer {marker!r}")
    return text


# ---------------------------------------------------------------------------
# Arm normalisation.
# ---------------------------------------------------------------------------

def arm_view(summary: dict, permutation: int, stratum: str, arm_scope: dict | None = None) -> dict:
    """Validate one (summary, permutation, stratum) cell and return its values.

    ``summary`` carries the source-level gates shared by all arms in the file;
    ``arm_scope`` carries the per-permutation stratum block (for arms 1/2 it is
    nested under ``permutation_seed1``/``permutation_seed2``).
    """
    where = f"perm{permutation}/{stratum}"
    if type(summary) is not dict or type(arm_scope) is not dict:
        raise SummaryContractError(f"{where}: summary dict required")
    _reject_forbidden_keys(summary, where)

    _bool(summary.get("execution_complete"), f"{where}.execution_complete")
    if summary["execution_complete"] is not True:
        raise SummaryContractError(f"{where}: execution_complete must be True")
    _bool(summary.get("publication_ready"), f"{where}.publication_ready")
    if summary["publication_ready"] is not False:
        raise SummaryContractError(f"{where}: publication_ready must remain False")
    if summary.get("low_role") != "clean CROSS":
        raise SummaryContractError(f"{where}: low_role must be 'clean CROSS'")
    if summary.get("cross_role") != "noisy CROSS":
        raise SummaryContractError(f"{where}: cross_role must be 'noisy CROSS'")
    if summary.get("direction") != "noise_minus_clean":
        raise SummaryContractError(f"{where}: direction must be 'noise_minus_clean'")

    if permutation == 42:
        _int(summary.get("permutation_seed"), f"{where}.permutation_seed")
        if summary["permutation_seed"] != 42:
            raise SummaryContractError(f"{where}: arm 42 requires permutation_seed 42")
    else:
        _exact_int_list(summary.get("permutation_seeds"), (1, 2), f"{where}.permutation_seeds")
        _exact_int_list(summary.get("training_seeds"), TRAINING_SEEDS, f"{where}.training_seeds")

    cell = arm_scope.get(stratum)
    if type(cell) is not dict:
        raise SummaryContractError(f"{where}: declared stratum {stratum!r} missing")

    _bool(cell.get("publication_ready"), f"{where}.publication_ready")
    if cell["publication_ready"] is not False:
        raise SummaryContractError(f"{where}: stratum publication_ready must remain False")
    if cell.get("training_identity_independence") != "unverified":
        raise SummaryContractError(f"{where}: training_identity_independence must be 'unverified'")
    _bool(cell.get("causal_source_evidence"), f"{where}.causal_source_evidence")
    if cell["causal_source_evidence"] is not False:
        raise SummaryContractError(f"{where}: causal_source_evidence must remain False")
    if cell.get("metric_version") != "empirical-roc-v2":
        raise SummaryContractError(f"{where}: metric_version must be 'empirical-roc-v2'")
    _exact_int_list(cell.get("training_seeds"), TRAINING_SEEDS, f"{where}.training_seeds")
    _require_text_markers(cell.get("scope"), REQUIRED_SCOPE_MARKERS, f"{where}.scope")
    for field in ("n_pairs", "n_subjects", "loo_valid", "loo_unavailable"):
        count = _int(cell.get(field), f"{where}.{field}")
        if count < (1 if field in ("n_pairs", "n_subjects") else 0):
            raise SummaryContractError(f"{where}: nonnegative counts and nonempty sample required")
    if cell["loo_valid"] + cell["loo_unavailable"] != cell["n_subjects"]:
        raise SummaryContractError(f"{where}: LOO coverage must partition subjects")

    boot = cell.get("bootstrap")
    if type(boot) is not dict:
        raise SummaryContractError(f"{where}.bootstrap: dict required")
    _bool(boot.get("shared_draws_across_all_models"), f"{where}.bootstrap.shared")
    if boot["shared_draws_across_all_models"] is not True:
        raise SummaryContractError(f"{where}: shared draws across all checkpoints required")
    _bool(boot.get("thresholds_reselected"), f"{where}.bootstrap.thresholds_reselected")
    if boot["thresholds_reselected"] is not True:
        raise SummaryContractError(f"{where}: thresholds must be reselected per resample")
    _int(boot.get("requested"), f"{where}.bootstrap.requested")
    _int(boot.get("valid"), f"{where}.bootstrap.valid")
    _int(boot.get("seed"), f"{where}.bootstrap.seed")
    if boot["requested"] != 2000 or boot["seed"] != 0 or boot["valid"] != boot["requested"]:
        raise SummaryContractError(f"{where}: all requested bootstrap draws must be valid")
    if (boot.get("positive_weight") != "owner multiplicity once"
            or boot.get("negative_weight") != "endpoint multiplicity product"):
        raise SummaryContractError(f"{where}: declared subject endpoint weights required")
    _require_text_markers(
        boot.get("conditioning"), REQUIRED_CONDITIONING_MARKERS, f"{where}.bootstrap.conditioning"
    )

    metrics = cell.get("metrics")
    if type(metrics) is not dict or set(metrics) != set(METRICS):
        raise SummaryContractError(f"{where}: exact ROC-v2 metric set required")

    out: dict = {"permutation": permutation, "metrics": {}}
    for metric in METRICS:
        mwhere = f"{where}.metrics.{metric}"
        stack = metrics[metric]
        if type(stack) is not dict or set(stack) != METRIC_STACK_KEYS:
            raise SummaryContractError(f"{mwhere}: exact clean/noisy/frozen metric stack required")
        low, cross, delta, frozen = stack["low"], stack["cross"], stack["cross_minus_low"], stack["frozen"]

        clean_mean = _flt(low.get("mean"), f"{mwhere}.low.mean")
        clean_sd = _flt(low.get("sd"), f"{mwhere}.low.sd")
        if clean_sd < 0:
            raise SummaryContractError(f"{mwhere}.low.sd: SD must be nonnegative")
        noisy_mean = _flt(cross.get("mean"), f"{mwhere}.cross.mean")
        noisy_sd = _flt(cross.get("sd"), f"{mwhere}.cross.sd")
        if noisy_sd < 0:
            raise SummaryContractError(f"{mwhere}.cross.sd: SD must be nonnegative")
        delta_mean = _flt(delta.get("mean"), f"{mwhere}.cross_minus_low.mean")
        delta_lo, delta_hi = _ci95(
            delta.get("mean_checkpoint_ci95"), f"{mwhere}.cross_minus_low.mean_checkpoint_ci95"
        )
        frozen_point = _flt(frozen.get("point"), f"{mwhere}.frozen.point")
        _ci95(frozen.get("ci95"), f"{mwhere}.frozen.ci95")

        per_seed = low.get("per_seed")
        if type(per_seed) is not dict or set(per_seed) != {str(s) for s in TRAINING_SEEDS}:
            raise SummaryContractError(f"{mwhere}.low.per_seed: exact training-seed keys required")
        for seed in TRAINING_SEEDS:
            _flt(per_seed[str(seed)].get("point"), f"{mwhere}.low.per_seed.{seed}.point")

        if abs(delta_mean - (noisy_mean - clean_mean)) > CONSISTENCY_TOL:
            raise SummaryContractError(
                f"{mwhere}: cross_minus_low.mean disagrees with cross.mean - low.mean"
            )

        out["metrics"][metric] = {
            "frozen": frozen_point,
            "clean_mean": clean_mean,
            "clean_sd": clean_sd,
            "noisy_mean": noisy_mean,
            "noisy_sd": noisy_sd,
            "delta_mean": delta_mean,
            "delta_ci95": (delta_lo, delta_hi),
        }
    return out


def normalize(summary42: dict, summary12: dict, stratum: str) -> dict[int, dict]:
    """Validate the three arms of one stratum and return them keyed by arm."""
    if stratum not in STRATA:
        raise SummaryContractError(f"declared stratum required, got {stratum!r}")
    if type(summary12) is not dict or "permutation_seed1" not in summary12 or "permutation_seed2" not in summary12:
        raise SummaryContractError("perm1/perm2 scope dicts required")
    if {key for key in summary12 if key.startswith("permutation_seed") and key != "permutation_seeds"} != {"permutation_seed1", "permutation_seed2"}:
        raise SummaryContractError("exact perm1/perm2 scope dicts required")
    arms = {
        42: arm_view(summary42, 42, stratum, summary42),
        1: arm_view(summary12, 1, stratum, summary12["permutation_seed1"]),
        2: arm_view(summary12, 2, stratum, summary12["permutation_seed2"]),
    }

    # The clean CROSS draw and the frozen backbone are shared protocol inputs;
    # every arm's declared clean/frozen values must agree, and each arm's own
    # noisy-minus-clean delta must be internally consistent.
    reference = arms[42]["metrics"]
    for arm, view in arms.items():
        for metric in METRICS:
            got, want = view["metrics"][metric], reference[metric]
            for field in ("frozen", "clean_mean", "clean_sd"):
                if abs(got[field] - want[field]) > CONSISTENCY_TOL:
                    raise SummaryContractError(
                        f"perm{arm}/{stratum}/{metric}: shared {field} inconsistent with arm 42"
                    )
    coverage = []
    for source in (summary42, summary12):
        cov = source.get("coverage")
        if type(cov) is not dict:
            raise SummaryContractError("coverage: shared coverage block required")
        fields = {"source_positives", "retained_positives", "source_25plus_positives", "retained_25plus_positives"}
        if set(cov) != fields or any(_int(cov[key], f"coverage.{key}") < 0 for key in fields):
            raise SummaryContractError("coverage: exact nonnegative integer counts required")
        coverage.append(cov)
    if coverage[0] != coverage[1]:
        raise SummaryContractError("coverage: shared clean/frozen coverage inconsistent across sources")
    return arms


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------

def _num(value: float) -> str:
    return f"{value:.4f}"


def _signed(value: float) -> str:
    return f"{value:+.4f}"


def table(summary42: dict, summary12: dict, stratum: str) -> str:
    """Return a compact supplement tabular: 5 metrics x 3 permutation arms."""
    arms = normalize(summary42, summary12, stratum)
    rows = []
    for arm in PERMUTATION_ARMS:
        for metric in METRICS:
            cell = arms[arm]["metrics"][metric]
            lo, hi = cell["delta_ci95"]
            rows.append(
                " & ".join(
                    (
                        str(arm),
                        METRIC_LABELS[metric],
                        _num(cell["frozen"]),
                        f"{_num(cell['clean_mean'])}$\\pm${_num(cell['clean_sd'])}",
                        f"{_num(cell['noisy_mean'])}$\\pm${_num(cell['noisy_sd'])}",
                        _signed(cell["delta_mean"]),
                        f"$[{_signed(lo)},{_signed(hi)}]$",
                    )
                )
                + r" \\"
            )
    return "\n".join(
        [
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\footnotesize",
            r"\begin{tabular}{@{}llrrrrr@{}}",
            r"\toprule",
            r"Perm. & Metric & Frozen & Clean mean$\pm$SD & Noisy mean$\pm$SD"
            r" & Noise$-$clean $\Delta$ & Paired mean-$\Delta$ 95\% CI \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def table_star(
    summary42: dict,
    summary12: dict,
    stratum: str,
    *,
    caption: str,
    label: str = "tab:noise-permutation",
) -> str:
    """Wrap :func:`table` in a compact ``table*`` float for the supplement."""
    if not caption.strip():
        raise SummaryContractError("caption required")
    body = table(summary42, summary12, stratum)
    note = (
        r"\par\vspace{2pt}\footnotesize\textit{Notes.} Mean$\pm$SD across training "
        r"seeds 42/1/2; $\Delta$ is noise$-$clean; the paired interval resamples "
        r"subjects over fixed checkpoints. Smaller EER is better. Nominal 95\% "
        r"subject intervals, not multiplicity-adjusted. TAR is at tested operating "
        r"thresholds, not deployment calibration. One fixed artificial permutation "
        r"per arm; not an empirical identity-error rate, training-seed population, "
        r"ensemble-score AUC, or causal-source comparison."
    )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            body,
            note,
            r"\end{table*}",
        ]
    )


def row_count(summary42: dict, summary12: dict, stratum: str) -> int:
    """Number of printed data rows (must be 15 for either stratum)."""
    text = table(summary42, summary12, stratum)
    # Subtract the one ``\\``-terminated header row.
    return sum(1 for line in text.splitlines() if r"\\" in line) - 1

