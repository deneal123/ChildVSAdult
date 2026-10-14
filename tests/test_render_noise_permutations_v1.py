"""Focused synthetic tests for the noise-permutation table renderer.

All fixtures are built in-memory; no native summary, score, embedding, image or
workbook file is read.  These tests assert the presentation contract, not any
scientific result.
"""
from __future__ import annotations

import copy
import math

import pytest

from scripts import render_noise_permutations_v1 as r

COVERAGE = {
    "source_positives": 2659,
    "retained_positives": 2654,
    "source_25plus_positives": 215,
    "retained_25plus_positives": 215,
}
SCOPE = (
    "raw noisy CROSS minus clean CROSS; smaller EER is better; test-derived ROC "
    "not deployment calibration; intervals not multiplicity corrected"
)
CONDITIONING = (
    "fixed checkpoints and protocol; not training-seed population or score ensemble"
)
FROZEN = {
    "roc_auc": 0.8870185140911416,
    "eer_interpolated": 0.1900,
    "eer_discrete_minimax": 0.2000,
    "tar@far=0.01": 0.4200,
    "tar@far=0.001": 0.3900,
}
CLEAN = {
    "roc_auc": 0.8865425106104032,
    "eer_interpolated": 0.1920371765887968,
    "eer_discrete_minimax": 0.2050,
    "tar@far=0.01": 0.4100,
    "tar@far=0.001": 0.3800,
}


def _metric(metric: str, noisy_mean: float, delta_lo: float, delta_hi: float) -> dict:
    clean = CLEAN[metric]
    return {
        "low": {
            "mean": clean,
            "sd": 0.00031177534654040746,
            "mean_checkpoint_ci95": [clean - 0.02, clean + 0.02],
            "per_seed": {
                "42": {"point": clean - 0.0001, "ci95": [clean - 0.02, clean + 0.02]},
                "1": {"point": clean + 0.0001, "ci95": [clean - 0.02, clean + 0.02]},
                "2": {"point": clean, "ci95": [clean - 0.02, clean + 0.02]},
            },
            "mean_delta_vs_frozen": -0.0004,
            "delta_vs_frozen_ci95": [-0.0074, 0.0076],
        },
        "cross": {
            "mean": noisy_mean,
            "sd": 0.000521365334748207,
            "mean_checkpoint_ci95": [noisy_mean - 0.02, noisy_mean + 0.02],
            "per_seed": {
                "42": {"point": noisy_mean, "ci95": [noisy_mean - 0.02, noisy_mean + 0.02]},
                "1": {"point": noisy_mean, "ci95": [noisy_mean - 0.02, noisy_mean + 0.02]},
                "2": {"point": noisy_mean, "ci95": [noisy_mean - 0.02, noisy_mean + 0.02]},
            },
            "mean_delta_vs_frozen": -0.02,
            "delta_vs_frozen_ci95": [-0.03, -0.01],
        },
        "cross_minus_low": {
            "mean": noisy_mean - clean,
            "sd": 0.0005676209244504899,
            "mean_checkpoint_ci95": [delta_lo, delta_hi],
            "per_seed": {
                "42": {"point": noisy_mean - clean, "ci95": [delta_lo, delta_hi]},
                "1": {"point": noisy_mean - clean, "ci95": [delta_lo, delta_hi]},
                "2": {"point": noisy_mean - clean, "ci95": [delta_lo, delta_hi]},
            },
            "loo_min": delta_lo,
            "loo_max": delta_hi,
        },
        "frozen": {"point": FROZEN[metric], "ci95": [FROZEN[metric] - 0.02, FROZEN[metric] + 0.02]},
    }


def _stratum_cell(noisy_offsets: dict[str, float], delta_bounds: dict[str, tuple[float, float]]) -> dict:
    return {
        "metrics": {
            metric: _metric(
                metric,
                CLEAN[metric] + noisy_offsets[metric],
                delta_bounds[metric][0],
                delta_bounds[metric][1],
            )
            for metric in r.METRICS
        },
        "metric_version": "empirical-roc-v2",
        "training_seeds": [42, 1, 2],
        "n_pairs": 5308,
        "n_subjects": 82,
        "loo_valid": 82,
        "loo_unavailable": 0,
        "bootstrap": {
            "seed": 0,
            "requested": 2000,
            "valid": 2000,
            "shared_draws_across_all_models": True,
            "positive_weight": "owner multiplicity once",
            "negative_weight": "endpoint multiplicity product",
            "thresholds_reselected": True,
            "conditioning": CONDITIONING,
        },
        "scope": SCOPE,
        "training_identity_independence": "unverified",
        "causal_source_evidence": False,
        "publication_ready": False,
    }


def _offsets(seed: int) -> dict[str, float]:
    return {
        "roc_auc": -0.0212 - 0.0004 * seed,
        "eer_interpolated": 0.0171,
        "eer_discrete_minimax": 0.0200,
        "tar@far=0.01": -0.0233,
        "tar@far=0.001": -0.0248,
    }


def _bounds(seed: int, metric: str) -> tuple[float, float]:
    base = {
        "roc_auc": (-0.0260, -0.0164),
        "eer_interpolated": (-0.0039, 0.0409),
        "eer_discrete_minimax": (-0.0020, 0.0450),
        "tar@far=0.01": (-0.0962, 0.0061),
        "tar@far=0.001": (-0.0857, 0.0067),
    }[metric]
    return (base[0] - 0.001 * seed, base[1] + 0.001 * seed)


def make_summaries():
    def one_stratum(seed: int) -> dict:
        return _stratum_cell(_offsets(seed), {m: _bounds(seed, m) for m in r.METRICS})

    summary42 = {
        "overall": one_stratum(0),
        "large_gap_25plus": one_stratum(0),
        "execution_complete": True,
        "publication_ready": False,
        "coverage": copy.deepcopy(COVERAGE),
        "permutation_seed": 42,
        "low_role": "clean CROSS",
        "cross_role": "noisy CROSS",
        "direction": "noise_minus_clean",
        "limitation": "fixed permutation/checkpoints",
    }
    summary12 = {
        "execution_complete": True,
        "publication_ready": False,
        "coverage": copy.deepcopy(COVERAGE),
        "permutation_seeds": [1, 2],
        "training_seeds": [42, 1, 2],
        "evaluator_version": "oriented-noise-evaluator-v2-root-e385",
        "low_role": "clean CROSS",
        "cross_role": "noisy CROSS",
        "direction": "noise_minus_clean",
        "limitation": "fixed permutations/checkpoints",
        "permutation_seed1": {
            "overall": one_stratum(1),
            "large_gap_25plus": one_stratum(1),
        },
        "permutation_seed2": {
            "overall": one_stratum(2),
            "large_gap_25plus": one_stratum(2),
        },
    }
    return summary42, summary12


# ---------------------------------------------------------------------------
# Happy path.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stratum", r.STRATA)
def test_table_has_fifteen_rows_per_stratum(stratum):
    summary42, summary12 = make_summaries()
    text = r.table(summary42, summary12, stratum)
    assert r.row_count(summary42, summary12, stratum) == 15
    assert text.count(r"\midrule") == 1
    assert text.count(r"\toprule") == 1 and text.count(r"\bottomrule") == 1
    for arm in ("42", "1", "2"):
        assert text.count(f"\n{arm} & ") == 5


def test_metric_labels_and_meanings_preserved():
    summary42, summary12 = make_summaries()
    text = r.table(summary42, summary12, "overall")
    for label in r.METRIC_LABELS.values():
        assert label in text
    assert "Interpolated EER" in text and "Discrete minimax EER" in text
    assert r"TAR@FAR=1\%" in text and r"TAR@FAR=0.1\%" in text
    assert "Frozen" in text and "Clean mean$\\pm$SD" in text and "Noisy mean$\\pm$SD" in text


def test_table_star_has_compact_width_and_nominal_interval_note():
    summary42, summary12 = make_summaries()
    text = r.table_star(summary42, summary12, "overall", caption="Noise sensitivity")
    assert r"\begin{table*}" in text and r"\end{table*}" in text
    assert r"\setlength{\tabcolsep}{3.5pt}" in text
    assert "not multiplicity-adjusted" in text
    assert "not deployment calibration" in text
    assert "not an empirical identity-error rate" in text


# ---------------------------------------------------------------------------
# Exact native types.
# ---------------------------------------------------------------------------

def test_float_seed_is_rejected():
    summary42, summary12 = make_summaries()
    summary12["permutation_seed1"]["overall"]["training_seeds"] = [42.0, 1, 2]
    with pytest.raises(r.SummaryContractError, match="exact int"):
        r.table(summary42, summary12, "overall")


def test_bool_publication_ready_is_rejected():
    summary42, summary12 = make_summaries()
    summary42["overall"]["publication_ready"] = 0
    with pytest.raises(r.SummaryContractError, match="exact bool"):
        r.table(summary42, summary12, "overall")


def test_int_metric_value_is_rejected():
    summary42, summary12 = make_summaries()
    summary42["overall"]["metrics"]["roc_auc"]["frozen"]["point"] = 1
    with pytest.raises(r.SummaryContractError, match="finite native float"):
        r.table(summary42, summary12, "overall")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected(bad):
    summary42, summary12 = make_summaries()
    summary42["overall"]["metrics"]["roc_auc"]["low"]["mean"] = bad
    with pytest.raises(r.SummaryContractError, match="finite native float"):
        r.table(summary42, summary12, "overall")


# ---------------------------------------------------------------------------
# Seed sets and arms.
# ---------------------------------------------------------------------------

def test_wrong_training_seed_set_is_rejected():
    summary42, summary12 = make_summaries()
    summary12["permutation_seed2"]["overall"]["training_seeds"] = [42, 1]
    with pytest.raises(r.SummaryContractError, match="exact list of 3"):
        r.table(summary42, summary12, "overall")


def test_reordered_training_seeds_are_rejected():
    summary42, summary12 = make_summaries()
    summary42["overall"]["training_seeds"] = [1, 42, 2]
    with pytest.raises(r.SummaryContractError, match="expected"):
        r.table(summary42, summary12, "overall")


def test_wrong_permutation_seed_on_arm42_is_rejected():
    summary42, summary12 = make_summaries()
    summary42["permutation_seed"] = 7
    with pytest.raises(r.SummaryContractError, match="arm 42"):
        r.table(summary42, summary12, "overall")


def test_wrong_permutation_seed_list_on_arm12_is_rejected():
    summary42, summary12 = make_summaries()
    summary12["permutation_seeds"] = [1, 3]
    with pytest.raises(r.SummaryContractError, match="permutation_seeds"):
        r.table(summary42, summary12, "overall")


def test_missing_second_permutation_arm_is_rejected():
    summary42, summary12 = make_summaries()
    del summary12["permutation_seed2"]
    with pytest.raises(r.SummaryContractError, match="perm1/perm2"):
        r.table(summary42, summary12, "overall")


# ---------------------------------------------------------------------------
# Scope gates.
# ---------------------------------------------------------------------------

def test_publication_ready_true_is_rejected():
    summary42, summary12 = make_summaries()
    summary12["permutation_seed1"]["overall"]["publication_ready"] = True
    with pytest.raises(r.SummaryContractError, match="must remain False"):
        r.table(summary42, summary12, "overall")


def test_execution_incomplete_is_rejected():
    summary42, summary12 = make_summaries()
    summary42["execution_complete"] = False
    with pytest.raises(r.SummaryContractError, match="execution_complete must be True"):
        r.table(summary42, summary12, "overall")


def test_identity_independence_verified_is_rejected():
    summary42, summary12 = make_summaries()
    summary42["overall"]["training_identity_independence"] = "verified"
    with pytest.raises(r.SummaryContractError, match="unverified"):
        r.table(summary42, summary12, "overall")


def test_causal_source_evidence_true_is_rejected():
    summary42, summary12 = make_summaries()
    summary12["permutation_seed2"]["large_gap_25plus"]["causal_source_evidence"] = True
    with pytest.raises(r.SummaryContractError, match="causal_source_evidence"):
        r.table(summary42, summary12, "large_gap_25plus")


def test_scope_missing_multiplicity_disclaimer_is_rejected():
    summary42, summary12 = make_summaries()
    summary42["overall"]["scope"] = "test-derived ROC not deployment calibration"
    with pytest.raises(r.SummaryContractError, match="not multiplicity"):
        r.table(summary42, summary12, "overall")


def test_conditioning_without_population_disclaimer_is_rejected():
    summary42, summary12 = make_summaries()
    summary42["overall"]["bootstrap"]["conditioning"] = "fixed checkpoints"
    with pytest.raises(r.SummaryContractError, match="conditioning"):
        r.table(summary42, summary12, "overall")


def test_incomplete_bootstrap_draws_are_rejected():
    summary42, summary12 = make_summaries()
    summary42["overall"]["bootstrap"]["valid"] = 1999
    with pytest.raises(r.SummaryContractError, match="valid"):
        r.table(summary42, summary12, "overall")


def test_unshared_bootstrap_draws_are_rejected():
    summary42, summary12 = make_summaries()
    summary42["overall"]["bootstrap"]["shared_draws_across_all_models"] = False
    with pytest.raises(r.SummaryContractError, match="shared draws"):
        r.table(summary42, summary12, "overall")


# ---------------------------------------------------------------------------
# Rejected labels.
# ---------------------------------------------------------------------------

def test_ensemble_metric_label_is_rejected():
    summary42, summary12 = make_summaries()
    summary42["overall"]["metrics"]["ensemble_auc"] = copy.deepcopy(
        summary42["overall"]["metrics"]["roc_auc"]
    )
    with pytest.raises(r.SummaryContractError, match="forbidden label"):
        r.table(summary42, summary12, "overall")


def test_population_key_is_rejected():
    summary42, summary12 = make_summaries()
    summary12["permutation_seed1"]["overall"]["seed_population_ci"] = [0.0, 1.0]
    with pytest.raises(r.SummaryContractError, match="population"):
        r.table(summary42, summary12, "overall")


def test_causal_key_is_rejected():
    summary42, summary12 = make_summaries()
    summary12["permutation_seed2"]["overall"]["causal_effect"] = 1.0
    with pytest.raises(r.SummaryContractError, match="causal"):
        r.table(summary42, summary12, "overall")


def test_extra_metric_in_stack_is_rejected():
    summary42, summary12 = make_summaries()
    summary42["overall"]["metrics"]["roc_auc"]["ensemble"] = {"point": 0.5}
    with pytest.raises(r.SummaryContractError, match="forbidden label"):
        r.table(summary42, summary12, "overall")


# ---------------------------------------------------------------------------
# Interval ordering and consistency.
# ---------------------------------------------------------------------------

def test_reversed_ci_is_rejected():
    summary42, summary12 = make_summaries()
    summary42["overall"]["metrics"]["roc_auc"]["cross_minus_low"]["mean_checkpoint_ci95"] = [
        -0.01,
        -0.03,
    ]
    with pytest.raises(r.SummaryContractError, match="ordered"):
        r.table(summary42, summary12, "overall")


def test_negative_sd_is_rejected():
    summary42, summary12 = make_summaries()
    summary42["overall"]["metrics"]["roc_auc"]["low"]["sd"] = -0.1
    with pytest.raises(r.SummaryContractError, match="nonnegative"):
        r.table(summary42, summary12, "overall")


def test_inconsistent_shared_clean_value_across_arms_is_rejected():
    summary42, summary12 = make_summaries()
    cell = summary42["overall"]["metrics"]["roc_auc"]
    cell["low"]["mean"] += 0.01
    cell["cross"]["mean"] += 0.01  # keep arm 42 internally consistent
    with pytest.raises(r.SummaryContractError, match="shared clean_mean"):
        r.table(summary42, summary12, "overall")


def test_inconsistent_shared_frozen_value_across_arms_is_rejected():
    summary42, summary12 = make_summaries()
    summary12["permutation_seed1"]["large_gap_25plus"]["metrics"]["tar@far=0.01"]["frozen"][
        "point"
    ] = 0.1234
    with pytest.raises(r.SummaryContractError, match="shared frozen"):
        r.table(summary42, summary12, "large_gap_25plus")


def test_inconsistent_coverage_across_sources_is_rejected():
    summary42, summary12 = make_summaries()
    summary12["coverage"]["retained_positives"] = 2653
    with pytest.raises(r.SummaryContractError, match="coverage"):
        r.table(summary42, summary12, "overall")


def test_delta_not_matching_cross_minus_low_is_rejected():
    summary42, summary12 = make_summaries()
    summary42["overall"]["metrics"]["roc_auc"]["cross_minus_low"]["mean"] = 0.5
    with pytest.raises(r.SummaryContractError, match="disagrees"):
        r.table(summary42, summary12, "overall")


def test_unknown_stratum_is_rejected():
    summary42, summary12 = make_summaries()
    with pytest.raises(r.SummaryContractError, match="declared stratum"):
        r.table(summary42, summary12, "small_gap")


def test_wrong_metric_set_is_rejected():
    summary42, summary12 = make_summaries()
    del summary42["overall"]["metrics"]["eer_discrete_minimax"]
    with pytest.raises(r.SummaryContractError, match="exact ROC-v2 metric set"):
        r.table(summary42, summary12, "overall")


def test_metrics_are_finite_in_output():
    summary42, summary12 = make_summaries()
    for stratum in r.STRATA:
        for line in r.table(summary42, summary12, stratum).splitlines():
            assert "nan" not in line and "inf" not in line
    assert math.isfinite(CLEAN["roc_auc"])


@pytest.mark.parametrize("field,value", [("training_seeds", [42., 1, 2]), ("training_seeds", [1, 42, 2])])
def test_top_level_training_seeds_are_exact(field, value):
    summary42, summary12 = make_summaries()
    summary12[field] = value
    with pytest.raises(r.SummaryContractError):
        r.table(summary42, summary12, "overall")


def test_unexpected_permutation_block_refused():
    summary42, summary12 = make_summaries()
    summary12["permutation_seed3"] = copy.deepcopy(summary12["permutation_seed1"])
    with pytest.raises(r.SummaryContractError, match="exact perm1/perm2"):
        r.table(summary42, summary12, "overall")


@pytest.mark.parametrize("field,value", [("seed", True), ("seed", 1), ("requested", 2000.), ("requested", 1000), ("positive_weight", "endpoint multiplicity product")])
def test_bootstrap_contract_refused(field, value):
    summary42, summary12 = make_summaries()
    summary42["overall"]["bootstrap"][field] = value
    with pytest.raises(r.SummaryContractError):
        r.table(summary42, summary12, "overall")


def test_float_coverage_cannot_equal_integer_coverage():
    summary42, summary12 = make_summaries()
    summary12["coverage"]["retained_positives"] = 2654.
    with pytest.raises(r.SummaryContractError):
        r.table(summary42, summary12, "overall")


@pytest.mark.parametrize("field,value", [("n_pairs", True), ("n_subjects", 0), ("loo_valid", 81)])
def test_typed_and_partitioned_counts_required(field, value):
    summary42, summary12 = make_summaries()
    summary42["overall"][field] = value
    with pytest.raises(r.SummaryContractError):
        r.table(summary42, summary12, "overall")

