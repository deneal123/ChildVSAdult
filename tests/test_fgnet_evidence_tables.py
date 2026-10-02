from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from age_gap.common.manifest import write_experiment_manifest
from scripts.render_fgnet_evidence import (
    cmc_curves,
    interval,
    render_comparators,
    render_errors,
    render_retrieval,
    verified_result,
)


def comparators():
    metrics = {"roc_auc": .8, "eer": .2, "tar@far=0.01": .3}
    strata = ("overall", "large_gap_25plus")
    runs = [{"seed": seed, "provenance": {"manifest_checkpoint_match": True},
             "paired_vs_frozen": {s: {"n_pairs": 430, "n_subjects": 75, "delta_ci95": [-.02, .04]} for s in strata},
             "leave_one_subject_out_vs_frozen": {s: {"minimum_delta_auc": .01, "maximum_delta_auc": .02} for s in strata}}
            for seed in (1, 2, 42)]
    method = {"runs": runs, "aggregate_across_seeds": {s: {k: {"mean": v, "std": .001, "n_seeds": 3} for k, v in metrics.items()} for s in strata}}
    return {"protocol": "endpoint_age_matched", "frozen_common_backbone": {"metrics": {s: metrics for s in strata}},
            "comparators": {k: copy.deepcopy(method) for k in ("mtlface-common-protocol", "cacon-common-protocol")}}


def retrieval():
    metric = {"n_queries": 57, "recall@1": .3, "recall@5": .5, "recall@10": .6, "mrr": .4}
    strata = ("overall", "gap_25_plus")
    test = {"frozen": {s: metric for s in strata}, "tuned": {s: metric for s in strata},
            "paired_bootstrap": {s: {"n_queries": 57, "delta_ci95": {"recall@1": [-.1, .1]}} for s in strata}}
    return {"primary_split": "test", "training_identity_independence": "unverified",
            "results": {f"tuned_seed{seed}": {"test": copy.deepcopy(test)} for seed in (1, 2, 42)}}


def test_comparator_caption_distinguishes_seed_std_from_subject_ci():
    rendered = render_comparators(comparators())
    assert "not intervals of seed means" in rendered
    assert "both pair endpoints" in rendered
    assert "unverified" in rendered
    assert "$[-0.0200,+0.0400]$" in rendered


@pytest.mark.parametrize("fault", ["protocol", "seed", "checkpoint", "sample"])
def test_comparator_rejects_inconsistent_evidence(fault):
    payload = comparators()
    run = payload["comparators"]["mtlface-common-protocol"]["runs"][0]
    if fault == "protocol":
        payload["protocol"] = "legacy_random"
    elif fault == "seed":
        run["seed"] = 42
    elif fault == "checkpoint":
        run["provenance"]["manifest_checkpoint_match"] = False
    else:
        run["paired_vs_frozen"]["overall"]["n_subjects"] = 74
    with pytest.raises(ValueError):
        render_comparators(payload)


def test_retrieval_is_conditional_on_fixed_gallery_not_independence():
    rendered = render_retrieval(retrieval())
    assert "gallery is fixed, not resampled" in rendered
    assert "intervals include zero" in rendered
    assert "Legacy training manifests are absent" in rendered


def test_retrieval_rejects_different_frozen_or_query_sample():
    payload = retrieval()
    payload["results"]["tuned_seed1"]["test"]["frozen"]["overall"]["recall@1"] = .9
    with pytest.raises(ValueError):
        render_retrieval(payload)


@pytest.mark.parametrize("bounds", [[float("nan"), 1], [2, 1], [1]])
def test_interval_rejects_nonfinite_or_malformed(bounds):
    with pytest.raises(ValueError):
        interval(bounds)


def test_result_binding_checks_inputs_and_outputs_not_summary_equality(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("{}", encoding="utf-8")
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"full": True}), encoding="utf-8")
    write_experiment_manifest(result.with_suffix(".manifest.json"), experiment="test",
                              parameters={}, metrics={"summary": True}, inputs=[source], outputs=[result])
    assert verified_result(result, tmp_path) == {"full": True}
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="inputs checksum"):
        verified_result(result, tmp_path)
    source.write_text("{}", encoding="utf-8")
    result.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="result checksum"):
        verified_result(result, tmp_path)


def test_cmc_truncated_before_gallery_size_need_not_reach_one():
    payload = retrieval()
    curve = [.3, .35, .4, .45, .5, .52, .54, .56, .58, .6]
    payload["results"]["frozen_reference"] = {"cmc_test": curve}
    for seed in (1, 2, 42):
        payload["results"][f"tuned_seed{seed}"].update(cmc_test_frozen=curve, cmc_test_tuned=curve)
    assert len(cmc_curves(payload)["Frozen"]) == 10
    payload["results"]["tuned_seed1"]["cmc_test_tuned"] = [*curve[:-1], .59]
    with pytest.raises(ValueError, match="Recall@K differ"):
        cmc_curves(payload)


@pytest.mark.parametrize("kind,filename,render", [
    ("COMPARATORS", "metrics/comparator_fgnet_endpoint_age_matched.json", render_comparators),
    ("RETRIEVAL", "metrics/fgnet_retrieval_20261002/fgnet_retrieval_study.json", render_retrieval),
    ("ERRORS", "metrics/fgnet_error_breakdown/fgnet_error_breakdown.json", render_errors),
])
def test_supplement_generated_blocks_match_bound_local_evidence(kind, filename, render):
    root = Path(__file__).resolve().parents[1]
    path = root / filename
    if not path.is_file():
        pytest.skip("controlled-access local experiment artifact is absent")
    expected = render(verified_result(path, root))
    text = (root / "latex/papers/journal-1-tbiom/en/supplement.tex").read_text(encoding="utf-8")
    begin, end = f"% BEGIN GENERATED FGNET {kind}", f"% END GENERATED FGNET {kind}"
    assert text.count(begin) == text.count(end) == 1
    assert text[text.index(begin):text.index(end) + len(end)] + "\n" == expected


def errors():
    sample = {"n_pos": 10, "n_neg": 10, "false_accepts": 0, "false_rejects": 3,
              "fmr": 0., "fnmr": .3, "fmr_ci95": None, "fnmr_ci95": [.1, .5],
              "conditional_on_dev_threshold": True, "threshold_selection_variance_included": False}
    strata = ("overall", "gap_25_plus", "child_lt13_to_adult_gt25", "blur_bin_0", "blur_bin_1", "blur_bin_2")
    point = {"dev_fmr": .007, "strata": {s: copy.deepcopy(sample) for s in strata}}
    return {"threshold_provenance": "development negatives only; frozen on test",
            "acceptance_convention": "strict: accepted iff cosine > threshold",
            "training_identity_independence": "unverified",
            "splits": {model: {"test": {"operating_points": {"0.01": copy.deepcopy(point)}}}
                       for model in ("frozen", "tuned_seed1", "tuned_seed2", "tuned_seed42")}}


def test_error_table_excludes_calibration_uncertainty_and_deployment_guarantee():
    result = render_errors(errors())
    assert "calibration uncertainty is excluded" in result
    assert "under-resolved nominal 0.1" in result
    assert "not reliable risk bounds" in result
    assert "0/10" in result and "3/10" in result


@pytest.mark.parametrize("fault", ["threshold", "counts", "boundary_ci", "estimand"])
def test_error_table_rejects_wrong_estimand_or_false_zero_risk(fault):
    payload = errors()
    sample = payload["splits"]["frozen"]["test"]["operating_points"]["0.01"]["strata"]["overall"]
    if fault == "threshold":
        payload["threshold_provenance"] = "test"
    elif fault == "counts":
        sample["fnmr"] = .9
    elif fault == "boundary_ci":
        sample["fmr_ci95"] = [0., 0.]
    else:
        sample["threshold_selection_variance_included"] = True
    with pytest.raises(ValueError):
        render_errors(payload)
