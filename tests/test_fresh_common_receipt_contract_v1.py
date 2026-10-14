"""Synthetic positive/negative tests for the E477 decoded-receipt validator.

No project receipt, artifact or byte is read; every fixture is an
in-memory synthetic mapping. These tests never assert scientific acceptance.
"""


import pytest

from scripts import fresh_common_receipt_contract_v1 as rv

HEX = "a" * 64


def positive():
    return {
        "schema_version": 1,
        "experiment": rv.EXPERIMENT,
        "observed_utc": "2026-10-09T15:38:43Z",
        "cell": "random_head_lr1e-06_s42",
        "producer_exit": 0,
        "reader_exit": 0,
        "accepted": True,
        "native_manifest": {
            "path": "metrics/common_mechanism_fresh_weak_headlow42_e460_20261009/summary.manifest.json",
            "bytes": 6162693,
            "sha256": HEX,
        },
        "reader_log": {"path": ".work/fresh_common_acceptance_e469/reader_e461.log", "bytes": 528, "sha256": HEX},
        "reader_source": {"path": ".work/check_common_fresh_e461.py", "bytes": 16974, "sha256": HEX},
        "coordinator_status": {"path": ".work/fresh_common_acceptance_e469/status.json", "bytes": 5448, "sha256": HEX},
        "reader_report": {
            "accepted": True,
            "cell": "random_head_lr1e-06_s42",
            "physical_targets": 33115,
            "outputs": 2,
            "stable_two_phase": True,
            "numerical_age_probe_refit": True,
            "probe_refit_inside_bootstrap": False,
            "numerical_cached_vector_replay": True,
            "replayed_probes": 6,
            "replayed_diagnostics": 4,
            "replayed_separability_families": 2,
            "neural_inference_replayed": False,
            "historical_weak_recovery": False,
            "fresh_weak_family": True,
            "native_training_history_schema_rechecked": False,
            "human_identity_audit": False,
            "mechanism_claim": False,
        },
        "publication_ready": False,
        "presentation_accepted": False,
        "human_identity_audit": False,
        "mechanism_complete": False,
    }


def test_positive_accepts_and_never_claims_bytes():
    out = rv.validate_receipt(positive())
    assert out["cell"] == "random_head_lr1e-06_s42"
    assert out["physical_targets"] == 33115
    assert out["decoded_accounting_only"] is True
    assert out["file_bytes_verified"] is False
    assert out["manifest_closed"] is False
    assert out["scientific_accepted"] is False


def test_canonical_cell_regex():
    assert rv.is_canonical_cell("random_head_lr1e-06_s42")
    assert rv.is_canonical_cell("lookalike_tail_lr1e-05_s2")
    for bad in ("random_head_lr1e-06_s7", "random_head_lr1e-07_s42", "random_head_lr1e-06_s42 ", "Random_head_lr1e-06_s42"):
        assert not rv.is_canonical_cell(bad)


@pytest.mark.parametrize("name,path", [
    ("native_manifest", "metrics/x/summary.manifest.json"),
    ("reader_log", ".work/x.log"),
    ("reader_source", ".work/x.py"),
    ("coordinator_status", ".work/s.json"),
])
def test_each_record_is_bound_to_its_domain(name, path):
    r = positive()
    r[name]["path"] = path
    assert rv.validate_receipt(r)["records"][name]["path"] == path
    r[name]["path"] = "data/private.bin"
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)


@pytest.mark.parametrize("path", [".work/file:stream", ".work/nul\x00.py", ".work/line\n.py", "R:/other.py", ".work//x.py"])
def test_unsafe_windows_or_control_path_rejected(path):
    r = positive()
    r["reader_source"]["path"] = path
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_exact_top_level_receipt_shape(change):
    r = positive()
    if change == "missing":
        del r["reader_report"]
    else:
        r["neural_inference_replayed"] = True
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)


def test_bool_int_type_exactness_rejected():
    r = positive()
    r["accepted"] = 1  # int, not bool
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)
    r = positive()
    r["producer_exit"] = False  # bool, not int
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)
    r = positive()
    r["reader_report"]["outputs"] = True  # bool, not int
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)


def test_exact_exits_required():
    for key in ("producer_exit", "reader_exit"):
        r = positive()
        r[key] = 1
        with pytest.raises(rv.ReceiptError):
            rv.validate_receipt(r)


def test_reader_cell_must_match_receipt_cell():
    r = positive()
    r["reader_report"]["cell"] = "random_tail_lr1e-05_s1"
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)


def test_report_flags_and_counts_exact():
    for key, bad in (
        ("neural_inference_replayed", True),
        ("fresh_weak_family", False),
        ("accepted", False),
        ("mechanism_claim", True),
        ("replayed_probes", 5),
        ("outputs", 3),
    ):
        r = positive()
        r["reader_report"][key] = bad
        with pytest.raises(rv.ReceiptError):
            rv.validate_receipt(r)


def test_positive_physical_targets_required():
    r = positive()
    r["reader_report"]["physical_targets"] = 0
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)
    r = positive()
    r["reader_report"]["physical_targets"] = -5
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)


def test_extra_or_missing_report_keys_rejected():
    r = positive()
    r["reader_report"]["extra"] = False
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)
    r = positive()
    del r["reader_report"]["mechanism_claim"]
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)


def test_four_record_schemas_and_hash_shape():
    r = positive()
    r["reader_log"]["extra"] = 1
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)
    r = positive()
    r["reader_source"]["sha256"] = "A" * 64  # uppercase
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)
    r = positive()
    r["coordinator_status"]["sha256"] = "a" * 63
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)
    r = positive()
    r["native_manifest"]["bytes"] = 0
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)
    r = positive()
    r["reader_log"]["path"] = "/abs/reader.log"
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)
    r = positive()
    r["native_manifest"]["path"] = ".work/summary.manifest.json"  # wrong domain
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)
    r = positive()
    r["reader_source"]["path"] = ".work/../escape.py"
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)


@pytest.mark.parametrize("flag", ["publication_ready", "presentation_accepted", "human_identity_audit", "mechanism_complete"])
def test_forbidden_true_flags_rejected(flag):
    r = positive()
    r[flag] = True
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)
    r = positive()
    r[flag] = 0  # int, not exact bool false
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)


def test_schema_version_and_experiment_exact():
    r = positive()
    r["schema_version"] = True
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)
    r = positive()
    r["experiment"] = "root-fresh-common-numerical-acceptance-e469"
    with pytest.raises(rv.ReceiptError):
        rv.validate_receipt(r)
