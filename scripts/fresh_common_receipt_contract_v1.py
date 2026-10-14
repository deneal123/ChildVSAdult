"""Decoded-accounting validator for the root fresh-common receipt (E461).

Pure stdlib. Validates ONLY the decoded fields of a root acceptance receipt:
exact bool/int types, schema and experiment identity, the canonical strong-cell
regex, reader-cell agreement, exact success exits, the exact reader-report
flags/counts and a positive physical-target count, the four record schemas with
their declared path domains and SHA-256 shape, and the four must-be-false
publication/claim flags.

It may not claim file bytes, manifest closure, numerical replay or scientific
acceptance: a caller must independently bind every referenced artifact before
any such statement.
"""

from __future__ import annotations

import re

__all__ = [
    "ReceiptError",
    "validate_receipt",
    "is_canonical_cell",
    "SCHEMA_VERSION",
    "EXPERIMENT",
]

SCHEMA_VERSION = 1
EXPERIMENT = "root-fresh-common-numerical-acceptance-e461"
CELL = re.compile(r"(random|lookalike)_(head|tail|full)_lr(1e-06|1e-05)_s(42|1|2)\Z")
_HEX = frozenset("0123456789abcdef")

# record name -> required relative path domain
_DOMAINS = {
    "native_manifest": "metrics/",
    "reader_log": ".work/",
    "reader_source": ".work/",
    "coordinator_status": ".work/",
}
_RECORD_KEYS = frozenset({"path", "bytes", "sha256"})

# exact reader-report booleans
_FLAGS = {
    "accepted": True,
    "stable_two_phase": True,
    "numerical_age_probe_refit": True,
    "probe_refit_inside_bootstrap": False,
    "numerical_cached_vector_replay": True,
    "neural_inference_replayed": False,
    "historical_weak_recovery": False,
    "fresh_weak_family": True,
    "native_training_history_schema_rechecked": False,
    "human_identity_audit": False,
    "mechanism_claim": False,
}
# exact reader-report integer counts
_COUNTS = {
    "outputs": 2,
    "replayed_probes": 6,
    "replayed_diagnostics": 4,
    "replayed_separability_families": 2,
}
_REPORT_KEYS = frozenset({"cell", "physical_targets"}) | set(_FLAGS) | set(_COUNTS)
_FORBIDDEN = (
    "publication_ready",
    "presentation_accepted",
    "human_identity_audit",
    "mechanism_complete",
)


class ReceiptError(ValueError):
    """One rejected decoded receipt; messages never quote private material."""


def is_canonical_cell(value):
    """True only for an exact canonical strong checkpoint spelling."""
    return type(value) is str and CELL.fullmatch(value) is not None


def _fail(where, reason):
    raise ReceiptError(f"{where}: {reason}")


def _record(name, record):
    if type(record) is not dict or set(record) != set(_RECORD_KEYS):
        _fail(name, "exact {path,bytes,sha256} record schema required")
    path, size, digest = record["path"], record["bytes"], record["sha256"]
    if (
        type(path) is not str or not path or "\\" in path or path.startswith("/")
        or ":" in path or any(ord(char) < 32 for char in path)
    ):
        _fail(name, "relative posix path required")
    if any(part in ("", ".", "..") for part in path.split("/")):
        _fail(name, "canonical non-traversing path segments required")
    if not path.startswith(_DOMAINS[name]):
        _fail(name, "path outside the declared domain")
    if type(size) is not int or size <= 0:
        _fail(name, "positive byte count required")
    if type(digest) is not str or len(digest) != 64 or not set(digest) <= _HEX:
        _fail(name, "64-char lowercase sha256 shape required")
    return {"path": path, "bytes": size, "sha256": digest}


def _report(receipt):
    rep = receipt.get("reader_report")
    if type(rep) is not dict or set(rep) != _REPORT_KEYS:
        _fail("reader_report", "exact report schema required")
    for key, want in _FLAGS.items():
        if type(rep[key]) is not bool or rep[key] is not want:
            _fail("reader_report", f"{key} must be exact bool {want}")
    for key, want in _COUNTS.items():
        if type(rep[key]) is not int or rep[key] != want:
            _fail("reader_report", f"{key} must be exact int {want}")
    if type(rep["physical_targets"]) is not int or rep["physical_targets"] <= 0:
        _fail("reader_report", "positive physical target count required")
    if not is_canonical_cell(rep["cell"]):
        _fail("reader_report", "canonical reader cell required")
    return rep


def validate_receipt(receipt):
    """Validate decoded receipt accounting; return a normalized non-claim summary."""
    if type(receipt) is not dict:
        _fail("receipt", "decoded mapping required")
    keys = {
        "schema_version", "experiment", "observed_utc", "cell", "producer_exit",
        "reader_exit", "accepted", "reader_report", *_DOMAINS, *_FORBIDDEN,
    }
    if set(receipt) != keys:
        _fail("receipt", "exact root receipt schema required")
    if type(receipt.get("schema_version")) is not int or receipt["schema_version"] != SCHEMA_VERSION:
        _fail("schema_version", f"must be exact int {SCHEMA_VERSION}")
    if type(receipt.get("experiment")) is not str or receipt["experiment"] != EXPERIMENT:
        _fail("experiment", "exact experiment identity required")
    if type(receipt.get("observed_utc")) is not str or not receipt["observed_utc"].strip():
        _fail("observed_utc", "non-empty string required")
    if type(receipt.get("accepted")) is not bool or receipt["accepted"] is not True:
        _fail("accepted", "must be exact bool true")
    for key in ("producer_exit", "reader_exit"):
        if type(receipt.get(key)) is not int or receipt[key] != 0:
            _fail(key, "must be exact int 0")
    if not is_canonical_cell(receipt.get("cell")):
        _fail("cell", "canonical strong cell required")
    report = _report(receipt)
    if report["cell"] != receipt["cell"]:
        _fail("reader_report", "reader cell must match the receipt cell")
    for key in _FORBIDDEN:
        if type(receipt.get(key)) is not bool or receipt[key] is not False:
            _fail(key, "must be exact bool false")
    records = {name: _record(name, receipt.get(name)) for name in _DOMAINS}
    return {
        "cell": receipt["cell"],
        "physical_targets": report["physical_targets"],
        "records": records,
        "publication_ready": False,
        "presentation_accepted": False,
        "human_identity_audit": False,
        "mechanism_complete": False,
        "decoded_accounting_only": True,
        "file_bytes_verified": False,
        "manifest_closed": False,
        "scientific_accepted": False,
    }
