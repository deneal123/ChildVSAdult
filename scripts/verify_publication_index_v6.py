"""Independent, stdlib-only verifier for a finished index-v6 physical snapshot.

Corrections over v2: alias bindings are preflighted before every read and re-resolved
between/after phases; the selection state carries full presence flags and binds every
existing gate's bytes; outside-root targets are allowed only when their canonical path
is on an explicit coordinator-supplied approved-file list (never inferred from native
data); recursive exact-type equality; pinned completion metadata and an independent
v1+v2 metadata projection recomputed from primary semantics (not by importing the
producer); completion proof claims are distinguished from historical native claims;
exact frozen v1 private-record predicate; fresh pre/post hash *and* size; privacy-
normalized errors; a redacted CLI failure path; and root-containment preflight before
every read.

It never imports the producers, ``export_public_evidence`` or helper integrity modules
and never executes worker/native benchmark code. It hashes only the files it is pointed
at (or that are on the approved list) locally and never publishes paths or digests.
It emits aggregate acceptance/counts only and makes no scientific-legality or
publication-readiness claim.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Frozen ancestry fragments (data copied from canonical sources at freeze time).
# ---------------------------------------------------------------------------
FROZEN_V1_INCLUDE = (
    "age_extraction_audit.manifest.json",
    "group_integrity_audit.manifest.json",
    "human_audit_pack_validation.manifest.json",
    "human_audit.manifest.json",
    "data_funnel.manifest.json",
    "pipeline_figure.manifest.json",
    "curation_lineage_20261003/summary.manifest.json",
    "curation_replay_global_*/summary.manifest.json",
    "constituent_integrity_20261003/summary.manifest.json",
    "constituent_policy_sensitivity_20261003/summary.manifest.json",
    "sensitivity_dedup.manifest.json",
    "independent_embedding_audit.manifest.json",
    "model_inventory.manifest.json",
    "pretraining_overlap_audit.manifest.json",
    "headline_stats.manifest.json",
    "fgnet_endpoint_subject_stats.manifest.json",
    "fgnet_endpoint_subject_stats_s*.manifest.json",
    "fgnet_endpoint_multiseed.manifest.json",
    "fgnet_retrieval_*/fgnet_retrieval_study.manifest.json",
    "fgnet_retrieval_*/retrieval_presentation.manifest.json",
    "fgnet_error_breakdown/fgnet_error_breakdown.manifest.json",
    "comparator_fgnet_endpoint_age_matched.manifest.json",
    "internal_endpoint_age_matched.manifest.json",
    "internal_metrics_v2_20261003/summary.manifest.json",
    "internal_metrics_v2_20261003/presentation.manifest.json",
    "fgnet_metrics_v2_20261003/summary.manifest.json",
    "fgnet_metrics_v2_20261003/presentation.manifest.json",
    "lfw_metrics_v2_20261003/summary.manifest.json",
    "lfw_bound_cache_*/lfw_bound_cache.manifest.json",
    "lfw_bound_evaluation_*/lfw_bound_evaluation.manifest.json",
    "lfw_bound_evaluation_*/lfw_presentation.manifest.json",
    "matched_arm_image_budget_*/summary.manifest.json",
    "delong_tests.manifest.json",
    "synthetic_parity.manifest.json",
    "sota_objectives.manifest.json",
    "noise_robust_baseline.manifest.json",
    "hardneg_multiseed.manifest.json",
    "hardneg_curve_full.manifest.json",
    "hardneg_curve_facenet_clean.manifest.json",
    "hardneg_curve_miner2.manifest.json",
    "hardneg_paired_pairs_clean.manifest.json",
    "subgroup_fmr_fnmr.manifest.json",
    "fairness_audit.manifest.json",
    "age_leakage.manifest.json",
    "headroom_audit.manifest.json",
    "multiseed_facenet_e10.manifest.json",
    "scaling_multiseed.manifest.json",
    "cross_platform_vk2reddit_3seed.manifest.json",
    "cross_source_reddit2vk_3seed.manifest.json",
    "cacd_vs/*.manifest.json",
    "cacd_serial_20261003/*.manifest.json",
    "cacd_serial_presentation_20261003/presentation.manifest.json",
    "sota_common_protocol/*.manifest.json",
    "strong_backbone_study/*.manifest.json",
    "strong_backbone_fixed8_bn_frozen_*/*.manifest.json",
    "adaface_fixed8_cuda_*/summary.manifest.json",
    "facenet_fixed8_validation*_v2_*/summary.manifest.json",
    "strong_queue_remaining*_v2_*/*/train/summary.manifest.json",
    "strong_queue_remaining*_v2_*/*/fgnet/summary.manifest.json",
    "strong_mechanism_*_v1_*/summary.manifest.json",
    "common_mechanism_*_v1_*/summary.manifest.json",
    "strong_native_trajector*_v1_*/summary.manifest.json",
    "native_person_age_baselines*_v2_*/summary.manifest.json",
    "weak_native_cache_keys_v2_*/summary.manifest.json",
    "strong_fixed8_roc_presentation*_v1_*/presentation.manifest.json",
    "matched_agegap_arms/*.manifest.json",
    "matched_agegap_fixed10_last/*.manifest.json",
)
FROZEN_V6_EXTRA_INCLUDE = (
    "common_mechanism_presentation*_v1_*/presentation.manifest.json",
    "age_probe_absolute_presentation*_v1_*/presentation.manifest.json",
    "common_pair_protocol*_v1_*/summary.manifest.json",
    "common_pair_protocol*_v2_*/summary.manifest.json",
    "common_mechanism_presentation*_v2_*/presentation.manifest.json",
    "age_probe_absolute_presentation*_v3_*/presentation.manifest.json",
    "common_pair_protocol*_v3_*/summary.manifest.json",
    "external_layout*_v2_*/summary.manifest.json",
    "scaling_layout*_v2_*/summary.manifest.json",
    "strong_native_trajector*_v3_*/summary.manifest.json",
    "strong_history_presentation*_v1_*/history.manifest.json",
    "strong_fixed_seed_mean*_v1_*/summary.manifest.json",
    "fixed_checkpoint_means_presentation*_v1_*/presentation.manifest.json",
    "checkpoint_validation_*_v6_*/summary.manifest.json",
    "fixed_validation_presentation*_v1_*/presentation.manifest.json",
    "oriented_cuda_noise_perm*_v2_*/noise_s*.manifest.json",
    "oriented_cuda_noise_perm*_v2_*/training-bound.manifest.json",
    "noise_permutations_presentation*_v1_*/presentation.manifest.json",
    "adaface_lookalike_head_low_s42_fgnet_*/summary.manifest.json",
    "common_mechanism_lookalike_head_low_s42_*/summary.manifest.json",
    "facenet_fixed8_validation_s*_e*_20261009/summary.manifest.json",
    "facenet_fixed8_validation_external*_20261009/summary.manifest.json",
)
FROZEN_V6_CURRENT_REQUIRED = (
    "common_mechanism_presentation11_v2_e339_20261009/presentation.manifest.json",
    "age_probe_absolute_presentation16_v3_full_e339_20261009/presentation.manifest.json",
    "strong_fixed8_roc_presentation11_v1_e339_20261009/presentation.manifest.json",
    "common_pair_protocol7_v2_20261008/summary.manifest.json",
    "external_layout_v2_20261008/summary.manifest.json",
    "scaling_layout_v2_20261008/summary.manifest.json",
    "strong_native_trajectories10_v3_e326_20261009/summary.manifest.json",
    "strong_history_presentation10_v1_layout2_e337_20261009/history.manifest.json",
    "strong_fixed_seed_mean4_v1_e350_20261009/summary.manifest.json",
    "fixed_checkpoint_means_presentation_v1_e362_20261009/presentation.manifest.json",
    "checkpoint_validation_tailhigh_s2_v6_e358_20261009/summary.manifest.json",
    "fixed_validation_presentation_v1_e368_20261009/presentation.manifest.json",
    "oriented_cuda_noise_perm1_v2_e381_20261009/training-bound.manifest.json",
    "oriented_cuda_noise_perm2_v2_e388_20261009/training-bound.manifest.json",
    "noise_perm12_cuda_fgnet_v2_e391_20261009/summary.manifest.json",
    "noise_permutations_presentation_v1_e397_20261009/presentation.manifest.json",
    "adaface_fixed8_cuda_lookalike_head_low_s42_e393_20261009/summary.manifest.json",
    "adaface_lookalike_head_low_s42_fgnet_e407_20261009/summary.manifest.json",
    "common_mechanism_lookalike_head_low_s42_e429_20261009/summary.manifest.json",
    "submission_visual_e434_20261009/summary.manifest.json",
)
CANONICAL_SOURCE_RELPATHS = (
    "scripts/build_publication_artifact_index_v6.py",
    "scripts/build_publication_artifact_index_v5.py",
    "scripts/build_publication_artifact_index_v4.py",
    "scripts/build_publication_artifact_index_v3.py",
    "scripts/build_publication_artifact_index_v2.py",
    "scripts/build_publication_artifact_index.py",
    "scripts/export_public_evidence.py",
    "src/age_gap/common/manifest.py",
    "src/age_gap/common/io.py",
)
SEMANTIC_CAMPAIGNS = ("strong_backbone_study", "matched_agegap_arms",
                      "matched_agegap_fixed10_last")
EXPERIMENT_NAME = "publication-artifact-index-audit-v6"
INDEX_FILENAME = "publication_artifact_index.json"
MANIFEST_FILENAME = "summary.manifest.json"

# ---------------------------------------------------------------------------
# Privacy patterns, re-declared from the frozen public-evidence policy.
# ---------------------------------------------------------------------------
PRIVATE_ROW_ID = re.compile(r"(?:vk_|reddit_)?-?\d{5,}_\d{5,}", re.IGNORECASE)
SECRET_MARKERS = re.compile(
    r"(vk1\.a\.|bearer\s+[a-z0-9._-]{20,}|client_secret\s*[=:]|api[_-]?key\s*[=:]|"
    r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|password\s*[=:])", re.IGNORECASE)
ABS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/(?:home|users|mnt|media|root|tmp)/)",
                      re.IGNORECASE)
EMBEDDED_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s\"'<>|()\[\],;]+|\\\\[^\s\"'<>|()\[\],;]+"
    r"|(?<![\w/])/(?:home|users|mnt|media|root|tmp)/[^\s\"'<>|()\[\],;]+)", re.IGNORECASE)
PRIVATE_PATH_MARKERS = ("/private/", "data/interim/faces/")
SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")
UNSAFE_KEY = re.compile(
    r"^(?:(?:image|crop|face|row|subject|person|pair)_?(?:sha1|sha256|md5|hash|ids?|path)|"
    r"(?:sha1|sha256|md5)_of_(?:image|crop|face|row|subject|pair)|"
    r"embeddings?|captions?|raw_responses?|(?:photo|crop|face)_url)$", re.IGNORECASE)


class CheckError(RuntimeError):
    """Base class; every message is a generic ``refused: <code>`` with no data."""


class RefusalError(CheckError):
    """Containment/preflight/scope refusal before a native read."""


class SchemaError(CheckError):
    """Malformed JSON, duplicate keys, non-finite numbers or wrong shape/type."""


class UnsafeContentError(CheckError):
    """Residual private/public-unsafe content in the sanitized public projection."""


class IntegrityError(CheckError):
    """Alias retarget, snapshot drift, binding/selection/metadata mismatch."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _is_int(value: Any) -> bool:
    return type(value) is int


def exact_equal(left: Any, right: Any) -> bool:
    """Recursive exact equality: same type at every level (so True != 1)."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(exact_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(exact_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


# ---------------------------------------------------------------------------
# Strict JSON (normalized errors, no key/value leakage).
# ---------------------------------------------------------------------------
def _reject_constant(_token: str) -> Any:
    raise SchemaError("refused: json-nonfinite")


def _parse_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise SchemaError("refused: json-nonfinite")
    return value


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise SchemaError("refused: json-duplicate-key")
        out[key] = value
    return out


def strict_loads(data: bytes) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise SchemaError("refused: not-utf8") from None
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicates,
                          parse_constant=_reject_constant, parse_float=_parse_float)
    except SchemaError:
        raise
    except json.JSONDecodeError:
        raise SchemaError("refused: json-malformed") from None


# ---------------------------------------------------------------------------
# Containment / approved external files / preflight.
# ---------------------------------------------------------------------------
def _canonical_dir(path: Path) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except (OSError, ValueError, RuntimeError):
        raise RefusalError("refused: root-unavailable") from None
    if not resolved.is_dir():
        raise RefusalError("refused: root-not-directory")
    return resolved


class ApprovedFiles:
    """Explicit coordinator-supplied allowlist of exact external canonical files."""

    def __init__(self, entries: dict[str, tuple[int | None, str | None]]):
        self._entries = entries

    @classmethod
    def load(cls, path: Path | None) -> ApprovedFiles:
        if path is None:
            return cls({})
        try:
            policy_bytes = Path(path).read_bytes()
            raw = strict_loads(policy_bytes)
        except (OSError, ValueError, UnicodeDecodeError):
            raise RefusalError("refused: allowlist-unreadable") from None
        if type(raw) is not dict or set(raw) != {"files"} or type(raw["files"]) is not list:
            raise RefusalError("refused: allowlist-shape")
        entries: dict[str, tuple[int | None, str | None]] = {}
        for item in raw["files"]:
            if (type(item) is not dict or set(item) != {"path", "bytes", "sha256"}
                    or type(item["path"]) is not str or not item["path"].strip()
                    or not Path(item["path"]).is_absolute()
                    or type(item["bytes"]) is not int or item["bytes"] < 0
                    or type(item["sha256"]) is not str
                    or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
                raise RefusalError("refused: allowlist-shape")
            try:
                key = str(Path(item["path"]).resolve(strict=True))
            except (OSError, ValueError, RuntimeError):
                raise RefusalError("refused: allowlist-unreadable") from None
            size = item.get("bytes")
            sha = item.get("sha256")
            if key in entries:
                raise RefusalError("refused: allowlist-duplicate-target")
            entries[key] = (size, sha)
        approved = cls(entries)
        approved._policy_alias = Path(path).absolute()
        approved._policy_target = approved._policy_alias.resolve(strict=True)
        approved._policy_bytes = policy_bytes
        if Path(path).read_bytes() != policy_bytes:
            raise IntegrityError("refused: allowlist-read-drift")
        approved._file_aliases = [(item["path"], str(Path(item["path"]).resolve(strict=True)))
                                 for item in raw["files"]]
        return approved

    def guard(self) -> None:
        if not hasattr(self, "_policy_alias"):
            return
        if (self._policy_alias.resolve(strict=True) != self._policy_target
                or self._policy_alias.read_bytes() != self._policy_bytes):
            raise IntegrityError("refused: allowlist-drift")
        for spelling, target in self._file_aliases:
            if str(Path(spelling).resolve(strict=True)) != target:
                raise IntegrityError("refused: allowlist-alias-drift")

    def __contains__(self, canonical: str) -> bool:
        return canonical in self._entries

    def expected(self, canonical: str) -> tuple[int | None, str | None]:
        return self._entries.get(canonical, (None, None))


def resolve_declared(root: Path, approved: ApprovedFiles, spelling: str) -> Path:
    """Resolve a declared path, refusing anything outside root that is not approved."""
    if not isinstance(spelling, str) or not spelling.strip():
        raise SchemaError("refused: empty-record-path")
    candidate = Path(spelling)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, ValueError, RuntimeError):
        raise RefusalError("refused: path-unresolvable") from None
    if resolved == root or root in resolved.parents:
        return resolved
    canonical = str(resolved)
    if canonical in approved:
        return resolved
    raise RefusalError("refused: path-outside-scope")


def _check_public_path(token: str) -> None:
    if not isinstance(token, str) or not token.strip():
        raise SchemaError("refused: empty-record-path")
    normalized = token.replace("\\", "/")
    if ".." in Path(normalized).parts:
        raise UnsafeContentError("refused: public-path-escape")
    if PRIVATE_ROW_ID.search(token) or SECRET_MARKERS.search(token):
        raise UnsafeContentError("refused: public-unsafe-path")
    if ABS_PATH.match(token) or EMBEDDED_PATH.search(token):
        raise UnsafeContentError("refused: public-absolute-path")
    if any(marker in normalized.lower() for marker in PRIVATE_PATH_MARKERS):
        raise UnsafeContentError("refused: public-private-path")


def scan_unsafe_public(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            text = str(key)
            if UNSAFE_KEY.match(text):
                raise UnsafeContentError("refused: unsafe-identifier-key")
            if PRIVATE_ROW_ID.search(text) or SECRET_MARKERS.search(text):
                raise UnsafeContentError("refused: unsafe-key")
            scan_unsafe_public(text)
            scan_unsafe_public(item)
    elif isinstance(value, list):
        for item in value:
            scan_unsafe_public(item)
    elif isinstance(value, str):
        if PRIVATE_ROW_ID.search(value):
            raise UnsafeContentError("refused: private-row-id")
        if SECRET_MARKERS.search(value):
            raise UnsafeContentError("refused: secret-marker")
        if ABS_PATH.match(value) or EMBEDDED_PATH.search(value):
            raise UnsafeContentError("refused: absolute-machine-path")
        if any(marker in value.replace("\\", "/").lower() for marker in PRIVATE_PATH_MARKERS):
            raise UnsafeContentError("refused: private-artifact-path")


# ---------------------------------------------------------------------------
# Public metadata projection, rebuilt from primary v1 + v2 semantics.
# ---------------------------------------------------------------------------
def _norm(path: str) -> str:
    return path.replace("\\", "/")


def _redact_path(token: str, root: Path) -> str:
    try:
        return Path(token).resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError, RuntimeError):
        return f"<external-artifact>/{Path(token).name}"


def hide_controlled_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: hide_controlled_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [hide_controlled_paths(item) for item in value]
    if isinstance(value, str):
        normalized = _norm(value).lower()
        if "/private/" in normalized or "data/interim/faces/" in normalized:
            return "<controlled-access-artifact>"
    return value


def sanitize_value(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if (ABS_PATH.match(str(key)) or PRIVATE_ROW_ID.search(str(key))
                    or EMBEDDED_PATH.search(str(key))):
                out["<redacted-key>"] = sanitize_value(item, root)
            else:
                out[key] = sanitize_value(item, root)
        return out
    if isinstance(value, list):
        return [sanitize_value(item, root) for item in value]
    if isinstance(value, str):
        out = PRIVATE_ROW_ID.sub("<redacted-row-id>", value)
        return EMBEDDED_PATH.sub(lambda m: _redact_path(m.group(0), root), out)
    return value


def public_value(value: Any, project_root: Path) -> Any:
    """v1 ``_public_value``: row ids masked, project-absolute paths made relative."""
    if isinstance(value, dict):
        return {key: public_value(item, project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [public_value(item, project_root) for item in value]
    if isinstance(value, str):
        if PRIVATE_ROW_ID.search(value):
            return "<private-row-artifact>"
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(project_root.resolve()).as_posix()
            except (ValueError, OSError, RuntimeError):
                return f"<external-artifact>/{candidate.name}"
    return value


def project_metadata(name: str, value: Any, root: Path) -> Any:
    projected = public_value(value, root) if name in {"command", "parameters"} else value
    return sanitize_value(hide_controlled_paths(projected), root)


def project_public_records(records: Any, root: Path, private_predicate) -> list[Any]:
    if not isinstance(records, list):
        return []
    kept = [r for r in records if not private_predicate(r)]
    return sanitize_value(hide_controlled_paths(public_value(kept, root)), root)


# ---------------------------------------------------------------------------
# Shared declared-record map: aliases used, full preflight, pre/post hash+size.
# ---------------------------------------------------------------------------
class Claim:
    __slots__ = ("size", "sha", "must_match", "private", "spellings")

    def __init__(self, size: int, sha: str, must_match: bool, private: bool):
        self.size = size
        self.sha = sha
        self.must_match = must_match
        self.private = private
        self.spellings: set[str] = set()


def project_integrity(root: Path, records: Any, approved: ApprovedFiles,
                      book: RecordBook | None) -> dict[str, Any]:
    """v1 raw verdict with the v2 hide_controlled_paths + sanitize_value projection."""
    return sanitize_value(hide_controlled_paths(
        record_integrity(root, records, approved, book)), root)


class RecordBook:
    def __init__(self, root: Path, approved: ApprovedFiles):
        self.root = root
        self.approved = approved
        self._claims: dict[str, Claim] = {}
        self._spelling_key: dict[str, str] = {}
        self._pre: dict[str, tuple[str, int]] = {}
        self._post: dict[str, tuple[str, int]] = {}
        self._parsed: dict[str, tuple[str, int]] = {}

    def _resolve(self, spelling: str) -> str:
        canonical = str(resolve_declared(self.root, self.approved, spelling))
        previous = self._spelling_key.get(spelling)
        if previous is not None and previous != canonical:
            raise IntegrityError("refused: alias-retarget")
        self._spelling_key[spelling] = canonical
        return canonical

    def declare(self, spelling: str, size: int, sha: str, *, must_match: bool,
                private: bool) -> str:
        if not _is_int(size) or size < 0 or not isinstance(sha, str) \
                or not SHA256_HEX.fullmatch(sha):
            raise SchemaError("refused: malformed-record")
        canonical = self._resolve(spelling)
        claim = self._claims.get(canonical)
        if claim is None:
            claim = Claim(size, sha.lower(), must_match, private)
            self._claims[canonical] = claim
        else:
            if must_match and claim.must_match and (claim.size != size or claim.sha != sha.lower()):
                raise IntegrityError("refused: alias-conflict")
            if must_match and not claim.must_match:
                claim.size, claim.sha = size, sha.lower()
            claim.must_match = claim.must_match or must_match
            claim.private = claim.private or private
        claim.spellings.add(spelling)
        # Re-resolve every alias now so a retarget anywhere is caught pre-read.
        for other in claim.spellings:
            if self._resolve(other) != canonical:
                raise IntegrityError("refused: alias-retarget")
        return canonical

    def parse_records(self, records: Any, *, public: bool, must_match: bool) -> None:
        if not isinstance(records, list):
            raise SchemaError("refused: records-not-list")
        for raw in records:
            if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
                raise SchemaError("refused: malformed-record")
            declared = raw["path"]
            if public:
                _check_public_path(declared)
            try:
                self.declare(declared, raw["bytes"], raw["sha256"], must_match=must_match,
                             private=_is_private_record(raw))
            except RefusalError:
                # Historical native records may name an approved-later external file that is
                # not on today's list; those are counted as unverifiable, never read or accepted.
                if must_match:
                    raise
                continue

    def canonical(self, spelling: str) -> str | None:
        return self._spelling_key.get(spelling)

    def _fresh(self, canonical: str) -> tuple[str, int] | None:
        self.guard_aliases(canonical)
        target = Path(canonical)
        return hash_file(target) if target.is_file() else None

    def guard_aliases(self, canonical: str | None = None) -> None:
        selected = ((canonical, self._claims[canonical]),) if canonical is not None else self._claims.items()
        for key, claim in selected:
            for spelling in claim.spellings:
                if str(resolve_declared(self.root, self.approved, spelling)) != key:
                    raise IntegrityError("refused: alias-retarget")

    def read_raw(self, spelling: str) -> bytes:
        canonical = self._resolve(spelling)
        if canonical not in self._claims:
            raise IntegrityError("refused: undeclared-read")
        self._claims[canonical].spellings.add(spelling)
        self.guard_aliases(canonical)
        raw = Path(canonical).read_bytes()
        self.guard_aliases(canonical)
        observed = (sha256_bytes(raw), len(raw))
        if canonical in self._parsed and self._parsed[canonical] != observed:
            raise IntegrityError("refused: parsed-byte-drift")
        if canonical in self._pre and self._pre[canonical] != observed:
            raise IntegrityError("refused: parsed-byte-drift")
        self._parsed[canonical] = observed
        return raw

    def read_json(self, spelling: str) -> Any:
        return strict_loads(self.read_raw(spelling))

    def snapshot_pre(self) -> None:
        self.approved.guard()
        self.guard_aliases()
        for canonical in self._claims:
            fresh = self._fresh(canonical)
            if fresh is not None:
                self._pre[canonical] = fresh
        self.guard_aliases()
        if any(self._pre.get(key) != value for key, value in self._parsed.items()):
            raise IntegrityError("refused: parsed-byte-drift")

    def snapshot_post(self) -> None:
        self.approved.guard()
        self.guard_aliases()
        for canonical in self._claims:
            fresh = self._fresh(canonical)
            if fresh is not None:
                self._post[canonical] = fresh
        self.guard_aliases()
        self.approved.guard()

    def present(self, canonical: str) -> bool:
        return canonical in self._pre

    def observed(self, canonical: str) -> tuple[str, int] | None:
        return self._pre.get(canonical)

    def verify_bindings(self) -> int:
        checked = 0
        for canonical, claim in self._claims.items():
            expected = self.approved.expected(canonical)
            if canonical in self.approved:
                actual = self._pre.get(canonical)
                if actual is None:
                    raise IntegrityError("refused: binding-missing")
                if expected[0] is not None and actual[1] != expected[0]:
                    raise IntegrityError("refused: approved-size-mismatch")
                if expected[1] is not None and actual[0] != expected[1]:
                    raise IntegrityError("refused: approved-sha-mismatch")
            if not claim.must_match:
                continue
            actual = self._pre.get(canonical)
            if actual is None:
                raise IntegrityError("refused: binding-missing")
            if actual[0] != claim.sha or actual[1] != claim.size:
                raise IntegrityError("refused: binding-mismatch")
            checked += 1
        return checked

    def verify_stability(self) -> int:
        if self._pre != self._post:
            raise IntegrityError("refused: snapshot-changed")
        for canonical in self._claims:
            if self._pre.get(canonical) is None and canonical in self._post:
                raise IntegrityError("refused: snapshot-appeared")
        return len(self._claims)


# ---------------------------------------------------------------------------
# Independent coverage / semantic recompute (v1 literals, v2 integrity semantics).
# ---------------------------------------------------------------------------
def effective_include() -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        (*FROZEN_V1_INCLUDE, *FROZEN_V6_EXTRA_INCLUDE, *FROZEN_V6_CURRENT_REQUIRED)))


def selected_manifest_paths(root: Path, approved: ApprovedFiles) -> set[Path]:
    metrics = root / "metrics"
    found: set[Path] = set()
    for pattern in effective_include():
        for match in metrics.glob(pattern):
            found.add(resolve_declared(root, approved, str(match)))
    return found


def semantic_gates(root: Path, approved: ApprovedFiles) -> dict[Path, bool]:
    metrics = root / "metrics"
    paths = {d / name
             for d in metrics.glob("lfw_bound_evaluation_*")
             for name in ("lfw_bound_evaluation.json", "lfw_bound_evaluation.manifest.json")}
    paths.update(d / "summary.json" for d in metrics.glob("strong_backbone_fixed8_bn_frozen_*"))
    paths.update(metrics / name / "summary.json" for name in SEMANTIC_CAMPAIGNS)
    gates: dict[Path, bool] = {}
    for path in paths:
        resolved = resolve_declared(root, approved, str(path))
        gates[resolved] = resolved.is_file()
    return gates


def selection_state(root: Path, approved: ApprovedFiles) -> tuple[frozenset[str], dict[str, bool]]:
    selected = frozenset(str(p) for p in selected_manifest_paths(root, approved))
    gates = {str(p): present for p, present in semantic_gates(root, approved).items()}
    return selected, gates


def expected_missing_literals(root: Path, approved: ApprovedFiles) -> list[str]:
    metrics = root / "metrics"
    literals = [p for p in effective_include() if not any(c in p for c in "*?[")]
    found = {p.relative_to(metrics).as_posix()
             for p in selected_manifest_paths(root, approved)}
    return sorted(name for name in literals if name not in found)


def _is_private_record(record: Any) -> bool:
    """Exact frozen v1 predicate (confirmed against primary source, not assumed)."""
    if not isinstance(record, dict):
        return False
    path = str(record.get("path", "")).replace("\\", "/").lower()
    return (path.startswith("data/interim/faces/") or "/private/" in path
            or Path(path).suffix in {".jpg", ".jpeg", ".png", ".npy", ".npz"}
            or bool(PRIVATE_ROW_ID.search(path)))


def record_integrity(root: Path, records: Any, approved: ApprovedFiles,
                     book: RecordBook | None = None) -> dict[str, Any]:
    """Mirror v2 ``IntegrityAudit.check``: bytes AND hash; v1 list shapes.

    Historical native records may reference unapproved external files; those are
    recorded as missing (unverifiable) rather than read or refused.
    """
    if not isinstance(records, list):
        raise SchemaError("refused: records-not-list")
    checked = 0
    private_missing = private_mismatch = 0
    missing: list[str] = []
    mismatch: list[str] = []
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get("path"), str)
                or not record["path"].strip() or not _is_int(record.get("bytes"))
                or record["bytes"] < 0 or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise SchemaError("refused: malformed-native-record")
        private = _is_private_record(record)
        try:
            target = resolve_declared(root, approved, record["path"])
        except RefusalError:
            if private:
                private_missing += 1
            else:
                missing.append(record["path"])
            continue
        actual = None
        if book is not None:
            canonical = book.canonical(record["path"])
            if canonical is not None and book.present(canonical):
                actual = book.observed(canonical)
        if actual is None and book is None:
            actual = hash_file(target) if target.is_file() else None
        if actual is None:
            if private:
                private_missing += 1
            else:
                missing.append(record["path"])
            continue
        checked += 1
        if actual[0].lower() != record["sha256"].lower() or actual[1] != record["bytes"]:
            if private:
                private_mismatch += 1
            else:
                mismatch.append(record["path"])
    valid = not missing and not mismatch and not private_missing and not private_mismatch
    return {"checked": checked, "missing": missing, "checksum_mismatch": mismatch,
            "private_missing_count": private_missing,
            "private_checksum_mismatch_count": private_mismatch, "valid": valid}


def campaign_incomplete(root: Path, book: RecordBook | None = None) -> list[str]:
    metrics = root / "metrics"
    incomplete: list[str] = []
    for evaluation_dir in sorted(metrics.glob("lfw_bound_evaluation_*")):
        if not ((evaluation_dir / "lfw_bound_evaluation.json").is_file()
                and (evaluation_dir / "lfw_bound_evaluation.manifest.json").is_file()):
            incomplete.append(f"{evaluation_dir.name}: completed result or manifest missing")
    for campaign_dir in sorted(metrics.glob("strong_backbone_fixed8_bn_frozen_*")):
        summary = campaign_dir / "summary.json"
        if not summary.is_file():
            incomplete.append(f"{campaign_dir.name}: corrected campaign summary missing")
            continue
        campaign = (book.read_json(str(summary)) if book else strict_loads(_read(root, summary))).get("campaign", {})
        if not campaign.get("complete", False):
            incomplete.append(
                f"{campaign_dir.name}: incomplete ({campaign.get('completed_runs', 0)}/"
                f"{campaign.get('expected_runs', 36)} runs)")
    strong = metrics / "strong_backbone_study" / "summary.json"
    if not strong.is_file():
        incomplete.append("strong-backbone summary missing")
    else:
        campaign = (book.read_json(str(strong)) if book else strict_loads(_read(root, strong))).get("campaign", {})
        if not campaign.get("complete", False):
            incomplete.append(
                f"strong-backbone campaign incomplete "
                f"({campaign.get('completed_runs', 0)}/{campaign.get('expected_runs', 36)} runs)")
    for name, label in (("matched_agegap_arms", "matched-age-gap"),
                        ("matched_agegap_fixed10_last", "fixed-epoch matched-age-gap")):
        summary = metrics / name / "summary.json"
        if summary.is_file():
            payload = book.read_json(str(summary)) if book else strict_loads(_read(root, summary))
            if not payload.get("complete", False):
                incomplete.append(
                    f"{label} campaign incomplete ({payload.get('completed_run_count', 0)}/"
                    f"{payload.get('expected_run_count', 0)} runs)")
    return incomplete


def _read(root: Path, path: Path) -> bytes:
    """Resolve + preflight a path, then read its bytes (no unapproved external read)."""
    target = resolve_declared(root, ApprovedFiles.load(None), str(path))
    return Path(target).read_bytes()


def recompute_invalid(root: Path, approved: ApprovedFiles,
                      book: RecordBook | None = None) -> list[str]:
    invalid: list[str] = []
    for manifest in sorted(selected_manifest_paths(root, approved)):
        payload = book.read_json(str(manifest)) if book else strict_loads(_read(root, manifest))
        if not (record_integrity(root, payload.get("inputs", []), approved, book)["valid"]
                and record_integrity(root, payload.get("outputs", []), approved, book)["valid"]):
            invalid.append(manifest.relative_to(root).as_posix())
    return invalid


# ---------------------------------------------------------------------------
# Output validation.
# ---------------------------------------------------------------------------
_INDEX_REQUIRED = {
    "schema_version", "scope", "entry_count", "missing_expected_manifests",
    "invalid_input_or_output_manifests", "incomplete_experiments", "complete", "entries",
    "index_audit_version", "publication_ready", "physical_snapshot_complete",
    "current_required_manifests", "limitation",
}
_ENTRY_REQUIRED = {
    "manifest", "manifest_sha256", "experiment", "created_at_utc", "command", "parameters",
    "git", "inputs", "outputs", "private_input_count", "private_output_count",
    "input_integrity", "output_integrity",
}


def precheck_index(index_bytes: bytes) -> None:
    index = strict_loads(index_bytes)
    if not isinstance(index, dict) or set(index) != _INDEX_REQUIRED:
        raise SchemaError("refused: index-shape")
    if not (_is_int(index["schema_version"]) and index["schema_version"] == 1):
        raise SchemaError("refused: index-schema-version")
    if not (_is_int(index["index_audit_version"]) and index["index_audit_version"] == 6):
        raise SchemaError("refused: index-audit-version")
    if index["publication_ready"] is not False:
        raise SchemaError("refused: publication-ready-not-false")
    if not _is_int(index["entry_count"]):
        raise SchemaError("refused: index-entry-count")
    if not isinstance(index["entries"], list):
        raise SchemaError("refused: index-entries")


def precheck_manifest(manifest_bytes: bytes) -> None:
    manifest = strict_loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise SchemaError("refused: manifest-not-object")
    if not (_is_int(manifest.get("schema_version")) and manifest["schema_version"] == 1):
        raise SchemaError("refused: manifest-schema-version")
    if not isinstance(manifest.get("inputs"), list) or not isinstance(manifest.get("outputs"), list):
        raise SchemaError("refused: manifest-shape")


def collect_index(index_bytes: bytes, root: Path, approved: ApprovedFiles,
                  book: RecordBook) -> None:
    index = strict_loads(index_bytes)
    for entry in index["entries"]:
        if not isinstance(entry, dict) or set(entry) != _ENTRY_REQUIRED:
            raise SchemaError("refused: entry-shape")
        declared = entry["manifest"]
        _check_public_path(declared)
        target = resolve_declared(root, approved, declared)
        if target.is_file():
            actual = hash_file(target)
            if actual[0] != str(entry["manifest_sha256"]).lower():
                raise IntegrityError("refused: listed-manifest-drift")
            book.declare(declared, actual[1], actual[0], must_match=True, private=False)
        # Sanitized public records are semantic projections, NEVER physical paths.
        native = book.read_json(declared) if target.is_file() else {}
        book.parse_records(native.get("inputs", []), public=False, must_match=False)
        book.parse_records(native.get("outputs", []), public=False, must_match=False)


def collect_manifest(manifest_bytes: bytes, root: Path, approved: ApprovedFiles,
                     book: RecordBook) -> None:
    manifest = strict_loads(manifest_bytes)
    book.parse_records(manifest["inputs"], public=False, must_match=True)
    for raw in manifest["outputs"]:
        if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
            raise SchemaError("refused: malformed-record")
        canonical = str(resolve_declared(root, approved, raw["path"]))
        existing = book.canonical(raw["path"])
        if existing == canonical and book.present(existing):
            continue
        book.declare(raw["path"], raw["bytes"], raw["sha256"], must_match=True, private=False)


def validate_index(index_bytes: bytes, root: Path, approved: ApprovedFiles,
                   book: RecordBook) -> dict[str, Any]:
    index = strict_loads(index_bytes)
    if not (_is_int(index["index_audit_version"]) and index["index_audit_version"] == 6):
        raise SchemaError("refused: index-audit-version")
    if index["publication_ready"] is not False:
        raise SchemaError("refused: publication-ready-not-false")
    if not isinstance(index["limitation"], str) or not index["limitation"].strip():
        raise SchemaError("refused: limitation-missing")
    expected_required = [f"metrics/{name}" for name in FROZEN_V6_CURRENT_REQUIRED]
    if not exact_equal(index["current_required_manifests"], expected_required):
        raise SchemaError("refused: current-required-mismatch")
    for field in ("missing_expected_manifests", "invalid_input_or_output_manifests",
                  "incomplete_experiments"):
        if not isinstance(index[field], list) or not all(isinstance(x, str) for x in index[field]):
            raise SchemaError("refused: index-list-field")
    if not (_is_int(index["entry_count"]) and index["entry_count"] >= 0):
        raise SchemaError("refused: index-entry-count")
    entries = index["entries"]
    if index["entry_count"] != len(entries):
        raise SchemaError("refused: index-entry-count")
    expected_complete = (not index["missing_expected_manifests"]
                         and not index["invalid_input_or_output_manifests"]
                         and not index["incomplete_experiments"])
    if index["complete"] is not expected_complete:
        raise SchemaError("refused: index-complete-flag")
    if index["physical_snapshot_complete"] is not index["complete"]:
        raise SchemaError("refused: index-snapshot-flag")

    if not exact_equal(index["missing_expected_manifests"],
                       expected_missing_literals(root, approved)):
        raise IntegrityError("refused: missing-manifests-drift")
    if not exact_equal(index["incomplete_experiments"], campaign_incomplete(root, book)):
        raise IntegrityError("refused: incomplete-drift")
    if not exact_equal(index["invalid_input_or_output_manifests"],
                       recompute_invalid(root, approved, book)):
        raise IntegrityError("refused: invalid-drift")

    listed: set[str] = set()
    for entry in entries:
        declared = entry["manifest"]
        _check_public_path(declared)
        canonical = str(resolve_declared(root, approved, declared))
        if canonical in listed:
            raise IntegrityError("refused: duplicate-listed-manifest")
        listed.add(canonical)
        if not (_is_int(entry["private_input_count"]) and _is_int(entry["private_output_count"])):
            raise SchemaError("refused: entry-private-count")
        target = Path(canonical)
        if not target.is_file() or not book.present(canonical):
            raise IntegrityError("refused: listed-manifest-missing")
        if book.observed(canonical)[0] != str(entry["manifest_sha256"]).lower():
            raise IntegrityError("refused: listed-manifest-drift")

        native = book.read_json(declared)
        native_inputs = native.get("inputs", [])
        native_outputs = native.get("outputs", [])
        if not exact_equal(entry["input_integrity"],
                           project_integrity(root, native_inputs, approved, book)):
            raise IntegrityError("refused: input-integrity-drift")
        if not exact_equal(entry["output_integrity"],
                           project_integrity(root, native_outputs, approved, book)):
            raise IntegrityError("refused: output-integrity-drift")
        private_inputs = sum(1 for r in native_inputs if _is_private_record(r))
        private_outputs = sum(1 for r in native_outputs if _is_private_record(r))
        if entry["private_input_count"] != private_inputs \
                or entry["private_output_count"] != private_outputs:
            raise IntegrityError("refused: private-count-drift")

        # Full independent v1+v2 metadata/record projection from primary semantics.
        expected_entry = {
            "experiment": project_metadata("experiment", native.get("experiment"), root),
            "created_at_utc": project_metadata("created_at_utc",
                                               native.get("created_at_utc"), root),
            "command": project_metadata("command", native.get("command"), root),
            "parameters": project_metadata("parameters", native.get("parameters"), root),
            "git": project_metadata("git", native.get("git"), root),
            "inputs": project_public_records(native_inputs, root, _is_private_record),
            "outputs": project_public_records(native_outputs, root, _is_private_record),
        }
        for field, want in expected_entry.items():
            if not exact_equal(entry[field], want):
                raise IntegrityError("refused: entry-projection-drift")

    selected = selected_manifest_paths(root, approved)
    if {Path(k) for k in listed} != selected:
        raise IntegrityError("refused: selection-coverage")
    scan_unsafe_public(index)
    return {"entry_count": index["entry_count"], "listed_manifests": len(listed),
            "selected_manifests": len(selected),
            "missing_count": len(index["missing_expected_manifests"]),
            "invalid_count": len(index["invalid_input_or_output_manifests"]),
            "incomplete_count": len(index["incomplete_experiments"]),
            "complete": index["complete"]}


def validate_manifest(manifest_bytes: bytes, root: Path, approved: ApprovedFiles,
                      index_path: Path, book: RecordBook) -> dict[str, Any]:
    manifest = strict_loads(manifest_bytes)
    for field in ("schema_version", "experiment", "created_at_utc", "command", "parameters",
                  "metrics", "inputs", "outputs", "runtime", "git"):
        if field not in manifest:
            raise SchemaError("refused: manifest-missing-keys")
    if not (_is_int(manifest["schema_version"]) and manifest["schema_version"] == 1):
        raise SchemaError("refused: manifest-schema-version")
    if manifest["experiment"] != EXPERIMENT_NAME:
        raise SchemaError("refused: manifest-experiment")
    if not isinstance(manifest["parameters"], dict) or not isinstance(manifest["metrics"], dict) \
            or not isinstance(manifest["command"], list):
        raise SchemaError("refused: manifest-shape")
    if manifest["parameters"].get("canonical_index_updated") is not False:
        raise SchemaError("refused: canonical-index-updated")
    if not exact_equal(manifest["parameters"].get("current_required"),
                       list(FROZEN_V6_CURRENT_REQUIRED)):
        raise SchemaError("refused: manifest-current-required")
    if manifest["metrics"].get("execution_complete") is not True:
        raise SchemaError("refused: manifest-execution-complete")
    if manifest["metrics"].get("publication_ready") is not False:
        raise SchemaError("refused: manifest-publication-ready")

    bound = {str(resolve_declared(root, approved, r["path"])) for r in manifest["inputs"]}
    required = {str(p) for p in selected_manifest_paths(root, approved)}
    required |= {str(p) for p, present in semantic_gates(root, approved).items() if present}
    sources = {str(resolve_declared(root, approved, rel)) for rel in CANONICAL_SOURCE_RELPATHS}
    if not sources <= bound:
        raise IntegrityError("refused: source-binding")
    if not required <= bound:
        raise IntegrityError("refused: selection-binding")
    expected_bound = required | sources
    for selected in selected_manifest_paths(root, approved):
        native = book.read_json(str(selected))
        for item in native.get("inputs", []) + native.get("outputs", []):
            try:
                target = resolve_declared(root, approved, item["path"])
            except RefusalError:
                # An unapproved historical input is unverifiable, not permission
                # to open it or to demand it in a physical completion proof.
                continue
            if target.is_file():
                expected_bound.add(str(target))
    if bound != expected_bound or len(bound) != len(manifest["inputs"]):
        raise IntegrityError("refused: physical-input-closure")
    if (len(manifest["outputs"]) != 1 or
            str(resolve_declared(root, approved, manifest["outputs"][0]["path"])) != str(index_path)):
        raise IntegrityError("refused: output-role-closure")
    if str(index_path) not in {str(resolve_declared(root, approved, r["path"]))
                               for r in manifest["outputs"]}:
        raise IntegrityError("refused: index-output-binding")

    index_json = book.read_json(str(index_path))
    canonical = str(index_path)
    observed = book.observed(canonical)
    if observed is None or observed != book._parsed.get(canonical):
        raise IntegrityError("refused: index-byte-drift")

    if not exact_equal(manifest["metrics"].get("observed_physical_inputs"),
                       len(manifest["inputs"])):
        raise SchemaError("refused: manifest-observed-inputs")
    pairs = {
        "entry_count": (manifest["metrics"].get("entry_count"), index_json["entry_count"]),
        "missing_count": (manifest["metrics"].get("missing_count"),
                          len(index_json["missing_expected_manifests"])),
        "invalid_count": (manifest["metrics"].get("invalid_count"),
                          len(index_json["invalid_input_or_output_manifests"])),
        "incomplete_count": (manifest["metrics"].get("incomplete_count"),
                             len(index_json["incomplete_experiments"])),
        "artifact_index_complete": (manifest["metrics"].get("artifact_index_complete"),
                                    index_json["complete"]),
        "physical_snapshot_complete": (manifest["metrics"].get("physical_snapshot_complete"),
                                       index_json["complete"]),
    }
    for got, want in pairs.values():
        if not exact_equal(got, want):
            raise SchemaError("refused: manifest-metric-drift")
    return {"binding_inputs": len(manifest["inputs"]), "bound_sources": len(sources)}


def verify(root: Path, out: Path, allowed_files: Path | None = None) -> dict[str, Any]:
    """Independently validate a finished index-v6 output directory. Counts only."""
    root = _canonical_dir(root)
    out = _canonical_dir(out)
    if allowed_files is not None and not Path(allowed_files).resolve().is_relative_to(root):
        raise RefusalError("refused: allowlist-outside-root")
    approved = ApprovedFiles.load(allowed_files)
    if root not in out.parents and out != root:
        raise RefusalError("refused: out-escape")
    index_path = resolve_declared(root, approved, str(out / INDEX_FILENAME))
    manifest_path = resolve_declared(root, approved, str(out / MANIFEST_FILENAME))

    pre_state = selection_state(root, approved)
    index_bytes = index_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    precheck_index(index_bytes)
    precheck_manifest(manifest_bytes)

    book = RecordBook(root, approved)
    book.declare(str(index_path), len(index_bytes), sha256_bytes(index_bytes),
                 must_match=True, private=False)
    book.declare(str(manifest_path), len(manifest_bytes), sha256_bytes(manifest_bytes),
                 must_match=True, private=False)
    # Canonical producing sources and all existing semantic gates are bound.
    for rel in CANONICAL_SOURCE_RELPATHS:
        source = resolve_declared(root, approved, rel)
        if source.is_file():
            sha, size = hash_file(source)
            book.declare(rel, size, sha, must_match=True, private=False)
    for gate, present in semantic_gates(root, approved).items():
        if present:
            sha, size = hash_file(gate)
            book.declare(str(gate), size, sha, must_match=True, private=False)

    collect_index(index_bytes, root, approved, book)
    collect_manifest(manifest_bytes, root, approved, book)

    book.snapshot_pre()
    for relative, fields in (
            ("scripts/build_publication_artifact_index.py", {"INCLUDE": FROZEN_V1_INCLUDE}),
            ("scripts/build_publication_artifact_index_v6.py", {
                "EXTRA_INCLUDE": FROZEN_V6_EXTRA_INCLUDE,
                "CURRENT_REQUIRED": FROZEN_V6_CURRENT_REQUIRED})):
        tree = ast.parse(book.read_raw(relative))
        assignments = {target.id: ast.literal_eval(node.value)
                       for node in tree.body if isinstance(node, ast.Assign)
                       for target in node.targets
                       if isinstance(target, ast.Name) and target.id in fields}
        if not exact_equal(assignments, fields):
            raise IntegrityError("refused: frozen-selection-contract-drift")
    index_summary = validate_index(index_bytes, root, approved, book)
    manifest_summary = validate_manifest(manifest_bytes, root, approved, index_path, book)
    book.snapshot_post()

    bound = book.verify_bindings()
    tracked = book.verify_stability()
    if selection_state(root, approved) != pre_state:
        raise IntegrityError("refused: selection-drift")

    return {
        "accepted_physical_snapshot": True,
        "publication_ready": False,
        "index_audit_version": 6,
        "entry_count": index_summary["entry_count"],
        "listed_manifests": index_summary["listed_manifests"],
        "selected_manifests": index_summary["selected_manifests"],
        "missing_count": index_summary["missing_count"],
        "invalid_count": index_summary["invalid_count"],
        "incomplete_count": index_summary["incomplete_count"],
        "complete": index_summary["complete"],
        "binding_verified": bound,
        "tracked_targets": tracked,
        "binding_inputs": manifest_summary["binding_inputs"],
        "bound_sources": manifest_summary["bound_sources"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allowed-files", type=Path, default=None)
    try:
        args = parser.parse_args()
        result = verify(args.root, args.out, args.allowed_files)
    except CheckError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
    except Exception:  # noqa: BLE001 - never expose native paths or identifiers
        print("refused: internal", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
