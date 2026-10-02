#!/usr/bin/env python3
"""Whitelist-only builder for a publication-evidence ZIP.

Exports ONLY explicit reviewed aggregate result JSON, checksum-verified against the
manifests declaring them, plus SANITIZED manifest projections (hidden input/output
records are counted, never listed) and a certificate of verdict counts and
original-vs-exported checksums. Membership is built from an explicit map, never by
scanning the output directory. This export is partial evidence, not publication clearance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPORT_POLICY = "tbiom-reviewed-aggregates-v1"
SCHEMA_VERSION = 1
ZIP_NAME = "public_evidence_bundle.zip"
ALLOWED_MEMBERS = "results/*.json + manifests/*.sanitized.json + README.txt + CERTIFICATE.json"
FIXED_MEMBERS = ("README.txt", "CERTIFICATE.json")
PRIVATE_ROW_ID = re.compile(r"(?:vk_|reddit_)?-?\d{5,}_\d{5,}", re.IGNORECASE)
SECRET_MARKERS = re.compile(
    r"(vk1\.a\.|bearer\s+[a-z0-9._-]{20,}|client_secret\s*[=:]|api[_-]?key\s*[=:]|"
    r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|password\s*[=:])", re.IGNORECASE)
ABS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/(?:home|users|mnt|media|root|tmp)/)", re.IGNORECASE)
# Non-anchored: catches absolute paths embedded inside command/tool strings.
EMBEDDED_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s\"'<>|()\[\],;]+|\\\\[^\s\"'<>|()\[\],;]+"
    r"|(?<![\w/])/(?:home|users|mnt|media|root|tmp)/[^\s\"'<>|()\[\],;]+)", re.IGNORECASE)
PRIVATE_PATH_MARKERS = ("/private/", "data/interim/faces/")
SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")
UNSAFE_KEY = re.compile(
    r"^(?:(?:image|crop|face|row|subject|person|pair)_?(?:sha1|sha256|md5|hash|ids?|path)|"
    r"(?:sha1|sha256|md5)_of_(?:image|crop|face|row|subject|pair)|"
    r"embeddings?|captions?|raw_responses?|(?:photo|crop|face)_url)$", re.IGNORECASE)
GATE_KEYS = {"training_identity_independence", "identity_independence",
             "checkpoint_training_provenance", "checkpoint_training_manifest_status",
             "training_manifest_status", "training_benchmark_identity_independence"}
# Explicit whitelist. Manifest is sibling "<name>.manifest.json"; member names are derived
# only from this tuple plus the two fixed files. No recursive globbing anywhere.
EXPORTS: tuple[str, ...] = (
    "metrics/fgnet_endpoint_multiseed.json",
    "metrics/fgnet_endpoint_subject_stats.json",
    "metrics/fgnet_endpoint_subject_stats_s1.json",
    "metrics/fgnet_endpoint_subject_stats_s2.json",
    "metrics/internal_endpoint_age_matched.json",
    "metrics/comparator_fgnet_endpoint_age_matched.json",
    "metrics/fgnet_retrieval_20261002/fgnet_retrieval_study.json",
    "metrics/fgnet_error_breakdown/fgnet_error_breakdown.json")
README_TEXT = (
    f"T-BIOM public evidence bundle (policy {EXPORT_POLICY}).\n"
    "Members: results/*.json reviewed aggregate results; manifests/*.sanitized.json "
    "SANITIZED projections (hidden records counted only); CERTIFICATE.json verdict counts, "
    "original-vs-exported checksums, unresolved gates. No faces/captions/embeddings, private "
    "rows, machine paths or per-image/checkpoint private hashes. Uncertainty and unresolved "
    "training-independence / not-complete flags preserved. Standalone exported evidence, NOT "
    "full reproduction or a publication-readiness claim.\n"
    "Build locally: python scripts/export_public_evidence.py --root . --out <new-output-dir>.\n"
    "Building requires all declared original inputs, including controlled-access artifacts. "
    "These are verified locally, not redistributed. Public projections list ORIGINAL artifact "
    "checksums; CERTIFICATE.json separately binds the EXPORTED result and projection bytes. "
    "This bundle covers eight selected results, not every manuscript table. Obtain any private "
    "inputs only through an approved controlled-access procedure; this export grants no access.\n")


class ExportError(RuntimeError):
    """A refused export."""


class UnsafeContentError(ExportError):
    """Private row, secret marker, machine path or per-image identifier key."""


class ChecksumError(ExportError):
    """Originals missing/mismatched vs manifest, or malformed manifest record."""


class ArchiveError(ExportError):
    """Archive membership, duplicates or overwrite violates the allowlist."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm(path: str) -> str:
    return path.replace("\\", "/")


def is_private_record(path: str) -> bool:
    p = _norm(path).lower()
    return (p.startswith("data/interim/faces/") or "/private/" in p or p.startswith("private/")
            or bool(PRIVATE_ROW_ID.search(p)))


def is_listable_record(path: str) -> bool:
    """Only public aggregate result JSON and code files may keep path+digest in a projection."""
    p = _norm(path)
    if is_private_record(p) or ".." in Path(p).parts:
        return False
    if p.startswith("scripts/") and p.endswith(".py"):
        return True
    return p in EXPORTS


def _redact_path(token: str, root: Path) -> str:
    try:
        return Path(token).resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return f"<external-artifact>/{Path(token).name}"


def sanitize_value(value: Any, root: Path) -> tuple[Any, int]:
    """Redact row IDs and machine/private paths in artifact-path strings; return (value, hits).

    Only path-like strings are altered; scientific scalar fields are left untouched.
    """
    if isinstance(value, dict):
        out, hits = {}, 0
        for key, item in value.items():
            if ABS_PATH.match(str(key)) or PRIVATE_ROW_ID.search(str(key)) or EMBEDDED_PATH.search(str(key)):
                out["<redacted-key>"], hit = sanitize_value(item, root)
                hits += hit + 1
            else:
                out[key], hit = sanitize_value(item, root)
                hits += hit
        return out, hits
    if isinstance(value, list):
        out, hits = [], 0
        for item in value:
            clean, hit = sanitize_value(item, root)
            out.append(clean)
            hits += hit
        return out, hits
    if isinstance(value, str):
        hits = 0
        out = value
        if PRIVATE_ROW_ID.search(out):
            out, n = PRIVATE_ROW_ID.subn("<redacted-row-id>", out)
            hits += n
        out, n = EMBEDDED_PATH.subn(lambda m: _redact_path(m.group(0), root), out)
        hits += n
        return out, hits
    return value, 0


def reject_identifiers(value: Any, where: str = "$") -> None:
    """Fail fast on private row identifiers, secret markers or per-image identifier keys.

    Artifact-path strings are handled by sanitize_value; identifiers here are unsafe by
    nature and are refused rather than silently rewritten.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            text = str(key)
            if UNSAFE_KEY.match(text):
                raise UnsafeContentError(f"unsafe per-image identifier key at {where}: {text}")
            if PRIVATE_ROW_ID.search(text) or SECRET_MARKERS.search(text):
                raise UnsafeContentError(f"unsafe key at {where}: {text}")
            reject_identifiers(item, f"{where}.{text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_identifiers(item, f"{where}[{index}]")
    elif isinstance(value, str):
        if PRIVATE_ROW_ID.search(value):
            raise UnsafeContentError(f"private row identifier at {where}")
        if SECRET_MARKERS.search(value):
            raise UnsafeContentError(f"secret marker at {where}")


def scan_unsafe(value: Any, where: str = "$") -> None:
    """Post-sanitation rejection of residual private rows, secrets, absolute/private paths
    or per-image identifier keys. Nothing unsafe may survive into a projection."""
    if isinstance(value, dict):
        for key, item in value.items():
            text = str(key)
            if UNSAFE_KEY.match(text):
                raise UnsafeContentError(f"unsafe per-image identifier key at {where}: {text}")
            if PRIVATE_ROW_ID.search(text) or SECRET_MARKERS.search(text):
                raise UnsafeContentError(f"unsafe key at {where}: {text}")
            scan_unsafe(text, f"{where}[key]")
            scan_unsafe(item, f"{where}.{text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            scan_unsafe(item, f"{where}[{index}]")
    elif isinstance(value, str):
        if PRIVATE_ROW_ID.search(value):
            raise UnsafeContentError(f"private row identifier at {where}")
        if SECRET_MARKERS.search(value):
            raise UnsafeContentError(f"secret marker at {where}")
        if ABS_PATH.match(value) or EMBEDDED_PATH.search(value):
            raise UnsafeContentError(f"absolute machine path at {where}")
        if any(marker in _norm(value).lower() for marker in PRIVATE_PATH_MARKERS):
            raise UnsafeContentError(f"private artifact path at {where}")


def _verify_records(records: list[Any], root: Path) -> dict[str, Any]:
    """Validate record shape, then verify existence and checksum; return counts only."""
    verified = missing = mismatch = 0
    for record in records:
        if not isinstance(record, dict):
            raise ChecksumError("malformed manifest record: not an object")
        declared, sha = record.get("path"), record.get("sha256")
        if not isinstance(declared, str) or not declared.strip():
            raise ChecksumError("malformed manifest record: empty path")
        if not isinstance(sha, str) or not SHA256_HEX.fullmatch(sha):
            raise ChecksumError("malformed manifest record: sha256 is not 64 hex digits")
        path = Path(declared)
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file():
            missing += 1
        elif sha256_file(resolved).lower() != sha.lower():
            mismatch += 1
        else:
            verified += 1
    return {"checked": len(records), "verified": verified, "missing": missing,
            "checksum_mismatch": mismatch, "valid": missing == 0 and mismatch == 0}


def verify_manifest(result: str, root: Path) -> dict[str, Any]:
    """Verify a result and every direct input/output. Aborts on any missing or mismatch."""
    result_path = root / result
    manifest_rel = result[: -len(".json")] + ".manifest.json"
    manifest_path = root / manifest_rel
    if not result_path.is_file():
        raise ChecksumError(f"missing reviewed result: {result}")
    if not manifest_path.is_file():
        raise ChecksumError(f"missing manifest: {manifest_rel}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or any(
        not isinstance(manifest.get(key), list) for key in ("inputs", "outputs")
    ):
        raise ChecksumError("manifest must declare input and output record lists")
    inputs = _verify_records(list(manifest.get("inputs", [])), root)
    outputs = _verify_records(list(manifest.get("outputs", [])), root)
    if not inputs["valid"] or not outputs["valid"]:
        raise ChecksumError(
            f"verification failed vs {manifest_rel}: inputs missing={inputs['missing']} "
            f"mismatch={inputs['checksum_mismatch']}; outputs missing={outputs['missing']} "
            f"mismatch={outputs['checksum_mismatch']}")
    declared = [r for r in manifest.get("outputs", [])
                if isinstance(r, dict) and _norm(str(r.get("path", ""))) == result]
    if not declared or str(declared[0].get("sha256", "")).lower() != sha256_file(result_path).lower():
        raise ChecksumError(f"exported result does not match declared output: {result}")
    return {"manifest": manifest_rel, "manifest_sha256": sha256_file(manifest_path),
            "inputs": inputs, "outputs": outputs,
            "declared_output_sha256": sha256_file(result_path)}


def _sanitized_manifest(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    def partition(records: list[Any]) -> tuple[list[Any], int, int]:
        public, hidden = [], 0
        for record in records:
            if isinstance(record, dict) and is_listable_record(str(record.get("path", ""))):
                public.append({"path": record.get("path"), "bytes": record.get("bytes"),
                               "sha256": record.get("sha256")})
            else:
                hidden += 1
        clean, hits = sanitize_value(public, root)
        return clean, hidden, hits
    public_inputs, hidden_inputs, hits_in = partition(manifest.get("inputs", []))
    public_outputs, hidden_outputs, hits_out = partition(manifest.get("outputs", []))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "note": ("sanitized projection; hidden input/output records are counted only, with no "
                 "private paths or checkpoint/biometric digests; only public aggregate JSON and "
                 "code files keep path+digest"),
        "experiment": manifest.get("experiment"), "created_at_utc": manifest.get("created_at_utc"),
        "command": sanitize_value(manifest.get("command"), root)[0],
        "parameters": sanitize_value(manifest.get("parameters"), root)[0],
        "public_inputs": public_inputs, "public_outputs": public_outputs,
        "hidden_input_records": hidden_inputs, "hidden_output_records": hidden_outputs,
        "redactions": hits_in + hits_out}
    scan_unsafe(payload, "sanitized-manifest")
    return payload


def _gates(value: Any, found: dict[str, Any]) -> dict[str, Any]:
    """Collect gate field values verbatim (never scrubbed as if they were paths)."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in GATE_KEYS and isinstance(item, (str, bool)):
                found.setdefault(key, item)
            _gates(item, found)
    elif isinstance(value, list):
        for item in value:
            _gates(item, found)
    return found


def _build_members(root: Path) -> tuple[dict[str, bytes], list[dict[str, Any]], dict[str, Any]]:
    """Build the explicit member map from EXPORTS only; never scan the output directory."""
    members: dict[str, bytes] = {}
    entries: list[dict[str, Any]] = []
    gates: dict[str, Any] = {}
    for result in EXPORTS:
        member = Path(result).name
        verification = verify_manifest(result, root)
        original = root / result
        payload = json.loads(original.read_text(encoding="utf-8"))
        reject_identifiers(payload, member)
        clean, redactions = sanitize_value(payload, root)
        scan_unsafe(clean, member)
        _gates(payload, gates)
        result_bytes = ((json.dumps(clean, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
                        if redactions else original.read_bytes())
        manifest = json.loads((root / verification["manifest"]).read_text(encoding="utf-8"))
        sanitized = _sanitized_manifest(manifest, root)
        manifest_bytes = (json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        result_member, manifest_member = f"results/{member}", f"manifests/{member}.sanitized.json"
        if result_member in members or manifest_member in members:
            raise ArchiveError(f"duplicate derived member for {result}")
        members[result_member], members[manifest_member] = result_bytes, manifest_bytes
        entries.append({
            "member": result_member, "manifest_member": manifest_member,
            "original_relpath": result, "original_sha256": sha256_file(original),
            "original_bytes": original.stat().st_size,
            "exported_sha256": sha256_bytes(result_bytes), "exported_bytes": len(result_bytes),
            "sanitized": bool(redactions), "redactions": redactions,
            "manifest_exported_sha256": sha256_bytes(manifest_bytes),
            "verification": verification})
    return members, entries, gates


def build_export(root: Path, out_dir: Path, overwrite: bool = False) -> Path:
    archive = out_dir / ZIP_NAME
    if archive.exists() and not overwrite:
        raise ArchiveError(f"refusing to overwrite existing archive (root decides): {archive}")
    out_dir.mkdir(parents=True, exist_ok=True)
    members, entries, gates = _build_members(root)
    certificate = {
        "schema_version": SCHEMA_VERSION, "export_policy": EXPORT_POLICY,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "exporter": {"path": "scripts/export_public_evidence.py",
                     "sha256": sha256_file(Path(__file__))},
        "scope": ("standalone reviewed aggregate evidence; NOT full experiment reproducibility "
                  "(private inputs and legacy training provenance excluded)"),
        "file_whitelist": list(EXPORTS), "archive_members_allowlist": ALLOWED_MEMBERS,
        "entries": entries,
        "unresolved": {"gates": gates, "asserts_training_independence": False,
                       "asserts_publication_ready": False, "asserts_experiment_complete": False}}
    cert_bytes = (json.dumps(certificate, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    readme_bytes = README_TEXT.encode("utf-8")
    members["CERTIFICATE.json"], members["README.txt"] = cert_bytes, readme_bytes
    expected = {"CERTIFICATE.json": sha256_bytes(cert_bytes), "README.txt": sha256_bytes(readme_bytes)}
    for entry in entries:
        expected[entry["member"]] = entry["exported_sha256"]
        expected[entry["manifest_member"]] = entry["manifest_exported_sha256"]
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    verify_archive(archive, expected)
    return archive


def verify_archive(archive: Path, expected: dict[str, str]) -> set[str]:
    """Verify exact membership, no duplicates/traversal, and every archived hash."""
    if not isinstance(expected, dict) or not expected:
        raise ArchiveError("exact expected member->sha256 map is required")
    allowed = set(FIXED_MEMBERS)
    for result in EXPORTS:
        allowed.add(f"results/{Path(result).name}")
        allowed.add(f"manifests/{Path(result).name}.sanitized.json")
    if set(expected) != allowed:
        raise ArchiveError("expected membership does not match reviewed EXPORTS")
    with zipfile.ZipFile(archive) as zf:
        names = [info.filename for info in zf.infolist()]
        if len(names) != len(set(names)):
            raise ArchiveError("duplicate archive members")
        for name in names:
            if "\\" in name:
                raise ArchiveError(f"backslash in member name: {name}")
            pure = Path(name)
            if pure.is_absolute() or name.startswith("/") or ".." in pure.parts:
                raise ArchiveError(f"unsafe archive member: {name}")
            if not (name in FIXED_MEMBERS
                    or re.fullmatch(r"results/[A-Za-z0-9_.-]+\.json", name)
                    or re.fullmatch(r"manifests/[A-Za-z0-9_.-]+\.sanitized\.json", name)):
                raise ArchiveError(f"member outside allowlist: {name}")
        if set(names) != set(expected):
            raise ArchiveError(f"archive membership mismatch: {sorted(set(names) ^ set(expected))}")
        for name, wanted in expected.items():
            if sha256_bytes(zf.read(name)) != wanted:
                raise ArchiveError(f"archived hash mismatch for {name}")
    return set(names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing bundle ZIP (root decides)")
    args = parser.parse_args(argv)
    print(f"wrote {build_export(args.root.resolve(), args.out.resolve(), args.overwrite)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
