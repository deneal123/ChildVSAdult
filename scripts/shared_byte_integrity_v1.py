"""Standalone stdlib shared-byte integrity phase.

Two phases for one canonical record set:

* :func:`audited_records` canonicalizes every declared path *before* any read,
  rejects aliases that disagree on ``(bytes, sha256)``, streams each distinct
  physical target exactly once per phase, and returns a stable canonical map.
* :func:`rehash_phase` takes that map and fresh-streams every target with no
  persistent cache and no size/mtime shortcut, rejecting mutation and escape.

Failures are privacy-safe: public subjects/details never carry a record path, a
canonical key or a SHA-256. Only stage, an indexed label, reason codes and byte
counts are exposed. :func:`parse_manifest_strict` parses and hashes the SAME
bytes, so a digest from a separate read is never trusted.

This component retains declared-alias bindings and preflights all targets before
any content read. It is not wired into a queue and has no
dependency beyond the standard library.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

__all__ = [
    "ByteIntegrityError",
    "audited_records",
    "rehash_phase",
    "parse_manifest_strict",
]

_BLOCK = 1 << 20
_HEX = frozenset("0123456789abcdef")
_TYPES = {"int": int, "bool": bool, "str": str}
_RECORD_KEYS = {"path", "bytes", "sha256"}


class ByteIntegrityError(Exception):
    """One rejected byte check, worded so private material can never leak."""

    def __init__(self, stage: str, subject: str, detail: str) -> None:
        super().__init__(f"{stage}: {subject}: {detail}")
        self.stage = stage
        self.subject = subject
        self.detail = detail


def _fail(stage: str, subject: str, detail: str) -> None:
    raise ByteIntegrityError(stage, subject, detail)


def _typed(value: object, kind: str, subject: str, stage: str):
    """Fail-closed gate: bool never passes as int, NaN/Inf never as number."""
    if kind in _TYPES:
        if type(value) is not _TYPES[kind]:
            _fail(stage, subject, f"reason=type expected={kind} got={type(value).__name__}")
    elif type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _fail(stage, subject, f"reason=type expected={kind} got={type(value).__name__}")
    return value


def _stream_file(path: Path) -> tuple[int, str]:
    """Stream a file through SHA-256; no size/mtime shortcut is ever accepted."""
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(_BLOCK), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _canonical(path: Path) -> Path | None:
    """Resolve without ever surfacing an OSError carrying a raw path."""
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


def _root_of(root: object) -> Path:
    resolved = _canonical(root)
    try:
        valid = resolved is not None and resolved.is_dir()
    except (OSError, RuntimeError, ValueError):
        valid = False
    if not valid:
        _fail("phase", "root", "reason=unresolvable-or-not-a-directory")
    return resolved


def _shape(text: str, subject: str, stage: str) -> str:
    """Reject absolute/escaping spellings before any resolution or read."""
    parts = tuple(part for part in text.replace("\\", "/").split("/") if part not in ("", "."))
    if text.startswith("/") or text.startswith("\\") or (len(text) > 1 and text[1] == ":") or ".." in parts:
        _fail(stage, subject, "reason=path-escapes-root")
    if not parts:
        _fail(stage, subject, "reason=empty-record-path")
    return "/".join(parts)


def _declared_target(relative: str, base: Path, subject: str, stage: str) -> tuple[str, Path]:
    """Canonicalize a declared path inside the root BEFORE it is ever read."""
    text = _shape(relative, subject, stage)
    target = _canonical(base / text)
    if target is None or not target.is_relative_to(base):
        _fail(stage, subject, "reason=path-escapes-root")
    return os.path.normcase(target.relative_to(base).as_posix()).replace("\\", "/"), target


def _malformed(size: object, sha: object, subject: str, stage: str) -> None:
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        _fail(stage, subject, f"reason=malformed-record bytes={size}")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in _HEX for c in sha):
        _fail(stage, subject, "reason=malformed-record sha256")


@dataclass(frozen=True, repr=False, eq=False)
class RecordSnapshot(Mapping):
    """Read-only canonical map; alias bindings cannot disappear on serialization."""

    _records: Mapping
    _aliases: tuple

    def __getitem__(self, key):
        size, sha = self._records[key]
        return {"bytes": size, "sha256": sha}

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)


def _check_aliases(snapshot, base, stage):
    for index, (declared, key) in enumerate(snapshot._aliases):
        current, _target = _declared_target(declared, base, f"alias[{index}]", stage)
        if current != key:
            _fail(stage, f"alias[{index}]", "reason=declared-target-changed")


def audited_records(records: object, root: object) -> RecordSnapshot:
    """Preflight every declaration, then stream each distinct contained target once."""
    stage = "audited-records"
    base = _root_of(root)
    if type(records) is not list:
        _fail(stage, "records", "reason=type expected=list")
    expected, targets, aliases = {}, {}, []
    for index, raw in enumerate(records):
        subject = f"record[{index}]"
        if type(raw) is not dict or set(raw) != _RECORD_KEYS:
            _fail(stage, subject, "reason=record-shape")
        relative = _typed(raw["path"], "str", subject, stage)
        size = _typed(raw["bytes"], "int", subject, stage)
        sha = _typed(raw["sha256"], "str", subject, stage)
        _malformed(size, sha, subject, stage)
        key, target = _declared_target(relative, base, subject, stage)
        declared = (size, sha)
        if key in expected and expected[key] != declared:
            _fail("record-alias-conflict", subject, "reason=aliases-disagree-on-bytes-or-hash")
        expected[key] = declared
        targets[key] = target
        aliases.append((relative, key))
    snapshot = RecordSnapshot(MappingProxyType(dict(sorted(expected.items()))), tuple(aliases))
    _check_aliases(snapshot, base, "target-escape")
    for index, key in enumerate(snapshot):
        subject = f"record[{index}]"
        size, sha = snapshot._records[key]
        # Re-resolve before each open, rather than trusting the preflight Path object.
        current, target = _declared_target(key, base, subject, "target-escape")
        if current != key:
            _fail("target-escape", subject, "reason=canonical-target-changed")
        try:
            actual = _stream_file(target)
        except (OSError, RuntimeError, ValueError):
            _fail("record-missing", subject, f"reason=unreadable declared_bytes={size}")
        if actual != (size, sha):
            _fail(stage, subject,
                  f"reason=byte-drift declared_bytes={size} actual_bytes={actual[0]} "
                  f"same_size={actual[0] == size}")
    _check_aliases(snapshot, base, "target-escape")
    return snapshot


def rehash_phase(expected: object, root: object) -> RecordSnapshot:
    """Fresh-stream all targets, with alias containment checked before and after."""
    stage = "rehash-phase"
    base = _root_of(root)
    if not isinstance(expected, RecordSnapshot):
        _fail("target-escape", "expected", "reason=alias-bound-snapshot-required")
    # Validate the complete map before any file content is opened.
    for index, key in enumerate(expected):
        subject = f"target[{index}]"
        _shape(_typed(key, "str", subject, stage), subject, "target-escape")
        size, sha = expected._records[key]
        _typed(size, "int", subject, stage)
        _typed(sha, "str", subject, stage)
        _malformed(size, sha, subject, stage)
    _check_aliases(expected, base, "target-escape")
    for index, key in enumerate(expected):
        subject = f"target[{index}]"
        size, sha = expected._records[key]
        current, target = _declared_target(key, base, subject, "target-escape")
        if current != key:
            _fail("target-escape", subject, "reason=canonical-target-changed")
        try:
            actual = _stream_file(target)
        except (OSError, RuntimeError, ValueError):
            _fail("target-missing", subject, f"reason=unreadable declared_bytes={size}")
        if actual != (size, sha):
            _fail(stage, subject,
                  f"reason=byte-drift declared_bytes={size} actual_bytes={actual[0]} "
                  f"same_size={actual[0] == size}")
    _check_aliases(expected, base, "target-escape")
    return expected


def parse_manifest_strict(raw: object) -> tuple[object, int, str]:
    """Parse strict JSON and hash the SAME bytes.

    Duplicate object keys and non-finite numbers are rejected. Returns
    ``(data, size, sha256)`` where the digest is computed from exactly the parsed
    bytes, never from a separate read.
    """
    stage = "manifest"
    if type(raw) is not bytes:
        _fail(stage, "bytes", "reason=type expected=bytes")

    def no_duplicates(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                _fail(stage, "bytes", "reason=duplicate-json-key")
            data[key] = value
        return data

    def reject_constant(name):
        _fail(stage, "bytes", "reason=non-finite-number")

    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            _fail(stage, "bytes", "reason=non-finite-number")
        return value

    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates,
                          parse_constant=reject_constant, parse_float=finite_float)
    except ByteIntegrityError:
        raise
    except (UnicodeDecodeError, ValueError):
        _fail(stage, "bytes", "reason=malformed-json")
    return data, len(raw), hashlib.sha256(raw).hexdigest()
