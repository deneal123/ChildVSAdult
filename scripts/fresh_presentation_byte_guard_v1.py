"""Preflight ALL declarations before any target-byte read; exact outside policy.

Callers must supply an already-bound inventory policy. This helper does not
approve an inventory, scientific result, native schema or presentation by itself.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


class ByteGuardError(ValueError):
    """Privacy-safe original-byte preflight or phase refusal."""


def shape(record):
    if (
        type(record) is not dict
        or set(record) != {"path", "bytes", "sha256"}
        or type(record["path"]) is not str
        or not record["path"]
        or type(record["bytes"]) is not int
        or record["bytes"] < 0
        or type(record["sha256"]) is not str
        or len(record["sha256"]) != 64
        or any(v not in "0123456789abcdef" for v in record["sha256"])
    ):
        raise ByteGuardError("invalid byte record")


def resolve(spelling, root):
    try:
        value = Path(spelling)
        return (value if value.is_absolute() else root / value).resolve(strict=True)
    except (OSError, ValueError, RuntimeError):
        raise ByteGuardError("unresolvable declaration") from None


def key(path):
    return os.path.normcase(str(path))


@dataclass(frozen=True)
class Snapshot:
    root: Path
    aliases: object
    physical: object


def preflight(records, root, allowed):
    """Resolve/authorize the WHOLE set without opening target contents."""
    if type(records) is not list or not records or type(allowed) is not dict:
        raise ByteGuardError("record or policy set invalid")
    # Python equality treats False == 0. Require typed policy claims before
    # comparing them, including entries not selected by this declaration set.
    for spelling, claim in allowed.items():
        if (
            type(spelling) is not str
            or not spelling
            or not Path(spelling).is_absolute()
            or os.path.normcase(spelling) != spelling
            or type(claim) is not tuple
            or len(claim) != 2
        ):
            raise ByteGuardError("invalid outside policy")
        try:
            shape(dict(path=spelling, bytes=claim[0], sha256=claim[1]))
        except ByteGuardError:
            raise ByteGuardError("invalid outside policy") from None
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ByteGuardError("root directory required")
    aliases, physical = {}, {}
    for record in records:
        shape(record)
        target = resolve(record["path"], root)
        name = key(target)
        claim = (record["bytes"], record["sha256"])
        if not target.is_relative_to(root) and allowed.get(name) != claim:
            raise ByteGuardError("unapproved outside target")
        if name in physical and physical[name][1] != claim:
            raise ByteGuardError("conflicting alias claims")
        aliases[record["path"]] = name
        physical[name] = (target, claim)
    return Snapshot(root, MappingProxyType(aliases), MappingProxyType(physical))


def alias_guard(snapshot):
    for spelling, expected in snapshot.aliases.items():
        if key(resolve(spelling, snapshot.root)) != expected:
            raise ByteGuardError("declaration retargeted")


def stream(path):
    digest, count = hashlib.sha256(), 0
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                count += len(block)
                digest.update(block)
    except OSError:
        raise ByteGuardError("target unreadable") from None
    return count, digest.hexdigest()


def phase(snapshot):
    """Fresh aliases and physical bytes every time, no size/mtime shortcut."""
    alias_guard(snapshot)
    for target, claim in snapshot.physical.values():
        if stream(target) != claim:
            raise ByteGuardError("original target bytes changed")
    alias_guard(snapshot)
    return len(snapshot.physical)
