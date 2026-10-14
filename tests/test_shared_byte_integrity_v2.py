"""Synthetic pytest suite for the shared-byte integrity phase (stdlib only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import shared_byte_integrity_v2 as sbi


def _record(path: str, raw: bytes) -> dict[str, object]:
    return {"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _write(root: Path, name: str, raw: bytes) -> dict[str, object]:
    (root / name).write_bytes(raw)
    return _record(name, raw)


def test_duplicate_declarations_stream_once(tmp_path, monkeypatch):
    calls = []
    original = sbi._stream_file

    def counted(path):
        calls.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(sbi, "_stream_file", counted)
    first = _write(tmp_path, "block.bin", b"shared-bytes")
    second = dict(first)
    out = sbi.audited_records([first, second], tmp_path)
    assert calls == ["block.bin"]
    assert out == {"block.bin": {"bytes": len(b"shared-bytes"),
                                 "sha256": first["sha256"]}}


def test_alias_agreement_and_disagreement(tmp_path):
    raw = b"canonical"
    good = _write(tmp_path, "one.bin", raw)
    also = {"path": "./one.bin", "bytes": good["bytes"], "sha256": good["sha256"]}
    assert sbi.audited_records([good, also], tmp_path)["one.bin"]["bytes"] == len(raw)
    bad = {"path": "one.bin", "bytes": len(raw) + 1,
           "sha256": hashlib.sha256(b"other").hexdigest()}
    with pytest.raises(sbi.ByteIntegrityError) as excinfo:
        sbi.audited_records([good, bad], tmp_path)
    assert excinfo.value.stage == "record-alias-conflict"


def test_same_size_drift(tmp_path):
    record = _write(tmp_path, "table.bin", b"AAAAAAAA")
    (tmp_path / "table.bin").write_bytes(b"BBBBBBBB")
    with pytest.raises(sbi.ByteIntegrityError) as excinfo:
        sbi.audited_records([record], tmp_path)
    assert "same_size=True" in excinfo.value.detail


def test_disappearing_file(tmp_path):
    record = _write(tmp_path, "gone.bin", b"gone")
    (tmp_path / "gone.bin").unlink()
    with pytest.raises(sbi.ByteIntegrityError) as excinfo:
        sbi.audited_records([record], tmp_path)
    assert excinfo.value.stage == "record-missing"


def test_path_escape(tmp_path):
    for spelling in ("../outside.bin", "/etc/passwd", "a/../../etc", "C:/windows/x"):
        record = {"path": spelling, "bytes": 0, "sha256": "0" * 64}
        with pytest.raises(sbi.ByteIntegrityError) as excinfo:
            sbi.audited_records([record], tmp_path)
        assert excinfo.value.stage == "audited-records"
        assert "escapes-root" in excinfo.value.detail


def test_resolved_alias_dedups_and_retarget_is_caught(tmp_path, monkeypatch):
    real = tmp_path / "real.bin"
    real.write_bytes(b"payload")
    original = sbi._canonical
    target = {"to": real}

    def fake(path):
        if Path(path).name in ("real.bin", "alias.bin"):
            return target["to"]
        return original(path)

    monkeypatch.setattr(sbi, "_canonical", fake)
    seen = _record("real.bin", b"payload")
    alias = {"path": "./alias.bin", "bytes": seen["bytes"], "sha256": seen["sha256"]}
    mapped = sbi.audited_records([seen, alias], tmp_path)
    assert list(mapped) == ["real.bin"]
    assert sbi.rehash_phase(mapped, tmp_path) == mapped
    (tmp_path / "other.bin").write_bytes(b"MOVED!!")
    target["to"] = tmp_path / "other.bin"
    with pytest.raises(sbi.ByteIntegrityError) as excinfo:
        sbi.rehash_phase(mapped, tmp_path)
    assert excinfo.value.stage == "target-escape"
    assert "declared-target-changed" in excinfo.value.detail


def test_mocked_resolution_escape(tmp_path, monkeypatch):
    original = sbi._canonical
    monkeypatch.setattr(sbi, "_canonical",
                        lambda p: original(tmp_path / ".." / "..") if Path(p).name == "z.bin" else original(p))
    record = {"path": "z.bin", "bytes": 0, "sha256": "0" * 64}
    with pytest.raises(sbi.ByteIntegrityError) as excinfo:
        sbi.audited_records([record], tmp_path)
    assert "escapes-root" in excinfo.value.detail


@pytest.mark.parametrize("mutation", [
    {"bytes": True},
    {"bytes": -1},
    {"bytes": 1.5},
    {"bytes": float("nan")},
    {"sha256": "XYZ"},
    {"sha256": "a" * 63},
    {"path": 7},
    {"extra": 1},
    {"bytes": "8"},
])
def test_malformed_records(tmp_path, mutation):
    record = _write(tmp_path, "ok.bin", b"ok")
    record.update(mutation)
    with pytest.raises(sbi.ByteIntegrityError):
        sbi.audited_records([record], tmp_path)


def test_malformed_container_and_empty_file(tmp_path):
    with pytest.raises(sbi.ByteIntegrityError):
        sbi.audited_records({"path": "a"}, tmp_path)
    empty = _write(tmp_path, "empty.bin", b"")
    assert sbi.audited_records([empty], tmp_path) == {
        "empty.bin": {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}}


def test_rehash_fresh_stream_no_cache(tmp_path, monkeypatch):
    record = _write(tmp_path, "fresh.bin", b"fresh")
    mapped = sbi.audited_records([record], tmp_path)
    calls = []
    original = sbi._stream_file

    def counted(path):
        calls.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(sbi, "_stream_file", counted)
    assert sbi.rehash_phase(mapped, tmp_path) == mapped
    assert sbi.rehash_phase(mapped, tmp_path) == mapped
    assert calls == ["fresh.bin", "fresh.bin"]


def test_rehash_mutation_and_escape(tmp_path):
    record = _write(tmp_path, "state.bin", b"state")
    mapped = sbi.audited_records([record], tmp_path)
    (tmp_path / "state.bin").write_bytes(b"STATE")
    with pytest.raises(sbi.ByteIntegrityError) as excinfo:
        sbi.rehash_phase(mapped, tmp_path)
    assert excinfo.value.stage == "rehash-phase"
    escaping = {"../outside.bin": {"bytes": 0, "sha256": "0" * 64}}
    with pytest.raises(sbi.ByteIntegrityError) as excinfo:
        sbi.rehash_phase(escaping, tmp_path)
    assert excinfo.value.stage in ("target-escape", "target-missing")


def test_public_errors_never_leak_paths_or_hashes(tmp_path):
    secret = "identity_998877_token.bin"
    raw = b"private-material"
    record = _write(tmp_path, secret, raw)
    (tmp_path / secret).write_bytes(b"PRIVATE-materiaL")
    with pytest.raises(sbi.ByteIntegrityError) as excinfo:
        sbi.audited_records([record], tmp_path)
    text = f"{excinfo.value} {excinfo.value.subject} {excinfo.value.detail}"
    assert secret not in text
    assert record["sha256"] not in text


def test_parse_manifest_same_bytes(tmp_path):
    raw = json.dumps({"b": 1, "a": [2, 3]}).encode("utf-8")
    data, size, sha = sbi.parse_manifest_strict(raw)
    assert data == {"b": 1, "a": [2, 3]}
    assert (size, sha) == (len(raw), hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("raw", [
    b'{"a": 1, "a": 2}',
    b'{"a": NaN}',
    b'{"a": Infinity}',
    b"{not json}",
    b"\xff\xfe",
    "not-bytes",
])
def test_parse_manifest_strict_rejects(raw):
    with pytest.raises(sbi.ByteIntegrityError) as excinfo:
        sbi.parse_manifest_strict(raw)
    assert "reason=" in excinfo.value.detail


def test_all_declarations_preflight_before_any_content_read(tmp_path, monkeypatch):
    good = _write(tmp_path, "good.bin", b"good")

    def forbidden(path):
        pytest.fail("No content may be opened before all declarations are valid")

    monkeypatch.setattr(sbi, "_stream_file", forbidden)
    with pytest.raises(sbi.ByteIntegrityError):
        sbi.audited_records([good, {"path": "../PRIVATE_998877.bin", "bytes": 0,
                                   "sha256": "0" * 64}], tmp_path)


def test_declared_alias_retarget_same_bytes_is_rejected(tmp_path, monkeypatch):
    first = _write(tmp_path, "first.bin", b"same")
    _write(tmp_path, "second.bin", b"same")
    original = sbi._canonical
    target = tmp_path / "first.bin"

    def resolve(path):
        if Path(path).name == "alias.bin":
            return target
        return original(path)

    monkeypatch.setattr(sbi, "_canonical", resolve)
    alias = dict(first, path="alias.bin")
    snapshot = sbi.audited_records([alias], tmp_path)
    target = tmp_path / "second.bin"
    with pytest.raises(sbi.ByteIntegrityError) as error:
        sbi.rehash_phase(snapshot, tmp_path)
    assert error.value.stage == "target-escape"
    assert "alias.bin" not in str(error.value)


def test_alias_binding_cannot_be_lost_by_serializing_map(tmp_path):
    record = _write(tmp_path, "entry.bin", b"entry")
    snapshot = sbi.audited_records([record], tmp_path)
    copied = dict(snapshot)
    with pytest.raises(sbi.ByteIntegrityError, match="alias-bound-snapshot-required"):
        sbi.rehash_phase(copied, tmp_path)
    with pytest.raises(TypeError):
        snapshot["entry.bin"] = copied["entry.bin"]
    copied["entry.bin"]["bytes"] = 99
    assert snapshot["entry.bin"]["bytes"] == 5


@pytest.mark.parametrize("raw", [b'{"x":1e999}', b'{"x":-1e999}'])
def test_overflowed_json_number_is_nonfinite(raw):
    with pytest.raises(sbi.ByteIntegrityError, match="non-finite-number"):
        sbi.parse_manifest_strict(raw)


@pytest.mark.parametrize("value", [None, 42, object()])
def test_bad_root_normalizes_errors(value):
    with pytest.raises(sbi.ByteIntegrityError):
        sbi.audited_records([], value)


def test_empty_record_path_refused_before_read(tmp_path, monkeypatch):
    monkeypatch.setattr(sbi, "_stream_file", lambda path: pytest.fail("must not open root"))
    with pytest.raises(sbi.ByteIntegrityError, match="empty-record-path"):
        sbi.audited_records([{"path": "", "bytes": 0, "sha256": "0" * 64}], tmp_path)


def test_malformed_snapshot_refused_before_read(tmp_path, monkeypatch):
    snapshot = sbi.RecordSnapshot({"PRIVATE_998877.bin": (-1, "0" * 64)}, ())
    monkeypatch.setattr(sbi, "_stream_file", lambda path: pytest.fail("must not open malformed target"))
    with pytest.raises(sbi.ByteIntegrityError) as error:
        sbi.rehash_phase(snapshot, tmp_path)
    assert "PRIVATE" not in str(error.value)

def _rec(name: str, raw: bytes) -> dict[str, object]:
    return {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def test_duplicate_alias_1000_resolves_spelling_once(tmp_path, monkeypatch):
    raw = b"shared-bytes"
    _rec("block.bin", raw)
    (tmp_path / "block.bin").write_bytes(raw)
    root = _rec("block.bin", raw)
    records = [dict(root) for _ in range(1000)]
    calls, streams = [], []
    orig_res, orig_stream = sbi._declared_target, sbi._stream_file
    monkeypatch.setattr(sbi, "_declared_target",
                        lambda *a, **k: (calls.append(a[0]), orig_res(*a, **k))[1])
    monkeypatch.setattr(sbi, "_stream_file",
                        lambda p: (streams.append(Path(p).name), orig_stream(p))[1])
    snap = sbi.audited_records(records, tmp_path)
    assert len(records) == 1000
    assert len(snap._aliases) == 1, "1000 identical spellings must collapse to one binding"
    assert calls.count("block.bin") <= 6, f"memoized re-resolution={calls}"
    assert streams == ["block.bin"], "one physical target streamed once per phase"


def test_distinct_spelling_alias_agreement_and_conflict(tmp_path):
    raw = b"canonical"
    (tmp_path / "one.bin").write_bytes(raw)
    good = _rec("one.bin", raw)
    alias = {"path": "./one.bin", "bytes": good["bytes"], "sha256": good["sha256"]}
    assert sbi.audited_records([good, alias], tmp_path)["one.bin"]["bytes"] == len(raw)
    bad = {"path": "./one.bin", "bytes": len(raw) + 1,
           "sha256": hashlib.sha256(b"other").hexdigest()}
    with pytest.raises(sbi.ByteIntegrityError) as err:
        sbi.audited_records([good, bad], tmp_path)
    assert err.value.stage == "record-alias-conflict"


def test_mid_preflight_retarget_caught_before_any_read(tmp_path, monkeypatch):
    for name, blob in (("a.bin", b"alpha"), ("b.bin", b"bravo")):
        (tmp_path / name).write_bytes(blob)
    orig = sbi._declared_target
    state = {"seen": 0}
    monkeypatch.setattr(sbi, "_stream_file",
                        lambda p: pytest.fail("no read before whole re-resolution"))

    def resolve(rel, base, subject, stage):
        if rel == "a.bin":
            state["seen"] += 1
            if state["seen"] > 1:  # memo pass resolved once; later pass is retargeted
                return orig("b.bin", base, subject, stage)
        return orig(rel, base, subject, stage)

    monkeypatch.setattr(sbi, "_declared_target", resolve)
    with pytest.raises(sbi.ByteIntegrityError) as err:
        sbi.audited_records([_rec("a.bin", b"alpha")], tmp_path)
    assert err.value.stage == "target-escape"
    assert "declared-target-changed" in err.value.detail


def test_fresh_hash_every_phase_and_mutation_rejected(tmp_path, monkeypatch):
    raw = b"fresh"
    (tmp_path / "fresh.bin").write_bytes(raw)
    snap = sbi.audited_records([_rec("fresh.bin", raw)], tmp_path)
    orig = sbi._stream_file
    calls = []
    monkeypatch.setattr(sbi, "_stream_file",
                        lambda p: (calls.append(Path(p).name), orig(p))[1])
    assert sbi.rehash_phase(snap, tmp_path) == snap
    assert sbi.rehash_phase(snap, tmp_path) == snap
    assert calls == ["fresh.bin", "fresh.bin"], "no persistent hash cache across phases"
    (tmp_path / "fresh.bin").write_bytes(b"FRESH")
    with pytest.raises(sbi.ByteIntegrityError) as err:
        sbi.rehash_phase(snap, tmp_path)
    assert "same_size=True" in err.value.detail


def test_aliases_rechecked_after_read(tmp_path, monkeypatch):
    raw = b"tail"
    (tmp_path / "tail.bin").write_bytes(raw)
    (tmp_path / "other.bin").write_bytes(b"other")
    orig = sbi._declared_target
    streams, state = [], {"calls": 0}
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(sbi, "_stream_file",
                        lambda p: (streams.append(Path(p).name), (len(raw), digest))[1])

    def resolve(rel, base, subject, stage):
        state["calls"] += 1
        if rel == "tail.bin" and state["calls"] == 4:  # post-read re-resolution
            return orig("other.bin", base, subject, stage)
        return orig(rel, base, subject, stage)

    monkeypatch.setattr(sbi, "_declared_target", resolve)
    with pytest.raises(sbi.ByteIntegrityError) as err:
        sbi.audited_records([_rec("tail.bin", raw)], tmp_path)
    assert err.value.stage == "target-escape"
    assert streams == ["tail.bin"], "read happened, so this is the after-read recheck"


def test_public_error_never_leaks_path_or_hash(tmp_path):
    secret, raw = "identity_998877_token.bin", b"private-material"
    (tmp_path / secret).write_bytes(b"PRIVATE-materiaL")
    rec = _rec(secret, raw)
    with pytest.raises(sbi.ByteIntegrityError) as err:
        sbi.audited_records([rec], tmp_path)
    text = f"{err.value} {err.value.subject} {err.value.detail}"
    assert secret not in text and rec["sha256"] not in text


def test_rehash_needs_alias_bound_snapshot_and_containment(tmp_path):
    raw = b"v"
    (tmp_path / "v.bin").write_bytes(raw)
    snap = sbi.audited_records([_rec("v.bin", raw)], tmp_path)
    with pytest.raises(sbi.ByteIntegrityError, match="alias-bound-snapshot-required"):
        sbi.rehash_phase(dict(snap), tmp_path)
    escaping = sbi.RecordSnapshot({"../outside.bin": (0, "0" * 64)}, ())
    with pytest.raises(sbi.ByteIntegrityError) as err:
        sbi.rehash_phase(escaping, tmp_path)
    assert err.value.stage in ("target-escape", "target-missing")
