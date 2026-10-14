"""Adversarial synthetic CPU tests for the v5 hardening PROPOSAL.

No GPU, no real weights/data/network. Each test exercises a real tiny file, including a
SUCCESSFUL real ``write_experiment_manifest`` publication. Passing these is NOT native
CUDA evidence and does not recover any historical epoch loss.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from age_gap.common.manifest import file_record
from scripts import measure_checkpoint_validation_cuda_v5 as v5
from scripts.shared_byte_integrity_v2 import ByteIntegrityError


def _rec(path: Path) -> dict:
    return dict(file_record(path))


def _root(tmp_path) -> Path:
    root = tmp_path / "root"
    (root / "metrics").mkdir(parents=True)
    return root


def _junction(link: Path, target: Path) -> None:
    """Real NTFS junction (symlink creation is often not privileged on Windows)."""
    import subprocess

    if link.exists():
        link.rmdir()
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.skip(f"junction unavailable: {result.stderr.strip()[:60]}")


# --- 1. actual alias retarget at freeze (equal-byte targets) --------------------------
def test_freeze_alias_retarget_equal_bytes(tmp_path):
    root = _root(tmp_path)
    src, far = root / "src", root / "far"
    src.mkdir()
    far.mkdir()
    (src / "a.bin").write_bytes(b"same-bytes")
    (far / "a.bin").write_bytes(b"same-bytes")  # equal bytes: only the retarget can be caught
    sub = root / "sub"
    _junction(sub, src)
    declared = _rec(src / "a.bin")
    alias = {"path": "sub/a.bin", "bytes": declared["bytes"], "sha256": declared["sha256"]}
    frozen = v5.freeze_ancestry([declared, alias], root)
    assert len(frozen.aliases) == 2 and len(frozen.physical) == 1
    v5.fresh_check(frozen, root, "ok")  # stable alias target passes
    _junction(sub, far)  # retarget to an equal-byte physical target
    with pytest.raises(ByteIntegrityError):
        v5.fresh_check(frozen, root, "retarget")


def test_freeze_guard_detects_retarget_during_freeze(tmp_path, monkeypatch):
    root = _root(tmp_path)
    src, far = root / "src", root / "far"
    src.mkdir()
    far.mkdir()
    (src / "a.bin").write_bytes(b"same-bytes")
    (far / "a.bin").write_bytes(b"same-bytes")
    sub = root / "sub"
    _junction(sub, src)
    declared = _rec(src / "a.bin")
    alias = {"path": "sub/a.bin", "bytes": declared["bytes"], "sha256": declared["sha256"]}
    real = v5.sbi.audited_records

    def retarget_during(records, base):
        result = real(records, base)  # freeze hashes the real target first
        _junction(sub, far)  # alias now points at equal bytes elsewhere
        return result

    monkeypatch.setattr(v5.sbi, "audited_records", retarget_during)
    with pytest.raises(ByteIntegrityError):
        v5.freeze_ancestry([declared, alias], root)


def test_freeze_alias_conflict_and_no_blessed_new_bytes(tmp_path):
    root = _root(tmp_path)
    a = root / "a.bin"
    a.write_bytes(b"original")
    good = _rec(a)
    alias = dict(good, path="sub/../a.bin")
    frozen = v5.freeze_ancestry([good, alias], root)
    assert len(frozen.aliases) == 2  # every original spelling kept
    assert len(frozen.physical) == 1  # one physical target
    v5.fresh_check(frozen, root, "ok")
    conflict = dict(good)
    conflict["sha256"] = "0" * 64
    with pytest.raises(ByteIntegrityError):
        v5.freeze_ancestry([good, conflict], root)
    # Byte drift present BEFORE freeze is never silently blessed.
    a.write_bytes(b"drifted-bytes")
    with pytest.raises(ByteIntegrityError):
        v5.freeze_ancestry([{**good, "path": "a.bin"}], root)


# --- 2. external initial byte drift caught at freeze ----------------------------------
def test_external_byte_drift_caught_at_freeze(tmp_path):
    root = _root(tmp_path)
    outside = tmp_path / "cache/w.pt"
    outside.parent.mkdir()
    outside.write_bytes(b"weights")
    record = _rec(outside)
    record["path"] = str(outside)
    allowed = {os.path.normcase(str(outside.resolve())): (record["bytes"], record["sha256"])}
    frozen = v5.freeze_ancestry([record], root, allowed)  # checked AT freeze
    assert frozen.outside_targets and frozen.physical
    bad = {os.path.normcase(str(outside.resolve())): (record["bytes"] + 1, record["sha256"])}
    with pytest.raises(ByteIntegrityError):
        v5.freeze_ancestry([record], root, bad)
    with pytest.raises(ByteIntegrityError):
        v5.freeze_ancestry([record], root, {})  # generic outside path refused


# --- 3. inventory pin + mutation after parse ------------------------------------------
def _inventory(root: Path, key: str, artifact: Path) -> Path:
    path = root / "metrics/model_inventory.json"
    path.write_text(json.dumps({"models": {key: {"artifact": _rec(artifact)}}}), encoding="utf-8")
    return path


def test_inventory_pin_mutation_after_parse(tmp_path):
    root = _root(tmp_path)
    w = tmp_path / "w.pt"
    w.write_bytes(b"weights")
    path = _inventory(root, "adaface_ir101", w)
    pin = v5.pin_inventory(path, "adaface_ir101")
    v5.verify_pinned_json(pin.pin, "ok")
    assert v5.allowed_outside_weight(pin)  # allowlist comes from the pinned record
    pin.pin.parsed["models"]["adaface_ir101"]["artifact"]["bytes"] = 999  # in-memory only
    path.write_text(
        json.dumps({"models": {"adaface_ir101": {"artifact": _rec(w)}}}), encoding="utf-8"
    )
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ByteIntegrityError):
        v5.verify_pinned_json(pin.pin, "after")


def test_inventory_artifact_types_strict(tmp_path):
    root = _root(tmp_path)
    w = tmp_path / "w.pt"
    w.write_bytes(b"weights")
    bad = dict(_rec(w))
    bad["bytes"] = True  # bool must not pass as int
    path = root / "metrics/model_inventory.json"
    path.write_text(json.dumps({"models": {"k": {"artifact": bad}}}), encoding="utf-8")
    with pytest.raises(ByteIntegrityError):
        v5.pin_inventory(path, "k")


# --- 4. component pin + mutation after parse ------------------------------------------
def test_component_pin_mutation_after_parse(tmp_path):
    root = _root(tmp_path)
    comp = root / "component.manifest.json"
    comp.write_text(
        json.dumps({"experiment": v5.v2.COMPONENT_EXPERIMENT, "metrics": {}}), encoding="utf-8"
    )
    pin = v5.pin_json(comp)
    assert pin.parsed["experiment"] == v5.v2.COMPONENT_EXPERIMENT
    comp.write_text(
        json.dumps({"experiment": v5.v2.COMPONENT_EXPERIMENT, "metrics": {"x": 1}}),
        encoding="utf-8",
    )
    with pytest.raises(ByteIntegrityError):
        v5.verify_pinned_json(pin, "after")


# --- 5. output retarget before first score write, no external bytes altered -----------
def test_output_retarget_before_first_write(tmp_path):
    root = _root(tmp_path)
    src = root / "inputs/a.bin"
    src.parent.mkdir()
    src.write_bytes(b"alpha")
    frozen = v5.freeze_ancestry([_rec(src)], root)
    guard = v5.make_output_guard(root / "out", root, [c["path"] for c in frozen.claims])
    guard = v5.lease_output(guard, frozen)
    owned = Path(guard.resolved)
    outside = tmp_path / "external"
    outside.mkdir()
    (outside / "keep.txt").write_bytes(b"untouched")
    # Real retarget: move the leased directory aside, recreate the same spelling.
    owned.rename(tmp_path / "moved")
    owned.mkdir()
    with pytest.raises(RuntimeError):
        v5.guard_leaf(guard, owned / "private/scores.npz", "before scores write")
    with pytest.raises(RuntimeError):
        v5.guard_output(guard, frozen, "before scores write")
    assert (outside / "keep.txt").read_bytes() == b"untouched"


def test_output_guard_rejects_escape_and_alias(tmp_path):
    root = _root(tmp_path)
    a = root / "a.bin"
    a.write_bytes(b"alpha")
    frozen = v5.freeze_ancestry([_rec(a)], root)
    with pytest.raises(ValueError):
        v5.make_output_guard(tmp_path / "outside", root)
    # A non-existent subdirectory still under the root resolves inside and is allowed.
    v5.make_output_guard(root / "deep" / "nested", root)
    with pytest.raises(ValueError):
        v5.make_output_guard(a, root, [str(a)])  # out IS a declared ancestry target
    with pytest.raises(ValueError):
        v5.make_output_guard(root, root, [str(a)])  # out is an ancestor of a target
    guard = v5.make_output_guard(root / "out", root, [c["path"] for c in frozen.claims])
    guarded = v5.lease_output(guard, frozen)
    with pytest.raises(RuntimeError):
        v5.guard_leaf(guarded, Path(guard.resolved) / "elsewhere.json", "leaf")


# --- 6. nested bool records -----------------------------------------------------------
def test_nested_bool_records_rejected(tmp_path):
    bad = {"path": "a.bin", "bytes": True, "sha256": "0" * 64}
    with pytest.raises(ByteIntegrityError):
        v5.strict_record(bad, "r")
    for left, right in (
        ({"a": True}, {"a": 1}),
        ([True], [1]),
        ({"x": {"y": False}}, {"x": {"y": 0}}),
    ):
        with pytest.raises(ValueError):
            v5._type_exact_equal(left, right, "k")
    v5._type_exact_equal({"a": 1}, {"a": 1}, "k")
    with pytest.raises(ByteIntegrityError):
        v5._records_type_exact([bad], [{"path": "a.bin", "bytes": 1, "sha256": "0" * 64}], "in")


# --- 7. successful REAL synthetic manifest write (positive path) -----------------------
def test_successful_real_manifest_write(tmp_path):
    root = _root(tmp_path)
    src = root / "inputs/a.bin"
    src.parent.mkdir()
    src.write_bytes(b"alpha")
    frozen = v5.freeze_ancestry([_rec(src)], root)
    guard = v5.lease_output(
        v5.make_output_guard(root / "out", root, [c["path"] for c in frozen.claims]), frozen
    )
    owned = Path(guard.resolved)
    private = owned / "private"
    private.mkdir()
    npz = private / "scores.npz"
    npz.write_bytes(b"diagnostics")
    summary = owned / "summary.json"
    payload = {"status": "completed", "execution_complete": True, "n": 3}
    v5._write_summary(summary, payload)
    claims = (file_record(summary), file_record(npz))
    target = owned / "summary.manifest.json"
    v5.publish_manifest(
        target,
        summary=summary,
        npz=npz,
        frozen=frozen,
        summary_payload=payload,
        params_out={"experiment": v5.EXPERIMENT, "seed": 42},
        guard=guard,
        root=root,
        cli=["--execute"],
        claimed_outputs=claims,
    )
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["experiment"] == v5.EXPERIMENT
    from age_gap.common.manifest import SCHEMA_VERSION

    assert written["schema_version"] == SCHEMA_VERSION
    assert [r["sha256"] for r in written["inputs"]] == [c["sha256"] for c in frozen.claims]
    assert written["outputs"] == list(claims)
    assert written["metrics"] == payload


def test_publish_detects_post_write_output_and_input_mutation(tmp_path, monkeypatch):
    root = _root(tmp_path)
    src = root / "inputs/a.bin"
    src.parent.mkdir()
    src.write_bytes(b"alpha")
    frozen = v5.freeze_ancestry([_rec(src)], root)
    guard = v5.lease_output(
        v5.make_output_guard(root / "out", root, [c["path"] for c in frozen.claims]), frozen
    )
    owned = Path(guard.resolved)
    (owned / "private").mkdir()
    npz = owned / "private/scores.npz"
    npz.write_bytes(b"diag")
    summary = owned / "summary.json"
    payload = {"status": "completed", "execution_complete": True}
    v5._write_summary(summary, payload)
    claims = (file_record(summary), file_record(npz))
    real = v5.write_experiment_manifest

    def mutate_output(path, **kwargs):
        result = real(path, **kwargs)
        npz.write_bytes(b"changed-after-write")
        return result

    monkeypatch.setattr(v5, "write_experiment_manifest", mutate_output)
    with pytest.raises(RuntimeError):
        v5.publish_manifest(
            owned / "summary.manifest.json",
            summary=summary,
            npz=npz,
            frozen=frozen,
            summary_payload=payload,
            params_out={"a": 1},
            guard=guard,
            root=root,
            cli=["--execute"],
            claimed_outputs=claims,
        )
    assert json.loads(summary.read_text(encoding="utf-8"))["execution_complete"] is False
    assert npz.exists()  # diagnostics preserved


def test_publish_rejects_wrong_experiment(tmp_path, monkeypatch):
    root = _root(tmp_path)
    src = root / "inputs/a.bin"
    src.parent.mkdir()
    src.write_bytes(b"alpha")
    frozen = v5.freeze_ancestry([_rec(src)], root)
    guard = v5.lease_output(
        v5.make_output_guard(root / "out", root, [c["path"] for c in frozen.claims]), frozen
    )
    owned = Path(guard.resolved)
    (owned / "private").mkdir()
    npz = owned / "private/scores.npz"
    npz.write_bytes(b"diag")
    summary = owned / "summary.json"
    payload = {"status": "completed", "execution_complete": True}
    v5._write_summary(summary, payload)
    target = owned / "summary.manifest.json"

    def fake_manifest(path, **kwargs):
        Path(path).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment": "wrong",
                    "parameters": {},
                    "metrics": payload,
                    "inputs": [],
                    "outputs": [],
                }
            ),
            encoding="utf-8",
        )
        return Path(path)

    monkeypatch.setattr(v5, "write_experiment_manifest", fake_manifest)
    with pytest.raises(RuntimeError):
        v5.publish_manifest(
            target,
            summary=summary,
            npz=npz,
            frozen=frozen,
            summary_payload=payload,
            params_out={},
            guard=guard,
            root=root,
            cli=[],
            claimed_outputs=(file_record(summary), file_record(npz)),
        )


# --- 8. closure + honest CLI + redaction ----------------------------------------------
def test_closure_missing_mandatory_seed_fails(tmp_path):
    (tmp_path / "src/age_gap").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        v5.producing_source_paths(tmp_path)


def test_command_line_honest_metadata(monkeypatch):
    import argparse

    args = argparse.Namespace(train="T", out="O")
    monkeypatch.delattr(v5.sys, "orig_argv", raising=False)
    argv, meta = v5.command_line(args)
    assert meta["command_reconstructed"] is True and meta["command_argv"] == argv
    assert "--train" in argv and "--out" in argv and argv[-1] == "--execute"
    fake = [
        "C:/py/python.exe",
        "-m",
        "scripts.measure_checkpoint_validation_cuda_v5",
        "--train",
        "T",
        "--out",
        "O",
        "--execute",
    ]
    monkeypatch.setattr(v5.sys, "orig_argv", fake, raising=False)
    argv2, meta2 = v5.command_line(args)
    assert argv2 == fake and meta2["command_source"] == "sys.orig_argv"
    assert meta2["command_reconstructed"] is False


def test_redaction_never_echoes_paths():
    exc = ByteIntegrityError("stage", "record[0]", "reason=byte-drift path=/secret sha=abcd")
    text = v5._redact(exc)
    assert "/secret" not in text and "abcd" not in text and "reason=byte-drift" in text
    assert v5._redact(RuntimeError("raw /secret/path")) == "RuntimeError: reason=internal"


def test_private_directory_is_an_explicit_owned_leaf(tmp_path):
    root = _root(tmp_path)
    guard = v5.lease_output(v5.make_output_guard(root / "out", root), None)
    private = Path(guard.resolved) / "private"
    assert v5.guard_leaf(guard, private, "before private mkdir") == private
    private.mkdir()
    assert v5.guard_leaf(guard, private, "after private mkdir") == private
    assert v5.guard_leaf(guard, private / "scores.npz", "before scores") == private / "scores.npz"


def test_private_junction_cannot_turn_allowance_into_external_path(tmp_path):
    root = _root(tmp_path)
    guard = v5.lease_output(v5.make_output_guard(root / "out", root), None)
    outside = tmp_path / "external"
    outside.mkdir()
    keep = outside / "scores.npz"
    keep.write_bytes(b"untouched")
    private = Path(guard.resolved) / "private"
    _junction(private, outside)
    with pytest.raises(RuntimeError):
        v5.guard_leaf(guard, private, "before mkdir")
    with pytest.raises(RuntimeError):
        v5.guard_leaf(guard, private / "scores.npz", "before write")
    assert keep.read_bytes() == b"untouched"


def test_output_hard_link_to_existing_external_file_is_refused(tmp_path):
    root = _root(tmp_path)
    guard = v5.lease_output(v5.make_output_guard(root / "out", root), None)
    outside = tmp_path / "external.json"
    outside.write_bytes(b"untouched")
    leaf = Path(guard.resolved) / "summary.json"
    os.link(outside, leaf)
    with pytest.raises(RuntimeError, match="hard links"):
        v5.guard_leaf(guard, leaf, "before write")
    assert outside.read_bytes() == b"untouched"


@pytest.mark.parametrize("field", ["seed", "gap_weight", "learning_rate", "batch_size"])
def test_native_protocol_numeric_booleans_are_refused(field, monkeypatch):
    params = dict(
        epochs_requested=8,
        epochs_executed=8,
        selected_epoch=8,
        seed=1,
        batch_size=16,
        gap_weight=0.0,
        margin=0.3,
        learning_rate=1e-6,
    )
    params[field] = True
    monkeypatch.setattr(v5.v2, "native_contract", lambda _: (params, []))
    with pytest.raises(ValueError):
        v5.native_contract_v5({})
