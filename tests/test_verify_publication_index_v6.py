"""Synthetic tests for independent index verification; no scientific acceptance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import verify_publication_index_v6 as chk


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rec(root: Path, path: Path) -> dict:
    resolved = Path(path).resolve()
    try:
        shown = resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        shown = str(resolved)
    return {"path": shown, "bytes": resolved.stat().st_size, "sha256": sha(resolved)}


def integrity(root: Path, records: list[dict], approved=None, book=None) -> dict:
    approved = approved or chk.ApprovedFiles.load(None)
    return chk.project_integrity(Path(root), records, approved, book)


def entry(root: Path, manifest: Path, payload: dict, approved=None) -> dict:
    inputs = payload.get("inputs", [])
    outputs = payload.get("outputs", [])
    return {
        "manifest": Path(manifest).resolve().relative_to(Path(root).resolve()).as_posix(),
        "manifest_sha256": sha(manifest),
        "experiment": chk.project_metadata("experiment", payload.get("experiment"), root),
        "created_at_utc": chk.project_metadata("created_at_utc", payload.get("created_at_utc"),
                                               root),
        "command": chk.project_metadata("command", payload.get("command"), root),
        "parameters": chk.project_metadata("parameters", payload.get("parameters"), root),
        "git": chk.project_metadata("git", payload.get("git"), root),
        "inputs": chk.project_public_records(inputs, root, chk._is_private_record),
        "outputs": chk.project_public_records(outputs, root, chk._is_private_record),
        "private_input_count": sum(1 for r in inputs if chk._is_private_record(r)),
        "private_output_count": sum(1 for r in outputs if chk._is_private_record(r)),
        "input_integrity": integrity(root, inputs, approved),
        "output_integrity": integrity(root, outputs, approved),
    }


def build(root: Path, *, complete: bool = True, drop_input: bool = False,
          external_file: Path | None = None, approve_external: bool = True,
          wrong_hash: bool = False, new_gate_dir: str | None = None) -> Path:
    """Build a self-consistent index-v6 tree; return the output directory.

    ``external_file`` adds a native input outside the root; ``approve_external``
    controls whether the matching allowlist entry is written. ``wrong_hash`` adds a
    historical native record whose digest does not match the file on disk.
    """
    root = Path(root).resolve()
    for rel in chk.CANONICAL_SOURCE_RELPATHS:
        source = root / rel
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# synthetic {rel}\n", encoding="utf-8")
        if rel == "scripts/build_publication_artifact_index.py":
            source.write_text(f"INCLUDE = {chk.FROZEN_V1_INCLUDE!r}\n", encoding="utf-8")
        if rel == "scripts/build_publication_artifact_index_v6.py":
            source.write_text(f"EXTRA_INCLUDE = {chk.FROZEN_V6_EXTRA_INCLUDE!r}\n"
                              f"CURRENT_REQUIRED = {chk.FROZEN_V6_CURRENT_REQUIRED!r}\n",
                              encoding="utf-8")

    shared = root / "data" / "aggregate.json"
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_text('{"ok": true}\n', encoding="utf-8")
    native_inputs = [rec(root, shared)]
    if wrong_hash:
        stale = root / "data" / "aggregate_copy.json"
        stale.write_text('{"stale": true}\n', encoding="utf-8")
        native_inputs.append({"path": "data/aggregate_copy.json",
                              "bytes": stale.stat().st_size, "sha256": "d" * 64})
    if drop_input:
        native_inputs.append({"path": "data/missing_aggregate.json", "bytes": 1,
                              "sha256": "c" * 64})
    approved_entries: list[dict] = []
    if external_file is not None:
        resolved_external = Path(external_file).resolve()
        native_inputs.append({"path": str(resolved_external),
                              "bytes": resolved_external.stat().st_size,
                              "sha256": sha(resolved_external)})
        if approve_external:
            approved_entries.append({"path": str(resolved_external),
                                     "bytes": resolved_external.stat().st_size,
                                     "sha256": sha(resolved_external)})

    private = root / "data" / "interim" / "faces" / "person_00001.jpg"
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_bytes(b"synthetic-private-bytes")
    native_inputs.append(rec(root, private))

    for rel in chk.effective_include():
        if any(c in rel for c in "*?["):
            continue
        manifest = root / "metrics" / rel
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(
            {"experiment": "synthetic", "created_at_utc": "2026-10-09T00:00:00+00:00",
             "command": ["synthetic"], "parameters": {}, "git": {},
             "inputs": native_inputs, "outputs": []}), encoding="utf-8")

    for name, payload in (
            ("strong_backbone_study", {"campaign": {"complete": True, "completed_runs": 36,
                                                    "expected_runs": 36}}),
            ("matched_agegap_arms", {"complete": True, "completed_run_count": 1,
                                     "expected_run_count": 1}),
            ("matched_agegap_fixed10_last", {"complete": True, "completed_run_count": 1,
                                             "expected_run_count": 1})):
        summary = root / "metrics" / name / "summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(payload), encoding="utf-8")

    if new_gate_dir:
        (root / "metrics" / new_gate_dir).mkdir(parents=True, exist_ok=True)

    allowlist_path = root / "out" / "approved_files.json" if approved_entries else None
    approved = chk.ApprovedFiles.load(None)
    if allowlist_path is not None:
        allowlist_path.parent.mkdir(parents=True, exist_ok=True)
        allowlist_path.write_text(json.dumps({"files": approved_entries}), encoding="utf-8")
        approved = chk.ApprovedFiles.load(allowlist_path)
    discovered = sorted(chk.selected_manifest_paths(root, approved))
    entries = [entry(root, m, json.loads(m.read_text(encoding="utf-8")), approved)
               for m in discovered]
    missing = chk.expected_missing_literals(root, approved)
    invalid = chk.recompute_invalid(root, approved)
    incomplete = chk.campaign_incomplete(root)
    is_complete = not missing and not invalid and not incomplete
    if not complete:
        is_complete = False

    out = root / "out"
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / chk.INDEX_FILENAME
    index = {
        "schema_version": 1,
        "scope": "T-BIOM publication experiments only; no face data or biometric templates",
        "entry_count": len(entries),
        "missing_expected_manifests": missing,
        "invalid_input_or_output_manifests": invalid,
        "incomplete_experiments": incomplete,
        "complete": is_complete,
        "entries": entries,
        "index_audit_version": 6,
        "publication_ready": False,
        "physical_snapshot_complete": is_complete,
        "current_required_manifests": [f"metrics/{r}" for r in chk.FROZEN_V6_CURRENT_REQUIRED],
        "limitation": "physical artifact integrity snapshot only; no publication clearance",
    }
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    bound = list(discovered)
    bound += [p for p, present in chk.semantic_gates(root, approved).items() if present]
    bound += [root / rel for rel in chk.CANONICAL_SOURCE_RELPATHS]
    for manifest in discovered:
        native = json.loads(manifest.read_text(encoding="utf-8"))
        for item in native["inputs"] + native["outputs"]:
            try:
                target = chk.resolve_declared(root, approved, item["path"])
            except chk.RefusalError:
                continue
            if target.is_file():
                bound.append(target)
    binding = [rec(root, p) for p in sorted({Path(p).resolve() for p in bound})]
    manifest = {
        "schema_version": 1,
        "experiment": chk.EXPERIMENT_NAME,
        "created_at_utc": "2026-10-09T00:00:00+00:00",
        "command": ["synthetic"],
        "parameters": {"canonical_index_updated": False,
                       "current_required": list(chk.FROZEN_V6_CURRENT_REQUIRED)},
        "metrics": {"execution_complete": True, "publication_ready": False,
                    "entry_count": len(entries),
                    "missing_count": len(missing),
                    "invalid_count": len(invalid),
                    "incomplete_count": len(incomplete),
                    "artifact_index_complete": is_complete,
                    "physical_snapshot_complete": is_complete,
                    "observed_physical_inputs": len(binding)},
        "inputs": binding,
        "outputs": [rec(root, index_path)],
        "runtime": {"python": "synthetic"},
        "git": {},
    }
    (out / chk.MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n",
                                             encoding="utf-8")
    return out


def allowlist(out: Path) -> Path | None:
    candidate = Path(out) / "approved_files.json"
    return candidate if candidate.is_file() else None


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def store(path: Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _verify(tmp_path, out, allowed=None):
    return chk.verify(Path(tmp_path), Path(out), allowed)


# -- 1. Baseline acceptance + alias-binding preflight ------------------------
def test_baseline_accepts_consistent_tree(tmp_path):
    out = build(tmp_path)
    result = _verify(tmp_path, out)
    assert result["accepted_physical_snapshot"] is True
    assert result["publication_ready"] is False
    assert result["index_audit_version"] == 6
    assert result["entry_count"] == result["listed_manifests"] == result["selected_manifests"]
    assert result["binding_verified"] > 0


def test_alias_conflict_refused(tmp_path):
    book = chk.RecordBook(tmp_path, chk.ApprovedFiles.load(None))
    target = tmp_path / "data" / "aggregate.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    for spelling in ("data/aggregate.json", "./data/aggregate.json"):
        book.declare(spelling, target.stat().st_size,
                     chk.sha256_bytes(target.read_bytes()), must_match=True, private=False)
    with pytest.raises(chk.IntegrityError):
        book.declare("data/aggregate.json", 1, "0" * 64, must_match=True, private=False)


def test_alias_retarget_during_phase_refused(tmp_path, monkeypatch):
    """A spelling that re-resolves to a new canonical path mid-verify is refused."""
    root = tmp_path.resolve()
    out = build(root)
    other = root / "data" / "other.json"
    other.write_text("{}", encoding="utf-8")
    calls = {"n": 0}
    original = chk.resolve_declared

    def fake(resolve_root, approved, spelling):
        if spelling == "data/aggregate.json":
            calls["n"] += 1
            if calls["n"] > 1:
                return other.resolve()
        return original(resolve_root, approved, spelling)

    monkeypatch.setattr(chk, "resolve_declared", fake)
    with pytest.raises(chk.CheckError):
        _verify(root, out)


# -- 2. Selection-state presence flags --------------------------------------
def test_new_gate_dir_is_selection_drift(tmp_path, monkeypatch):
    """A gate path appearing during verify must be caught even when empty."""
    out = build(tmp_path)
    original = chk.validate_manifest

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        (tmp_path / "metrics" / "lfw_bound_evaluation_late_20261009").mkdir(
            parents=True, exist_ok=True)
        return result

    monkeypatch.setattr(chk, "validate_manifest", spy)
    with pytest.raises(chk.CheckError):
        _verify(tmp_path, out)


def test_selection_state_carries_presence_flags(tmp_path):
    build(tmp_path)
    approved = chk.ApprovedFiles.load(None)
    (tmp_path / "metrics" / "lfw_bound_evaluation_empty_20261009").mkdir(parents=True)
    _, gates = chk.selection_state(tmp_path, approved)
    assert gates, "gate map must not be empty"
    assert all(isinstance(v, bool) for v in gates.values())
    absent = [k for k, v in gates.items() if v is False]
    assert absent, "an empty gate dir must be tracked as present=False"
    assert set(gates) == {str(p) for p in chk.semantic_gates(tmp_path, approved)}


# -- 3. Approved external inputs --------------------------------------------
def test_approved_external_input_accepted(tmp_path):
    external = tmp_path.parent / f"{tmp_path.name}_native_cache.bin"
    external.write_bytes(b"native-model-cache")
    out = build(tmp_path, external_file=external)
    allow = allowlist(out)
    assert allow is not None
    result = _verify(tmp_path, out, allow)
    assert result["accepted_physical_snapshot"] is True


def test_unapproved_external_input_not_read(tmp_path):
    """An unapproved external target must never be opened; it is reported unverifiable."""
    external = tmp_path.parent / f"{tmp_path.name}_unapproved.bin"
    external.write_bytes(b"must-not-be-read")
    out = build(tmp_path, external_file=external, approve_external=False)
    original = chk.hash_file
    opened: list[str] = []

    def spy(path):
        if Path(path).resolve() == external.resolve():
            opened.append(str(path))
        return original(path)

    chk.hash_file = spy
    try:
        result = _verify(tmp_path, out)
    finally:
        chk.hash_file = original
    assert opened == []
    assert result["invalid_count"] >= 1
    assert result["complete"] is False


def test_approved_external_size_mismatch_refused(tmp_path):
    external = tmp_path.parent / f"{tmp_path.name}_sized.bin"
    external.write_bytes(b"native-model-cache")
    out = build(tmp_path, external_file=external)
    allow = allowlist(out)
    payload = json.loads(Path(allow).read_text())
    payload["files"][0]["bytes"] = 1
    Path(allow).write_text(json.dumps(payload))
    with pytest.raises(chk.IntegrityError):
        _verify(tmp_path, out, allow)


def test_outside_root_rejected_without_allowlist(tmp_path):
    build(tmp_path)
    outside = tmp_path.parent / "elsewhere"
    with pytest.raises(chk.RefusalError):
        chk.resolve_declared(tmp_path, chk.ApprovedFiles.load(None), str(outside / "x.json"))


# -- 4. Recursive exact typing ----------------------------------------------
@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_exact_equal_rejects_nested_non_int(tmp_path, value):
    assert chk.exact_equal(1, value) is False


def test_exact_equal_dict_nested_bool():
    assert chk.exact_equal({"a": [{"b": 1}]}, {"a": [{"b": True}]}) is False
    assert chk.exact_equal({"a": [{"b": 1}]}, {"a": [{"b": 1}]}) is True


def test_bool_schema_version_refused(tmp_path):
    build(tmp_path)
    index_path = tmp_path / "out" / chk.INDEX_FILENAME
    payload = json.loads(index_path.read_text())
    payload["schema_version"] = True
    index_path.write_text(json.dumps(payload))
    with pytest.raises(chk.SchemaError):
        chk.precheck_index(index_path.read_bytes())


def test_bool_metric_refused(tmp_path):
    out = build(tmp_path)
    manifest_path = tmp_path / "out" / chk.MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text())
    payload["metrics"]["entry_count"] = True
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises((chk.SchemaError, chk.IntegrityError)):
        _verify(tmp_path, out)


# -- 5. Pinned metadata / index contract ------------------------------------
def test_publication_ready_true_refused(tmp_path):
    build(tmp_path)
    index_path = tmp_path / "out" / chk.INDEX_FILENAME
    payload = json.loads(index_path.read_text())
    payload["publication_ready"] = True
    index_path.write_text(json.dumps(payload))
    with pytest.raises(chk.SchemaError):
        chk.precheck_index(index_path.read_bytes())


def test_current_required_drift_refused(tmp_path):
    build(tmp_path)
    index_path = tmp_path / "out" / chk.INDEX_FILENAME
    payload = json.loads(index_path.read_text())
    payload["current_required_manifests"] = []
    index_path.write_text(json.dumps(payload))
    book = chk.RecordBook(tmp_path, chk.ApprovedFiles.load(None))
    with pytest.raises(chk.SchemaError):
        chk.validate_index(index_path.read_bytes(), tmp_path,
                           chk.ApprovedFiles.load(None), book)


def test_index_update_flag_refused(tmp_path):
    out = build(tmp_path)
    manifest_path = tmp_path / "out" / chk.MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text())
    payload["parameters"]["canonical_index_updated"] = True
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(chk.SchemaError):
        _verify(tmp_path, out)


def test_execution_complete_false_refused(tmp_path):
    out = build(tmp_path)
    manifest_path = tmp_path / "out" / chk.MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text())
    payload["metrics"]["execution_complete"] = False
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(chk.SchemaError):
        _verify(tmp_path, out)


def test_header_metadata_projection_drift_refused(tmp_path):
    out = build(tmp_path)
    index_path = tmp_path / "out" / chk.INDEX_FILENAME
    payload = json.loads(index_path.read_text())
    payload["entries"][0]["command"] = ["tampered"]
    index_path.write_text(json.dumps(payload))
    with pytest.raises(chk.IntegrityError):
        _verify(tmp_path, out)


# -- 6. Completion proof vs historical native claims ------------------------
def test_wrong_native_hash_makes_entry_invalid_not_refused(tmp_path):
    """A stale historical native digest must show up as invalid, not abort binding."""
    out = build(tmp_path, wrong_hash=True)
    result = _verify(tmp_path, out)
    assert result["invalid_count"] >= 1
    assert result["complete"] is False


# -- 7. Frozen private-record predicate -------------------------------------
@pytest.mark.parametrize("record,private", [
    ({"path": "data/interim/faces/a.bin"}, True),
    ({"path": "x/private/y.json"}, True),
    ({"path": "photo.JPG"}, True),
    ({"path": "tensor.npy"}, True),
    ({"path": "vk_00012_00034.json"}, True),
    ({"path": "metrics/summary.json"}, False),
    ({"path": "data/aggregate.json"}, False),
])
def test_private_record_predicate(record, private):
    assert chk._is_private_record(record) is private


# -- 8. Fresh size + hash pre/post ------------------------------------------
def test_size_only_change_detected(tmp_path):
    """Same-size in-place rewrite must be caught by fresh hashing in both snapshots."""
    build(tmp_path)
    book = chk.RecordBook(tmp_path, chk.ApprovedFiles.load(None))
    target = tmp_path / "data" / "aggregate.json"
    book.declare("data/aggregate.json", target.stat().st_size,
                 chk.sha256_bytes(target.read_bytes()), must_match=True, private=False)
    book.snapshot_pre()
    target.write_text('{"ok": false}', encoding="utf-8")  # same length, different bytes
    book.snapshot_post()
    with pytest.raises(chk.IntegrityError):
        book.verify_stability()


# -- 9/11. Containment + preflight read sites -------------------------------
def test_canonical_dir_refuses_file(tmp_path):
    target = tmp_path / "a_file.txt"
    target.write_text("x")
    with pytest.raises(chk.RefusalError):
        chk._canonical_dir(target)


def test_listed_manifest_outside_root_refused(tmp_path):
    out = build(tmp_path)
    index_path = tmp_path / "out" / chk.INDEX_FILENAME
    payload = json.loads(index_path.read_text())
    payload["entries"][0]["manifest"] = "../evil/summary.manifest.json"
    index_path.write_text(json.dumps(payload))
    with pytest.raises(chk.CheckError):
        _verify(tmp_path, out)


# -- 10. Redacted errors ----------------------------------------------------
def test_error_messages_are_generic():
    for exc in (chk.RefusalError("refused: path-outside-scope"),
                chk.SchemaError("refused: json-duplicate-key"),
                chk.IntegrityError("refused: alias-conflict")):
        assert str(exc).startswith("refused: ")
        assert "=" not in str(exc)
        assert "\\" not in str(exc)


def test_verify_never_leaks_paths_on_failure(tmp_path):
    out = build(tmp_path)
    secret = tmp_path / "data" / "secret_native.bin"
    secret.write_bytes(b"x")
    index_path = tmp_path / "out" / chk.INDEX_FILENAME
    payload = json.loads(index_path.read_text())
    payload["entries"][0]["inputs"] = [{"path": str(secret), "bytes": 1, "sha256": "a" * 64}]
    index_path.write_text(json.dumps(payload))
    try:
        _verify(tmp_path, out)
    except chk.CheckError as exc:
        assert "secret_native" not in str(exc)
        assert str(secret) not in str(exc)


# -- JSON hardening ---------------------------------------------------------
def test_duplicate_keys_refused():
    with pytest.raises(chk.SchemaError):
        chk.strict_loads(b'{"a": 1, "a": 2}')


def test_overflow_number_refused():
    with pytest.raises(chk.SchemaError):
        chk.strict_loads(b'{"a": 1e999}')


def test_non_utf8_refused():
    with pytest.raises(chk.SchemaError):
        chk.strict_loads(b"\xff\xfe{")


# -- Public-projection safety ----------------------------------------------
def test_duplicate_listed_manifest_refused(tmp_path):
    out = build(tmp_path)
    index_path = tmp_path / "out" / chk.INDEX_FILENAME
    payload = json.loads(index_path.read_text())
    payload["entries"].append(dict(payload["entries"][0]))
    payload["entry_count"] = len(payload["entries"])
    index_path.write_text(json.dumps(payload))
    with pytest.raises(chk.IntegrityError):
        _verify(tmp_path, out)


def test_public_row_id_refused(tmp_path):
    out = build(tmp_path)
    index_path = tmp_path / "out" / chk.INDEX_FILENAME
    payload = json.loads(index_path.read_text())
    payload["entries"][0]["parameters"] = {"vk_00012_00034": 1}
    index_path.write_text(json.dumps(payload))
    with pytest.raises((chk.UnsafeContentError, chk.IntegrityError)):
        _verify(tmp_path, out)


def test_public_absolute_path_refused(tmp_path):
    out = build(tmp_path)
    index_path = tmp_path / "out" / chk.INDEX_FILENAME
    payload = json.loads(index_path.read_text())
    payload["entries"][0]["parameters"] = {"where": "/home/user/private/data.json"}
    index_path.write_text(json.dumps(payload))
    with pytest.raises((chk.UnsafeContentError, chk.IntegrityError)):
        _verify(tmp_path, out)


def test_cli_redacts_on_failure(tmp_path, capsys, monkeypatch):
    (tmp_path / "out").mkdir()
    monkeypatch.setattr("sys.argv", ["check", "--root", str(tmp_path),
                                     "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as exc:
        chk.main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("refused: ")
    assert str(tmp_path) not in captured.err


def test_each_alias_is_guarded_even_after_first_valid(tmp_path, monkeypatch):
    target, other = tmp_path / 'a.json', tmp_path / 'b.json'
    target.write_bytes(b'{}')
    other.write_bytes(b'{}')
    book = chk.RecordBook(tmp_path, chk.ApprovedFiles.load(None))
    for alias in ('a.json', './a.json'):
        book.declare(alias, 2, chk.sha256_bytes(b'{}'), must_match=True, private=False)
    original = chk.resolve_declared
    monkeypatch.setattr(chk, 'resolve_declared', lambda root, approved, alias:
                        other if alias == './a.json' else original(root, approved, alias))
    with pytest.raises(chk.IntegrityError, match='alias-retarget'):
        book._fresh(str(target.resolve()))


def test_single_fresh_target_does_not_scan_entire_claim_map(tmp_path):
    class NoScan(dict):
        def items(self):
            raise AssertionError('quadratic scan')
    target = tmp_path / 'a.json'
    target.write_bytes(b'{}')
    book = chk.RecordBook(tmp_path, chk.ApprovedFiles.load(None))
    key = book.declare('a.json', 2, chk.sha256_bytes(b'{}'), must_match=True, private=False)
    book._claims = NoScan(book._claims)
    assert book._fresh(key) == (chk.sha256_bytes(b'{}'), 2)


@pytest.mark.parametrize('mutation', ['omit-native', 'duplicate-proof'])
def test_physical_completion_closure_is_exact(tmp_path, mutation):
    out = build(tmp_path)
    path = out / chk.MANIFEST_FILENAME
    proof = load(path)
    if mutation == 'omit-native':
        proof['inputs'] = [r for r in proof['inputs'] if r['path'] != 'data/aggregate.json']
    else:
        proof['inputs'].append(dict(proof['inputs'][0]))
    proof['metrics']['observed_physical_inputs'] = len(proof['inputs'])
    store(path, proof)
    with pytest.raises(chk.IntegrityError, match='physical-input-closure'):
        _verify(tmp_path, out)


def test_public_external_projection_is_not_a_physical_filename(tmp_path, monkeypatch):
    external = tmp_path.parent / f'{tmp_path.name}_model.bin'
    external.write_bytes(b'known-model')
    out = build(tmp_path, external_file=external)
    expected = {'path': f'<external-artifact>/{external.name}',
                'bytes': 11, 'sha256': chk.sha256_bytes(b'known-model')}
    assert expected in load(out/chk.INDEX_FILENAME)['entries'][0]['inputs']
    original = chk.resolve_declared
    def no_placeholder(root, approved, spelling):
        assert '<external-artifact>' not in spelling
        return original(root, approved, spelling)
    monkeypatch.setattr(chk, 'resolve_declared', no_placeholder)
    assert _verify(tmp_path, out, allowlist(out))['accepted_physical_snapshot']


def test_parsed_bytes_must_equal_fresh_snapshot(tmp_path):
    target = tmp_path / 'data.json'
    target.write_bytes(b'{"n":1}')
    book = chk.RecordBook(tmp_path, chk.ApprovedFiles.load(None))
    book.declare('data.json', 7, chk.sha256_bytes(target.read_bytes()),
                 must_match=True, private=False)
    book.snapshot_pre()
    target.write_bytes(b'{"n":2}')
    with pytest.raises(chk.IntegrityError, match='parsed-byte-drift'):
        book.read_json('data.json')


@pytest.mark.parametrize('mutation', ['bool-size', 'missing-hash', 'duplicate'])
def test_external_policy_strict_records(tmp_path, mutation):
    target = tmp_path / 'model.bin'
    target.write_bytes(b'x')
    item = {'path': str(target.resolve()), 'bytes': 1, 'sha256': chk.sha256_bytes(b'x')}
    rows = [item]
    if mutation == 'bool-size':
        item['bytes'] = True
    elif mutation == 'missing-hash':
        del item['sha256']
    else:
        rows.append(dict(item))
    policy = tmp_path / 'policy.json'
    store(policy, {'files': rows})
    with pytest.raises(chk.RefusalError):
        chk.ApprovedFiles.load(policy)


def test_cli_internal_error_is_redacted(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(chk, 'verify', lambda *args: (_ for _ in ()).throw(
        KeyError('private-person-identifier')))
    monkeypatch.setattr('sys.argv', ['check', '--out', str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        chk.main()
    assert error.value.code == 2
    assert capsys.readouterr().err == 'refused: internal\n'



@pytest.mark.parametrize("value", [True, 4, 5, 5.0, 6.0])
def test_wrong_or_bool_v6_index_version_refused(tmp_path, value):
    out = build(tmp_path)
    target = out / chk.INDEX_FILENAME
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["index_audit_version"] = value
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(chk.CheckError):
        chk.verify(tmp_path, out)


def test_v6_sources_and_history_requirements_are_explicit():
    assert "scripts/build_publication_artifact_index_v4.py" in chk.CANONICAL_SOURCE_RELPATHS
    assert "scripts/build_publication_artifact_index_v6.py" in chk.CANONICAL_SOURCE_RELPATHS
    assert "scripts/build_publication_artifact_index_v5.py" in chk.CANONICAL_SOURCE_RELPATHS
    assert len(chk.FROZEN_V6_CURRENT_REQUIRED) == 20
    assert "strong_native_trajectories10_v3_e326_20261009/summary.manifest.json" in chk.FROZEN_V6_CURRENT_REQUIRED
    assert "strong_history_presentation10_v1_layout2_e337_20261009/history.manifest.json" in chk.FROZEN_V6_CURRENT_REQUIRED
