"""Synthetic, GPU-free tests for the recovery-v3 shared-byte trajectory runner.

No torch, no weights/images/arrays, no CUDA, no real campaign or native corpus.
All fixtures live in a temp root; the runner is imported normally.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_strong_trajectories_v3 as v3
from scripts import shared_byte_integrity_v1 as sbi

NATIVE = v3.NATIVE_EXPERIMENT


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(root: Path, relative: str, data: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def record(root: Path, relative: str) -> dict:
    data = (root / relative).read_bytes()
    return {"path": relative.replace("\\", "/"), "bytes": len(data), "sha256": sha(data)}


def history(base: float = 0.5) -> list:
    return [
        {
            "epoch": epoch,
            "training_loss": base / epoch,
            "validation_auc": 0.5 + epoch / 100.0,
            "mean_gradient_norm": 0.1 + epoch / 1000.0,
            "epoch_seconds": 1.0 + epoch / 100.0,
        }
        for epoch in range(1, 9)
    ]


def parameters(**overrides) -> dict:
    params = dict(
        backbone="adaface_ir101", epochs_requested=8, epochs_executed=8, selected_epoch=8,
        checkpoint_selection="last_epoch", batchnorm_policy="frozen_all", trainable_scope="head",
        learning_rate=1e-6, batch_size=16, margin=0.3, gap_weight=0.0, crops_dir="faces",
        seed=42, device="cuda", threads=1, negative="random",
        pairs_file="data/processed/pairs.jsonl",
    )
    params.update(overrides)
    return params


def metrics(**overrides) -> dict:
    values = dict(
        training_complete=True, evaluation_complete=False, publication_ready=False,
        peak_allocated_bytes=1024, peak_reserved_bytes=2048, train_pairs=100,
        validation_pairs=20, unique_crop_files=120, history=history(),
    )
    values.update(overrides)
    return values


class Fixture:
    """One synthetic root with a dependency tree and one native cell."""

    def __init__(self, root: Path, *, name="cell0", params=None, mets=None):
        self.root = root
        # Dependency tree (mandatory relative paths bound to this root).
        self.dep_root = root / "deps"
        for rel in v3.MANDATORY_DEPENDENCIES:
            write(self.dep_root, rel, f"# synthetic {rel}\n".encode())
        self.own_module = write(self.dep_root, "own_runner.py", b"# synthetic own module\n")
        # Native cell under the native base root.
        self.shared = "shared/pairs.jsonl"
        write(root, self.shared, b'{"pair":"shared"}')
        write(root, f"{name}/summary.json", b'{"training_complete": true}')
        write(root, f"{name}/private/checkpoint.pt", b"CHECKPOINT-BYTES")
        write(root, f"{name}/private/checkpoint.manifest.json", b'{"experiment": "component"}')
        self.native_inputs = [record(root, self.shared)]
        self.native_outputs = [
            record(root, f"{name}/summary.json"),
            record(root, f"{name}/private/checkpoint.pt"),
            record(root, f"{name}/private/checkpoint.manifest.json"),
        ]
        self.manifest = self.write_manifest(
            root / name / "train.manifest.json",
            parameters() if params is None else params,
            metrics() if mets is None else mets,
            self.native_inputs,
            self.native_outputs,
        )

    def write_manifest(self, path, params, mets, inputs, outputs):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(schema_version=1, experiment=NATIVE, parameters=params,
                       metrics=mets, inputs=inputs, outputs=outputs)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def aggregate(self, bindings=None, out="out", **kwargs):
        return v3.aggregate(
            [self.manifest] if bindings is None else bindings,
            self.root / out, root=self.root, dependency_root=self.dep_root,
            own_module=self.own_module, **kwargs)


def run(fixture: Fixture, out="out", **kwargs):
    return fixture.aggregate(out=out, **kwargs)


# ---------------------------------------------------------------------------
# Success and closure properties.
# ---------------------------------------------------------------------------


def test_success_declares_every_input_exactly_once(tmp_path):
    fixture = Fixture(tmp_path / "root")
    assert run(fixture) == 1
    out = fixture.root / "out"
    manifest = json.loads((out / "summary.manifest.json").read_text(encoding="utf-8"))
    declared = [row["path"] for row in manifest["inputs"]]
    assert len(declared) == len(set(declared))  # unique
    # Every native declared input and every native output role appears.
    expected_native = {
        fixture.root.joinpath(r["path"]).resolve() for r in
        fixture.native_inputs + fixture.native_outputs
    }
    for target in expected_native:
        assert any(Path(p).resolve() == target for p in declared)
    # The binding manifest itself appears.
    assert any(Path(p).resolve() == fixture.manifest.resolve() for p in declared)
    # Every mandatory dependency appears.
    for rel in v3.MANDATORY_DEPENDENCIES:
        dep = (fixture.dep_root / rel).resolve()
        assert any(Path(p).resolve() == dep for p in declared)
    # Cross-check each declared input record really hashes to its listed value.
    for row in manifest["inputs"]:
        path = Path(row["path"])
        if not path.is_absolute():
            path = v3.PROJECT_ROOT / path
        data = path.read_bytes()
        assert row == {"path": row["path"], "bytes": len(data), "sha256": sha(data)}
    # And every phase-1 target is declared.
    assert manifest["parameters"]["distinct_byte_targets"] == len(manifest["inputs"])


def test_success_curves_and_null_validation_loss(tmp_path):
    fixture = Fixture(tmp_path / "root")
    assert run(fixture) == 1
    payload = json.loads((fixture.root / "out" / "summary.json").read_text(encoding="utf-8"))
    assert payload["validation_loss"] is None
    assert payload["historical_loss_available"] is False
    assert payload["checkpoint_reselection"] is False
    assert payload["publication_clearance"] is False
    cell = payload["cells"]["random|head|lr1e-06|seed42"]
    assert cell["validation_loss"] is None
    assert cell["curves"]["training_loss"]["first"] == pytest.approx(0.5)
    assert cell["curves"]["validation_auc"]["values"] == [pytest.approx(0.51 + i / 100) for i in range(8)]
    assert cell["descriptive"]["validation_loss"] is None


def test_shared_records_streamed_once_per_phase(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path / "root")
    calls = []
    original = sbi._stream_file
    monkeypatch.setattr(sbi, "_stream_file",
                        lambda path: (calls.append(Path(path).name), original(path))[1])
    assert run(fixture) == 1
    for name in set(calls):
        assert calls.count(name) >= 2, name


# ---------------------------------------------------------------------------
# 1. Omission / input-declaration assertions.
# ---------------------------------------------------------------------------


def test_missing_native_record_is_never_omitted(tmp_path):
    fixture = Fixture(tmp_path / "root")
    assert run(fixture) == 1
    manifest = json.loads((fixture.root / "out" / "summary.manifest.json").read_text(encoding="utf-8"))
    # Removing a declared input from the manifest must fail the published guard.
    manifest["inputs"] = manifest["inputs"][:-1]
    with pytest.raises(v3.TrajectoryError) as excinfo:
        v3._validate_published_manifest(fixture.root / "out" / "summary.manifest.json",
                                        manifest["inputs"] + [{}], manifest["outputs"],
                                        [fixture.root / "out" / "summary.json"])
    assert "publication" in excinfo.value.stage


def test_duplicate_input_path_rejected(tmp_path):
    fixture = Fixture(tmp_path / "root")
    dep_snapshot = sbi.audited_records([record(fixture.dep_root, "own_runner.py")], fixture.dep_root)
    native_snapshot = sbi.audited_records(fixture.native_inputs, fixture.root)
    shared = fixture.root / fixture.shared
    with pytest.raises(v3.TrajectoryError) as excinfo:
        v3._expected_inputs([shared, shared], native_snapshot, dep_snapshot,
                            fixture.root, fixture.dep_root)
    assert "duplicate-input-path" in excinfo.value.detail


def test_all_output_roles_required(tmp_path):
    fixture = Fixture(tmp_path / "root")
    payload = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    payload["outputs"] = payload["outputs"][:2]
    fixture.manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(v3.TrajectoryError) as excinfo:
        run(fixture)
    assert excinfo.value.stage == "preflight"


# ---------------------------------------------------------------------------
# 2. Mandatory sources / missing source.
# ---------------------------------------------------------------------------


def test_missing_mandatory_dependency_rejected(tmp_path):
    fixture = Fixture(tmp_path / "root")
    (fixture.dep_root / v3.MANDATORY_DEPENDENCIES[0]).unlink()
    with pytest.raises(v3.TrajectoryError) as excinfo:
        run(fixture)
    assert excinfo.value.stage == "dependencies"


def test_empty_dependencies_cannot_bypass_mandatory_set(tmp_path):
    fixture = Fixture(tmp_path / "root")
    # Passing an explicit empty/None dependencies set still enforces mandatory.
    (fixture.dep_root / v3.MANDATORY_DEPENDENCIES[2]).unlink()
    for extra in (None, []):
        with pytest.raises(v3.TrajectoryError) as excinfo:
            fixture.aggregate(dependencies=extra)
        assert excinfo.value.stage == "dependencies"


def test_extra_dependency_outside_root_rejected(tmp_path):
    fixture = Fixture(tmp_path / "root")
    outside = write(tmp_path, "outside_dep.py", b"# outside")
    with pytest.raises(v3.TrajectoryError) as excinfo:
        fixture.aggregate(dependencies=[outside])
    assert excinfo.value.stage == "dependencies"


def test_source_mutation_during_phases_rejected(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path / "root")
    target = fixture.dep_root / v3.MANDATORY_DEPENDENCIES[1]
    original = v3._cell

    def mutate(cell, hist):
        target.write_bytes(b"# mutated source\n")
        return original(cell, hist)

    monkeypatch.setattr(v3, "_cell", mutate)
    with pytest.raises(sbi.ByteIntegrityError) as excinfo:
        run(fixture)
    assert excinfo.value.stage == "rehash-phase"
    assert "byte-drift" in excinfo.value.detail
    assert not (fixture.root / "out" / "summary.manifest.json").exists()


def test_snapshot_lookup_does_not_scan_the_manifest(tmp_path):
    class LookupOnly(dict):
        def __iter__(self):
            raise AssertionError("quadratic snapshot scan")
    path = write(tmp_path, "shared.bin", b"x")
    value = {"bytes": 1, "sha256": sha(b"x")}
    assert v3._snapshot_value(LookupOnly({"shared.bin": value}), {},
                              tmp_path, tmp_path, path) == (1, sha(b"x"))


@pytest.mark.parametrize("schema", [True, False, 1.0, "1", None, 2])
def test_schema_version_is_a_strict_integer_one(tmp_path, schema):
    fixture = Fixture(tmp_path / "root")
    data = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    data["schema_version"] = schema
    fixture.manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(v3.TrajectoryError, match="schema-version"):
        run(fixture)


def test_binding_alias_is_checked_after_derivation(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path / "root")
    alias = fixture.root / "binding_alias.json"
    alias.write_bytes(fixture.manifest.read_bytes())
    other = write(fixture.root, "other_manifest.json", fixture.manifest.read_bytes())
    original_resolve = Path.resolve
    state = {"retarget": False}
    def resolve(path, *args, **kwargs):
        if path == alias:
            return other if state["retarget"] else fixture.manifest
        return original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    original_cell = v3._cell
    def retarget(cell, hist):
        state["retarget"] = True
        return original_cell(cell, hist)
    monkeypatch.setattr(v3, "_cell", retarget)
    with pytest.raises(sbi.ByteIntegrityError, match="alias"):
        fixture.aggregate(bindings=[alias])
    assert not (fixture.root / "out" / "summary.manifest.json").exists()


def test_native_source_overlap_is_counted_once(tmp_path):
    fixture = Fixture(tmp_path / "root")
    data = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    data["inputs"].append(record(fixture.root, "deps/own_runner.py"))
    fixture.manifest.write_text(json.dumps(data), encoding="utf-8")
    assert run(fixture) == 1
    manifest = json.loads((fixture.root / "out" / "summary.manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameters"]["distinct_byte_targets"] == len(manifest["inputs"])


def test_manifest_metric_mutation_removes_only_own_completion_marker(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path / "root")
    writer = v3.write_experiment_manifest
    def wrong_metrics(*args, **kwargs):
        kwargs["metrics"] = dict(kwargs["metrics"], publication_ready=True)
        return writer(*args, **kwargs)
    monkeypatch.setattr(v3, "write_experiment_manifest", wrong_metrics)
    with pytest.raises(v3.TrajectoryError, match="metrics-differ"):
        run(fixture)
    assert (fixture.root / "out" / "summary.json").is_file()
    assert not (fixture.root / "out" / "summary.manifest.json").exists()


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_published_equality_rejects_nested_numeric_type_changes(value):
    assert not v3._exact({"a": [{"b": 1}]}, {"a": [{"b": value}]})


def test_extra_dependency_alias_remains_bound(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path / "root")
    alias = write(fixture.dep_root, "extra_alias.py", b"# alias\n")
    other = write(fixture.dep_root, "extra_other.py", b"# alias\n")
    state = {"retarget": False}
    original_resolve = Path.resolve
    def resolve(path, *args, **kwargs):
        if path == alias and state["retarget"]:
            return original_resolve(other, *args, **kwargs)
        return original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    original_cell = v3._cell
    def retarget(cell, hist):
        state["retarget"] = True
        return original_cell(cell, hist)
    monkeypatch.setattr(v3, "_cell", retarget)
    with pytest.raises(sbi.ByteIntegrityError, match="alias"):
        fixture.aggregate(dependencies=[alias])
    assert not (fixture.root / "out" / "summary.manifest.json").exists()


# ---------------------------------------------------------------------------
# 3. Destination containment / retarget.
# ---------------------------------------------------------------------------


def test_output_root_escape_rejected(tmp_path):
    fixture = Fixture(tmp_path / "root")
    with pytest.raises(v3.TrajectoryError) as excinfo:
        v3.aggregate([fixture.manifest], tmp_path / "outside_out", root=fixture.root,
                     dependency_root=fixture.dep_root, own_module=fixture.own_module)
    assert excinfo.value.stage == "output"


def test_destination_retarget_rejected(tmp_path):
    fixture = Fixture(tmp_path / "root")
    decoy = fixture.root / "elsewhere"
    decoy.mkdir(parents=True, exist_ok=True)
    with pytest.raises(v3.TrajectoryError) as excinfo:
        fixture.aggregate(destination_resolver=lambda _item: decoy)
    assert "retargeted-after-creation" in excinfo.value.detail


def test_fresh_destination_required(tmp_path):
    fixture = Fixture(tmp_path / "root")
    (fixture.root / "out").mkdir()
    with pytest.raises(FileExistsError):
        run(fixture)
    assert v3.main(["--bindings", str(fixture.manifest), "--out", str(fixture.root / "out")]) == 1


# ---------------------------------------------------------------------------
# 4. Binding alias retarget.
# ---------------------------------------------------------------------------


def test_binding_retarget_before_read_rejected(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path / "root")
    other = write(fixture.root, "other.bin", b"OTHER")
    real = v3._canonical
    state = {"n": 0}

    def flaky(path):
        resolved = real(path)
        if Path(path) == fixture.manifest:
            state["n"] += 1
            if state["n"] > 1:  # first call during _binding_alias, later during read
                return real(other)
        return resolved

    monkeypatch.setattr(v3, "_canonical", flaky)
    with pytest.raises(v3.TrajectoryError) as excinfo:
        run(fixture)
    assert "binding-retargeted" in excinfo.value.detail


# ---------------------------------------------------------------------------
# 5/6. After-phase2 mutation, published-file hashing, strict JSON.
# ---------------------------------------------------------------------------


def test_after_phase2_record_mutation_rejected(tmp_path, monkeypatch):
    fixture = Fixture(tmp_path / "root")
    shared = fixture.root / fixture.shared
    original = v3._expected_inputs

    def mutate(input_paths, native, deps, base, dep_base):
        result = original(input_paths, native, deps, base, dep_base)
        shared.write_bytes(b'{"pair":"MUTATED-AFTER-PHASE2"}')
        return result

    monkeypatch.setattr(v3, "_expected_inputs", mutate)
    with pytest.raises(sbi.ByteIntegrityError) as excinfo:
        run(fixture)
    assert excinfo.value.stage == "rehash-phase"


def test_published_manifest_hash_guard(tmp_path):
    fixture = Fixture(tmp_path / "root")
    assert run(fixture) == 1
    target = fixture.root / "out" / "summary.manifest.json"
    summary = fixture.root / "out" / "summary.json"
    expected_inputs = json.loads(target.read_text(encoding="utf-8"))["inputs"]
    expected_outputs = [{"path": r["path"], "bytes": r["bytes"], "sha256": r["sha256"]}
                        for r in json.loads(target.read_text(encoding="utf-8"))["outputs"]]
    # Tamper the published manifest bytes.
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["inputs"] = payload["inputs"][:-1]
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(v3.TrajectoryError) as excinfo:
        v3._validate_published_manifest(target, expected_inputs, expected_outputs, [summary])
    assert "inputs-changed-during-publication" in excinfo.value.detail
    # The validator never deletes arbitrary paths; only aggregate() cleans its
    # own fresh, still-bound completion marker after a failure.
    assert target.exists()
    # Tamper the published summary file bytes after write.
    assert run(fixture, out="out2") == 1
    target2 = fixture.root / "out2" / "summary.manifest.json"
    summary2 = fixture.root / "out2" / "summary.json"
    data = json.loads(target2.read_text(encoding="utf-8"))
    exp_in = data["inputs"]
    exp_out = [{"path": r["path"], "bytes": r["bytes"], "sha256": r["sha256"]} for r in data["outputs"]]
    summary2.write_bytes(b"{}")
    with pytest.raises(v3.TrajectoryError) as excinfo:
        v3._validate_published_manifest(target2, exp_in, exp_out, [summary2])
    assert "published-output-changed-after-write" in excinfo.value.detail


def test_strict_json_duplicate_and_overflow(tmp_path):
    fixture = Fixture(tmp_path / "root")
    fixture.manifest.write_bytes(b'{"experiment": "a", "experiment": "b"}')
    with pytest.raises(sbi.ByteIntegrityError) as excinfo:
        run(fixture)
    assert "duplicate-json-key" in excinfo.value.detail

    fixture2 = Fixture(tmp_path / "root2")
    fixture2.manifest.write_bytes(b'{"experiment": 1e999}')
    with pytest.raises(sbi.ByteIntegrityError) as excinfo:
        run(fixture2)
    assert "non-finite-number" in excinfo.value.detail

    fixture3 = Fixture(tmp_path / "root3")
    fixture3.manifest.write_bytes(b"{not json")
    with pytest.raises(sbi.ByteIntegrityError) as excinfo:
        run(fixture3)
    assert "malformed-json" in excinfo.value.detail


# ---------------------------------------------------------------------------
# Empty bindings, bool schema, contract, privacy.
# ---------------------------------------------------------------------------


def test_empty_bindings_rejected(tmp_path):
    fixture = Fixture(tmp_path / "root")
    with pytest.raises(v3.TrajectoryError) as excinfo:
        v3.aggregate([], fixture.root / "out", root=fixture.root,
                     dependency_root=fixture.dep_root, own_module=fixture.own_module)
    assert "empty-bindings" in excinfo.value.detail


def test_duplicate_binding_rejected(tmp_path):
    fixture = Fixture(tmp_path / "root")
    with pytest.raises(v3.TrajectoryError) as excinfo:
        fixture.aggregate(bindings=[fixture.manifest, fixture.manifest])
    assert "duplicate-binding" in excinfo.value.detail


@pytest.mark.parametrize("field,value", [
    ("epochs_executed", True),
    ("selected_epoch", True),
    ("batch_size", True),
    ("seed", True),
    ("threads", False),
])
def test_bool_is_not_int(tmp_path, field, value):
    fixture = Fixture(tmp_path / "root", params=parameters(**{field: value}))
    with pytest.raises(v3.TrajectoryError) as excinfo:
        run(fixture)
    assert "expected=int" in excinfo.value.detail


@pytest.mark.parametrize("field,value", [
    ("epochs_executed", 7), ("selected_epoch", 7), ("checkpoint_selection", "best_val"),
    ("batchnorm_policy", "trainable_all"), ("batch_size", 32), ("margin", 0.5),
    ("gap_weight", 0.1), ("crops_dir", "aligned"), ("device", "cpu"),
])
def test_fixed8_contract_rejections(tmp_path, field, value):
    fixture = Fixture(tmp_path / "root", params=parameters(**{field: value}))
    with pytest.raises(v3.TrajectoryError):
        run(fixture)


def test_fabricated_validation_loss_rejected(tmp_path):
    fixture = Fixture(tmp_path / "root", mets=metrics(validation_loss=0.1))
    with pytest.raises(v3.TrajectoryError) as excinfo:
        run(fixture)
    assert "fabricated-validation-loss" in excinfo.value.detail


def test_history_bool_epoch_rejected(tmp_path):
    rows = history()
    rows[0]["epoch"] = True
    fixture = Fixture(tmp_path / "root", mets=metrics(history=rows))
    with pytest.raises(v3.TrajectoryError) as excinfo:
        run(fixture)
    assert excinfo.value.stage == "history"


def test_duplicate_cell_rejected(tmp_path):
    root = tmp_path / "root"
    first = Fixture(root, name="cell0")
    second = Fixture(root, name="cell1")
    with pytest.raises(v3.TrajectoryError) as excinfo:
        v3.aggregate([first.manifest, second.manifest], root / "out", root=root,
                     dependency_root=first.dep_root, own_module=first.own_module)
    assert "duplicate-native-cell" in excinfo.value.detail


def test_privacy_safe_error(tmp_path):
    fixture = Fixture(tmp_path / "root")
    payload = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    payload["inputs"] = [{"path": "identity_name_private.bin", "bytes": 999,
                          "sha256": sha(b"SECRET")}]
    fixture.manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(sbi.ByteIntegrityError) as excinfo:
        run(fixture)
    message = str(excinfo.value)
    assert "identity_name_private" not in message
    assert sha(b"SECRET") not in message
    assert str(tmp_path).replace("\\", "/") not in message.replace("\\", "/")


def test_main_ok_and_fail_closed(tmp_path, capsys):
    fixture = Fixture(tmp_path / "root")
    # main uses the real dependency root; run the injected-root path directly.
    assert fixture.aggregate(out="fresh") == 1
    assert (fixture.root / "fresh" / "summary.manifest.json").is_file()
    assert v3.main(["--bindings", str(fixture.manifest),
                    "--out", str(fixture.root / "fresh")]) == 1
    captured = capsys.readouterr().out
    assert json.loads(captured.strip())["ok"] is False
