"""Synthetic orchestration tests; no native training, CUDA or provenance claims."""

import itertools
import json
import subprocess

import pytest

from scripts import run_strong_queue_v2 as queue


def cells():
    return [dict(negative=n, scope=s, lr=lr, seed=seed) for n, s, lr, seed in
            itertools.product(("random", "lookalike"), ("head", "tail", "full"),
                              (1e-6, 1e-5), (42, 1, 2))]


def test_requires_exact_full_matrix_no_duplicates_or_bool_seeds():
    plan = dict(cells=cells(), completed=[])
    assert len(queue.validate_plan(plan)[0]) == 36
    plan["cells"][-1] = plan["cells"][0]
    with pytest.raises(ValueError):
        queue.validate_plan(plan)
    plan = dict(cells=cells(), completed=[])
    plan["cells"][0]["seed"] = True
    with pytest.raises(ValueError):
        queue.validate_plan(plan)


def test_dry_run_does_not_verify_claims_create_outputs_or_train(monkeypatch, tmp_path):
    path, out = tmp_path / "plan.json", tmp_path / "out"
    path.write_text(json.dumps(dict(cells=cells(), completed=[])), encoding="utf-8")
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **k: pytest.fail("training called"))
    result = queue.run(path, out)
    assert not result["training_launched"] and not result["completion_verified"]
    assert not out.exists()


def test_first_training_failure_stops_before_scoring_or_next_cell(monkeypatch, tmp_path):
    path, out = tmp_path / "plan.json", tmp_path / "out"
    path.write_text(json.dumps(dict(cells=cells(), completed=[])), encoding="utf-8")
    commands = []

    def fail(command, **kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(queue.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        queue.run(path, out, execute=True)
    assert len(commands) == 1 and commands[0][2] == "scripts.run_strong_backbone_cuda_cell_v1"
    ledger = json.loads((out / "queue.json").read_text())
    assert ledger["status"] == "failed" and ledger["completed"] == []
    with pytest.raises(FileExistsError):
        queue.run(path, out, execute=True)


def test_unverified_exclusion_fails_before_root_creation(monkeypatch, tmp_path):
    all_cells = cells()
    completed = [dict(cell=c, training_manifest="unverified.json", external_manifest="unverified_external.json")
                 for c in all_cells[1:]]
    path, out = tmp_path / "plan.json", tmp_path / "out"
    path.write_text(json.dumps(dict(cells=all_cells[:1], completed=completed)), encoding="utf-8")
    monkeypatch.setattr(queue, "verify_cell", lambda *args: (_ for _ in ()).throw(ValueError("unverified")))
    with pytest.raises(ValueError, match="unverified"):
        queue.run(path, out, execute=True)
    assert not out.exists()


def test_training_identity_must_match_explicit_excluded_cell(monkeypatch, tmp_path):
    native = dict(parameters=dict(negative="random", trainable_scope="head", learning_rate=1e-6, seed=1))
    monkeypatch.setattr(queue, "readiness", lambda paths: ({}, [native]))
    with pytest.raises(ValueError, match="identity"):
        queue.verify_cell(cells()[0], tmp_path / "training.json", tmp_path / "external.json")


def test_shared_input_mutation_rejected(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("original", encoding="utf-8")
    snapshot = {path: queue.file_record(path)}
    queue.verify_snapshot(snapshot)
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="shared scientific"):
        queue.verify_snapshot(snapshot)


def test_new_arm_ancestry_is_locked_after_first_use(monkeypatch, tmp_path):
    path = tmp_path / "new_arm_input.json"
    path.write_text("original", encoding="utf-8")
    monkeypatch.setattr(queue, "paths_from", lambda native: [path])
    snapshot = {}
    queue.extend_snapshot(snapshot, [{}])
    assert path in snapshot
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="previously bound"):
        queue.extend_snapshot(snapshot, [{}])
    with pytest.raises(RuntimeError, match="shared scientific"):
        queue.verify_snapshot(snapshot)


def test_external_must_bind_exact_training_manifest(monkeypatch, tmp_path):
    native = dict(parameters=dict(negative="random", trainable_scope="head", learning_rate=1e-6, seed=42))
    monkeypatch.setattr(queue, "readiness", lambda paths: ({}, [native]))
    monkeypatch.setattr(queue, "verified", lambda path, experiment: dict(metrics=dict(execution_complete=True), inputs=[]))
    training = tmp_path / "training.json"
    training.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind"):
        queue.verify_cell(cells()[0], training, tmp_path / "external.json")
