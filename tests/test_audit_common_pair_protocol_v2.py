"""Synthetic tests for the v2 canonical-cell shared-pair linkage audit.

These tests exercise only tiny local fixtures. They do not run any native audit
against live artifacts. Canonical v2 never imports or mutates canonical v1;
unrelated tests may import v1 independently in the same pytest process.
"""

import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_common_pair_protocol_v2 as audit

HERE = Path(__file__).resolve()
V1_SHA256 = "8908d90e4256279ad312b5c47407ff7fda038c29191ffaba009bf929778eef1e"


ROOT = HERE.parents[1]

CANONICAL_NAMES = [
    f"{family}_{view}_lr{lr}_s{seed}"
    for family, view, lr, seed in itertools.product(
        ("random", "lookalike"), ("head", "tail", "full"),
        ("1e-06", "1e-05"), ("42", "1", "2"),
    )
]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def path_of(record):
    path = Path(record["path"])
    return (path if path.is_absolute() else audit.PROJECT_ROOT / path).resolve()


def file_record(path):
    from age_gap.common.manifest import file_record as canonical

    return canonical(path)


def make_cache(root):
    cache = root / "data/external/fgnet_crops.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    people = np.arange(650) % 82
    ages = np.arange(650) % 30
    np.savez(cache, subjects=people, ages=ages)
    return cache, people, ages


def build(root, names, weak_tag="frozen"):
    cache, people, ages = make_cache(root)
    weak_path = root / f"weak_{weak_tag}/manifest.json"
    weak = dict(experiment="fgnet-endpoint-age-matched-error-breakdown", inputs=[], outputs=[])
    for key in audit.WEAK_MODELS:
        path = weak_path.parent / f"private/private_embeddings_{key}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic opaque embedding fixture " + f"{weak_tag}-{key}".encode())
        weak["outputs"].append(file_record(path))
    weak["inputs"] = [file_record(cache)]
    save(weak_path, weak)
    result = []
    for name in names:
        parent_path = root / name / "summary.manifest.json"
        private = parent_path.parent / "private"
        private.mkdir(parents=True)
        left = np.arange(5308) % 82
        labels = (np.arange(5308) % 2).astype(int)
        right = np.where(labels == 1, left, (left + 1) % 82)
        np.savez(private / "scores.npz", left=left, right=right, labels=labels,
                 subject_a=people[left], subject_b=people[right],
                 source_gap=np.abs(ages[left] - ages[right]))
        (private / "frozen_embeddings.npz").write_bytes(b"same opaque frozen fixture")
        parent = dict(experiment="adaface-fixed8-cuda-full-fgnet-roc-v2",
                      metrics=dict(execution_complete=True),
                      parameters=dict(protocol_seed=42, tolerance=2, actual_cells=[name]),
                      inputs=[file_record(cache)],
                      outputs=[file_record(private / "scores.npz"),
                               file_record(private / "frozen_embeddings.npz")])
        save(parent_path, parent)
        common = dict(experiment="common-image-index-mechanism-v1",
                      command=["producer", "--strong", str(parent_path),
                               "--weak", str(weak_path)],
                      metrics=dict(execution_complete=True,
                                   results=dict(strong=dict(diagnostics={name: {}}))),
                      inputs=[file_record(parent_path), file_record(weak_path),
                              *parent["inputs"], *parent["outputs"], *weak["outputs"]])
        path = root / f"common_{name}/summary.manifest.json"
        save(path, common)
        result.append(path)
    return result


def mutate_parent(binding, callback):
    common = json.loads(binding.read_text(encoding="utf-8"))
    parent_path = audit.argument(common, "--strong")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    callback(parent, parent_path)
    save(parent_path, parent)
    # Refresh the selected parent and output bindings in the common manifest.
    keep = [r for r in common["inputs"] if path_of(r) != parent_path.resolve()
            and path_of(r) not in {path_of(o) for o in parent["outputs"]}]
    common["inputs"] = [file_record(parent_path), *keep, *parent["outputs"]]
    save(binding, common)


def mutate_common(binding, callback):
    common = json.loads(binding.read_text(encoding="utf-8"))
    callback(common)
    save(binding, common)


def rebind_weak(binding, new_weak_path):
    common = json.loads(binding.read_text(encoding="utf-8"))
    old_weak = audit.argument(common, "--weak")
    command = common["command"]
    command[command.index("--weak") + 1] = str(new_weak_path)
    new_weak = json.loads(new_weak_path.read_text(encoding="utf-8"))
    keep = [r for r in common["inputs"]
            if path_of(r) != old_weak and path_of(r).parent != old_weak.parent / "private"]
    common["inputs"] = [*keep, file_record(new_weak_path), *new_weak["outputs"]]
    save(binding, common)


@pytest.fixture
def bindings(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    return build(tmp_path, ["random_head_lr1e-06_s42", "random_tail_lr1e-06_s42"])


def test_v1_source_is_preserved():
    v1 = ROOT / "scripts/audit_common_pair_protocol_v1.py"
    assert hashlib.sha256(v1.read_bytes()).hexdigest() == V1_SHA256
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "from scripts.audit_common_pair_protocol_v1" not in source
    assert "import scripts.audit_common_pair_protocol_v1" not in source


def test_all_36_canonical_names_stay_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    found = build(tmp_path, CANONICAL_NAMES)
    assert len(found) == 36
    payload, paths = audit.audit(found)
    assert payload["checked_cells"] == sorted(CANONICAL_NAMES)
    assert payload["checked_cell_count"] == 36
    assert payload["expected_canonical_cell_count"] == 36
    # Even at full canonical coverage the audit stays partial.
    assert payload["mechanism_complete"] is False
    assert payload["publication_ready"] is False
    assert payload["execution_complete"] is True
    assert payload["n_images"] == 650 and payload["n_pairs"] == 5308
    assert payload["n_recorded_persons"] == 82
    assert payload["exact_pair_arrays_equal"] is True
    assert payload["frozen_embedding_bytes_equal"] is True
    assert payload["weak_embedding_bindings_equal"] is True
    for phrase in ("not full ancestry hash", "human identity clearance",
                   "training provenance", "seed-population inference"):
        assert phrase in payload["limitation"]
    assert found[0] in paths


def test_high_lr_head_and_lookalike_full_and_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    names = ["random_head_lr1e-05_s42", "lookalike_full_lr1e-06_s42",
             "lookalike_tail_lr1e-05_s2"]
    found = build(tmp_path, names)
    payload, _ = audit.audit(found)
    assert payload["checked_cell_count"] == 3
    assert payload["checked_cells"] == sorted(names)
    assert payload["mechanism_complete"] is False


def test_pair_array_difference_rejected(bindings):
    def change(parent, parent_path):
        path = parent_path.parent / "private/scores.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["right"][0] = 2
        arrays["subject_b"][0] = 2
        arrays["source_gap"][0] = 2
        np.savez(path, **arrays)
        parent["outputs"][0] = file_record(path)
    mutate_parent(bindings[1], change)
    with pytest.raises(ValueError, match="reference differs"):
        audit.audit(bindings)


def test_frozen_bytes_difference_rejected(bindings):
    def change(parent, parent_path):
        path = parent_path.parent / "private/frozen_embeddings.npz"
        path.write_bytes(b"different frozen fixture")
        parent["outputs"][1] = file_record(path)
    mutate_parent(bindings[1], change)
    with pytest.raises(ValueError, match="reference differs"):
        audit.audit(bindings)


def test_weak_cache_reference_difference_rejected(bindings, tmp_path):
    other_path = tmp_path / "weak_other/manifest.json"
    other = dict(experiment="fgnet-endpoint-age-matched-error-breakdown",
                 inputs=[file_record(tmp_path / "data/external/fgnet_crops.npz")], outputs=[])
    for key in audit.WEAK_MODELS:
        path = other_path.parent / f"private/private_embeddings_{key}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"other weak fixture " + key.encode())
        other["outputs"].append(file_record(path))
    save(other_path, other)
    rebind_weak(bindings[1], other_path)
    with pytest.raises(ValueError, match="reference differs"):
        audit.audit(bindings)


def test_swapped_endpoint_metadata_rejected(bindings):
    def change(parent, parent_path):
        path = parent_path.parent / "private/scores.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["subject_a"], arrays["subject_b"] = arrays["subject_b"], arrays["subject_a"]
        np.savez(path, **arrays)
        parent["outputs"][0] = file_record(path)
    mutate_parent(bindings[1], change)
    with pytest.raises(ValueError, match="endpoint metadata"):
        audit.audit(bindings)


def test_bound_metadata_not_bound_rejected(bindings, tmp_path):
    def change(parent, parent_path):
        decoy = tmp_path / "decoy_cache.npz"
        decoy.parent.mkdir(parents=True, exist_ok=True)
        np.savez(decoy, subjects=np.arange(650) % 82, ages=np.arange(650) % 30)
        parent["inputs"][0] = file_record(decoy)
    mutate_parent(bindings[1], change)
    with pytest.raises(ValueError, match="exact native record"):
        audit.audit(bindings)


def test_common_execution_flag_must_be_true(bindings):
    mutate_common(bindings[0], lambda c: c["metrics"].update(execution_complete=1))
    with pytest.raises(ValueError, match="completed common-index"):
        audit.audit(bindings)


def test_strong_execution_flag_must_be_true(bindings):
    def flag(parent, parent_path):
        parent["metrics"]["execution_complete"] = 1
    mutate_parent(bindings[1], flag)
    with pytest.raises(ValueError, match="parent protocols"):
        audit.audit(bindings)


def test_malformed_pair_shapes_rejected(bindings):
    def coverage(parent, parent_path):
        path = parent_path.parent / "private/scores.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["labels"] = arrays["labels"][:-1]
        np.savez(path, **arrays)
        parent["outputs"][0] = file_record(path)
    mutate_parent(bindings[1], coverage)
    with pytest.raises(ValueError, match="coverage"):
        audit.audit(bindings)


def test_malformed_pair_ndim_rejected(bindings):
    def shape(parent, parent_path):
        path = parent_path.parent / "private/scores.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["left"] = arrays["left"].reshape(-1, 2)
        arrays["right"] = arrays["right"].reshape(-1, 2)
        np.savez(path, **arrays)
        parent["outputs"][0] = file_record(path)
    mutate_parent(bindings[1], shape)
    with pytest.raises(ValueError, match="valid shared image-index pairs"):
        audit.audit(bindings)


def test_protocol_and_body_cell_mismatch(bindings):
    mutate_common(bindings[1], lambda c: c["metrics"]["results"]["strong"].update(
        diagnostics={"random_full_lr1e-05_s2": {}}))
    with pytest.raises(ValueError, match="not linked to strong parent"):
        audit.audit(bindings)


def test_non_canonical_cell_name_rejected(bindings):
    mutate_parent(bindings[1], lambda p, path: p["parameters"].update(
        actual_cells=["random_head_lr1e-04_s42"]))
    with pytest.raises(ValueError, match="unique canonical"):
        audit.audit(bindings)


def test_duplicate_and_empty_rejected(bindings):
    with pytest.raises(ValueError, match="unique canonical"):
        audit.audit([bindings[0], bindings[0]])
    with pytest.raises(ValueError, match="at least one"):
        audit.audit([])


@pytest.mark.parametrize("kind", ["81_people", "float_people", "negative_age", "nan_age", "matrix_meta"])
def test_entire_chronological_metadata_contract_checked(tmp_path, monkeypatch, kind):
    original_cache = make_cache

    def changed_cache(root):
        path, people, ages = original_cache(root)
        if kind == "81_people":
            people = np.arange(650) % 81
        elif kind == "float_people":
            people = people.astype(float)
        elif kind == "negative_age":
            ages[0] = -1
        elif kind == "nan_age":
            ages = ages.astype(float)
            ages[0] = np.nan
        else:
            people, ages = people[:, None], ages[:, None]
        np.savez(path, subjects=people, ages=ages)
        return path, people, ages

    monkeypatch.setattr(sys.modules[__name__], "make_cache", changed_cache)
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    found = build(tmp_path, ["random_head_lr1e-05_s42"])
    with pytest.raises(ValueError, match="coverage"):
        audit.audit(found)


@pytest.mark.parametrize("key,value", [("protocol_seed", 42.0), ("tolerance", 2.0),
                                      ("tolerance", True)])
def test_protocol_integer_types_not_coerced(bindings, key, value):
    mutate_parent(bindings[0], lambda p, _: p["parameters"].update({key: value}))
    with pytest.raises(ValueError, match="parent protocols"):
        audit.audit(bindings)


@pytest.mark.parametrize("command", [["--strong"], ["--strong", "--weak"],
                                     "--strong file", ["--strong", 1]])
def test_parent_argument_shape_rejected(command):
    with pytest.raises(ValueError, match="explicit parent"):
        audit.argument({"command": command}, "--strong")


def _prepare_final_sources(tmp_path, bindings, monkeypatch):
    monkeypatch.setattr(audit, "__file__", str(bindings[0]))
    for name in ("scripts/run_common_mechanism_v1.py", "scripts/evaluate_oriented_cuda_v1.py",
                 "src/age_gap/common/manifest.py", "src/age_gap/common/io.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture source")


def test_source_mutation_after_publication_removes_manifest(tmp_path, monkeypatch, bindings):
    _prepare_final_sources(tmp_path, bindings, monkeypatch)
    original = audit.write_experiment_manifest
    victim = tmp_path / "scripts/run_common_mechanism_v1.py"

    def mutate(*args, **kwargs):
        target = original(*args, **kwargs)
        victim.write_bytes(b"mutated source")
        return target

    monkeypatch.setattr(audit, "write_experiment_manifest", mutate)
    out = tmp_path / "out_mutate"
    monkeypatch.setattr(sys, "argv",
                        ["audit", "--bindings", *map(str, bindings), "--out", str(out)])
    with pytest.raises(RuntimeError, match="changed"):
        audit.main()
    assert not (out / "summary.manifest.json").exists()


def test_source_removal_after_publication_removes_manifest(tmp_path, monkeypatch, bindings):
    _prepare_final_sources(tmp_path, bindings, monkeypatch)
    original = audit.write_experiment_manifest
    victim = tmp_path / "scripts/evaluate_oriented_cuda_v1.py"

    def remove(*args, **kwargs):
        target = original(*args, **kwargs)
        os.unlink(victim)
        return target

    monkeypatch.setattr(audit, "write_experiment_manifest", remove)
    out = tmp_path / "out_remove"
    monkeypatch.setattr(sys, "argv",
                        ["audit", "--bindings", *map(str, bindings), "--out", str(out)])
    with pytest.raises(RuntimeError, match="changed"):
        audit.main()
    assert not (out / "summary.manifest.json").exists()
