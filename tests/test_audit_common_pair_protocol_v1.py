"""Tiny selected-record fixtures; not independent verification of native research."""

import json

import numpy as np
import pytest

from age_gap.common.manifest import file_record
from scripts import audit_common_pair_protocol_v1 as audit


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def bindings(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    cache = tmp_path / "data/external/fgnet_crops.npz"
    cache.parent.mkdir(parents=True)
    people = np.arange(650) % 82
    ages = np.arange(650) % 30
    np.savez(cache, subjects=people, ages=ages)
    weak_path = tmp_path / "weak/manifest.json"
    weak = dict(experiment="fgnet-endpoint-age-matched-error-breakdown",
                inputs=[file_record(cache)], outputs=[])
    for key in ("frozen", "tuned_seed42", "tuned_seed1", "tuned_seed2"):
        path = weak_path.parent / f"private/private_embeddings_{key}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic opaque embedding fixture " + key.encode())
        weak["outputs"].append(file_record(path))
    save(weak_path, weak)
    result = []
    for name in ("random_head_lr1e-06_s42", "random_tail_lr1e-06_s42"):
        parent_path = tmp_path / name / "summary.manifest.json"
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
        path = tmp_path / f"common_{name}/summary.manifest.json"
        save(path, common)
        result.append(path)
    return result


def mutate_parent(binding, callback):
    common = json.loads(binding.read_text(encoding="utf-8"))
    parent_path = audit.argument(common, "--strong")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    callback(parent, parent_path)
    save(parent_path, parent)
    common["inputs"] = [file_record(parent_path),
                        *[r for r in common["inputs"]
                          if r["path"] != file_record(parent_path)["path"]]]
    # Refresh selected output bindings only for fixtures simulating a different valid protocol.
    for record in parent["outputs"]:
        common["inputs"] = [r for r in common["inputs"] if r["path"] != record["path"]]
        common["inputs"].append(record)
    save(binding, common)


def test_exact_pair_linkage_and_limits(bindings):
    payload, paths = audit.audit(bindings)
    assert payload["checked_cell_count"] == 2
    assert payload["n_pairs"] == 5308 and payload["n_recorded_persons"] == 82
    assert payload["exact_pair_arrays_equal"] is True
    assert payload["mechanism_complete"] is False and payload["publication_ready"] is False
    assert "not full native-ancestry" in payload["limitation"]
    assert bindings[0] in paths


def test_same_counts_do_not_prove_same_pairs(bindings):
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


def test_parent_protocol_mismatch(bindings):
    mutate_parent(bindings[1], lambda parent, path: parent["parameters"].update(tolerance=3))
    with pytest.raises(ValueError, match="parent protocols"):
        audit.audit(bindings)


def test_frozen_bytes_mismatch(bindings):
    def change(parent, parent_path):
        path = parent_path.parent / "private/frozen_embeddings.npz"
        path.write_bytes(b"different frozen fixture")
        parent["outputs"][1] = file_record(path)
    mutate_parent(bindings[1], change)
    with pytest.raises(ValueError, match="reference differs"):
        audit.audit(bindings)


def test_duplicate_empty_and_tampered_file_rejected(bindings):
    with pytest.raises(ValueError, match="unique"):
        audit.audit([bindings[0], bindings[0]])
    with pytest.raises(ValueError, match="at least one"):
        audit.audit([])
    common = json.loads(bindings[0].read_text(encoding="utf-8"))
    parent_path = audit.argument(common, "--strong")
    parent_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exact native record"):
        audit.audit(bindings)


def test_final_manifest_mutation_refused(bindings, tmp_path, monkeypatch):
    import sys

    # Real run source records are deliberately replaced by one tiny fixture source.
    monkeypatch.setattr(audit, "__file__", str(bindings[0]))
    for name in ("scripts/run_common_mechanism_v1.py", "scripts/evaluate_oriented_cuda_v1.py",
                 "src/age_gap/common/manifest.py", "src/age_gap/common/io.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture source")
    original = audit.write_experiment_manifest

    def mutate(*args, **kwargs):
        target = original(*args, **kwargs)
        (tmp_path / "scripts/run_common_mechanism_v1.py").write_bytes(b"mutated source")
        return target

    monkeypatch.setattr(audit, "write_experiment_manifest", mutate)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["audit", "--bindings", *map(str, bindings),
                                    "--out", str(out)])
    with pytest.raises(RuntimeError, match="changed"):
        audit.main()
    assert not (out / "summary.manifest.json").exists()
