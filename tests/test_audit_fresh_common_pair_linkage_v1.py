"""Generated typed arrays and temporary byte records; no native image/model IO."""

import json

import numpy as np
import pytest

from age_gap.common.manifest import file_record
from scripts import audit_fresh_common_pair_linkage_v1 as audit


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    people = np.arange(650) % 82
    ages = np.arange(650) % 70
    cache = tmp_path / "data/external/fgnet_crops.npz"
    cache.parent.mkdir(parents=True)
    np.savez_compressed(cache, subjects=people, ages=ages)
    left = np.arange(5308) % 650
    right = (left + np.where(np.arange(5308) % 2, 1, 82)) % 650
    frame = dict(
        left=left,
        right=right,
        labels=(people[left] == people[right]).astype(int),
        subject_a=people[left],
        subject_b=people[right],
        source_gap=np.abs(ages[left] - ages[right]),
    )
    vectors = np.zeros((650, 512), np.float32)
    vectors[:, 0] = 1
    cell = "random_head_lr1e-06_s42"
    paths = [tmp_path / key / "summary.manifest.json" for key in ("strong", "weak", "common")]
    parents = []
    for path, keys in (
        (paths[0], ("frozen", cell)),
        (paths[1], ("frozen", "tuned_s1", "tuned_s2", "tuned_s42")),
    ):
        private = path.parent / "private"
        private.mkdir(parents=True)
        np.savez_compressed(private / "scores.npz", **frame)
        for key in keys:
            leaf = f"{key}_embeddings.npz" if path == paths[0] else f"embeddings_{key}.npz"
            np.savez_compressed(private / leaf, indices=np.arange(650), embeddings=vectors)
        outputs = [file_record(p) for p in private.iterdir()]
        if path == paths[0]:
            native = dict(
                experiment="adaface-fixed8-cuda-full-fgnet-roc-v2",
                parameters=dict(actual_cells=[cell], protocol_seed=42, tolerance=2),
                metrics=dict(execution_complete=True, publication_ready=False),
            )
        else:
            native = dict(
                experiment="facenet-fixed8-validation-full-fgnet-cuda-v3",
                parameters=dict(
                    native_experiment="facenet-fixed8-validation-loss-v2",
                    device="cuda",
                    image_inference=True,
                    evaluator_version="weak-fixed8-evaluator-v3-root-e421",
                    protocol_seed=42,
                    tolerance=2,
                    n_boot=2000,
                    bootstrap_seed=0,
                    actual_cells=["tuned_s1", "tuned_s2", "tuned_s42"],
                ),
                metrics=dict(
                    execution_complete=True,
                    publication_ready=False,
                    seeds=[1, 2, 42],
                    actual_cells=["tuned_s1", "tuned_s2", "tuned_s42"],
                    n_crops=650,
                    bindings=["n1", "n2", "n42"],
                ),
            )
        native.update(inputs=[file_record(cache)], outputs=outputs)
        path.write_text(json.dumps(native), encoding="utf-8")
        parents.append(native)
    common = dict(
        experiment="common-image-index-mechanism-v2-fresh-weak",
        command=["producer", "--strong", str(paths[0]), "--weak", str(paths[1])],
        metrics=dict(
            execution_complete=True,
            strong_cells=[cell],
            weak_keys=["frozen", "tuned_s1", "tuned_s2", "tuned_s42"],
            weak_upstream_bindings=["n1", "n2", "n42"],
        ),
        inputs=[
            file_record(cache),
            file_record(paths[0]),
            file_record(paths[1]),
            *parents[0]["outputs"],
            *parents[1]["outputs"],
        ],
    )
    paths[2].parent.mkdir()
    paths[2].write_text(json.dumps(common), encoding="utf-8")
    return paths, common, frame


def test_exact_fresh_selected_records(tmp_path, monkeypatch):
    paths, _, _ = fixture(tmp_path, monkeypatch)
    result, selected = audit.audit(paths[2])
    assert result["exact_pair_arrays_equal"] is True
    assert result["typed_vectors_checked"] == 6
    assert result["full_ancestry_reverified"] is False
    assert result["numerical_probe_replay"] is False
    assert result["publication_ready"] is False
    assert len(selected) == 12


def test_rehashed_pair_permutation_rejected(tmp_path, monkeypatch):
    paths, common, frame = fixture(tmp_path, monkeypatch)
    pair = paths[1].parent / "private/scores.npz"
    old = file_record(pair)
    np.savez_compressed(pair, **{k: v[::-1] for k, v in frame.items()})
    native = json.loads(paths[1].read_text())
    native["outputs"] = [file_record(pair) if r == old else r for r in native["outputs"]]
    old_parent = file_record(paths[1])
    paths[1].write_text(json.dumps(native), encoding="utf-8")
    common["inputs"] = [
        file_record(pair) if r == old else file_record(paths[1]) if r == old_parent else r
        for r in common["inputs"]
    ]
    paths[2].write_text(json.dumps(common), encoding="utf-8")
    with pytest.raises(ValueError, match="pair arrays differ"):
        audit.audit(paths[2])


def test_offroot_argument_rejected_before_read(tmp_path, monkeypatch):
    paths, common, _ = fixture(tmp_path, monkeypatch)
    common["command"][2] = str(tmp_path.parent)
    paths[2].write_text(json.dumps(common), encoding="utf-8")
    with pytest.raises(ValueError, match="inside project"):
        audit.audit(paths[2])


def test_unbound_changed_vectors_rejected(tmp_path, monkeypatch):
    paths, _, _ = fixture(tmp_path, monkeypatch)
    vector = paths[1].parent / "private/embeddings_tuned_s1.npz"
    vector.write_bytes(b"unbound synthetic mutation")
    with pytest.raises(ValueError, match="not bound"):
        audit.audit(paths[2])
