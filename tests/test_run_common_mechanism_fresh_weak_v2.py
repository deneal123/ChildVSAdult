"""Synthetic IO/statistic stubs, never native checkpoints or model execution."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import run_common_mechanism_fresh_weak_v2 as producer


def fixture(tmp_path, monkeypatch, *, mismatch=False):
    monkeypatch.setattr(producer, "PROJECT_ROOT", tmp_path)
    cache = tmp_path / "data/external/fgnet_crops.npz"
    cache.parent.mkdir(parents=True)
    people, ages = np.arange(650) % 82, np.arange(650) % 64
    np.savez_compressed(cache, subjects=people, ages=ages)
    left = np.arange(5308) % 650
    right = (left + np.where(np.arange(5308) % 2, 1, 82)) % 650
    pairs = dict(
        left=left,
        right=right,
        labels=(people[left] == people[right]).astype(int),
        subject_a=people[left],
        subject_b=people[right],
        source_gap=np.abs(ages[left] - ages[right]),
    )
    strong_path, weak_path = [
        tmp_path / name / "summary.manifest.json" for name in ("strong", "weak")
    ]

    def record(path):
        return dict(path=str(path))

    monkeypatch.setattr(producer, "file_record", record)
    vectors = np.zeros((650, 512), dtype=np.float32)
    vectors[:, 0] = 1
    for path, names in (
        (strong_path, ["frozen", "random_head_lr1e-06_s42"]),
        (weak_path, ["frozen", "tuned_s1", "tuned_s2", "tuned_s42"]),
    ):
        private = path.parent / "private"
        private.mkdir(parents=True)
        stored = {k: v.copy() for k, v in pairs.items()}
        if mismatch and path == weak_path:
            stored = {k: v[::-1].copy() for k, v in stored.items()}
        np.savez_compressed(private / "scores.npz", **stored)
        for name in names:
            filename = f"{name}_embeddings.npz" if path == strong_path else f"embeddings_{name}.npz"
            np.savez_compressed(private / filename, indices=np.arange(650), embeddings=vectors)
    strong = dict(
        experiment="adaface-fixed8-cuda-full-fgnet-roc-v2",
        parameters=dict(actual_cells=["random_head_lr1e-06_s42"], protocol_seed=42, tolerance=2),
        metrics=dict(execution_complete=True, publication_ready=False),
        inputs=[record(cache)],
        outputs=[record(p) for p in (strong_path.parent / "private").iterdir()],
    )
    weak = dict(
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
            bindings=["native1", "native2", "native42"],
        ),
        inputs=[record(cache)],
        outputs=[record(p) for p in (weak_path.parent / "private").iterdir()],
    )
    monkeypatch.setattr(
        producer, "verified", lambda path, experiment: strong if path == strong_path else weak
    )
    monkeypatch.setattr(producer, "script_ancestry", lambda *args: [])
    monkeypatch.setattr(producer, "age_probe", lambda *args: dict(synthetic_probe=True))
    monkeypatch.setattr(producer, "drift", lambda *args: dict(synthetic_drift=True))
    monkeypatch.setattr(producer, "paired_age_error", lambda *args: dict(synthetic_age=True))
    monkeypatch.setattr(producer, "infer", lambda scores, *args: dict(synthetic_keys=list(scores)))
    captured = {}
    monkeypatch.setattr(
        producer, "write_experiment_manifest", lambda path, **kwargs: captured.update(kwargs)
    )
    import scripts.evaluate_oriented_cuda_v1 as native_validator

    monkeypatch.setattr(native_validator, "validate_written_inputs", lambda *args: None)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["test", "--strong", str(strong_path), "--weak", str(weak_path), "--out", str(out)],
    )
    return out, captured


def test_distinct_new_lineage_and_no_neural_or_legacy_recovery(tmp_path, monkeypatch):
    out, captured = fixture(tmp_path, monkeypatch)
    producer.main()
    summary = json.loads((out / "summary.json").read_bytes())
    assert captured["experiment"] == "common-image-index-mechanism-v2-fresh-weak"
    assert captured["metrics"] == summary
    assert summary["fresh_weak_lineage"] is True
    assert summary["legacy_weak_history_recovered"] is False
    assert summary["neural_inference_replayed"] is False
    assert summary["mechanism_complete"] is summary["publication_ready"] is False
    assert summary["weak_keys"] == ["frozen", "tuned_s1", "tuned_s2", "tuned_s42"]
    assert len(summary["results"]["weak"]["diagnostics"]) == 3
    assert set(json.loads((out / "private/age_probes.json").read_bytes())["weak"]) == set(
        summary["weak_keys"]
    )


def test_pair_order_drift_before_any_diagnostic_output(tmp_path, monkeypatch):
    out, _ = fixture(tmp_path, monkeypatch, mismatch=True)
    with pytest.raises(ValueError, match="arrays differ"):
        producer.main()
    assert not out.exists()


def test_existing_destination_not_overwritten(tmp_path, monkeypatch):
    out, _ = fixture(tmp_path, monkeypatch)
    out.mkdir()
    with pytest.raises(FileExistsError):
        producer.main()


def test_source_no_old_cache_or_renderer_reinterpretation():
    source = Path(producer.__file__).read_text(encoding="utf-8")
    assert "private_embeddings_tuned_seed" not in source
    assert "weak = verified(args.weak, WEAK_EXPERIMENT)" in source
    assert "exact_pairs(strong_pairs, weak_pairs, people, ages)" in source
    assert '"common-image-index-mechanism-v2-fresh-weak"' in source
    assert "script_ancestry" in source


def test_actual_literal_script_closure_includes_new_helper_and_weak_lineage():
    root = Path(producer.__file__).resolve().parents[1]
    source = set(producer.script_ancestry(Path(producer.__file__), root))
    source.update(
        producer.script_ancestry(root / "scripts/evaluate_facenet_fixed8_cuda_v3.py", root)
    )
    expected = {
        "run_common_mechanism_fresh_weak_v2.py",
        "fresh_weak_common_binding_v1.py",
        "evaluate_facenet_fixed8_cuda_v3.py",
        "run_facenet_validation_cuda_v2.py",
        "mechanism_diagnostics_v1.py",
        "identity_separability_v1.py",
        "strong_subject_inference_v1.py",
    }
    assert expected <= {path.name for path in source}
    assert all(path.is_relative_to(root / "scripts") and path.is_file() for path in source)
