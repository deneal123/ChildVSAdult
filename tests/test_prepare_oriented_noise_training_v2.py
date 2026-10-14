"""Metadata-only tiny fixtures: no data, weights, CUDA or optimizer execution."""
from pathlib import Path

import pytest

from scripts import prepare_oriented_noise_training_v2 as producer


@pytest.mark.parametrize("seed", [1, 2])
def test_valid_permutation_seed(seed):
    assert producer.canonical_permutation_seed(seed) == seed


@pytest.mark.parametrize("seed", [True, False, 1., "1", None, 0, 3, 42])
def test_noncanonical_seed_rejected(seed):
    with pytest.raises(ValueError):
        producer.canonical_permutation_seed(seed)


@pytest.mark.parametrize("seed", [1, 2])
def test_matching_bound_preparation(seed, monkeypatch):
    monkeypatch.setattr(producer, "file_record", lambda path: {"path": str(path)})
    manifest = dict(parameters=dict(seed=seed), inputs=[dict(path="clean")], outputs=[dict(path="noise")])
    producer.require_preparation(manifest, "clean", "noise", seed)


@pytest.mark.parametrize("wrong", [True, 1., "1", None, 42, 2])
def test_wrong_prepared_seed_rejected(wrong, monkeypatch):
    monkeypatch.setattr(producer, "file_record", lambda path: {"path": str(path)})
    manifest = dict(parameters=dict(seed=wrong), inputs=[dict(path="clean")], outputs=[dict(path="noise")])
    with pytest.raises(ValueError):
        producer.require_preparation(manifest, "clean", "noise", 1)


@pytest.mark.parametrize("side", ["inputs", "outputs"])
def test_missing_original_pair_binding_rejected(side, monkeypatch):
    monkeypatch.setattr(producer, "file_record", lambda path: {"path": str(path)})
    manifest = dict(parameters=dict(seed=1), inputs=[dict(path="clean")], outputs=[dict(path="noise")])
    manifest[side] = []
    with pytest.raises(ValueError):
        producer.require_preparation(manifest, "clean", "noise", 1)


def test_existing_output_rejected_before_native_reads(tmp_path, monkeypatch):
    out = tmp_path / "existing"
    out.mkdir()
    monkeypatch.setattr(producer, "verified", lambda *args: pytest.fail("must not read native inputs"))
    with pytest.raises(FileExistsError):
        producer.prepare(Path("preflight"), Path("noise"), out, permutation_seed=1)


def test_noncanonical_seed_rejected_before_native_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(producer, "verified", lambda *args: pytest.fail("must not read native inputs"))
    with pytest.raises(ValueError):
        producer.prepare(Path("preflight"), Path("noise"), tmp_path / "out", permutation_seed=True)
