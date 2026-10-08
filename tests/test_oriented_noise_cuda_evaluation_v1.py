from pathlib import Path

import numpy as np
import pytest

from scripts import evaluate_oriented_noise_cuda_v1 as evaluator
from scripts.evaluate_oriented_noise_cuda_v1 import aligned_clean


def fixture():
    left, right = np.array([0, 1]), np.array([0, 2])
    labels, gaps, subjects = np.array([1, 0]), np.array([25, 25]), np.array(["a", "b", "c"])
    data = dict(
        left=left.copy(),
        right=right.copy(),
        labels=labels.copy(),
        source_gap=gaps.copy(),
        subject_a=subjects[left],
        subject_b=subjects[right],
        frozen=np.array([0.9, 0.1]),
    )
    for seed in (42, 1, 2):
        data[f"cross_s{seed}"] = np.array([0.8, 0.2])
    return data, left, right, labels, gaps, subjects


def test_same_order_clean_scores():
    values = fixture()
    scores, frozen = aligned_clean(*values)
    assert scores.shape == (3, 2)
    assert frozen.tolist() == [0.9, 0.1]


@pytest.mark.parametrize(
    "field", ["left", "right", "labels", "source_gap", "subject_a", "subject_b"]
)
def test_misalignment_rejected(field):
    values = fixture()
    data = values[0]
    if data[field].dtype.kind in "US":
        data[field][0] = "wrong"
    else:
        data[field][0] += 1
    with pytest.raises(ValueError, match="metadata"):
        aligned_clean(*values)


def test_nonfinite_rejected():
    values = fixture()
    values[0]["cross_s42"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        aligned_clean(*values)


def readiness_fixture(monkeypatch):
    """Synthetic linkage records; real hashing is covered by native execution."""
    binding = Path("campaign/training-bound.manifest.json")
    native = dict(
        parameters=evaluator.PROTOCOL | dict(device="cuda", permutation_seed=42),
        metrics=dict(training_complete=True, completed_cells=3),
        outputs=[],
    )
    cells, components, calls = {}, {}, []
    for seed in evaluator.SEEDS:
        ckpt = binding.parent / "private" / f"noise_s{seed}.pt"
        component = ckpt.with_suffix(".manifest.json")
        cell = binding.parent / f"noise_s{seed}.manifest.json"
        records = [dict(path=str(p)) for p in (ckpt, component, cell)]
        native["outputs"].extend(records)
        cells[str(cell)] = dict(
            metrics=dict(training_complete=True),
            parameters=dict(seed=seed, device="cuda", permutation_seed=42),
            outputs=records[:2],
        )
        components[str(component)] = dict(outputs=records[:1])

    def verified(path, experiment):
        if experiment == "oriented-uniform-cuda-noise-training":
            return native
        if experiment == "oriented-native-cuda-noise-cell":
            return cells[str(path)]
        assert experiment == "pair-contrastive-backbone-finetune"
        return components[str(path)]

    monkeypatch.setattr(evaluator, "verified", verified)
    monkeypatch.setattr(evaluator, "file_record", lambda p: dict(path=str(p)))
    monkeypatch.setattr(evaluator, "training_contract", lambda *args: calls.append(args))
    return binding, native, cells, components, calls


def test_three_seed_readiness_links_each_component(monkeypatch):
    binding, _, _, _, calls = readiness_fixture(monkeypatch)
    _, selected = evaluator.readiness(binding)
    assert set(selected) == {42, 1, 2}
    assert [call[2] for call in calls] == list(evaluator.SEEDS)
    assert all(call[1].name == "partial_noise_arm.jsonl" for call in calls)


@pytest.mark.parametrize(
    "change",
    [
        "incomplete",
        "missing_cell",
        "cpu",
        "wrong_permutation",
        "wrong_seed",
        "unbound_component",
        "unbound_checkpoint",
    ],
)
def test_readiness_rejects_incomplete_or_incompatible_records(monkeypatch, change):
    binding, native, cells, components, _ = readiness_fixture(monkeypatch)
    cell = cells[str(binding.parent / "noise_s42.manifest.json")]
    component = components[str(binding.parent / "private/noise_s42.manifest.json")]
    if change == "incomplete":
        native["metrics"]["completed_cells"] = 2
    elif change == "missing_cell":
        native["outputs"].pop(2)
    elif change == "cpu":
        native["parameters"]["device"] = "cpu"
    elif change == "wrong_permutation":
        cell["parameters"]["permutation_seed"] = 1
    elif change == "wrong_seed":
        cell["parameters"]["seed"] = 1
    elif change == "unbound_component":
        cell["outputs"] = cell["outputs"][:1]
    else:
        component["outputs"] = []
    with pytest.raises(ValueError):
        evaluator.readiness(binding)
