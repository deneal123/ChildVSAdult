"""Synthetic CPU contracts only: no pretrained model, native data or CUDA run."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.strong_validation_telemetry_v1 import (
    ValidationCollector,
    installed_validation_collector,
)


class Pairs(Dataset):
    def __init__(self, weights=(1, 1, 1, 1, 1)):
        self.labels = [1, 0, 1, 0, 1]
        self.weights = weights
        self.cosines = [0.8, 0.7, 0.2, 0.1, 0.5]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        c = self.cosines[idx]
        return (
            torch.tensor([1.0, 0.0]),
            torch.tensor([c, np.sqrt(1 - c * c)], dtype=torch.float32),
            torch.tensor(float(self.labels[idx])),
            torch.tensor(float(self.weights[idx])),
        )


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value


def test_same_auc_rng_forward_count_and_next_shuffle():
    trainer = importlib.import_module("age_gap.training.finetune")
    ds = Pairs()
    torch.manual_seed(1234)
    original = Tiny()
    expected_auc = trainer._val_auc(original, ds, "cpu", 2)
    expected_state = torch.get_rng_state().clone()
    expected_next = list(DataLoader(list(range(17)), batch_size=17, shuffle=True))[0]
    torch.manual_seed(1234)
    measured = Tiny()
    collector = ValidationCollector()
    observed_auc = collector(measured, ds, "cpu", 2)
    assert torch.equal(torch.get_rng_state(), expected_state)
    observed_next = list(DataLoader(list(range(17)), batch_size=17, shuffle=True))[0]
    assert torch.equal(observed_next, expected_next)
    assert observed_auc == expected_auc
    assert measured.calls == original.calls == 6
    assert not measured.training


def test_global_weight_mass_not_mean_of_batch_means():
    ds = Pairs(weights=(1, 4, 2, 3, 8))
    collector = ValidationCollector()
    collector(Tiny(), ds, "cpu", 2)
    result = collector.records[0]
    assert result["validation_denominator"] == 18
    assert result["validation_pairs"] == 5
    assert result["validation_numerator"] == pytest.approx(7.4, abs=1e-6)
    assert result["validation_loss"] == pytest.approx(7.4 / 18, abs=1e-7)
    batch_mean_average = (1.8 / 5 + 1.6 / 5 + 4 / 8) / 3
    assert result["validation_loss"] != pytest.approx(batch_mean_average)


def test_unit_weights_match_pinned_training_loss_convention():
    collector = ValidationCollector()
    collector(Tiny(), Pairs(), "cpu", 2)
    assert collector.records[0]["validation_loss"] == pytest.approx(1.9 / 5, abs=1e-7)


def test_epoch_sequence_separate_from_component_history():
    collector = ValidationCollector()
    collector(Tiny(), Pairs(), "cpu", 2)
    collector(Tiny(), Pairs(), "cpu", 2)
    assert [r["epoch"] for r in collector.records] == [1, 2]
    assert set(collector.records[0]) == {
        "epoch",
        "validation_loss",
        "validation_numerator",
        "validation_denominator",
        "validation_pairs",
        "validation_auc",
    }


@pytest.mark.parametrize("margin", [True, 0, 0.2, float("nan")])
def test_invalid_margin(margin):
    with pytest.raises(ValueError, match="margin"):
        ValidationCollector(margin)


@pytest.mark.parametrize("weights", [(0, 1, 1, 1, 1), (-1, 1, 1, 1, 1), (float("nan"), 1, 1, 1, 1)])
def test_bad_weights_no_record(weights):
    collector = ValidationCollector()
    with pytest.raises(ValueError, match="weights"):
        collector(Tiny(), Pairs(weights), "cpu", 2)
    assert collector.records == []


@pytest.mark.parametrize("batch", [0, True, 1.5])
def test_invalid_batch(batch):
    with pytest.raises(ValueError, match="batch"):
        ValidationCollector()(Tiny(), Pairs(), "cpu", batch)


def test_label_metadata_drift_refused():
    ds = Pairs()
    original = ds.__class__.__getitem__

    class Drift(Pairs):
        def __getitem__(self, idx):
            a, b, y, w = original(self, idx)
            return a, b, 1 - y, w

    collector = ValidationCollector()
    with pytest.raises(ValueError, match="label order"):
        collector(Tiny(), Drift(), "cpu", 2)
    assert not collector.records


def test_context_restores_on_success_and_error():
    original = object()
    trainer = SimpleNamespace(_val_auc=original)
    collector = ValidationCollector()
    with installed_validation_collector(trainer, collector):
        assert trainer._val_auc is collector
    assert trainer._val_auc is original
    with pytest.raises(RuntimeError), installed_validation_collector(trainer, collector):
        raise RuntimeError("synthetic failure")
    assert trainer._val_auc is original


def test_used_collector_not_installed():
    collector = ValidationCollector()
    collector(Tiny(), Pairs(), "cpu", 2)
    with (
        pytest.raises(ValueError, match="fresh"),
        installed_validation_collector(SimpleNamespace(_val_auc=object()), collector),
    ):
        pass


def test_extra_default_validation_iterator_changes_rng_but_explicit_generator_does_not():
    torch.manual_seed(123)
    initial = torch.get_rng_state().clone()
    list(DataLoader([1], batch_size=1, shuffle=False, num_workers=0))
    assert not torch.equal(initial, torch.get_rng_state())
    torch.set_rng_state(initial)
    list(
        DataLoader(
            [1],
            batch_size=1,
            shuffle=False,
            num_workers=0,
            generator=torch.Generator().manual_seed(1),
        )
    )
    assert torch.equal(initial, torch.get_rng_state())


def test_real_module_seam_installed_and_restored():
    trainer = importlib.import_module("age_gap.training.finetune")
    original = trainer._val_auc
    collector = ValidationCollector()
    with installed_validation_collector(trainer, collector):
        assert trainer.finetune.__globals__["_val_auc"] is collector
        trainer._val_auc(Tiny(), Pairs(), "cpu", 2)
        assert len(collector.records) == 1
    assert trainer._val_auc is original
    assert trainer.finetune.__globals__["_val_auc"] is original


def test_single_class_fails_before_iterator_or_forward():
    ds = Pairs()
    ds.labels = [1] * 5
    model = Tiny()
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="two-class"):
        ValidationCollector()(model, ds, "cpu", 2)
    assert model.calls == 0 and torch.equal(before, torch.get_rng_state())


def test_tiny_dropout_training_trajectory_exact_on_cpu():
    """Tiny generated arrays only; not native weights or a CUDA guarantee."""
    trainer = importlib.import_module("age_gap.training.finetune")

    class SyntheticNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.4))

        def forward(self, value):
            return torch.nn.functional.normalize(self.net(value), dim=1)

    def run(validation):
        torch.manual_seed(987)
        model = SyntheticNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = trainer.ContrastivePairLoss(margin=0.3)
        loader = DataLoader(Pairs(), batch_size=2, shuffle=True)
        aucs = []
        for _ in range(3):
            model.train()
            for a, b, y, w in loader:
                optimizer.zero_grad()
                loss_fn(model(a), model(b), y, weights=w).backward()
                optimizer.step()
            aucs.append(validation(model, Pairs(), "cpu", 2))
        return model.state_dict(), torch.get_rng_state().clone(), aucs

    expected, expected_rng, expected_auc = run(trainer._val_auc)
    collector = ValidationCollector()
    actual, actual_rng, actual_auc = run(collector)
    assert actual_auc == expected_auc
    assert torch.equal(actual_rng, expected_rng)
    assert all(torch.equal(actual[key], expected[key]) for key in expected)
    assert len(collector.records) == 3
