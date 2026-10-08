"""CPU synthetic producer guards and objective parity; not a native GPU run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from age_gap.training.losses import ContrastivePairLoss
from scripts import run_facenet_validation_cuda_v2 as producer

spec = importlib.util.spec_from_file_location(
    "facenet_validation_cuda_v2_reload", producer.__file__
)


class EmbeddingDataset(torch.utils.data.Dataset):
    """In-memory (emb_a, emb_b, label, weight) rows; no file IO."""

    def __init__(self, a, b, y, w):
        self.a = torch.as_tensor(a, dtype=torch.float32)
        self.b = torch.as_tensor(b, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.w = torch.as_tensor(w, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.a[i], self.b[i], self.y[i], self.w[i]


class ScaledModel(nn.Module):
    """Parameterised identity on embeddings so optimizers have something to update."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.scale


def _manual_objective(za, zb, y, w, margin):
    cos = (za * zb).sum(-1)
    per_pair = y * (1 - cos) + (1 - y) * np.maximum(cos - margin, 0.0)
    return per_pair, cos


def test_new_experiment_type_and_fixed_budget():
    assert producer.EXPERIMENT == "facenet-fixed8-validation-loss-v2"
    assert producer.EXPERIMENT != "pair-contrastive-backbone-finetune"
    b = producer.FIXED_BUDGET
    assert (b["epochs_requested"], b["epochs_executed"], b["selected_epoch"]) == (8, 8, 8)
    assert b["checkpoint_selection"] == "last_epoch"
    assert b["batchnorm_policy"] == "frozen_all"
    assert b["gap_weight"] == 0.0 and b["margin"] == 0.3
    # Root protocol choice: closest to the completed strong head lowLR cell.
    assert b["batch_size"] == 16 and b["learning_rate"] == 1e-6
    assert producer.SEEDS == (42, 1, 2)


def test_pair_objective_matches_formula():
    rng = np.random.default_rng(0)
    za = rng.normal(size=(7, 4)).astype(np.float32)
    zb = rng.normal(size=(7, 4)).astype(np.float32)
    y = np.array([1, 0, 1, 0, 1, 0, 1], np.float32)
    w = np.array([1, 2, 1, 1, 3, 1, 1], np.float32)
    per_pair, cos = producer.pair_objective(
        torch.from_numpy(za), torch.from_numpy(zb),
        torch.from_numpy(y), torch.from_numpy(w), margin=0.3,
    )
    exp_pair, exp_cos = _manual_objective(za, zb, y, w, 0.3)
    assert np.allclose(cos.numpy(), exp_cos, atol=1e-6)
    assert np.allclose(per_pair.numpy(), exp_pair, atol=1e-6)


# --------------------------------------------------------------------------------------
# Optimizer objective parity with production ContrastivePairLoss (weighted MEAN, not SUM)
# --------------------------------------------------------------------------------------
def test_loss_and_grad_parity_vs_production_unequal_weights_partial_batch():
    """Identical initial model, unequal weights, partial batch of 4 rows."""
    rng = np.random.default_rng(7)
    za = torch.from_numpy(rng.normal(size=(4, 4)).astype(np.float32))
    zb = torch.from_numpy(rng.normal(size=(4, 4)).astype(np.float32))
    y = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    w = torch.tensor([1.0, 2.0, 0.5, 3.0], dtype=torch.float32)
    assert float(w.sum()) != 1.0

    def via_producer():
        m = ScaledModel()
        loss, _per, _cos = producer.batch_step_loss(
            m, za, zb, y, w, ContrastivePairLoss(margin=0.3)
        )
        loss.backward()
        return float(loss.detach()), m.scale.grad.detach().clone()

    def via_production():
        m = ScaledModel()
        loss = ContrastivePairLoss(margin=0.3)(m(za), m(zb), y, weights=w)
        loss.backward()
        return float(loss.detach()), m.scale.grad.detach().clone()

    p_loss, p_grad = via_producer()
    r_loss, r_grad = via_production()
    assert p_loss == pytest.approx(r_loss, abs=1e-7)
    assert torch.allclose(p_grad, r_grad, atol=1e-7)

    # The old SUM objective must be detectably different (this was the blocker).
    m = ScaledModel()
    per_pair, _ = producer.pair_objective(m(za), m(zb), y, w, 0.3)
    ((per_pair * w).sum()).backward()
    sum_grad = m.scale.grad.detach().clone()
    assert not torch.allclose(p_grad, sum_grad, atol=1e-6)
    assert torch.allclose(sum_grad, p_grad * float(w.sum()), atol=1e-5)


def test_train_epoch_matches_production_reference_on_shuffled_partial_batches():
    """5 rows, batch_size=3 -> partial last batch; replay DataLoader's seeded permutation."""
    rng = np.random.default_rng(11)
    n, bs = 5, 3
    ds = EmbeddingDataset(
        rng.normal(size=(n, 4)), rng.normal(size=(n, 4)),
        [1, 0, 1, 0, 1], [1.0, 2.0, 1.0, 3.0, 0.5],
    )
    # Mirror the producer loader exactly (same generator + shuffle) to get batch order.
    batches = list(
        DataLoader(
            ds, batch_size=bs, shuffle=True,
            generator=torch.Generator().manual_seed(0),
        )
    )
    assert len(batches) == 2 and len(batches[-1][0]) == 2  # partial last batch of 2

    m_ref = ScaledModel()
    opt_ref = torch.optim.SGD(m_ref.parameters(), lr=1e-2)
    loss_fn = ContrastivePairLoss(margin=0.3)
    for ta, tb, y, w in batches:
        opt_ref.zero_grad()
        loss_fn(m_ref(ta), m_ref(tb), y, weights=w).backward()
        opt_ref.step()

    m_prod = ScaledModel()
    producer.train_epoch(
        m_prod, ds, optimizer=torch.optim.SGD(m_prod.parameters(), lr=1e-2),
        loss_fn=ContrastivePairLoss(margin=0.3), device="cpu", batch_size=bs,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.allclose(m_prod.scale.detach(), m_ref.scale.detach(), atol=1e-6)

    # SUM-objective reference lands elsewhere (detects the fixed blocker end to end).
    m_sum = ScaledModel()
    opt_sum = torch.optim.SGD(m_sum.parameters(), lr=1e-2)
    for ta, tb, y, w in batches:
        opt_sum.zero_grad()
        per_pair, _ = producer.pair_objective(m_sum(ta), m_sum(tb), y, w, 0.3)
        ((per_pair * w).sum()).backward()
        opt_sum.step()
    assert not torch.allclose(m_prod.scale.detach(), m_sum.scale.detach(), atol=1e-6)


def test_train_epoch_global_denominator_and_finite_diagnostics():
    a = np.array([[1, 0, 0, 0]] * 4, np.float32)
    b = np.array(
        [[0.5, 0, 0, 0], [0.8, 0, 0, 0], [0.2, 0, 0, 0], [0.9, 0, 0, 0]], np.float32,
    )
    ds = EmbeddingDataset(a, b, [1, 0, 1, 0], [1.0, 1.0, 2.0, 1.0])
    model = ScaledModel()
    out = producer.train_epoch(
        model, ds, optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        loss_fn=ContrastivePairLoss(margin=0.3), device="cpu", batch_size=2,
        generator=torch.Generator().manual_seed(0),
    )
    assert out["train_denominator"] == pytest.approx(5.0)
    assert np.isfinite(out["train_loss"]) and np.isfinite(out["mean_gradient_norm"])
    assert out["mean_gradient_norm"] > 0
    assert out["train_batches"] == 2


def test_validation_loss_is_global_not_mean_of_batch_means():
    a = np.eye(4, dtype=np.float32)[[0, 1, 2, 3, 0]]
    b = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [-1, 0, 0, 0],
         [0, -1, 0, 0], [0.2, 0.9, 0, 0]], np.float32,
    )
    y = [1, 0, 1, 0, 1]
    w = [1.0, 2.0, 1.0, 3.0, 1.0]
    ds = EmbeddingDataset(a, b, y, w)
    out = producer.evaluate_epoch(
        nn.Identity(), ds, device="cpu", batch_size=2, margin=0.3
    )
    exp_pair, _ = _manual_objective(a, b, np.array(y, np.float32), np.array(w), 0.3)
    wv = np.array(w)
    expected_global = float((exp_pair * wv).sum() / wv.sum())
    batch_means = []
    for i in range(0, 5, 2):
        sl = slice(i, min(i + 2, 5))
        batch_means.append(float((exp_pair[sl] * wv[sl]).sum() / wv[sl].sum()))
    wrong = float(np.mean(batch_means))
    assert out["validation_denominator"] == pytest.approx(float(wv.sum()))
    assert out["validation_loss"] == pytest.approx(expected_global, abs=1e-6)
    assert abs(out["validation_loss"] - wrong) > 1e-4


def test_evaluate_epoch_fails_closed_on_nonfinite_or_out_of_range_auc(monkeypatch):
    import age_gap.evaluation.metrics as metrics

    ds = EmbeddingDataset(np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32), [1, 0, 1, 0], [1, 1, 1, 1])
    for bad in (float("nan"), float("inf"), -0.1, 1.5):
        monkeypatch.setattr(metrics, "roc_auc", lambda *a, _b=bad, **k: _b)
        with pytest.raises(RuntimeError, match="invalid validation_auc|undefined"):
            producer.evaluate_epoch(nn.Identity(), ds, device="cpu", batch_size=2, margin=0.3)
    monkeypatch.setattr(metrics, "roc_auc", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    with pytest.raises(RuntimeError, match="undefined"):
        producer.evaluate_epoch(nn.Identity(), ds, device="cpu", batch_size=2, margin=0.3)


def test_validate_pairs_rejects_invalid_weights_and_labels():
    good_y, good_w = torch.tensor([1.0, 0.0]), torch.tensor([1.0, 2.0])
    producer.validate_pairs(good_y, good_w)
    for bad_w in (torch.tensor([-1.0, 2.0]), torch.tensor([0.0, 0.0]),
                  torch.tensor([float("nan"), 1.0]), torch.tensor([float("inf"), 1.0])):
        with pytest.raises(ValueError, match="weights"):
            producer.validate_pairs(good_y, bad_w)
    for bad_y in (torch.tensor([1.0, 2.0]), torch.tensor([float("nan"), 0.0])):
        with pytest.raises(ValueError, match="labels"):
            producer.validate_pairs(bad_y, good_w)


def test_require_cuda_fails_closed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA required"):
        producer.require_cuda()


def test_import_does_not_probe_gpu(monkeypatch):
    """Re-executing the module must not touch CUDA (no import-time device probe)."""
    calls = []

    def boom():
        calls.append(True)
        raise AssertionError("import-time CUDA probe")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert calls == []


def test_seed_everything_is_reproducible():
    g1 = producer.seed_everything(42)
    d1 = (random.random(), float(np.random.rand()), torch.rand(1).item())
    g2 = producer.seed_everything(42)
    d2 = (random.random(), float(np.random.rand()), torch.rand(1).item())
    assert d1 == d2
    assert torch.randperm(5, generator=g1).tolist() == torch.randperm(5, generator=g2).tolist()


# --------------------------------------------------------------------------------------
# Device-free ancestry / checkpoint fail-closed helpers (tiny fake files, not real hashes)
# --------------------------------------------------------------------------------------
def test_pretrained_weight_resolution_and_missing_fails_closed(tmp_path):
    weight = producer.resolve_facenet_casia_weight(tmp_path)
    assert weight == tmp_path / "checkpoints" / producer.CASIA_WEIGHT_NAME
    assert not weight.exists()
    with pytest.raises(FileNotFoundError, match="refusing implicit download"):
        producer.require_pretrained_weight(weight)
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"fake-weights")
    assert producer.require_pretrained_weight(weight) == weight


def test_weight_record_must_match_inventory(tmp_path):
    weight = tmp_path / "casia.pt"
    weight.write_bytes(b"fake-weights")
    good = {
        "path": str(weight),
        "bytes": weight.stat().st_size,
        "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
    }
    assert producer.validate_weight_record(weight, good)["sha256"] == good["sha256"]
    for key, bad in (("bytes", -1), ("sha256", "0" * 64), ("path", str(tmp_path / "other.pt"))):
        mutated = dict(good)
        mutated[key] = bad
        with pytest.raises(RuntimeError, match="mismatch|differs"):
            producer.validate_weight_record(weight, mutated)


def test_inventory_weight_record_reads_facenet_casia(tmp_path):
    inv = tmp_path / "model_inventory.json"
    inv.write_text(json.dumps({"models": {"facenet_casia": {"artifact": {"sha256": "x"}}}}))
    assert producer.inventory_weight_record(inv)["sha256"] == "x"
    inv.write_text(json.dumps({"models": {}}))
    with pytest.raises(RuntimeError, match="lacks facenet_casia"):
        producer.inventory_weight_record(inv)


def test_bind_ancestry_detects_fake_source_and_weight_mutation(tmp_path):
    weight = tmp_path / "casia.pt"
    weight.write_bytes(b"weight-one")
    source = tmp_path / "facenet_module.py"
    source.write_text("VALUE = 1\n")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("{}\n")
    inputs = [weight, source, pairs]
    before = producer.bind_ancestry(inputs, "init-digest")
    assert before == producer.bind_ancestry(inputs, "init-digest")
    assert before["input_count"] == 3
    assert before["model_init_state_sha256"] == "init-digest"
    source.write_text("VALUE = 2\n")
    assert before != producer.bind_ancestry(inputs, "init-digest")
    weight.write_bytes(b"weight-two")
    assert before != producer.bind_ancestry(inputs, "init-digest")


def test_source_ancestry_paths_includes_weight_inventory_and_facenet(tmp_path):
    weight = tmp_path / "casia.pt"
    weight.write_bytes(b"w")
    inv = tmp_path / "inv.json"
    inv.write_text("{}")
    script = tmp_path / "s.py"
    script.write_text("# script\n")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("{}\n")
    crop = tmp_path / "c.jpg"
    crop.write_bytes(b"c")
    paths = producer.source_ancestry_paths(
        script=script, weight=weight, inventory=inv, pairs=pairs, crops=[crop],
        root=producer.PROJECT_ROOT,
    )
    resolved = {str(p) for p in paths}
    assert str(script.resolve()) in resolved
    assert str(weight.resolve()) in resolved
    assert str(inv.resolve()) in resolved
    assert str(pairs.resolve()) in resolved
    assert str(crop.resolve()) in resolved
    assert any("facenet_pytorch" in p for p in resolved)
    assert len(resolved) == len(paths)  # deduplicated


def test_state_dict_digest_order_independent_and_value_sensitive():
    s1 = {"a": torch.ones(2, 2), "b": torch.zeros(1)}
    s2 = {"b": torch.zeros(1), "a": torch.ones(2, 2)}
    assert producer.state_dict_digest(s1) == producer.state_dict_digest(s2)
    s3 = {"a": torch.ones(2, 2), "b": torch.zeros(1) + 1}
    assert producer.state_dict_digest(s1) != producer.state_dict_digest(s3)


def _fake_checkpoint(path, epochs=8, bad_tensor=False):
    history = [
        {"epoch": e, "train_loss": 0.5, "validation_loss": 0.4, "validation_auc": 0.7,
         "mean_gradient_norm": 0.1, "epoch_seconds": 1.0}
        for e in range(1, epochs + 1)
    ]
    state = {"w": torch.ones(2, 2) if not bad_tensor else torch.full((2, 2), float("nan"))}
    torch.save({"state_dict": state, "history": history, "selected_epoch": epochs}, path)


def test_verify_completed_checkpoint_fails_closed(tmp_path):
    good = tmp_path / "good.pt"
    _fake_checkpoint(good, 8)
    producer.verify_completed_checkpoint(good, 8)  # no raise
    short = tmp_path / "short.pt"
    _fake_checkpoint(short, 3)
    with pytest.raises(RuntimeError, match="ordered epochs|selected_epoch"):
        producer.verify_completed_checkpoint(short, 8)
    nan = tmp_path / "nan.pt"
    _fake_checkpoint(nan, 8, bad_tensor=True)
    with pytest.raises(RuntimeError, match="non-finite checkpoint tensor"):
        producer.verify_completed_checkpoint(nan, 8)


def test_label_guard_does_not_create_a_cpu_reference_tensor(monkeypatch):
    labels, weights = torch.tensor([0., 1.]), torch.tensor([1., 1.])

    def forbidden(*args, **kwargs):
        raise AssertionError("label guard must not allocate a CPU reference tensor")

    monkeypatch.setattr(torch, "tensor", forbidden)
    producer.validate_pairs(labels, weights)


@pytest.mark.parametrize("labels,weights", [
    (torch.tensor([[0., 1.]]), torch.tensor([1., 1.])),
    (torch.tensor([0., 1.]), torch.tensor([1.])),
    (torch.tensor([]), torch.tensor([])),
])
def test_label_weight_vectors_must_align(labels, weights):
    with pytest.raises(ValueError, match="aligned"):
        producer.validate_pairs(labels, weights)


def test_finite_weights_with_overflowing_sum_rejected():
    labels = torch.tensor([0., 1.])
    weights = torch.full((2,), torch.finfo(torch.float32).max)
    with pytest.raises(ValueError, match="finite positive total mass"):
        producer.validate_pairs(labels, weights)


def mocked_run(tmp_path, monkeypatch):
    """Fake CUDA hooks/epochs, real tiny ancestry/checkpoint IO: not GPU evidence."""
    from argparse import Namespace

    from age_gap.models import backbones
    from age_gap.training import finetune

    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"fake-initial-weight")
    inventory = tmp_path / "metrics/model_inventory.json"
    inventory.parent.mkdir()
    inventory.write_text(json.dumps({"models": {"facenet_casia": {
        "artifact": producer.file_record(weight)}}}), encoding="utf-8")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("{}\n", encoding="utf-8")
    source = tmp_path / "source.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    inputs = [source, weight, inventory, pairs]

    class FakeBackbone(nn.Module):
        trainable_scopes = {"head": ("weight", "bias")}

        def __init__(self):
            super().__init__()
            self.net = nn.Linear(2, 2)

        def to(self, device):
            assert device == "cuda"
            return self  # deliberately no real CUDA tensor or device initialization

    class FakeDataset:
        _items = [(pairs, pairs)]

        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 2

    monkeypatch.setattr(producer, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(producer, "resolve_facenet_casia_weight", lambda: weight)
    monkeypatch.setattr(producer, "source_ancestry_paths", lambda **kwargs: inputs)
    monkeypatch.setattr(producer, "seed_everything", lambda seed: torch.Generator().manual_seed(seed))
    monkeypatch.setattr(backbones, "make_backbone", lambda *args, **kwargs: FakeBackbone())
    monkeypatch.setattr(finetune, "ImagePairDataset", FakeDataset)
    monkeypatch.setattr(finetune, "_bb_prep", lambda model: None)
    monkeypatch.setattr(producer, "train_epoch", lambda *args, **kwargs: {
        "train_loss": .2, "train_numerator": .4, "train_denominator": 2.,
        "mean_gradient_norm": .1, "train_batches": 1, "train_pairs": 2})
    monkeypatch.setattr(producer, "evaluate_epoch", lambda *args, **kwargs: {
        "validation_loss": .3, "validation_numerator": .6,
        "validation_denominator": 2., "validation_auc": .75, "validation_pairs": 2})
    for name, function in {
        "is_available": lambda: True,
        "device_count": lambda: 1,
        "reset_peak_memory_stats": lambda: None,
        "max_memory_allocated": lambda: 0,
        "max_memory_reserved": lambda: 0,
        "synchronize": lambda: None,
        "get_device_name": lambda: "mock-only-not-a-real-GPU",
    }.items():
        monkeypatch.setattr(torch.cuda, name, function)
    return Namespace(out=tmp_path / "out", seed=42, lr=1e-6, pairs=str(pairs)), source


def test_mocked_run_binds_original_inputs_and_measured_epoch_fields(tmp_path, monkeypatch):
    args, source = mocked_run(tmp_path, monkeypatch)
    original = producer.file_record(source)
    manifest = producer.run(args)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["inputs"][0] == original
    assert len(payload["metrics"]["history"]) == 8
    assert all(row["validation_loss"] == .3 for row in payload["metrics"]["history"])
    assert payload["metrics"]["evaluation_complete"] is False
    assert payload["metrics"]["publication_ready"] is False


def test_mutation_during_manifest_write_rejected_against_initial_snapshot(tmp_path, monkeypatch):
    args, source = mocked_run(tmp_path, monkeypatch)
    writer = producer.write_experiment_manifest

    def changed_writer(path, **kwargs):
        source.write_text("VERSION = 2\n", encoding="utf-8")
        return writer(path, **kwargs)

    monkeypatch.setattr(producer, "write_experiment_manifest", changed_writer)
    with pytest.raises(RuntimeError, match="inputs changed"):
        producer.run(args)
    assert not (args.out / "summary.manifest.json").exists()
