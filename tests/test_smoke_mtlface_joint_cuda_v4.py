"""Synthetic CPU guards for the CUDA-only joint smoke producer (cuda-smoke-v4).

No GPU, no optimizer step, no real weights/faces/inference anywhere: tiny fake modules,
a recording ``FakeCuda`` seam and fake stage functions with refusal optimizers. Passing
these tests is NOT native CUDA smoke evidence.
"""

# ruff: noqa: E402

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT / "scripts" / "smoke_mtlface_joint_cuda_v4.py"

from age_gap.common.manifest import file_record
from scripts.smoke_mtlface_joint_cuda_v4 import (
    AGE_BOUNDARIES,
    EXPERIMENT,
    build_selection,
    run_joint_smoke,
    run_smoke,
)
from scripts.smoke_mtlface_joint_cuda_v4 import (
    main as smoke_main,
)


class _NoStep:
    def __init__(self):
        self.steps = 0

    def step(self, *args, **kwargs):
        self.steps += 1
        raise AssertionError("no optimizer step is allowed in synthetic tests")


class FakeCuda:
    """Recording CUDA seam; never touches a GPU."""

    def __init__(self):
        self.calls: list[tuple] = []

    def require(self):
        self.calls.append(("require",))

    def seed(self, seed):
        self.calls.append(("seed", seed))
        return seed

    def device(self):
        return "cuda"

    def transfer(self, value, device):
        self.calls.append(("transfer", type(value).__name__, device))
        return value

    def reset_peak(self):
        self.calls.append(("reset_peak",))

    def peaks(self):
        return dict(peak_allocated_bytes=111, peak_reserved_bytes=222)

    def binding(self):
        return dict(device="cuda", gpu_name="fake-gpu", torch_version="0.0", cuda_version="0.0")

    def synchronize(self):
        self.calls.append(("synchronize",))


# --------------------------------------------------------------------------------------
# Tiny modules
# --------------------------------------------------------------------------------------
class TinyAdapter(nn.Module):
    def __init__(self, *, nan=False):
        super().__init__()
        self.stem = nn.Conv2d(3, 3, 1)
        self.norm = nn.BatchNorm2d(3)
        self.head = nn.Linear(3, 4)
        self.nan = nan
        self.policy = None

    def preprocess(self, image):
        return image

    def set_batchnorm_policy(self, policy="frozen_all"):
        if policy != "frozen_all":
            raise ValueError("frozen BN required")
        self.policy = policy
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def forward(self, images):
        maps = self.stem(images)
        pooled = maps.mean((2, 3))
        if self.nan:
            return pooled * float("nan")
        return pooled


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3, 4))

    @property
    def classes(self):
        return 4


class TinyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(3, 3, 1)


class TinyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 1)


class TinyDataset:
    def __init__(self, records):
        self.records = records
        self.n_classes = 4

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        age = 0 if row.age is None else row.age
        return torch.full((3, 4, 4), float(index + 1)), index, age


def _records(*ages):
    return tuple(SimpleNamespace(age=age) for age in ages)


def _fake_stages(record, *, nan=False):
    def recognition_step_fn(model, head, optimizer, batch):
        record.setdefault("order", []).append("recognition")
        assert isinstance(optimizer, _NoStep)
        images, identities, ages = batch
        with torch.no_grad():
            for value in list(model.parameters()) + list(head.parameters()):
                value.add_(0.05)
        return dict(total=1.0, gradient_norm=0.5)

    def fas_step_fn(
        model,
        generator,
        discriminator,
        g_opt,
        d_opt,
        images,
        targets,
        groups,
        *,
        generator_bn_policy,
    ):
        record.setdefault("order", []).append("fas")
        assert isinstance(g_opt, _NoStep) and isinstance(d_opt, _NoStep)
        assert generator_bn_policy == "adapt"
        with torch.no_grad():
            generator.conv3.weight.add_(1.0)
            discriminator.conv1.weight.add_(1.0)
        return dict(discriminator_loss=1.0, generator_loss=1.0, generator_updated=True)

    return recognition_step_fn, fas_step_fn


def _no_step_optimizers(model, head, generator, discriminator):
    return _NoStep(), _NoStep(), _NoStep()


@pytest.fixture
def fake(monkeypatch):
    import scripts.smoke_mtlface_joint_cuda_v4 as module

    monkeypatch, module = monkeypatch, module
    calls: list[str] = []

    def no_op_assert(modules, device):
        calls.append("assert_on_device")
        assert device == "cuda"
        assert set(modules) == {"model", "identity_head", "generator", "discriminator"}

    monkeypatch.setattr(module, "_assert_on_device", no_op_assert)

    def fake_verify(tensors, device):
        calls.append("verify_tensors")
        assert device == "cuda" and len(tuple(tensors)) == 5

    monkeypatch.setattr(module, "_verify_tensor_devices", fake_verify)
    record: dict = {}
    recognition_step_fn, fas_step_fn = _fake_stages(record)
    runtime = FakeCuda()
    dataset = TinyDataset(_records(0, 11, None))
    return dict(
        module=module,
        calls=calls,
        record=record,
        runtime=runtime,
        dataset=dataset,
        recognition_step_fn=recognition_step_fn,
        fas_step_fn=fas_step_fn,
    )


# --------------------------------------------------------------------------------------
# Import / dry run
# --------------------------------------------------------------------------------------
def test_import_does_not_probe_gpu(monkeypatch):
    calls = []

    def boom():
        calls.append(True)
        raise AssertionError("import-time CUDA probe")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    spec = importlib.util.spec_from_file_location("smoke_cuda_v4_reload", PRODUCER)
    fresh = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fresh
    try:
        spec.loader.exec_module(fresh)
    finally:
        sys.modules.pop(spec.name, None)
    assert calls == []


def test_dry_run_touches_nothing(tmp_path, monkeypatch, capsys):
    import scripts.smoke_mtlface_joint_cuda_v4 as module

    def boom(*args, **kwargs):
        raise AssertionError("dry-run must not probe or read")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    monkeypatch.setattr(module, "load_prerequisites", boom)
    out = tmp_path / "out"
    smoke_main(
        [
            "--faces",
            str(tmp_path / "f.json"),
            "--initialization",
            str(tmp_path / "i.json"),
            "--smoke",
            str(tmp_path / "s.json"),
            "--weights",
            str(tmp_path / "w.bin"),
            "--out",
            str(out),
        ]
    )
    assert "dry-run" in capsys.readouterr().out
    assert not out.exists()


# --------------------------------------------------------------------------------------
# Fail-closed CUDA
# --------------------------------------------------------------------------------------
def test_cuda_unavailable_refused_before_models(tmp_path, monkeypatch):
    import scripts.smoke_mtlface_joint_cuda_v4 as module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def refuse(*args, **kwargs):
        raise AssertionError("no model may be constructed without CUDA")

    monkeypatch.setattr(module, "load_prerequisites", refuse)
    with pytest.raises(RuntimeError, match="CUDA required"):
        run_smoke(
            initialization_path=tmp_path / "i.json",
            faces_path=tmp_path / "f.json",
            smoke_path=tmp_path / "s.json",
            weights=tmp_path / "w.bin",
            out=tmp_path / "out",
            model_factory=refuse,
        )
    assert not (tmp_path / "out").exists()


def test_zero_devices_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="no device is visible"):
        run_smoke(
            initialization_path=tmp_path / "i.json",
            faces_path=tmp_path / "f.json",
            smoke_path=tmp_path / "s.json",
            weights=tmp_path / "w.bin",
            out=tmp_path / "out",
        )


# --------------------------------------------------------------------------------------
# Exact selection + refusal
# --------------------------------------------------------------------------------------
def test_selection_is_first_explicit_and_first_missing():
    dataset = TinyDataset(_records(None, 5, 41, None))
    indices, images, identities, ages, targets, groups = build_selection(dataset)
    assert indices == [1, 0]
    assert images.shape == (2, 3, 4, 4) and targets.shape == (2, 3, 4, 4)
    assert torch.equal(targets[0], targets[1]) and torch.equal(targets[0], images[0])
    assert groups.tolist() == [0, 0]
    assert sum(boundary < 41 for boundary in AGE_BOUNDARIES) == 4
    assert ages.tolist() == [5, 0] and identities.tolist() == [1, 0]


def test_missing_age_stratum_refused():
    with pytest.raises(ValueError, match="missing age"):
        build_selection(TinyDataset(_records(10, 30)))
    with pytest.raises(ValueError, match="missing age"):
        build_selection(TinyDataset(_records(None, None)))


# --------------------------------------------------------------------------------------
# run_joint_smoke: ordering, device routing, immutability, frozen BN, NoStep
# --------------------------------------------------------------------------------------
def _run_real_smoke(fake, *, nan=False, model=None):
    model = model if model is not None else TinyAdapter(nan=nan)
    head = TinyHead()
    generator = TinyGenerator()
    discriminator = TinyDiscriminator()
    recognition_step_fn, fas_step_fn = _fake_stages(fake["record"], nan=nan)
    return run_joint_smoke(
        model,
        head,
        fake["dataset"],
        generator,
        discriminator,
        runtime=fake["runtime"],
        seed=42,
        optimizer_factory=_no_step_optimizers,
        recognition_step_fn=recognition_step_fn,
        fas_step_fn=fas_step_fn,
    )


def test_verify_tensor_devices_rejects_cpu_tensor():
    import scripts.smoke_mtlface_joint_cuda_v4 as module

    with pytest.raises(ValueError, match="not on cuda"):
        module._verify_tensor_devices((torch.zeros(1),), "cuda")


def test_run_joint_smoke_transfers_everything_before_optimizers(fake):
    result, indices = _run_real_smoke(fake)
    assert indices == [0, 2]
    assert fake["record"]["order"] == ["recognition", "fas"]
    calls = fake["runtime"].calls
    assert calls[0] == ("require",)
    assert ("seed", 42) in calls
    module_names = {call[1] for call in calls if call[0] == "transfer"}
    assert {
        "TinyAdapter",
        "TinyHead",
        "TinyGenerator",
        "TinyDiscriminator",
        "Tensor",
    } <= module_names
    assert all(call[2] == "cuda" for call in calls if call[0] == "transfer")
    assert len([c for c in calls if c[0] == "transfer" and c[1] == "Tensor"]) == 5
    assert fake["calls"] == ["assert_on_device", "verify_tensors"]
    assert result["recognition_buffers_frozen"] is True
    assert result["recognizer_state_unchanged_during_fas"] is True
    assert result["identity_head_unchanged_during_fas"] is True
    assert result["embeddings_finite"] is True
    assert result["training_complete"] is False
    assert result["scientific_evaluation_complete"] is False
    assert result["full_method_parity"] is False


def test_run_joint_smoke_rejects_cpu_batch_tensor(fake, monkeypatch):
    import scripts.smoke_mtlface_joint_cuda_v4 as module

    def refuse(tensors, device):
        raise ValueError("batch tensor not on cuda")

    monkeypatch.setattr(module, "_verify_tensor_devices", refuse)
    with pytest.raises(ValueError, match="not on cuda"):
        _run_real_smoke(fake)


def test_run_joint_smoke_refuses_nonfinite_embedding(fake):
    with pytest.raises(FloatingPointError, match="nonfinite post-joint embedding"):
        _run_real_smoke(fake, nan=True)


def test_frozen_recognition_buffer_change_refused(fake):
    def bad_recognition(model, head, optimizer, batch):
        with torch.no_grad():
            for value in model.parameters():
                value.add_(0.05)
            model.norm.running_mean.add_(1.0)  # buffer must stay frozen
        return dict(total=1.0, gradient_norm=0.5)

    model, head = TinyAdapter(), TinyHead()
    with pytest.raises(ValueError, match="frozen BN/buffer changed"):
        run_joint_smoke(
            model,
            head,
            fake["dataset"],
            TinyGenerator(),
            TinyDiscriminator(),
            runtime=fake["runtime"],
            seed=42,
            optimizer_factory=_no_step_optimizers,
            recognition_step_fn=bad_recognition,
            fas_step_fn=_fake_stages(fake["record"])[1],
        )


def test_recognition_no_update_refused(fake):
    def frozen_recognition(model, head, optimizer, batch):
        return dict(total=1.0, gradient_norm=0.0)

    with pytest.raises(ValueError, match="update required"):
        run_joint_smoke(
            TinyAdapter(),
            TinyHead(),
            fake["dataset"],
            TinyGenerator(),
            TinyDiscriminator(),
            runtime=fake["runtime"],
            seed=42,
            optimizer_factory=_no_step_optimizers,
            recognition_step_fn=frozen_recognition,
            fas_step_fn=_fake_stages(fake["record"])[1],
        )


def test_recognizer_change_during_fas_refused(fake):
    def bad_fas(
        model,
        generator,
        discriminator,
        g_opt,
        d_opt,
        images,
        targets,
        groups,
        *,
        generator_bn_policy,
    ):
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
            generator.conv3.weight.add_(1.0)
            discriminator.conv1.weight.add_(1.0)
        return dict(generator_updated=True)

    with pytest.raises(ValueError, match="recognizer state changed during FAS"):
        run_joint_smoke(
            TinyAdapter(),
            TinyHead(),
            fake["dataset"],
            TinyGenerator(),
            TinyDiscriminator(),
            runtime=fake["runtime"],
            seed=42,
            optimizer_factory=_no_step_optimizers,
            recognition_step_fn=_fake_stages(fake["record"])[0],
            fas_step_fn=bad_fas,
        )


def test_no_generator_update_refused(fake):
    def idle_fas(
        model,
        generator,
        discriminator,
        g_opt,
        d_opt,
        images,
        targets,
        groups,
        *,
        generator_bn_policy,
    ):
        return dict(generator_updated=False)

    with pytest.raises(ValueError, match="actual G/D parameter update required"):
        run_joint_smoke(
            TinyAdapter(),
            TinyHead(),
            fake["dataset"],
            TinyGenerator(),
            TinyDiscriminator(),
            runtime=fake["runtime"],
            seed=42,
            optimizer_factory=_no_step_optimizers,
            recognition_step_fn=_fake_stages(fake["record"])[0],
            fas_step_fn=idle_fas,
        )


def test_recognizer_bn_frozen_after_set_policy(fake):
    model = TinyAdapter()
    model.train()
    _run_real_smoke(fake, model=model)
    assert model.policy == "frozen_all"
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            assert not module.training


# --------------------------------------------------------------------------------------
# run_smoke: fail-closed ordering, fresh output, manifest, source drift
# --------------------------------------------------------------------------------------
def _smoke_kwargs(tmp_path, module, monkeypatch, runtime, record):
    paths = {
        name: tmp_path / f"{name}.json" for name in ("faces", "initialization", "smoke", "weights")
    }
    for path in paths.values():
        path.write_text("{}", encoding="utf-8")
    contract = dict(weights=file_record(paths["weights"]))
    init = dict(metrics=dict(identity_classes=4), inputs=[], outputs=[])
    faces = dict(inputs=[], outputs=[])
    smoke = dict(inputs=[], outputs=[])
    monkeypatch.setattr(module, "load_prerequisites", lambda **kw: (init, faces, smoke, contract))

    class Base:
        def __init__(self):
            self.backbone = nn.Identity()
            self.separation = nn.Identity()
            self.age_head = nn.Identity()
            self.age_adversary = nn.Identity()

    base = Base()

    def model_factory(weights, classes):
        record["calls"].append("model_factory")
        return base, TinyHead(), contract

    def adapter_factory(_base):
        record["calls"].append("adapter_factory")
        return TinyAdapter()

    def dataset_factory(model, faces_path):
        record["calls"].append("dataset_factory")
        return TinyDataset(_records(0, None)), faces

    def modules_factory():
        record["calls"].append("modules_factory")
        return TinyGenerator(), TinyDiscriminator()

    def smoke_runner(model, head, dataset, generator, discriminator, *, runtime, seed):
        record["calls"].append("smoke_runner")
        record["device"] = runtime.device()
        assert ("reset_peak",) in runtime.calls
        return (
            dict(
                recognition=dict(total=1.0),
                fas=dict(generator_updated=True),
                recognizer_state_unchanged_during_fas=True,
                identity_head_unchanged_during_fas=True,
                embeddings_finite=True,
                updated_recognition_parameter_tensors=3,
                updated_classifier_parameter_tensors=4,
                target_group=0,
                training_complete=False,
            ),
            [0, 1],
        )

    def require_fn():
        record["calls"].append("require")

    def seed_fn(seed):
        record["calls"].append(("seed", seed))

    return paths, dict(
        initialization_path=paths["initialization"],
        faces_path=paths["faces"],
        smoke_path=paths["smoke"],
        weights=paths["weights"],
        out=tmp_path / "smoke",
        runtime=runtime,
        model_factory=model_factory,
        adapter_factory=adapter_factory,
        dataset_factory=dataset_factory,
        modules_factory=modules_factory,
        smoke_runner=smoke_runner,
        require_fn=require_fn,
        seed_fn=seed_fn,
    )


def test_run_smoke_full_order_and_manifest(tmp_path, monkeypatch):
    import scripts.smoke_mtlface_joint_cuda_v4 as module

    record = dict(calls=[])
    paths, kwargs = _smoke_kwargs(tmp_path, module, monkeypatch, FakeCuda(), record)
    result = run_smoke(**kwargs)
    assert record["calls"][0] == "require"
    assert record["calls"].index("model_factory") > 0
    assert record["calls"].index("smoke_runner") > record["calls"].index("model_factory")
    assert ("seed", 42) in record["calls"]
    assert record["calls"].index(("seed", 42)) < record["calls"].index("model_factory")
    assert record["device"] == "cuda"
    assert result["cuda_smoke_complete"] is True and result["cuda_probed"] is True
    assert result["full_method_parity"] is False
    assert result["peak_allocated_bytes"] == 111 and result["peak_reserved_bytes"] == 222
    native = json.loads((kwargs["out"] / "summary.manifest.json").read_text())
    assert native["experiment"] == EXPERIMENT
    assert native["parameters"]["device"] == "cuda"
    assert native["parameters"]["implementation_mode"] == "injected-test"
    assert native["parameters"]["cuda_smoke_required"] is True
    assert native["parameters"]["native_cpu_prerequisite_device"] == "cpu"
    assert native["parameters"]["native_cpu_prerequisite_is_training_proof"] is False
    assert native["parameters"]["gpu_name"] == "fake-gpu"
    selection = json.loads((kwargs["out"] / "private" / "selection.json").read_text())
    assert selection == {"dataset_indices": [0, 1]}
    bound = {r["sha256"] for r in native["outputs"]}
    assert file_record(kwargs["out"] / "summary.json")["sha256"] in bound


def test_run_smoke_require_precedes_everything(tmp_path, monkeypatch):
    import scripts.smoke_mtlface_joint_cuda_v4 as module

    record = dict(calls=[])

    def refuse():
        record["calls"].append("require")
        raise RuntimeError("CUDA required; no CPU fallback")

    paths, kwargs = _smoke_kwargs(tmp_path, module, monkeypatch, FakeCuda(), record)
    kwargs["require_fn"] = refuse
    with pytest.raises(RuntimeError, match="CUDA required"):
        run_smoke(**kwargs)
    assert record["calls"] == ["require"]
    assert not kwargs["out"].exists()


def test_run_smoke_fresh_output_refused(tmp_path, monkeypatch):
    import scripts.smoke_mtlface_joint_cuda_v4 as module

    record = dict(calls=[])
    paths, kwargs = _smoke_kwargs(tmp_path, module, monkeypatch, FakeCuda(), record)
    kwargs["out"].mkdir()
    with pytest.raises(FileExistsError):
        run_smoke(**kwargs)
    assert record["calls"] == []


def test_run_smoke_rejects_non_cuda_binding(tmp_path, monkeypatch):
    import scripts.smoke_mtlface_joint_cuda_v4 as module

    record = dict(calls=[])
    runtime = FakeCuda()
    runtime.binding = lambda: dict(device="cpu", gpu_name="x", torch_version="0", cuda_version="0")
    paths, kwargs = _smoke_kwargs(tmp_path, module, monkeypatch, runtime, record)
    with pytest.raises(ValueError, match="cuda smoke device evidence required"):
        run_smoke(**kwargs)
    assert not (kwargs["out"] / "summary.manifest.json").exists()


def test_run_smoke_source_drift_refused(tmp_path, monkeypatch):
    import scripts.smoke_mtlface_joint_cuda_v4 as module

    record = dict(calls=[])
    paths, kwargs = _smoke_kwargs(tmp_path, module, monkeypatch, FakeCuda(), record)
    real = module.file_record
    seen: dict[str, int] = {}

    def flaky(path):
        item = dict(real(path))
        key = str(Path(path).resolve())
        if key == str(PRODUCER.resolve()):
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 2:
                item["sha256"] = "0" * 64
        return item

    monkeypatch.setattr(module, "file_record", flaky)
    with pytest.raises(RuntimeError, match="sources changed"):
        run_smoke(**kwargs)
    assert not (kwargs["out"] / "summary.manifest.json").exists()


@pytest.mark.parametrize("changes", [
    {"seed": True}, {"seed": 0}, {"seed": 1}, {"seed": 2**32},
    {"scope": "full"}, {"batch_size": 64}, {"batch_size": True},
])
def test_smoke_fixed_protocol_refuses_before_any_cuda_probe(tmp_path, monkeypatch, changes):
    import scripts.smoke_mtlface_joint_cuda_v4 as module
    record = dict(calls=[])
    _, kwargs = _smoke_kwargs(tmp_path, module, monkeypatch, FakeCuda(), record)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        run_smoke(**kwargs)
    assert record["calls"] == [] and not kwargs["out"].exists()


def test_state_comparison_explicitly_moves_both_operands_to_cpu(monkeypatch):
    import scripts.smoke_mtlface_joint_cuda_v4 as module
    calls = []
    original = torch.Tensor.cpu
    def tracked(value, *args, **kwargs):
        calls.append(tuple(value.shape))
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "cpu", tracked)
    assert module._same_cpu_tensor(torch.tensor([1., 2.]), torch.tensor([1., 2.]))
    assert calls == [(2,), (2,)]
    assert not module._same_cpu_tensor(torch.tensor([1]), torch.tensor([1.]))


def test_partial_manifest_write_exception_leaves_no_completion_marker(tmp_path, monkeypatch):
    import scripts.smoke_mtlface_joint_cuda_v4 as module
    record = dict(calls=[])
    _, kwargs = _smoke_kwargs(tmp_path, module, monkeypatch, FakeCuda(), record)
    def fail(path, **kw):
        path.write_text('{"partial":', encoding='utf-8')
        raise RuntimeError('partial write test')
    monkeypatch.setattr(module, 'write_experiment_manifest', fail)
    with pytest.raises(RuntimeError, match='partial write'):
        run_smoke(**kwargs)
    assert not (kwargs['out'] / 'summary.manifest.json').exists()


def test_nonfinite_injected_metrics_never_publish_completion(tmp_path, monkeypatch):
    import scripts.smoke_mtlface_joint_cuda_v4 as module
    record = dict(calls=[])
    _, kwargs = _smoke_kwargs(tmp_path, module, monkeypatch, FakeCuda(), record)
    original = kwargs['smoke_runner']
    def bad(*args, **kw):
        metrics, indices = original(*args, **kw)
        metrics['recognition']['total'] = float('nan')
        return metrics, indices
    kwargs['smoke_runner'] = bad
    with pytest.raises((ValueError, FloatingPointError)):
        run_smoke(**kwargs)
    assert not (kwargs['out'] / 'summary.manifest.json').exists()
