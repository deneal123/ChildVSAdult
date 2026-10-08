"""Synthetic CPU guards for the CUDA-only joint MTLFace campaign producer (v4).

These tests never run a GPU, never take a real optimizer step, never run real-face
inference and never load real weights: everything is tiny fake IO, tiny modules and
injected device seams. Passing them is NOT native CUDA evidence.
"""

# ruff: noqa: E402

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT / "scripts" / "run_mtlface_joint_campaign_cuda_v4.py"

from age_gap.common.manifest import file_record
from scripts.mtlface_target_stream_v3 import TargetAgeStream
from scripts.mtlface_training_v2 import FaceRecord, RecognitionDataset
from scripts.run_mtlface_joint_campaign_cuda_v4 import (
    CampaignConfig,
    _weights_snapshot,
    load_inputs,
    require_cuda,
    run_campaign,
)
from scripts.run_mtlface_joint_campaign_cuda_v4 import (
    main as producer_main,
)


# --------------------------------------------------------------------------------------
# Tiny synthetic modules (no forward pass, no optimizer step in tests)
# --------------------------------------------------------------------------------------
class TinyRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 3, 1)
        self.norm = nn.BatchNorm2d(3)

    def set_batchnorm_policy(self, policy="frozen_all"):
        if policy != "frozen_all":
            raise ValueError("frozen BN required")
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


class TinyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)


class TinyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)


class _NoStep:
    def step(self, *args, **kwargs):
        raise AssertionError("no optimizer step is allowed in synthetic tests")


class FakeCuda:
    """Injected CUDA seam; records ordering and device routing without touching a GPU."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.transferred: list[int] = []

    def require(self):
        self.calls.append(("require",))
        require_cuda()

    def seed(self, seed):
        self.calls.append(("seed", seed))
        return torch.Generator().manual_seed(seed)

    def device(self):
        return "cuda"

    def transfer(self, module, device):
        self.calls.append(("transfer", type(module).__name__, device))
        self.transferred.append(id(module))
        assert device == "cuda"
        return module

    def reset_peak(self):
        self.calls.append(("reset_peak",))

    def peaks(self):
        return dict(peak_allocated_bytes=1234, peak_reserved_bytes=5678)

    def binding(self):
        return dict(device="cuda", gpu_name="fake-gpu", torch_version="0.0", cuda_version="0.0")

    def synchronize(self):
        self.calls.append(("synchronize",))


def _synthetic_target_stream(tmp_path, seed=42):
    rows, bindings = [], {}
    for i, age in enumerate((0, 11, 21, 31, 41, 51, 61, None)):
        path = tmp_path / f"{i}.jpg"
        assert cv2.imwrite(str(path), np.full((4, 4, 3), i, dtype=np.uint8))
        data = path.read_bytes()
        bindings[path.resolve()] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        rows.append(FaceRecord(path, i, age))
    dataset = RecognitionDataset(rows, lambda image: image.transpose(2, 0, 1).astype(np.float32))
    return TargetAgeStream(dataset, torch.Generator().manual_seed(seed), bindings)


def _ledger(samples, batches, before, after, complete):
    steps = dict(fr=batches, generator=batches, discriminator=batches)
    return dict(
        epoch_complete=complete,
        completed_source_samples=samples,
        source_unique_rows=samples,
        source_decoded_views=samples,
        target_sampled_draws=samples,
        target_decoded_views=samples,
        completed_batches=batches,
        optimizer_steps=steps,
        optimizer_step_attempts=steps,
        entry_forward_images=dict(
            encoder_fr=samples,
            encoder_fas=2 * samples,
            generator_fas=samples,
            discriminator_fas=3 * samples,
        ),
        target_stream_before=before,
        target_stream_after=after,
    )


def _make_fake_epoch_runner(record):
    def fake_epoch(
        model,
        head,
        generator,
        discriminator,
        fr_opt,
        g_opt,
        d_opt,
        stream,
        *,
        batch_size,
        source_rng,
        device,
        generator_bn_policy,
        progress,
    ):
        assert device == "cuda"
        assert source_rng.device.type == "cpu"
        assert stream.generator.device.type == "cpu"
        record["device"] = device
        record["policy"] = generator_bn_policy
        samples = len(stream.dataset)
        batches = math.ceil(samples / batch_size)
        before = copy.deepcopy(stream.ledger())
        stream.draw_images(samples)
        after = copy.deepcopy(stream.ledger())
        for module in (model, head, generator, discriminator):
            with torch.no_grad():
                for value in module.parameters():
                    value.add_(0.01)
        for _ in range(batches):
            if progress is not None:
                progress(_ledger(samples, batches, before, after, complete=False))
        return dict(
            ledger=_ledger(samples, batches, before, after, complete=True),
            sample_weighted_means=dict(
                recognition=dict(total=1.0, gradient_norm=0.5),
                fas=dict(
                    discriminator_loss=1.0,
                    generator_loss=1.0,
                    discriminator_gradient_norm=0.5,
                    generator_gradient_norm=0.5,
                ),
            ),
        )

    return fake_epoch


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import scripts.run_mtlface_joint_campaign_cuda_v4 as module

    stream = _synthetic_target_stream(tmp_path)
    paths = {
        name: tmp_path / f"{name}.json"
        for name in ("faces", "initialization", "smoke", "joint_smoke", "targets", "weights")
    }
    for path in paths.values():
        path.write_text("{}", encoding="utf-8")
    faces = dict(
        inputs=[file_record(row.path) for row in stream.records],
        outputs=[],
        metrics=dict(counts=dict(retained_images=8, retained_people=8)),
    )
    init = dict(
        inputs=[file_record(paths["weights"])], outputs=[], metrics=dict(identity_classes=8)
    )
    empty = dict(inputs=[], outputs=[])
    contract = dict(weights=file_record(paths["weights"]))
    monkeypatch.setattr(
        module, "load_inputs", lambda *a: (init, faces, empty, contract, empty, empty)
    )

    created = []
    record: dict = {}
    cuda = FakeCuda()

    def models(weights, classes, *, seed, scope):
        torch.manual_seed(seed)
        values = (
            TinyRecognizer(),
            nn.Linear(3, classes),
            dict(seed=seed, weights=file_record(weights)),
            TinyGenerator(),
            TinyDiscriminator(),
        )
        created.append(values)
        return values

    def dataset(model, path):
        return stream.dataset, faces

    def optimizers(modules, config):
        record["optimizer_after_transfers"] = list(cuda.calls)
        return _NoStep(), _NoStep(), _NoStep()

    return dict(
        paths=paths,
        out=tmp_path / "campaign",
        config=CampaignConfig(2, 3, 0.001, 0.001, 0.001, "frozen"),
        model_factory=models,
        dataset_factory=dataset,
        epoch_runner=_make_fake_epoch_runner(record),
        optimizer_factory=optimizers,
        cuda=cuda,
        _record=record,
        _created=created,
    )


def _run(setup, **over):
    kwargs = {key: value for key, value in setup.items() if not key.startswith("_")}
    kwargs.update(over)
    return run_campaign(**kwargs)


# --------------------------------------------------------------------------------------
# Import / dry-run: no CUDA probe, no native verification, no model, no output
# --------------------------------------------------------------------------------------
def test_import_does_not_probe_gpu(monkeypatch):
    calls = []

    def boom():
        calls.append(True)
        raise AssertionError("import-time CUDA probe")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    fresh_spec = importlib.util.spec_from_file_location("mtlface_cuda_v4_reload", PRODUCER)
    fresh = importlib.util.module_from_spec(fresh_spec)
    sys.modules[fresh_spec.name] = fresh
    try:
        fresh_spec.loader.exec_module(fresh)
    finally:
        sys.modules.pop(fresh_spec.name, None)
    assert calls == []


def test_dry_run_touches_nothing(tmp_path, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise AssertionError("dry-run must not probe CUDA or read prerequisites")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    monkeypatch.setattr("scripts.run_mtlface_joint_campaign_cuda_v4.load_inputs", boom)
    out = tmp_path / "out"
    producer_main(
        [
            "--faces",
            str(tmp_path / "f.json"),
            "--initialization",
            str(tmp_path / "i.json"),
            "--smoke",
            str(tmp_path / "s.json"),
            "--joint-smoke",
            str(tmp_path / "j.json"),
            "--targets",
            str(tmp_path / "t.json"),
            "--weights",
            str(tmp_path / "w.bin"),
            "--out",
            str(out),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--fr-lr",
            "0.1",
            "--g-lr",
            "0.1",
            "--d-lr",
            "0.1",
            "--generator-bn-policy",
            "adapt",
            "--scope",
            "head",
            "--betas",
            "0.5",
            "0.99",
        ]
    )
    assert "dry-run" in capsys.readouterr().out
    assert not out.exists()


# --------------------------------------------------------------------------------------
# Fail-closed CUDA
# --------------------------------------------------------------------------------------
def test_cuda_unavailable_refused_before_models(setup, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def refuse(*args, **kwargs):
        raise AssertionError("no model may be constructed without CUDA")

    setup["cuda"] = None
    setup["model_factory"] = refuse
    with pytest.raises(RuntimeError, match="CUDA required"):
        _run(setup)
    assert not setup["out"].exists()


def test_require_cuda_rejects_zero_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="no device is visible"):
        require_cuda()


# --------------------------------------------------------------------------------------
# CUDA device routing, ordering, frozen BN and caller-owned CPU RNGs
# --------------------------------------------------------------------------------------
def test_full_campaign_routes_all_modules_to_cuda(setup):
    result = _run(setup)
    assert result["training_complete"]
    assert result["completed_seeds"] == [42, 1, 2]
    assert result["device_binding"]["gpu_name"] == "fake-gpu"
    assert result["peak_allocated_bytes"] == 1234 and result["peak_reserved_bytes"] == 5678
    calls = setup["cuda"].calls
    assert calls[0] == ("require",)
    assert ("transfer", "TinyRecognizer", "cuda") in calls
    assert ("transfer", "Linear", "cuda") in calls
    assert ("transfer", "TinyGenerator", "cuda") in calls
    assert ("transfer", "TinyDiscriminator", "cuda") in calls
    assert len(setup["cuda"].transferred) == 12  # 4 modules x 3 seeds
    assert len(setup["_created"]) == 3
    last_transfer = max(i for i, call in enumerate(calls) if call[0] == "transfer")
    assert setup["_record"]["optimizer_after_transfers"][last_transfer][0] == "transfer"
    assert setup["_record"]["device"] == "cuda"
    assert setup["_record"]["policy"] == "frozen"
    for model, *_rest in setup["_created"]:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                assert not module.training
    native = json.loads((setup["out"] / "summary.manifest.json").read_text())
    assert native["experiment"] == "mtlface-common-joint-campaign-cuda-v4"
    assert native["parameters"]["device"] == "cuda"
    assert native["parameters"]["implementation_mode"] == "injected-test"
    assert native["parameters"]["checkpoint_selection"] == "last_epoch"
    assert native["parameters"]["cuda_training_required"] is True
    assert "gpu_name" in native["parameters"] and "torch_version" in native["parameters"]


def test_seed_contract_binds_all_seeds_and_checkpoints(setup):
    result = _run(setup)
    native = json.loads((setup["out"] / "summary.manifest.json").read_text())
    bound = {r["sha256"] for r in native["outputs"]}
    checkpoints = []
    for seed in result["completed_seeds"]:
        cell = setup["out"] / "private" / f"seed_{seed}"
        checkpoint = cell / "last.pt"
        assert file_record(checkpoint)["sha256"] in bound
        checkpoints.append(torch.load(checkpoint, weights_only=True))
        history = json.loads((cell / "history.json").read_text())
        assert len(history) == 2
        assert (
            history[1]["ledger"]["target_stream_before"]
            == history[0]["ledger"]["target_stream_after"]
        )
        for epoch in (1, 2):
            path = cell / f"epoch_{epoch}_batches.jsonl"
            assert file_record(path)["sha256"] in bound
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == 3
    assert set(checkpoints[0]) == {"model", "identity_head", "generator", "discriminator"}
    assert not torch.equal(
        checkpoints[0]["model"]["stem.weight"], checkpoints[1]["model"]["stem.weight"]
    )
    cell = json.loads((setup["out"] / "private" / "seed_42" / "cell.manifest.json").read_text())
    assert cell["parameters"]["device"] == "cuda"
    assert cell["metrics"]["device_binding"]["device"] == "cuda"
    assert cell["metrics"]["changed_parameter_tensors"]["model"] >= 1


# --------------------------------------------------------------------------------------
# Preflight path is CPU prerequisite evidence only
# --------------------------------------------------------------------------------------
def test_preflight_constructs_no_models_and_probes_no_cuda(setup, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("preflight must not construct a neural model")

    def no_probe():
        raise AssertionError("preflight must not probe CUDA")

    setup["model_factory"] = refuse
    setup["cuda"] = None
    monkeypatch.setattr(torch.cuda, "is_available", no_probe)
    result = _run(setup, preflight_only=True)
    assert result["preflight_complete"] and not result["training_complete"]
    assert result["cuda_probed"] is False
    assert not (setup["out"] / "private").exists()
    native = json.loads((setup["out"] / "summary.manifest.json").read_text())
    assert native["experiment"] == "mtlface-common-joint-campaign-cuda-v4-preflight"
    assert native["parameters"]["device"] == "cpu"
    assert native["parameters"]["cuda_probed"] is False


# --------------------------------------------------------------------------------------
# Refusals: coverage, partial logging, fresh output, config, source drift
# --------------------------------------------------------------------------------------
def test_wrong_epoch_coverage_refused(setup):
    original = setup["epoch_runner"]

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result["ledger"]["source_unique_rows"] = 1
        return result

    setup["epoch_runner"] = corrupt
    with pytest.raises(ValueError, match="full joint"):
        _run(setup)
    assert not (setup["out"] / "summary.manifest.json").exists()


def test_partial_batch_logging_refused(setup):
    original = setup["epoch_runner"]

    def skip_progress(*args, **kwargs):
        kwargs["progress"] = None
        return original(*args, **kwargs)

    setup["epoch_runner"] = skip_progress
    with pytest.raises(ValueError, match="persisted batch coverage"):
        _run(setup)


def test_fresh_output_refused(setup):
    setup["out"].mkdir()
    with pytest.raises(FileExistsError):
        _run(setup)


@pytest.mark.parametrize(
    "key,value",
    [
        ("epochs", 0),
        ("epochs", True),
        ("batch_size", -1),
        ("fr_lr", float("nan")),
        ("g_lr", 0),
        ("seeds", (42, 42)),
        ("seeds", (-1,)),
        ("seeds", (2**32,)),
        ("seeds", (2**63 - 10001,)),
        ("seeds", (True,)),
        ("scope", "implicit"),
        ("generator_bn_policy", "implicit"),
        ("betas", (1.0, 0.99)),
    ],
)
def test_invalid_config_refused(setup, key, value):
    setup["config"] = replace(setup["config"], **{key: value})
    with pytest.raises(ValueError):
        _run(setup)
    assert not setup["out"].exists()


def test_source_change_during_run_refused(setup, monkeypatch):
    import scripts.run_mtlface_joint_campaign_cuda_v4 as module

    real = module.file_record
    seen: dict[str, int] = {}

    def flaky(path):
        record = dict(real(path))
        key = str(Path(path).resolve())
        if key == str(PRODUCER.resolve()):
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 2:
                record["sha256"] = "0" * 64
        return record

    monkeypatch.setattr(module, "file_record", flaky)
    with pytest.raises(RuntimeError, match="input/code changed"):
        _run(setup)
    assert not (setup["out"] / "summary.manifest.json").exists()


# --------------------------------------------------------------------------------------
# CPU prerequisite identity is preserved verbatim
# --------------------------------------------------------------------------------------
def _native_prerequisites(setup):
    paths = setup["paths"]
    joint = dict(
        inputs=[file_record(paths[name]) for name in ("faces", "initialization", "weights")],
        metrics=dict(
            real_crop_joint_smoke_complete=True,
            training_complete=False,
            scientific_evaluation_complete=False,
        ),
        parameters=dict(scope="head", device="cpu"),
    )
    targets = dict(
        inputs=[file_record(paths["faces"])],
        metrics=dict(
            metadata_sampling_preflight_complete=True,
            training_complete=False,
            scientific_evaluation_complete=False,
            retained_images=8,
            retained_recorded_identities=8,
        ),
        parameters=dict(policy="uniform-group-then-uniform-row-with-replacement"),
    )
    return joint, targets


def _patch_native(monkeypatch, setup, joint, targets):
    import scripts.run_mtlface_joint_campaign_cuda_v4 as module

    faces = dict(metrics=dict(counts=dict(retained_images=8, retained_people=8)))
    init = dict(metrics=dict(identity_classes=8))
    paths = setup["paths"]
    monkeypatch.setattr(module, "load_prerequisites", lambda **kw: (init, faces, {}, {}))
    monkeypatch.setattr(
        module,
        "verified",
        lambda path, experiment: joint if path == paths["joint_smoke"] else targets,
    )


def test_native_cpu_prerequisite_identity_preserved(setup, monkeypatch):
    joint, targets = _native_prerequisites(setup)
    _patch_native(monkeypatch, setup, joint, targets)
    assert load_inputs(setup["paths"], setup["config"])[4:] == (joint, targets)

    # A CUDA-claiming joint-smoke prerequisite must be rejected: CPU identity is fixed.
    joint["parameters"]["device"] = "cuda"
    with pytest.raises(ValueError, match="scope/device mismatch"):
        load_inputs(setup["paths"], setup["config"])


@pytest.mark.parametrize(
    "corruption",
    [
        None,
        "joint_incomplete",
        "target_incomplete",
        "face_binding",
        "weight_binding",
        "target_count",
        "target_policy",
        "joint_scope",
        "training_claim",
    ],
)
def test_joint_and_target_prerequisite_contracts(setup, monkeypatch, corruption):
    joint, targets = _native_prerequisites(setup)
    if corruption == "joint_incomplete":
        joint["metrics"]["real_crop_joint_smoke_complete"] = False
    elif corruption == "target_incomplete":
        targets["metrics"]["metadata_sampling_preflight_complete"] = False
    elif corruption == "face_binding":
        targets["inputs"] = []
    elif corruption == "weight_binding":
        joint["inputs"] = joint["inputs"][:2]
    elif corruption == "target_count":
        targets["metrics"]["retained_images"] = 7
    elif corruption == "target_policy":
        targets["parameters"]["policy"] = "random-unmatched"
    elif corruption == "joint_scope":
        joint["parameters"]["scope"] = "full"
    elif corruption == "training_claim":
        targets["metrics"]["training_complete"] = True
    _patch_native(monkeypatch, setup, joint, targets)
    if corruption is None:
        assert load_inputs(setup["paths"], setup["config"])[4:] == (joint, targets)
    else:
        with pytest.raises(ValueError):
            load_inputs(setup["paths"], setup["config"])


# --------------------------------------------------------------------------------------
# Snapshot helpers
# --------------------------------------------------------------------------------------
def test_saved_snapshot_has_independent_storage():
    module = nn.Linear(2, 2)
    snapshot = _weights_snapshot(dict(model=module))
    before = copy.deepcopy(snapshot)
    with torch.no_grad():
        module.weight.add_(1)
    assert torch.equal(snapshot["model"]["weight"], before["model"]["weight"])


def test_maximum_numpy_seed_has_valid_configuration():
    config = CampaignConfig(1, 1, 0.001, 0.001, 0.001, "frozen", seeds=(2**32 - 1,))
    config.validate()


def test_nonfinite_final_snapshot_refused():
    module = nn.Linear(2, 2)
    with torch.no_grad():
        module.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        _weights_snapshot(dict(model=module))
