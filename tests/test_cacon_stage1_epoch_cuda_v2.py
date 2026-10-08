"""Synthetic CPU guards for the CACon stage-1 CUDA epoch v2 correction.

The v2 fix is the explicit three-view ``.to`` transfer to the resolved actual
CUDA device plus a trusted-type anchor. All feasible checks below run on CPU
with a synthetic transfer/guard and tiny CPU forward/backward only. There is no
GPU, no real faces/weights/model inference and no real optimizer step, so a
passing test is NOT native CUDA execution evidence.
"""

# ruff: noqa: E402

from __future__ import annotations

import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

SUBDIR = Path(__file__).resolve().parents[1]
if str(SUBDIR) not in sys.path:
    sys.path.insert(0, str(SUBDIR))

from age_gap.common.manifest import file_record  # noqa: E402
from age_gap.training.sota_common import ProjectionHead, cacon_nt_xent  # noqa: E402
from scripts.cacon_dataset_v2 import ThreeViewDataset, ThreeViewRecord  # noqa: E402
from scripts.cacon_stage1_epoch_cuda_v2 import (  # noqa: E402
    OBJECTIVE,
    PRIMARY_LOCATOR,
    CaconStage1EpochFailure,
    CudaTensorGuard,
    CudaTransfer,
    _native_verified,
    _uniform_device_selection,
    run_cacon_stage1_epoch,
)


class SyntheticGuard:
    """Synthetic device seam: records calls and the requested device identity."""

    name = "synthetic-injected"

    def __init__(self):
        self.calls = []

    def require_module(self, module, role):
        self.calls.append(("module", role, None))

    def require_tensor(self, tensor, role, device=None):
        self.calls.append(("tensor", role, None if device is None else str(device)))

    def tensor_roles(self):
        return [role for kind, role, _ in self.calls if kind == "tensor"]


class FakeCudaTransfer:
    """Synthetic transfer seam; never claims native and only moves to the given device."""

    name = "synthetic-transfer"
    native = False

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.moves = []

    def resolve_device(self, backbone, projector):
        return self.device

    def to_device(self, tensor, device):
        self.moves.append(str(device))
        return tensor.to(device)


class LyingTransfer(FakeCudaTransfer):
    """Reports a CUDA device but returns the tensors unmoved (must be caught).

    ``to_device`` is deliberately a no-op so no real CUDA allocation can ever
    happen in tests; the device-identity check must refuse the unmoved CPU view.
    """

    def __init__(self):
        super().__init__("cuda:0")

    def to_device(self, tensor, device):
        self.moves.append(str(device))
        return tensor


class SpoofGuard(CudaTensorGuard):
    """Subclass overriding the native checks: must never be marked native."""

    name = "spoof-cuda"

    def require_module(self, module, role):
        return None

    def require_tensor(self, tensor, role, device=None):
        return None


class CountingOptimizer:
    """Deliberately not a torch optimizer: counts attempts, never updates parameters."""

    def __init__(self, parameters):
        self.param_groups = [{"params": list(parameters)}]
        self.attempts = 0

    def zero_grad(self, set_to_none=True):
        for parameter in self.param_groups[0]["params"]:
            parameter.grad = None

    def step(self):
        self.attempts += 1


class MiniBackbone(nn.Module):
    def __init__(self, dimension=8):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4, dimension)

    def forward(self, images):
        x = torch.relu(self.conv(images))
        return self.fc(x.mean(dim=(2, 3)))


class TailCountingProjector(ProjectionHead):
    def __init__(self, dimension=8):
        super().__init__(dimension)
        self.batches = 0

    def forward(self, x):
        self.batches += 1
        return super().forward(x)


def write_image(path, value):
    assert cv2.imwrite(str(path), np.full((8, 6, 3), value, dtype=np.uint8))


def make_dataset(tmp_path, count, *, seed=42):
    records = []
    for index in range(count):
        source = tmp_path / f"s{index}.png"
        generated = tmp_path / f"g{index}.png"
        write_image(source, index)
        write_image(generated, 100 + index)
        records.append(ThreeViewRecord(f"face{index}", file_record(source), file_record(generated)))

    def preprocess(image):
        return image.transpose(2, 0, 1).astype(np.float32) / 255

    return ThreeViewDataset(records, preprocess, np.random.default_rng(seed))


def make_setup(tmp_path, count, *, seed=3):
    torch.manual_seed(seed)
    dataset = make_dataset(tmp_path, count)
    backbone = MiniBackbone()
    projector = TailCountingProjector()
    optimizer = CountingOptimizer([*backbone.parameters(), *projector.parameters()])
    return dict(
        backbone=backbone,
        projector=projector,
        optimizer=optimizer,
        dataset=dataset,
        torch_generator=torch.Generator().manual_seed(seed),
        device_guard=SyntheticGuard(),
        device_transfer=FakeCudaTransfer("cpu"),
    )


def run(setup, **overrides):
    kwargs = dict(
        batch_size=2,
        device="cuda",
        bn_policy="frozen_all",
        tail_policy="reject",
        temperature=0.1,
    )
    kwargs.update(overrides)
    return run_cacon_stage1_epoch(**setup, **kwargs)


# ---------------------------------------------------------------------------------------
# Trusted-type anchor: native verification cannot be fabricated by subclassing
# ---------------------------------------------------------------------------------------
def test_native_verification_requires_exact_trusted_types():
    assert _native_verified(CudaTensorGuard(), CudaTransfer()) is True
    assert _native_verified(SpoofGuard(), CudaTransfer()) is False
    assert _native_verified(CudaTensorGuard(), FakeCudaTransfer()) is False
    assert _native_verified(SpoofGuard(), FakeCudaTransfer()) is False


def test_spoof_guard_is_never_reported_native(tmp_path):
    setup = make_setup(tmp_path, 4)
    setup["device_guard"] = SpoofGuard()
    setup["device_transfer"] = CudaTransfer()
    with pytest.raises(CaconStage1EpochFailure, match="subclasses overriding") as info:
        run(setup)
    assert info.value.ledger["native_cuda_verified"] is False
    assert info.value.ledger["optimizer_steps"] == 0


def test_native_guard_with_unapproved_transfer_refused_before_step(tmp_path):
    setup = make_setup(tmp_path, 4)
    setup["device_guard"] = CudaTensorGuard()
    setup["device_transfer"] = FakeCudaTransfer("cpu")
    with pytest.raises(CaconStage1EpochFailure, match="approved CudaTransfer seam"):
        run(setup)


def test_all_synthetic_runs_are_flagged_non_native(tmp_path):
    ledger = run(make_setup(tmp_path, 4))
    assert ledger["native_cuda_verified"] is False
    assert ledger["device_guard"] == "synthetic-injected"
    assert ledger["device_transfer"] == "synthetic-transfer"
    assert ledger["declared_transfer_native"] is False


# ---------------------------------------------------------------------------------------
# Device-identity contract (resolved actual device, not merely "any CUDA")
# ---------------------------------------------------------------------------------------
def test_mixed_backbone_projector_devices_refused():
    with pytest.raises(ValueError, match="mixed-device"):
        _uniform_device_selection({"cuda:0"}, {"cuda:1"})
    with pytest.raises(ValueError, match="single device identity"):
        _uniform_device_selection({"cuda:0", "cuda:1"}, {"cuda:0"})


def test_uniform_device_selection_returns_shared_identity():
    assert _uniform_device_selection({"cuda:1"}, {"cuda:1"}) == "cuda:1"


def test_resolve_device_refuses_mixed_modules(monkeypatch):
    import scripts.cacon_stage1_epoch_cuda_v2 as module

    monkeypatch.setattr(
        module,
        "_module_devices",
        lambda mod: {"cuda:0"} if mod == "backbone" else {"cuda:1"},
    )
    with pytest.raises(ValueError, match="mixed-device"):
        CudaTransfer().resolve_device("backbone", "projector")


def test_resolved_device_must_match_module_device(tmp_path, monkeypatch):
    import scripts.cacon_stage1_epoch_cuda_v2 as module

    setup = make_setup(tmp_path, 4)
    monkeypatch.setattr(module, "_module_devices", lambda mod: {"cuda:0"})
    transfer = FakeCudaTransfer("cuda:1")  # lies: modules are cuda:0
    setup["device_transfer"] = transfer
    with pytest.raises(CaconStage1EpochFailure, match="does not match the resolved device") as info:
        run(setup)
    assert info.value.ledger["resolved_device"] == "cuda:1"
    assert info.value.ledger["device_consistent"] is False


def test_transfer_that_does_not_move_tensors_is_refused(tmp_path, monkeypatch):
    import scripts.cacon_stage1_epoch_cuda_v2 as module

    setup = make_setup(tmp_path, 4)
    monkeypatch.setattr(module, "_module_devices", lambda mod: {"cuda:0"})
    setup["device_transfer"] = LyingTransfer()  # returns real CPU tensors
    with pytest.raises(
        CaconStage1EpochFailure, match="transferred view device does not match"
    ) as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_explicit_transfer_is_invoked_once_per_view_with_resolved_device(tmp_path):
    setup = make_setup(tmp_path, 6)
    ledger = run(setup)
    # 3 views per batch across 3 batches
    assert setup["device_transfer"].moves == ["cpu"] * 9
    assert ledger["device_consistent"] is True
    assert ledger["resolved_device"] == "cpu"


def test_native_guard_rejects_cpu_batches_absent_transfer_seam(tmp_path):
    # Native guard + native transfer on this CPU host: the run is refused before any
    # step because there is no CPU fallback and no batch ever reaches the guard.
    setup = make_setup(tmp_path, 4)
    setup.pop("device_transfer")
    setup["device_guard"] = CudaTensorGuard()
    with pytest.raises(CaconStage1EpochFailure, match="CUDA") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0
    assert info.value.ledger["samples"] == 0
    assert info.value.ledger["native_cuda_verified"] is False


def test_native_guard_rejects_cpu_tensor_directly():
    with pytest.raises(ValueError, match="CPU/other-device fallback refused"):
        CudaTensorGuard().require_tensor(torch.zeros(2, 3), "input-view")


def test_native_guard_enforces_requested_device_identity():
    class NotATensor:
        pass

    # A non-torch object is a type error; exact-device mismatch needs a real CUDA
    # tensor and is covered by the native-guard CPU rejection below on this host.
    with pytest.raises(TypeError, match="torch.Tensor"):
        CudaTensorGuard().require_tensor(NotATensor(), "input-view")


def test_native_transfer_refuses_cpu_backbone_without_fallback():
    class CpuBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 3)

        def forward(self, x):
            return self.fc(x)

    backbone = CpuBackbone()
    with pytest.raises(ValueError, match="no CPU fallback"):
        CudaTransfer().resolve_device(backbone, backbone)


def test_native_guard_refuses_actual_dataset_cpu_batch(tmp_path):
    # The authentic ThreeViewDataset necessarily yields CPU tensors; the native guard
    # must refuse each of them directly when no approved transfer seam is applied.
    dataset = make_dataset(tmp_path, 2)
    views = dataset[0]
    assert len(views) == 3
    guard = CudaTensorGuard()
    for view in views:
        assert view.device.type == "cpu"
        with pytest.raises(ValueError, match="CPU/other-device fallback refused"):
            guard.require_tensor(view, "input-view")


# ---------------------------------------------------------------------------------------
# Finite gates: projected features and gradients
# ---------------------------------------------------------------------------------------
def test_nonfinite_projected_features_refused_before_step(tmp_path):
    setup = make_setup(tmp_path, 4)

    class NanFeatureProjector(ProjectionHead):
        def forward(self, x):
            return torch.full((x.shape[0], 128), float("nan"))

    setup["projector"] = NanFeatureProjector(8)
    optimizer = CountingOptimizer(
        [*setup["backbone"].parameters(), *setup["projector"].parameters()]
    )
    setup["optimizer"] = optimizer
    with pytest.raises(CaconStage1EpochFailure, match="nonfinite projected features") as info:
        run(setup)
    assert info.value.ledger["features_finite"] is False
    assert info.value.ledger["optimizer_step_attempts"] == 0
    assert optimizer.attempts == 0


def test_nonfinite_gradient_refused_before_step(tmp_path):
    setup = make_setup(tmp_path, 4)
    target = setup["backbone"].conv.weight
    target.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    with pytest.raises(CaconStage1EpochFailure, match="nonfinite parameter gradient") as info:
        run(setup)
    ledger = info.value.ledger
    assert ledger["gradients_finite"] is False
    assert ledger["optimizer_step_attempts"] == 0
    assert setup["optimizer"].attempts == 0


def test_nonfinite_input_refused_before_step(tmp_path):
    setup = make_setup(tmp_path, 4)

    class PoisonedDataset(ThreeViewDataset):
        def __getitem__(self, index):
            tensors = super().__getitem__(index)
            tensors[0][0, 0, 0] = float("nan")
            return tensors

    base = setup["dataset"]
    setup["dataset"] = PoisonedDataset(base.records, base.preprocess, base.rng)
    with pytest.raises(CaconStage1EpochFailure, match="nonfinite three-view input") as info:
        run(setup)
    assert info.value.ledger["inputs_finite"] is False
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_nonfinite_loss_refused_before_step(tmp_path, monkeypatch):
    setup = make_setup(tmp_path, 4)
    import scripts.cacon_stage1_epoch_cuda_v2 as module

    monkeypatch.setattr(module, "cacon_nt_xent", lambda *a, **k: torch.tensor(float("nan")))
    with pytest.raises(CaconStage1EpochFailure, match="nonfinite NT-Xent loss") as info:
        run(setup)
    assert info.value.ledger["losses_finite"] is False
    assert info.value.ledger["optimizer_step_attempts"] == 0


# ---------------------------------------------------------------------------------------
# Retained v1 contracts
# ---------------------------------------------------------------------------------------
def test_primary_locator_and_equations_declared():
    assert PRIMARY_LOCATOR == "https://arxiv.org/html/2312.11195v2#S2.SS3"
    assert "Eq.6" in OBJECTIVE and "mined person labels unused" in OBJECTIVE


def test_full_coverage_continuing_rng_and_counters(tmp_path):
    setup = make_setup(tmp_path, 6)
    before_generator = copy.deepcopy(setup["torch_generator"].get_state())
    projections_before = [copy.deepcopy(p.detach()) for p in setup["projector"].parameters()]
    reports = []
    ledger = run(setup, progress=lambda row: reports.append(copy.deepcopy(row)))
    assert ledger["epoch_complete"] is True and ledger["single_epoch_coverage"] is True
    assert ledger["samples"] == ledger["source_unique_rows"] == 6
    assert ledger["completed_batches"] == 3
    assert ledger["encoder_image_instances"] == 18
    assert ledger["optimizer_steps"] == ledger["optimizer_step_attempts"] == 3
    assert ledger["sample_weighted_loss"] > 0
    assert ledger["sample_weighted_gradient_norm"] >= 0
    assert ledger["mined_person_labels_used"] is False
    assert ledger["full_method_parity"] is False and ledger["publication_ready"] is False
    assert setup["projector"].batches == 3
    assert not torch.equal(before_generator, setup["torch_generator"].get_state())
    assert ledger["torch_generator_initial_sha256"] != ledger["torch_generator_current_sha256"]
    assert sorted(ledger["ordered_source_indices"]) == list(range(6))
    assert [row["samples"] for row in reports] == [2, 4, 6]
    assert not any(
        not torch.equal(p.detach(), prior)
        for p, prior in zip(setup["projector"].parameters(), projections_before, strict=True)
    )


def test_frozen_bn_policy_preserves_backbone_running_stats(tmp_path):
    torch.manual_seed(4)
    dataset = make_dataset(tmp_path, 4)

    class BNBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(3)
            self.conv = nn.Conv2d(3, 3, 3, padding=1)

        def forward(self, images):
            return self.bn(self.conv(images)).mean(dim=(2, 3))

    backbone = BNBackbone()
    backbone.train()
    projector = ProjectionHead(3)
    optimizer = CountingOptimizer([*backbone.parameters(), *projector.parameters()])
    bn = backbone.bn
    before = (
        bn.running_mean.detach().clone(),
        bn.running_var.detach().clone(),
        int(bn.num_batches_tracked),
    )
    setup = dict(
        backbone=backbone,
        projector=projector,
        optimizer=optimizer,
        dataset=dataset,
        torch_generator=torch.Generator().manual_seed(4),
        device_guard=SyntheticGuard(),
        device_transfer=FakeCudaTransfer("cpu"),
    )
    ledger = run(setup, bn_policy="frozen_all")
    assert ledger["bn_modules"] == 1 and ledger["bn_running_stats_unchanged"] is True
    assert torch.equal(bn.running_mean, before[0])
    assert int(bn.num_batches_tracked) == before[2]
    assert run(setup, bn_policy="train_all")["bn_running_stats_unchanged"] is False


def test_tail_reject_refused_before_generator_draw(tmp_path):
    setup = make_setup(tmp_path, 5)
    before = copy.deepcopy(setup["torch_generator"].get_state())
    with pytest.raises(CaconStage1EpochFailure, match="tail singleton") as info:
        run(setup, tail_policy="reject")
    assert info.value.ledger["optimizer_steps"] == 0
    assert torch.equal(before, setup["torch_generator"].get_state())


def test_tail_merge_reports_actual_merged_batch_count(tmp_path):
    setup = make_setup(tmp_path, 5)
    ledger = run(setup, tail_policy="merge_tail")
    assert ledger["tail_merged"] is True
    assert ledger["expected_batches"] == ledger["completed_batches"] == 2
    assert ledger["optimizer_steps"] == 2
    assert ledger["encoder_image_instances"] == 15
    assert sorted(len(row) for row in ledger["batch_partitions"]) == [2, 3]
    assert ledger["sample_weighted_loss"] > 0


def test_optimizer_duplicate_and_partial_ownership_refused(tmp_path):
    setup = make_setup(tmp_path, 4)
    setup["optimizer"] = CountingOptimizer(
        [
            *setup["backbone"].parameters(),
            *setup["projector"].parameters(),
            *setup["backbone"].parameters(),
        ]
    )
    with pytest.raises(CaconStage1EpochFailure, match="duplicated parameter"):
        run(setup)
    setup = make_setup(tmp_path, 4, seed=5)
    setup["optimizer"] = CountingOptimizer(setup["backbone"].parameters())
    with pytest.raises(CaconStage1EpochFailure, match="exactly once"):
        run(setup)


def test_batch_size_below_two_and_device_refused(tmp_path):
    with pytest.raises(CaconStage1EpochFailure, match="batch size >= 2"):
        run(make_setup(tmp_path, 4), batch_size=1)
    with pytest.raises(CaconStage1EpochFailure, match="no CPU fallback"):
        run(make_setup(tmp_path, 4), device="cpu")


def test_failing_optimizer_preserves_partial_attempts(tmp_path):
    setup = make_setup(tmp_path, 4)

    class Failing(CountingOptimizer):
        def step(self):
            super().step()
            raise RuntimeError("injected optimizer failure")

    setup["optimizer"] = Failing(
        [*setup["backbone"].parameters(), *setup["projector"].parameters()]
    )
    with pytest.raises(CaconStage1EpochFailure, match="injected") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 1
    assert info.value.ledger["optimizer_steps"] == 0
    assert info.value.ledger["epoch_complete"] is False


def test_canonical_nt_xent_reused():
    # The module must call the canonical objective, not a local reimplementation.
    import scripts.cacon_stage1_epoch_cuda_v2 as module

    assert module.cacon_nt_xent is cacon_nt_xent


def test_wrong_types_refused(tmp_path):
    setup = make_setup(tmp_path, 4)
    with pytest.raises(CaconStage1EpochFailure, match="ProjectionHead"):
        run({**setup, "projector": nn.Linear(8, 8)})
    with pytest.raises(CaconStage1EpochFailure, match="ThreeViewDataset"):
        run({**setup, "dataset": [1, 2, 3]})


@pytest.mark.parametrize(
    "key,value",
    [
        ("bn_policy", "implicit"),
        ("tail_policy", "silent"),
        ("temperature", float("nan")),
        ("temperature", 0.0),
        ("device", "cpu"),
        ("progress", True),
    ],
)
def test_invalid_contracts_refused_before_mutation(tmp_path, key, value):
    setup = make_setup(tmp_path, 4)
    gen_before = copy.deepcopy(setup["torch_generator"].get_state())
    with pytest.raises((CaconStage1EpochFailure, TypeError, ValueError)):
        run(setup, **{key: value})
    assert torch.equal(gen_before, setup["torch_generator"].get_state())
    assert setup["optimizer"].attempts == 0


def test_native_guard_refuses_transfer_subclasses_before_device_claim(tmp_path):
    class TransferSubclass(CudaTransfer):
        def resolve_device(self, backbone, projector):
            return torch.device("cpu")

    setup = make_setup(tmp_path, 4)
    setup["device_guard"] = CudaTensorGuard()
    setup["device_transfer"] = TransferSubclass()
    with pytest.raises(CaconStage1EpochFailure, match="approved CudaTransfer seam") as info:
        run(setup)
    assert info.value.ledger["native_cuda_verified"] is False
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_repeated_epochs_chain_caller_rng_state(tmp_path):
    setup = make_setup(tmp_path, 6)
    first = run(setup)
    second = run(setup)
    assert first["torch_generator_current_sha256"] == second["torch_generator_initial_sha256"]
    assert first["epoch_complete"] and second["epoch_complete"]
    assert sorted(second["ordered_source_indices"]) == list(range(6))


def test_mutation_of_unread_cache_record_stops_next_batch(tmp_path):
    setup = make_setup(tmp_path, 4)

    def mutate(ledger):
        if ledger["completed_batches"] == 1:
            unread = ledger["batch_partitions"][1][0]
            setup["dataset"].records[unread].source_record["sha256"] = "0" * 64

    with pytest.raises(CaconStage1EpochFailure, match="hash") as info:
        run(setup, progress=mutate)
    assert info.value.ledger["completed_batches"] == 1
    assert info.value.ledger["optimizer_steps"] == 1
    assert info.value.ledger["epoch_complete"] is False


def test_duplicate_face_ids_fail_before_optimizer_attempt(tmp_path):
    setup = make_setup(tmp_path, 4)
    setup["dataset"].records = tuple(
        ThreeViewRecord("same", row.source_record, row.generated_record)
        for row in setup["dataset"].records)
    with pytest.raises(CaconStage1EpochFailure, match="unique nonempty") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_single_sample_epoch_never_creates_contrastive_step(tmp_path):
    setup = make_setup(tmp_path, 1)
    with pytest.raises(CaconStage1EpochFailure, match="singleton") as info:
        run(setup, tail_policy="merge_tail")
    assert info.value.ledger["optimizer_step_attempts"] == 0
