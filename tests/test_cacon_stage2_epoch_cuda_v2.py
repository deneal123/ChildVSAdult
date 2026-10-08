"""Synthetic CPU tests for the CACon stage-2 v2 label-provenance/ownership hardening.

All checks run on CPU with a synthetic guard/transfer and tiny CPU
forward/backward only. There is no GPU, no real faces/weights/model inference
and no real optimizer step: the counting fake optimizer never updates
parameters, so a passing suite is NOT native CUDA execution evidence. The
canonical ``FinalLinearStage`` and the stage-1 trusted guard/transfer and
observed ledger are reused unmodified.
"""

# ruff: noqa: E402

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from scripts.cacon_stage1_epoch_cuda_v2 import (  # noqa: E402
    CudaTensorGuard,
    CudaTransfer,
    _native_verified,
)
from scripts.cacon_stage2_epoch_cuda_v2 import (  # noqa: E402
    REPRESENTATION_DECLARATION,
    SUPPORTED_DATASET_TYPES,
    CaconStage2EpochFailure,
    Stage2LabelDataset,
    Stage2Record,
    run_cacon_stage2_epoch,
)
from scripts.cacon_supervised_v2 import FinalLinearStage  # noqa: E402


class SyntheticGuard:
    name = "synthetic-injected"
    synthetic = True

    def __init__(self):
        self.calls = []

    def require_module(self, module, role):
        self.calls.append(("module", role, None))

    def require_tensor(self, tensor, role, device=None):
        self.calls.append(("tensor", role, None if device is None else str(device)))

    def tensor_roles(self):
        return [role for kind, role, _ in self.calls if kind == "tensor"]


class FakeCudaTransfer:
    name = "synthetic-transfer"
    native = False
    synthetic = True

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.moves = []

    def resolve_device(self, backbone, projector):
        return self.device

    def to_device(self, tensor, device):
        self.moves.append(str(device))
        return tensor.to(device)


class UnmovedTransfer(FakeCudaTransfer):
    def __init__(self):
        super().__init__("cuda:0")

    def to_device(self, tensor, device):
        self.moves.append(str(device))
        return tensor


class SpoofGuard(CudaTensorGuard):
    name = "spoof-cuda"

    def require_module(self, module, role):
        return None

    def require_tensor(self, tensor, role, device=None):
        return None


class UndeclaredGuard:
    """Non-native guard that never declares itself synthetic: must be refused."""

    name = "undeclared"

    def require_module(self, module, role):
        return None

    def require_tensor(self, tensor, role, device=None):
        return None


class CountingOptimizer:
    """Not a torch optimizer: counts attempts and never updates parameters."""

    def __init__(self, parameters):
        self.param_groups = [{"params": list(parameters)}]
        self.attempts = 0

    def zero_grad(self, set_to_none=True):
        for parameter in self.param_groups[0]["params"]:
            parameter.grad = None

    def step(self):
        self.attempts += 1


class MiniBackbone(nn.Module):
    """Tiny CNN backbone with BatchNorm so the frozen-BN contract is non-vacuous."""

    def __init__(self, dimension=8):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, dimension)

    def forward(self, images):
        x = torch.relu(self.bn(self.conv(images)))
        return self.fc(x.mean(dim=(2, 3)))


class DisconnectedClassifier(nn.Module):
    """Classifier with an unused trainable parameter: no actual gradient."""

    def __init__(self, dimension, classes):
        super().__init__()
        self.linear = nn.Linear(dimension, classes)
        self.unused = nn.Parameter(torch.zeros(dimension))

    def forward(self, features):
        return self.linear(features)


def write_image(path, value):
    import cv2

    assert cv2.imwrite(str(path), np.full((8, 6, 3), value, dtype=np.uint8))


def preprocess(image):
    return image.transpose(2, 0, 1).astype(np.float32) / 255


def make_dataset(tmp_path, count, *, classes=3):
    records = []
    for index in range(count):
        image = tmp_path / f"img{index}.png"
        write_image(image, index)
        records.append(Stage2Record(f"face{index}", index % classes, image))
    return Stage2LabelDataset(records, preprocess)


def make_setup(tmp_path, count, *, classes=3, seed=5):
    torch.manual_seed(seed)
    dataset = make_dataset(tmp_path, count, classes=classes)
    backbone = MiniBackbone()
    stage = FinalLinearStage(backbone, 8, dataset.n_classes)
    optimizer = CountingOptimizer(list(stage.classifier.parameters()))
    return dict(
        stage=stage,
        optimizer=optimizer,
        dataset=dataset,
        torch_generator=torch.Generator().manual_seed(seed),
        device_guard=SyntheticGuard(),
        device_transfer=FakeCudaTransfer("cpu"),
    )


def run(setup, **overrides):
    kwargs = dict(batch_size=2, device="cuda", bn_policy="frozen_all")
    kwargs.update(overrides)
    return run_cacon_stage2_epoch(**setup, **kwargs)


# ----------------------------------------------------------- (1) label provenance
def test_actual_batch_labels_are_verified_and_recorded(tmp_path):
    setup = make_setup(tmp_path, 5)
    ledger = run(setup)
    assert ledger["label_provenance_verified"] is True
    assert ledger["label_provenance_batches"] == ledger["expected_batches"]
    assert ledger["last_batch_observed_labels"] == ledger["last_batch_declared_labels"]
    for row in ledger["batch_summaries"]:
        assert row["observed_labels"] == row["identities"]
    assert setup["optimizer"].attempts == 3


def test_in_range_flipped_label_refused_at_zero_attempts(tmp_path, monkeypatch):
    setup = make_setup(tmp_path, 4)  # labels cycle 0,1,2
    original = Stage2LabelDataset.__getitem__

    def flipped(self, index):
        tensor, label = original(self, index)
        return tensor, (label + 1) % 3  # in-range flip, not a range violation

    monkeypatch.setattr(Stage2LabelDataset, "__getitem__", flipped)
    with pytest.raises(CaconStage2EpochFailure, match="label provenance refused") as info:
        run(setup)
    ledger = info.value.ledger
    assert ledger["label_provenance_verified"] is False
    assert ledger["optimizer_step_attempts"] == 0
    assert ledger["samples"] == 0
    assert setup["optimizer"].attempts == 0


def test_later_batch_label_corruption_preserves_earlier_counts(tmp_path, monkeypatch):
    setup = make_setup(tmp_path, 4)
    generator = torch.Generator().manual_seed(5)
    order = torch.randperm(4, generator=generator).tolist()
    second_batch = set(order[2:])
    original = Stage2LabelDataset.__getitem__

    def corrupt_late(self, index):
        tensor, label = original(self, index)
        return (tensor, (label + 1) % 3) if index in second_batch else (tensor, label)

    monkeypatch.setattr(Stage2LabelDataset, "__getitem__", corrupt_late)
    with pytest.raises(CaconStage2EpochFailure, match="label provenance refused") as info:
        run(setup)
    ledger = info.value.ledger
    # The first batch completed and stepped; the second was refused before forward.
    assert ledger["completed_batches"] == 1
    assert ledger["optimizer_step_attempts"] == 1
    assert ledger["optimizer_steps"] == 1
    assert ledger["samples"] == 2
    assert ledger["label_provenance_batches"] == 1
    assert setup["optimizer"].attempts == 1


# -------------------------------------------- (2) exact types + entry revalidation
def test_supported_dataset_types_are_exact_concretes():
    from age_gap.training.sota_common import MTLFaceDataset

    assert Stage2LabelDataset in SUPPORTED_DATASET_TYPES
    assert MTLFaceDataset in SUPPORTED_DATASET_TYPES


def test_forged_dataset_subclass_refused(tmp_path):
    class Forged(Stage2LabelDataset):
        pass

    setup = make_setup(tmp_path, 4)
    dataset = setup["dataset"]
    forged = Forged(dataset.records, preprocess)
    setup["dataset"] = forged
    with pytest.raises(CaconStage2EpochFailure, match="forged subclasses") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_mutated_records_refused_at_entry(tmp_path):
    setup = make_setup(tmp_path, 4)
    dataset = setup["dataset"]
    mutable = list(dataset.records)
    mutable[0] = Stage2Record("face0", 99, mutable[0].path)  # out of range after construction
    dataset.records = tuple(mutable)
    with pytest.raises(CaconStage2EpochFailure, match="n_classes|contiguous") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_stale_n_classes_refused_at_entry(tmp_path):
    setup = make_setup(tmp_path, 4)
    setup["dataset"].n_classes = 99
    with pytest.raises(CaconStage2EpochFailure, match="stale dataset n_classes") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_duplicate_face_ids_in_mutated_records_refused(tmp_path):
    setup = make_setup(tmp_path, 4)
    dataset = setup["dataset"]
    mutable = list(dataset.records)
    mutable[1] = Stage2Record(mutable[0].face_id, mutable[1].identity, mutable[1].path)
    dataset.records = tuple(mutable)
    with pytest.raises(CaconStage2EpochFailure, match="unique") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_empty_face_id_in_mutated_records_refused(tmp_path):
    setup = make_setup(tmp_path, 4)
    dataset = setup["dataset"]
    mutable = list(dataset.records)
    mutable[0] = Stage2Record("", mutable[0].identity, mutable[0].path)
    dataset.records = tuple(mutable)
    with pytest.raises(CaconStage2EpochFailure, match="nonempty string face IDs") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_boolean_label_in_mutated_records_refused(tmp_path):
    setup = make_setup(tmp_path, 4)
    dataset = setup["dataset"]
    mutable = list(dataset.records)
    mutable[0] = Stage2Record(mutable[0].face_id, True, mutable[0].path)
    dataset.records = tuple(mutable)
    with pytest.raises(CaconStage2EpochFailure, match="non-boolean integer") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_mtlface_dataset_is_revalidated_at_entry(tmp_path, monkeypatch):
    import age_gap.training.sota_common as sota
    from age_gap.training.sota_common import FaceItem, MTLFaceDataset

    paths = []
    for index in range(4):
        image = tmp_path / f"mtl{index}.png"
        write_image(image, index)
        paths.append(image)
    monkeypatch.setattr(
        sota,
        "common_protocol_faces",
        lambda split="train", crops_dir="faces": [
            FaceItem(path, index % 3, None) for index, path in enumerate(paths)
        ],
    )
    dataset = MTLFaceDataset(preprocess)
    assert type(dataset) is MTLFaceDataset

    setup = make_setup(tmp_path, 4)
    setup["dataset"] = dataset
    ledger = run(setup)
    assert ledger["dataset_type"] == "MTLFaceDataset"
    assert ledger["source_entry_revalidated"] is True
    assert ledger["samples"] == 4 and ledger["encoder_image_instances"] == 4

    # Mutating the live items must be caught at entry, not trusted from construction.
    dataset.items[0] = FaceItem(paths[0], 7, None)
    setup2 = make_setup(tmp_path, 4)
    setup2["dataset"] = dataset
    with pytest.raises(CaconStage2EpochFailure, match="contiguous|n_classes") as info:
        run(setup2)
    assert info.value.ledger["optimizer_step_attempts"] == 0


# ------------------------------------------- (3) per-batch optimizer ownership
def test_ownership_mutation_in_progress_callback_fails_before_next_step(tmp_path):
    setup = make_setup(tmp_path, 4)
    optimizer = setup["optimizer"]

    def hostile(_ledger):
        optimizer.param_groups[0]["params"].append(setup["stage"].classifier.bias)

    with pytest.raises(CaconStage2EpochFailure, match="duplicated") as info:
        run(setup, progress=hostile)
    ledger = info.value.ledger
    assert ledger["completed_batches"] == 1
    assert ledger["optimizer_step_attempts"] == 1
    assert setup["optimizer"].attempts == 1


def test_encoder_parameter_injected_mid_epoch_fails_before_next_step(tmp_path):
    setup = make_setup(tmp_path, 4)
    optimizer = setup["optimizer"]
    encoder_parameter = next(setup["stage"].backbone.parameters())

    def hostile(_ledger):
        optimizer.param_groups[0]["params"][0] = encoder_parameter

    with pytest.raises(CaconStage2EpochFailure, match="exactly once|identity") as info:
        run(setup, progress=hostile)
    assert info.value.ledger["optimizer_step_attempts"] == 1


def test_ownership_checks_count_matches_batches(tmp_path):
    ledger = run(make_setup(tmp_path, 4))  # 2 batches
    assert ledger["optimizer_ownership_checks"] == 3  # entry + one per batch


def test_encoder_parameter_in_optimizer_refused_at_entry(tmp_path):
    setup = make_setup(tmp_path, 4)
    setup["optimizer"] = CountingOptimizer(
        [*setup["stage"].classifier.parameters(), next(setup["stage"].backbone.parameters())]
    )
    with pytest.raises(CaconStage2EpochFailure, match="exactly once") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_partial_classifier_ownership_refused(tmp_path):
    setup = make_setup(tmp_path, 4)
    setup["optimizer"] = CountingOptimizer([setup["stage"].classifier.weight])
    with pytest.raises(CaconStage2EpochFailure, match="exactly once") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0


# --------------------------------------------------- (4) mandatory actual gradients
def test_disconnected_classifier_parameter_refused_at_zero_attempts(tmp_path):
    setup = make_setup(tmp_path, 4)
    stage = setup["stage"]
    stage.classifier = DisconnectedClassifier(8, stage.classes)
    setup["optimizer"] = CountingOptimizer(list(stage.classifier.parameters()))
    with pytest.raises(CaconStage2EpochFailure, match="no actual gradient") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0
    assert setup["optimizer"].attempts == 0


def test_nonfinite_classifier_gradient_refused_before_step(tmp_path):
    setup = make_setup(tmp_path, 4)
    stage = setup["stage"]
    handle = stage.classifier.weight.register_hook(lambda g: torch.full_like(g, float("nan")))
    try:
        with pytest.raises(CaconStage2EpochFailure, match="nonfinite classifier gradient") as info:
            run(setup)
    finally:
        handle.remove()
    assert info.value.ledger["gradients_finite"] is False
    assert info.value.ledger["optimizer_step_attempts"] == 0
    assert setup["optimizer"].attempts == 0


# ------------------------------------------------------------- trust / device
def test_counterfeit_final_linear_stage_refused(tmp_path):
    class Counterfeit(FinalLinearStage):
        pass

    setup = make_setup(tmp_path, 4)
    real = setup["stage"]
    fake = Counterfeit(real.backbone, 8, real.classes)
    setup["stage"] = fake
    setup["optimizer"] = CountingOptimizer(list(fake.classifier.parameters()))
    with pytest.raises(CaconStage2EpochFailure, match="authentic FinalLinearStage") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_native_verification_requires_exact_types():
    assert _native_verified(CudaTensorGuard(), CudaTransfer()) is True
    assert _native_verified(SpoofGuard(), CudaTransfer()) is False
    assert _native_verified(CudaTensorGuard(), FakeCudaTransfer()) is False


def test_synthetic_seam_is_declared_and_never_certifies_authentic_source(tmp_path):
    ledger = run(make_setup(tmp_path, 4))
    assert ledger["native_cuda_verified"] is False
    assert ledger["authentic_source_certified"] is False
    assert ledger["declared_synthetic_seam"] is True
    assert ledger["device_guard"] == "synthetic-injected"


def test_spoof_guard_and_undeclared_guard_refused(tmp_path):
    setup = make_setup(tmp_path, 4)
    setup["device_guard"] = SpoofGuard()
    with pytest.raises(CaconStage2EpochFailure, match="subclasses overriding"):
        run(setup)

    setup = make_setup(tmp_path, 4)
    setup["device_guard"] = UndeclaredGuard()
    with pytest.raises(CaconStage2EpochFailure, match="synthetic=True"):
        run(setup)


def test_native_guard_requires_native_transfer(tmp_path):
    setup = make_setup(tmp_path, 4)
    setup["device_guard"] = CudaTensorGuard()
    setup["device_transfer"] = FakeCudaTransfer("cpu")
    with pytest.raises(CaconStage2EpochFailure, match="approved CudaTransfer seam"):
        run(setup)


def test_native_guard_rejects_cpu_batches_without_transfer(tmp_path):
    setup = make_setup(tmp_path, 4)
    setup["device_guard"] = CudaTensorGuard()
    setup.pop("device_transfer")
    with pytest.raises(CaconStage2EpochFailure, match="CUDA") as info:
        run(setup)
    assert info.value.ledger["samples"] == 0


def test_resolved_device_must_match_module_device(tmp_path, monkeypatch):
    import scripts.cacon_stage2_epoch_cuda_v2 as module

    setup = make_setup(tmp_path, 4)
    monkeypatch.setattr(module, "_module_devices", lambda mod: {"cuda:0"})
    setup["device_transfer"] = FakeCudaTransfer("cuda:1")
    with pytest.raises(CaconStage2EpochFailure, match="match the resolved device") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_mixed_backbone_classifier_devices_refused(tmp_path, monkeypatch):
    import scripts.cacon_stage2_epoch_cuda_v2 as module

    setup = make_setup(tmp_path, 4)
    monkeypatch.setattr(
        module,
        "_module_devices",
        lambda mod: {"cuda:0"} if isinstance(mod, MiniBackbone) else {"cuda:1"},
    )
    with pytest.raises(CaconStage2EpochFailure, match="mixed-device") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_transfer_that_does_not_move_tensors_is_refused(tmp_path, monkeypatch):
    import scripts.cacon_stage2_epoch_cuda_v2 as module

    setup = make_setup(tmp_path, 4)
    monkeypatch.setattr(module, "_module_devices", lambda mod: {"cuda:0"})
    setup["device_transfer"] = UnmovedTransfer()
    with pytest.raises(CaconStage2EpochFailure, match="transferred batch device") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_transfer_is_invoked_exactly_once_per_tensor_per_batch(tmp_path):
    setup = make_setup(tmp_path, 6)
    transfer = FakeCudaTransfer("cpu")
    setup["device_transfer"] = transfer
    run(setup)
    assert transfer.moves == ["cpu"] * 6


# ------------------------------------------------------------ finite gates
def test_nonfinite_input_refused_before_step(tmp_path, monkeypatch):
    setup = make_setup(tmp_path, 4)
    original = Stage2LabelDataset.__getitem__

    def corrupt(self, index):
        tensor, label = original(self, index)
        tensor[0, 0, 0] = float("nan")
        return tensor, label

    monkeypatch.setattr(Stage2LabelDataset, "__getitem__", corrupt)
    with pytest.raises(CaconStage2EpochFailure, match="nonfinite input") as info:
        run(setup)
    assert info.value.ledger["inputs_finite"] is False
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_nonfinite_logits_refused_before_step(tmp_path, monkeypatch):
    setup = make_setup(tmp_path, 4)
    stage = setup["stage"]
    monkeypatch.setattr(
        stage, "forward", lambda images: torch.full((len(images), stage.classes), float("inf"))
    )
    with pytest.raises(CaconStage2EpochFailure, match="nonfinite classifier logits") as info:
        run(setup)
    assert info.value.ledger["logits_finite"] is False
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_nonfinite_loss_refused_before_step(tmp_path, monkeypatch):
    import scripts.cacon_stage2_epoch_cuda_v2 as module

    setup = make_setup(tmp_path, 4)
    stage = setup["stage"]
    monkeypatch.setattr(stage, "forward", lambda images: torch.zeros(len(images), stage.classes))
    monkeypatch.setattr(module, "_cross_entropy", lambda *a, **k: torch.tensor(float("nan")))
    with pytest.raises(CaconStage2EpochFailure, match="nonfinite cross-entropy loss") as info:
        run(setup)
    assert info.value.ledger["losses_finite"] is False
    assert info.value.ledger["optimizer_step_attempts"] == 0


# ------------------------------------------------------- coverage / contracts
def test_single_pass_coverage_face_order_and_encoder_exposure(tmp_path):
    setup = make_setup(tmp_path, 7)
    ledger = run(setup)
    assert ledger["epoch_complete"] is True and ledger["single_epoch_coverage"] is True
    assert ledger["samples"] == 7 and ledger["source_unique_rows"] == 7
    assert ledger["encoder_image_instances"] == 7
    assert ledger["generator_image_instances"] == 0
    assert ledger["fas_encoder_image_instances"] == 0
    assert ledger["completed_batches"] == 4 and ledger["expected_batches"] == 4
    assert ledger["optimizer_step_attempts"] == 4 and ledger["optimizer_steps"] == 4
    assert ledger["input_face_ids"] == [f"face{i}" for i in range(7)]
    assert ledger["ordered_face_ids"] == [
        f"face{index}" for index in ledger["ordered_source_indices"]
    ]
    assert sorted(ledger["ordered_source_indices"]) == list(range(7))
    assert ledger["sample_weighted_loss"] is not None


def test_tail_singleton_batch_is_a_legitimate_stage2_step(tmp_path):
    ledger = run(make_setup(tmp_path, 5))
    assert [row["size"] for row in ledger["batch_summaries"]] == [2, 2, 1]
    assert ledger["optimizer_steps"] == 3 and ledger["samples"] == 5
    assert ledger["bn_running_stats_unchanged"] is True


def test_batch_size_one_covers_every_sample(tmp_path):
    ledger = run(make_setup(tmp_path, 3), batch_size=1)
    assert [row["size"] for row in ledger["batch_summaries"]] == [1, 1, 1]
    assert ledger["optimizer_steps"] == 3 and ledger["encoder_image_instances"] == 3


def test_freeze_survives_caller_train_and_embedding_is_unmoved(tmp_path):
    setup = make_setup(tmp_path, 4)
    stage = setup["stage"]
    before_state = copy.deepcopy(stage.backbone.state_dict())
    images = next(iter(torch.utils.data.DataLoader(setup["dataset"], batch_size=4)))[0]
    before_features = stage.verification_features(
        images, representation="backbone", normalize=False
    )

    ledger = run(setup, progress=lambda _: stage.train())

    assert ledger["backbone_frozen"] is True and ledger["backbone_eval"] is True
    assert ledger["encoder_gradients_absent"] is True
    assert ledger["bn_running_stats_unchanged"] is True
    assert not stage.backbone.training
    assert all(not p.requires_grad and p.grad is None for p in stage.backbone.parameters())
    for key, value in stage.backbone.state_dict().items():
        torch.testing.assert_close(value, before_state[key])
    torch.testing.assert_close(
        before_features,
        stage.verification_features(images, representation="backbone", normalize=False),
    )


def test_continuing_generator_advances_and_is_deterministic(tmp_path):
    setup = make_setup(tmp_path, 6)
    ledger = run(setup)
    assert ledger["torch_generator_current_sha256"] != ledger["torch_generator_initial_sha256"]
    assert (
        run(make_setup(tmp_path, 6))["ordered_source_indices"] == ledger["ordered_source_indices"]
    )


def test_ledger_declaration_and_honest_flags(tmp_path):
    ledger = run(make_setup(tmp_path, 4))
    assert "does not improve or move" in REPRESENTATION_DECLARATION
    assert ledger["full_method_parity"] is False
    assert ledger["common_budget_matched"] is False
    assert ledger["scientific_evaluation_complete"] is False
    assert ledger["publication_ready"] is False
    assert ledger["mined_identity_labels_used"] is True
    assert ledger["mined_identity_labels_human_cleared"] is False
    assert ledger["never_score_external"] is True and ledger["never_select_checkpoint"] is True


def test_class_count_mismatch_refused_before_step(tmp_path):
    setup = make_setup(tmp_path, 4)
    stage = setup["stage"]
    mismatched = FinalLinearStage(stage.backbone, 8, stage.classes + 1)
    setup["stage"] = mismatched
    setup["optimizer"] = CountingOptimizer(list(mismatched.classifier.parameters()))
    with pytest.raises(CaconStage2EpochFailure, match="class count must match") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_declared_source_face_id_order_enforced(tmp_path):
    setup = make_setup(tmp_path, 4)
    declared = [record.face_id for record in setup["dataset"].records]
    assert run(setup, source_face_ids=declared)["source_face_ids_declared"] is True
    with pytest.raises(CaconStage2EpochFailure, match="declared source face-ID list") as info:
        run(setup, source_face_ids=list(reversed(declared)))
    assert info.value.ledger["optimizer_step_attempts"] == 0


def test_cpu_device_and_bn_policy_and_generator_contracts(tmp_path):
    with pytest.raises(CaconStage2EpochFailure, match="no CPU fallback"):
        run(make_setup(tmp_path, 4), device="cpu")
    with pytest.raises(CaconStage2EpochFailure, match="frozen_all BatchNorm policy"):
        run(make_setup(tmp_path, 4), bn_policy="train_all")
    setup = make_setup(tmp_path, 4)
    setup["torch_generator"] = np.random.default_rng(0)
    with pytest.raises(CaconStage2EpochFailure, match="CPU torch.Generator"):
        run(setup)


@pytest.mark.parametrize("classes", [True, 3.0])
def test_root_class_count_requires_actual_integer(tmp_path, classes):
    setup = make_setup(tmp_path, 4)
    setup["dataset"].n_classes = classes
    with pytest.raises(CaconStage2EpochFailure, match="non-boolean integer dataset") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0
    assert info.value.ledger["authentic_source_certified"] is False


def test_root_changed_source_paths_refused_before_second_step(tmp_path):
    setup = make_setup(tmp_path, 4)
    dataset = setup["dataset"]

    def mutate(_):
        records = list(dataset.records)
        first = records[0]
        records[0] = Stage2Record(first.face_id, first.identity, records[1].path)
        dataset.records = tuple(records)

    with pytest.raises(CaconStage2EpochFailure, match="source records changed") as info:
        run(setup, progress=mutate)
    assert info.value.ledger["optimizer_step_attempts"] == 1
    assert setup["optimizer"].attempts == 1


def test_root_two_epochs_continue_generator_and_keep_full_source_coverage(tmp_path):
    setup = make_setup(tmp_path, 7)
    first = run(setup)
    second = run(setup)
    assert second["torch_generator_initial_sha256"] == first["torch_generator_current_sha256"]
    assert first["samples"] == second["samples"] == 7
    assert first["encoder_image_instances"] == second["encoder_image_instances"] == 7
    assert setup["optimizer"].attempts == 8


def test_root_transfer_cannot_change_identity_labels(tmp_path):
    setup = make_setup(tmp_path, 4)

    class CorruptLabels(FakeCudaTransfer):
        def to_device(self, tensor, device):
            result = super().to_device(tensor, device)
            return (result + 1) % 3 if result.dtype == torch.int64 else result

    setup["device_transfer"] = CorruptLabels()
    with pytest.raises(CaconStage2EpochFailure, match="transferred labels changed") as info:
        run(setup)
    assert info.value.ledger["optimizer_step_attempts"] == 0
    assert info.value.ledger["label_provenance_verified"] is False
