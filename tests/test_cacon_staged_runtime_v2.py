"""Pure synthetic integration tests for the CACon staged-runtime wrapper v2.

Chain the canonical ``run_cacon_stage1_epoch`` and ``run_cacon_stage2_epoch``
through the v2 transition wrapper with tiny CPU modules and counting fake
optimizers that never update parameters. Datasets are authentic (tiny generated
PNGs, cache-hash verified) but synthetic in content.

No GPU, no real faces/weights/model inference, no real ``torch.optim`` step, no
downloads. The three seeds exercised are a regression sweep, not scientific
replications, and a passing suite is NOT native CUDA, budget or scientific
evidence.
"""

# ruff: noqa: E402

from __future__ import annotations

import hashlib
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
from age_gap.training.sota_common import ProjectionHead  # noqa: E402
from scripts.cacon_dataset_v2 import ThreeViewDataset, ThreeViewRecord  # noqa: E402
from scripts.cacon_stage1_epoch_cuda_v2 import (  # noqa: E402
    CudaTensorGuard,
    CudaTransfer,
)
from scripts.cacon_stage2_epoch_cuda_v2 import (  # noqa: E402
    Stage2LabelDataset,
    Stage2Record,
)
from scripts.cacon_staged_runtime_v2 import (  # noqa: E402
    PROSPECTIVE_ALLOCATION,
    CaconTransitionFailure,
    _place_classifier,
    _resolve_stage2_placement,
    _source_snapshot,
    run_cacon_staged_training,
)
from scripts.cacon_supervised_v2 import FinalLinearStage  # noqa: E402


class SyntheticGuard:
    name = "synthetic-injected"
    synthetic = True

    def __init__(self):
        self.calls = []

    def require_module(self, module, role):
        self.calls.append(("module", role))

    def require_tensor(self, tensor, role, device=None):
        self.calls.append(("tensor", role))


class FakeCudaTransfer:
    """Synthetic transfer seam; declares non-native and moves to a real device."""

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


class CountingOptimizer:
    """Deliberately not a torch optimizer: counts steps and never updates params."""

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
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, dimension)

    def forward(self, images):
        x = torch.relu(self.bn(self.conv(images)))
        return self.fc(x.mean(dim=(2, 3)))


def write_image(path, value):
    assert cv2.imwrite(str(path), np.full((8, 6, 3), value, dtype=np.uint8))


def float32_preprocess(image):
    return image.transpose(2, 0, 1).astype(np.float32) / 255


def float64_preprocess(image):
    return image.transpose(2, 0, 1).astype(np.float64) / 255


def make_three_view(tmp_path, count, preprocess, *, seed=42):
    records = []
    for index in range(count):
        source = tmp_path / f"s{index}.png"
        generated = tmp_path / f"g{index}.png"
        write_image(source, index)
        write_image(generated, 100 + index)
        records.append(ThreeViewRecord(f"face{index}", file_record(source), file_record(generated)))
    return ThreeViewDataset(records, preprocess, np.random.default_rng(seed))


def make_stage2(tmp_path, count, preprocess, *, classes=3):
    records = []
    for index in range(count):
        source = tmp_path / f"s{index}.png"
        records.append(Stage2Record(f"face{index}", index % classes, source))
    return Stage2LabelDataset(records, preprocess)


def make_setup(tmp_path, count=6, *, classes=3, seed=5, float64=False):
    torch.manual_seed(seed)
    preprocess = float64_preprocess if float64 else float32_preprocess
    stage1_dataset = make_three_view(tmp_path, count, preprocess, seed=seed)
    stage2_dataset = make_stage2(tmp_path, count, preprocess, classes=classes)
    backbone = MiniBackbone()
    projector = ProjectionHead(8)
    if float64:
        backbone.double()
        projector.double()
    state = {"stage1": 0, "stage2": 0}
    observed = {}

    def stage1_factory(back, proj):
        state["stage1"] += 1
        return CountingOptimizer([*back.parameters(), *proj.parameters()])

    def stage2_factory(stage):
        state["stage2"] += 1
        observed["stage"] = stage
        observed["factory_classifier_device"] = {
            str(parameter.device) for parameter in stage.classifier.parameters()
        }
        observed["factory_classifier_dtype"] = {
            parameter.dtype for parameter in stage.classifier.parameters()
        }
        return CountingOptimizer(list(stage.classifier.parameters()))

    return dict(
        backbone=backbone,
        projector=projector,
        stage1_dataset=stage1_dataset,
        stage2_dataset=stage2_dataset,
        stage1_optimizer_factory=stage1_factory,
        stage2_optimizer_factory=stage2_factory,
        torch_generator=torch.Generator().manual_seed(seed),
        dimension=8,
        batch_size=2,
        device="cuda",
        bn_policy="frozen_all",
        tail_policy="reject",
        contrastive_passes=4,
        final_linear_passes=4,
        synthetic_probe=False,
        device_guard=SyntheticGuard(),
        device_transfer=FakeCudaTransfer("cpu"),
        _state=state,
        _observed=observed,
    )


def run(setup, **overrides):
    kwargs = {key: value for key, value in setup.items() if not key.startswith("_")}
    kwargs.update(overrides)
    return run_cacon_staged_training(**kwargs)


def _stage2_optimizer(setup):
    return setup["_observed"].get("optimizer")


# ---------------------------------------------------------------------------------------
# (1) Classifier placement on the resolved device/dtype BEFORE the optimizer factory
# ---------------------------------------------------------------------------------------
def test_classifier_is_placed_before_stage2_factory_via_dtype(tmp_path):
    # A float64 backbone makes placement observable on CPU: a classifier moved
    # after the factory would be seen as float32 inside the factory.
    setup = make_setup(tmp_path, 6, float64=True)
    ledger = run(setup)
    observed = setup["_observed"]
    assert observed["factory_classifier_dtype"] == {torch.float64}
    assert observed["factory_classifier_device"] == {"cpu"}
    assert ledger["stage2_classifier_dtype"] == "torch.float64"
    assert ledger["stage2_classifier_device"] == "cpu"
    assert ledger["stage2_placement_before_factory"] is True
    assert observed["factory_classifier_device"] == {ledger["stage2_classifier_device"]}
    assert (
        str(next(iter(observed["factory_classifier_dtype"]))) == ledger["stage2_classifier_dtype"]
    )


def test_resolved_placement_matches_observed_stage1_device(tmp_path):
    setup = make_setup(tmp_path, 6)
    ledger = run(setup)
    device, dtype = _resolve_stage2_placement(
        setup["backbone"], setup["projector"], setup["device_transfer"]
    )
    assert ledger["transition_device_matches_stage1"] is True
    assert str(device) == ledger["stage1_ledgers"][0]["resolved_device"]
    assert dtype == torch.float32
    assert all(str(parameter.device) == str(device) for parameter in setup["backbone"].parameters())


def test_noop_placement_is_refused_with_float64_backbone(tmp_path):
    setup = make_setup(tmp_path, 6, float64=True)
    import scripts.cacon_staged_runtime_v2 as wrapper

    original = wrapper._place_classifier
    wrapper._place_classifier = lambda classifier, device, dtype: classifier  # no-op
    try:
        with pytest.raises(CaconTransitionFailure, match="placement") as info:
            run(setup)
    finally:
        wrapper._place_classifier = original
    assert info.value.ledger["passes_completed"] == 4
    assert info.value.ledger["stage2_placement_before_factory"] is False


def test_factory_observes_no_parameter_movement(tmp_path):
    # The factory receives the already-placed classifier; nothing moves after it.
    setup = make_setup(tmp_path, 6, float64=True)
    ledger = run(setup)
    observed = setup["_observed"]
    assert observed["factory_classifier_dtype"] == {torch.float64}
    assert ledger["stage2_classifier_dtype"] == "torch.float64"


def test_backbone_object_is_never_moved_or_changed(tmp_path):
    setup = make_setup(tmp_path, 6)
    backbone = setup["backbone"]
    before = [parameter.data_ptr() for parameter in backbone.parameters()]
    ledger = run(setup)
    assert ledger["backbone_object_preserved"] is True
    assert ledger["same_backbone_object"] is True
    assert [parameter.data_ptr() for parameter in backbone.parameters()] == before
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_place_classifier_rejects_wrong_dtype():
    class InertModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, 2))

        def to(self, *args, **kwargs):  # deliberately ignores the requested dtype
            return self

    with pytest.raises(RuntimeError, match="placement onto the resolved dtype failed"):
        _place_classifier(InertModule(), torch.device("cpu"), torch.float64)


# ---------------------------------------------------------------------------------------
# (2) Entry validation before ANY stage1 step
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("dimension", [0, -3, True, "8", 1.5, None])
def test_invalid_dimension_fails_at_zero_stage1_attempts(tmp_path, dimension):
    setup = make_setup(tmp_path, 6)
    with pytest.raises(CaconTransitionFailure, match="dimension") as info:
        run(setup, dimension=dimension)
    assert info.value.ledger["passes_completed"] == 0
    assert setup["_state"]["stage1"] == 0
    assert info.value.ledger["stage2_placement_before_factory"] is False


@pytest.mark.parametrize("normalize_input", [0, 1, None, "yes"])
def test_invalid_normalize_input_fails_at_zero_stage1_attempts(tmp_path, normalize_input):
    setup = make_setup(tmp_path, 6)
    with pytest.raises(CaconTransitionFailure, match="normalize_input") as info:
        run(setup, normalize_input=normalize_input)
    assert info.value.ledger["passes_completed"] == 0
    assert setup["_state"]["stage1"] == 0


def test_stale_stage2_n_classes_fails_at_zero_stage1_attempts(tmp_path):
    setup = make_setup(tmp_path, 6)
    setup["stage2_dataset"].n_classes = 99
    with pytest.raises(CaconTransitionFailure, match="contiguous label set") as info:
        run(setup)
    assert info.value.ledger["passes_completed"] == 0
    assert setup["_state"]["stage1"] == 0


def test_boolean_stage2_n_classes_fails_at_zero_stage1_attempts(tmp_path):
    setup = make_setup(tmp_path, 6)
    setup["stage2_dataset"].n_classes = True
    with pytest.raises(CaconTransitionFailure, match="n_classes") as info:
        run(setup)
    assert setup["_state"]["stage1"] == 0
    assert info.value.ledger["passes_completed"] == 0


def test_noncontiguous_stage2_labels_fail_at_zero_stage1_attempts(tmp_path):
    setup = make_setup(tmp_path, 6)
    broken = setup["stage2_dataset"]
    broken.records = tuple(broken.records[:-1]) + (
        Stage2Record(broken.records[-1].face_id, 7, broken.records[-1].path),
    )
    broken.n_classes = len({record.identity for record in broken.records})
    with pytest.raises(CaconTransitionFailure, match="contiguous") as info:
        run(setup)
    assert info.value.ledger["passes_completed"] == 0
    assert setup["_state"]["stage1"] == 0


def test_boolean_stage2_label_fails_at_zero_stage1_attempts(tmp_path):
    setup = make_setup(tmp_path, 6)
    broken = setup["stage2_dataset"]
    broken.records = (
        Stage2Record(broken.records[0].face_id, True, broken.records[0].path),
        *broken.records[1:],
    )
    with pytest.raises(CaconTransitionFailure, match="integer stage2 identity labels") as info:
        run(setup)
    assert setup["_state"]["stage1"] == 0
    assert info.value.ledger["stage2_classes"] == 0


def test_invalid_dimension_does_not_freeze_backbone_early(tmp_path):
    setup = make_setup(tmp_path, 6)
    backbone = setup["backbone"]
    with pytest.raises(CaconTransitionFailure, match="dimension"):
        run(setup, dimension=0)
    assert all(parameter.requires_grad for parameter in backbone.parameters())
    assert backbone.training is True


# ---------------------------------------------------------------------------------------
# (3) Exact authentic type + source revalidation across callbacks
# ---------------------------------------------------------------------------------------
def test_forged_three_view_subclass_refused(tmp_path):
    class ForgedThreeViewDataset(ThreeViewDataset):
        pass

    setup = make_setup(tmp_path, 6)
    forged = ForgedThreeViewDataset(
        list(setup["stage1_dataset"].records), float32_preprocess, np.random.default_rng(0)
    )
    setup["stage1_dataset"] = forged
    with pytest.raises(CaconTransitionFailure, match="exact type") as info:
        run(setup)
    assert info.value.ledger["passes_completed"] == 0
    assert setup["_state"]["stage1"] == 0


def test_callback_stage2_path_mutation_detected_before_next_step(tmp_path):
    setup = make_setup(tmp_path, 6)
    state = {"calls": 0}
    original_path = setup["stage2_dataset"].records[0].path
    other = tmp_path / "other.png"
    write_image(other, 9)

    def mutating(row):
        state["calls"] += 1
        if state["calls"] == 4:  # after the 4th stage1 pass, before stage2
            dataset = setup["stage2_dataset"]
            dataset.records = (
                Stage2Record(dataset.records[0].face_id, dataset.records[0].identity, other),
                *dataset.records[1:],
            )

    with pytest.raises(CaconTransitionFailure, match="changed during the run") as info:
        run(setup, progress=mutating)
    ledger = info.value.ledger
    assert ledger["passes_completed"] == 4
    assert len(ledger["stage1_ledgers"]) == 4
    assert all(row["single_epoch_coverage"] for row in ledger["stage1_ledgers"])
    assert ledger["stage2_source_records"] == 0
    assert original_path is not None


def test_callback_stage1_source_path_mutation_detected_before_next_step(tmp_path):
    setup = make_setup(tmp_path, 6)
    state = {"calls": 0}

    def mutating(row):
        state["calls"] += 1
        if state["calls"] == 1:
            setup["stage1_dataset"].records[0].source_record["path"] = "missing.png"

    with pytest.raises(CaconTransitionFailure, match="changed during the run") as info:
        run(setup, progress=mutating)
    ledger = info.value.ledger
    assert ledger["passes_completed"] == 1
    assert len(ledger["stage1_ledgers"]) == 1
    assert ledger["stage1_source_records"] == 6


def test_callback_label_mutation_detected_before_next_step(tmp_path):
    setup = make_setup(tmp_path, 6)
    state = {"calls": 0}

    def mutating(row):
        state["calls"] += 1
        if state["calls"] == 4:
            dataset = setup["stage2_dataset"]
            dataset.records = (
                Stage2Record(
                    dataset.records[0].face_id,
                    dataset.records[0].identity + 1,
                    dataset.records[0].path,
                ),
                *dataset.records[1:],
            )

    with pytest.raises(CaconTransitionFailure, match="changed during the run") as info:
        run(setup, progress=mutating)
    assert info.value.ledger["passes_completed"] == 4


def test_source_contract_revalidated_before_every_pass(tmp_path):
    setup = make_setup(tmp_path, 6)
    ledger = run(setup)
    # before each of the 8 passes, at the transition and after the final pass
    assert ledger["source_contract_revalidations"] == 4 + 4 + 1 + 1
    expected = hashlib.sha256(
        repr(_source_snapshot(setup["stage1_dataset"], setup["stage2_dataset"])).encode()
    ).hexdigest()
    assert ledger["source_contract_sha256"] == expected


# ---------------------------------------------------------------------------------------
# (4) Observed aggregation from completed AND partial stage ledgers
# ---------------------------------------------------------------------------------------
def test_failed_stage2_pass_aggregates_observed_partial_counts(tmp_path, monkeypatch):
    setup = make_setup(tmp_path, 6)
    calls = {"count": 0}
    original = Stage2LabelDataset.__getitem__

    def failing(self, index):
        calls["count"] += 1
        if calls["count"] > 3:  # 3 batches settle, then refuse the 4th dataset read
            raise RuntimeError("synthetic mid-pass stage2 failure")
        return original(self, index)

    monkeypatch.setattr(Stage2LabelDataset, "__getitem__", failing)
    with pytest.raises(CaconTransitionFailure, match="stage2 pass") as info:
        run(setup)
    ledger = info.value.ledger
    partial = ledger["stage2_ledgers"][0]
    assert partial["optimizer_steps"] > 0
    assert ledger["stage2_optimizer_steps"] == partial["optimizer_steps"]
    assert ledger["stage2_optimizer_step_attempts"] == partial["optimizer_step_attempts"]
    assert ledger["stage2_source_records"] == partial["samples"]
    assert ledger["stage2_encoder_image_instances"] == partial["encoder_image_instances"]
    assert ledger["observed_optimizer_steps"] == (
        ledger["stage1_optimizer_steps"] + partial["optimizer_steps"]
    )
    assert ledger["observed_samples"] == ledger["stage1_source_records"] + partial["samples"]
    assert ledger["stage1_source_records"] == 4 * 6  # four completed stage1 passes


def test_failed_first_stage2_pass_keeps_stage1_observed_counts(tmp_path, monkeypatch):
    setup = make_setup(tmp_path, 6)

    def failing(self, index):
        raise RuntimeError("synthetic immediate stage2 failure")

    monkeypatch.setattr(Stage2LabelDataset, "__getitem__", failing)
    with pytest.raises(CaconTransitionFailure, match="stage2 pass") as info:
        run(setup)
    ledger = info.value.ledger
    assert ledger["passes_completed"] == 4
    assert ledger["stage1_source_records"] == 24
    assert ledger["stage1_encoder_image_instances"] == 4 * 3 * 6
    assert ledger["stage1_optimizer_steps"] == 12
    assert ledger["stage2_source_records"] == 0
    assert ledger["observed_optimizer_steps"] == 12


# ---------------------------------------------------------------------------------------
# (5) Per-pass RNG continuity and callback reseeding rejection
# ---------------------------------------------------------------------------------------
def test_rng_continuity_checked_at_every_pass_boundary(tmp_path):
    setup = make_setup(tmp_path, 6)
    ledger = run(setup)
    assert ledger["rng_continuity_checks"] == 4 + 4
    assert ledger["generator_advanced"] is True
    assert ledger["numpy_rng_advanced"] is True
    # every stage1 pass advanced the generator; the stage2 stage continues the stream
    shas = [row["torch_generator_current_sha256"] for row in ledger["stage1_ledgers"]]
    assert len(set(shas)) == len(shas)
    assert (
        ledger["stage1_ledgers"][-1]["torch_generator_current_sha256"]
        == (ledger["stage2_ledgers"][0]["torch_generator_initial_sha256"])
    )


def test_callback_reseeding_generator_is_rejected_with_honest_counts(tmp_path):
    setup = make_setup(tmp_path, 6)
    generator = setup["torch_generator"]
    state = {"calls": 0}

    def reseeding(row):
        state["calls"] += 1
        if state["calls"] == 2:
            generator.manual_seed(1234)

    with pytest.raises(CaconTransitionFailure, match="reseeded or mutated") as info:
        run(setup, progress=reseeding)
    ledger = info.value.ledger
    assert ledger["passes_completed"] == 2
    assert len(ledger["stage1_ledgers"]) == 2
    assert ledger["stage1_source_records"] == 12


def test_callback_replacing_numpy_rng_is_rejected(tmp_path):
    setup = make_setup(tmp_path, 6)
    state = {"calls": 0}

    def replacing(row):
        state["calls"] += 1
        if state["calls"] == 1:
            setup["stage1_dataset"].rng = np.random.default_rng(0)

    with pytest.raises(CaconTransitionFailure, match="replaced the NumPy RNG") as info:
        run(setup, progress=replacing)
    ledger = info.value.ledger
    assert ledger["passes_completed"] == 1
    assert ledger["stage1_source_records"] == 6


def test_benign_progress_callback_is_accepted(tmp_path):
    setup = make_setup(tmp_path, 6)
    seen = []
    ledger = run(
        setup, progress=lambda row: seen.append((row["passes_completed"], row["complete"]))
    )
    assert [count for count, _ in seen] == list(range(1, 9))
    assert seen[-1][1] is False
    assert ledger["passes_completed"] == 8 and ledger["complete"] is True


# ---------------------------------------------------------------------------------------
# Allocation, source contract and non-claims
# ---------------------------------------------------------------------------------------
def test_prospective_4_plus_4_runs_and_reconciles(tmp_path):
    setup = make_setup(tmp_path, 6)
    ledger = run(setup)
    assert PROSPECTIVE_ALLOCATION == (4, 4)
    assert ledger["allocation"] == "prospective_4+4"
    assert ledger["allocation_matches_prospective"] is True
    assert ledger["passes_completed"] == 8
    assert ledger["source_records_per_pass"] == 6
    assert ledger["stage1_source_records"] == 24
    assert ledger["stage2_source_records"] == 24
    assert ledger["observed_samples"] == 48
    assert ledger["stage1_encoder_image_instances"] == 4 * 3 * 6
    assert ledger["stage2_encoder_image_instances"] == 4 * 6
    assert ledger["total_encoder_image_instances"] == 4 * 3 * 6 + 4 * 6
    assert ledger["observed_optimizer_steps"] == 12 + 12
    assert ledger["complete"] is True
    assert len(ledger["stage1_ledgers"]) == 4 and len(ledger["stage2_ledgers"]) == 4


def test_factories_are_called_once_and_optimizer_is_continued(tmp_path):
    setup = make_setup(tmp_path, 6)
    ledger = run(setup)
    assert setup["_state"]["stage1"] == 1  # one continuing stage1 optimizer across 4 passes
    assert setup["_state"]["stage2"] == 1  # one stage2 optimizer across 4 passes
    assert ledger["observed_optimizer_steps"] == 12 + 12


def test_partial_run_leaves_training_flags_false(tmp_path, monkeypatch):
    setup = make_setup(tmp_path, 6)

    def failing(self, index):
        raise RuntimeError("synthetic stage2 failure")

    monkeypatch.setattr(Stage2LabelDataset, "__getitem__", failing)
    with pytest.raises(CaconTransitionFailure, match="stage2 pass") as info:
        run(setup)
    ledger = info.value.ledger
    assert ledger["complete"] is False
    assert ledger["training_complete"] is False
    assert ledger["full_method_parity"] is False
    assert ledger["scientific_evaluation_complete"] is False
    assert ledger["publication_ready"] is False
    assert ledger["stage1_source_records"] == 24  # observed, never a planned zero
    assert ledger["observed_optimizer_steps"] == 12


def test_completed_run_still_never_claims_training_or_science(tmp_path):
    setup = make_setup(tmp_path, 6)
    ledger = run(setup)
    assert ledger["complete"] is True
    assert ledger["training_complete"] is False
    assert ledger["full_method_parity"] is False
    assert ledger["common_budget_matched"] is False
    assert ledger["scientific_evaluation_complete"] is False
    assert ledger["publication_ready"] is False


def test_short_probe_requires_explicit_tag(tmp_path):
    setup = make_setup(tmp_path, 6)
    with pytest.raises(CaconTransitionFailure, match="synthetic_probe=True") as info:
        run(setup, contrastive_passes=1, final_linear_passes=1)
    assert info.value.ledger["passes_completed"] == 0
    assert setup["_state"]["stage1"] == 0


def test_short_probe_is_tagged_not_scientific(tmp_path):
    setup = make_setup(tmp_path, 6)
    ledger = run(setup, contrastive_passes=1, final_linear_passes=1, synthetic_probe=True)
    assert ledger["allocation"] == "synthetic_probe"
    assert ledger["allocation_matches_prospective"] is False
    assert ledger["full_method_parity"] is False
    assert ledger["common_budget_matched"] is False
    assert ledger["scientific_evaluation_complete"] is False
    assert ledger["publication_ready"] is False


def test_three_seed_regression_sweep(tmp_path):
    observed = []
    for seed in (42, 1, 2):
        seed_dir = tmp_path / f"seed{seed}"
        seed_dir.mkdir()
        ledger = run(make_setup(seed_dir, 6, seed=seed))
        assert ledger["complete"] is True
        observed.append(ledger["generator_current_sha256"])
    assert len(set(observed)) == 3


def test_source_mismatch_is_rejected_at_zero_attempts(tmp_path):
    setup = make_setup(tmp_path, 6)
    broken = setup["stage2_dataset"]
    broken.records = tuple(broken.records[:-1]) + (
        Stage2Record("face-other", 0, broken.records[-1].path),
    )
    with pytest.raises(CaconTransitionFailure, match="must match exactly") as info:
        run(setup)
    assert info.value.ledger["passes_completed"] == 0
    assert setup["_state"]["stage1"] == 0


def test_generated_image_cannot_be_a_supervised_source(tmp_path):
    setup = make_setup(tmp_path, 6)
    from age_gap.common.io import PROJECT_ROOT

    generated_path = (
        PROJECT_ROOT / setup["stage1_dataset"].records[3].generated_record["path"]
    ).resolve()
    broken = setup["stage2_dataset"]
    broken.records = (
        *broken.records[:3],
        Stage2Record(broken.records[3].face_id, broken.records[3].identity, generated_path),
        *broken.records[4:],
    )
    with pytest.raises(CaconTransitionFailure, match="excluded from the supervised source") as info:
        run(setup)
    assert info.value.ledger["passes_completed"] == 0


def test_duplicate_face_ids_rejected(tmp_path):
    setup = make_setup(tmp_path, 6)
    broken = setup["stage2_dataset"]
    broken.records = tuple(broken.records[:-1]) + (
        Stage2Record(broken.records[0].face_id, 0, broken.records[-1].path),
    )
    with pytest.raises(CaconTransitionFailure, match="unique and nonempty") as info:
        run(setup)
    assert info.value.ledger["stage1_ledgers"] == []


def test_same_backbone_frozen_and_projector_excluded(tmp_path):
    setup = make_setup(tmp_path, 6)
    projector = setup["projector"]
    ledger = run(setup)
    assert ledger["stale_stage1_gradients_present_before_transition"] is True
    assert ledger["stale_gradients_cleared"] is True
    assert ledger["encoder_frozen"] is True
    assert ledger["projector_not_in_stage2_optimizer"] is True
    assert ledger["projector_not_forwarded_in_stage2"] is True
    assert all(parameter.requires_grad for parameter in projector.parameters())
    for row in ledger["stage2_ledgers"]:
        assert row["classifier_trainable_parameter_count"] == row["optimizer_parameter_count"]
        assert row["backbone_frozen"] is True and row["encoder_gradients_absent"] is True


def test_stage2_optimizer_receives_placed_classifier(tmp_path):
    setup = make_setup(tmp_path, 6)
    ledger = run(setup)
    stage = setup["_observed"]["stage"]
    assert type(stage) is FinalLinearStage
    assert stage.backbone is setup["backbone"]
    assert stage.classes == ledger["stage2_classes"] == setup["stage2_dataset"].n_classes
    assert all(
        str(parameter.device) == ledger["stage2_classifier_device"]
        for parameter in stage.classifier.parameters()
    )


def test_non_frozen_all_bn_policy_refused(tmp_path):
    setup = make_setup(tmp_path, 6)
    with pytest.raises(CaconTransitionFailure, match="frozen_all") as info:
        run(setup, bn_policy="train_all")
    assert info.value.ledger["passes_completed"] == 0


def test_projector_in_stage2_optimizer_refused(tmp_path):
    setup = make_setup(tmp_path, 6)
    projector = setup["projector"]

    def greedy_factory(stage):
        return CountingOptimizer([*stage.classifier.parameters(), *projector.parameters()])

    setup["stage2_optimizer_factory"] = greedy_factory
    with pytest.raises(CaconTransitionFailure, match="projector must not be supervised") as info:
        run(setup)
    assert info.value.ledger["projector_not_in_stage2_optimizer"] is False
    assert info.value.ledger["stage2_source_records"] == 0


def test_synthetic_seams_never_certify_native(tmp_path):
    setup = make_setup(tmp_path, 6)
    ledger = run(setup)
    assert ledger["native_cuda_verified"] is False
    assert ledger["declared_synthetic_seam"] is True
    for row in ledger["stage1_ledgers"]:
        assert row["native_cuda_verified"] is False
    for row in ledger["stage2_ledgers"]:
        assert row["native_cuda_verified"] is False
        assert row["authentic_source_certified"] is False


def test_native_guard_plus_synthetic_transfer_refused(tmp_path):
    setup = make_setup(tmp_path, 6)
    setup["device_guard"] = CudaTensorGuard()
    setup["device_transfer"] = FakeCudaTransfer("cpu")
    with pytest.raises(CaconTransitionFailure, match="approved CudaTransfer seam") as info:
        run(setup)
    assert info.value.ledger["passes_completed"] == 0


def test_full_native_seams_refuse_cpu_host_before_any_step(tmp_path):
    setup = make_setup(tmp_path, 6)
    setup["device_guard"] = CudaTensorGuard()
    setup["device_transfer"] = CudaTransfer()
    with pytest.raises(CaconTransitionFailure, match="CUDA") as info:
        run(setup)
    ledger = info.value.ledger
    assert ledger["passes_completed"] == 0
    assert ledger["native_cuda_verified"] is False
    assert ledger["total_encoder_image_instances"] == 0


def test_ledger_carries_non_claims_and_no_equal_compute(tmp_path):
    setup = make_setup(tmp_path, 6)
    ledger = run(setup)
    assert ledger["equal_compute_claim"] is False
    assert ledger["stage1_encoder_image_instances"] != ledger["stage2_encoder_image_instances"]
    assert ledger["full_method_parity"] is False
    assert ledger["common_budget_matched"] is False
    assert ledger["scientific_evaluation_complete"] is False
    assert ledger["publication_ready"] is False
    assert ledger["external_metrics_used"] is False
    assert ledger["checkpoint_selection"] == "none"
    assert ledger["generator_cache_production_costs_included"] is False
    assert ledger["third_view_production_unproven"] is True
    assert ledger["non_claims"] and all(isinstance(item, str) for item in ledger["non_claims"])


def test_input_type_validation(tmp_path):
    setup = make_setup(tmp_path, 6)
    with pytest.raises(CaconTransitionFailure, match="canonical ProjectionHead") as info:
        run(setup, projector=nn.Linear(8, 8))
    assert info.value.ledger["passes_completed"] == 0
    setup = make_setup(tmp_path, 6)
    with pytest.raises(CaconTransitionFailure, match="exact concrete stage2 dataset") as info:
        run(setup, stage2_dataset=nn.Linear(3, 2))
    assert info.value.ledger["passes_completed"] == 0
    setup = make_setup(tmp_path, 6)
    with pytest.raises(CaconTransitionFailure, match="continuing CPU torch.Generator") as info:
        run(setup, torch_generator=object())
    assert info.value.ledger["passes_completed"] == 0
    setup = make_setup(tmp_path, 6)
    with pytest.raises(CaconTransitionFailure, match="positive stage pass counts") as info:
        run(setup, contrastive_passes=0)
    assert info.value.ledger["passes_completed"] == 0



@pytest.mark.parametrize("stage", [1, 2])
def test_optimizer_factory_cannot_reseed_continuing_stream(tmp_path, stage):
    setup = make_setup(tmp_path, 6)
    name = f"stage{stage}_optimizer_factory"
    original = setup[name]
    def reseeding(*args):
        setup["torch_generator"].manual_seed(987)
        return original(*args)
    setup[name] = reseeding
    with pytest.raises(CaconTransitionFailure, match="optimizer factory"):
        run(setup)


def test_callback_cannot_replace_source_bytes_under_same_path(tmp_path):
    setup = make_setup(tmp_path, 6)
    def changing(row):
        if row["passes_completed"] == 4:
            write_image(setup["stage2_dataset"].records[0].path, 244)
    with pytest.raises(CaconTransitionFailure, match="bytes changed") as info:
        run(setup, progress=changing)
    assert info.value.ledger["passes_completed"] == 4
    assert info.value.ledger["stage2_source_records"] == 0


def test_callback_cannot_rebind_hash_to_changed_bytes(tmp_path):
    setup = make_setup(tmp_path, 6)
    def changing(row):
        if row["passes_completed"] == 1:
            path = setup["stage2_dataset"].records[0].path
            write_image(path, 244)
            setup["stage1_dataset"].records[0].source_record.update(file_record(path))
    with pytest.raises(CaconTransitionFailure, match="changed during the run") as info:
        run(setup, progress=changing)
    assert info.value.ledger["passes_completed"] == 1


def test_callback_invalid_class_declaration_refused_before_next_pass(tmp_path):
    setup = make_setup(tmp_path, 6)
    def changing(row):
        if row["passes_completed"] == 1:
            setup["stage2_dataset"].n_classes = True
    with pytest.raises(CaconTransitionFailure, match="n_classes") as info:
        run(setup, progress=changing)
    assert info.value.ledger["passes_completed"] == 1
