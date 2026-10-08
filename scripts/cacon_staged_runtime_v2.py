"""CACon staged-runtime orchestration, v2 (device/dtype, entry validation, observed ledgers).

Corrects five evidence gaps found in the v1 transition wrapper while keeping the
underlying canonical stage-1/stage-2 epochs and the 4+4 declared adaptation:

1. **Stage-2 classifier placement.** v1 built the canonical ``FinalLinearStage``
   whose new ``classifier`` is created on CPU, so a CUDA backbone made native
   stage-2 impossible (mixed-device). v2 resolves the *actual* stage-1 module
   device and floating dtype and explicitly transfers the **new classifier** to
   that device/dtype **before** the stage-2 optimizer factory runs. The backbone
   object is never moved or changed.
2. **Entry validation.** v1 validated ``dimension``/``normalize_input`` and the
   stage-2 class/label declarations only when ``FinalLinearStage`` was constructed,
   i.e. after four stage-1 passes. v2 validates all of them, plus live identity
   label contiguity, paths, IDs and lengths, before **any** stage-1 optimizer step,
   and without constructing the stage early (the backbone is not frozen early).
3. **Exact authentic stage-1 type + source revalidation.** v1 used ``isinstance``,
   so a forged subclass could claim authenticity. v2 requires the exact
   ``ThreeViewDataset`` type, snapshots the actual source IDs, source paths,
   generated paths and identity labels of both datasets, and revalidates that
   snapshot before every pass and at the transition, so a progress callback that
   mutates a record path, ID or label is refused before the next step.
4. **Observed, not planned, aggregation.** v1 assigned planned counts
   (``passes * N``) after the loops, so a failed stage-2 pass left the wrapper
   samples/steps at false zeros even though the partial row had real steps. v2
   aggregates observed samples/exposure/attempted/returned steps from all
   completed **and partial** stage ledgers.
5. **Per-pass RNG continuity.** v1 only checked a final advanced flag. v2 checks
   at every pass boundary that the continuing generator advances, the NumPy RNG
   advances during stage-1 passes, and neither the RNG object nor either stream is
   replaced or reseeded by a progress callback — with honest partial counts on
   refusal.

Scope / non-claims
------------------
Runtime orchestration only: not full CACon, no generator training, no cache/lineage
or production-cost proof, no native CUDA smoke evidence on this host, not a matched
budget, not a scientific evaluation, not publication ready, no author parity. The
4+4 allocation is a declared common-protocol adaptation from
``docs/SOTABudgetProtocol.md``; equal compute is not claimed (stage-1 registers
``3N`` encoder image instances per pass, stage-2 ``N`` frozen-encoder instances).
No external metric, no checkpoint selection, no artifact IO. Mined identity labels
are recorded, not human cleared.

Worker/test constraints
-----------------------
No GPU, no real faces/weights/model inference, no real ``torch.optim`` step, no
downloads: tests use tiny generated images, CPU forward/backward and a counting
fake optimizer that never updates parameters. Passing tests are NOT native CUDA or
scientific evidence.
"""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from torch import nn

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record
from age_gap.training.sota_common import MTLFaceDataset, ProjectionHead
from scripts.cacon_dataset_v2 import ThreeViewDataset
from scripts.cacon_stage1_epoch_cuda_v2 import (
    CudaTensorGuard,
    CudaTransfer,
    _generator_sha,
    run_cacon_stage1_epoch,
)
from scripts.cacon_stage2_epoch_cuda_v2 import (
    Stage2LabelDataset,
    _require_seams,
    run_cacon_stage2_epoch,
)
from scripts.cacon_supervised_v2 import FinalLinearStage

PROSPECTIVE_ALLOCATION = (4, 4)
PRIMARY_LOCATOR = "https://arxiv.org/html/2312.11195v2#S2.SS1"
PROTOCOL_LOCATOR = "docs/SOTABudgetProtocol.md#prospective-stage-allocation-2026-10-08"
OBJECTIVE = (
    "sequence the prospective 4+4 CACon adaptation: four stage-1 three-view instance "
    "NT-Xent source passes, then four stage-2 frozen-encoder final-linear source passes"
)
SCOPE = (
    "staged-runtime orchestration over the canonical stage1/stage2 epochs; not full "
    "CACon, not generator/cache lineage or production-cost proof, not a native CUDA "
    "smoke test, not a matched budget, not a scientific evaluation, not publication ready"
)
NON_CLAIMS = (
    "synthetic seams and synthetic-probe allocations never certify training, full method "
    "parity, a matched budget, a scientific evaluation or publication readiness",
    "equal-compute is not claimed: stage1 registers 3N encoder image instances per pass, "
    "stage2 registers N frozen-encoder instances",
    "third-view generation and generator training are separate unproven costs, not free",
    "no external metric, no checkpoint selection, no artifact IO",
    "mined identity labels are recorded, not human cleared",
)


class CaconTransitionFailure(RuntimeError):
    """Transition failure carrying the observed partial ledger (and stage ledgers)."""

    def __init__(self, message, ledger):
        super().__init__(message)
        self.ledger = ledger


def _numpy_rng_sha(rng):
    return hashlib.sha256(repr(rng.bit_generator.state).encode()).hexdigest()


def _three_view_source(dataset):
    rows = list(dataset.records)
    face_ids = [row.face_id for row in rows]
    source_paths = [str((PROJECT_ROOT / row.source_record["path"]).resolve()) for row in rows]
    generated_paths = [str((PROJECT_ROOT / row.generated_record["path"]).resolve()) for row in rows]
    return face_ids, source_paths, generated_paths


def _supervised_source(dataset):
    if type(dataset) is Stage2LabelDataset:
        return (
            [record.face_id for record in dataset.records],
            [str(record.path.resolve()) for record in dataset.records],
        )
    if type(dataset) is MTLFaceDataset:
        return (
            [item.path.stem for item in dataset.items],
            [str(item.path.resolve()) for item in dataset.items],
        )
    raise TypeError(
        "exact concrete Stage2LabelDataset or canonical MTLFaceDataset required for stage2"
    )


def _stage2_label_values(dataset):
    if type(dataset) is Stage2LabelDataset:
        return [record.identity for record in dataset.records]
    return [item.identity for item in dataset.items]


def _validated_stage2_classes(dataset):
    labels = _stage2_label_values(dataset)
    if any(isinstance(label, bool) or not isinstance(label, int) for label in labels):
        raise ValueError("non-boolean integer stage2 identity labels required")
    if any(label < 0 for label in labels):
        raise ValueError("nonnegative stage2 identity labels required")
    classes = len(set(labels))
    if set(labels) != set(range(classes)):
        raise ValueError("contiguous stage2 identity labels starting at zero required")
    declared = dataset.n_classes
    if isinstance(declared, bool) or not isinstance(declared, int):
        raise ValueError("non-boolean integer stage2 n_classes required")
    if declared != classes:
        raise ValueError("stage2 n_classes must match the live contiguous label set")
    return classes


def _source_snapshot(stage1_dataset, stage2_dataset):
    ids1, source1, generated1 = _three_view_source(stage1_dataset)
    ids2, paths2 = _supervised_source(stage2_dataset)
    labels2 = _stage2_label_values(stage2_dataset)
    return (
        tuple(ids1),
        tuple(source1),
        tuple(generated1),
        tuple(ids2),
        tuple(paths2),
        tuple(labels2),
        tuple(tuple(sorted(row.source_record.items())) for row in stage1_dataset.records),
        tuple(tuple(sorted(row.generated_record.items())) for row in stage1_dataset.records),
        stage2_dataset.n_classes,
    )


def _validate_source_contract(stage1_dataset, stage2_dataset):
    """Exact face-ID and source-path agreement, generated images excluded."""
    ids1, source1, generated1, ids2, paths2, _labels2 = _source_snapshot(
        stage1_dataset, stage2_dataset
    )[:6]
    if not ids1 or not source1:
        raise ValueError("nonempty stage1 three-view source required")
    if not all(isinstance(face_id, str) and face_id for face_id in ids1) or len(set(ids1)) != len(
        ids1
    ):
        raise ValueError("stage1 source face IDs must be unique and nonempty")
    if not all(isinstance(face_id, str) and face_id for face_id in ids2) or len(set(ids2)) != len(
        ids2
    ):
        raise ValueError("stage2 source face IDs must be unique and nonempty")
    if len(source1) != len(set(source1)):
        raise ValueError("stage1 source paths must be unique")
    if len(paths2) != len(set(paths2)):
        raise ValueError("stage2 source paths must be unique")
    if ids1 != ids2:
        raise ValueError("stage1/stage2 source face IDs must match exactly, in order")
    if set(paths2) & set(generated1):
        raise ValueError("generated third-view images must be excluded from the supervised source")
    if source1 != paths2:
        raise ValueError("stage1/stage2 source paths must match exactly, in order")
    if len(ids1) != len(stage1_dataset) or len(ids2) != len(stage2_dataset):
        raise ValueError("live source list must match the dataset length")
    _verify_bound_bytes(stage1_dataset)
    return len(ids1)


def _resolve_stage2_placement(backbone, projector, transfer):
    """Actual stage-1 module device + single floating dtype for classifier placement."""
    device = torch.device(transfer.resolve_device(backbone, projector))
    dtypes = {parameter.dtype for parameter in backbone.parameters()}
    if len(dtypes) != 1:
        raise ValueError("backbone must own a single floating dtype for classifier placement")
    dtype = next(iter(dtypes))
    if not dtype.is_floating_point:
        raise ValueError("floating-point backbone dtype required for classifier placement")
    return device, dtype


def _place_classifier(classifier, device, dtype):
    """Move only the new classifier; the backbone object is never moved here."""
    classifier.to(device=device, dtype=dtype)
    if {str(tensor.device) for tensor in classifier.parameters()} != {str(device)}:
        raise RuntimeError("classifier placement onto the resolved device failed")
    if {parameter.dtype for parameter in classifier.parameters()} != {dtype}:
        raise RuntimeError("classifier placement onto the resolved dtype failed")
    return classifier


def _base_ledger(device):
    return dict(
        primary_locator=PRIMARY_LOCATOR,
        protocol_locator=PROTOCOL_LOCATOR,
        objective=OBJECTIVE,
        scope=SCOPE,
        non_claims=list(NON_CLAIMS),
        task="cacon_staged_runtime_orchestration",
        complete=False,
        training_complete=False,
        full_method_parity=False,
        common_budget_matched=False,
        scientific_evaluation_complete=False,
        publication_ready=False,
        equal_compute_claim=False,
        native_cuda_verified=False,
        declared_synthetic_seam=False,
        allocation="unresolved",
        allocation_matches_prospective=False,
        synthetic_probe=False,
        contrastive_passes=0,
        final_linear_passes=0,
        passes_completed=0,
        stage1_passes_observed=0,
        stage2_passes_observed=0,
        batch_size=None,
        device=device if isinstance(device, str) else None,
        resolved_device=None,
        dimension=None,
        normalize_input=None,
        stage2_classes=0,
        stage2_classifier_device=None,
        stage2_classifier_dtype=None,
        stage2_placement_before_factory=False,
        transition_device_matches_stage1=False,
        backbone_object_preserved=False,
        source_contract_verified=False,
        source_contract_sha256=None,
        source_contract_revalidations=0,
        source_cache_membership_verified=False,
        generated_images_excluded_from_supervised_source=False,
        same_backbone_object=False,
        stale_stage1_gradients_present_before_transition=None,
        stale_gradients_cleared=False,
        encoder_frozen=False,
        projector_not_in_stage2_optimizer=False,
        projector_not_forwarded_in_stage2=False,
        transition_policy="continuous_in_memory_last_state",
        external_metrics_used=False,
        checkpoint_selection="none",
        source_records_per_pass=0,
        stage1_source_records=0,
        stage2_source_records=0,
        total_source_records=0,
        observed_samples=0,
        stage1_encoder_image_instances=0,
        stage2_encoder_image_instances=0,
        total_encoder_image_instances=0,
        stage1_optimizer_step_attempts=0,
        stage1_optimizer_steps=0,
        stage2_optimizer_step_attempts=0,
        stage2_optimizer_steps=0,
        observed_optimizer_step_attempts=0,
        observed_optimizer_steps=0,
        generator_initial_sha256=None,
        generator_current_sha256=None,
        generator_advanced=False,
        numpy_rng_initial_sha256=None,
        numpy_rng_current_sha256=None,
        numpy_rng_advanced=False,
        rng_continuity_checks=0,
        generator_cache_production_costs_included=False,
        third_view_production_unproven=True,
        stage1_ledgers=[],
        stage2_ledgers=[],
        state_snapshot_unavailable=False,
    )


def run_cacon_staged_training(
    backbone,
    projector,
    stage1_dataset,
    stage2_dataset,
    stage1_optimizer_factory,
    stage2_optimizer_factory,
    torch_generator,
    *,
    dimension,
    batch_size,
    device,
    bn_policy="frozen_all",
    tail_policy="reject",
    temperature=0.1,
    contrastive_passes=4,
    final_linear_passes=4,
    synthetic_probe=False,
    normalize_input=False,
    device_guard=None,
    device_transfer=None,
    progress=None,
):
    """Run the staged CACon adaptation and reconcile observed ledgers.

    Every source, type, label, policy, device/dtype and seam contract is validated
    before the first stage-1 optimizer step. Any stage or transition failure raises
    ``CaconTransitionFailure`` carrying the observed partial ledger, including the
    completed and partial stage ledgers, with observed sample/exposure/step counts.
    """
    ledger = _base_ledger(device)
    rng_identity = [None]
    rng_last = [None]

    def apply_observed():
        stage1 = ledger["stage1_ledgers"]
        stage2 = ledger["stage2_ledgers"]
        rows = stage1 + stage2
        ledger["stage1_passes_observed"] = len(stage1)
        ledger["stage2_passes_observed"] = len(stage2)
        for name, part in (("stage1", stage1), ("stage2", stage2)):
            ledger[f"{name}_source_records"] = sum(row.get("samples", 0) for row in part)
            ledger[f"{name}_encoder_image_instances"] = sum(
                row.get("encoder_image_instances", 0) for row in part
            )
            ledger[f"{name}_optimizer_step_attempts"] = sum(
                row.get("optimizer_step_attempts", 0) for row in part
            )
            ledger[f"{name}_optimizer_steps"] = sum(row.get("optimizer_steps", 0) for row in part)
        ledger["total_source_records"] = (
            ledger["stage1_source_records"] + ledger["stage2_source_records"]
        )
        ledger["observed_samples"] = ledger["total_source_records"]
        ledger["total_encoder_image_instances"] = (
            ledger["stage1_encoder_image_instances"] + ledger["stage2_encoder_image_instances"]
        )
        ledger["observed_optimizer_step_attempts"] = (
            ledger["stage1_optimizer_step_attempts"] + ledger["stage2_optimizer_step_attempts"]
        )
        ledger["observed_optimizer_steps"] = (
            ledger["stage1_optimizer_steps"] + ledger["stage2_optimizer_steps"]
        )
        if rows:
            ledger["native_cuda_verified"] = all(row.get("native_cuda_verified") for row in rows)
            ledger["declared_synthetic_seam"] = not ledger["native_cuda_verified"]
            ledger["resolved_device"] = next(
                (row.get("resolved_device") for row in rows if row.get("resolved_device")),
                ledger["resolved_device"],
            )

    def snapshot():
        apply_observed()
        row = copy.deepcopy(ledger)
        if isinstance(torch_generator, torch.Generator):
            row["generator_current_sha256"] = _generator_sha(torch_generator)
        if isinstance(stage1_dataset, ThreeViewDataset) and isinstance(
            stage1_dataset.rng, np.random.Generator
        ):
            row["numpy_rng_current_sha256"] = _numpy_rng_sha(stage1_dataset.rng)
        return row

    def rng_state():
        generator = (
            _generator_sha(torch_generator)
            if isinstance(torch_generator, torch.Generator)
            else None
        )
        rng = stage1_dataset.rng if isinstance(stage1_dataset, ThreeViewDataset) else None
        numpy_sha = _numpy_rng_sha(rng) if isinstance(rng, np.random.Generator) else None
        return {
            "generator": generator,
            "numpy": numpy_sha,
            "numpy_id": id(rng) if rng is not None else None,
        }

    def boundary_check(context, pre, *, require_numpy):
        """Per-pass RNG continuity: advance, no replacement, and record the state."""
        post = rng_state()
        if post["numpy_id"] != rng_identity[0]:
            raise RuntimeError(f"{context}: continuing NumPy RNG was replaced")
        if pre is None or post["generator"] == pre["generator"]:
            raise RuntimeError(f"{context}: continuing generator did not advance")
        if require_numpy and (pre["numpy"] is None or post["numpy"] == pre["numpy"]):
            raise RuntimeError(f"{context}: NumPy RNG did not advance")
        ledger["rng_continuity_checks"] += 1
        ledger["generator_current_sha256"] = post["generator"]
        ledger["numpy_rng_current_sha256"] = post["numpy"]
        rng_last[0] = post

    def callback_guard(context):
        """The progress callback must not reseed, replace or mutate either stream."""
        if rng_last[0] is None:
            return
        post = rng_state()
        if post["numpy_id"] != rng_identity[0]:
            raise RuntimeError(f"{context}: progress callback replaced the NumPy RNG")
        if post != rng_last[0]:
            raise RuntimeError(
                f"{context}: progress callback reseeded or mutated the continuing generator/RNG"
            )

    def require_unchanged(context):
        if rng_last[0] is None:
            return
        post = rng_state()
        if post["numpy_id"] != rng_identity[0]:
            raise RuntimeError(f"{context}: continuing NumPy RNG was replaced")
        if post != rng_last[0]:
            raise RuntimeError(f"{context}: continuing generator/RNG changed unexpectedly")

    try:
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone nn.Module required")
        if not isinstance(projector, ProjectionHead):
            raise TypeError("canonical ProjectionHead projector required")
        if type(stage1_dataset) is not ThreeViewDataset:
            raise TypeError(
                "authentic ThreeViewDataset cache consumer required for stage1 (exact type, "
                "no counterfeit subclass)"
            )
        if type(stage2_dataset) not in (Stage2LabelDataset, MTLFaceDataset):
            raise TypeError("exact concrete stage2 dataset required")
        if not callable(stage1_optimizer_factory) or not callable(stage2_optimizer_factory):
            raise TypeError("callable stage optimizer factories required")
        if not isinstance(torch_generator, torch.Generator) or torch_generator.device.type != "cpu":
            raise ValueError("caller-owned continuing CPU torch.Generator required")
        if not isinstance(stage1_dataset.rng, np.random.Generator):
            raise ValueError("continuing NumPy RNG required on the stage1 dataset")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 2:
            raise ValueError("stage1 requires an integer batch size >= 2")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("positive integer embedding dimension required")
        if not isinstance(normalize_input, bool):
            raise ValueError("explicit boolean normalize_input policy required")
        if isinstance(contrastive_passes, bool) or not isinstance(contrastive_passes, int):
            raise ValueError("integer contrastive pass count required")
        if isinstance(final_linear_passes, bool) or not isinstance(final_linear_passes, int):
            raise ValueError("integer final-linear pass count required")
        if contrastive_passes < 1 or final_linear_passes < 1:
            raise ValueError("positive stage pass counts required")
        if not isinstance(synthetic_probe, bool):
            raise ValueError("explicit boolean synthetic_probe flag required")
        if (
            contrastive_passes,
            final_linear_passes,
        ) != PROSPECTIVE_ALLOCATION and not synthetic_probe:
            raise ValueError(
                "non-prospective pass counts require an explicit synthetic_probe=True tag "
                "(shorter runs are not scientific evaluations)"
            )
        if bn_policy != "frozen_all":
            raise ValueError("staged runtime requires the declared frozen_all BatchNorm policy")
        if progress is not None and not callable(progress):
            raise TypeError("callable progress callback required")

        # Seam trust before any optimizer work; default to the approved native pair.
        guard = CudaTensorGuard() if device_guard is None else device_guard
        transfer = CudaTransfer() if device_transfer is None else device_transfer
        _require_seams(guard, transfer)

        # Full entry validation (classes/labels/source) BEFORE any stage1 step.
        classes = _validated_stage2_classes(stage2_dataset)
        original = _source_snapshot(stage1_dataset, stage2_dataset)
        total = _validate_source_contract(stage1_dataset, stage2_dataset)
        rng_identity[0] = id(stage1_dataset.rng)
        ledger["source_contract_verified"] = True
        ledger["source_cache_membership_verified"] = True
        ledger["generated_images_excluded_from_supervised_source"] = True
        ledger["source_contract_sha256"] = hashlib.sha256(repr(original).encode()).hexdigest()
        ledger["source_records_per_pass"] = total
        ledger["dimension"] = dimension
        ledger["normalize_input"] = normalize_input
        ledger["stage2_classes"] = classes
        ledger["synthetic_probe"] = bool(synthetic_probe)
        ledger["allocation"] = (
            "prospective_4+4"
            if (contrastive_passes, final_linear_passes) == PROSPECTIVE_ALLOCATION
            else "synthetic_probe"
        )
        ledger["allocation_matches_prospective"] = (
            contrastive_passes,
            final_linear_passes,
        ) == PROSPECTIVE_ALLOCATION
        ledger["contrastive_passes"] = contrastive_passes
        ledger["final_linear_passes"] = final_linear_passes
        ledger["batch_size"] = batch_size
        ledger["generator_initial_sha256"] = _generator_sha(torch_generator)
        ledger["generator_current_sha256"] = ledger["generator_initial_sha256"]
        ledger["numpy_rng_initial_sha256"] = _numpy_rng_sha(stage1_dataset.rng)
        ledger["numpy_rng_current_sha256"] = ledger["numpy_rng_initial_sha256"]

        rng_last[0] = rng_state()
        stage1_optimizer = stage1_optimizer_factory(backbone, projector)
        require_unchanged("stage1 optimizer factory")
        for index in range(contrastive_passes):
            _revalidate_source(original, stage1_dataset, stage2_dataset)
            ledger["source_contract_revalidations"] += 1
            require_unchanged("before pass")
            pre = rng_state()
            try:
                stage1_ledger = run_cacon_stage1_epoch(
                    backbone,
                    projector,
                    stage1_optimizer,
                    stage1_dataset,
                    torch_generator,
                    batch_size=batch_size,
                    device=device,
                    bn_policy=bn_policy,
                    tail_policy=tail_policy,
                    temperature=temperature,
                    device_guard=guard,
                    device_transfer=transfer,
                )
            except Exception as error:  # preserve the observed stage1 partial ledger
                partial = getattr(error, "ledger", None)
                if partial is not None:
                    ledger["stage1_ledgers"].append(partial)
                raise RuntimeError(f"stage1 pass {index} failed: {error}") from error
            ledger["stage1_ledgers"].append(stage1_ledger)
            ledger["passes_completed"] += 1
            boundary_check(f"stage1 pass {index}", pre, require_numpy=True)
            if progress is not None:
                progress(snapshot())
                callback_guard(f"stage1 pass {index}")

        # Transition: same backbone object; place the NEW classifier before the factory.
        _revalidate_source(original, stage1_dataset, stage2_dataset)
        ledger["source_contract_revalidations"] += 1
        require_unchanged("transition")
        ledger["stale_stage1_gradients_present_before_transition"] = any(
            parameter.grad is not None for parameter in backbone.parameters()
        )
        classifier_device, classifier_dtype = _resolve_stage2_placement(
            backbone, projector, transfer
        )
        stage1_device = next(
            (
                row.get("resolved_device")
                for row in ledger["stage1_ledgers"]
                if row.get("resolved_device")
            ),
            None,
        )
        ledger["transition_device_matches_stage1"] = (
            stage1_device is None or str(classifier_device) == stage1_device
        )
        if not ledger["transition_device_matches_stage1"]:
            raise RuntimeError("resolved stage2 device does not match the observed stage1 device")
        stage = FinalLinearStage(backbone, dimension, classes, normalize_input=normalize_input)
        _place_classifier(stage.classifier, classifier_device, classifier_dtype)
        # Independent of the placement helper: the contract is the observed placement.
        classifier_devices = {str(parameter.device) for parameter in stage.classifier.parameters()}
        classifier_dtypes = {parameter.dtype for parameter in stage.classifier.parameters()}
        if classifier_devices != {str(classifier_device)} or classifier_dtypes != {
            classifier_dtype
        }:
            raise RuntimeError(
                "stage2 classifier placement onto the resolved device/dtype failed before factory"
            )
        ledger["stage2_classifier_device"] = str(classifier_device)
        ledger["stage2_classifier_dtype"] = str(classifier_dtype)
        ledger["stage2_placement_before_factory"] = True
        ledger["same_backbone_object"] = stage.backbone is backbone
        ledger["backbone_object_preserved"] = stage.backbone is backbone
        ledger["stale_gradients_cleared"] = all(
            parameter.grad is None for parameter in backbone.parameters()
        )
        ledger["encoder_frozen"] = (
            all(not parameter.requires_grad for parameter in backbone.parameters())
            and not backbone.training
        )
        if not (
            ledger["same_backbone_object"]
            and ledger["stale_gradients_cleared"]
            and ledger["encoder_frozen"]
        ):
            raise RuntimeError("stage2 transition did not freeze the shared backbone object")
        projector_ids = {id(parameter) for parameter in projector.parameters()}
        ledger["projector_not_forwarded_in_stage2"] = not any(
            module is projector for module in stage.modules()
        )

        stage2_optimizer = stage2_optimizer_factory(stage)
        require_unchanged("stage2 optimizer factory")
        stage2_ids = [
            id(parameter)
            for group in stage2_optimizer.param_groups
            for parameter in group["params"]
        ]
        ledger["projector_not_in_stage2_optimizer"] = not (set(stage2_ids) & projector_ids)
        if not (
            ledger["projector_not_in_stage2_optimizer"]
            and ledger["projector_not_forwarded_in_stage2"]
        ):
            raise RuntimeError("projector must not be supervised or forwarded in stage2")
        owned_devices = {
            str(parameter.device)
            for group in stage2_optimizer.param_groups
            for parameter in group["params"]
        }
        if owned_devices != {str(classifier_device)}:
            raise RuntimeError("stage2 optimizer owns a parameter off the resolved device")

        for index in range(final_linear_passes):
            _revalidate_source(original, stage1_dataset, stage2_dataset)
            ledger["source_contract_revalidations"] += 1
            pre = rng_state()
            try:
                stage2_ledger = run_cacon_stage2_epoch(
                    stage,
                    stage2_optimizer,
                    stage2_dataset,
                    torch_generator,
                    batch_size=batch_size,
                    device=device,
                    bn_policy=bn_policy,
                    source_face_ids=list(original[0]),
                    device_guard=guard,
                    device_transfer=transfer,
                )
            except Exception as error:  # preserve the observed stage2 partial ledger
                partial = getattr(error, "ledger", None)
                if partial is not None:
                    ledger["stage2_ledgers"].append(partial)
                raise RuntimeError(f"stage2 pass {index} failed: {error}") from error
            ledger["stage2_ledgers"].append(stage2_ledger)
            ledger["passes_completed"] += 1
            boundary_check(f"stage2 pass {index}", pre, require_numpy=False)
            if progress is not None:
                progress(snapshot())
                callback_guard(f"stage2 pass {index}")

        # Final source + RNG revalidation after the last pass/callback.
        _revalidate_source(original, stage1_dataset, stage2_dataset)
        ledger["source_contract_revalidations"] += 1
        require_unchanged("final")
        apply_observed()

        # Observed counters must reconcile with the protocol allocation when complete.
        if ledger["stage1_source_records"] != contrastive_passes * total:
            raise RuntimeError("stage1 observed source coverage does not reconcile")
        if ledger["stage2_source_records"] != final_linear_passes * total:
            raise RuntimeError("stage2 observed source coverage does not reconcile")
        if ledger["stage1_encoder_image_instances"] != contrastive_passes * 3 * total:
            raise RuntimeError("stage1 encoder instance total does not reconcile")
        if ledger["stage2_encoder_image_instances"] != final_linear_passes * total:
            raise RuntimeError("stage2 encoder instance total does not reconcile")
        if ledger["total_source_records"] != (contrastive_passes + final_linear_passes) * total:
            raise RuntimeError("total source presentations do not reconcile")
        if any(row.get("samples") != total for row in ledger["stage1_ledgers"]):
            raise RuntimeError("stage1 per-pass observed coverage mismatch")
        if any(row.get("samples") != total for row in ledger["stage2_ledgers"]):
            raise RuntimeError("stage2 per-pass observed coverage mismatch")
        ledger["generator_advanced"] = (
            _generator_sha(torch_generator) != ledger["generator_initial_sha256"]
        )
        ledger["numpy_rng_advanced"] = (
            _numpy_rng_sha(stage1_dataset.rng) != ledger["numpy_rng_initial_sha256"]
        )
        if not (ledger["generator_advanced"] and ledger["numpy_rng_advanced"]):
            raise RuntimeError("continuing generator/RNG did not advance across the staged passes")
        ledger["complete"] = True
        return snapshot()
    except CaconTransitionFailure:
        raise
    except Exception as error:
        try:
            partial = snapshot()
        except Exception:
            partial = dict(ledger)
            partial["state_snapshot_unavailable"] = True
        raise CaconTransitionFailure(str(error), partial) from error


def _verify_bound_bytes(dataset):
    for row in dataset.records:
        for record in (row.source_record, row.generated_record):
            if file_record(PROJECT_ROOT / record["path"]) != record:
                raise ValueError("bound source/generated bytes changed; step refused")


def _revalidate_source(original, stage1_dataset, stage2_dataset):
    _validated_stage2_classes(stage2_dataset)
    if _source_snapshot(stage1_dataset, stage2_dataset) != original:
        raise ValueError(
            "source records (IDs/paths/generated paths/labels) changed during the run; step refused"
        )


    _verify_bound_bytes(stage1_dataset)
