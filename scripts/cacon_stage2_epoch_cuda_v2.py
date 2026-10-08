"""CACon stage-2 frozen-encoder final-linear epoch, v2 (label-provenance and ownership hardening).

Corrects four evidence gaps in the delivered ``proposal_stage2`` v1 runtime while
keeping the canonical ``FinalLinearStage`` API, the native CUDA guard/transfer,
and the prospective 4+4 protocol adaptation unchanged:

1. **Label provenance.** v1 only range-checked the batch labels, so an in-range
   label flip could train while the ledger logged the declared originals. v2
   compares the actual int64 batch labels against the sampler/source identity
   order for that exact batch *before* forward and step, and records the observed
   and declared labels separately.
2. **Exact dataset types and entry revalidation.** v1 used ``isinstance`` while
   declaring the dataset authentic. v2 requires the *exact* concrete supported
   types (documented adaptation for the local ``Stage2LabelDataset``), refuses
   forged subclasses, and revalidates at epoch entry from the live records:
   ordered nonempty unique face IDs, non-boolean nonnegative integer labels,
   contiguous label coverage and class count, including canonical
   ``MTLFaceDataset``. Constructor checks alone do not protect mutated records or
   a stale ``n_classes``.
3. **Per-batch optimizer ownership.** v1 checked ownership once. v2 re-runs the
   strict classifier-only ownership check before every step and compares the
   original parameter object identities, so a progress callback that mutates
   ``param_groups`` fails before the next step.
4. **Actual finite gradients.** v1 skipped a classifier parameter whose ``grad``
   was ``None``. v2 requires every trainable classifier parameter to carry a
   finite actual gradient and computes a finite aggregate norm before the step.

Scope / non-claims
------------------
Runtime component only. Not full CACon, not generator/cache lineage proof, not a
matched-budget or scientific-evaluation result, not a checkpoint selector, not
author parity. Mined identity labels are used for classifier supervision and are
recorded as mined, never human cleared. No external scoring.

Worker/test constraints
-----------------------
No GPU, no real faces/weights/model inference, no real optimizer step: tests use
tiny CPU modules plus a counting fake optimizer that never updates parameters. A
passing test is NOT native CUDA execution evidence.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn.functional import cross_entropy as _cross_entropy
from torch.utils.data import DataLoader, Dataset

from age_gap.training.sota_common import MTLFaceDataset
from scripts.cacon_stage1_epoch_cuda_v2 import (
    CudaTensorGuard,
    CudaTransfer,
    _generator_sha,
    _module_devices,
    _native_verified,
    _ObservedOptimizer,
    _PartitionSampler,
    _uniform_device_selection,
)
from scripts.cacon_supervised_v2 import FinalLinearStage

PRIMARY_LOCATOR = "https://arxiv.org/html/2312.11195v2#S2.SS1"
PROTOCOL_LOCATOR = "docs/SOTABudgetProtocol.md#prospective-stage-allocation-2026-10-08"
OBJECTIVE = (
    "section 2.1 label-based final-linear training: frozen recognition encoder plus a "
    "trainable linear classifier optimized with cross-entropy over mined identity labels"
)
SCOPE = (
    "stage-2 frozen-encoder final-linear runtime pass + common CUDA epoch adaptation; "
    "not the full CACon method, not stage-1 contrastive training, not a generator/cache "
    "lineage proof, not a matched budget and not an external evaluation"
)
REPRESENTATION_DECLARATION = (
    "training the final-linear classifier does not improve or move the frozen backbone "
    "embedding; the verification representation (backbone vs logits) is a caller "
    "declaration via FinalLinearStage.verification_features; this module never computes "
    "an external score and never selects a checkpoint"
)
BN_POLICIES = ("frozen_all",)
NON_CLAIMS = (
    "no GPU/native CUDA execution is asserted by tests",
    "no external scoring, no checkpoint selection, no publication artifact",
    "no generator training, no third-view synthesis, no projection head",
    "mined identity labels are recorded, not human cleared",
)


class CaconStage2EpochFailure(RuntimeError):
    """Epoch failure carrying the observed partial ledger."""

    def __init__(self, message, ledger):
        super().__init__(message)
        self.ledger = ledger


@dataclass(frozen=True)
class Stage2Record:
    """Explicit source-bound label record; face ID order is preserved by the dataset."""

    face_id: str
    identity: int
    path: Path


class Stage2LabelDataset(Dataset):
    """Local explicit source-bound identity dataset reusing canonical label semantics.

    Contiguous non-negative identity labels, unique nonempty face IDs and finite
    three-channel CHW preprocessing mirror the canonical ``MTLFaceDataset`` /
    ``RecognitionDataset`` contracts, but the record list is explicit so the live
    records can be revalidated at epoch entry. Constructor checks alone do not
    protect against mutated records or a stale ``n_classes``.
    """

    def __init__(self, records, preprocess):
        self.records = tuple(records)
        if not self.records:
            raise ValueError("nonempty source-bound records required")
        if not all(isinstance(record, Stage2Record) for record in self.records):
            raise ValueError("Stage2Record items required")
        face_ids = [record.face_id for record in self.records]
        if not all(isinstance(fid, str) and fid for fid in face_ids) or len(set(face_ids)) != len(
            face_ids
        ):
            raise ValueError("unique nonempty face IDs required")
        labels = set()
        for record in self.records:
            if isinstance(record.identity, bool) or not isinstance(record.identity, int):
                raise ValueError("nonnegative integer identity labels required")
            if record.identity < 0:
                raise ValueError("nonnegative integer identity labels required")
            labels.add(record.identity)
        if labels != set(range(len(labels))):
            raise ValueError("contiguous identity labels starting at zero required")
        self.n_classes = len(labels)
        if not callable(preprocess):
            raise ValueError("callable preprocessing required")
        self.preprocess = preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(record.path))
        if image is None:
            raise RuntimeError(f"cannot read {record.path}")
        tensor = torch.from_numpy(np.asarray(self.preprocess(image)))
        if (
            tensor.ndim != 3
            or tensor.shape[0] != 3
            or not tensor.is_floating_point()
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("finite three-channel CHW preprocessing required")
        return tensor, record.identity


def _bn_snapshot(module):
    snapshot = {}
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            snapshot[id(child)] = (
                None if child.running_mean is None else child.running_mean.detach().clone(),
                None if child.running_var is None else child.running_var.detach().clone(),
                None
                if child.num_batches_tracked is None
                else child.num_batches_tracked.detach().clone(),
            )
    return snapshot


def _bn_unchanged(snapshot, module):
    def same(current, expected):
        if expected is None:
            return current is None
        return current is not None and torch.equal(current, expected)

    for child in module.modules():
        if id(child) not in snapshot:
            continue
        mean, var, count = snapshot[id(child)]
        if (
            not same(child.running_mean, mean)
            or not same(child.running_var, var)
            or not same(child.num_batches_tracked, count)
        ):
            return False
    return True


def _bn_all_eval(module):
    return all(
        not child.training
        for child in module.modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
    )


def _stage2_partitions(total, batch_size, generator):
    """Full single-epoch coverage with no drop_last and no tail merge.

    A tail singleton batch is retained: the encoder BatchNorm is frozen, so a
    batch of one is a legitimate final-linear step.
    """
    order = torch.randperm(total, generator=generator).tolist()
    partitions = [order[index : index + batch_size] for index in range(0, total, batch_size)]
    flat = [index for part in partitions for index in part]
    if sorted(flat) != list(range(total)):
        raise ValueError("exact single-epoch source index coverage required")
    return partitions


# Documented exact supported concrete dataset types. ``Stage2LabelDataset`` is the
# local explicit source-bound adaptation and ``MTLFaceDataset`` is the canonical
# common-protocol dataset; both are accepted only as their exact concrete types.
SUPPORTED_DATASET_TYPES = (Stage2LabelDataset, MTLFaceDataset)


def _validated_source_list(dataset, stage):
    """Exact-type source list with entry-time revalidation of live records.

    Constructor checks do not protect against mutated ``records``/``items`` or a
    stale ``n_classes``, so the ordered face IDs and identity labels are rebuilt
    from the live dataset and checked again here.
    """
    if type(dataset) is Stage2LabelDataset:
        records = list(dataset.records)
        source = [
            (record.face_id, record.identity) if isinstance(record, Stage2Record) else None
            for record in records
        ]
        declared_classes = dataset.n_classes
    elif type(dataset) is MTLFaceDataset:
        items = list(dataset.items)
        source = [(item.path.stem, item.identity) for item in items]
        declared_classes = dataset.n_classes
    else:
        raise TypeError(
            "exact concrete Stage2LabelDataset or canonical MTLFaceDataset required; "
            "forged subclasses are refused"
        )
    if not source:
        raise ValueError("nonempty source list required")
    if any(pair is None for pair in source):
        raise ValueError("every record must be a Stage2Record")
    face_ids = [face_id for face_id, _ in source]
    if not all(isinstance(face_id, str) and face_id for face_id in face_ids):
        raise ValueError("nonempty string face IDs required")
    if len(set(face_ids)) != len(face_ids):
        raise ValueError("ordered source face IDs must be unique")
    labels = [identity for _, identity in source]
    if any(isinstance(identity, bool) or not isinstance(identity, int) for identity in labels):
        raise ValueError("non-boolean integer identity labels required")
    if any(identity < 0 for identity in labels):
        raise ValueError("nonnegative identity labels required")
    classes = len(set(labels))
    if set(labels) != set(range(classes)):
        raise ValueError("contiguous identity labels starting at zero required")
    if isinstance(declared_classes, bool) or not isinstance(declared_classes, int):
        raise ValueError("non-boolean integer dataset n_classes required")
    if classes != declared_classes:
        raise ValueError("stale dataset n_classes does not match the live records")
    if classes != stage.classes:
        raise ValueError("dataset class count must match the classifier output count")
    return source, classes


def _classifier_parameters(classifier):
    parameters = [p for p in classifier.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("at least one trainable classifier parameter required")
    return parameters


def _check_classifier_ownership(optimizer, parameters, *, expected_objects, context):
    """Strict classifier-only ownership; re-run before every step.

    Compares parameter object identity against the originally captured objects so
    a mutated ``param_groups`` (e.g. from a progress callback) is refused.
    """
    if not hasattr(optimizer, "param_groups"):
        raise TypeError("optimizer with param_groups required")
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    ids = [id(p) for p in actual]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{context}: optimizer must not own a duplicated parameter")
    if set(ids) != {id(p) for p in parameters} or len(ids) != len(parameters):
        raise ValueError(
            f"{context}: optimizer must own every trainable classifier parameter exactly "
            "once (no frozen encoder parameters)"
        )
    if len(expected_objects) != len(parameters) or any(
        p is not q for p, q in zip(parameters, expected_objects, strict=True)
    ):
        raise ValueError(
            f"{context}: optimizer parameter object identity changed (mutated param_groups)"
        )
    return parameters


def _base_ledger(device):
    return dict(
        primary_locator=PRIMARY_LOCATOR,
        protocol_locator=PROTOCOL_LOCATOR,
        objective=OBJECTIVE,
        scope=SCOPE,
        representation_declaration=REPRESENTATION_DECLARATION,
        non_claims=list(NON_CLAIMS),
        epoch_complete=False,
        training_complete=False,
        full_method_parity=False,
        common_budget_matched=False,
        scientific_evaluation_complete=False,
        publication_ready=False,
        native_cuda_verified=False,
        authentic_source_certified=False,
        declared_synthetic_seam=False,
        device_guard="unresolved",
        device_transfer="unresolved",
        declared_transfer_native=None,
        declared_device=device if isinstance(device, str) else None,
        resolved_device=None,
        device_consistent=False,
        dataset_type=None,
        source_entry_revalidated=False,
        bn_policy=None,
        bn_modules=0,
        bn_running_stats_unchanged=None,
        backbone_frozen=False,
        backbone_eval=False,
        encoder_gradients_absent=None,
        encoder_image_instances=0,
        generator_image_instances=0,
        fas_encoder_image_instances=0,
        classifier_trainable_parameter_count=0,
        optimizer_parameter_count=0,
        optimizer_ownership_checks=0,
        label_provenance_verified=True,
        label_provenance_batches=0,
        batch_size=None,
        source_face_ids_declared=False,
        mined_identity_labels_used=True,
        mined_identity_labels_human_cleared=False,
        trained_representation="final_linear_classifier",
        frozen_representation="backbone",
        verification_representation_required=True,
        never_score_external=True,
        never_select_checkpoint=True,
        expected_samples=0,
        expected_batches=0,
        source_unique_rows=0,
        samples=0,
        completed_batches=0,
        optimizer_step_attempts=0,
        optimizer_steps=0,
        batch_partitions=[],
        ordered_source_indices=[],
        input_face_ids=[],
        ordered_face_ids=[],
        batch_summaries=[],
        single_epoch_coverage=False,
        inputs_finite=True,
        logits_finite=True,
        losses_finite=True,
        gradients_finite=True,
        gradient_missing_parameters=[],
        sample_weighted_loss=None,
        sample_weighted_gradient_norm=None,
        last_batch_loss=None,
        last_batch_gradient_norm=None,
        last_batch_size=0,
        last_batch_indices=[],
        last_batch_observed_labels=[],
        last_batch_declared_labels=[],
        torch_generator_initial_sha256=None,
        torch_generator_current_sha256=None,
    )


def _require_seams(guard, transfer):
    """Trusted native seam or an explicit declared synthetic seam; never ambiguous."""
    for name, seam, required in (
        ("guard", guard, ("require_module", "require_tensor")),
        ("transfer", transfer, ("resolve_device", "to_device")),
    ):
        if not all(hasattr(seam, method) for method in required):
            raise TypeError(f"device {name} must expose {'/'.join(required)}")
    if isinstance(guard, CudaTensorGuard) and type(guard) is not CudaTensorGuard:
        raise ValueError(
            "CudaTensorGuard subclasses overriding native checks refused; "
            "use the exact trusted guard or an explicitly synthetic guard"
        )
    native = type(guard) is CudaTensorGuard
    if native:
        if type(transfer) is not CudaTransfer:
            raise ValueError(
                "native CudaTensorGuard requires the approved CudaTransfer seam; "
                "injected transfer refused"
            )
    else:
        if not getattr(guard, "synthetic", False):
            raise ValueError("non-native guard must explicitly declare synthetic=True")
        if type(transfer) is not CudaTransfer and not getattr(transfer, "synthetic", False):
            raise ValueError("non-native transfer must explicitly declare synthetic=True")
    return native


def run_cacon_stage2_epoch(
    stage,
    optimizer,
    dataset,
    torch_generator,
    *,
    batch_size,
    device,
    bn_policy,
    source_face_ids=None,
    device_guard=None,
    device_transfer=None,
    progress=None,
):
    """One full source pass of the CACon stage-2 frozen-encoder final-linear epoch.

    Actual batch labels are verified against the sampler/source identity order
    before every forward and step; dataset records and class count are
    revalidated at entry; classifier-only optimizer ownership is re-checked
    before every step; and every trainable classifier parameter must carry a
    finite actual gradient before the step.
    """
    ledger = _base_ledger(device)
    phase = "idle"
    handles = []
    seen = set()
    parameters = []
    expected_objects = []
    loss_total = grad_total = 0.0
    weight_total = 0

    def snapshot():
        row = copy.deepcopy(ledger)
        row["torch_generator_current_sha256"] = _generator_sha(torch_generator)
        return row

    def entry_hook(module, args, output):
        if phase != "forward" or not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("unaccountable encoder entry forward")
        ledger["encoder_image_instances"] += len(args[0])

    try:
        if type(stage) is not FinalLinearStage:
            raise TypeError(
                "authentic FinalLinearStage required (exact type, no counterfeit subclass)"
            )
        if type(dataset) not in SUPPORTED_DATASET_TYPES:
            raise TypeError(
                "exact concrete Stage2LabelDataset or canonical MTLFaceDataset required; "
                "forged subclasses and foreign datasets are refused"
            )
        ledger["dataset_type"] = type(dataset).__name__
        if not isinstance(torch_generator, torch.Generator) or torch_generator.device.type != "cpu":
            raise ValueError("caller-owned continuing CPU torch.Generator required")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("positive integer batch size required")
        if device != "cuda":
            raise ValueError('declared "cuda" device required; no CPU fallback')
        if bn_policy not in BN_POLICIES:
            raise ValueError("stage-2 requires explicit frozen_all BatchNorm policy")
        if progress is not None and not callable(progress):
            raise TypeError("callable progress callback required")

        guard = CudaTensorGuard() if device_guard is None else device_guard
        transfer = CudaTransfer() if device_transfer is None else device_transfer
        native = _require_seams(guard, transfer)
        ledger["device_guard"] = getattr(guard, "name", "unnamed-injected")
        ledger["device_transfer"] = getattr(transfer, "name", "unnamed-injected")
        ledger["declared_transfer_native"] = bool(getattr(transfer, "native", False))
        ledger["declared_synthetic_seam"] = not native

        stage.train()  # canonical freeze policy; re-applied for every entry
        backbone, classifier = stage.backbone, stage.classifier
        if not isinstance(backbone, nn.Module) or not isinstance(classifier, nn.Module):
            raise TypeError("stage backbone/classifier modules required")
        if any(parameter.requires_grad for parameter in backbone.parameters()):
            raise ValueError("frozen encoder required; backbone parameters must not train")
        if not all(parameter.requires_grad for parameter in classifier.parameters()):
            raise ValueError("all classifier parameters must be trainable")
        if backbone.training:
            raise ValueError("frozen encoder must stay in eval mode")
        if not _bn_all_eval(backbone):
            raise ValueError("frozen encoder BatchNorm must stay in eval mode")
        ledger["backbone_frozen"] = True
        ledger["backbone_eval"] = True

        backbone_devices = _module_devices(backbone)
        classifier_devices = _module_devices(classifier)
        _uniform_device_selection(backbone_devices, classifier_devices)
        resolved_device = transfer.resolve_device(backbone, classifier)
        expected_device = str(resolved_device)
        ledger["resolved_device"] = expected_device
        if backbone_devices != {expected_device} or classifier_devices != {expected_device}:
            raise ValueError("stage modules do not match the resolved device")
        guard.require_module(backbone, "backbone")
        guard.require_module(classifier, "classifier")
        # Exact types are a trust precondition, not proof of a CUDA device; only the
        # exact native pair may certify authentic/native source execution.
        ledger["native_cuda_verified"] = _native_verified(guard, transfer)
        # Device validation alone does not certify a source or completed pass.
        ledger["authentic_source_certified"] = False
        ledger["device_consistent"] = True

        if stage.classes < 1:
            raise ValueError("positive classifier class count required")
        source_list, classes = _validated_source_list(dataset, stage)
        ledger["source_entry_revalidated"] = True
        if len(source_list) != len(dataset):
            raise ValueError("live source list length must match the dataset length")
        source_paths = tuple(
            str(item.path) for item in (
                dataset.records if type(dataset) is Stage2LabelDataset else dataset.items
            )
        )
        face_ids = [face_id for face_id, _ in source_list]
        ledger["input_face_ids"] = list(face_ids)
        if source_face_ids is not None:
            declared = list(source_face_ids)
            if declared != face_ids:
                raise ValueError(
                    "declared source face-ID list must match the dataset order exactly"
                )
            ledger["source_face_ids_declared"] = True

        parameters = _classifier_parameters(classifier)
        expected_objects = list(parameters)
        _check_classifier_ownership(
            optimizer, parameters, expected_objects=expected_objects, context="epoch entry"
        )
        ledger["classifier_trainable_parameter_count"] = len(parameters)
        ledger["optimizer_parameter_count"] = sum(
            len(group["params"]) for group in optimizer.param_groups
        )
        ledger["optimizer_ownership_checks"] = 1
        initial_generator = _generator_sha(torch_generator)
        partitions = _stage2_partitions(len(source_list), batch_size, torch_generator)
        ledger.update(
            bn_policy=bn_policy,
            batch_size=batch_size,
            expected_samples=len(source_list),
            expected_batches=len(partitions),
            batch_partitions=[list(part) for part in partitions],
            torch_generator_initial_sha256=initial_generator,
            torch_generator_current_sha256=initial_generator,
        )
        bn_snapshot = _bn_snapshot(backbone)
        ledger["bn_modules"] = len(bn_snapshot)

        handles.append(backbone.register_forward_hook(entry_hook))
        observed = _ObservedOptimizer(optimizer, ledger)
        sampler = _PartitionSampler(partitions)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
        if loader.num_workers != 0:
            raise RuntimeError(
                "num_workers=0 required; continuing dataset RNG must not be duplicated"
            )
        for batch in loader:
            current_source, _ = _validated_source_list(dataset, stage)
            current_paths = tuple(
                str(item.path) for item in (
                    dataset.records if type(dataset) is Stage2LabelDataset else dataset.items
                )
            )
            if current_source != source_list or current_paths != source_paths:
                raise ValueError("source records changed during the epoch; step refused")
            if len(batch) not in (2, 3):
                raise RuntimeError("(image, identity[, age]) dataset item required")
            images, labels = batch[0], batch[1]
            size = len(images)
            batch_indices = list(sampler.used[-1])
            if len(batch_indices) != size:
                raise RuntimeError("sampler/dataset batch size mismatch")
            if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[2:]) < 1:
                raise ValueError("actual BCHW three-channel input tensors required")
            if not images.is_floating_point():
                raise ValueError("floating-point image batch required")
            if (
                not isinstance(labels, torch.Tensor)
                or labels.dtype != torch.int64
                or labels.ndim != 1
                or len(labels) != size
            ):
                raise ValueError("aligned int64 identity labels required")
            declared_labels = [source_list[index][1] for index in batch_indices]
            expected_labels = torch.tensor(declared_labels, dtype=torch.int64)
            # Label provenance: the actual batch labels must equal the source identity
            # order for exactly this batch, before any forward or step.
            if not torch.equal(labels.detach().cpu(), expected_labels):
                ledger["label_provenance_verified"] = False
                raise ValueError(
                    "actual batch labels do not match the sampler/source identity order; "
                    "label provenance refused"
                )
            # Explicit transfer to the resolved actual CUDA device, then guard.
            moved = transfer.to_device(images, resolved_device)
            moved_labels = transfer.to_device(labels, resolved_device)
            if str(moved.device) != expected_device or str(moved_labels.device) != expected_device:
                raise ValueError("transferred batch device does not match the resolved device")
            guard.require_tensor(moved, "input-images", device=resolved_device)
            if not torch.isfinite(moved).all():
                ledger["inputs_finite"] = False
                raise FloatingPointError("nonfinite input batch; step refused")
            if (moved_labels < 0).any() or (moved_labels >= stage.classes).any():
                raise ValueError("identity labels outside the declared class range")
            if not torch.equal(moved_labels.detach().cpu(), expected_labels):
                ledger["label_provenance_verified"] = False
                raise ValueError("transferred labels changed provenance; refused")
            ledger["label_provenance_batches"] += 1

            phase = "forward"
            logits = stage(moved)
            phase = "idle"
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 2
                or logits.shape != (size, stage.classes)
                or not logits.is_floating_point()
            ):
                raise ValueError("declared (B, classes) logit tensor required")
            if str(logits.device) != expected_device:
                raise ValueError("logit device does not match the resolved device")
            guard.require_tensor(logits, "logits", device=resolved_device)
            if not torch.isfinite(logits).all():
                ledger["logits_finite"] = False
                raise FloatingPointError("nonfinite classifier logits; step refused")
            if not _bn_all_eval(backbone):
                raise ValueError("frozen encoder BatchNorm left eval mode during the epoch")

            # Re-run strict classifier-only ownership before this step.
            _check_classifier_ownership(
                optimizer,
                _classifier_parameters(stage.classifier),
                expected_objects=expected_objects,
                context=f"batch {ledger['completed_batches']}",
            )
            ledger["optimizer_ownership_checks"] += 1

            observed.zero_grad(set_to_none=True)
            loss = _cross_entropy(logits, moved_labels)
            if not torch.isfinite(loss):
                ledger["losses_finite"] = False
                raise FloatingPointError("nonfinite cross-entropy loss; step refused")
            loss.backward()
            grad_norm = 0.0
            for parameter in parameters:
                if parameter.grad is None:
                    ledger["gradients_finite"] = False
                    ledger["gradient_missing_parameters"].append(
                        tuple(parameter.shape) if hasattr(parameter, "shape") else None
                    )
                    raise FloatingPointError(
                        "classifier parameter carries no actual gradient; step refused"
                    )
                if str(parameter.grad.device) != expected_device:
                    raise ValueError(
                        "classifier gradient device does not match the resolved device"
                    )
                guard.require_tensor(parameter.grad, "classifier-gradient", device=resolved_device)
                if not torch.isfinite(parameter.grad).all():
                    ledger["gradients_finite"] = False
                    raise FloatingPointError("nonfinite classifier gradient; step refused")
                grad_norm += float(parameter.grad.detach().double().square().sum().item())
            grad_norm = math.sqrt(grad_norm)
            if not math.isfinite(grad_norm):
                ledger["gradients_finite"] = False
                raise FloatingPointError("nonfinite aggregate classifier gradient; step refused")
            observed.step()

            seen.update(batch_indices)
            ledger["samples"] += size
            ledger["source_unique_rows"] = len(seen)
            ledger["completed_batches"] += 1
            ledger["last_batch_size"] = size
            ledger["last_batch_indices"] = batch_indices
            ledger["last_batch_loss"] = float(loss.detach().item())
            ledger["last_batch_gradient_norm"] = float(grad_norm)
            ledger["last_batch_observed_labels"] = [int(v) for v in moved_labels.detach().cpu()]
            ledger["last_batch_declared_labels"] = list(declared_labels)
            loss_total += float(loss.detach().item()) * size
            grad_total += grad_norm * size
            weight_total += size
            ledger["sample_weighted_loss"] = loss_total / weight_total
            ledger["sample_weighted_gradient_norm"] = grad_total / weight_total
            ledger["batch_summaries"].append(
                dict(
                    indices=batch_indices,
                    face_ids=[face_ids[index] for index in batch_indices],
                    identities=declared_labels,
                    observed_labels=[int(v) for v in moved_labels.detach().cpu()],
                    size=size,
                    loss=float(loss.detach().item()),
                    gradient_norm=float(grad_norm),
                )
            )
            if progress is not None:
                progress(snapshot())

        ledger["ordered_source_indices"] = [index for part in sampler.used for index in part]
        ledger["ordered_face_ids"] = [face_ids[index] for index in ledger["ordered_source_indices"]]
        ledger["encoder_gradients_absent"] = all(
            parameter.grad is None for parameter in backbone.parameters()
        )
        total = len(source_list)
        if (
            ledger["samples"] != total
            or ledger["source_unique_rows"] != total
            or ledger["completed_batches"] != len(partitions)
            or ledger["optimizer_steps"] != len(partitions)
            or ledger["optimizer_step_attempts"] != len(partitions)
            or ledger["encoder_image_instances"] != total
            or ledger["label_provenance_batches"] != len(partitions)
            or ledger["label_provenance_verified"] is not True
            or sorted(ledger["ordered_source_indices"]) != list(range(total))
            or [list(part) for part in sampler.used] != [list(part) for part in partitions]
            or ledger["encoder_gradients_absent"] is not True
        ):
            raise RuntimeError("observed single-epoch coverage/exposure/step mismatch")
        ledger["single_epoch_coverage"] = True
        ledger["bn_running_stats_unchanged"] = _bn_unchanged(bn_snapshot, backbone)
        if not ledger["bn_running_stats_unchanged"]:
            raise RuntimeError("frozen_all BatchNorm policy did not preserve running statistics")
        ledger["torch_generator_current_sha256"] = _generator_sha(torch_generator)
        ledger["epoch_complete"] = True
        ledger["authentic_source_certified"] = (
            ledger["native_cuda_verified"] and ledger["source_entry_revalidated"]
            and ledger["label_provenance_verified"] and ledger["single_epoch_coverage"]
        )
        return snapshot()
    except Exception as error:
        ledger["epoch_complete"] = False
        ledger["single_epoch_coverage"] = False
        try:
            partial = snapshot()
        except Exception:
            partial = dict(ledger)
            partial["state_snapshot_unavailable"] = True
        raise CaconStage2EpochFailure(str(error), partial) from error
    finally:
        for handle in handles:
            handle.remove()
