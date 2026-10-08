"""CACon stage-1 three-view instance NT-Xent epoch (CUDA adaptation, observed accounting).

Primary locator
---------------
CACon paper, section 2.3 "Modified Contrastive Loss"
(``https://arxiv.org/html/2312.11195v2#S2.SS3``), Eq.(4) cosine similarity,
Eq.(5) two-view NT-Xent and Eq.(6) the three-view modification: the anchor
maximizes similarity to both own-image positives (z_j, z_k) while other images
in the batch, including other source images, are negatives.

v2 correction over v1
---------------------
The authentic ``ThreeViewDataset`` necessarily yields CPU tensors
(``torch.from_numpy``), so v1's direct ``guard.require_tensor(input-view)``
made a native CUDA epoch impossible. v2 adds the mandatory
``CudaTransfer`` seam: the three views are explicitly moved to the **resolved
actual CUDA device** of the already-transferred backbone/projector, and that
device identity (not merely "some CUDA") is enforced for modules, inputs and
gradients. A mixed ``cuda:0``/``cuda:1`` binding is refused.

Trust anchor
------------
``native_cuda_verified`` is derived from *exact trusted types*
(``type(guard) is CudaTensorGuard`` and ``type(transfer) is CudaTransfer``).
A subclass that overrides the checks is therefore marked synthetic, never
native. A native guard with an unapproved (non-``CudaTransfer``) seam is
refused before any step. Tests inject deliberately synthetic seams, so a
passing test never implies native CUDA execution.

Scope
-----
Stage-1 instance objective plus the common CUDA epoch adaptation only. Not the
full CACon method, no generator training, no stage-2 linear training, no
external scoring, no checkpoint selection, no artifact IO, no budget proof.
Mined person identity labels are never used.

Worker/test constraints
-----------------------
No GPU, no real faces/weights/model inference, no real optimizer step: tests
use tiny CPU modules plus a counting fake optimizer that never updates
parameters. Passing is NOT native CUDA evidence.
"""

from __future__ import annotations

import copy
import hashlib
import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from age_gap.training.sota_common import ProjectionHead, cacon_nt_xent
from scripts.cacon_dataset_v2 import ThreeViewDataset

PRIMARY_LOCATOR = "https://arxiv.org/html/2312.11195v2#S2.SS3"
OBJECTIVE = (
    "section 2.3 ('Modified Contrastive Loss') Eq.6 three-view NT-Xent: anchor z_i maximizes "
    "similarity to its two own-source positives z_j/z_k over a 3B denominator; other-source "
    "images are negatives; mined person labels unused"
)
SCOPE = (
    "stage-1 instance objective + common CUDA epoch adaptation; not the full CACon "
    "method, generator training, stage-2 linear training, budget matching or publication evidence"
)
BN_POLICIES = ("frozen_all", "train_all")
TAIL_POLICIES = ("reject", "merge_tail")


class CaconStage1EpochFailure(RuntimeError):
    """A refused or failed epoch preserves its observed partial ledger."""

    def __init__(self, message, ledger):
        super().__init__(message)
        self.ledger = copy.deepcopy(ledger)


def _module_devices(module):
    return {str(tensor.device) for tensor in [*module.parameters(), *module.buffers()]}


def _uniform_device_selection(backbone_devices, projector_devices):
    """Require one shared device identity; a mixed cuda:0/cuda:1 binding is refused."""
    if len(backbone_devices) != 1 or len(projector_devices) != 1:
        raise ValueError("backbone/projector must each own a single device identity")
    if backbone_devices != projector_devices:
        raise ValueError("backbone/projector mixed-device binding refused")
    return next(iter(backbone_devices))


class CudaTensorGuard:
    """Native fail-closed CUDA guard enforcing one shared device identity.

    It refuses any non-CUDA tensor, any module spanning more than one device
    and any tensor whose device differs from the resolved device.
    """

    name = "native-cuda"

    def require_module(self, module, role):
        if not isinstance(module, nn.Module):
            raise TypeError(f"{role} must be an nn.Module")
        tensors = [*module.parameters(), *module.buffers()]
        if not tensors:
            raise ValueError(f"{role} must own at least one parameter or buffer")
        devices = {str(tensor.device) for tensor in tensors}
        if len(devices) != 1:
            raise ValueError(f"{role} spans multiple devices; mixed-device module refused")
        for tensor in tensors:
            self.require_tensor(tensor, role)

    def require_tensor(self, tensor, role, device=None):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{role} must be a torch.Tensor")
        if tensor.device.type != "cuda" or not tensor.is_cuda:
            raise ValueError(f"{role} must reside on CUDA; CPU/other-device fallback refused")
        if device is not None and str(tensor.device) != str(device):
            raise ValueError(
                f"{role} device {tensor.device} does not match resolved device {device}; "
                "mixed-device binding refused"
            )


class CudaTransfer:
    """Approved native CUDA transfer seam (exact type is the trust anchor).

    ``resolve_device`` binds the single actual CUDA device shared by the
    backbone and projector; ``to_device`` performs the real ``.to`` move of the
    authentic CPU three-view batch. Only an exact ``CudaTransfer`` instance can
    contribute to a native verification; subclasses are treated as synthetic.
    """

    name = "native-cuda-transfer"
    native = True

    def resolve_device(self, backbone, projector):
        device = _uniform_device_selection(_module_devices(backbone), _module_devices(projector))
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("CUDA-bound backbone/projector required; no CPU fallback")
        return device

    def to_device(self, tensor, device):
        return tensor.to(device)


def _native_verified(guard, transfer):
    return type(guard) is CudaTensorGuard and type(transfer) is CudaTransfer


class _ObservedOptimizer:
    """Count step attempts and returned steps around the caller's optimizer.

    No optimizer method or hook is replaced; a thrown step may have partially
    mutated state and counts as an attempt but not a returned step.
    """

    def __init__(self, optimizer, ledger):
        self.optimizer, self.ledger = optimizer, ledger

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def step(self, *args, **kwargs):
        self.ledger["optimizer_step_attempts"] += 1
        result = self.optimizer.step(*args, **kwargs)
        self.ledger["optimizer_steps"] += 1
        return result


class _PartitionSampler:
    """Emit predeclared index batches and record the actual ordered batches used."""

    def __init__(self, partitions):
        self.partitions = [list(part) for part in partitions]
        self.used = []

    def __iter__(self):
        for part in self.partitions:
            self.used.append(list(part))
            yield list(part)

    def __len__(self):
        return len(self.partitions)


def _base_ledger(device):
    return dict(
        primary_locator=PRIMARY_LOCATOR,
        objective=OBJECTIVE,
        scope=SCOPE,
        epoch_complete=False,
        training_complete=False,
        full_method_parity=False,
        common_budget_matched=False,
        scientific_evaluation_complete=False,
        publication_ready=False,
        native_cuda_verified=False,
        device_guard="unresolved",
        device_transfer="unresolved",
        declared_device=device if isinstance(device, str) else None,
        resolved_device=None,
        device_consistent=False,
        declared_transfer_native=None,
        bn_policy=None,
        bn_modules=0,
        bn_running_stats_unchanged=None,
        tail_policy=None,
        tail_merged=False,
        tail_singleton_indices=[],
        batch_size=None,
        temperature=None,
        mined_person_labels_used=False,
        recorded_person_collisions_treated_as_negatives=True,
        primary_equations=(
            "Eq.4 cosine similarity, Eq.5 two-view NT-Xent, Eq.6 three-view modified NT-Xent"
        ),
        positive_definition="own-source two augmented views positive (3N rows, three per sample)",
        negative_definition="other-source images negative",
        expected_samples=0,
        expected_batches=0,
        expected_encoder_image_instances=0,
        source_unique_rows=0,
        samples=0,
        encoder_image_instances=0,
        completed_batches=0,
        optimizer_step_attempts=0,
        optimizer_steps=0,
        optimizer_parameter_count=0,
        batch_partitions=[],
        ordered_source_indices=[],
        face_ids_ordered=[],
        single_epoch_coverage=False,
        inputs_finite=True,
        features_finite=True,
        losses_finite=True,
        gradients_finite=True,
        sample_weighted_loss=None,
        sample_weighted_gradient_norm=None,
        last_batch_loss=None,
        last_batch_gradient_norm=None,
        last_batch_size=0,
        last_batch_indices=[],
        batch_summaries=[],
        torch_generator_initial_sha256=None,
        torch_generator_current_sha256=None,
    )


def _trainable_parameters(modules):
    return [p for module in modules for p in module.parameters() if p.requires_grad]


def _require_optimizer_ownership(optimizer, modules):
    trainable = _trainable_parameters(modules)
    expected = [id(p) for p in trainable]
    if not expected:
        raise ValueError("at least one trainable backbone/projector parameter required")
    if not hasattr(optimizer, "param_groups"):
        raise TypeError("optimizer with param_groups required")
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len(set(actual)):
        raise ValueError("optimizer must not own a duplicated parameter")
    if set(actual) != set(expected) or len(actual) != len(expected):
        raise ValueError("optimizer must own each trainable parameter exactly once")
    return trainable


def _source_partitions(total, batch_size, tail_policy, generator):
    # A declared tail contract is validated before the continuing generator is drawn.
    if total == 1 or total % batch_size == 1:
        if tail_policy == "reject":
            raise ValueError(
                "tail singleton has no other-source negatives; declare merge_tail or change batch size"
            )
        if total == 1:
            raise ValueError("cannot merge_tail: the whole epoch would be a single singleton batch")
    order = torch.randperm(total, generator=generator).tolist()
    partitions = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    tail_merged = False
    tail_singleton_indices = []
    if partitions and len(partitions[-1]) == 1:
        tail_singleton_indices = list(partitions[-1])
        partitions[-2].extend(partitions.pop())
        tail_merged = True
    if any(len(part) < 2 for part in partitions):
        raise ValueError(
            "every stage-1 batch must hold at least two samples for other-source negatives"
        )
    flat = [index for part in partitions for index in part]
    if sorted(flat) != list(range(total)):
        raise ValueError("exact single-epoch source index coverage required")
    return partitions, tail_merged, tail_singleton_indices


def _apply_bn_policy(modules, policy):
    snapshot, bn_modules = {}, 0
    for module in modules:
        module.train()
        for m in module.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                bn_modules += 1
                if policy == "frozen_all":
                    m.eval()
                else:
                    m.train()
                snapshot[id(m)] = (
                    None if m.running_mean is None else m.running_mean.detach().clone(),
                    None if m.running_var is None else m.running_var.detach().clone(),
                    None
                    if m.num_batches_tracked is None
                    else m.num_batches_tracked.detach().clone(),
                )
    return snapshot, bn_modules


def _bn_running_stats_unchanged(snapshot, modules):
    def same(current, expected):
        if expected is None:
            return current is None
        return current is not None and torch.equal(current, expected)

    for module in modules:
        for m in module.modules():
            if id(m) not in snapshot:
                continue
            mean, var, count = snapshot[id(m)]
            if (
                not same(m.running_mean, mean)
                or not same(m.running_var, var)
                or not same(m.num_batches_tracked, count)
            ):
                return False
    return True


def _generator_sha(generator):
    return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()


def run_cacon_stage1_epoch(
    backbone,
    projector,
    optimizer,
    dataset,
    torch_generator,
    *,
    batch_size,
    device,
    bn_policy,
    tail_policy,
    temperature=0.1,
    encoder_probe=None,
    device_guard=None,
    device_transfer=None,
    progress=None,
):
    """One full source pass of the CACon stage-1 instance objective on CUDA.

    The three authentic CPU views are explicitly transferred to the resolved
    actual CUDA device before any guard check. Every contract (types, policies,
    index/tail, optimizer ownership, device identity) is validated before the
    first optimizer step. Failures attach the observed partial ledger.
    """
    ledger = _base_ledger(device)
    phase = "idle"
    handles = []
    seen = set()
    parameters = []
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
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone nn.Module required")
        if not isinstance(projector, ProjectionHead):
            raise TypeError("canonical ProjectionHead projector required")
        if not isinstance(dataset, ThreeViewDataset):
            raise TypeError("authentic ThreeViewDataset cache consumer required")
        if len(dataset) < 1:
            raise ValueError("nonempty dataset required")
        face_ids = [row.face_id for row in dataset.records]
        if not all(isinstance(face_id, str) and face_id for face_id in face_ids) or len(
            set(face_ids)
        ) != len(face_ids):
            raise ValueError("unique nonempty declared face IDs required")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 2:
            raise ValueError("stage-1 NT-Xent requires an integer batch size >= 2")
        if device != "cuda":
            raise ValueError('declared "cuda" device required; no CPU fallback')
        if bn_policy not in BN_POLICIES:
            raise ValueError("explicit frozen_all/train_all BN policy required")
        if tail_policy not in TAIL_POLICIES:
            raise ValueError("explicit reject/merge_tail tail policy required")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("finite positive NT-Xent temperature required")
        if progress is not None and not callable(progress):
            raise TypeError("callable progress callback required")
        if not isinstance(torch_generator, torch.Generator) or torch_generator.device.type != "cpu":
            raise ValueError("caller-owned continuing CPU torch.Generator required")
        guard = CudaTensorGuard() if device_guard is None else device_guard
        if not (hasattr(guard, "require_module") and hasattr(guard, "require_tensor")):
            raise TypeError("device guard must expose require_module/require_tensor")
        transfer = CudaTransfer() if device_transfer is None else device_transfer
        if not (hasattr(transfer, "resolve_device") and hasattr(transfer, "to_device")):
            raise TypeError("device transfer must expose resolve_device/to_device")
        # Trust precheck: a native guard requires the approved transfer seam, and a
        # subclass that overrides the native checks is refused outright (never native).
        if isinstance(guard, CudaTensorGuard) and type(guard) is not CudaTensorGuard:
            raise ValueError(
                "CudaTensorGuard subclasses overriding native checks refused; "
                "use the exact trusted guard or an explicitly synthetic guard"
            )
        if type(guard) is CudaTensorGuard and type(transfer) is not CudaTransfer:
            raise ValueError(
                "native CudaTensorGuard requires the approved CudaTransfer seam; "
                "injected transfer refused"
            )
        ledger["device_guard"] = getattr(guard, "name", "unnamed-injected")
        ledger["device_transfer"] = getattr(transfer, "name", "unnamed-injected")
        ledger["declared_transfer_native"] = bool(getattr(transfer, "native", False))
        if encoder_probe is None:
            encoder_probe = backbone
        if not isinstance(encoder_probe, nn.Module) or not (
            encoder_probe is backbone or any(encoder_probe is child for child in backbone.modules())
        ):
            raise ValueError("encoder entry probe must belong to the backbone")

        backbone_devices = _module_devices(backbone)
        projector_devices = _module_devices(projector)
        _uniform_device_selection(backbone_devices, projector_devices)
        resolved_device = transfer.resolve_device(backbone, projector)
        expected_device = str(resolved_device)
        ledger["resolved_device"] = expected_device
        if backbone_devices != {expected_device}:
            raise ValueError("backbone/projector device does not match the resolved device")
        guard.require_module(backbone, "backbone")
        guard.require_module(projector, "projector")
        # Exact types are only a trust precondition, not proof of a CUDA device.
        ledger["native_cuda_verified"] = _native_verified(guard, transfer)
        ledger["device_consistent"] = True

        parameters = _require_optimizer_ownership(optimizer, (backbone, projector))
        initial_generator = _generator_sha(torch_generator)
        partitions, tail_merged, tail_singleton_indices = _source_partitions(
            len(dataset), batch_size, tail_policy, torch_generator
        )
        if any(len(part) == 1 for part in partitions):
            raise ValueError("singleton batch cannot produce other-source negatives")

        ledger.update(
            bn_policy=bn_policy,
            tail_policy=tail_policy,
            tail_merged=tail_merged,
            tail_singleton_indices=tail_singleton_indices,
            batch_size=batch_size,
            temperature=float(temperature),
            expected_samples=len(dataset),
            expected_batches=len(partitions),
            expected_encoder_image_instances=3 * len(dataset),
            optimizer_parameter_count=len(parameters),
            batch_partitions=[list(part) for part in partitions],
            torch_generator_initial_sha256=initial_generator,
            torch_generator_current_sha256=initial_generator,
        )
        bn_snapshot, bn_modules = _apply_bn_policy((backbone, projector), bn_policy)
        ledger["bn_modules"] = bn_modules

        handles.append(encoder_probe.register_forward_hook(entry_hook))
        observed = _ObservedOptimizer(optimizer, ledger)
        sampler = _PartitionSampler(partitions)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
        if loader.num_workers != 0:
            raise RuntimeError(
                "num_workers=0 required; continuing dataset RNG must not be duplicated"
            )
        for views in loader:
            if len(views) != 3:
                raise RuntimeError("three views per item required")
            first, second, third = views
            size = len(first)
            batch_indices = list(sampler.used[-1])
            if len(batch_indices) != size:
                raise RuntimeError("sampler/dataset batch size mismatch")
            shapes = {tuple(t.shape) for t in (first, second, third)}
            if first.ndim != 4 or first.shape[1] != 3 or min(first.shape[2:]) < 1:
                raise ValueError("actual BCHW three-channel three-view tensors required")
            if len(shapes) != 1 or not first.is_floating_point():
                raise ValueError("equal floating-point BCHW three-view shapes required")
            # Explicit transfer to the resolved actual CUDA device, then guard.
            moved = [transfer.to_device(view, resolved_device) for view in (first, second, third)]
            for tensor in moved:
                if str(tensor.device) != expected_device:
                    raise ValueError("transferred view device does not match the resolved device")
                guard.require_tensor(tensor, "input-view", device=resolved_device)
                if not torch.isfinite(tensor).all():
                    ledger["inputs_finite"] = False
                    raise FloatingPointError("nonfinite three-view input; step refused")
            merged = torch.cat(moved, dim=0)
            if str(merged.device) != expected_device:
                raise ValueError("merged batch device does not match the resolved device")
            guard.require_tensor(merged, "merged-3N-input", device=resolved_device)
            phase = "forward"
            features = projector(backbone(merged))
            phase = "idle"
            if (
                not isinstance(features, torch.Tensor)
                or features.ndim != 2
                or features.shape[0] != 3 * size
                or not features.is_floating_point()
            ):
                raise ValueError("declared (3N, projection) feature tensor required")
            if str(features.device) != expected_device:
                raise ValueError("feature device does not match the resolved device")
            guard.require_tensor(features, "features", device=resolved_device)
            if not torch.isfinite(features).all():
                ledger["features_finite"] = False
                raise FloatingPointError("nonfinite projected features; step refused")
            first_projection, second_projection, aged_projection = features.chunk(3)
            loss = cacon_nt_xent(first_projection, second_projection, aged_projection, temperature)
            if not torch.isfinite(loss):
                ledger["losses_finite"] = False
                raise FloatingPointError("nonfinite NT-Xent loss; step refused")
            observed.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = 0.0
            for parameter in parameters:
                if parameter.grad is None:
                    continue
                if str(parameter.grad.device) != expected_device:
                    raise ValueError("gradient device does not match the resolved device")
                guard.require_tensor(parameter.grad, "gradient", device=resolved_device)
                if not torch.isfinite(parameter.grad).all():
                    ledger["gradients_finite"] = False
                    raise FloatingPointError("nonfinite parameter gradient; step refused")
                grad_norm += float(parameter.grad.detach().double().square().sum().item())
            grad_norm = math.sqrt(grad_norm)
            observed.step()
            seen.update(batch_indices)
            ledger["samples"] += size
            ledger["source_unique_rows"] = len(seen)
            ledger["completed_batches"] += 1
            ledger["last_batch_size"] = size
            ledger["last_batch_indices"] = batch_indices
            ledger["last_batch_loss"] = float(loss.detach().item())
            ledger["last_batch_gradient_norm"] = float(grad_norm)
            loss_total += float(loss.detach().item()) * size
            grad_total += grad_norm * size
            weight_total += size
            ledger["sample_weighted_loss"] = loss_total / weight_total
            ledger["sample_weighted_gradient_norm"] = grad_total / weight_total
            ledger["batch_summaries"].append(
                dict(
                    indices=batch_indices,
                    face_ids=[face_ids[index] for index in batch_indices],
                    size=size,
                    loss=float(loss.detach().item()),
                    gradient_norm=float(grad_norm),
                )
            )
            if progress is not None:
                progress(snapshot())

        ledger["ordered_source_indices"] = [index for part in sampler.used for index in part]
        ledger["face_ids_ordered"] = [face_ids[index] for index in ledger["ordered_source_indices"]]
        total = len(dataset)
        if (
            ledger["samples"] != total
            or ledger["source_unique_rows"] != total
            or ledger["completed_batches"] != len(partitions)
            or ledger["optimizer_steps"] != len(partitions)
            or ledger["optimizer_step_attempts"] != len(partitions)
            or ledger["encoder_image_instances"] != 3 * total
            or sorted(ledger["ordered_source_indices"]) != list(range(total))
            or [list(part) for part in sampler.used] != [list(part) for part in partitions]
        ):
            raise RuntimeError("observed single-epoch coverage/step/exposure mismatch")
        ledger["single_epoch_coverage"] = True
        ledger["bn_running_stats_unchanged"] = _bn_running_stats_unchanged(
            bn_snapshot, (backbone, projector)
        )
        if bn_policy == "frozen_all" and not ledger["bn_running_stats_unchanged"]:
            raise RuntimeError("frozen_all BN policy did not preserve running statistics")
        ledger["torch_generator_current_sha256"] = _generator_sha(torch_generator)
        ledger["epoch_complete"] = True
        return snapshot()
    except Exception as error:
        ledger["epoch_complete"] = False
        ledger["single_epoch_coverage"] = False
        try:
            partial = snapshot()
        except Exception:
            partial = dict(ledger)
            partial["state_snapshot_unavailable"] = True
        raise CaconStage1EpochFailure(str(error), partial) from error
    finally:
        for handle in handles:
            handle.remove()
