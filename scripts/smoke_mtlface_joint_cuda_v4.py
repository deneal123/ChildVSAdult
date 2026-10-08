"""CUDA-only real-crop joint recognition+FAS smoke; native CUDA execution still required.

This is a faithful port of the immutable CPU smoke ``scripts/smoke_mtlface_joint_v3.py``
with *every* selection rule, stage order, immutability check and artifact preserved, and
only the execution device changed:

* the native CPU prerequisite identity is preserved: prerequisites are still loaded with
  ``load_prerequisites(..., device="cpu")`` and those remain CPU evidence only;
* the actual smoke is CUDA-only and fail closed: ``CudaRuntime.require`` runs before any
  model construction, there is no CPU fallback, and the declared/observed device must be
  ``cuda``;
* python/numpy/torch/CUDA are seeded, model/head/generator/discriminator are transferred
  to CUDA and *every* batch tensor is moved to CUDA before the optimizers are constructed;
* the actual GPU name, torch/CUDA versions and peak allocated/reserved bytes are persisted.

The two-row selection is preserved exactly: the first explicit-age record and the first
missing-age record, the first explicit target image repeated twice, and the target group
derived from the explicit age against boundaries (10,20,30,40,50,60). The recognition step
runs first (frozen recognition BN, model/head update required), then the FAS D/G step
(adapt generator BN, recognizer and identity head proven unchanged, actual G and D weight
updates required), and finite post-joint embeddings are required.

A dry run (no ``--execute``) touches nothing: no CUDA probe, no native prerequisite read,
no model construction, no output. Output is always fresh (no implicit resume); a failed run
leaves no success marker because the manifest is written last and removed on drift. This is
a smoke, never scientific training / full-coverage / common-budget evidence.

No downloads, no checkpoint selection, and no full-method-parity / publication claim.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import inspect
import json
import random
from pathlib import Path

import numpy as np
import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.build_mtlface_recognition_v2 import build_model
from scripts.mtlface_epoch_v2 import load_bound_dataset
from scripts.mtlface_fas_networks_v2 import AgingModule, PatchDiscriminator, provenance
from scripts.mtlface_fas_training_v2 import fas_step
from scripts.mtlface_joint_adapter_v2 import CommonJointAdapter
from scripts.mtlface_training_v2 import recognition_step, recognition_weights_snapshot
from scripts.run_mtlface_campaign_v2 import _require_finite, collect_sources, load_prerequisites
from scripts.run_mtlface_joint_campaign_cuda_v4 import CudaRuntime

EXPERIMENT = "mtlface-common-real-crop-joint-smoke-cuda-v4"
RECOGNITION_BN_POLICY = "frozen_all"
GENERATOR_BN_POLICY = "adapt"
SELECTION = "first explicit and first missing age; first explicit target repeated twice"
AGE_BOUNDARIES = (10, 20, 30, 40, 50, 60)


def flags():
    return dict(
        full_method_parity=False,
        common_budget_matched=False,
        scientific_evaluation_complete=False,
        publication_ready=False,
    )


def seed_all(seed: int) -> None:
    """Seed python/numpy/torch/CUDA before the smoke touches any model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device_batch(runtime, values, device="cuda"):
    return tuple(runtime.transfer(value, device) for value in values)


def _default_optimizers(model, head, generator, discriminator):
    fr_opt = torch.optim.SGD(
        [p for m in (model, head) for p in m.parameters() if p.requires_grad], lr=1e-5
    )
    g_opt = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.99))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.99))
    return fr_opt, g_opt, d_opt


def _default_modules():
    return AgingModule(), PatchDiscriminator()


def build_selection(dataset):
    """Exact v3 two-row selection: first explicit age, first missing age."""
    explicit = next((i for i, row in enumerate(dataset.records) if row.age is not None), None)
    missing = next((i for i, row in enumerate(dataset.records) if row.age is None), None)
    if explicit is None or missing is None:
        raise ValueError("explicit and missing age smoke rows required")
    indices = [explicit, missing]
    items = [dataset[index] for index in indices]
    images = torch.stack([item[0] for item in items])
    identities = torch.tensor([item[1] for item in items], dtype=torch.int64)
    ages = torch.tensor([item[2] for item in items], dtype=torch.int64)
    target_group = sum(dataset.records[explicit].age > boundary for boundary in AGE_BOUNDARIES)
    groups = torch.full((2,), target_group, dtype=torch.int64)
    targets = images[:1].repeat(2, 1, 1, 1)
    return indices, images, identities, ages, targets, groups


def _configure_threads() -> None:
    """Pin single-threaded execution; interop threads may only be set once per process."""
    torch.set_num_threads(1)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _verify_tensor_devices(tensors, device):
    for tensor in tensors:
        if tensor.device.type != device:
            raise ValueError(f"batch tensor not on {device}")


def _assert_on_device(modules, device):
    for role, module in modules.items():
        for name, value in list(module.named_parameters()) + list(module.named_buffers()):
            if value.device.type != device:
                raise ValueError(f"{role}.{name} not on {device}")


def _same_cpu_tensor(left, right):
    """Compare states on one device and require matching tensor dtype/shape."""
    return (left.dtype == right.dtype and left.shape == right.shape
            and torch.equal(left.detach().cpu(), right.detach().cpu()))


def _validate_protocol(seed, scope, batch_size):
    if type(seed) is not int or seed != 42 or scope != "head":
        raise ValueError("historical smoke requires seed42 and head scope")
    if type(batch_size) is not int or batch_size != 2:
        raise ValueError("fixed two-row smoke requires batch_size=2")


def run_joint_smoke(
    model,
    head,
    dataset,
    generator,
    discriminator,
    *,
    runtime,
    seed,
    optimizer_factory=None,
    recognition_step_fn=None,
    fas_step_fn=None,
):
    """CUDA-only port of the v3 smoke: same selection, same FR-then-D/G order."""
    _validate_protocol(seed, "head", 2)
    optimizer_factory = optimizer_factory or _default_optimizers
    recognition_step_fn = recognition_step_fn or recognition_step
    fas_step_fn = fas_step_fn or fas_step
    runtime.require()
    runtime.seed(seed)
    indices, images, identities, ages, targets, groups = build_selection(dataset)
    modules = {
        "model": model,
        "identity_head": head,
        "generator": generator,
        "discriminator": discriminator,
    }
    # Transfer every module to CUDA BEFORE any optimizer is constructed.
    for module in modules.values():
        runtime.transfer(module, "cuda")
    _assert_on_device(modules, "cuda")
    # And every batch tensor, also before optimizers exist.
    images, identities, ages = _to_device_batch(runtime, (images, identities, ages))
    targets, groups = _to_device_batch(runtime, (targets, groups))
    _verify_tensor_devices((images, identities, ages, targets, groups), "cuda")

    if not hasattr(model, "set_batchnorm_policy"):
        raise ValueError("recognition model must expose set_batchnorm_policy")
    model.set_batchnorm_policy(RECOGNITION_BN_POLICY)
    before_fr = recognition_weights_snapshot(model, head)
    fr_opt, g_opt, d_opt = optimizer_factory(model, head, generator, discriminator)
    print("cuda joint smoke: recognition step", flush=True)
    fr = recognition_step_fn(model, head, fr_opt, (images, identities, ages))
    after_fr = recognition_weights_snapshot(model, head)
    updated_fr = sum(
        not _same_cpu_tensor(value, before_fr["model"][name]) for name, value in model.named_parameters()
    )
    updated_head = sum(
        not _same_cpu_tensor(value, before_fr["identity_head"][name])
        for name, value in head.named_parameters()
    )
    for name, value in model.named_buffers():
        if not _same_cpu_tensor(value, before_fr["model"][name]):
            raise ValueError("recognition frozen BN/buffer changed")
    if not updated_fr or not updated_head:
        raise ValueError("recognition model/head update required")
    del before_fr
    before_g = generator.conv3.weight.detach().clone()
    before_d = discriminator.conv1.weight.detach().clone()
    print("cuda joint smoke: FAS D/G step", flush=True)
    fas = fas_step_fn(
        model,
        generator,
        discriminator,
        g_opt,
        d_opt,
        images,
        targets,
        groups,
        generator_bn_policy=GENERATOR_BN_POLICY,
    )
    for name, value in model.state_dict().items():
        if not _same_cpu_tensor(value, after_fr["model"][name]):
            raise ValueError("recognizer state changed during FAS")
    for name, value in head.state_dict().items():
        if not _same_cpu_tensor(value, after_fr["identity_head"][name]):
            raise ValueError("identity head changed during FAS")
    if torch.equal(generator.conv3.weight, before_g) or torch.equal(
        discriminator.conv1.weight, before_d
    ):
        raise ValueError("actual G/D parameter update required")
    model.eval()
    with torch.no_grad():
        embeddings = model(images)
    if (embeddings.ndim != 2 or embeddings.shape[0] != 2 or embeddings.shape[1] < 1
            or not embeddings.is_floating_point() or not torch.isfinite(embeddings).all()):
        raise FloatingPointError("nonfinite post-joint embedding")
    return dict(
        recognition=fr,
        fas=fas,
        updated_recognition_parameter_tensors=updated_fr,
        updated_classifier_parameter_tensors=updated_head,
        recognition_buffers_frozen=True,
        recognizer_state_unchanged_during_fas=True,
        identity_head_unchanged_during_fas=True,
        embeddings_finite=True,
        embedding_shape=list(embeddings.shape),
        target_group=int(groups[0]),
        joint_steps_complete=True,
        training_complete=False,
        scientific_evaluation_complete=False,
        full_method_parity=False,
        publication_ready=False,
    ), indices


def run_smoke(
    *,
    initialization_path,
    faces_path,
    smoke_path,
    weights,
    out,
    seed=42,
    scope="head",
    batch_size=2,
    runtime=None,
    model_factory=None,
    adapter_factory=None,
    dataset_factory=None,
    modules_factory=None,
    smoke_runner=None,
    require_fn=None,
    seed_fn=None,
):
    """Execute the CUDA smoke. Requires explicit construction; never dry-runs itself."""
    _validate_protocol(seed, scope, batch_size)
    runtime = runtime if runtime is not None else CudaRuntime()
    require_fn = require_fn or runtime.require
    seed_fn = seed_fn or runtime.seed
    injected_fns = [
        fn
        for fn in (model_factory, adapter_factory, dataset_factory, modules_factory, smoke_runner,
                   require_fn, seed_fn)
        if fn is not None
    ]
    injected = bool(injected_fns) or not isinstance(runtime, CudaRuntime)
    model_factory = model_factory or functools.partial(_default_models, scope=scope, seed=seed)
    adapter_factory = adapter_factory or _default_adapter
    dataset_factory = dataset_factory or _default_dataset
    modules_factory = modules_factory or _default_modules
    smoke_runner = smoke_runner or run_joint_smoke
    if out.exists():
        raise FileExistsError("fresh output required; implicit resume refused")
    _configure_threads()
    # Fail closed BEFORE any model construction or optimizer creation.
    require_fn()
    init, faces, smoke, contract = load_prerequisites(
        initialization_path=initialization_path,
        faces_path=faces_path,
        smoke_path=smoke_path,
        weights=weights,
        scope=scope,
        device="cpu",
    )
    extras = [Path(__file__).resolve()]
    extras += [Path(inspect.getsourcefile(fn)) for fn in injected_fns]
    extras += [
        PROJECT_ROOT / ("scripts/" + name + ".py")
        for name in (
            "smoke_mtlface_joint_v3",
            "run_mtlface_joint_campaign_cuda_v4",
            "mtlface_fas_networks_v2",
            "mtlface_fas_training_v2",
            "mtlface_joint_adapter_v2",
            "mtlface_training_v2",
            "mtlface_recognition_v2",
            "mtlface_epoch_v2",
            "build_mtlface_recognition_v2",
            "run_mtlface_campaign_v2",
        )
    ]
    sources = collect_sources(
        initialization_path=initialization_path,
        faces_path=faces_path,
        smoke_path=smoke_path,
        weights=weights,
        initialization=init,
        faces=faces,
        smoke=smoke,
        extra_sources=extras,
    )
    before = [file_record(path) for path in sources]

    def stable():
        if before != [file_record(path) for path in sources]:
            raise RuntimeError("cuda joint smoke sources changed; completion not credited")

    stable()
    seed_fn(seed)
    runtime.reset_peak()
    base, head, metadata = model_factory(weights, init["metrics"]["identity_classes"])
    if metadata != contract:
        raise ValueError("initialization metadata mismatch")
    model = adapter_factory(base)
    del base
    dataset, native = dataset_factory(model, faces_path)
    if native != faces or dataset.n_classes != head.classes:
        raise ValueError("dataset contract mismatch")
    generator, discriminator = modules_factory()
    stable()
    metrics, indices = smoke_runner(
        model, head, dataset, generator, discriminator, runtime=runtime, seed=seed
    )
    peaks = runtime.peaks()
    binding = runtime.binding()
    if binding.get("device") != "cuda":
        raise ValueError("cuda smoke device evidence required")
    metrics = metrics | dict(
        real_crop_joint_smoke_complete=True,
        cuda_smoke_complete=True,
        cuda_probed=True,
        **flags(),
        **binding,
        **peaks,
    )
    _require_finite(metrics, "cuda smoke metrics")
    if before != [file_record(path) for path in sources]:
        raise RuntimeError("cuda joint smoke sources changed")
    out.mkdir(parents=True)
    private = out / "private"
    private.mkdir()
    selection = private / "selection.json"
    selection.write_text(json.dumps(dict(dataset_indices=indices)) + "\n", encoding="utf-8")
    summary = out / "summary.json"
    summary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest = out / "summary.manifest.json"
    parameters = (
        dict(
            seed=seed,
            scope=scope,
            device="cuda",
            cuda_smoke_required=True,
            threads=1,
            selection=SELECTION,
            selection_not_uniform_training_stream=True,
            recognition_optimizer="SGD",
            recognition_lr=1e-5,
            fas_optimizer="Adam",
            fas_lr=1e-4,
            fas_betas=[0.5, 0.99],
            generator_bn_policy=GENERATOR_BN_POLICY,
            recognition_bn_policy=RECOGNITION_BN_POLICY,
            batch_size=batch_size,
            recognition_steps=1,
            fas_steps=1,
            model_discarded=True,
            implementation_mode="injected-test" if injected else "native-cuda-smoke",
            architecture_reference=provenance(),
            native_cpu_prerequisite_device="cpu",
            native_cpu_prerequisite_is_training_proof=False,
        )
        | binding
        | peaks
    )
    try:
        write_experiment_manifest(
            manifest,
            experiment=EXPERIMENT,
            parameters=parameters,
            metrics=metrics,
            inputs=sources,
            outputs=[summary, selection],
        )
        stable()
    except BaseException:
        manifest.unlink(missing_ok=True)
        raise
    print(json.dumps(metrics))
    return metrics


def _default_adapter(base):
    return CommonJointAdapter(base.backbone, base.separation, base.age_head, base.age_adversary)


def _default_models(weights, classes, *, scope, seed):
    base, head, metadata = build_model(weights, classes, seed=seed, scope=scope)
    return base, head, metadata


def _default_dataset(model, faces_path):
    return load_bound_dataset(faces_path, model.preprocess)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("faces", "initialization", "smoke", "weights", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            f"dry-run: {EXPERIMENT} out={args.out}; no CUDA probe, no native prerequisite "
            "verification, no model construction, no output. Use --execute to run the "
            "CUDA-only smoke (CUDA host required)."
        )
        return
    run_smoke(
        initialization_path=args.initialization,
        faces_path=args.faces,
        smoke_path=args.smoke,
        weights=args.weights,
        out=args.out,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
