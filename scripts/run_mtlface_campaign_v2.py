"""Serial per-seed recognition-stage campaign runner for the MTLFace v2 stack.

Scope: this is a *recognition-stage adaptation* campaign built from the local
`arcface_r50_casia` common backbone, the spatial age/identity adapter and a
CosFace identity head. It is NOT a complete joint MTLFace/FAS reproduction and
must never claim ``full_joint_fas``, ``scientific_evaluation_complete`` or
``publication_ready``. It does not download data and does not run neural
training by itself when imported (only :func:`main` executes the campaign).

Protocol (fixed, declared before any outcome is inspected):
  * explicit epoch count and ``checkpoint_selection="last_epoch"``;
  * one continuing ``torch.Generator`` per seed, reused across epochs;
  * plain SGD over *every* trainable parameter of model + identity head once;
  * full dataset epochs (``num_workers=0``, ``drop_last=False``);
  * frozen BNs, CPU, single thread.

Fail-closed rules:
  * prerequisite manifests are hash-verified before and after the run;
  * all transitive prerequisite inputs, the actual code/settings/dependencies
    and the initial backbone weights are bound into every native manifest;
  * nonfinite epoch metrics or nonfinite snapshot tensors abort the cell;
  * a cell is only ``training_complete`` after all epochs and the last
    checkpoint are written; the campaign is only complete when all seeds are.

The heavy pieces (`model_factory`, `dataset_factory`, `epoch_runner`) are
dependency-injected so the campaign can be proven with tiny synthetic fakes.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest

CELL_EXPERIMENT = "mtlface-common-recognition-campaign-cell"
CAMPAIGN_EXPERIMENT = "mtlface-common-recognition-campaign"
PREFLIGHT_EXPERIMENT = "mtlface-common-recognition-campaign-preflight"
INITIALIZATION_EXPERIMENT = "mtlface-common-recognition-initialization"
SMOKE_EXPERIMENT = "mtlface-common-recognition-real-crop-smoke"
FACES_EXPERIMENT = "sota-recognition-face-list"

UNSUPPORTED_FLAGS = ("full_joint_fas", "scientific_evaluation_complete", "publication_ready")

PROTOCOL: dict[str, Any] = dict(
    backbone="arcface_r50_casia",
    scope="head",
    epochs=1,
    checkpoint_selection="last_epoch",
    optimizer="SGD",
    learning_rate=1e-5,
    batch_size=64,
    device="cpu",
    threads=1,
    batchnorm_policy="frozen_all",
    seeds=(42, 1, 2),
    generator="one continuing torch.Generator per seed; shuffle=True; num_workers=0; drop_last=False",
    sampling="uniform row shuffle; full dataset epoch; no early stopping; no observed-metric selection",
    scope_note="recognition-stage adaptation; not full joint MTLFace/FAS; no scientific claim",
)


# --------------------------------------------------------------------------- #
# path + record helpers
# --------------------------------------------------------------------------- #
def _ensure_importable() -> None:
    """Allow running this file directly from outside the repo root."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _snapshot(model: Any, identity_head: Any) -> dict[str, Any]:
    _ensure_importable()
    from scripts.mtlface_training_v2 import recognition_weights_snapshot

    return recognition_weights_snapshot(model, identity_head)


def resolve(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def verified(path: Path | str, experiment: str) -> dict[str, Any]:
    """Load a native manifest and hash-verify every bound input and output."""
    native = json.loads(Path(path).read_text(encoding="utf-8"))
    if native.get("experiment") != experiment:
        raise ValueError(f"wrong prerequisite type: expected {experiment}")
    for record in native["inputs"] + native["outputs"]:
        if file_record(resolve(record["path"])) != record:
            raise ValueError("prerequisite hash mismatch")
    return native


def _dedupe(paths: Iterable[Path]) -> list[Path]:
    seen: dict[str, Path] = {}
    for path in paths:
        resolved = path.resolve()
        seen.setdefault(str(resolved), resolved)
    return list(seen.values())


def _default_sources() -> list[Path]:
    root = PROJECT_ROOT
    sources = [
        Path(__file__).resolve(),
        root / "scripts/build_mtlface_recognition_v2.py",
        root / "scripts/mtlface_epoch_v2.py",
        root / "scripts/mtlface_training_v2.py",
        root / "scripts/mtlface_recognition_v2.py",
        root / "scripts/mtlface_components_v2.py",
        root / "scripts/prepare_sota_face_list.py",
        root / "scripts/check_mtlface_initialization.py",
        root / "scripts/smoke_mtlface_recognition_v2.py",
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    sources += sorted((root / "src/age_gap").rglob("*.py"))
    sources += sorted((root / "src/age_gap/settings").rglob("*.toml"))
    return sources


def _transitive(native: dict[str, Any]) -> list[Path]:
    return [resolve(record["path"]) for record in native["inputs"] + native["outputs"]]


def collect_sources(
    *,
    initialization_path: Path,
    faces_path: Path,
    smoke_path: Path,
    weights: Path,
    initialization: dict[str, Any],
    faces: dict[str, Any],
    smoke: dict[str, Any],
    extra_sources: Sequence[Path] | None,
) -> list[Path]:
    """Bind prerequisite transitive inputs + actual sources/settings/weights."""
    paths: list[Path] = [
        initialization_path,
        faces_path,
        smoke_path,
        weights,
        *_default_sources(),
        *(extra_sources or []),
    ]
    for native in (initialization, faces, smoke):
        paths.extend(_transitive(native))
    return _dedupe(paths)


# --------------------------------------------------------------------------- #
# prerequisite + numeric contracts
# --------------------------------------------------------------------------- #
def _reject_unsupported_claims(metrics: dict[str, Any], where: str) -> None:
    """Reject any prerequisite that actually claims a full-method outcome.

    Absent flags are tolerated for older prerequisites; present-and-truthy is not.
    """
    for key in UNSUPPORTED_FLAGS:
        if metrics.get(key):
            raise ValueError(f"{where}: unsupported claim {key} is not allowed")


def _require_unsupported_flags_false(metrics: dict[str, Any], where: str) -> None:
    """Our own artifacts must state the unsupported flags explicitly as False."""
    for key in UNSUPPORTED_FLAGS:
        if metrics.get(key) is not False:
            raise ValueError(f"{where}: {key} must be exactly False in this campaign")


def _contract_fields() -> tuple[str, ...]:
    # Seed-dependent fields are intentionally excluded: the seed=42
    # initialization must not force every seeded head to be identical.
    return (
        "backbone",
        "scope",
        "channels",
        "spatial_size",
        "embedding_dimension",
        "cosface_scale",
        "cosface_margin",
        "batchnorm_policy",
        "full_joint_fas",
        "weights",
    )


def load_prerequisites(
    *,
    initialization_path: Path,
    faces_path: Path,
    smoke_path: Path,
    weights: Path,
    scope: str,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate completed prerequisite flags and weight/backbone/count contracts."""
    initialization = verified(initialization_path, INITIALIZATION_EXPERIMENT)
    smoke = verified(smoke_path, SMOKE_EXPERIMENT)
    faces = verified(faces_path, FACES_EXPERIMENT)
    from scripts.prepare_sota_face_list import POLICY

    if (
        faces.get("parameters") != POLICY
        or faces["metrics"].get("preparation_complete") is not True
        or faces["metrics"].get("retained_crops_decoded") is not True
    ):
        raise ValueError("completed decoded face-list policy required")

    init_metrics = initialization["metrics"]
    if init_metrics.get("initialization_complete") is not True:
        raise ValueError("completed initialization prerequisite required")
    if init_metrics.get("training_complete") is True:
        raise ValueError("initialization prerequisite must not claim training")
    _reject_unsupported_claims(init_metrics, "initialization")

    smoke_metrics = smoke["metrics"]
    if smoke_metrics.get("real_crop_smoke_complete") is not True:
        raise ValueError("completed real-crop smoke prerequisite required")
    if smoke_metrics.get("training_complete") is True:
        raise ValueError("smoke prerequisite must not claim training")
    _reject_unsupported_claims(smoke_metrics, "smoke")

    contract = init_metrics.get("initialization")
    if not isinstance(contract, dict):
        raise ValueError("initialization metadata contract required")
    if contract.get("scope") != scope:
        raise ValueError("initialization scope contract mismatch")
    if contract.get("batchnorm_policy") != PROTOCOL["batchnorm_policy"]:
        raise ValueError("initialization batchnorm contract mismatch")
    if contract.get("full_joint_fas") is not False:
        raise ValueError("initialization must not claim full joint FAS")

    weights_record = file_record(weights)
    if contract.get("weights") != weights_record:
        raise ValueError("initialization weight contract mismatch")
    if weights_record not in initialization["inputs"]:
        raise ValueError("initialization does not bind the declared weights")
    if weights_record not in smoke["inputs"]:
        raise ValueError("smoke does not bind the declared weights")
    if file_record(initialization_path) not in smoke["inputs"]:
        raise ValueError("smoke does not bind the initialization prerequisite")
    if file_record(faces_path) not in smoke["inputs"]:
        raise ValueError("smoke does not bind the face-list prerequisite")

    smoke_parameters = smoke.get("parameters", {})
    if smoke_parameters.get("scope") != scope:
        raise ValueError("smoke scope contract mismatch")
    if smoke_parameters.get("device") != device:
        raise ValueError("smoke device contract mismatch")
    if smoke_parameters.get("optimizer") != PROTOCOL["optimizer"]:
        raise ValueError("smoke optimizer contract mismatch")

    classes = init_metrics.get("identity_classes")
    if isinstance(classes, bool) or not isinstance(classes, int) or classes < 1:
        raise ValueError("positive identity class count required")
    counts = faces.get("metrics", {}).get("counts", {})
    if counts.get("retained_people") != classes:
        raise ValueError("face-list identity count does not match initialization classes")
    return initialization, faces, smoke, contract


def _check_seeded_contract(
    metadata: dict[str, Any], contract: dict[str, Any], *, seed: int, weights: Path
) -> None:
    for key in _contract_fields():
        if metadata.get(key) != contract.get(key):
            raise ValueError(f"seeded model contract mismatch for {key}")
    if metadata.get("seed") != seed:
        raise ValueError("seeded model reported the wrong seed")
    if metadata.get("weights") != file_record(weights):
        raise ValueError("seeded model weights differ from the bound weights")


def _require_finite(value: Any, where: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite(item, f"{where}[{index}]")
    elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"nonfinite metric at {where}")


def _require_finite_snapshot(snapshot: dict[str, Any]) -> None:
    for module in ("model", "identity_head"):
        for key, tensor in snapshot[module].items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise FloatingPointError(f"nonfinite snapshot tensor {module}.{key}")


def validate_epoch(result, samples, batch_size):
    if not isinstance(result, dict):
        raise ValueError("epoch result dictionary required")
    for key, expected in (("samples", samples), ("batches", math.ceil(samples / batch_size))):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError("full epoch sample/batch coverage required")
    means = result.get("sample_weighted_means")
    if not isinstance(means, dict) or not {"total", "gradient_norm"} <= means.keys():
        raise ValueError("numeric total loss and gradient norm required")
    for value in means.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("numeric epoch metrics required")
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite metric in epoch means")
    if means["gradient_norm"] < 0:
        raise ValueError("nonnegative gradient norm required")


# --------------------------------------------------------------------------- #
# default (heavy) dependencies
# --------------------------------------------------------------------------- #
def default_model_factory(weights: Path, classes: int, *, seed: int, scope: str):
    _ensure_importable()
    from scripts.build_mtlface_recognition_v2 import build_model

    return build_model(weights, classes, seed=seed, scope=scope)


def default_dataset_factory(model: Any, faces_path: Path):
    _ensure_importable()
    from scripts.mtlface_epoch_v2 import load_bound_dataset

    return load_bound_dataset(faces_path, model.preprocess)


def default_epoch_runner(
    model, identity_head, optimizer, dataset, *, batch_size, generator, device
):
    _ensure_importable()
    from scripts.mtlface_epoch_v2 import run_recognition_epoch

    return run_recognition_epoch(
        model,
        identity_head,
        optimizer,
        dataset,
        batch_size=batch_size,
        generator=generator,
        device=device,
    )


# --------------------------------------------------------------------------- #
# campaign
# --------------------------------------------------------------------------- #
def run_campaign(
    *,
    faces: Path,
    initialization: Path,
    smoke: Path,
    weights: Path,
    out: Path,
    seeds: Sequence[int] = PROTOCOL["seeds"],
    epochs: int = PROTOCOL["epochs"],
    scope: str = PROTOCOL["scope"],
    learning_rate: float = PROTOCOL["learning_rate"],
    batch_size: int = PROTOCOL["batch_size"],
    device: str = PROTOCOL["device"],
    model_factory: Callable[..., Any] | None = None,
    dataset_factory: Callable[..., Any] | None = None,
    epoch_runner: Callable[..., Any] | None = None,
    extra_sources: Sequence[Path] | None = None,
    command: Sequence[str] | None = None,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Run the fixed recognition-stage campaign serially over seeds.

    All heavy collaborators can be injected for bounded synthetic verification.
    """
    faces, initialization, smoke, weights, out = (
        Path(faces),
        Path(initialization),
        Path(smoke),
        Path(weights),
        Path(out),
    )
    seeds = list(seeds)
    if not isinstance(preflight_only, bool):
        raise ValueError("boolean preflight_only required")
    if scope not in ("head", "tail", "full"):
        raise ValueError("head/tail/full scope required")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("positive integer epoch count required")
    if isinstance(learning_rate, bool) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("finite positive learning rate required")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("positive integer batch size required")
    if device != "cpu":
        raise ValueError("explicit CPU device required")
    if not seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("nonempty nonnegative integer seeds required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("unique seeds required")
    if out.exists():
        raise FileExistsError("fresh output directory required")
    torch.set_num_threads(PROTOCOL["threads"])

    injected = any(item is not None for item in (model_factory, dataset_factory, epoch_runner))
    _ensure_importable()
    model_factory = model_factory or default_model_factory
    dataset_factory = dataset_factory or default_dataset_factory
    epoch_runner = epoch_runner or default_epoch_runner

    init_native, faces_native, smoke_native, contract = load_prerequisites(
        initialization_path=initialization,
        faces_path=faces,
        smoke_path=smoke,
        weights=weights,
        scope=scope,
        device=device,
    )
    classes = init_native["metrics"]["identity_classes"]
    resolved_protocol = dict(PROTOCOL)
    resolved_protocol.update(
        seeds=seeds,
        epochs=epochs,
        scope=scope,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        implementation_mode="injected-test" if injected else "native-recognition",
        interop_threads=torch.get_num_interop_threads(),
        preflight_only=preflight_only,
    )

    sources = collect_sources(
        initialization_path=initialization,
        faces_path=faces,
        smoke_path=smoke,
        weights=weights,
        initialization=init_native,
        faces=faces_native,
        smoke=smoke_native,
        extra_sources=[
            *(extra_sources or []),
            *[
                Path(inspect.getsourcefile(fn))
                for fn in (model_factory, dataset_factory, epoch_runner)
            ],
        ],
    )
    before = [file_record(path) for path in sources]

    out.mkdir(parents=True)
    if preflight_only:
        metrics = dict(
            preflight_complete=True,
            prerequisite_hashes_verified=True,
            identity_classes=classes,
            retained_images=faces_native["metrics"]["counts"]["retained_images"],
            source_files=len(sources),
            training_complete=False,
            full_joint_fas=False,
            scientific_evaluation_complete=False,
            publication_ready=False,
        )
        summary = out / "summary.json"
        summary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        if [file_record(path) for path in sources] != before:
            raise RuntimeError("preflight sources changed")
        manifest = out / "summary.manifest.json"
        write_experiment_manifest(
            manifest,
            experiment=PREFLIGHT_EXPERIMENT,
            parameters=resolved_protocol,
            metrics=metrics,
            inputs=sources,
            outputs=[summary],
            command=list(command) if command is not None else None,
        )
        if [file_record(path) for path in sources] != before:
            manifest.unlink(missing_ok=True)
            raise RuntimeError("preflight sources changed during manifest write")
        verified(manifest, PREFLIGHT_EXPERIMENT)
        return metrics
    private = out / "private"
    private.mkdir()

    cell_manifests: list[Path] = []
    cell_outputs: list[Path] = []
    cells: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint = private / f"seed_{seed}_last.pt"
        history_path = private / f"seed_{seed}_history.json"
        cell_manifest = private / f"seed_{seed}.manifest.json"
        if checkpoint.exists() or history_path.exists() or cell_manifest.exists():
            raise FileExistsError(f"fresh output required for seed {seed}")

        torch.manual_seed(seed)
        model, identity_head, metadata = model_factory(weights, classes, seed=seed, scope=scope)
        _check_seeded_contract(metadata, contract, seed=seed, weights=weights)
        model.to(device)
        identity_head.to(device)
        dataset, dataset_native = dataset_factory(model, faces)
        if (
            getattr(dataset, "n_classes", None) != classes
            or dataset_native != faces_native
            or len(dataset) != faces_native["metrics"]["counts"]["retained_images"]
            or len(dataset) < 1
        ):
            raise ValueError("dataset classifier/count/native contract mismatch")

        optimizer = torch.optim.SGD(
            [
                p
                for module in (model, identity_head)
                for p in module.parameters()
                if p.requires_grad
            ],
            lr=learning_rate,
        )
        generator = torch.Generator().manual_seed(seed)
        history: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            result = epoch_runner(
                model,
                identity_head,
                optimizer,
                dataset,
                batch_size=batch_size,
                generator=generator,
                device=device,
            )
            validate_epoch(result, len(dataset), batch_size)
            _require_finite(result, f"seed{seed}.epoch{epoch}")
            history.append({"epoch": epoch} | dict(result))

        snapshot = _snapshot(model, identity_head)
        _require_finite_snapshot(snapshot)
        torch.save(snapshot, checkpoint)
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

        if [file_record(path) for path in sources] != before:
            checkpoint.unlink(missing_ok=True)
            history_path.unlink(missing_ok=True)
            raise RuntimeError("campaign sources changed during training; cell not credited")

        cell_metrics = dict(
            training_complete=True,
            epochs_requested=epochs,
            epochs_executed=epochs,
            selected_epoch=epochs,
            checkpoint_selection=PROTOCOL["checkpoint_selection"],
            seed=seed,
            identity_classes=classes,
            dataset_samples=len(dataset),
            history=history,
            full_joint_fas=False,
            scientific_evaluation_complete=False,
            publication_ready=False,
        )
        _require_unsupported_flags_false(cell_metrics, f"seed{seed}")
        write_experiment_manifest(
            cell_manifest,
            experiment=CELL_EXPERIMENT,
            parameters=resolved_protocol | {"seed": seed},
            metrics=cell_metrics,
            inputs=sources,
            outputs=[checkpoint, history_path],
            command=list(command) if command is not None else None,
        )
        if [file_record(path) for path in sources] != before:
            cell_manifest.unlink(missing_ok=True)
            raise RuntimeError("campaign sources changed during manifest write")
        verified(cell_manifest, CELL_EXPERIMENT)
        cell_manifests.append(cell_manifest)
        cell_outputs.extend([checkpoint, history_path])
        cells.append(
            dict(
                seed=seed,
                manifest=file_record(cell_manifest),
                checkpoint=file_record(checkpoint),
                history=file_record(history_path),
            )
        )
        del model, identity_head, optimizer, dataset, snapshot

    if [file_record(path) for path in sources] != before:
        raise RuntimeError("campaign sources changed; campaign not credited")
    for manifest in cell_manifests:
        verified(manifest, CELL_EXPERIMENT)

    campaign_metrics = dict(
        campaign_complete=True,
        training_complete=True,
        seeds_requested=seeds,
        completed_seeds=seeds,
        epochs_executed=epochs,
        checkpoint_selection=PROTOCOL["checkpoint_selection"],
        cells=cells,
        full_joint_fas=False,
        scientific_evaluation_complete=False,
        publication_ready=False,
    )
    _require_unsupported_flags_false(campaign_metrics, "campaign")
    summary = out / "summary.json"
    summary.write_text(json.dumps(campaign_metrics, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        out / "summary.manifest.json",
        experiment=CAMPAIGN_EXPERIMENT,
        parameters=resolved_protocol,
        metrics=campaign_metrics,
        inputs=sources,
        outputs=[summary, *cell_manifests, *cell_outputs],
        command=list(command) if command is not None else None,
    )
    if [file_record(path) for path in sources] != before:
        (out / "summary.manifest.json").unlink(missing_ok=True)
        raise RuntimeError("campaign sources changed during final manifest write")
    verified(out / "summary.manifest.json", CAMPAIGN_EXPERIMENT)
    return campaign_metrics


def seeds_from(text: str) -> list[int]:
    try:
        seeds = [int(part) for part in text.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("at least one nonnegative seed required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=seeds_from, default=list(PROTOCOL["seeds"]))
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--scope", choices=("head", "tail", "full"), required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default=PROTOCOL["device"])
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    _ensure_importable()
    torch.set_num_threads(PROTOCOL["threads"])
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    result = run_campaign(
        faces=args.faces,
        initialization=args.initialization,
        smoke=args.smoke,
        weights=args.weights,
        out=args.out,
        seeds=args.seeds,
        epochs=args.epochs,
        scope=args.scope,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
        command=None,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
