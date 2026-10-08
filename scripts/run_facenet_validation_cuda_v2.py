"""New fixed8 FaceNet CUDA producer with measured validation objective.

Fresh head/random-negative runs: batch16, default LR1e-6, last epoch, frozen BN.
This is not historical weak-training recovery or an identical-network claim.
Existing checksum-bound trainers, evaluations and running queue remain untouched.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest

EXPERIMENT = "facenet-fixed8-validation-loss-v2"
CROPS_DIR = "faces"
CASIA_WEIGHT_NAME = "20180408-102900-casia-webface.pt"
FIXED_BUDGET = dict(
    backbone="facenet",
    epochs_requested=8,
    epochs_executed=8,
    selected_epoch=8,
    checkpoint_selection="last_epoch",
    batchnorm_policy="frozen_all",
    trainable_scope="head",
    learning_rate=1e-6,
    batch_size=16,
    margin=0.3,
    gap_weight=0.0,
    crops_dir=CROPS_DIR,
)
SEEDS = (42, 1, 2)


# --------------------------------------------------------------------------------------
# Fail-closed device / seeding helpers (executed by run, never at import time)
# --------------------------------------------------------------------------------------
def require_cuda() -> None:
    """Fail closed: CUDA is mandatory, there is no CPU fallback for the producer."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback for the v2 producer")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA reported available but no device is visible")


def seed_everything(seed: int) -> torch.Generator:
    """Seed python/numpy/torch and return the explicit shuffle generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def _seed_worker(worker_id: int) -> None:
    """Deterministic per-worker seeding for any DataLoader worker process."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _trainable_parameters(backbone: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in backbone.parameters() if p.requires_grad]


def _freeze_batchnorm(backbone: torch.nn.Module) -> None:
    """frozen_all: BatchNorm keeps eval statistics; affine params follow scope."""
    for module in backbone.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def _set_scope(backbone: torch.nn.Module, scope: str) -> None:
    """Unfreeze only the backbone-declared scope (head/tail); other bytes stay frozen."""
    bb: Any = backbone
    names = bb.trainable_scopes[scope]
    for name, param in bb.net.named_parameters():
        param.requires_grad = any(token in name for token in names)


# --------------------------------------------------------------------------------------
# Objective: production parity for the optimizer, global numerator/denominator for telemetry
# --------------------------------------------------------------------------------------
def pair_objective(za, zb, labels, weights, margin: float):
    """Per-pair y*(1-cos) + (1-y)*relu(cos-margin); cos from L2-normalised embeddings."""
    cos = (za * zb).sum(dim=-1)
    per_pair = labels * (1.0 - cos) + (1.0 - labels) * torch.relu(cos - margin)
    return per_pair, cos


def validate_pairs(y: torch.Tensor, w: torch.Tensor) -> None:
    """Device-local guards; no CPU label tensor beside CUDA labels."""
    if y.ndim != 1 or w.shape != y.shape or not y.numel() or y.device != w.device:
        raise ValueError("aligned nonempty same-device label/weight vectors required")
    mass = float(w.sum())
    if (not torch.isfinite(w).all() or bool((w < 0).any())
            or not math.isfinite(mass) or mass <= 0):
        raise ValueError("finite nonnegative weights with finite positive total mass required")
    if not torch.isfinite(y).all() or not bool(((y == 0) | (y == 1)).all()):
        raise ValueError("binary finite labels required")


def batch_step_loss(backbone, ta, tb, y, w, loss_fn):
    """One forward pass; backward objective is exactly the production weighted mean.

    ``loss_fn`` is the immutable ``age_gap.training.losses.ContrastivePairLoss``:
    ``(per_pair * w).sum() / w.sum().clamp_min(1e-8)``.  Telemetry (per-pair, cos) is
    computed separately from the same embeddings.
    """
    za, zb = backbone(ta), backbone(tb)
    loss = loss_fn(za, zb, y, weights=w)
    per_pair, cos = pair_objective(za, zb, y, w, loss_fn.margin)
    return loss, per_pair, cos


def train_epoch(
    backbone: torch.nn.Module,
    dataset,
    *,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: str,
    batch_size: int,
    generator: torch.Generator,
) -> dict[str, float]:
    """One training epoch; returns the global weighted objective and diagnostics."""
    backbone.train()
    _freeze_batchnorm(backbone)
    params = _trainable_parameters(backbone)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        worker_init_fn=_seed_worker,
        drop_last=False,
    )
    numerator = denominator = 0.0
    grad_norm_total = 0.0
    batches = 0
    for ta, tb, y, w in loader:
        ta, tb = ta.to(device), tb.to(device)
        y, w = y.to(device).float(), w.to(device).float()
        validate_pairs(y, w)
        optimizer.zero_grad()
        # Optimizer objective == production ContrastivePairLoss (weighted MEAN per batch).
        loss, per_pair, _ = batch_step_loss(backbone, ta, tb, y, w, loss_fn)
        loss.backward()
        grad_sq = sum(
            float(torch.sum(p.grad.detach() ** 2)) for p in params if p.grad is not None
        )
        grad_norm_total += grad_sq**0.5
        optimizer.step()
        # Telemetry accumulates the GLOBAL numerator/denominator independently.
        numerator += float((per_pair.detach() * w).sum())
        denominator += float(w.sum().detach())
        batches += 1
    if denominator <= 0 or batches == 0:
        raise RuntimeError("empty or zero-weight training epoch")
    return dict(
        train_loss=numerator / denominator,
        train_numerator=numerator,
        train_denominator=denominator,
        mean_gradient_norm=grad_norm_total / batches,
        train_batches=batches,
        train_pairs=int(len(dataset)),
    )


@torch.no_grad()
def evaluate_epoch(
    backbone: torch.nn.Module,
    dataset,
    *,
    device: str,
    batch_size: int,
    margin: float,
) -> dict[str, float]:
    """Validation objective loss over the whole split as one numerator/denominator.

    Fails closed on non-finite or out-of-range AUC immediately, never at json dump time.
    """
    from age_gap.evaluation.metrics import roc_auc

    backbone.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    numerator = denominator = 0.0
    scores: list[float] = []
    labels: list[float] = []
    for ta, tb, y, w in loader:
        ta, tb = ta.to(device), tb.to(device)
        y, w = y.to(device).float(), w.to(device).float()
        validate_pairs(y, w)
        per_pair, cos = pair_objective(backbone(ta), backbone(tb), y, w, margin)
        numerator += float((per_pair * w).sum())
        denominator += float(w.sum())
        scores.extend(cos.cpu().tolist())
        labels.extend(y.cpu().tolist())
    if denominator <= 0:
        raise RuntimeError("empty or zero-weight validation split")
    try:
        auc = float(roc_auc(np.asarray(scores), np.asarray(labels)))
    except ValueError as exc:
        raise RuntimeError(f"validation AUC undefined: {exc}") from None
    if not math.isfinite(auc) or not (0.0 <= auc <= 1.0):
        raise RuntimeError(f"invalid validation_auc {auc!r}; failing closed")
    return dict(
        validation_loss=numerator / denominator,
        validation_numerator=numerator,
        validation_denominator=denominator,
        validation_auc=auc,
        validation_pairs=int(len(dataset)),
    )


# --------------------------------------------------------------------------------------
# Ancestry binding: real weight file, inventory, sources, pairs and crops
# --------------------------------------------------------------------------------------
def state_dict_digest(state: dict[str, torch.Tensor]) -> str:
    """Order-independent digest of an initialisation state_dict (kept as EXTRA evidence)."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def resolve_facenet_casia_weight(torch_home: Path | None = None) -> Path:
    """Exact cache path facenet-pytorch uses for ``pretrained='casia-webface'``."""
    base = torch_home
    if base is None:
        base = Path(
            os.path.expanduser(
                os.getenv(
                    "TORCH_HOME",
                    os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"),
                )
            )
        )
    return base / "checkpoints" / CASIA_WEIGHT_NAME


def require_pretrained_weight(weight: Path) -> Path:
    """Fail closed BEFORE make_backbone so facenet-pytorch cannot download implicitly."""
    weight = Path(weight)
    if not weight.is_file():
        raise FileNotFoundError(
            f"pretrained CASIA weight missing at {weight}; refusing implicit download"
        )
    return weight


def inventory_weight_record(inventory: Path) -> dict[str, Any]:
    """Read /models/facenet_casia/artifact from the project model inventory."""
    payload = json.loads(Path(inventory).read_text(encoding="utf-8"))
    try:
        record = payload["models"]["facenet_casia"]["artifact"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"model inventory lacks facenet_casia artifact: {exc}") from None
    return record


def validate_weight_record(weight: Path, inventory_record: dict[str, Any]) -> dict[str, Any]:
    """Exact file_record of the real weight must match the inventory record."""
    record = file_record(weight)
    if Path(str(inventory_record["path"])).resolve() != Path(weight).resolve():
        raise RuntimeError("pretrained weight path differs from model_inventory.json")
    for key in ("bytes", "sha256"):
        if record[key] != inventory_record.get(key):
            raise RuntimeError(
                f"pretrained weight {key} mismatch vs model_inventory.json"
            )
    return record


def bind_ancestry(inputs: list[Path], init_digest: str) -> dict[str, Any]:
    """Full before/after file_records for every declared input (weight, sources, crops)."""
    records = [file_record(p) for p in inputs]
    return {
        "inputs": records,
        "input_count": len(records),
        "model_init_state_sha256": init_digest,
    }


def source_ancestry_paths(
    *,
    script: Path,
    weight: Path,
    inventory: Path,
    pairs: Path,
    crops: list[Path],
    root: Path = PROJECT_ROOT,
) -> list[Path]:
    """Assemble the canonical ancestry input list (deduplicated, order preserved)."""
    import facenet_pytorch

    facenet_sources = sorted(Path(facenet_pytorch.__file__).resolve().parent.rglob("*.py"))
    paths: list[Path] = [
        Path(script).resolve(),
        Path(weight).resolve(),
        Path(inventory).resolve(),
        Path(pairs).resolve(),
        root / "pyproject.toml",
        root / "uv.lock",
        root / "src/age_gap/settings/settings.toml",
        *sorted((root / "src/age_gap").rglob("*.py")),
        root / "scripts/run_strong_backbone_cuda_cell_v1.py",
        root / "scripts/run_oriented_campaign.py",
        root / "scripts/run_restricted_matched_campaign.py",
        root / "scripts/evaluate_oriented_cuda_v1.py",
        root / "scripts/validation_pair_loss_v1.py",
        *facenet_sources,
        *[Path(p) for p in crops],
    ]
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(Path(path).resolve())
        if key not in seen:
            seen.add(key)
            out.append(Path(path).resolve())
    return out


def verify_completed_checkpoint(checkpoint: Path, epochs: int = 8) -> None:
    """Fail closed unless the checkpoint holds epochs 1..N with finite tensors."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("selected_epoch") != epochs:
        raise RuntimeError(f"checkpoint selected_epoch must be {epochs}")
    history = saved.get("history", [])
    if [row.get("epoch") for row in history] != list(range(1, epochs + 1)):
        raise RuntimeError(f"checkpoint must carry ordered epochs 1..{epochs}")
    for row in history:
        for key in ("train_loss", "validation_loss", "validation_auc",
                    "mean_gradient_norm", "epoch_seconds"):
            if key not in row or not math.isfinite(float(row[key])):
                raise RuntimeError(f"non-finite checkpoint diagnostic {key} at {row.get('epoch')}")
    for name, tensor in saved["state_dict"].items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"non-finite checkpoint tensor: {name}")


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def _collect_crops(train_ds, val_ds) -> list[Path]:
    ids = {p for ds in (train_ds, val_ds) for item in ds._items for p in item[:2]}
    return sorted(ids)


def run(args) -> Path:
    if args.out.exists():
        raise FileExistsError(f"fresh output directory required: {args.out}")
    if type(args.seed) is not int or args.seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if isinstance(args.lr, bool) or args.lr not in (1e-6, 1e-5):
        raise ValueError("learning rate must be finite and positive")
    require_cuda()

    weight = require_pretrained_weight(resolve_facenet_casia_weight())
    inventory = PROJECT_ROOT / "metrics/model_inventory.json"
    validate_weight_record(weight, inventory_weight_record(inventory))

    from age_gap.models.backbones import make_backbone
    from age_gap.training.finetune import ImagePairDataset, _bb_prep
    from age_gap.training.losses import ContrastivePairLoss

    pairs = (PROJECT_ROOT / args.pairs).resolve()
    if not pairs.is_file():
        raise FileNotFoundError("canonical pair file is missing; producer never mines it")
    if not torch.cuda.is_available():  # re-check after heavy imports
        raise RuntimeError("CUDA disappeared after import; aborting")
    device = "cuda"
    torch.set_num_threads(1)
    generator = seed_everything(args.seed)

    # Weight already verified to exist and match the inventory, so this cannot download.
    backbone = make_backbone("facenet", pretrained=True).to(device)
    init_digest = state_dict_digest(backbone.state_dict())
    prep = _bb_prep(backbone)
    train_ds = ImagePairDataset(
        split="train", pairs_file=str(pairs), gap_weight=0.0, preprocess=prep,
        crops_dir=CROPS_DIR,
    )
    val_ds = ImagePairDataset(
        split="val", pairs_file=str(pairs), preprocess=prep, crops_dir=CROPS_DIR
    )
    if not len(train_ds) or not len(val_ds):
        raise ValueError("nonempty train and validation splits required")

    crops = _collect_crops(train_ds, val_ds)
    inputs = source_ancestry_paths(
        script=Path(__file__), weight=weight, inventory=inventory, pairs=pairs, crops=crops
    )
    for path in inputs:
        if not Path(path).is_file():
            raise FileNotFoundError(f"declared ancestry input missing: {path}")
    before = bind_ancestry(inputs, init_digest)

    _set_scope(backbone, FIXED_BUDGET["trainable_scope"])
    params = _trainable_parameters(backbone)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    loss_fn = ContrastivePairLoss(margin=FIXED_BUDGET["margin"])

    args.out.mkdir(parents=True)
    checkpoint = args.out / "private/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    history: list[dict[str, Any]] = []
    for epoch in range(1, FIXED_BUDGET["epochs_requested"] + 1):
        started = monotonic()
        train = train_epoch(
            backbone, train_ds, optimizer=optimizer, loss_fn=loss_fn, device=device,
            batch_size=FIXED_BUDGET["batch_size"], generator=generator,
        )
        valid = evaluate_epoch(
            backbone, val_ds, device=device,
            batch_size=FIXED_BUDGET["batch_size"], margin=loss_fn.margin,
        )
        row = {"epoch": epoch, **train, **valid,
               "epoch_seconds": monotonic() - started,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        if not all(math.isfinite(float(row[k])) for k in (
            "train_loss", "validation_loss", "validation_auc",
            "mean_gradient_norm", "epoch_seconds"
        )):
            raise RuntimeError(f"non-finite diagnostics at epoch {epoch}")
        history.append(row)
        torch.save(
            {"state_dict": copy.deepcopy(backbone.state_dict()), "history": history,
             "selected_epoch": epoch, "checkpoint_selection": "last_epoch",
             "backbone": "facenet", "crops_dir": CROPS_DIR,
             "batchnorm_policy": "frozen_all"},
            checkpoint,
        )
        print(f"[{EXPERIMENT}] epoch {epoch}/8 train={row['train_loss']:.6f} "
              f"val={row['validation_loss']:.6f} auc={row['validation_auc']:.6f}")

    torch.cuda.synchronize()
    verify_completed_checkpoint(checkpoint, FIXED_BUDGET["epochs_requested"])

    after = bind_ancestry(inputs, init_digest)
    if before != after:
        raise RuntimeError("declared inputs changed during training; nothing published")

    summary = args.out / "summary.json"
    summary.write_text(
        json.dumps(
            {"experiment": EXPERIMENT, "seed": args.seed, "history": history,
             "training_complete": True, "evaluation_complete": False,
             "publication_ready": False,
             "limitation": "single fresh fixed8 cell; no legacy weak recovery claim"},
            indent=2, allow_nan=False,
        ) + "\n", encoding="utf-8",
    )
    target = args.out / "summary.manifest.json"
    expected_inputs = before["inputs"]
    write_experiment_manifest(
        target,
        experiment=EXPERIMENT,
        parameters={
            **FIXED_BUDGET,
            "learning_rate": args.lr,
            "seed": args.seed,
            "device": device,
            "threads": 1,
            "gap_weight": 0.0,
            "margin": FIXED_BUDGET["margin"],
            "pairs_file": str(pairs),
            "model_init_state_sha256": init_digest,
            "pretrained_weight_sha256": before["inputs"][1]["sha256"],
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(),
            "ancestry_input_count": before["input_count"],
        },
        metrics={"history": history, "training_complete": True,
                 "evaluation_complete": False, "publication_ready": False,
                 "selected_epoch": 8},
        inputs=inputs,
        outputs=[summary, checkpoint],
    )
    from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs

    # Guard: refuse a published manifest if any declared input changed during writes.
    validate_written_inputs(target, expected_inputs)
    native = json.loads(target.read_text(encoding="utf-8"))
    if [r["epoch"] for r in native["metrics"]["history"]] != list(range(1, 9)):
        raise ValueError("manifest must carry eight ordered completed epochs")
    if native["metrics"].get("training_complete") is not True:
        raise ValueError("completed training flag required")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=list(SEEDS), required=True)
    parser.add_argument("--lr", type=float, choices=[1e-6, 1e-5], default=1e-6)
    parser.add_argument("--pairs", default="data/processed/pairs.jsonl")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(f"dry-run: {EXPERIMENT} seed={args.seed} lr={args.lr}; no CUDA probe, no training")
        return
    print(run(args))


if __name__ == "__main__":
    main()
