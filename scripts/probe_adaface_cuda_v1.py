"""Bounded real-crop CUDA recovery diagnostic; never a scientific trained cell."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import ImagePairDataset, _apply_batchnorm_policy, _bb_prep
from age_gap.training.losses import ContrastivePairLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh diagnostic output directory required")
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no implicit CPU fallback")
    started = time.monotonic()
    pairs = PROJECT_ROOT / "data/processed/pairs.jsonl"
    weights = PROJECT_ROOT / "models/adaface_ir101.pt"
    sources = sorted((PROJECT_ROOT / "src/age_gap").rglob("*.py"))
    inputs = [pairs, weights, Path(__file__), *sources,
              PROJECT_ROOT / "src/age_gap/settings/settings.toml",
              PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "uv.lock"]
    before = {str(path): file_record(path) for path in inputs}
    model = make_backbone("adaface_ir101", pretrained=True).cuda()
    dataset = ImagePairDataset("train", pairs_file=str(pairs), preprocess=_bb_prep(model))
    if len(dataset) < 48:
        raise RuntimeError("at least 48 usable training pairs required")
    # Explicit probe-only sampling; this is not the campaign DataLoader schedule.
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:48].tolist()
    crops = sorted({path for i in indices for path in dataset._items[i][:2]})
    inputs.extend(crops)
    before.update({str(path): file_record(path) for path in crops})
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    loss_fn = ContrastivePairLoss(margin=0.3)
    probe_parameter = next(model.parameters())
    initial = probe_parameter.detach().cpu().clone()
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        batch = [dataset[i] for i in indices[16 * step:16 * (step + 1)]]
        first, second, labels, pair_weights = [torch.stack(list(values)).cuda() for values in zip(*batch, strict=True)]
        model.train()
        _apply_batchnorm_policy(model, "frozen_all")
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(first), model(second), labels, weights=pair_weights)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("nonfinite gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("nonfinite parameter")
        model.eval()
        with torch.no_grad():
            embedding = model(first)
        if not torch.isfinite(embedding).all():
            raise RuntimeError("nonfinite eval embedding")
        torch.cuda.synchronize()
        row = dict(step=step + 1, pairs=16, loss=float(loss.detach()),
                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                   peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
                   elapsed_seconds=time.monotonic() - started)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if torch.equal(initial, probe_parameter.detach().cpu()):
        raise RuntimeError("probe parameter did not change")
    for path in inputs:
        if file_record(path) != before[str(path)]:
            raise RuntimeError("diagnostic input changed during execution")
    metrics = dict(diagnostic_complete=True, training_complete=False, scientific_evaluation_complete=False,
                   full_method_parity=False, optimizer_steps=3, eval_forwards=3, pairs_presented=48,
                   probe_parameter_changed=True, steps=rows)
    result = args.out / "summary.json"
    result.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_experiment_manifest(
        args.out / "summary.manifest.json", experiment="adaface-real-crop-cuda-recovery-diagnostic",
        parameters=dict(backbone="adaface_ir101", scope="full", batch_size=16, seed=42, lr=1e-6,
                        batchnorm_policy="frozen_all", device="cuda", threads=1,
                        sampling="probe-only CPU randperm seed42 first48 usable train pairs",
                        torch_cuda=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                        cudnn_enabled=torch.backends.cudnn.enabled, cudnn_benchmark=torch.backends.cudnn.benchmark,
                        gpu_name=torch.cuda.get_device_name(),
                        limitation="three steps only; not long-run stability or a scientific checkpoint"),
        metrics=metrics, inputs=inputs, outputs=[result],
    )


if __name__ == "__main__":
    main()
