"""Bounded CPU numerical probe; never a cached trainer or scientific checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from age_gap.models.facenet import FaceNetBackbone
from age_gap.training.finetune import (
    ImagePairDataset,
    _apply_batchnorm_policy,
    _bb_prep,
    _set_trainable,
)
from age_gap.training.losses import ContrastivePairLoss
from scripts.run_oriented_campaign import verified

HEAD = {"last_linear.weight", "last_bn.weight", "last_bn.bias"}
RTOL, ATOL = 1e-6, 1e-7


def require_contract(model):
    from facenet_pytorch import InceptionResnetV1

    if type(model) is not FaceNetBackbone or type(model.net) is not InceptionResnetV1:
        raise ValueError("exact FaceNet/InceptionResnetV1 implementation required")
    if model.net.classify:
        raise ValueError("classification output is unsupported")
    active = {name for name, p in model.net.named_parameters() if p.requires_grad}
    if active != HEAD:
        raise ValueError("only all three native head tensors may be trainable")
    if any(m.training for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)):
        raise ValueError("all BatchNorm running statistics must be frozen")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("CPU float32 only")


def capture_features(model, images):
    """Capture native avgpool output; restore every training flag, leave RNG unchanged."""
    require_contract(model)
    flags = [(module, module.training) for module in model.modules()]
    rng = torch.get_rng_state().clone()
    captured = []
    hook = model.net.avgpool_1a.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach().clone())
    )
    try:
        model.eval()
        with torch.no_grad():
            model(images)
    finally:
        hook.remove()
        for module, training in flags:
            module.training = training
    if not torch.equal(rng, torch.get_rng_state()):
        raise RuntimeError("feature capture advanced CPU RNG")
    if len(captured) != 1 or captured[0].shape != (len(images), 1792, 1, 1):
        raise ValueError("unexpected native pre-dropout feature shape")
    if not torch.isfinite(captured[0]).all():
        raise ValueError("nonfinite trunk features")
    return captured[0]


def cached_forward(model, features):
    require_contract(model)
    if (
        features.ndim != 4
        or features.shape[1:] != (1792, 1, 1)
        or features.shape[0] < 1
        or features.requires_grad
        or features.device.type != "cpu"
        or features.dtype != torch.float32
        or not torch.isfinite(features).all()
    ):
        raise ValueError("detached finite CPU float32 pre-dropout features required")
    x = model.net.dropout(features)
    x = model.net.last_linear(x.reshape(len(x), -1))
    x = model.net.last_bn(x)
    # Both normalizations are present in the original net + project wrapper.
    return F.normalize(F.normalize(x, p=2, dim=1), dim=-1)


def compare(reference, candidate):
    if (
        reference.shape != candidate.shape
        or not torch.isfinite(reference).all()
        or not torch.isfinite(candidate).all()
    ):
        raise ValueError("nonfinite or incompatible compared tensors")
    return {
        "max_abs": float((reference.detach() - candidate.detach()).abs().max()),
        "within_tolerance": bool(torch.allclose(reference, candidate, rtol=RTOL, atol=ATOL)),
        "bitwise_equal": bool(torch.equal(reference, candidate)),
    }


def probe(base, batch, *, seed, sizes):
    require_contract(base)
    if not sizes or any(size < 1 or size > len(batch[0]) for size in sizes):
        raise ValueError("invalid probe batch sizes")
    ref, alt = copy.deepcopy(base), copy.deepcopy(base)
    first, second, labels, weights = batch
    # Deliberately extract the maximum batch then slice: tests cache re-batching too.
    features_a = capture_features(base, first)
    features_b = capture_features(base, second)
    opts = [
        torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-5)
        for model in (ref, alt)
    ]
    loss_fn = ContrastivePairLoss(margin=0.3)
    torch.manual_seed(seed)
    rows = []
    for index, size in enumerate(sizes, 1):
        for model in (ref, alt):
            model.train()
            _apply_batchnorm_policy(model, "frozen_all")
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        rng = torch.get_rng_state().clone()
        za, zb = ref(first[:size]), ref(second[:size])
        end_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        ca, cb = cached_forward(alt, features_a[:size]), cached_forward(alt, features_b[:size])
        if not torch.equal(end_rng, torch.get_rng_state()):
            raise RuntimeError("native and cached dropout RNG advancement differ")
        losses = [
            loss_fn(a, b, labels[:size], weights=weights[:size]) for a, b in ((za, zb), (ca, cb))
        ]
        comparisons = {
            "output_a": compare(za, ca),
            "output_b": compare(zb, cb),
            "loss": compare(*losses),
        }
        for loss in losses:
            loss.backward()
        refs, alts = dict(ref.net.named_parameters()), dict(alt.net.named_parameters())
        for name in sorted(HEAD):
            comparisons[f"gradient:{name}"] = compare(refs[name].grad, alts[name].grad)
        for opt in opts:
            opt.step()
        for name in sorted(HEAD):
            comparisons[f"parameter:{name}"] = compare(refs[name], alts[name])
            for state in ("step", "exp_avg", "exp_avg_sq"):
                comparisons[f"adam:{name}:{state}"] = compare(
                    opts[0].state[refs[name]][state], opts[1].state[alts[name]][state]
                )
        for model in (ref, alt):
            for name, tensor in model.net.state_dict().items():
                if name not in HEAD and not torch.equal(tensor, base.net.state_dict()[name]):
                    raise RuntimeError("frozen state changed")
            model.eval()
        with torch.no_grad():
            comparisons["evaluation"] = compare(
                ref(first[:size]), cached_forward(alt, features_a[:size])
            )
        rows.append(
            dict(
                seed=seed,
                step=index,
                batch_size=size,
                comparisons=comparisons,
                within_tolerance=all(item["within_tolerance"] for item in comparisons.values()),
            )
        )
    if all(torch.equal(dict(base.net.named_parameters())[name], refs[name]) for name in HEAD):
        raise RuntimeError("head never updated")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    torch.set_num_threads(1)
    started = time.monotonic()
    preflight = PROJECT_ROOT / "metrics/oriented_training_preflight_20261003/summary.manifest.json"
    native = verified(preflight, "oriented-uniform-training-preflight")
    pairs = (
        PROJECT_ROOT / "metrics/oriented_exposure_matching_20261003/private/low_candidate_arm.jsonl"
    )
    base = FaceNetBackbone()
    _set_trainable(base, "head")
    base.train()
    _apply_batchnorm_policy(base, "frozen_all")
    ds = ImagePairDataset("train", pairs_file=str(pairs), preprocess=_bb_prep(base))
    if len(ds) != 1200:
        raise ValueError("expected current oriented LOW training rows")
    paths = sorted({path for item in ds._items[:64] for path in item[:2]})
    source_paths = [
        PROJECT_ROOT / record["path"]
        for record in native["inputs"]
        if not record["path"].startswith("data/interim/faces/")
    ]
    inputs = list(dict.fromkeys([preflight, pairs, Path(__file__), *source_paths, *paths]))
    before = {str(path): file_record(path) for path in inputs}
    for record in native["inputs"]:
        path = PROJECT_ROOT / record["path"]
        if str(path) in before and before[str(path)] != record:
            raise RuntimeError("native preflight input changed before probe")
    batch = [torch.stack(list(values)) for values in zip(*(ds[i] for i in range(64)), strict=True)]
    rows = []
    for seed in (42, 1, 2):
        rows.extend(probe(base, batch, seed=seed, sizes=(64, 48, 64)))
        print(
            json.dumps(
                {
                    "seed": seed,
                    "within_tolerance": all(
                        r["within_tolerance"] for r in rows if r["seed"] == seed
                    ),
                }
            ),
            flush=True,
        )
    for path in inputs:
        if file_record(path) != before[str(path)]:
            raise RuntimeError("probe input changed")
    summary = dict(
        diagnostic_complete=True,
        bounded_tolerance_passed=all(r["within_tolerance"] for r in rows),
        accelerator_qualified=False,
        training_complete=False,
        scientific_evaluation_complete=False,
        publication_ready=False,
        elapsed_seconds=time.monotonic() - started,
        steps=rows,
    )
    args.out.mkdir(parents=True)
    result = args.out / "summary.json"
    result.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_experiment_manifest(
        args.out / "summary.manifest.json",
        experiment="facenet-head-cache-numerical-probe",
        parameters=dict(
            device="cpu",
            threads=1,
            scope="head",
            batchnorm_policy="frozen_all",
            seeds=[42, 1, 2],
            batches=[64, 48, 64],
            rtol=RTOL,
            atol=ATOL,
            rows="probe-only first64 LOW training rows; not campaign sampler",
            limitation="nine shadow Adam steps, not full trajectory, speedup or cache provenance qualification",
        ),
        metrics=summary,
        inputs=inputs,
        outputs=[result],
    )


if __name__ == "__main__":
    main()
