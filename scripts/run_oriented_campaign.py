"""Separate prepare/smoke/serial-train stages for uniform oriented age-gap controls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest

PROTOCOL = dict(
    backbone="facenet",
    epochs_requested=10,
    epochs_executed=10,
    selected_epoch=10,
    checkpoint_selection="last_epoch",
    batchnorm_policy="frozen_all",
    trainable_scope="head",
    learning_rate=3e-5,
    batch_size=64,
    margin=0.3,
    gap_weight=0.0,
    crops_dir="faces",
    seeds=[42, 1, 2],
    device="cpu",
    threads=1,
    sampling="uniform row shuffle; drop_last=false; no diagnostic weights",
    scope="conditional within-source age-gap sensitivity, not causal source evidence",
)


def verified(path, experiment):
    from scripts.run_restricted_matched_campaign import verify_records

    native = json.loads(path.read_text(encoding="utf-8"))
    if native.get("experiment") != experiment:
        raise ValueError("wrong prerequisite type")
    verify_records(native["inputs"] + native["outputs"])
    return native


def paths_from(native):
    return [
        Path(r["path"]) if Path(r["path"]).is_absolute() else PROJECT_ROOT / r["path"]
        for r in native["inputs"] + native["outputs"]
    ]


def training_contract(native, arm_path, seed):
    p = native.get("parameters", {})
    keys = (
        "backbone",
        "epochs_requested",
        "epochs_executed",
        "selected_epoch",
        "checkpoint_selection",
        "batchnorm_policy",
        "trainable_scope",
        "learning_rate",
        "batch_size",
        "margin",
        "gap_weight",
        "crops_dir",
    )
    if native.get("experiment") != "pair-contrastive-backbone-finetune" or any(
        p.get(k) != PROTOCOL[k] for k in keys
    ):
        raise ValueError("actual training protocol mismatch")
    if (
        p.get("seed") != seed
        or Path(p.get("pairs_file", "")).resolve() != arm_path.resolve()
        or file_record(arm_path) not in native["inputs"]
    ):
        raise ValueError("training arm/seed linkage mismatch")


def prepare(out):
    from facenet_pytorch.models import inception_resnet_v1

    from age_gap.training.finetune import ImagePairDataset
    from scripts.run_restricted_matched_campaign import audit_crops
    from scripts.train_matched_agegap_arms import validate_arm_files

    root = PROJECT_ROOT
    nuisance = root / "metrics/oriented_nuisance_20261003/summary.manifest.json"
    prepared = verified(nuisance, "oriented-fresh-nuisance-joint-audit")
    if prepared["metrics"].get("execution_complete") is not True:
        raise ValueError("completed oriented audit required")
    arms = {
        a: root
        / f"metrics/oriented_exposure_matching_20261003/private/{a.lower()}_candidate_arm.jsonl"
        for a in ("LOW", "CROSS")
    }
    records = prepared["inputs"] + prepared["outputs"]
    if any(file_record(p) not in records for p in arms.values()):
        raise ValueError("arm not bound by prerequisite")
    clusters = root / "data/processed/person_clusters.jsonl"
    groups = {}
    for r in read_jsonl(clusters):
        if r["identity_group_id"] in groups and groups[r["identity_group_id"]] != r["person_id"]:
            raise ValueError("conflicting person mapping")
        groups[r["identity_group_id"]] = r["person_id"]
    validation = validate_arm_files(arms["LOW"], arms["CROSS"], group_to_person=groups)
    inventory = root / "metrics/model_inventory.json"
    weight = (
        Path(inception_resnet_v1.get_torch_home()) / "checkpoints/20180408-102900-casia-webface.pt"
    )
    if (
        file_record(weight)
        != json.loads(inventory.read_text(encoding="utf-8"))["models"]["facenet_casia"]["artifact"]
    ):
        raise ValueError("actual initial weight differs from inventory")
    sources = [
        Path(__file__),
        root / "scripts/run_restricted_matched_campaign.py",
        root / "scripts/train_matched_agegap_arms.py",
    ]
    sources += sorted((root / "src/age_gap").rglob("*.py"))
    sources += sorted((root / "src/age_gap/settings").rglob("*.toml"))
    sources += sorted(Path(inception_resnet_v1.__file__).parent.rglob("*.py"))
    inputs = list(
        dict.fromkeys(
            p.resolve()
            for p in [
                *paths_from(prepared),
                nuisance,
                inventory,
                weight,
                *sources,
                root / "uv.lock",
                root / "pyproject.toml",
            ]
        )
    )
    before = [file_record(p) for p in inputs]
    loaded = {a: list(read_jsonl(path)) for a, path in arms.items()}
    crops, crop_audit = audit_crops(loaded)
    crop_paths = [
        Path(r["path"]) if Path(r["path"]).is_absolute() else root / r["path"] for r in crops
    ]
    counts = {}
    for arm, path in arms.items():
        counts[arm] = {}
        for split in ("train", "val", "test"):
            expected = sum(r["split"] == split for r in loaded[arm])
            observed = len(ImagePairDataset(split=split, pairs_file=str(path), gap_weight=0.0))
            if observed != expected:
                raise ValueError("loader dropped rows")
            counts[arm][split] = observed
    if before != [file_record(p) for p in inputs] or crops != [file_record(p) for p in crop_paths]:
        raise ValueError("inputs changed during preflight")
    result = dict(
        protocol=PROTOCOL,
        crop_audit=crop_audit,
        arm_validation=validation,
        loader_counts=counts,
        preflight_complete=True,
        training_complete=False,
        runtime_stability_checked=False,
        publication_ready=False,
    )
    output = out / "summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    target = out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="oriented-uniform-training-preflight",
        parameters=PROTOCOL,
        metrics=result,
        inputs=[*inputs, *crop_paths],
        outputs=[output],
    )
    if [*before, *crops] != json.loads(target.read_text(encoding="utf-8"))["inputs"]:
        target.unlink()
        raise ValueError("inputs changed during preflight manifest write")
    print(json.dumps(result, indent=2), flush=True)


def cpu_setup():
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_available():
        raise ValueError("explicit CPU-only environment required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return torch


def smoke(preflight, out):
    native = verified(preflight, "oriented-uniform-training-preflight")
    if native["metrics"].get("preflight_complete") is not True:
        raise ValueError("complete preflight required")
    torch = cpu_setup()
    from torch.utils.data import DataLoader

    from age_gap.models.backbones import make_backbone
    from age_gap.training.finetune import ImagePairDataset, _apply_batchnorm_policy, _set_trainable
    from age_gap.training.losses import ContrastivePairLoss

    torch.manual_seed(42)
    arm = (
        PROJECT_ROOT / "metrics/oriented_exposure_matching_20261003/private/low_candidate_arm.jsonl"
    )
    model = make_backbone("facenet", pretrained=True).cpu()
    _set_trainable(model, "head")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-5)
    loss_fn = ContrastivePairLoss(margin=0.3)
    losses, stages = [], {}
    bn_before = {
        k: v.clone()
        for k, v in model.state_dict().items()
        if "running_" in k or "num_batches_tracked" in k
    }
    for stage, split in (("train", "train"), ("evaluation", "val")):
        ds = ImagePairDataset(split=split, pairs_file=str(arm), gap_weight=0.0)
        loader = DataLoader(
            ds, batch_size=64, shuffle=stage == "train", generator=torch.Generator().manual_seed(42)
        )
        model.train(stage == "train")
        _apply_batchnorm_policy(model, "frozen_all")
        seen = 0
        for ta, tb, y, w in loader:
            with torch.set_grad_enabled(stage == "train"):
                optimizer.zero_grad()
                loss = loss_fn(model(ta), model(tb), y, weights=w)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite smoke loss")
                if stage == "train":
                    loss.backward()
                    if any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    ):
                        raise ValueError("nonfinite smoke gradient")
                    optimizer.step()
            losses.append(float(loss.detach()))
            seen += 1
            print(f"smoke {stage} batch {seen}/2", flush=True)
            if seen == 2:
                break
        if seen != 2:
            raise ValueError("two full smoke batches per stage required")
        stages[stage] = seen
    if any(not torch.equal(v, model.state_dict()[k]) for k, v in bn_before.items()):
        raise ValueError("BN running buffers changed")
    verified(preflight, "oriented-uniform-training-preflight")
    result = dict(
        stages=stages,
        finite_losses=losses,
        frozen_bn_buffers_unchanged=True,
        smoke_complete=True,
        training_complete=False,
        publication_ready=False,
        scope="bounded actual FaceNet CPU train/eval probe, not long-run stability proof",
    )
    output = out / "summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        out / "summary.manifest.json",
        experiment="oriented-cpu-runtime-smoke",
        parameters={"seed": 42, "batch_size": 64, "train_batches": 2, "eval_batches": 2},
        metrics=result,
        inputs=[*paths_from(native), preflight, Path(__file__)],
        outputs=[output],
    )


def train(preflight, smoke_path, models, out):
    native = verified(preflight, "oriented-uniform-training-preflight")
    probe = verified(smoke_path, "oriented-cpu-runtime-smoke")
    if (
        probe["metrics"].get("smoke_complete") is not True
        or file_record(preflight) not in probe["inputs"]
    ):
        raise ValueError("completed smoke on the same preflight required")
    if models.exists():
        raise FileExistsError("fresh model directory required")
    models.mkdir(parents=True)
    cpu_setup()
    from age_gap.training.finetune import finetune

    completed = []
    for arm in ("low", "cross"):
        pairs = (
            PROJECT_ROOT
            / f"metrics/oriented_exposure_matching_20261003/private/{arm}_candidate_arm.jsonl"
        )
        for seed in (42, 1, 2):
            verified(preflight, "oriented-uniform-training-preflight")
            checkpoint = models / f"{arm}_s{seed}.pt"
            print(f"serial training {arm} seed{seed}", flush=True)
            finetune(
                epochs=10,
                lr=3e-5,
                batch_size=64,
                margin=0.3,
                patience=10,
                trainable_scope="head",
                gap_weight=0.0,
                backbone_name="facenet",
                crops_dir="faces",
                ckpt_out=checkpoint,
                seed=seed,
                pairs_file=str(pairs),
                checkpoint_selection="last_epoch",
                batchnorm_policy="frozen_all",
            )
            manifest = checkpoint.with_suffix(".manifest.json")
            actual = verified(manifest, "pair-contrastive-backbone-finetune")
            training_contract(actual, pairs, seed)
            verified(preflight, "oriented-uniform-training-preflight")
            binding = out / f"{arm}_s{seed}.manifest.json"
            write_experiment_manifest(
                binding,
                experiment="oriented-source-crop-bound-training-cell",
                parameters=PROTOCOL | {"seed": seed, "arm": arm},
                metrics={"training_complete": True, "publication_ready": False},
                inputs=[*paths_from(native), preflight, *paths_from(probe), smoke_path],
                outputs=[checkpoint, manifest],
            )
            completed.extend([checkpoint, manifest, binding])
    verified(preflight, "oriented-uniform-training-preflight")
    write_experiment_manifest(
        out / "training-bound.manifest.json",
        experiment="oriented-uniform-serial-training",
        parameters=PROTOCOL,
        metrics={
            "training_complete": True,
            "completed_cells": 6,
            "evaluation_complete": False,
            "publication_ready": False,
        },
        inputs=[*paths_from(native), preflight, *paths_from(probe), smoke_path],
        outputs=completed,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "smoke", "train"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--preflight", type=Path)
    p.add_argument("--smoke", type=Path)
    p.add_argument("--models", type=Path)
    args = p.parse_args()
    if (
        args.stage != "prepare"
        and args.preflight is None
        or args.stage == "train"
        and (args.smoke is None or args.models is None)
    ):
        p.error("stage prerequisites required")
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    args.out.mkdir(parents=True)
    if args.stage == "prepare":
        prepare(args.out)
    elif args.stage == "smoke":
        smoke(args.preflight, args.out)
    else:
        train(args.preflight, args.smoke, args.models, args.out)


if __name__ == "__main__":
    main()
