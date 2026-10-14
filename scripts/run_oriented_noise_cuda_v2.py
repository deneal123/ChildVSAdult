"""Version2 CUDA noise sensitivity: permutation1/2, training seeds42/1/2, fixed10/last."""

import argparse
import importlib
import json
from pathlib import Path

import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.run_oriented_campaign import PROTOCOL, paths_from, training_contract, verified


def _same_schema(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(_same_schema(actual[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(_same_schema(a, b) for a, b in zip(actual, expected, strict=True))
    return actual == expected


def require_noise(native, pairs, permutation_seed):
    if type(permutation_seed) is not int or permutation_seed not in (1, 2):
        raise ValueError("canonical permutation seed1/2 required")
    if not _same_schema(native["parameters"], PROTOCOL | dict(permutation_seed=permutation_seed)):
        raise ValueError("noise preflight protocol mismatch")
    metrics = native["metrics"]
    if metrics.get("preflight_complete") is not True or type(metrics.get("permutation_seed")) is not int or metrics.get("permutation_seed") != permutation_seed:
        raise ValueError("completed selected-permutation preflight required")
    if metrics.get("training_seeds") != [42, 1, 2] or any(type(s) is not int for s in metrics["training_seeds"]):
        raise ValueError("training seeds42/1/2 required, distinct from permutation seed")
    if file_record(pairs) not in native["inputs"]:
        raise ValueError("noise pair file not bound")
    contract = metrics["clean_noise_contract"]
    if any(
        not _same_schema(contract.get(key), expected)
        for key, expected in {
            "positive_pairs": 600,
            "negative_pairs": 600,
            "all_train_images": 1096,
            "recorded_train_people": 500,
            "fixed_negatives_and_heldout_unchanged": True,
            "endpoint_image_multiplicities_unchanged": True,
        }.items()
    ):
        raise ValueError("clean/noise budget or exposure mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--permutation-seed", required=True, type=int, choices=(1, 2))
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh noise campaign required")
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required, no silent CPU fallback")
    preflight, pairs, permutation_seed = args.preflight, args.pairs, args.permutation_seed
    native = verified(preflight, "oriented-uniform-noise-training-preflight-v2")
    require_noise(native, pairs, permutation_seed)
    clean_binding = (
        PROJECT_ROOT / "metrics/oriented_cuda_campaign_20261004/training-bound.manifest.json"
    )
    clean = verified(clean_binding, "oriented-uniform-cuda-serial-training")
    if (
        not _same_schema(clean["parameters"], PROTOCOL | dict(device="cuda"))
        or clean["metrics"].get("completed_cells") != 6
    ):
        raise ValueError("completed compatible CUDA clean campaign required")
    inputs = list(
        dict.fromkeys(
            [Path(__file__), PROJECT_ROOT / "scripts/run_oriented_noise_cuda_v1.py",
             preflight, clean_binding, *paths_from(native), *paths_from(clean)]
        )
    )
    before = [file_record(path) for path in inputs]
    args.out.mkdir(parents=True)
    trainer = importlib.import_module("age_gap.training.finetune")
    original = trainer.torch_device
    trainer.torch_device = lambda: "cuda"
    outputs = []
    try:
        for seed in (42, 1, 2):
            checkpoint = args.out / "private" / f"noise_s{seed}.pt"
            print(f"CUDA noise permutation{permutation_seed} training seed{seed}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            trainer.finetune(
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
            torch.cuda.synchronize()
            component = checkpoint.with_suffix(".manifest.json")
            actual = verified(component, "pair-contrastive-backbone-finetune")
            training_contract(actual, pairs, seed)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if any(not torch.isfinite(value).all() for value in saved["state_dict"].values()):
                raise ValueError("nonfinite noise checkpoint")
            del saved
            if [file_record(path) for path in inputs] != before:
                raise RuntimeError("noise campaign inputs changed")
            cell = args.out / f"noise_s{seed}.manifest.json"
            write_experiment_manifest(
                cell,
                experiment="oriented-native-cuda-noise-cell-v2",
                parameters=PROTOCOL
                | dict(
                    device="cuda",
                    seed=seed,
                    permutation_seed=permutation_seed,
                    torch_version=torch.__version__,
                    cuda_version=torch.version.cuda,
                    cudnn_version=torch.backends.cudnn.version(),
                ),
                metrics=dict(
                    training_complete=True,
                    publication_ready=False,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                ),
                inputs=inputs,
                outputs=[checkpoint, component],
            )
            outputs.extend([checkpoint, component, cell])
    finally:
        trainer.torch_device = original
    summary = args.out / "summary.json"
    summary.write_text(
        json.dumps(
            dict(
                training_complete=True,
                completed_cells=3,
                evaluation_complete=False,
                publication_ready=False,
                permutation_seed=permutation_seed,
                scope="supervision perturbation, not co-occurrence or causal-source evidence",
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_experiment_manifest(
        args.out / "training-bound.manifest.json",
        experiment="oriented-uniform-cuda-noise-training-v2",
        parameters=PROTOCOL | dict(device="cuda", permutation_seed=permutation_seed),
        metrics=dict(
            training_complete=True,
            completed_cells=3,
            evaluation_complete=False,
            publication_ready=False,
        ),
        inputs=inputs,
        outputs=[summary, *outputs],
    )


if __name__ == "__main__":
    main()


