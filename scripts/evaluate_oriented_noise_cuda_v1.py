"""Full ROC-v2 noisy-minus-clean CUDA control; completed three-seed inputs only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.evaluate_oriented_controls import SEEDS, paired_infer
from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs
from scripts.run_oriented_campaign import PROTOCOL, paths_from, training_contract, verified


def aligned_clean(data, left, right, labels, gaps, subjects):
    expected = dict(
        left=left,
        right=right,
        labels=labels,
        source_gap=gaps,
        subject_a=subjects[left],
        subject_b=subjects[right],
    )
    if any(not np.array_equal(data[key], value) for key, value in expected.items()):
        raise ValueError("saved clean pair order/metadata differs")
    scores = np.stack([data[f"cross_s{seed}"] for seed in SEEDS])
    frozen = data["frozen"]
    if (
        scores.shape != (3, len(labels))
        or frozen.shape != labels.shape
        or not np.isfinite(scores).all()
        or not np.isfinite(frozen).all()
    ):
        raise ValueError("finite aligned clean/frozen scores required")
    return scores, frozen


def readiness(path):
    native = verified(path, "oriented-uniform-cuda-noise-training")
    if native["parameters"] != PROTOCOL | dict(device="cuda", permutation_seed=42):
        raise ValueError("compatible noise CUDA protocol required")
    if (
        native["metrics"].get("training_complete") is not True
        or native["metrics"].get("completed_cells") != 3
    ):
        raise ValueError("three completed noise seeds required")
    pairs = (
        PROJECT_ROOT
        / "data/interim/oriented_partial_noise_20261003/seed42/private/partial_noise_arm.jsonl"
    )
    selected = {}
    for seed in SEEDS:
        checkpoint = path.parent / "private" / f"noise_s{seed}.pt"
        component_path = checkpoint.with_suffix(".manifest.json")
        cell_path = path.parent / f"noise_s{seed}.manifest.json"
        if any(
            file_record(p) not in native["outputs"] for p in (checkpoint, component_path, cell_path)
        ):
            raise ValueError("noise cell/component/checkpoint not bound")
        cell = verified(cell_path, "oriented-native-cuda-noise-cell")
        if (
            cell["metrics"].get("training_complete") is not True
            or cell["parameters"].get("seed") != seed
            or cell["parameters"].get("device") != "cuda"
            or cell["parameters"].get("permutation_seed") != 42
        ):
            raise ValueError("noise cell protocol mismatch")
        if any(file_record(p) not in cell["outputs"] for p in (checkpoint, component_path)):
            raise ValueError("noise cell linkage mismatch")
        component = verified(component_path, "pair-contrastive-backbone-finetune")
        training_contract(component, pairs, seed)
        if file_record(checkpoint) not in component["outputs"]:
            raise ValueError("component does not bind noise checkpoint")
        selected[seed] = checkpoint
    return native, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh noise evaluation output required")
    binding = args.campaign / "training-bound.manifest.json"
    native, checkpoints = readiness(binding)
    clean_path = PROJECT_ROOT / "metrics/oriented_cuda_fgnet_20261004/summary.manifest.json"
    clean = verified(clean_path, "oriented-cuda-controls-full-fgnet-roc-v2")
    if clean["metrics"].get("execution_complete") is not True:
        raise ValueError("completed clean evaluation required")
    if not args.execute:
        print("three completed noise seeds and clean scores ready; no inference")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    from age_gap.training.finetune import load_finetuned
    from scripts.reevaluate_fgnet_metrics_v2 import protocol

    torch.set_num_threads(1)
    cache = PROJECT_ROOT / "data/external/fgnet_crops.npz"
    deps = [
        PROJECT_ROOT / f"scripts/{name}.py"
        for name in (
            "evaluate_oriented_controls",
            "evaluate_oriented_cuda_v1",
            "benchmark_metrics_v2",
            "verification_metrics_v2",
            "reevaluate_fgnet_metrics_v2",
            "run_oriented_campaign",
        )
    ]
    inputs = list(
        dict.fromkeys(
            [
                Path(__file__),
                binding,
                clean_path,
                cache,
                *deps,
                *paths_from(native),
                *paths_from(clean),
            ]
        )
    )
    before = [file_record(path) for path in inputs]
    with np.load(cache, allow_pickle=False) as data:
        crops, subjects, ages = data["crops"], data["subjects"], data["ages"]
    if crops.shape != (len(subjects), 112, 112, 3) or crops.dtype != np.uint8:
        raise ValueError("full aligned BGR crops required")
    left, right, labels, gaps, coverage = protocol(subjects, ages)
    with np.load(clean_path.parent / "private/scores.npz", allow_pickle=False) as data:
        clean_scores, frozen = aligned_clean(data, left, right, labels, gaps, subjects)
    embeddings, noise = {}, []
    for seed in SEEDS:
        model = load_finetuned(checkpoints[seed], "cuda").eval()
        vectors = []
        with torch.no_grad():
            for start in range(0, len(crops), 32):
                batch = np.stack(
                    [model.preprocess(image, True) for image in crops[start : start + 32]]
                )
                vectors.append(model(torch.from_numpy(batch).to("cuda")).cpu().numpy())
        vectors = np.concatenate(vectors)
        if (
            vectors.shape != (len(crops), 512)
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        ):
            raise ValueError("finite full unit embeddings required")
        embeddings[seed] = vectors
        noise.append(np.sum(vectors[left] * vectors[right], axis=1))
        del model
        print(f"CUDA noise full pool embedded: seed{seed}", flush=True)
    noise = np.stack(noise)
    result = {}
    for name, mask in (("overall", np.ones(len(labels), bool)), ("large_gap_25plus", gaps >= 25)):
        result[name] = paired_infer(
            clean_scores[:, mask],
            noise[:, mask],
            labels[mask],
            subjects[left][mask],
            subjects[right][mask],
            frozen=frozen[mask],
        )
    result.update(
        execution_complete=True,
        publication_ready=False,
        coverage=coverage,
        permutation_seed=42,
        low_role="clean CROSS",
        cross_role="noisy CROSS",
        direction="noise_minus_clean",
        limitation="fixed permutation/checkpoints; not empirical identity-error rate, co-occurrence or causal-source evidence",
    )
    if before != [file_record(path) for path in inputs]:
        raise RuntimeError("noise evaluation input changed")
    private = args.out / "private"
    private.mkdir(parents=True)
    outputs = []
    for seed, vectors in embeddings.items():
        path = private / f"noise_embeddings_s{seed}.npz"
        np.savez_compressed(path, indices=np.arange(len(crops)), embeddings=vectors)
        outputs.append(path)
    scores = private / "scores.npz"
    np.savez_compressed(
        scores,
        clean=clean_scores,
        noise=noise,
        frozen=frozen,
        labels=labels,
        subject_a=subjects[left],
        subject_b=subjects[right],
        left=left,
        right=right,
        source_gap=gaps,
    )
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = args.out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="oriented-cuda-noise-full-fgnet-roc-v2",
        parameters=dict(
            device="cuda",
            permutation_seed=42,
            training_seeds=list(SEEDS),
            bootstrap_seed=0,
            n_boot=2000,
            protocol_seed=42,
            tolerance=2,
        ),
        metrics=result,
        inputs=inputs,
        outputs=[summary, scores, *outputs],
    )
    validate_written_inputs(target, before)


if __name__ == "__main__":
    main()
