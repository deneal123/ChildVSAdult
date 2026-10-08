"""Full FG-NET CUDA inference only after six verified CUDA training cells."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.evaluate_oriented_controls import SEEDS, paired_infer
from scripts.run_oriented_campaign import PROTOCOL, paths_from, training_contract, verified
from scripts.run_oriented_cuda_campaign_v1 import check_cell


def validate_written_inputs(target, expected):
    """Do not publish a completed manifest for inputs changed during output writes."""
    actual = json.loads(target.read_text(encoding="utf-8"))["inputs"]
    if actual != expected:
        target.unlink()  # Only this invocation's newly created completion manifest.
        raise RuntimeError("inputs changed during manifest publication")


def readiness(campaign):
    binding = campaign / "training-bound.manifest.json"
    native = verified(binding, "oriented-uniform-cuda-serial-training")
    if native["parameters"] != PROTOCOL | dict(device="cuda"):
        raise ValueError("CUDA campaign protocol mismatch")
    if (
        native["metrics"].get("completed_cells") != 6
        or native["metrics"].get("training_complete") is not True
    ):
        raise ValueError("six completed CUDA cells required")
    selected = {}
    for record in native["outputs"]:
        path = PROJECT_ROOT / record["path"]
        cell = verified(path, "oriented-native-cuda-training-cell")
        arm, seed = cell["parameters"]["arm"], cell["parameters"]["seed"]
        if arm not in ("low", "cross") or seed not in SEEDS:
            raise ValueError("unexpected cell")
        check_cell(path, arm, seed)
        key = f"{arm}_s{seed}"
        if key in selected:
            raise ValueError("duplicate cell")
        checkpoints = [
            PROJECT_ROOT / r["path"] for r in cell["outputs"] if r["path"].endswith(".pt")
        ]
        if len(checkpoints) != 1:
            raise ValueError("one bound checkpoint required")
        checkpoint = checkpoints[0]
        component_path = checkpoint.with_suffix(".manifest.json")
        if file_record(component_path) not in cell["outputs"]:
            raise ValueError("component manifest not bound by CUDA cell")
        component = verified(component_path, "pair-contrastive-backbone-finetune")
        pair_path = (
            PROJECT_ROOT
            / f"metrics/oriented_exposure_matching_20261003/private/{arm}_candidate_arm.jsonl"
        )
        training_contract(component, pair_path, seed)
        if file_record(checkpoint) not in component["outputs"]:
            raise ValueError("component does not bind selected checkpoint")
        selected[key] = checkpoints[0]
    if set(selected) != {f"{a}_s{s}" for a in ("low", "cross") for s in SEEDS}:
        raise ValueError("missing CUDA cell")
    return native, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh evaluation output required")
    native, checkpoints = readiness(args.campaign)
    if not args.execute:
        print("six verified CUDA cells; no inference performed")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    from age_gap.models.backbones import make_backbone
    from age_gap.training.finetune import load_finetuned
    from scripts.reevaluate_fgnet_metrics_v2 import protocol

    torch.set_num_threads(1)
    cache = PROJECT_ROOT / "data/external/fgnet_crops.npz"
    dependencies = [
        PROJECT_ROOT / f"scripts/{name}.py"
        for name in (
            "evaluate_oriented_controls",
            "benchmark_metrics_v2",
            "verification_metrics_v2",
            "reevaluate_fgnet_metrics_v2",
            "run_oriented_cuda_campaign_v1",
            "run_oriented_campaign",
        )
    ]
    inputs = list(
        dict.fromkeys(
            [
                args.campaign / "training-bound.manifest.json",
                Path(__file__),
                cache,
                *dependencies,
                *paths_from(native),
            ]
        )
    )
    before = [file_record(path) for path in inputs]
    with np.load(cache, allow_pickle=False) as data:
        crops, subjects, ages = data["crops"], data["subjects"], data["ages"]
    if crops.shape != (len(subjects), 112, 112, 3) or crops.dtype != np.uint8:
        raise ValueError("full aligned BGR crop cache required")
    left, right, labels, gaps, coverage = protocol(subjects, ages)
    scores, embeddings = {}, {}
    for key in ("frozen", *checkpoints):
        model = (
            make_backbone("facenet", pretrained=True).to("cuda").eval()
            if key == "frozen"
            else load_finetuned(checkpoints[key], "cuda").eval()
        )
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
        embeddings[key] = vectors
        scores[key] = np.sum(vectors[left] * vectors[right], axis=1)
        del model
        print(f"CUDA full crop pool embedded: {key}", flush=True)
    low, cross = [
        np.stack([scores[f"{arm}_s{seed}"] for seed in SEEDS]) for arm in ("low", "cross")
    ]
    result = {}
    for name, mask in (("overall", np.ones(len(labels), bool)), ("large_gap_25plus", gaps >= 25)):
        result[name] = paired_infer(
            low[:, mask],
            cross[:, mask],
            labels[mask],
            subjects[left][mask],
            subjects[right][mask],
            frozen=scores["frozen"][mask],
        )
    result.update(
        coverage=coverage,
        execution_complete=True,
        publication_ready=False,
        strata_are_not_independent=True,
        causal_source_evidence=False,
        limitation="within-source age-gap control; cached FG-NET preprocessing; human overlap adjudication pending",
    )
    if before != [file_record(path) for path in inputs]:
        raise RuntimeError("evaluation inputs changed")
    private = args.out / "private"
    private.mkdir(parents=True)
    outputs = []
    for key, vectors in embeddings.items():
        path = private / f"embeddings_{key}.npz"
        np.savez_compressed(path, indices=np.arange(len(crops)), embeddings=vectors)
        outputs.append(path)
    stored = private / "scores.npz"
    np.savez_compressed(
        stored,
        **scores,
        labels=labels,
        source_gap=gaps,
        subject_a=subjects[left],
        subject_b=subjects[right],
        left=left,
        right=right,
    )
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_experiment_manifest(
        args.out / "summary.manifest.json",
        experiment="oriented-cuda-controls-full-fgnet-roc-v2",
        parameters=dict(
            device="cuda", n_boot=2000, bootstrap_seed=0, protocol_seed=42, tolerance=2
        ),
        metrics=result,
        inputs=inputs,
        outputs=[summary, stored, *outputs],
    )
    validate_written_inputs(args.out / "summary.manifest.json", before)


if __name__ == "__main__":
    main()
