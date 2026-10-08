"""Full-pool ROC-v2 FG-NET evaluation of completed native fixed8 AdaFace cells."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs
from scripts.run_oriented_campaign import paths_from, verified
from scripts.run_strong_backbone_cuda_cell_v1 import contract
from scripts.strong_subject_inference_v1 import infer


def readiness(bindings):
    selected, natives = {}, []
    for path in bindings:
        native = verified(path, "adaface-fixed8-native-cuda-training-cell")
        if native["metrics"].get("training_complete") is not True:
            raise ValueError("completed native cell required")
        # Frozen inference must use the same current weights that initialized every cell.
        pretrained = PROJECT_ROOT / "models/adaface_ir101.pt"
        if file_record(pretrained) not in native["inputs"]:
            raise ValueError("shared frozen initialization weights not bound")
        p = native["parameters"]
        if (
            p.get("device") != "cuda"
            or p.get("scope", p.get("trainable_scope")) not in {"head", "tail", "full"}
            or p.get("seed") not in {42, 1, 2}
            or p.get("negative") not in {"random", "lookalike"}
            or p.get("learning_rate") not in {1e-6, 1e-5}
        ):
            raise ValueError("native CUDA cell metadata differs")
        checkpoint = path.parent / "private/checkpoint.pt"
        component = checkpoint.with_suffix(".manifest.json")
        if any(file_record(f) not in native["outputs"] for f in (checkpoint, component)):
            raise ValueError("checkpoint/component not bound")
        actual = verified(component, "pair-contrastive-backbone-finetune")
        pairs = PROJECT_ROOT / (
            "data/processed/pairs.jsonl"
            if p["negative"] == "random"
            else "data/processed/experiments/pairs_lookalike.jsonl"
        )
        contract(
            actual,
            pairs,
            checkpoint,
            scope=p["trainable_scope"],
            lr=p["learning_rate"],
            seed=p["seed"],
        )
        for key, value in actual["parameters"].items():
            if p.get(key) != value:
                raise ValueError("native/component parameter mismatch")
        if any(r not in native["inputs"] for r in actual["inputs"]):
            raise ValueError("native does not bind component inputs")
        key = f"{p['negative']}_{p['trainable_scope']}_lr{p['learning_rate']:.0e}_s{p['seed']}"
        if key in selected:
            raise ValueError("duplicate cell binding")
        selected[key] = checkpoint
        natives.append(native)
    if not selected:
        raise ValueError("at least one completed cell required")
    return selected, natives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh strong evaluation destination required")
    selected, natives = readiness(args.bindings)
    if not args.execute:
        print(f"{len(selected)} completed cells verified; no inference")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    from age_gap.models.backbones import make_backbone
    from age_gap.training.finetune import load_finetuned
    from scripts.reevaluate_fgnet_metrics_v2 import protocol

    torch.set_num_threads(1)
    cache = PROJECT_ROOT / "data/external/fgnet_crops.npz"
    deps = [
        PROJECT_ROOT / f"scripts/{name}.py"
        for name in (
            "evaluate_oriented_cuda_v1",
            "run_oriented_campaign",
            "run_restricted_matched_campaign",
            "run_strong_backbone_cuda_cell_v1",
            "strong_subject_inference_v1",
            "benchmark_metrics_v2",
            "verification_metrics_v2",
            "reevaluate_fgnet_metrics_v2",
        )
    ]
    inputs = list(
        dict.fromkeys(
            [
                Path(__file__),
                cache,
                *deps,
                *args.bindings,
                *(p for native in natives for p in paths_from(native)),
            ]
        )
    )
    before = [file_record(p) for p in inputs]
    with np.load(cache, allow_pickle=False) as data:
        crops, subjects, ages = data["crops"], data["subjects"], data["ages"]
    if crops.shape != (len(subjects), 112, 112, 3) or crops.dtype != np.uint8:
        raise ValueError("full aligned BGR crop cache required")
    left, right, labels, gaps, coverage = protocol(subjects, ages)
    scores, embeddings = {}, {}
    torch.cuda.reset_peak_memory_stats()
    for key in ("frozen", *selected):
        model = (
            make_backbone("adaface_ir101", pretrained=True).cuda().eval()
            if key == "frozen"
            else load_finetuned(selected[key], "cuda").eval()
        )
        rows = []
        with torch.no_grad():
            for start in range(0, len(crops), 16):
                batch = np.stack([model.preprocess(im, True) for im in crops[start : start + 16]])
                rows.append(model(torch.from_numpy(batch).cuda()).cpu().numpy())
        vectors = np.concatenate(rows)
        if (
            vectors.shape != (len(crops), 512)
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        ):
            raise ValueError("complete finite unit embeddings required")
        embeddings[key] = vectors
        scores[key] = np.sum(vectors[left] * vectors[right], axis=1)
        del model
        torch.cuda.empty_cache()
        print(f"full FG-NET pool embedded: {key}", flush=True)
    results = {
        name: infer(
            {k: v[mask] for k, v in scores.items()},
            labels[mask],
            subjects[left][mask],
            subjects[right][mask],
        )
        for name, mask in (
            ("overall", np.ones(len(labels), bool)),
            ("large_gap_25plus", gaps >= 25),
        )
    }
    results.update(
        execution_complete=True,
        publication_ready=False,
        coverage=coverage,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        limitation="actual fixed checkpoints; both strata share FG-NET; completed matrix and representation diagnostics remain separate",
    )
    if before != [file_record(p) for p in inputs]:
        raise RuntimeError("evaluation inputs changed")
    private = args.out / "private"
    private.mkdir(parents=True)
    outputs = []
    for key, vectors in embeddings.items():
        path = private / f"{key}_embeddings.npz"
        np.savez_compressed(path, indices=np.arange(len(crops)), embeddings=vectors)
        outputs.append(path)
    path = private / "scores.npz"
    np.savez_compressed(
        path,
        **scores,
        labels=labels,
        left=left,
        right=right,
        subject_a=subjects[left],
        subject_b=subjects[right],
        source_gap=gaps,
    )
    outputs.append(path)
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = args.out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="adaface-fixed8-cuda-full-fgnet-roc-v2",
        parameters=dict(
            device="cuda",
            protocol_seed=42,
            tolerance=2,
            n_boot=2000,
            bootstrap_seed=0,
            actual_cells=list(selected),
        ),
        metrics=results,
        inputs=inputs,
        outputs=[summary, *outputs],
    )
    validate_written_inputs(target, before)


if __name__ == "__main__":
    main()
