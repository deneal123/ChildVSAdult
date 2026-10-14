"""Separate telemetry-native FG-NET evaluator (source only; no native access).

Explicit adaptation of scripts/evaluate_strong_cuda_fgnet_v1.py for the separately
versioned telemetry experiment. It does NOT import, reload or monkeypatch the
immutable v1 evaluator's globals: it re-declares distinct telemetry experiment
literals, re-uses the shared pure helpers (verified/paths_from/contract/infer/
validate_written_inputs) and adds the decoded telemetry accounting check to
readiness(). The v1 numerical main() pipeline and CUDA guard are preserved verbatim
in structure.

Binding policy
--------------
readiness() requires that each native train manifest declares at least the minimum
producer source bindings. That is a *declaration over the manifest's own input
records*, NOT a bounded proof that the bound set equals the full literal source
ancestry. main() separately binds the new helper, the full literal local script
ancestry and the whole src tree, before and after the pipeline.

This module is a source draft. Nothing here is native byte-reader acceptance,
native bitwise equivalence or causal mechanism evidence.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs
from scripts.fresh_presentation_byte_guard_v1 import shape
from scripts.run_oriented_campaign import paths_from, verified
from scripts.run_strong_backbone_cuda_cell_v1 import contract
from scripts.run_strong_validation_telemetry_cuda_v1 import local_script_ancestry
from scripts.strong_subject_inference_v1 import infer
from scripts.strong_telemetry_accounting_v1 import (
    required_train_producer_bindings,
    validate_telemetry_record,
)

# Distinct telemetry literals. The v1 native experiment identifier is deliberately
# absent from this module: no telemetry module may consume or relabel a v1 cell.
TELEMETRY_NATIVE_EXPERIMENT = "adaface-fixed8-native-cuda-validation-telemetry-v1"
COMPONENT_EXPERIMENT = "pair-contrastive-backbone-finetune"
TELEMETRY_EXTERNAL_EXPERIMENT = (
    "adaface-fixed8-native-cuda-validation-telemetry-full-fgnet-roc-v3"
)
CELL_SCOPES = {"head", "tail", "full"}
CELL_SEEDS = {42, 1, 2}
CELL_NEGATIVES = {"random", "lookalike"}
CELL_RATES = {1e-6, 1e-5}
WEIGHTS_REL = "models/adaface_ir101.pt"
CROP_CACHE_REL = "data/external/fgnet_crops.npz"
PIPELINE_DEPS = (
    "evaluate_oriented_cuda_v1",
    "run_oriented_campaign",
    "run_restricted_matched_campaign",
    "run_strong_backbone_cuda_cell_v1",
    "strong_subject_inference_v1",
    "benchmark_metrics_v2",
    "verification_metrics_v2",
    "reevaluate_fgnet_metrics_v2",
    "strong_telemetry_accounting_v1",
    "strong_validation_telemetry_v1",
    "run_strong_validation_telemetry_cuda_v1",
)


def declared_producer_sources(native, root):
    """Require the manifest to declare at least the minimum producer bindings.

    Declaration check over the native manifest's own input records only. It is NOT
    a proof that the bound set equals the full literal source ancestry; main() binds
    that separately. Returns the resolved required paths.
    """
    root = root.resolve()
    required = [(root / rel).resolve() for rel in required_train_producer_bindings()]
    bound = {}
    for record in native.get("inputs", []):
        shape(record)
        path = Path(record["path"])
        path = (path if path.is_absolute() else root / path).resolve()
        claim = (record["bytes"], record["sha256"])
        if path in bound and bound[path] != claim:
            raise ValueError("conflicting native source aliases")
        bound[path] = claim
    if any(path not in bound for path in required):
        raise ValueError("telemetry producer source bindings incomplete")
    for path in required:
        current = file_record(path)
        shape(current)
        if (current["bytes"], current["sha256"]) != bound[path]:
            raise ValueError("producer source bytes differ from native declaration")
    return required


def fixed_metadata(params):
    """Exact producer types, not bool/int equality or a shadow scope alias."""
    if (
        type(params.get("seed")) is not int or params["seed"] not in CELL_SEEDS
        or type(params.get("learning_rate")) is not float
        or params["learning_rate"] not in CELL_RATES
        or type(params.get("trainable_scope")) is not str
        or params["trainable_scope"] not in CELL_SCOPES
        or type(params.get("negative")) is not str
        or params["negative"] not in CELL_NEGATIVES
        or params.get("device") != "cuda"
    ):
        raise ValueError("telemetry native CUDA cell metadata differs")
    if "scope" in params and params["scope"] != params["trainable_scope"]:
        raise ValueError("conflicting native scope alias")
    expected = dict(backbone="adaface_ir101", epochs_requested=8, epochs_executed=8,
                    selected_epoch=8, checkpoint_selection="last_epoch",
                    batchnorm_policy="frozen_all", batch_size=16, margin=0.3,
                    gap_weight=0.0, crops_dir="faces")
    if any(type(params.get(k)) is not type(v) or params[k] != v for k, v in expected.items()):
        raise ValueError("typed fixed8 metadata differs")


def readiness(bindings):
    """Telemetry readiness: accounting + unchanged fixed8 component/weight/pair link."""
    selected, natives = {}, []
    for path in bindings:
        native = verified(path, TELEMETRY_NATIVE_EXPERIMENT)
        if type(native.get("schema_version")) is not int or native["schema_version"] != 1:
            raise ValueError("typed native manifest schema required")
        if native["metrics"].get("training_complete") is not True:
            raise ValueError("completed telemetry native cell required")
        # Frozen inference must use the same current weights that initialized every cell.
        pretrained = PROJECT_ROOT / WEIGHTS_REL
        if file_record(pretrained) not in native["inputs"]:
            raise ValueError("shared frozen initialization weights not bound")
        p = native["parameters"]
        fixed_metadata(p)
        declared_producer_sources(native, PROJECT_ROOT)
        # New decoded telemetry accounting surface, checked before any linkage work.
        validate_telemetry_record(native, native["metrics"].get("validation_pairs"))
        checkpoint = path.parent / "private/checkpoint.pt"
        component = checkpoint.with_suffix(".manifest.json")
        if any(file_record(f) not in native["outputs"] for f in (checkpoint, component)):
            raise ValueError("checkpoint/component not bound")
        actual = verified(component, COMPONENT_EXPERIMENT)
        if type(actual.get("schema_version")) is not int or actual["schema_version"] != 1:
            raise ValueError("typed component manifest schema required")
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
            if type(p.get(key)) is not type(value) or p[key] != value:
                raise ValueError("native/component parameter mismatch")
        if any(r not in native["inputs"] for r in actual["inputs"]):
            raise ValueError("native does not bind component inputs")
        key = f"{p['negative']}_{p['trainable_scope']}_lr{p['learning_rate']:.0e}_s{p['seed']}"
        if key in selected:
            raise ValueError("duplicate cell binding")
        selected[key] = checkpoint
        natives.append(native)
    if not selected:
        raise ValueError("at least one completed telemetry cell required")
    return selected, natives


def evaluation_inputs(script, cache, natives, bindings, root, ancestry=local_script_ancestry):
    """New helper + fixed pipeline deps + full literal ancestry + whole src tree."""
    deps = [root / f"scripts/{name}.py" for name in PIPELINE_DEPS]
    return list(
        dict.fromkeys(
            [
                script,
                cache,
                *deps,
                *ancestry(script, root),
                *sorted((root / "src/age_gap").rglob("*.py")),
                *bindings,
                *(p for native in natives for p in paths_from(native)),
            ]
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh telemetry evaluation destination required")
    selected, natives = readiness(args.bindings)
    if not args.execute:
        print(f"{len(selected)} telemetry cells verified; no inference")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    from age_gap.models.backbones import make_backbone
    from age_gap.training.finetune import load_finetuned
    from scripts.reevaluate_fgnet_metrics_v2 import protocol

    torch.set_num_threads(1)
    root = PROJECT_ROOT
    cache = root / CROP_CACHE_REL
    inputs = evaluation_inputs(Path(__file__), cache, natives, args.bindings, root)
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
        limitation="telemetry-native fixed checkpoints; evaluation compatibility is not native bitwise acceptance; both strata share FG-NET; completed matrix and representation diagnostics remain separate",
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
        experiment=TELEMETRY_EXTERNAL_EXPERIMENT,
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


def cli():
    """Keep failures from disclosing private weights/checkpoint paths."""
    try:
        main()
    except Exception:
        print(json.dumps(dict(execution_complete=False,
                              reason="telemetry-evaluator-refusal-no-private-details")))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
