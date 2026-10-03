"""Fresh CPU CACD scoring with mapped crops and one model process at a time."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest

KEYS = ("frozen", "tuned_seed42", "tuned_seed1", "tuned_seed2")
PREPROCESSING = {
    "function": "age_gap.models.facenet.preprocess_bgr",
    "input_size": 160,
    "channel_order": "RGB",
    "normalization": "(x - 127.5) / 128",
    "source_crop_format": "BGR uint8 112x112",
    "shared_between_frozen_and_tuned": True,
}
MEMBERS = {"a.npy", "b.npy", "issame.npy", "fold_ids.npy", "miss_rate.npy"}


def records_match(records, root=PROJECT_ROOT):
    if not records:
        raise ValueError("nonempty native records required")
    for record in records:
        path = Path(record["path"])
        if file_record(path if path.is_absolute() else root / path) != record:
            raise ValueError("native file checksum/size changed")


def verify(path, experiment, root=PROJECT_ROOT):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["experiment"] != experiment:
        raise ValueError("unexpected native experiment")
    records_match(payload["inputs"] + payload["outputs"], root)
    return payload


def write_bound(path, experiment, result, inputs, outputs, *, expected_inputs=None):
    before = [file_record(p) for p in inputs]
    if expected_inputs is not None and before != expected_inputs:
        raise ValueError("inputs changed before manifest write")
    write_experiment_manifest(
        path,
        experiment=experiment,
        parameters={"device": "cpu", "threads": 1, "batch_size": 8, "publication_ready": False},
        metrics=result,
        inputs=inputs,
        outputs=outputs,
    )
    if json.loads(path.read_text(encoding="utf-8"))["inputs"] != before:
        path.unlink()
        raise ValueError("inputs changed during manifest write; marker withdrawn")


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_plan(out):
    manifest = verify(out / "plan.manifest.json", "cacd-vs-serial-plan")
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    if (
        manifest["metrics"] != plan
        or plan["roles"] != list(KEYS)
        or plan["preprocessing"] != PREPROCESSING
        or plan["batch_size"] != 8
        or plan["threads"] != 1
        or plan["device"] != "cpu"
        or plan["seed"] != 0
        or plan["n_boot"] != 2000
        or plan["model_processes_simultaneous"] != 1
    ):
        raise ValueError("plan/native metrics or roles/preprocessing mismatch")
    records = plan["checkpoint_records"]
    if (
        set(records) != set(KEYS)
        or len({record["sha256"] for record in records.values()}) != 4
        or any(record not in manifest["inputs"] for record in records.values())
        or plan["source_record"] not in manifest["inputs"]
    ):
        raise ValueError("plan source/checkpoint records not bound to native inputs")
    return plan, manifest


def execution_plan(out):
    plan, manifest = read_plan(out)
    if plan["image_inference_authorized_by_execute"] is not True:
        raise ValueError("input-only preflight cannot be reused as an execution plan")
    return plan, manifest


def completed_phase(path, experiment, result_path, outputs):
    native = verify(path, experiment)
    if (
        native["metrics"]["execution_complete"] is not True
        or json.loads(result_path.read_text(encoding="utf-8")) != native["metrics"]
        or native["outputs"] != [file_record(p) for p in [result_path, *outputs]]
    ):
        raise ValueError("completed phase output/result binding mismatch")
    return native


def read_prepared(out):
    native = completed_phase(
        out / "prepared.manifest.json",
        "cacd-vs-mapped-crops",
        out / "prepared.json",
        [out / "private/crops" / name for name in sorted(MEMBERS)],
    )
    if (
        native["metrics"]["n_pairs"] != 4000
        or native["metrics"]["n_crop_rows"] != 8000
        or native["metrics"]["side_order"] != "all a then all b"
        or file_record(out / "plan.manifest.json") not in native["inputs"]
    ):
        raise ValueError("prepared dimensions/order/plan binding mismatch")
    return native


def extract_mapped(source, private):
    """Exact allowlist, no full-array decompression in RAM or path-based extraction."""
    private.mkdir()
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != MEMBERS:
            raise ValueError("unexpected/duplicate aligned archive members")
        for name in sorted(MEMBERS):
            with archive.open(name) as src, (private / name).open("xb") as dst:
                shutil.copyfileobj(src, dst, length=128 * 1024)


def phase_prepare(out):
    import numpy as np

    from scripts.reevaluate_cacd_metrics_v2 import canonical

    plan, native = execution_plan(out)
    inputs = [
        Path(r["path"]) if Path(r["path"]).is_absolute() else PROJECT_ROOT / r["path"]
        for r in native["inputs"]
    ] + [out / "plan.json", out / "plan.manifest.json"]
    before = [file_record(p) for p in inputs]
    private = out / "private/crops"
    source = PROJECT_ROOT / "data/external/cacd_vs_aligned.npz"
    extract_mapped(source, private)
    for side in ("a", "b"):
        array = np.load(private / f"{side}.npy", mmap_mode="r", allow_pickle=False)
        if (
            array.shape != (4000, 112, 112, 3)
            or array.dtype != np.uint8
            or not array.flags.c_contiguous
        ):
            raise ValueError("canonical C-order BGR uint8 crops required")
    labels = np.load(private / "issame.npy", allow_pickle=False)
    folds = np.load(private / "fold_ids.npy", allow_pickle=False)
    canonical(labels, folds)
    fallback = float(np.load(private / "miss_rate.npy", allow_pickle=False))
    if not np.isfinite(fallback) or not 0 <= fallback <= 0.05:
        raise ValueError("invalid alignment fallback fraction")
    result = {
        "n_pairs": 4000,
        "n_crop_rows": 8000,
        "side_order": "all a then all b",
        "alignment_fallback_fraction": fallback,
        "execution_complete": True,
        "publication_ready": False,
    }
    output = out / "prepared.json"
    if before != [file_record(p) for p in inputs]:
        raise ValueError("source/plan changed during mapped extraction")
    atomic_json(output, result)
    write_bound(
        out / "prepared.manifest.json",
        "cacd-vs-mapped-crops",
        result,
        inputs,
        [output, *(private / name for name in sorted(MEMBERS))],
        expected_inputs=before,
    )


def stream_embeddings(a, b, embed_batch, destination, *, batch_size=8, dimension=512):
    import numpy as np

    if a.shape != b.shape or len(a) < 1 or batch_size < 1 or dimension < 1 or destination.exists():
        raise ValueError(
            "matching nonempty sides, positive budget and fresh embedding file required"
        )
    target = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float32, shape=(2 * len(a), dimension)
    )
    for side_index, side in enumerate((a, b)):
        for start in range(0, len(side), batch_size):
            stop = min(start + batch_size, len(side))
            vectors = np.asarray(embed_batch(side[start:stop]), dtype=np.float32)
            if (
                vectors.shape != (stop - start, dimension)
                or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
            ):
                raise ValueError("finite normalized complete embedding batch required")
            target[side_index * len(a) + start : side_index * len(a) + stop] = vectors
            if stop % 400 == 0 or stop == len(side):
                print(f"crop rows {side_index * len(a) + stop}/{2 * len(a)}", flush=True)
    target.flush()
    del target


def phase_score(out, key):
    import numpy as np
    import torch

    from age_gap.models.facenet import preprocess_bgr
    from scripts.fgnet_retrieval_study import (
        PREPROCESSING as SHARED_PREPROCESSING,
    )
    from scripts.fgnet_retrieval_study import (
        configure_torch_threads,
        load_frozen_model,
        load_tuned_model,
    )

    plan, _ = execution_plan(out)
    read_prepared(out)
    if key not in KEYS or SHARED_PREPROCESSING != PREPROCESSING:
        raise ValueError("unknown role or changed shared preprocessing")
    inputs = [
        out / "plan.json",
        out / "plan.manifest.json",
        out / "prepared.manifest.json",
        *(out / "private/crops" / name for name in sorted(MEMBERS)),
    ]
    before = [file_record(p) for p in inputs]
    weight = Path(plan["checkpoint_records"][key]["path"])
    weight = weight if weight.is_absolute() else PROJECT_ROOT / weight
    configure_torch_threads(1)
    model = load_frozen_model(weight) if key == "frozen" else load_tuned_model(weight)
    a, b = (
        np.load(out / "private/crops" / f"{side}.npy", mmap_mode="r", allow_pickle=False)
        for side in ("a", "b")
    )

    def embed(batch):
        tensor = np.stack([preprocess_bgr(crop) for crop in batch])
        with torch.inference_mode():
            return model(torch.from_numpy(tensor)).detach().cpu().numpy()

    destination = out / "private" / f"embeddings_{key}.npy"
    stream_embeddings(a, b, embed, destination)
    read_plan(out)
    if before != [file_record(p) for p in inputs]:
        raise ValueError("mapped crops/plan changed during scoring")
    result = {
        "role": key,
        "checkpoint_record": plan["checkpoint_records"][key],
        "n_crop_rows": 8000,
        "embedding_dimension": 512,
        "side_order": "all a then all b",
        "execution_complete": True,
        "publication_ready": False,
    }
    output = out / f"{key}.json"
    atomic_json(output, result)
    write_bound(
        out / f"{key}.manifest.json",
        "cacd-vs-serial-checkpoint",
        result,
        inputs,
        [output, destination],
        expected_inputs=before,
    )


def paired_scores(vectors, *, n_pairs=4000, dimension=512, chunk_size=128):
    """Preserve float32 cosine summation and avoid crossing the A/B boundary."""
    import numpy as np

    if (
        vectors.shape != (2 * n_pairs, dimension)
        or vectors.dtype != np.float32
        or n_pairs < 1
        or chunk_size < 1
    ):
        raise ValueError("unexpected indexed embedding shape/dtype or budget")
    scores = np.empty(n_pairs, dtype=np.float32)
    for start in range(0, n_pairs, chunk_size):
        stop = min(start + chunk_size, n_pairs)
        scores[start:stop] = np.sum(
            vectors[start:stop] * vectors[n_pairs + start : n_pairs + stop], axis=1
        )
    return scores


def phase_aggregate(out):
    import numpy as np

    from scripts.cacd_metrics_v2 import infer
    from scripts.reevaluate_cacd_metrics_v2 import canonical

    plan, _ = execution_plan(out)
    prepared = read_prepared(out)
    output, score_path = out / "summary.json", out / "private/scores.npz"
    if any(p.exists() for p in (output, score_path, out / "summary.manifest.json")):
        raise FileExistsError("fresh aggregation outputs required")
    inputs = [
        out / "plan.json",
        out / "plan.manifest.json",
        out / "prepared.manifest.json",
        *(out / f"{key}.manifest.json" for key in KEYS),
        *(out / "private" / f"embeddings_{key}.npy" for key in KEYS),
        *(out / "private/crops" / name for name in sorted(MEMBERS)),
    ]
    before = [file_record(p) for p in inputs]
    labels, folds = (
        np.load(out / "private/crops" / f"{name}.npy", allow_pickle=False)
        for name in ("issame", "fold_ids")
    )
    canonical(labels, folds)
    scores = {}
    for key in KEYS:
        path = out / "private" / f"embeddings_{key}.npy"
        manifest = completed_phase(
            out / f"{key}.manifest.json", "cacd-vs-serial-checkpoint", out / f"{key}.json", [path]
        )
        row = manifest["metrics"]
        if (
            row["role"] != key
            or row["checkpoint_record"] != plan["checkpoint_records"][key]
            or row["n_crop_rows"] != 8000
            or row["embedding_dimension"] != 512
            or row["side_order"] != "all a then all b"
            or row["execution_complete"] is not True
            or file_record(out / "plan.manifest.json") not in manifest["inputs"]
            or file_record(out / "prepared.manifest.json") not in manifest["inputs"]
        ):
            raise ValueError("checkpoint completed-role binding mismatch")
        vectors = np.load(path, mmap_mode="r", allow_pickle=False)
        scores[key] = paired_scores(vectors)
        del vectors
    result = {
        "inference": infer(scores, labels, folds, n_boot=plan["n_boot"], seed=plan["seed"]),
        "plan": plan,
        "alignment_fallback_fraction": prepared["metrics"]["alignment_fallback_fraction"],
        "execution_complete": True,
        "legacy_results_rewritten": False,
        "publication_ready": False,
    }
    read_plan(out)
    if before != [file_record(p) for p in inputs]:
        raise ValueError("aggregation inputs changed")
    np.savez_compressed(score_path, **scores, labels=labels, folds=folds)
    atomic_json(output, result)
    write_bound(
        out / "summary.manifest.json",
        "cacd-vs-local-serial-4checkpoint-roc-v2",
        result,
        inputs,
        [output, score_path],
        expected_inputs=before,
    )


def child_command(out, phase, role=None):
    command = [
        sys.executable,
        "-m",
        "scripts.reevaluate_cacd_serial",
        "--out",
        str(out),
        "--execute",
        "--phase",
        phase,
    ]
    return command + (["--role", role] if role is not None else [])


def controller(out, execute):
    if out.exists():
        raise FileExistsError("fresh destination required; partial/legacy results never reused")
    root = PROJECT_ROOT
    source = root / "data/external/cacd_vs_aligned.npz"
    source_manifest = source.with_suffix(".manifest.json")
    native = verify(source_manifest, "cacd-vs-alignment-cache")
    if file_record(source) not in native["outputs"]:
        raise ValueError("source is not a declared aligned output")
    weights = {
        "frozen": Path.home() / ".cache/torch/checkpoints/20180408-102900-casia-webface.pt",
        "tuned_seed42": root / "models/bb_facenet_seed42.pt",
        "tuned_seed1": root / "models/bb_facenet_seed1.pt",
        "tuned_seed2": root / "models/bb_facenet_seed2.pt",
    }
    weight_records = {key: file_record(weights[key]) for key in KEYS}
    if len({r["sha256"] for r in weight_records.values()}) != 4:
        raise ValueError("distinct four checkpoint files required")
    paths = [
        source,
        source_manifest,
        *(
            Path(record["path"]) if Path(record["path"]).is_absolute() else root / record["path"]
            for record in native["inputs"]
        ),
        *weights.values(),
        Path(__file__),
        root / "uv.lock",
        root / "pyproject.toml",
        *(
            root / "scripts" / name
            for name in (
                "reevaluate_cacd_metrics_v2.py",
                "cacd_metrics_v2.py",
                "benchmark_metrics_v2.py",
                "verification_metrics_v2.py",
                "evaluate_lfw_bound.py",
                "fgnet_retrieval_study.py",
                "export_lfw_evidence.py",
            )
        ),
        *sorted((root / "src/age_gap").rglob("*.py")),
        *sorted((root / "src/age_gap").rglob("*.toml")),
    ]
    before = [file_record(p) for p in paths]
    plan = {
        "roles": list(KEYS),
        "checkpoint_records": weight_records,
        "source_record": before[0],
        "preprocessing": PREPROCESSING,
        "batch_size": 8,
        "threads": 1,
        "device": "cpu",
        "seed": 0,
        "n_boot": 2000,
        "model_processes_simultaneous": 1,
        "image_inference_authorized_by_execute": execute,
        "canonical_array_validation": "performed by mapped preparation phase before any model inference",
        "execution_complete": False,
        "subject_metadata_available": False,
        "scope": "fresh local scoring and pair CI; original alignment replay and person independence unverified",
        "publication_ready": False,
    }
    out.mkdir(parents=True)
    atomic_json(out / "plan.json", plan)
    write_bound(
        out / "plan.manifest.json",
        "cacd-vs-serial-plan",
        plan,
        paths,
        [out / "plan.json"],
        expected_inputs=before,
    )
    if before != json.loads((out / "plan.manifest.json").read_text(encoding="utf-8"))["inputs"]:
        (out / "plan.manifest.json").unlink()
        raise ValueError("inputs changed during plan creation")
    read_plan(out)
    if not execute:
        print(
            "Bound input-only preflight; array validation and image inference not performed",
            flush=True,
        )
        return
    logs = out / "private/logs"
    logs.mkdir(parents=True)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    for phase, role in [("prepare", None), *(("score", key) for key in KEYS), ("aggregate", None)]:
        label = role or phase
        print(f"CACD serial phase {label} started; log private/logs/{label}.log", flush=True)
        with (logs / f"{label}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                child_command(out, phase, role),
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            failed = {
                "phase": phase,
                "role": role,
                "exit_code": completed.returncode,
                "execution_complete": False,
                "publication_ready": False,
                "cause": "inspect private phase log",
            }
            atomic_json(out / "FAILED.json", failed)
            write_bound(
                out / "FAILED.manifest.json",
                "cacd-vs-serial-incomplete",
                failed,
                [out / "plan.json", out / "plan.manifest.json", logs / f"{label}.log"],
                [out / "FAILED.json"],
            )
            raise RuntimeError(
                f"CACD phase {label} exited {completed.returncode}; no completed campaign claim"
            )
        print(f"CACD serial phase {label} completed", flush=True)
    read_plan(out)
    verify(out / "summary.manifest.json", "cacd-vs-local-serial-4checkpoint-roc-v2")
    print(
        "CACD serial complete; four checkpoints/scores/paired pair CI bound; publication_ready=false",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--phase", choices=("prepare", "score", "aggregate"))
    parser.add_argument("--role", choices=KEYS)
    args = parser.parse_args()
    if args.phase is None:
        controller(args.out.resolve(), args.execute)
    else:
        if not args.execute or (args.phase == "score") != (args.role is not None):
            parser.error("internal phases need --execute and only score needs --role")
        {
            "prepare": phase_prepare,
            "score": lambda out: phase_score(out, args.role),
            "aggregate": phase_aggregate,
        }[args.phase](args.out.resolve())


if __name__ == "__main__":
    main()
