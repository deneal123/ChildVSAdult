"""Fresh CPU-only, source/crop-bound fixed10 LOW/CROSS campaign.

The existing scientific trainer is unchanged. Preparation and all decoded
crop bytes are checked before launching; all declared inputs are checked again
before publishing a completed campaign manifest. Existing outputs are refused,
so an orphan checkpoint cannot silently bypass provenance checks.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from age_gap.common.io import PROJECT_ROOT, read_jsonl, write_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from age_gap.models.facenet import preprocess_bgr
from age_gap.training.finetune import _crop_path


def verify_records(records, root=PROJECT_ROOT):
    if not records:
        raise ValueError("nonempty declared input records required")
    for record in records:
        path = Path(record["path"])
        if not path.is_absolute():
            path = root / path
        try:
            if file_record(path) != record:
                raise ValueError("declared input size/checksum changed")
        except OSError:
            raise ValueError("declared input is unavailable") from None


def audit_crops(arms, *, resolver=_crop_path):
    """No missing-file row dropping; decode/preprocess every unique input image."""
    ids = sorted({r[f"face_{s}"] for rows in arms.values() for r in rows for s in ("a", "b")})
    records = []
    shapes = {}
    for index, face in enumerate(ids):
        path = resolver(face)
        try:
            before = file_record(path)
        except OSError:
            raise ValueError("required crop is missing; no row dropping") from None
        image = cv2.imread(str(path))
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("required crop cannot be decoded")
        tensor = preprocess_bgr(image)
        if tensor.shape != (3, 160, 160) or tensor.dtype != np.float32 or not np.isfinite(tensor).all():
            raise ValueError("invalid FaceNet crop preprocessing")
        if before != file_record(path):
            raise ValueError("crop changed during decode")
        records.append(before)
        shape = f"{image.shape[0]}x{image.shape[1]}"
        shapes[shape] = shapes.get(shape, 0) + 1
        if (index + 1) % 2000 == 0:
            print(f"verified crops {index + 1}/{len(ids)}", flush=True)
    return records, {"unique_crops": len(records), "all_decodable": True,
                     "preprocessing": "BGR -> RGB160 INTER_AREA; (x-127.5)/128; float32 CHW",
                     "source_shape_counts": shapes, "missing_rows_dropped": 0}


def require_fresh(paths):
    for path in paths:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError("fresh empty campaign/model directories required; no checkpoint reuse")


def command(arms, models, out):
    return [sys.executable, "-m", "scripts.train_matched_agegap_arms",
            "--low-arm", str(arms / "private/low_arm.jsonl"),
            "--cross-arm", str(arms / "private/cross_arm.jsonl"),
            "--models-dir", str(models), "--result-dir", str(out),
            "--seeds", "42", "1", "2", "--epochs", "10", "--patience", "10",
            "--checkpoint-selection", "last_epoch", "--batch-size", "64",
            "--learning-rate", "3e-5", "--trainable-scope", "head",
            "--endpoint-age-tolerance", "2", "--bootstrap-resamples", "2000", "--device", "cpu"]


def validate_completed(summary):
    protocol = summary.get("training_protocol", {})
    if (summary.get("complete") is not True or summary.get("completed_run_count") != 6
            or summary.get("expected_run_count") != 6 or summary.get("seeds") != [42, 1, 2]
            or protocol.get("epochs") != 10 or protocol.get("checkpoint_selection") != "last_epoch"):
        raise ValueError("full fixed10 three-seed campaign not completed")
    runs = summary.get("runs", [])
    expected = {(arm, seed) for arm in ("low", "cross") for seed in (42, 1, 2)}
    if (len(runs) != 6 or {(r["arm"], r["seed"]) for r in runs} != expected
            or any(r.get("selected_epoch") != 10 for r in runs)):
        raise ValueError("six distinct arm/seed fixed10 runs required")


def finalize(out, models_dir, *, records, preflight, snapshot, input_paths, parameters):
    """Publish a binding only after verifying scientific output and actual budgets."""
    target = out / "campaign-bound.manifest.json"
    try:
        verify_records(records)
        pre = json.loads(preflight.read_text(encoding="utf-8"))
        verify_records(pre["inputs"] + pre["outputs"])
        completed = out / "summary.json"
        summary = json.loads(completed.read_text(encoding="utf-8"))
        validate_completed(summary)
        training_manifests = sorted(models_dir.glob("*.manifest.json"))
        if len(training_manifests) != 6:
            raise ValueError("six completed training manifests required")
        for row in summary["runs"]:
            path = models_dir / f"{row['run_id']}.manifest.json"
            payload = json.loads(path.read_text())
            params = payload["parameters"]
            if (payload.get("experiment") != "pair-contrastive-backbone-finetune"
                    or params.get("seed") != row["seed"] or params.get("epochs_executed") != 10
                    or params.get("selected_epoch") != 10 or params.get("epochs_requested") != 10
                    or params.get("checkpoint_selection") != "last_epoch"
                    or params.get("batchnorm_policy") != "adapt_all"):
                raise ValueError("actual training budget/BN mismatch")
            verify_records(payload["inputs"] + payload["outputs"])
            result = out / f"{row['run_id']}.manifest.json"
            evaluation = json.loads(result.read_text())
            verify_records(evaluation["inputs"] + evaluation["outputs"])
        summary_manifest = out / "summary.manifest.json"
        evidence = json.loads(summary_manifest.read_text())
        verify_records(evidence["inputs"] + evidence["outputs"])
        outputs = [completed, summary_manifest, *training_manifests, *sorted(models_dir.glob("*.pt")),
                   *[out / f"{r['run_id']}{suffix}" for r in summary["runs"] for suffix in (".json", ".manifest.json")]]
        write_experiment_manifest(target,
            experiment="restricted-matched-source-crop-bound-campaign",
            parameters=parameters | {"seeds": [42, 1, 2], "epochs": 10, "training_completed": True},
            metrics={"training_completed": True, "publication_ready": False,
                     "scope": "restricted image-budget control, not causal longitudinal-source proof"},
            inputs=[*input_paths, preflight, snapshot], outputs=outputs)
        verify_records(records)
        verify_records(pre["inputs"] + pre["outputs"])
    except Exception:
        if target.exists():
            target.unlink()
        (out / "campaign-status.json").write_text(json.dumps({
            "status": "failed_provenance_or_completion_check", "publication_ready": False}) + "\n")
        raise
    (out / "campaign-status.json").write_text(json.dumps({"status": "completed", "publication_ready": False}) + "\n")
    return completed


def run(arms_dir, models_dir, out, *, execute=False, threads=2):
    from facenet_pytorch.models import inception_resnet_v1

    from scripts.build_restricted_matched_arms import restrict_arms
    from scripts.train_matched_agegap_arms import validate_arm_files

    if type(threads) is not int or threads < 1:
        raise ValueError("positive CPU thread count required")
    arms_dir, models_dir, out = map(lambda p: Path(p).resolve(), (arms_dir, models_dir, out))
    require_fresh([models_dir, out])
    preparation = arms_dir / "summary.manifest.json"
    prepared = json.loads(preparation.read_text(encoding="utf-8"))
    if prepared.get("experiment") != "restricted-positive-image-low-cross-arms":
        raise ValueError("unexpected preparation manifest")
    verify_records(prepared["inputs"] + prepared["outputs"])
    seed = prepared["parameters"]["seed"]
    originals = [PROJECT_ROOT / "data/interim/matched_agegap_arms" / f"{name}_arm.jsonl" for name in ("low", "cross")]
    paths = [arms_dir / "private" / f"{name}_arm.jsonl" for name in ("low", "cross")]
    clusters = PROJECT_ROOT / "data/processed/person_clusters.jsonl"
    groups = {}
    for r in read_jsonl(clusters):
        key, value = r["identity_group_id"], r["person_id"]
        if not isinstance(key, str) or not isinstance(value, str) or not key or not value:
            raise ValueError("nonempty string cluster mapping required")
        if key in groups and groups[key] != value:
            raise ValueError("contradictory cluster mapping")
        groups[key] = value
    canonical = PROJECT_ROOT / "data/processed/pairs.jsonl"
    known = {tuple(sorted((r["face_a"], r["face_b"]))) for r in read_jsonl(canonical) if r.get("label") == 1}
    regenerated, _ = restrict_arms({name: list(read_jsonl(p)) for name, p in
                                   zip(("LOW", "CROSS"), originals, strict=True)}, groups,
                                   seed=seed, known_positive_pairs=known)
    loaded = {name: list(read_jsonl(p)) for name, p in zip(("LOW", "CROSS"), paths, strict=True)}
    if loaded != regenerated:
        raise ValueError("prepared arms differ from strict regenerated construction")
    validation = validate_arm_files(*paths, group_to_person=groups)
    inventory = PROJECT_ROOT / "metrics/model_inventory.json"
    artifact = json.loads(inventory.read_text())["models"]["facenet_casia"]["artifact"]
    weight = Path(inception_resnet_v1.get_torch_home()) / "checkpoints/20180408-102900-casia-webface.pt"
    if file_record(weight) != artifact:
        raise ValueError("actual FaceNet cache weight does not match model inventory")
    sources = [Path(__file__), PROJECT_ROOT / "scripts/train_matched_agegap_arms.py",
               PROJECT_ROOT / "scripts/build_restricted_matched_arms.py",
               PROJECT_ROOT / "scripts/build_matched_agegap_arms.py",
               PROJECT_ROOT / "scripts/audit_matched_arm_image_budget.py"]
    for category in ("common", "models", "evaluation", "training", "settings"):
        sources.extend(sorted((PROJECT_ROOT / "src/age_gap" / category).rglob("*.py")))
    sources.extend(sorted((PROJECT_ROOT / "src/age_gap/settings").rglob("*.toml")))
    sources.extend(sorted(Path(inception_resnet_v1.__file__).parent.rglob("*.py")))
    inputs = [preparation, *paths, *originals, clusters, canonical, inventory, weight,
              PROJECT_ROOT / "data/external/fgnet_crops.npz", *sources]
    before = [file_record(p) for p in inputs]
    crop_records, crop_audit = audit_crops(loaded)
    verify_records(before + crop_records)
    verify_records(prepared["inputs"] + prepared["outputs"])
    summary = {"crop_audit": crop_audit, "arm_validation": validation,
               "threads": threads, "device": "cpu", "batchnorm_policy": "adapt_all",
               "checkpoint_reuse": False, "publication_ready": False,
               "training_completed": False, "training_identity_independence": "unverified"}
    print(json.dumps(summary), flush=True)
    if not execute:
        return summary
    require_fresh([models_dir, out])
    out.mkdir(parents=True, exist_ok=True)
    private = out / "private"
    private.mkdir()
    snapshot = private / "preflight_inputs.jsonl"
    write_jsonl(snapshot, before + crop_records)
    audit = out / "preflight.json"
    audit.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    input_paths = [Path(r["path"]) if Path(r["path"]).is_absolute() else PROJECT_ROOT / r["path"]
                   for r in before + crop_records]
    preflight = write_experiment_manifest(out / "preflight.manifest.json",
        experiment="restricted-matched-campaign-preflight", parameters=summary,
        metrics=crop_audit, inputs=input_paths, outputs=[snapshot, audit])
    verify_records(before + crop_records)
    preflight_record = file_record(preflight)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    status = out / "campaign-status.json"
    status.write_text(json.dumps({"status": "running", "publication_ready": False}) + "\n")
    result = subprocess.run(command(arms_dir, models_dir, out), cwd=PROJECT_ROOT, env=env, check=False)
    if result.returncode:
        status.write_text(json.dumps({"status": "failed", "exit_code": result.returncode,
                                      "publication_ready": False}) + "\n")
        raise RuntimeError("restricted campaign subprocess failed; partial artifacts retained")
    return finalize(out, models_dir, records=before + crop_records + [preflight_record],
                    preflight=preflight, snapshot=snapshot, input_paths=input_paths, parameters=summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", type=Path, default=PROJECT_ROOT / "data/interim/restricted_matched_arms_20261003")
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    run(args.arms, args.models, args.out, execute=args.execute, threads=args.threads)


if __name__ == "__main__":
    main()
