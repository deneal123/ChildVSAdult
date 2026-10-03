"""Pre-score exact-age subset sensitivity of the saved internal matched protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.audit_age_label_balance import paired_rows


def select_blocks(rows, group_people):
    paired_rows(rows, explicit_targets=False)
    half = len(rows) // 2
    positives, negatives = rows[:half], rows[half:]
    exact, owners, impostors = [], [], []
    for p, n in zip(positives, negatives, strict=True):
        if p.get("split") != "test" or n.get("split") != "test" or p["age_gap"] < 25:
            raise ValueError("saved 25+ test observations required")
        error = abs(p["age_b"] - n["age_b"])
        if error > 2:
            raise ValueError("original endpoint tolerance exceeded")
        groups = [p["identity_group_a"], p["identity_group_b"], n["identity_group_a"], n["identity_group_b"]]
        if any(g not in group_people or not isinstance(group_people[g], str) or not group_people[g].strip() for g in groups):
            raise ValueError("nonmissing recorded person mapping required")
        a, b, c, d = [group_people[g] for g in groups]
        if a != b or a != c or a == d:
            raise ValueError("recorded-person labels contradict matched block")
        owners.append(a)
        impostors.append(d)
        exact.append(error == 0)
    mask = np.asarray(exact, bool)
    if not mask.any():
        raise ValueError("no exact-age matched blocks")
    return mask, np.asarray(owners), np.asarray(impostors)


def auc_gain(frozen, tuned, weights):
    labels = np.r_[np.ones(len(weights), int), np.zeros(len(weights), int)]
    doubled = np.r_[weights, weights]
    frozen_auc = float(roc_auc_score(labels, frozen, sample_weight=doubled))
    tuned_auc = float(roc_auc_score(labels, tuned, sample_weight=doubled))
    return [frozen_auc, tuned_auc, tuned_auc - frozen_auc]


def compare_subsets(frozen, tuned, exact, owners, impostors, *, n_boot=2000, seed=42):
    frozen, tuned = np.asarray(frozen, float), np.asarray(tuned, float)
    exact, owners, impostors = np.asarray(exact), np.asarray(owners), np.asarray(impostors)
    n = len(exact)
    if not n or exact.dtype.kind != "b" or not exact.any() or frozen.shape != (2 * n,) or tuned.shape != frozen.shape or owners.shape != (n,) or impostors.shape != (n,):
        raise ValueError("aligned nonempty matched vectors and exact mask required")
    if not np.isfinite(frozen).all() or not np.isfinite(tuned).all() or (owners == impostors).any():
        raise ValueError("finite scores and distinct block endpoints required")
    if any(ids.dtype.kind not in "US" or any(not str(v).strip() for v in ids) for ids in (owners, impostors)):
        raise ValueError("nonmissing recorded-person string identifiers required")
    if type(n_boot) is not int or n_boot < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive resample count and nonnegative seed required")
    subjects, inverse = np.unique(np.r_[owners, impostors], return_inverse=True)
    a, b = inverse[:n], inverse[n:]
    ones = np.ones(n)
    full, strict = auc_gain(frozen, tuned, ones), auc_gain(frozen, tuned, exact.astype(float))
    rng, draws = np.random.default_rng(seed), []
    for _ in range(n_boot):
        m = np.bincount(rng.integers(len(subjects), size=len(subjects)), minlength=len(subjects))
        weights = m[a] * m[b]
        if not (weights * exact).any():
            continue
        f, s = auc_gain(frozen, tuned, weights), auc_gain(frozen, tuned, weights * exact)
        draws.append([*f, *s, s[2] - f[2]])
    if not draws:
        raise ValueError("no valid exact/full joint subject resamples")
    intervals = np.percentile(np.asarray(draws), [2.5, 97.5], axis=0).T.tolist()
    results = {}
    labels = np.r_[np.ones(n, int), np.zeros(n, int)]
    for tag, mask, point, offset in (("original_tolerance", np.ones(n, bool), full, 0), ("exact_age_subset", exact, strict, 3)):
        selected = np.r_[mask, mask]
        lowfar = {}
        for name, scores in (("frozen", frozen), ("tuned", tuned)):
            far, tar, _ = roc_curve(labels[selected], scores[selected], drop_intermediate=False)
            lowfar[name] = {str(target): float(tar[far <= target].max()) for target in (.01, .001)}
        results[tag] = {"positive_pairs": int(mask.sum()), "negative_pairs": int(mask.sum()),
            "recorded_people": len(np.unique(np.r_[owners[mask], impostors[mask]])),
            "frozen_auc": point[0], "tuned_auc": point[1], "gain_auc": point[2],
            "ci95_frozen_tuned_gain": intervals[offset:offset + 3], "empirical_tar_at_far": lowfar}
    results["exact_minus_original_gain"] = {"point": strict[2] - full[2], "ci95": intervals[6]}
    results["bootstrap"] = {"seed": seed, "requested": n_boot, "valid": len(draws),
        "recorded_people": len(subjects), "method": "joint recorded-person dyad-multiplicity product; same weight for positive and negative matched block",
        "conditioning": "one fixed seed42 checkpoint and selected saved pairs; not training-seed uncertainty or causal age effect"}
    results["lowfar_scope"] = "descriptive empirical ROC operating points, no CI or deployable calibrated threshold; accepts tied scores together"
    results["identity_independence"] = "recorded person clusters only; missed-merge human audit pending"
    results["publication_ready"] = False
    return results


def load_local_model(path, *, frozen):
    import torch

    from age_gap.models.backbones import make_backbone
    data = torch.load(path, map_location="cpu", weights_only=frozen)
    model = make_backbone("facenet", pretrained=False).to("cpu")
    if frozen:
        mismatch = model.net.load_state_dict(data, strict=False)
        allowed = {"logits.weight", "logits.bias"}
    else:
        if data.get("backbone", "facenet") != "facenet" or data.get("seed", 42) != 42 or data.get("crops_dir", "faces") != "faces":
            raise ValueError("local seed42 FaceNet faces checkpoint required")
        mismatch = model.load_state_dict(data["state_dict"], strict=False)
        allowed = {"net.logits.weight", "net.logits.bias"}
    if mismatch.missing_keys or set(mismatch.unexpected_keys) - allowed:
        raise ValueError("embedding parameters not completely loaded")
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-weights", type=Path, default=Path.home() / ".cache/torch/checkpoints/20180408-102900-casia-webface.pt")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models/bb_facenet_seed42.pt")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists() or args.threads < 1:
        raise ValueError("fresh output directory and positive CPU thread count required")
    root = PROJECT_ROOT
    pair_path = root / "data/processed/experiments/pairs_internal_endpoint_age_matched.jsonl"
    original_manifest = root / "metrics/internal_endpoint_age_matched.manifest.json"
    reference = json.loads(original_manifest.read_text(encoding="utf-8"))
    if file_record(pair_path) not in reference["outputs"] or file_record(args.checkpoint) not in reference["inputs"]:
        raise ValueError("original pair/checkpoint binding mismatch")
    rows = list(read_jsonl(pair_path))
    people_path = root / "data/processed/person_clusters.jsonl"
    group_people = {}
    for row in read_jsonl(people_path):
        group, person = row["identity_group_id"], row["person_id"]
        if group in group_people and group_people[group] != person:
            raise ValueError("contradictory recorded-person mapping")
        group_people[group] = person
    mask, owners, impostors = select_blocks(rows, group_people)
    faces = sorted({r[f"face_{side}"] for r in rows for side in ("a", "b")})
    crops = [root / "data/interim/faces" / f"{face}.jpg" for face in faces]
    if any(file_record(path) not in reference["inputs"] for path in crops):
        raise ValueError("original crop binding mismatch")
    source_paths = [Path(__file__), root / "scripts/audit_age_label_balance.py",
                    *sorted((root / "src/age_gap").rglob("*.py")),
                    *sorted((root / "src/age_gap").rglob("*.toml"))]
    inputs = [pair_path, original_manifest, people_path, args.base_weights, args.checkpoint, *crops, *source_paths]
    before = [file_record(path) for path in inputs]
    plan = {"rule": "retain original whole matched blocks iff both endpoint ages exactly equal; no re-mining or score-based selection",
            "original_blocks": len(mask), "exact_blocks": int(mask.sum()), "retained_fraction": float(mask.mean()),
            "original_recorded_people": len(np.unique(np.r_[owners, impostors])),
            "exact_recorded_people": len(np.unique(np.r_[owners[mask], impostors[mask]])),
            "bootstrap_seed": 42, "n_boot": 2000, "training_seed": 42, "device": "cpu", "threads": args.threads,
            "selection_based_on_scores": False, "public_preregistration": False, "publication_ready": False,
            "limits": "subset changes age/identity composition; does not test causal shortcut use or exact rematching coverage; legacy training provenance unverified"}
    args.out.mkdir(parents=True)
    plan_path = args.out / "protocol.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(args.out / "protocol.manifest.json", experiment="internal-exact-age-sensitivity-protocol",
                              parameters=plan, metrics=plan, inputs=inputs, outputs=[plan_path])
    if before != [file_record(path) for path in inputs]:
        (args.out / "protocol.manifest.json").unlink()
        raise ValueError("protocol inputs changed")
    print(json.dumps(plan), flush=True)
    if not args.execute:
        return
    import cv2
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    score_arrays = []
    face_indices = {face: i for i, face in enumerate(faces)}
    left = [face_indices[r["face_a"]] for r in rows]
    right = [face_indices[r["face_b"]] for r in rows]
    for tag, weight, frozen in (("frozen", args.base_weights, True), ("tuned", args.checkpoint, False)):
        model, vectors = load_local_model(weight, frozen=frozen), []
        with torch.inference_mode():
            for start in range(0, len(crops), 16):
                images = [cv2.imread(str(path)) for path in crops[start:start + 16]]
                if any(image is None for image in images):
                    raise ValueError("bound crop cannot be decoded")
                batch = torch.from_numpy(np.stack([model.preprocess(image, bgr=True) for image in images]))
                output = model(batch).cpu().numpy()
                if output.ndim != 2 or len(output) != len(images) or not np.isfinite(output).all():
                    raise ValueError("invalid embedding batch")
                vectors.append(output)
                print(f"{tag}: embedded {min(start + 16, len(crops))}/{len(crops)}", flush=True)
        embeddings = np.concatenate(vectors)
        score_arrays.append(np.sum(embeddings[left] * embeddings[right], axis=1))
        del model, embeddings, vectors
    result = compare_subsets(*score_arrays, mask, owners, impostors)
    result["protocol"] = plan
    result["historical_point_reference"] = reference["metrics"]["model_comparison_same_matched_subset"]
    if before != [file_record(path) for path in inputs]:
        raise ValueError("evaluation inputs changed")
    private = args.out / "private"
    private.mkdir()
    scores_path = private / "scores.npz"
    np.savez_compressed(scores_path, frozen=score_arrays[0], tuned=score_arrays[1], exact=mask)
    output = args.out / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = args.out / "summary.manifest.json"
    write_experiment_manifest(target, experiment="internal-exact-age-sensitivity", parameters=plan,
                              metrics=result, inputs=[*inputs, plan_path, args.out / "protocol.manifest.json"], outputs=[output, scores_path])
    if json.loads(target.read_text(encoding="utf-8"))["inputs"][:len(before)] != before:
        target.unlink()
        raise ValueError("manifest inputs changed; completed marker withdrawn")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
