"""Explicit six-member LFW aggregate export, without altering bound evaluator code."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, sha256_file
from scripts.export_public_evidence import scan_unsafe
from scripts.render_lfw_evidence import METRICS, render

DIRECTORY = "metrics/lfw_bound_evaluation_20261002"
MODEL_KEYS = {"frozen", "tuned_seed42", "tuned_seed1", "tuned_seed2"}
METRIC_KEYS = {key for key, _ in METRICS}
MEMBERS = {"results/lfw.json", "tables/lfw.tex", "manifests/evaluation.sanitized.json",
           "manifests/presentation.sanitized.json", "README.txt", "CERTIFICATE.json"}
ZIP_NAME = "lfw_evidence_bundle.zip"
TOP_KEYS = {"accuracy_acceptance", "bootstrap", "cache_full_transform_replay", "delta_definition",
            "detection_sensitivity", "legacy_accuracy_protocol", "models", "n_negative", "n_pairs",
            "n_positive", "n_subjects", "n_unique_crops_embedded", "negative_events_at_nominal_far",
            "package_versions", "protocol", "provenance", "publication_ready", "roc_scope",
            "seed_aggregate", "seeds", "threshold_grid", "training_identity_independence"}


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("unreviewed result fields; export refused")


def numeric(value):
    if isinstance(value, dict):
        for item in value.values():
            numeric(item)
    elif isinstance(value, list):
        for item in value:
            numeric(item)
    elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numeric aggregate required")


def scalar(value):
    if isinstance(value, (dict, list)):
        raise ValueError("scalar aggregate required, not a vector")
    numeric(value)


def ci(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("exactly two interval endpoints required")
    for item in value:
        scalar(item)
    if not -1 <= value[0] <= value[1] <= 1:
        raise ValueError("invalid bounded interval")


def project_result(raw):
    keys(raw, TOP_KEYS)
    render(raw)  # rejects wrong protocol, fold count, bootstrap semantics and invalid CIs
    if raw["publication_ready"] is not False or raw["training_identity_independence"] != "unverified":
        raise ValueError("unreviewed publication/independence gate")
    keys(raw["models"], MODEL_KEYS)
    keys(raw["provenance"], MODEL_KEYS)
    for row in raw["provenance"].values():
        keys(row, {"checkpoint_name", "checkpoint_sha256", "status"})
    for name, model in raw["models"].items():
        expected = {"metrics", "official_folds", "subject_ci95"}
        if name != "frozen":
            expected.add("paired_gain")
        keys(model, expected)
        for field in ("metrics", "subject_ci95"):
            keys(model[field], METRIC_KEYS)
        for value in model["metrics"].values():
            scalar(value)
        for value in model["subject_ci95"].values():
            ci(value)
        for row in model["official_folds"]:
            keys(row, {"fold", "threshold", "accuracy"})
            for value in row.values():
                scalar(value)
        if name != "frozen":
            keys(model["paired_gain"], METRIC_KEYS)
            for row in model["paired_gain"].values():
                keys(row, {"delta", "subject_ci95"})
                scalar(row["delta"])
                ci(row["subject_ci95"])
        numeric(model)
    keys(raw["seed_aggregate"], METRIC_KEYS)
    for row in raw["seed_aggregate"].values():
        keys(row, {"mean", "std", "n_seeds", "delta_mean_vs_frozen", "fixed_checkpoint_mean_gain_subject_ci95"})
        numeric(row)
        for key in ("mean", "std", "n_seeds", "delta_mean_vs_frozen"):
            scalar(row[key])
        ci(row["fixed_checkpoint_mean_gain_subject_ci95"])
    for key in METRIC_KEYS:
        lo, hi = raw["seed_aggregate"][key]["fixed_checkpoint_mean_gain_subject_ci95"]
        if (key == "roc_auc" and lo <= 0) or (key != "roc_auc" and not lo <= 0 <= hi):
            raise ValueError("README inference no longer matches; new review required")
    detection = raw["detection_sensitivity"]
    keys(detection, {"endpoint_fallback_count", "endpoint_fallback_fraction", "pairs_with_any_fallback",
                     "both_detected_pairs", "both_detected_positive", "both_detected_negative", "both_detected_roc", "scope"})
    keys(detection["both_detected_roc"], MODEL_KEYS)
    for row in detection["both_detected_roc"].values():
        keys(row, METRIC_KEYS - {"accuracy_official_folds"})
        for value in row.values():
            scalar(value)
    public_detection = {key: copy.deepcopy(value) for key, value in detection.items() if key != "scope"}
    numeric(public_detection)
    for key, value in public_detection.items():
        if key != "both_detected_roc":
            scalar(value)
    public_detection["scope"] = "descriptive retained-pair ROC subset; no subset CI or canonical accuracy"
    boot = raw["bootstrap"]
    counts = {key: boot[key] for key in ("n_requested", "seed", "n_valid_roc", "n_valid_accuracy", "train_test_shared_person_count_per_fold")}
    keys(counts["train_test_shared_person_count_per_fold"], map(str, range(10)))
    numeric(counts)
    for value in counts.values():
        for item in (value.values() if isinstance(value, dict) else [value]):
            if type(item) is not int or item < 0:
                raise ValueError("nonnegative scalar bootstrap counts required")
    public_boot = {**counts, "sampling_unit": "person", "positive_weight": "one shared person multiplicity",
                   "negative_weight": "product of both endpoint multiplicities", "accuracy_threshold_reselection": True,
                   "conditional_on_fixed_trained_checkpoints_and_official_folds": True,
                   "includes_training_seed_population_uncertainty": False,
                   "folds_are_subject_disjoint": all(v == 0 for v in counts["train_test_shared_person_count_per_fold"].values()),
                   "boundary_percentile_intervals_do_not_establish_zero_risk": True}
    if boot != public_boot:
        raise ValueError("unreviewed bootstrap fields or semantics")
    grid = raw["threshold_grid"]
    keys(grid, {"minimum", "maximum", "count", "sha256_float64_little_endian", "tie_break"})
    if (grid["minimum"], grid["maximum"], grid["count"], grid["tie_break"]) != (-1, 1, 4001, "lowest threshold"):
        raise ValueError("unreviewed accuracy grid")
    if not re.fullmatch(r"[0-9a-f]{64}", grid["sha256_float64_little_endian"]):
        raise ValueError("invalid grid checksum")
    versions = raw["package_versions"]
    keys(versions, {"torch", "facenet-pytorch", "numpy", "scikit-learn", "Pillow"})
    if any(not isinstance(v, str) or not re.fullmatch(r"\d+(?:\.\d+){1,3}(?:[+.][A-Za-z0-9]+)?", v) for v in versions.values()):
        raise ValueError("unreviewed package version format")
    numeric_fields = {key: raw[key] for key in ("n_pairs", "n_subjects", "n_positive", "n_negative", "n_unique_crops_embedded", "negative_events_at_nominal_far")}
    keys(numeric_fields["negative_events_at_nominal_far"], {"0.01", "0.001"})
    numeric(numeric_fields)
    for key, value in numeric_fields.items():
        if key != "negative_events_at_nominal_far" and (type(value) is not int or value < 0):
            raise ValueError("scalar observation counts required")
    if numeric_fields["negative_events_at_nominal_far"] != {"0.01": 30.0, "0.001": 3.0}:
        raise ValueError("unexpected FAR event budget")
    # Never copy original prose, checkpoint filenames/hashes or manifest commands.
    result = {**numeric_fields, "protocol": raw["protocol"], "seeds": [42, 1, 2],
              "models": copy.deepcopy(raw["models"]), "seed_aggregate": copy.deepcopy(raw["seed_aggregate"]),
              "bootstrap": public_boot, "detection_sensitivity": public_detection,
              "threshold_grid": copy.deepcopy(grid), "package_versions": copy.deepcopy(versions),
              "cache_full_transform_replay": True, "accuracy_acceptance": "cosine >= threshold",
              "delta_definition": "tuned minus frozen; EER improvement negative, other improvements positive",
              "roc_scope": "test-derived descriptive ROC operating points, not deployment calibration",
              "legacy_accuracy_protocol": "historical interleaved results remain separate",
              "checkpoint_training_provenance": "legacy checkpoints; training manifests not independently verified",
              "checkpoint_records_omitted": len(raw["provenance"]),
              "training_identity_independence": "unverified", "publication_ready": False}
    scan_unsafe(result)
    return result


def verify_manifest(path, root, experiment):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != experiment:
        raise ValueError("unexpected experiment manifest")
    for field in ("inputs", "outputs"):
        if not isinstance(manifest.get(field), list) or not manifest[field]:
            raise ValueError("nonempty record lists required")
        for record in manifest[field]:
            try:
                source = Path(record["path"])
                if not source.is_absolute():
                    source = root / source
                if file_record(source) != record:
                    raise ValueError("record checksum/size mismatch")
            except (OSError, KeyError, TypeError):
                raise ValueError("missing or malformed declared record") from None
    return manifest


def verify_archive(path, expected):
    if set(expected) != MEMBERS:
        raise ValueError("expected map does not match reviewed membership")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != MEMBERS:
            raise ValueError("archive membership mismatch or duplicates")
        for name in names:
            data = archive.read(name)
            if digest(data) != expected[name]:
                raise ValueError("archive member checksum mismatch")
            scan_unsafe(json.loads(data) if name.endswith(".json") else data.decode("utf-8"))


def build_export(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("new empty export directory required")
    directory = root / DIRECTORY
    source, table = directory / "lfw_bound_evaluation.json", directory / "lfw_table.tex"
    paths = [source, table, directory / "lfw_bound_evaluation.manifest.json", directory / "lfw_presentation.manifest.json"]
    before = [file_record(path) for path in paths]
    evaluation = verify_manifest(paths[2], root, "lfw-bound-3seed-subject-evaluation")
    presentation = verify_manifest(paths[3], root, "lfw-source-bound-table-presentation")
    if (file_record(source) not in evaluation["outputs"] or file_record(table) not in presentation["outputs"]
            or file_record(source) not in presentation["inputs"] or file_record(paths[2]) not in presentation["inputs"]):
        raise ValueError("result/table/evaluation linkage missing")
    raw = json.loads(source.read_text(encoding="utf-8"))
    result = project_result(raw)
    generated = render(raw).encode("utf-8")
    if table.read_text(encoding="utf-8") != generated.decode("utf-8"):
        raise ValueError("table differs from generated verified result")
    members = {"results/lfw.json": encode(result), "tables/lfw.tex": generated}
    for label, manifest, path in (("evaluation", evaluation, paths[2]), ("presentation", presentation, paths[3])):
        members[f"manifests/{label}.sanitized.json"] = encode({
            "schema_version": 1, "experiment": manifest["experiment"], "original_manifest_sha256": sha256_file(path),
            "locally_verified_input_records": len(manifest["inputs"]),
            "locally_verified_output_records": len(manifest["outputs"]),
            "record_paths_and_digests_omitted": True, "commands_and_machine_paths_omitted": True,
            "publication_ready": False})
    members["README.txt"] = (
        b"Separate LFW aggregate evidence, not full reproduction or publication clearance.\n"
        b"All declared direct source inputs/outputs verified locally; private data not redistributed.\n"
        b"Result omits checkpoint filenames/digests and replaces original prose with reviewed descriptions.\n"
        b"Original and exported checksums differ and are explicitly labelled in CERTIFICATE.json.\n"
        b"Training provenance and mined-train identity independence are unverified.\n"
        b"Accuracy/EER/low-FAR gain intervals include zero; AUC gain is positive in these fixed checkpoints.\n"
        b"Build: python -m scripts.export_lfw_evidence --root <project> --out <new-directory>.\n"
    )
    members["CERTIFICATE.json"] = encode({
        "policy": "tbiom-reviewed-lfw-aggregates-v1", "publication_ready": False,
        "full_reproduction": False, "training_identity_independence": "unverified",
        "checkpoint_training_provenance": "unverified", "original_result_sha256": sha256_file(source),
        "exported_result_sha256": digest(members["results/lfw.json"]),
        "original_table_sha256": sha256_file(table), "exported_table_sha256": digest(generated),
        "exporter_sha256": sha256_file(Path(__file__)),
        "members_excluding_certificate": {name: digest(data) for name, data in members.items()},
        "private_records_listed": False, "checkpoint_records_omitted": len(raw["provenance"])})
    for name, data in members.items():
        scan_unsafe(json.loads(data) if name.endswith(".json") else data.decode())
    if before != [file_record(path) for path in paths]:
        raise ValueError("export source artifacts changed")
    out.mkdir(parents=True, exist_ok=True)
    target, temporary = out / ZIP_NAME, out / (ZIP_NAME + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        verify_archive(temporary, {name: digest(data) for name, data in members.items()})
        verify_manifest(paths[2], root, "lfw-bound-3seed-subject-evaluation")
        verify_manifest(paths[3], root, "lfw-source-bound-table-presentation")
        if before != [file_record(path) for path in paths]:
            raise ValueError("export sources changed while writing archive")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(build_export(args.root, args.out))


if __name__ == "__main__":
    main()
