"""Allowlisted CACD serial aggregate/table export; not data or publication clearance."""
from __future__ import annotations

import argparse
import copy
import json
import re
import zipfile
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, sha256_file, write_experiment_manifest
from scripts.cacd_metrics_v2 import ALL_METRICS
from scripts.evaluate_lfw_bound import THRESHOLD_SHA256
from scripts.export_lfw_evidence import digest, encode, keys, verify_manifest
from scripts.export_public_evidence import scan_unsafe
from scripts.export_roc_v2_evidence import bounded, count, sanitized
from scripts.render_cacd_metrics_v2 import summary, table, validate

DIRECTORY = "metrics/cacd_serial_20261003"
PRESENTATION = "metrics/cacd_serial_presentation_20261003"
EXPERIMENT = "cacd-vs-local-serial-4checkpoint-roc-v2"
PRESENTATION_EXPERIMENT = "cacd-serial-roc-v2-presentation"
ZIP_NAME = "cacd_evidence_bundle.zip"
TABLES = {"cacd_table.tex": table, "cacd_main_summary.tex": summary}
MEMBERS = {
    "README.txt", "CERTIFICATE.json", "results/cacd.json",
    "manifests/evaluation.sanitized.json", "manifests/presentation.sanitized.json",
    *{f"tables/{name}" for name in TABLES},
}
POLICY = "tbiom-cacd-serial-aggregate-v1"
README = (
    b"CACD-VS source-bound serial frozen/three-checkpoint aggregates and generated TeX blocks.\n"
    b"4000 pairs, ten folds. Shared fold/class-stratified pair bootstrap; not subject or training-seed-population CI or an ensemble.\n"
    b"Accuracy thresholds use other folds and are reselected per draw; ROC thresholds are not deployment-calibrated.\n"
    b"Nearly unchanged AUC coexists with worse EER, TAR at FAR 1% and fixed-grid accuracy. No universal improvement claim.\n"
    b"Original and exported aggregate checksums are distinct; direct native records verified locally. No raw scores, arrays, images, captions, identity records or model weights redistributed.\n"
    b"Manifest projections omit paths, record digests, parameters and commands. Preprocessing shared but original alignment replay unverified.\n"
    b"Original checkpoint training provenance, benchmark independence, person crosswalk, ethics/access and disclosure review remain unresolved.\n"
    b"This is partial aggregate evidence, not full reproduction, privacy certification, redistribution permission or publication clearance.\n"
    b"Build: python -m scripts.export_cacd_evidence --root <project> --out <new-directory>.\n"
)


def project(raw):
    keys(raw, {"inference", "plan", "alignment_fallback_fraction", "execution_complete",
               "legacy_results_rewritten", "publication_ready"})
    data = validate(raw)
    if raw["legacy_results_rewritten"] is not False:
        raise ValueError("legacy rewrite not permitted")
    bounded(raw["alignment_fallback_fraction"])
    keys(data, {"metric_version", "n_pairs", "n_positive", "n_negative", "n_folds", "models",
                "three_checkpoint_aggregate", "accuracy", "bootstrap", "scope",
                "training_identity_independence", "publication_ready"})
    for field in ("n_pairs", "n_positive", "n_negative", "n_folds"):
        count(data[field])
    keys(data["models"], {"frozen", "tuned_seed42", "tuned_seed1", "tuned_seed2"})
    for model in data["models"].values():
        keys(model, ALL_METRICS)
        for metric in model.values():
            keys(metric, {"point", "pair_ci95", "delta_vs_frozen", "paired_pair_delta_ci95"})
            if not 0 <= metric["pair_ci95"][0] <= metric["pair_ci95"][1] <= 1:
                raise ValueError("point interval outside metric bounds")
    for row in data["three_checkpoint_aggregate"].values():
        keys(row, {"mean", "sd", "pair_mean_ci95", "mean_checkpoint_delta", "paired_pair_delta_ci95"})
        if not 0 <= row["pair_mean_ci95"][0] <= row["pair_mean_ci95"][1] <= 1:
            raise ValueError("mean interval outside metric bounds")
    boot = data["bootstrap"]
    keys(boot, {"seed", "requested", "valid", "sampling_unit", "percentile_method",
                "shared_draws_across_models", "subject_metadata_available", "conditioning"})
    for field in ("seed", "requested", "valid"):
        count(boot[field])
    if boot["percentile_method"] != "linear" or boot["valid"] != boot["requested"]:
        raise ValueError("completed linear-percentile bootstrap required")
    accuracy = data["accuracy"]
    keys(accuracy, {"threshold_grid", "threshold_grid_sha256", "threshold_selection", "scope",
                    "legacy_grid_equivalence_claimed"})
    if (accuracy["threshold_grid"] != "fixed [-1,1],4001 points; lowest-threshold tie break"
            or not isinstance(accuracy["threshold_grid_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", accuracy["threshold_grid_sha256"])
            or accuracy["threshold_grid_sha256"] != THRESHOLD_SHA256
            or accuracy["legacy_grid_equivalence_claimed"] is not False):
        raise ValueError("unreviewed accuracy grid semantics")
    plan = raw["plan"]
    if (plan.get("subject_metadata_available") is not False
            or plan.get("publication_ready") is not False
            or plan.get("execution_complete") is not False):
        raise ValueError("historical input-only plan required; no person crosswalk")
    preprocessing = {
        "function": "age_gap.models.facenet.preprocess_bgr", "input_size": 160,
        "channel_order": "RGB", "normalization": "(x - 127.5) / 128",
        "source_crop_format": "BGR uint8 112x112", "shared_between_frozen_and_tuned": True,
    }
    if (plan.get("preprocessing") != preprocessing
            or any(type(plan["preprocessing"][key]) is not type(value) for key, value in preprocessing.items())):
        raise ValueError("unreviewed preprocessing")
    # Never redistribute the plan's weight/cache records, machine paths or arbitrary prose.
    public = copy.deepcopy(data)
    public["scope"] = "conditional pair CI; reused subjects/photos unaccounted for; no deployment calibration or multiplicity correction"
    public["accuracy"]["scope"] = "fixed-grid leave-one-fold-out estimand, not published-algorithm reproduction"
    result = {
        "inference": public, "preprocessing": preprocessing,
        "alignment_fallback_fraction": raw["alignment_fallback_fraction"],
        "execution_complete": True, "legacy_results_rewritten": False,
        "checkpoint_training_provenance": "unverified", "original_alignment_replay": "unverified",
        "publication_ready": False,
    }
    scan_unsafe(result)
    return result


def verify_archive(path, expected):
    if set(expected) != MEMBERS:
        raise ValueError("unreviewed expected membership")
    with zipfile.ZipFile(path) as archive:
        info = archive.infolist()
        names = [item.filename for item in info]
        if len(names) != len(set(names)) or set(names) != MEMBERS:
            raise ValueError("archive membership mismatch or duplicates")
        if (archive.comment or any(item.comment or item.extra or item.file_size > 1_000_000 for item in info)
                or sum(item.file_size for item in info) > 4_000_000):
            raise ValueError("unreviewed ZIP metadata or size")
        for name in names:
            value = archive.read(name)
            if digest(value) != expected[name]:
                raise ValueError("archive checksum mismatch")
            scan_unsafe(json.loads(value) if name.endswith(".json") else value.decode("utf-8"))


def check_evidence(root, path):
    """Verify original native linkage even when an attacker rehashes the ZIP certificate."""
    root = Path(root).resolve()
    source, mp = root / DIRECTORY / "summary.json", root / DIRECTORY / "summary.manifest.json"
    pp = root / PRESENTATION / "presentation.manifest.json"
    native = verify_manifest(mp, root, EXPERIMENT)
    presentation = verify_manifest(pp, root, PRESENTATION_EXPERIMENT)
    raw = json.loads(source.read_bytes())
    if raw != native.get("metrics") or file_record(source) not in native["outputs"]:
        raise ValueError("native aggregate linkage mismatch")
    if any(file_record(item) not in presentation["inputs"] for item in (source, mp)):
        raise ValueError("presentation linkage mismatch")
    if presentation.get("metrics") != {"pair_level_only": True, "legacy_results_pooled": False, "publication_ready": False}:
        raise ValueError("unreviewed presentation gates")
    with zipfile.ZipFile(path) as archive:
        certificate = json.loads(archive.read("CERTIFICATE.json"))
        keys(certificate, {"policy", "exporter_sha256", "original_aggregate_sha256", "exported_aggregate_sha256",
            "members_excluding_certificate", "publication_ready", "full_reproduction", "privacy_certified",
            "ethics_clearance_attested", "disclosure_review_completed", "training_identity_independence", "private_records_listed"})
        if (certificate["policy"] != POLICY or certificate["exporter_sha256"] != sha256_file(Path(__file__))
                or certificate["training_identity_independence"] != "unverified"
                or any(certificate[key] is not False for key in ("publication_ready", "full_reproduction", "privacy_certified",
                    "ethics_clearance_attested", "disclosure_review_completed", "private_records_listed"))):
            raise ValueError("unreviewed certificate policy or clearance claim")
        verify_archive(path, {**certificate["members_excluding_certificate"],
            "CERTIFICATE.json": digest(archive.read("CERTIFICATE.json"))})
        if archive.read("README.txt") != README:
            raise ValueError("unreviewed archive prose")
        exported = archive.read("results/cacd.json")
        if (certificate["original_aggregate_sha256"] != sha256_file(source)
                or certificate["exported_aggregate_sha256"] != digest(exported)
                or json.loads(exported) != project(raw)):
            raise ValueError("projection or original aggregate binding mismatch")
        for label, manifest, manifest_path in (("evaluation", native, mp), ("presentation", presentation, pp)):
            if json.loads(archive.read(f"manifests/{label}.sanitized.json")) != sanitized(manifest, sha256_file(manifest_path)):
                raise ValueError("sanitized native manifest mismatch")
        for name, renderer in TABLES.items():
            table_path = pp.parent / name
            expected = table_path.read_bytes()
            if (file_record(table_path) not in presentation["outputs"]
                    or expected.decode("utf-8").replace("\r\n", "\n") != renderer(raw)
                    or archive.read(f"tables/{name}") != expected):
                raise ValueError("generated source/archive presentation mismatch")
    return [source, mp, pp, *(pp.parent / name for name in TABLES), Path(__file__)]


def build_export(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("fresh empty export destination required")
    source = root / DIRECTORY / "summary.json"
    mp = source.with_suffix(".manifest.json")
    pp = root / PRESENTATION / "presentation.manifest.json"
    paths = [source, mp, pp, Path(__file__)]
    paths.extend(root / PRESENTATION / name for name in TABLES)
    before = [file_record(path) for path in paths]
    source_bytes, manifest_bytes, presentation_bytes = (path.read_bytes() for path in (source, mp, pp))
    if any(digest(value) != record["sha256"] for value, record in zip(
        (source_bytes, manifest_bytes, presentation_bytes), before[:3], strict=True
    )):
        raise ValueError("input changed before parsing")
    native = verify_manifest(mp, root, EXPERIMENT)
    presentation = verify_manifest(pp, root, PRESENTATION_EXPERIMENT)
    if native != json.loads(manifest_bytes) or presentation != json.loads(presentation_bytes):
        raise ValueError("manifest changed during verification")
    raw = json.loads(source_bytes)
    if raw != native.get("metrics") or before[0] not in native["outputs"]:
        raise ValueError("native aggregate linkage mismatch")
    if any(record not in presentation["inputs"] for record in before[:2]):
        raise ValueError("presentation missing native/aggregate linkage")
    if presentation.get("metrics") != {"pair_level_only": True, "legacy_results_pooled": False, "publication_ready": False}:
        raise ValueError("unreviewed presentation gates")
    members = {
        "results/cacd.json": encode(project(raw)),
        "manifests/evaluation.sanitized.json": encode(sanitized(native, digest(manifest_bytes))),
        "manifests/presentation.sanitized.json": encode(sanitized(presentation, digest(presentation_bytes))),
    }
    for filename, renderer in TABLES.items():
        path = root / PRESENTATION / filename
        generated = path.read_bytes()
        if (file_record(path) not in presentation["outputs"]
                or generated.decode("utf-8").replace("\r\n", "\n") != renderer(raw)):
            raise ValueError("generated table/summary bytes or binding differ")
        members[f"tables/{filename}"] = generated
    members["README.txt"] = README
    members["CERTIFICATE.json"] = encode({
        "policy": POLICY, "exporter_sha256": sha256_file(Path(__file__)),
        "original_aggregate_sha256": digest(source_bytes),
        "exported_aggregate_sha256": digest(members["results/cacd.json"]),
        "members_excluding_certificate": {name: digest(value) for name, value in members.items()},
        "publication_ready": False, "full_reproduction": False, "privacy_certified": False,
        "ethics_clearance_attested": False, "disclosure_review_completed": False,
        "training_identity_independence": "unverified", "private_records_listed": False,
    })
    # Bind all direct presentation/evaluation inputs, including private records locally,
    # without placing their paths/digests in the distributable archive.
    for manifest in (native, presentation):
        for record in manifest["inputs"] + manifest["outputs"]:
            path = Path(record["path"])
            paths.append(path if path.is_absolute() else root / path)
    script_root = Path(__file__).resolve().parent
    paths.extend(script_root / name for name in (
        "export_lfw_evidence.py", "export_public_evidence.py", "export_roc_v2_evidence.py",
        "render_cacd_metrics_v2.py", "cacd_metrics_v2.py",
    ))
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    initial = {record["path"]: record for record in before}
    before = [file_record(path) for path in paths]
    if any(initial[record["path"]] != record for record in before if record["path"] in initial):
        raise ValueError("direct inputs changed while collecting dependencies")
    out.mkdir(parents=True, exist_ok=True)
    target, temporary = out / ZIP_NAME, out / (ZIP_NAME + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, value in members.items():
                archive.writestr(name, value)
        verify_archive(temporary, {name: digest(value) for name, value in members.items()})
        verify_manifest(mp, root, EXPERIMENT)
        verify_manifest(pp, root, PRESENTATION_EXPERIMENT)
        check_evidence(root, temporary)
        if before != [file_record(path) for path in paths]:
            raise ValueError("inputs changed during export")
        temporary.replace(target)
        manifest_path = out / "export.manifest.json"
        write_experiment_manifest(
            manifest_path,
            experiment="cacd-serial-allowlisted-evidence-export",
            parameters={"policy": POLICY, "private_record_paths_in_zip": False},
            metrics={"member_count": len(MEMBERS), "archive_native_linkage_verified": True,
                "publication_ready": False, "privacy_certified": False, "full_reproduction": False},
            inputs=paths,
            outputs=[target],
        )
        written = json.loads(manifest_path.read_bytes())
        if before != written["inputs"]:
            manifest_path.unlink()
            target.unlink()
            raise ValueError("dependencies changed during manifest write; export withdrawn")
    except Exception:
        if target.exists():
            target.unlink()
        receipt = out / "export.manifest.json"
        if receipt.exists():
            receipt.unlink()
        raise
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(build_export(args.root, args.out))


if __name__ == "__main__":
    main()
