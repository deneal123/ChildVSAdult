"""Schema-allowlisted ROC-v2 aggregate export, separate from historical bundles."""
from __future__ import annotations

import argparse
import copy
import json
import zipfile
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, sha256_file
from scripts.benchmark_metrics_v2 import KEYS, METRICS
from scripts.export_lfw_evidence import ci, digest, encode, keys, scalar, verify_manifest
from scripts.export_public_evidence import scan_unsafe
from scripts.render_fgnet_metrics_v2 import headline, operating
from scripts.render_internal_metrics_v2 import render as internal_table

EXPORTS = {
    "fgnet": ("fgnet_metrics_v2_20261003", "fgnet-headline-full-cache-roc-v2"),
    "internal": ("internal_metrics_v2_20261003", "internal-roc-v2-paired-inference"),
    "lfw": ("lfw_metrics_v2_20261003", "lfw-saved-score-roc-v2-compatibility"),
}
PRESENTATIONS = {
    "fgnet": ("fgnet-roc-v2-presentation", {"headline_table.tex": headline, "operating_table.tex": operating}),
    "internal": ("internal-roc-v2-table-presentation", {"internal_table.tex": internal_table}),
}
MEMBERS = {"README.txt", "CERTIFICATE.json"} | {
    f"{folder}/{name}{suffix}" for name in EXPORTS
    for folder, suffix in (("results", ".json"), ("manifests", ".sanitized.json"))
} | {f"manifests/{name}.presentation.sanitized.json" for name in PRESENTATIONS} | {
    f"tables/{filename}" for _, tables in PRESENTATIONS.values() for filename in tables
}
ZIP_NAME = "roc_v2_evidence_bundle.zip"
MAX_MEMBER_BYTES = 1_000_000
MAX_TOTAL_BYTES = 4_000_000


def count(value):
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative scalar integer count required")


def bounded(value, *, signed=False):
    scalar(value)
    if not (-1 if signed else 0) <= value <= 1:
        raise ValueError("bounded scalar metric required")


def point_ci(value, *, signed=False):
    keys(value, {"point", "ci95"})
    bounded(value["point"], signed=signed)
    ci(value["ci95"])


def external(raw):
    keys(raw, {"metric_version", "n_pairs", "n_positive", "n_negative", "n_subjects", "models",
        "three_checkpoint_aggregate", "bootstrap", "scope", "training_identity_independence", "publication_ready"})
    if (raw["metric_version"] != "empirical-roc-v2" or raw["training_identity_independence"] != "unverified"
            or raw["publication_ready"] is not False):
        raise ValueError("ROC-v2 unresolved publication/identity gates required")
    for field in ("n_pairs", "n_positive", "n_negative", "n_subjects"):
        count(raw[field])
    if raw["n_positive"] + raw["n_negative"] != raw["n_pairs"]:
        raise ValueError("inconsistent class counts")
    keys(raw["models"], KEYS)
    for model in raw["models"].values():
        keys(model, METRICS)
        for row in model.values():
            keys(row, {"point", "ci95", "delta_vs_frozen", "delta_ci95"})
            bounded(row["point"])
            bounded(row["delta_vs_frozen"], signed=True)
            ci(row["ci95"])
            ci(row["delta_ci95"])
    keys(raw["three_checkpoint_aggregate"], METRICS)
    for row in raw["three_checkpoint_aggregate"].values():
        keys(row, {"mean", "sd", "mean_ci95", "mean_checkpoint_delta", "delta_ci95"})
        bounded(row["mean"])
        bounded(row["sd"])
        bounded(row["mean_checkpoint_delta"], signed=True)
        ci(row["mean_ci95"])
        ci(row["delta_ci95"])
    boot = raw["bootstrap"]
    keys(boot, {"seed", "requested", "valid", "positive_weight", "negative_weight", "shared_draws_across_models",
        "thresholds_reselected", "conditioning"})
    for key in ("seed", "requested", "valid"):
        count(boot[key])
    expected = {"positive_weight": "one shared person multiplicity",
        "negative_weight": "both-endpoint multiplicity product", "shared_draws_across_models": True,
        "thresholds_reselected": True,
        "conditioning": "fixed checkpoints and benchmark protocol; not training-seed population or ensemble-score AUC"}
    if any(type(boot[k]) is not type(v) or boot[k] != v for k, v in expected.items()) or not 2 <= boot["valid"] <= boot["requested"]:
        raise ValueError("unreviewed bootstrap semantics")
    result = {k: copy.deepcopy(raw[k]) for k in ("metric_version", "n_pairs", "n_positive", "n_negative", "n_subjects",
        "models", "three_checkpoint_aggregate", "bootstrap", "training_identity_independence", "publication_ready")}
    result["scope"] = "test-derived ROC; both EER definitions; fixed checkpoint person CI; no deployment calibration or multiplicity correction"
    return result


def project(name, raw):
    if name == "fgnet":
        keys(raw, {"protocol", "overall", "large_gap_25plus", "coverage", "cached_images", "cached_subjects", "scope",
            "reused_cache_preprocessing", "original_error_protocol_reused", "image_inference_performed",
            "source_embedding_model_count", "publication_ready"})
        public_overall = external(raw["overall"])
        public_large_gap = external(raw["large_gap_25plus"])
        headline(raw)
        coverage = raw["coverage"]
        keys(coverage, {"source_positives", "retained_positives", "source_25plus_positives", "retained_25plus_positives"})
        for value in coverage.values():
            count(value)
        keys(raw["reused_cache_preprocessing"], {"function", "input_size", "channel_order", "normalization", "source_crop_format", "shared_between_frozen_and_tuned"})
        for field in ("cached_images", "cached_subjects", "source_embedding_model_count"):
            count(raw[field])
        result = {key: copy.deepcopy(raw[key]) for key in ("coverage", "cached_images", "cached_subjects", "source_embedding_model_count")}
        result.update(overall=public_overall, large_gap_25plus=public_large_gap,
            protocol="global endpoint-age-matched, seed42/tolerance2; source-positive-gap strata",
            scope="650 cached images, not all raw FG-NET photos; cached preprocessing not replayed; no split-local scores",
            original_error_protocol_reused=False, image_inference_performed=False, publication_ready=False)
    elif name == "lfw":
        keys(raw, {"roc_v2", "compatibility", "official_fold_accuracy_recomputed", "accuracy_scope", "image_inference_performed",
            "legacy_results_rewritten", "training_identity_independence", "publication_ready"})
        if any(raw[k] is not False for k in ("official_fold_accuracy_recomputed", "image_inference_performed", "legacy_results_rewritten", "publication_ready")) or raw["training_identity_independence"] != "unverified":
            raise ValueError("unreviewed LFW audit gates")
        compat = raw["compatibility"]
        keys(compat, {"comparisons", "max_absolute_difference", "matches_within_1e_minus_12", "metric_mapping", "scope"})
        count(compat["comparisons"])
        bounded(compat["max_absolute_difference"])
        expected_map = {"roc_auc": "roc_auc", "eer": "eer_discrete_minimax", "tar@far=0.01": "tar@far=0.01", "tar@far=0.001": "tar@far=0.001"}
        if compat["metric_mapping"] != expected_map or compat["matches_within_1e_minus_12"] is not True or compat["max_absolute_difference"] > 1e-12:
            raise ValueError("compatibility findings changed; review required")
        result = {"roc_v2": external(raw["roc_v2"]), "compatibility": {k: copy.deepcopy(compat[k]) for k in compat if k != "scope"},
            "official_fold_accuracy_recomputed": False, "image_inference_performed": False,
            "legacy_results_rewritten": False, "training_identity_independence": "unverified", "publication_ready": False,
            "scope": "actual saved-score point/CI compatibility only, not universal equivalence; original official-fold accuracy separate"}
    elif name == "internal":
        keys(raw, {"metric_version", "subsets", "exact_minus_original_model_delta", "bootstrap", "scope", "eer_scope",
            "deployment_calibration", "verified_identity_independence", "publication_ready", "source_protocol"})
        if (raw["metric_version"] != "empirical-roc-v2" or raw["deployment_calibration"] is not False
                or raw["verified_identity_independence"] is not False or raw["publication_ready"] is not False):
            raise ValueError("unresolved internal identity/publication gates required")
        keys(raw["subsets"], {"original_tolerance", "exact_age_subset"})
        public_subsets = {}
        for name_key, subset in raw["subsets"].items():
            keys(subset, {"n_positive", "n_negative", "recorded_people", "metrics", "operating_points"})
            for field in ("n_positive", "n_negative", "recorded_people"):
                count(subset[field])
            keys(subset["metrics"], METRICS)
            for metric in subset["metrics"].values():
                keys(metric, {"frozen", "tuned", "tuned_minus_frozen"})
                for model, row in metric.items():
                    point_ci(row, signed=model == "tuned_minus_frozen")
            # Scalar aggregate operating points only, never row scores or embeddings.
            keys(subset["operating_points"], {"frozen", "tuned"})
            for ops in subset["operating_points"].values():
                keys(ops, {"0.01", "0.001"})
                for far, row in ops.items():
                    keys(row, {"tar", "far_achieved", "similarity_threshold", "reject_all"})
                    bounded(row["tar"])
                    bounded(row["far_achieved"])
                    if row["far_achieved"] > float(far) or type(row["reject_all"]) is not bool:
                        raise ValueError("invalid operating point")
                    if row["similarity_threshold"] is not None:
                        bounded(row["similarity_threshold"], signed=True)
            public_subsets[name_key] = copy.deepcopy(subset)
        keys(raw["exact_minus_original_model_delta"], METRICS)
        for row in raw["exact_minus_original_model_delta"].values():
            point_ci(row, signed=True)
        boot = raw["bootstrap"]
        keys(boot, {"seed", "requested", "valid", "recorded_people", "sampling", "scope", "threshold_reselection", "multiple_comparisons_corrected"})
        if (boot["sampling"] != "joint recorded-person dyad-multiplicity product; positive/negative matched block same weight"
                or boot["threshold_reselection"] is not True or boot["multiple_comparisons_corrected"] is not False):
            raise ValueError("unreviewed internal bootstrap semantics")
        for key in ("seed", "requested", "valid", "recorded_people"):
            count(boot[key])
        if not 2 <= boot["valid"] <= boot["requested"]:
            raise ValueError("invalid internal resample counts")
        keys(raw["source_protocol"], {"rule", "original_blocks", "exact_blocks", "retained_fraction", "original_recorded_people",
            "exact_recorded_people", "bootstrap_seed", "n_boot", "training_seed", "device", "threads", "selection_based_on_scores",
            "public_preregistration", "publication_ready", "limits"})
        protocol = raw["source_protocol"]
        if any(protocol[k] is not False for k in ("selection_based_on_scores", "public_preregistration", "publication_ready")):
            raise ValueError("internal selection/publication gates changed")
        public_protocol = {}
        for k in ("original_blocks", "exact_blocks", "original_recorded_people", "exact_recorded_people", "bootstrap_seed", "n_boot", "training_seed", "threads"):
            count(protocol[k])
            public_protocol[k] = protocol[k]
        bounded(protocol["retained_fraction"])
        if (protocol["original_blocks"] <= 0 or protocol["exact_blocks"] > protocol["original_blocks"]
                or protocol["retained_fraction"] != protocol["exact_blocks"] / protocol["original_blocks"]):
            raise ValueError("invalid subset retention denominator")
        public_protocol.update(retained_fraction=protocol["retained_fraction"], selection_based_on_scores=False, public_preregistration=False,
            rule="pre-score exact endpoint-age whole-block subset; no re-mining")
        public_boot = {k: boot[k] for k in ("seed", "requested", "valid", "recorded_people", "threshold_reselection", "multiple_comparisons_corrected")}
        public_boot.update(sampling=boot["sampling"], scope="conditional fixed seed42 checkpoint and selected test ROC; not training-seed population")
        result = {"metric_version": "empirical-roc-v2", "subsets": public_subsets,
            "exact_minus_original_model_delta": copy.deepcopy(raw["exact_minus_original_model_delta"]),
            "bootstrap": public_boot, "source_protocol": public_protocol, "deployment_calibration": False,
            "verified_identity_independence": False, "publication_ready": False,
            "scope": "subset changes composition; AUC gain not proof of shortcut removal; legacy training and human identity independence unverified"}
        internal_table(raw)
    else:
        raise ValueError("unreviewed result name")
    scan_unsafe(result)
    return result


def sanitized(manifest, original_sha256):
    return {"original_manifest_sha256": original_sha256, "experiment": manifest["experiment"],
        "locally_verified_input_records": len(manifest["inputs"]), "locally_verified_output_records": len(manifest["outputs"]),
        "record_paths_and_digests_omitted": True, "parameters_and_commands_omitted": True, "publication_ready": False}


def verify_archive(path, expected):
    if set(expected) != MEMBERS:
        raise ValueError("unreviewed expected archive membership")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != MEMBERS:
            raise ValueError("archive membership mismatch or duplicates")
        info = archive.infolist()
        if (archive.comment or any(i.comment or i.extra or i.file_size > MAX_MEMBER_BYTES for i in info)
                or sum(i.file_size for i in info) > MAX_TOTAL_BYTES):
            raise ValueError("unreviewed ZIP metadata or oversized aggregate archive")
        for name in names:
            data = archive.read(name)
            if digest(data) != expected[name]:
                raise ValueError("archive member checksum mismatch")
            scan_unsafe(json.loads(data) if name.endswith(".json") else data.decode("utf-8"))


def build_export(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("new empty export directory required")
    members, bindings, paths, manifests, before = {}, {}, [], [], []
    for name, (directory, experiment) in EXPORTS.items():
        source = root / "metrics" / directory / "summary.json"
        mp = source.with_suffix(".manifest.json")
        paths.extend((source, mp))
        source_record, manifest_record = file_record(source), file_record(mp)
        before.extend((source_record, manifest_record))
        source_bytes, manifest_bytes = source.read_bytes(), mp.read_bytes()
        if digest(source_bytes) != source_record["sha256"] or digest(manifest_bytes) != manifest_record["sha256"]:
            raise ValueError("aggregate/manifest changed before parsing")
        manifest = verify_manifest(mp, root, experiment)
        if manifest != json.loads(manifest_bytes):
            raise ValueError("manifest changed during verification")
        manifests.append((mp, experiment))
        if file_record(source) not in manifest["outputs"]:
            raise ValueError("aggregate not bound as output")
        raw = json.loads(source_bytes)
        if raw != manifest["metrics"]:
            raise ValueError("aggregate differs from native metrics")
        members[f"results/{name}.json"] = encode(project(name, raw))
        members[f"manifests/{name}.sanitized.json"] = encode(sanitized(manifest, digest(manifest_bytes)))
        bindings[name] = {"original_aggregate_sha256": digest(source_bytes),
            "exported_aggregate_sha256": digest(members[f"results/{name}.json"])}
        if name in PRESENTATIONS:
            pe, tables = PRESENTATIONS[name]
            pp = source.parent / "presentation.manifest.json"
            presentation_record = file_record(pp)
            presentation_bytes = pp.read_bytes()
            if digest(presentation_bytes) != presentation_record["sha256"]:
                raise ValueError("presentation manifest changed before parsing")
            presentation = verify_manifest(pp, root, pe)
            if presentation != json.loads(presentation_bytes):
                raise ValueError("presentation manifest changed during verification")
            manifests.append((pp, pe))
            paths.append(pp)
            before.append(presentation_record)
            if any(file_record(p) not in presentation["inputs"] for p in (source, mp)):
                raise ValueError("presentation missing result/native-manifest linkage")
            for filename, renderer in tables.items():
                table_path = source.parent / filename
                paths.append(table_path)
                before.append(file_record(table_path))
                if file_record(table_path) not in presentation["outputs"] or table_path.read_text(encoding="utf-8") != renderer(raw):
                    raise ValueError("table differs from generated bound result")
                members[f"tables/{filename}"] = renderer(raw).encode("utf-8")
            members[f"manifests/{name}.presentation.sanitized.json"] = encode(sanitized(presentation, digest(presentation_bytes)))
    paths.append(Path(__file__))
    before.append(file_record(Path(__file__)))
    members["README.txt"] = (
        b"Three ROC-v2 aggregates and three generated tables; separate from historical evidence.\n"
        b"Manifest projections omit record paths/digests, parameters, commands and machine paths.\n"
        b"All declared direct input/output records are locally verified; no private scores, images, captions, embeddings or person IDs redistributed.\n"
        b"Original and exported aggregate checksums are separately labelled. Not full reproduction or publication clearance.\n"
        b"FG-NET: 650 cached photos, not all raw photos; original global matching, not split-local error scores.\n"
        b"Internal: exact subset changes composition; fixed seed42 recorded-person CI, not a human identity audit.\n"
        b"LFW: actual saved-score/CI compatibility, not universal equivalence; official-fold accuracy remains separate.\n"
        b"Both EER definitions retained. Test ROC thresholds are reselected, not calibrated for deployment.\n"
        b"AUC gains do not establish reliable low-FAR improvement, objective equivalence or causal longitudinal-source superiority.\n"
        b"Fixed-checkpoint intervals do not estimate training-seed population uncertainty; multiple comparisons not corrected.\n"
        b"Training provenance, benchmark independence, ethics/access and disclosure review remain unresolved.\n"
        b"Schema/marker checks are not privacy certification; no grant of access or redistribution rights.\n"
        b"Build: python -m scripts.export_roc_v2_evidence --root <project> --out <new-directory>.\n"
    )
    members["CERTIFICATE.json"] = encode({"policy": "tbiom-reviewed-roc-v2-aggregates-v1", "aggregate_bindings": bindings,
        "exporter_sha256": sha256_file(Path(__file__)), "members_excluding_certificate": {n: digest(b) for n, b in members.items()},
        "publication_ready": False, "full_reproduction": False, "privacy_certified": False,
        "ethics_clearance_attested": False, "disclosure_review_completed": False,
        "training_identity_independence": "unverified", "private_records_listed": False})
    for name, data in members.items():
        scan_unsafe(json.loads(data) if name.endswith(".json") else data.decode("utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    target, temporary = out / ZIP_NAME, out / (ZIP_NAME + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        verify_archive(temporary, {n: digest(b) for n, b in members.items()})
        for mp, experiment in manifests:
            verify_manifest(mp, root, experiment)
        if before != [file_record(p) for p in paths]:
            raise ValueError("export source artifacts changed")
        temporary.replace(target)
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
