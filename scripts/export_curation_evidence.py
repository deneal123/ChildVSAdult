"""Eight-member, schema-allowlisted export of three curation aggregate audits."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import zipfile
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, sha256_file
from scripts.export_lfw_evidence import verify_manifest
from scripts.export_public_evidence import scan_unsafe

ZIP_NAME = "curation_evidence_bundle.zip"
EXPORTS = {
    "lineage": ("curation_lineage_20261003", "historical-curation-lineage"),
    "reconstruction": ("curation_replay_global_20261003_v2", "shadow-global-id-curation-replay"),
    "integrity": ("constituent_policy_sensitivity_20261003", "constituent-integrity-sensitivity"),
}
MEMBERS = {"README.txt", "CERTIFICATE.json"} | {
    f"{folder}/{name}{suffix}" for name in EXPORTS
    for folder, suffix in (("results", ".json"), ("manifests", ".sanitized.json"))
}
PAIR_KEYS = {f"{split}_{label}" for split in ("train", "val", "test") for label in ("positive", "negative")}
SCHEMAS = {
    "lineage": {
        **dict.fromkeys(("post_groups", "mapped_post_groups", "mapped_person_clusters",
                        "unmapped_fallback_groups", "unmapped_fallback_faces", "pre_prune_groups",
                        "noisy_groups_dropped", "retained_groups", "pre_prune_faces",
                        "faces_in_dropped_noisy_groups", "dedup_faces_removed_from_retained_groups",
                        "retained_unique_faces", "low_consistency_multiface_groups_pre_prune",
                        "low_consistency_multiface_groups_retained", "dedup_face_set_mismatched_groups",
                        "dedup_face_order_only_mismatches"), int),
        "current_redundant_file_replays_final_faces": False,
        "scope": "record lineage only; no true-person or historical detector validation",
    },
    "reconstruction": {
        **dict.fromkeys(("pre_groups", "expected_retained_groups", "actual_retained_groups",
                        "noisy_groups_dropped", "missing_group_count", "extra_group_count",
                        "face_set_mismatched_groups", "face_order_only_mismatches", "metadata_mismatched_groups",
                        "dedup_removed_from_retained_groups", "expected_retained_unique_faces",
                        "actual_retained_unique_faces", "globally_removed_pre_face_memberships",
                        "detected_redundant_face_memberships", "detected_redundant_unique_ids",
                        "pre_cross_group_shared_face_ids", "pre_empty_groups", "final_cross_group_shared_face_ids",
                        "final_empty_groups", "embedding_cache_rows", "embedding_unique_ids",
                        "identical_duplicate_extra_rows", "pre_faces_without_embeddings"), int),
        "ordered_faces_replayed": True,
        "full_group_records_replayed": True,
        "scope": "new deterministic reconstruction; original dedup file/provenance not recovered",
        "cleanup_policy": "global redundant face-ID set, including shared memberships",
    },
    "integrity": {
        **dict.fromkeys(("pre_groups", "retained_groups", "any_constituent_noisy_pre_groups",
                        "retained_groups_with_noisy_constituent", "retained_unique_faces_in_affected_groups",
                        "all_constituents_cached_single_pre_groups", "all_constituents_cached_single_retained_groups",
                        "positive_groups", "retained_groups_unresolved_after_known_noisy_exclusion"), int),
        "constituent_coverage": dict.fromkeys(("singleton_groups", "constituent_posts",
                    "missing_constituent_posts", "unknown_constituent_posts", "groups_with_missing_constituents",
                    "groups_with_unknown_constituents", "merged_groups"), int),
        "representative_by_constituent_noisy": dict.fromkeys(("collage|any_noisy", "meme|any_noisy",
                    "missing|no_noisy", "multi_person|any_noisy", "single|any_noisy", "single|no_noisy",
                    "unknown|any_noisy", "unknown|no_noisy"), int),
        **{key: dict.fromkeys(PAIR_KEYS, int) for key in (
            "original_pair_counts", "pair_counts_after_any_noisy_group_exclusion",
            "pair_counts_after_all_cached_single_filter")},
        "positive_group_representative_categories": dict.fromkeys(("missing", "single", "unknown"), int),
        "positive_group_single_fraction_all": float,
        "positive_group_single_fraction_cached": float,
        "scope": "cached automatic categories; sensitivity only, no true-identity purity or retrained effect",
        "policy_scope": "known-noisy exclusion does not certify missing/unknown as clean; all-cached-single is a separate automatic sensitivity arm",
    },
}
README = (
    "Three curation aggregate audits, not a complete dataset or reproduction package.\n"
    "Only schema-allowlisted scalar counts, fractions, record-equality flags and fixed scope text.\n"
    "All direct manifest inputs/outputs are verified locally; private inputs are not redistributed.\n"
    "Sanitized manifests omit record paths, record digests, parameters and commands.\n"
    "Lineage: original empty redundant-face file does not replay the final snapshot.\n"
    "Reconstruction: new global-face-ID shadow replay matches ordered records and metadata,\n"
    "not original historical provenance, true-person purity or detector/LLM validation.\n"
    "One empty final group and two pre-prune shared face IDs are retained in the evidence.\n"
    "Integrity: cached automatic decisions only, not independent human annotations.\n"
    "17 retained groups are candidates, not confirmed errors; 669 other groups unresolved.\n"
    "Known-noisy and all-cached-single filters are unbalanced counts sensitivity, not training arms.\n"
    "No performance effect, rebalancing, retraining or adjudication has been completed here.\n"
    "99.1% single is conditional on cached representatives; overall fraction is 97.7%.\n"
    "No identity-independence, ethics clearance, controlled-access grant or submission-readiness claim.\n"
    "Schema/marker checks are not a privacy certification; small-cell disclosure review is pending.\n"
    "Original aggregate and sanitized/exported checksums are separately labelled in the certificate.\n"
    "Build: python -m scripts.export_curation_evidence --root <project> --out <new-directory>.\n"
)


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(value, schema):
    if isinstance(schema, dict):
        if not isinstance(value, dict) or value.keys() != schema.keys():
            raise ValueError("unreviewed aggregate fields; export refused")
        for key, item in schema.items():
            validate(value[key], item)
    elif schema is int:
        if type(value) is not int or value < 0:
            raise ValueError("nonnegative integer count required")
    elif schema is float:
        if type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("finite scalar fraction required")
    elif type(value) is not type(schema) or value != schema:
        raise ValueError("unreviewed scope or record-equality flag")


def project_result(name, raw):
    validate(raw, SCHEMAS[name])
    result = copy.deepcopy(raw)
    scan_unsafe(result)
    return result


def validate_consistency(results):
    lineage, replay, integrity = (results[name] for name in EXPORTS)
    if not (lineage["pre_prune_groups"] == replay["pre_groups"] == integrity["pre_groups"]
            and lineage["retained_groups"] == replay["actual_retained_groups"] == integrity["retained_groups"]
            and lineage["retained_unique_faces"] == replay["actual_retained_unique_faces"]
            and lineage["dedup_faces_removed_from_retained_groups"] == replay["dedup_removed_from_retained_groups"]):
        raise ValueError("inconsistent audited snapshots")
    categories = integrity["positive_group_representative_categories"]
    total = sum(categories.values())
    cached = total - categories["missing"]
    if (total != integrity["positive_groups"] or total == 0 or cached == 0
            or integrity["positive_group_single_fraction_all"] != categories["single"] / total
            or integrity["positive_group_single_fraction_cached"] != categories["single"] / cached):
        raise ValueError("incorrect positive-group denominators")
    # The reviewed README contains these observations. Changed findings need a new review.
    if (integrity["retained_groups_with_noisy_constituent"] != 17
            or integrity["retained_groups_unresolved_after_known_noisy_exclusion"] != 669
            or replay["final_empty_groups"] != 1 or replay["pre_cross_group_shared_face_ids"] != 2
            or round(integrity["positive_group_single_fraction_all"] * 100, 1) != 97.7
            or round(integrity["positive_group_single_fraction_cached"] * 100, 1) != 99.1):
        raise ValueError("README findings changed; new export review required")


def verify_archive(path, expected):
    if set(expected) != MEMBERS:
        raise ValueError("unreviewed expected membership")
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
    members, results, bindings, sources, before = {}, {}, {}, [], []
    for name, (directory, experiment) in EXPORTS.items():
        source = root / "metrics" / directory / "summary.json"
        manifest_path = source.with_name("summary.manifest.json")
        sources.extend((source, manifest_path))
        source_record, manifest_record = file_record(source), file_record(manifest_path)
        before.extend((source_record, manifest_record))
        manifest = verify_manifest(manifest_path, root, experiment)
        if source_record not in manifest["outputs"]:
            raise ValueError("aggregate result not bound as manifest output")
        raw = json.loads(source.read_text(encoding="utf-8"))
        if raw != manifest.get("metrics"):
            raise ValueError("aggregate differs from manifest metrics")
        results[name] = project_result(name, raw)
        members[f"results/{name}.json"] = encode(results[name])
        members[f"manifests/{name}.sanitized.json"] = encode({
            "schema_version": 1, "experiment": experiment,
            "original_manifest_sha256": manifest_record["sha256"],
            "locally_verified_input_records": len(manifest["inputs"]),
            "locally_verified_output_records": len(manifest["outputs"]),
            "record_paths_and_digests_omitted": True, "parameters_and_commands_omitted": True,
            "publication_ready": False, "full_reproduction": False,
        })
        bindings[name] = {"original_aggregate_sha256": source_record["sha256"],
                          "exported_aggregate_sha256": digest(members[f"results/{name}.json"])}
    validate_consistency(results)
    members["README.txt"] = README.encode("utf-8")
    members["CERTIFICATE.json"] = encode({
        "policy": "tbiom-reviewed-curation-aggregates-v1", "publication_ready": False,
        "full_reproduction": False, "human_adjudication_completed": False,
        "disclosure_review_completed": False,
        "historical_pipeline_provenance_recovered": False, "training_identity_independence": "unverified",
        "aggregate_bindings": bindings, "exporter_sha256": sha256_file(Path(__file__)),
        "members_excluding_certificate": {name: digest(data) for name, data in members.items()},
        "private_records_listed": False,
    })
    for name, data in members.items():
        scan_unsafe(json.loads(data) if name.endswith(".json") else data.decode("utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    target, temporary = out / ZIP_NAME, out / (ZIP_NAME + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        verify_archive(temporary, {name: digest(data) for name, data in members.items()})
        for directory, experiment in EXPORTS.values():
            verify_manifest(root / "metrics" / directory / "summary.manifest.json", root, experiment)
        if before != [file_record(path) for path in sources]:
            raise ValueError("export sources changed while writing archive")
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
