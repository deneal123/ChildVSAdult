"""Build a compact index of machine-readable artifacts used by the T-BIOM paper."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import sha256_file

METRICS = PROJECT_ROOT / "metrics"
OUTPUT = METRICS / "publication_artifact_index.json"
PRIVATE_ROW_ID = re.compile(r"(?:vk_|reddit_)?-?\d{5,}_\d{5,}", re.IGNORECASE)
INCLUDE = (
    "age_extraction_audit.manifest.json",
    "group_integrity_audit.manifest.json",
    "human_audit_pack_validation.manifest.json",
    "human_audit.manifest.json",
    "data_funnel.manifest.json",
    "pipeline_figure.manifest.json",
    "curation_lineage_20261003/summary.manifest.json",
    "curation_replay_global_*/summary.manifest.json",
    "constituent_integrity_20261003/summary.manifest.json",
    "constituent_policy_sensitivity_20261003/summary.manifest.json",
    "sensitivity_dedup.manifest.json",
    "independent_embedding_audit.manifest.json",
    "model_inventory.manifest.json",
    "pretraining_overlap_audit.manifest.json",
    "headline_stats.manifest.json",
    "fgnet_endpoint_subject_stats.manifest.json",
    "fgnet_endpoint_subject_stats_s*.manifest.json",
    "fgnet_endpoint_multiseed.manifest.json",
    "fgnet_retrieval_*/fgnet_retrieval_study.manifest.json",
    "fgnet_retrieval_*/retrieval_presentation.manifest.json",
    "fgnet_error_breakdown/fgnet_error_breakdown.manifest.json",
    "comparator_fgnet_endpoint_age_matched.manifest.json",
    "internal_endpoint_age_matched.manifest.json",
    "internal_metrics_v2_20261003/summary.manifest.json",
    "internal_metrics_v2_20261003/presentation.manifest.json",
    "fgnet_metrics_v2_20261003/summary.manifest.json",
    "fgnet_metrics_v2_20261003/presentation.manifest.json",
    "lfw_metrics_v2_20261003/summary.manifest.json",
    "lfw_bound_cache_*/lfw_bound_cache.manifest.json",
    "lfw_bound_evaluation_*/lfw_bound_evaluation.manifest.json",
    "lfw_bound_evaluation_*/lfw_presentation.manifest.json",
    "matched_arm_image_budget_*/summary.manifest.json",
    "delong_tests.manifest.json",
    "synthetic_parity.manifest.json",
    "sota_objectives.manifest.json",
    "noise_robust_baseline.manifest.json",
    "hardneg_multiseed.manifest.json",
    "hardneg_curve_full.manifest.json",
    "hardneg_curve_facenet_clean.manifest.json",
    "hardneg_curve_miner2.manifest.json",
    "hardneg_paired_pairs_clean.manifest.json",
    "subgroup_fmr_fnmr.manifest.json",
    "fairness_audit.manifest.json",
    "age_leakage.manifest.json",
    "headroom_audit.manifest.json",
    "multiseed_facenet_e10.manifest.json",
    "scaling_multiseed.manifest.json",
    "cross_platform_vk2reddit_3seed.manifest.json",
    "cross_source_reddit2vk_3seed.manifest.json",
    "cacd_vs/*.manifest.json",
    "sota_common_protocol/*.manifest.json",
    "strong_backbone_study/*.manifest.json",
    "strong_backbone_fixed8_bn_frozen_*/*.manifest.json",
    "matched_agegap_arms/*.manifest.json",
    "matched_agegap_fixed10_last/*.manifest.json",
)


def _resolve_record(record: dict[str, Any]) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _private_record(record: dict[str, Any]) -> bool:
    path = str(record.get("path", "")).replace("\\", "/").lower()
    return (path.startswith("data/interim/faces/") or "/private/" in path
            or Path(path).suffix in {".jpg", ".jpeg", ".png", ".npy", ".npz"}
            or bool(PRIVATE_ROW_ID.search(path)))


def _record_integrity(records: list[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    missing: list[str] = []
    mismatched: list[str] = []
    private_missing = private_mismatched = 0
    for record in records:
        path = _resolve_record(record)
        if not path.is_file():
            if _private_record(record):
                private_missing += 1
            else:
                missing.append(str(record["path"]))
            continue
        checked += 1
        if sha256_file(path) != record.get("sha256"):
            if _private_record(record):
                private_mismatched += 1
            else:
                mismatched.append(str(record["path"]))
    return {
        "checked": checked,
        "missing": missing,
        "checksum_mismatch": mismatched,
        "private_missing_count": private_missing,
        "private_checksum_mismatch_count": private_mismatched,
        "valid": not missing and not mismatched and private_missing == 0 and private_mismatched == 0,
    }


def _public_value(value: Any) -> Any:
    """Remove machine/user-specific absolute paths from the publication index."""
    if isinstance(value, dict):
        return {key: _public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    if isinstance(value, str):
        if PRIVATE_ROW_ID.search(value):
            return "<private-row-artifact>"
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            except ValueError:
                return f"<external-artifact>/{candidate.name}"
    return value


def _public_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep private per-crop identifiers/checksums in local manifests, not the upload index."""
    public = [record for record in records if not _private_record(record)]
    return _public_value(public), len(records) - len(public)


def main() -> None:
    expected_literal: list[str] = []
    manifest_paths: set[Path] = set()
    for pattern in INCLUDE:
        matches = set(METRICS.glob(pattern))
        if not any(char in pattern for char in "*?["):
            expected_literal.append(pattern)
        manifest_paths.update(matches)

    entries: list[dict[str, Any]] = []
    for path in sorted(manifest_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        public_inputs, private_input_count = _public_records(payload.get("inputs", []))
        public_outputs, private_output_count = _public_records(payload.get("outputs", []))
        entry = {
                "manifest": path.relative_to(PROJECT_ROOT).as_posix(),
                "manifest_sha256": sha256_file(path),
                "experiment": payload.get("experiment"),
                "created_at_utc": payload.get("created_at_utc"),
                "command": _public_value(payload.get("command")),
                "parameters": _public_value(payload.get("parameters")),
                "git": payload.get("git"),
                "inputs": public_inputs,
                "outputs": public_outputs,
                "private_input_count": private_input_count,
                "private_output_count": private_output_count,
                "input_integrity": _record_integrity(payload.get("inputs", [])),
                "output_integrity": _record_integrity(payload.get("outputs", [])),
            }
        entries.append(entry)

    found_names = {path.relative_to(METRICS).as_posix() for path in manifest_paths}
    missing_expected = sorted(name for name in expected_literal if name not in found_names)
    invalid = [
        entry["manifest"]
        for entry in entries
        if not entry["input_integrity"]["valid"] or not entry["output_integrity"]["valid"]
    ]
    incomplete_experiments: list[str] = []
    for evaluation_dir in sorted(METRICS.glob("lfw_bound_evaluation_*")):
        if not ((evaluation_dir / "lfw_bound_evaluation.json").is_file()
                and (evaluation_dir / "lfw_bound_evaluation.manifest.json").is_file()):
            incomplete_experiments.append(f"{evaluation_dir.name}: completed result or manifest missing")
    for campaign_dir in sorted(METRICS.glob("strong_backbone_fixed8_bn_frozen_*")):
        corrected_summary = campaign_dir / "summary.json"
        if not corrected_summary.is_file():
            incomplete_experiments.append(f"{campaign_dir.name}: corrected campaign summary missing")
            continue
        corrected_campaign = json.loads(corrected_summary.read_text(encoding="utf-8")).get("campaign", {})
        if not corrected_campaign.get("complete", False):
            incomplete_experiments.append(
                f"{campaign_dir.name}: incomplete ({corrected_campaign.get('completed_runs', 0)}/"
                f"{corrected_campaign.get('expected_runs', 36)} runs)"
            )
    strong_summary = METRICS / "strong_backbone_study" / "summary.json"
    if not strong_summary.is_file():
        incomplete_experiments.append("strong-backbone summary missing")
    else:
        strong_payload = json.loads(strong_summary.read_text(encoding="utf-8"))
        if not strong_payload.get("campaign", {}).get("complete", False):
            completed = strong_payload.get("campaign", {}).get("completed_runs", 0)
            expected = strong_payload.get("campaign", {}).get("expected_runs", 36)
            incomplete_experiments.append(
                f"strong-backbone campaign incomplete ({completed}/{expected} runs)"
            )
    matched_summary = METRICS / "matched_agegap_arms" / "summary.json"
    if matched_summary.is_file():
        matched_payload = json.loads(matched_summary.read_text(encoding="utf-8"))
        if not matched_payload.get("complete", False):
            completed = matched_payload.get("completed_run_count", 0)
            expected = matched_payload.get("expected_run_count", 0)
            incomplete_experiments.append(
                f"matched-age-gap campaign incomplete ({completed}/{expected} runs)"
            )
    fixed_summary = METRICS / "matched_agegap_fixed10_last" / "summary.json"
    if fixed_summary.is_file():
        fixed_payload = json.loads(fixed_summary.read_text(encoding="utf-8"))
        if not fixed_payload.get("complete", False):
            completed = fixed_payload.get("completed_run_count", 0)
            expected = fixed_payload.get("expected_run_count", 0)
            incomplete_experiments.append(
                f"fixed-epoch matched-age-gap campaign incomplete ({completed}/{expected} runs)"
            )
    payload = {
        "schema_version": 1,
        "scope": "T-BIOM publication experiments only; no face data or biometric templates",
        "entry_count": len(entries),
        "missing_expected_manifests": missing_expected,
        "invalid_input_or_output_manifests": invalid,
        "incomplete_experiments": incomplete_experiments,
        "complete": not missing_expected and not invalid and not incomplete_experiments,
        "entries": entries,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"wrote {OUTPUT}: entries={len(entries)}, missing={len(missing_expected)}, "
        f"invalid={len(invalid)}, incomplete={len(incomplete_experiments)}, "
        f"complete={payload['complete']}"
    )


if __name__ == "__main__":
    main()
