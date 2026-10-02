from __future__ import annotations

import hashlib
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "latex/papers/journal-1-tbiom/en/main.tex"
SUPPLEMENT = ROOT / "latex/papers/journal-1-tbiom/en/supplement.tex"


def _latex_int(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def _fmt3(value: float) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _fmt_decimal(value: float, decimals: int) -> str:
    quantum = Decimal(1).scaleb(-decimals)
    return f"{Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP):.{decimals}f}"


def test_age_audit_table_matches_machine_readable_artifact() -> None:
    metrics = json.loads(
        (ROOT / "metrics/age_extraction_audit.json").read_text(encoding="utf-8")
    )
    main = MAIN.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    rows = [
        (metrics["agree_empty"] + metrics["agree_ages"], metrics["agreement_pct"]),
        (metrics["llm_filled"], metrics["llm_filled_pct"]),
        (metrics["age_mismatch"], metrics["age_mismatch_pct"]),
        (metrics["llm_missed"], metrics["llm_missed_pct"]),
    ]

    for count, percentage in rows:
        assert _latex_int(count) in main
        assert f"{percentage:.1f}\\%" in main
    for percentage in (
        metrics["agreement_pct"],
        metrics["age_mismatch_pct"],
        metrics["llm_filled_pct"],
        metrics["llm_missed_pct"],
    ):
        assert f"{percentage:.1f}\\%" in supplement
    assert (ROOT / "metrics/age_extraction_audit.manifest.json").is_file()


def test_group_integrity_table_matches_machine_readable_artifact() -> None:
    metrics = json.loads(
        (ROOT / "metrics/group_integrity_audit.json").read_text(encoding="utf-8")
    )
    main = MAIN.read_text(encoding="utf-8")
    expected = {
        "single": metrics["categories"]["single"],
        "multi_person": metrics["categories"]["multi_person"],
        "unknown": metrics["categories"]["unknown"],
        "meme_or_collage": metrics["categories"]["meme"]
        + metrics["categories"]["collage"],
    }

    for count in expected.values():
        assert _latex_int(count) in main
    assert (ROOT / "metrics/group_integrity_audit.manifest.json").is_file()


def test_dataset_scale_table_matches_funnel_and_dedup_artifacts() -> None:
    funnel = json.loads((ROOT / "metrics/data_funnel.json").read_text(encoding="utf-8"))["funnel"]
    dedup = json.loads((ROOT / "metrics/sensitivity_dedup.json").read_text(encoding="utf-8"))
    main = MAIN.read_text(encoding="utf-8")
    expected = (
        funnel["posts"],
        funnel["photos"],
        funnel["face_records"],
        funnel["usable_faces"],
        funnel["person_clusters"],
        dedup["no_dedup_positives"],
        funnel["curated_faces"],
        funnel["identity_groups"],
        funnel["positive_pairs"],
    )

    for count in expected:
        assert _latex_int(count) in main
    assert (ROOT / "metrics/data_funnel.manifest.json").is_file()
    assert (ROOT / "metrics/sensitivity_dedup.manifest.json").is_file()


def test_dedup_sensitivity_table_matches_machine_readable_artifact() -> None:
    metrics = json.loads((ROOT / "metrics/sensitivity_dedup.json").read_text(encoding="utf-8"))
    main = MAIN.read_text(encoding="utf-8")

    for threshold in ("0.93", "0.95", "0.97", "0.99"):
        row = metrics["thresholds"][threshold]
        assert threshold in main
        assert _latex_int(row["redundant_faces"]) in main
        assert _latex_int(row["positives"]) in main
        assert f"{row['reduction_pct']:.1f}\\%" in main


def test_objective_table_matches_machine_readable_artifact() -> None:
    metrics = json.loads((ROOT / "metrics/sota_objectives.json").read_text(encoding="utf-8"))
    main = MAIN.read_text(encoding="utf-8")

    for model in ("frozen", "+pairs", "+arcface", "+cosface", "+sphereface"):
        row = metrics[model]
        for key in ("fgnet.large_gap", "our.25+", "LFW.acc", "agedb_30.roc"):
            assert f"{row[key]:.3f}" in main
    assert (ROOT / "metrics/sota_objectives.manifest.json").is_file()


def test_hard_negative_table_matches_machine_readable_artifact() -> None:
    metrics = json.loads((ROOT / "metrics/hardneg_multiseed.json").read_text(encoding="utf-8"))
    supplement = SUPPLEMENT.read_text(encoding="utf-8")

    for key in ("fgnet.large_gap", "fgnet.roc", "LFW.acc", "agedb_30.roc", "calfw.roc"):
        assert f"{metrics['mean'][key]:.3f}" in supplement
        assert f"{metrics['std'][key]:.3f}" in supplement
    assert (ROOT / "metrics/hardneg_multiseed.manifest.json").is_file()


def test_operating_point_table_matches_machine_readable_artifact() -> None:
    metrics = json.loads((ROOT / "metrics/headline_stats.json").read_text(encoding="utf-8"))
    main = MAIN.read_text(encoding="utf-8")

    for benchmark in ("fgnet.large_gap", "fgnet.overall", "AgeDB-30", "CALFW", "LFW"):
        frozen = metrics[benchmark]["frozen"]
        tuned = metrics[benchmark]["+pairs"]
        for value in (
            frozen["auc"],
            tuned["auc"],
            frozen["eer"],
            tuned["eer"],
            frozen["tar@far1e-2"],
            tuned["tar@far1e-2"],
            frozen["tar@far1e-3"],
            tuned["tar@far1e-3"],
        ):
            assert _fmt3(value) in main
    assert (ROOT / "metrics/headline_stats.manifest.json").is_file()


def test_synthetic_parity_table_matches_machine_readable_artifact() -> None:
    metrics = json.loads((ROOT / "metrics/synthetic_parity.json").read_text(encoding="utf-8"))
    main = MAIN.read_text(encoding="utf-8")

    for model in ("frozen", "+real", "+syn_fran_fixed", "+syn_fran_parity"):
        for key in ("fgnet.large_gap", "our.25+"):
            assert _fmt3(metrics["models"][model][key]) in main
    assert f"median {metrics['gap_stats']['median']:.0f}~y" in main
    assert (ROOT / "metrics/synthetic_parity.manifest.json").is_file()


def test_noise_robust_objective_row_matches_machine_readable_artifact() -> None:
    metrics = json.loads(
        (ROOT / "metrics/noise_robust_baseline.json").read_text(encoding="utf-8")
    )
    main = MAIN.read_text(encoding="utf-8")

    for key in ("fgnet.large_gap", "our.25+", "LFW.acc", "agedb_30.roc"):
        assert _fmt3(metrics["+arcface_sc3"][key]) in main
    assert (ROOT / "metrics/noise_robust_baseline.manifest.json").is_file()


def test_lookalike_table_matches_machine_readable_artifacts() -> None:
    full = json.loads((ROOT / "metrics/hardneg_curve_full.json").read_text(encoding="utf-8"))
    facenet = json.loads(
        (ROOT / "metrics/hardneg_curve_facenet_clean.json").read_text(encoding="utf-8")
    )
    main = MAIN.read_text(encoding="utf-8")
    tags = ("random", "rank50", "rank10", "rank1")

    for model in ("adaface_ir101", "arcface_r100", "adaface_ir50"):
        for tag in tags:
            assert _fmt3(full[model][f"{tag} | 25+"]["auc"]) in main
    for model in ("facenet", "facenet:tuned"):
        for tag in tags:
            assert _fmt3(facenet[model][f"{tag} | 25+"]["auc"]) in main
    assert "1208" in main
    assert r"31\,590" in main
    assert "141" in main
    assert "4762" in main
    for name in (
        "hardneg_curve_full.manifest.json",
        "hardneg_curve_facenet_clean.manifest.json",
        "hardneg_curve_miner2.manifest.json",
        "hardneg_paired_pairs_clean.manifest.json",
    ):
        assert (ROOT / "metrics" / name).is_file()


def test_publication_artifact_index_has_no_checksum_failures() -> None:
    index = json.loads(
        (ROOT / "metrics/publication_artifact_index.json").read_text(encoding="utf-8")
    )

    assert index["entry_count"] >= 45
    assert index["invalid_input_or_output_manifests"] == []
    assert re.search(r"(?:vk_|reddit_)?-?\d{5,}_\d{5,}", json.dumps(index)) is None
    assert "data/interim/faces/" not in json.dumps(index)
    human_manifest = ROOT / "metrics/human_audit.manifest.json"
    assert ("human_audit.manifest.json" in index["missing_expected_manifests"]) != (
        human_manifest.is_file()
    )
    strong_summary = json.loads(
        (ROOT / "metrics/strong_backbone_study/summary.json").read_text(encoding="utf-8")
    )
    strong_manifest = json.loads(
        (ROOT / "metrics/strong_backbone_study/summary.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    strong_inputs = {record["path"] for record in strong_manifest["inputs"]}
    assert "models/adaface_ir101.pt" in strong_inputs
    assert "metrics/model_inventory.json" in strong_inputs
    provenance_required = {"models/adaface_ir101.pt", "metrics/model_inventory.json"}
    strong_dir = ROOT / "metrics/strong_backbone_study"
    for manifest_path in [
        *strong_dir.glob("run_*.manifest.json"),
        *strong_dir.glob("frozen_*.manifest.json"),
    ]:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        input_paths = {record["path"] for record in manifest_payload["inputs"]}
        assert provenance_required <= input_paths
    strong_gate_present = any(
        item.startswith("strong-backbone campaign incomplete")
        for item in index["incomplete_experiments"]
    )
    assert strong_gate_present != strong_summary["campaign"]["complete"]
    assert all(entry["input_integrity"]["valid"] for entry in index["entries"])
    assert all(entry["output_integrity"]["valid"] for entry in index["entries"])
    assert all("metrics" not in entry for entry in index["entries"])
    for entry in index["entries"]:
        manifest = ROOT / entry["manifest"]
        assert manifest.is_file()
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == entry["manifest_sha256"]
        for record in [*entry["inputs"], *entry["outputs"]]:
            if str(record["path"]).startswith("<external-artifact>/"):
                continue
            artifact = ROOT / record["path"]
            assert artifact.is_file()
            assert artifact.stat().st_size == record["bytes"]
            assert hashlib.sha256(artifact.read_bytes()).hexdigest() == record["sha256"]
    serialized = json.dumps(index).lower()
    assert "c:\\\\users\\\\" not in serialized
    assert "r:\\\\" not in serialized
    assert "/home/" not in serialized


def test_cacd_vs_results_match_machine_readable_artifacts() -> None:
    frozen = json.loads(
        (ROOT / "metrics/cacd_vs/facenet_frozen.json").read_text(encoding="utf-8")
    )["metrics"]
    tuned = json.loads(
        (ROOT / "metrics/cacd_vs/fine_tuned_summary.json").read_text(encoding="utf-8")
    )["metrics"]
    main = MAIN.read_text(encoding="utf-8")

    assert f"{frozen['accuracy_10fold']:.4f}" in main
    historical = SUPPLEMENT.read_text(encoding="utf-8")
    assert f"{frozen['roc_auc']:.4f}" in historical
    assert f"{frozen['eer']:.4f}" in main
    for key in ("accuracy_10fold", "eer"):
        assert f"{tuned[key]['mean']:.4f}" in main
    assert f"{tuned['roc_auc']['mean']:.4f}" in historical


def test_common_protocol_sota_table_matches_machine_readable_artifacts() -> None:
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    for method in ("mtlface", "cacon"):
        summary = json.loads(
            (
                ROOT
                / f"metrics/sota_common_protocol/{method}_arcface_r50_casia_e8_summary.json"
            ).read_text(encoding="utf-8")
        )
        for key in (
            "LFW.acc",
            "agedb_30.roc",
            "calfw.roc",
            "cacd_vs.roc",
            "fgnet.roc",
            "fgnet.large_gap",
            "our.25+",
        ):
            mean = f"{summary['metrics'][key]['mean']:.4f}".removeprefix("0")
            std = f"{summary['metrics'][key]['std']:.4f}".removeprefix("0")
            assert f"{mean}$\\pm${std}" in supplement


def test_bidirectional_cross_source_table_matches_machine_readable_artifacts() -> None:
    vk_to_reddit = json.loads(
        (ROOT / "metrics/cross_platform_vk2reddit_3seed.json").read_text(encoding="utf-8")
    )
    reddit_to_vk = json.loads(
        (ROOT / "metrics/cross_source_reddit2vk_3seed.json").read_text(encoding="utf-8")
    )
    main = MAIN.read_text(encoding="utf-8")

    assert _fmt3(vk_to_reddit["models"]["frozen"]["overall_auc"]) in main
    for key in ("overall_auc", "eer"):
        row = vk_to_reddit["fine_tuned_aggregate"][key]
        assert _fmt3(row["mean"]) in main
        assert _fmt3(row["std"]) in main
    for key in ("fgnet.large_gap", "our.25+"):
        frozen = reddit_to_vk["frozen"][key]
        tuned = reddit_to_vk["fine_tuned_aggregate"][key]
        assert _fmt3(frozen) in main
        assert _fmt3(tuned["mean"]) in main
        assert _fmt3(tuned["std"]) in main
    assert (ROOT / "metrics/cross_platform_vk2reddit_3seed.manifest.json").is_file()
    assert (ROOT / "metrics/cross_source_reddit2vk_3seed.manifest.json").is_file()


def test_quality_distribution_table_matches_data_funnel_artifact() -> None:
    metrics = json.loads((ROOT / "metrics/data_funnel.json").read_text(encoding="utf-8"))
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    distributions = metrics["retained_vs_rejected"]
    table_fields = {
        "image_width": 0,
        "face_width": 0,
        "blur_var": 0,
        "det_score": 3,
        "face_quality_score": 3,
        "pose_yaw_proxy_abs": 3,
    }

    for field, decimals in table_fields.items():
        for population in ("retained", "rejected"):
            row = distributions[field][population]
            for statistic in ("median", "p05", "p95"):
                assert _fmt_decimal(row[statistic], decimals) in supplement
    for value in metrics["apparent_age"]["p05_p25_median_p75_p95"]:
        assert f"{value:.0f}" in supplement
    for count in metrics["positive_age_gap"]["buckets"].values():
        assert _latex_int(count) in supplement


def test_human_audit_pack_claims_match_validator_artifact() -> None:
    metrics = json.loads(
        (ROOT / "metrics/human_audit_pack_validation.json").read_text(encoding="utf-8")
    )
    supplement = SUPPLEMENT.read_text(encoding="utf-8")

    assert f"{metrics['tasks_per_annotator']:,}" in supplement
    assert f"{metrics['opaque_image_files']:,}" in supplement
    assert all(count == 400 for count in metrics["counts_by_type"].values())
    assert metrics["source_identifier_leaks"] == 0
    assert metrics["automatic_decision_leaks"] == 0
    assert metrics["image_checksum_mismatches"] == 0
    assert metrics["repeated_source_occurrences"] == 0
    assert metrics["automatic_label_counts_by_type"]["near_duplicate"] == {
        "distinct_photo": 128,
        "near_duplicate": 272,
    }
    assert metrics["automatic_label_counts_by_type"]["group_integrity"] == {
        "different_identity": 200,
        "same_identity": 200,
    }
    assert metrics["near_duplicate_similarity"]["min"] < 0.97
    assert metrics["near_duplicate_similarity"]["max"] >= 0.97
    assert f"{100 * metrics['age_baseline_coverage']['regex']:.1f}\\%" in supplement
    assert metrics["age_baseline_coverage"]["gigachat"] == 1.0
    assert metrics["age_baseline_coverage"]["local"] == 1.0
    assert (ROOT / "metrics/human_audit_pack_validation.manifest.json").is_file()


def test_age_shortcut_claim_is_bounded_to_the_weak_backbone_probe() -> None:
    main = MAIN.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")

    assert "mechanistically tied to removing an age shortcut" not in main
    assert "The mechanism is removal of an age shortcut" not in supplement
    assert "overstated failure of the frozen model" in main
    assert "not elimination of an age shortcut" in main
    assert "low-FAR TAR declines" in main


def test_endpoint_matched_internal_claim_tracks_artifact() -> None:
    result = json.loads(
        (ROOT / "metrics/internal_endpoint_age_matched.json").read_text(encoding="utf-8")
    )
    comparison = result["model_comparison_same_matched_subset"]
    interval = result["facenet_paired_subject_bootstrap"]["ci95"]["delta_comparison_minus_reference"]
    main = MAIN.read_text(encoding="utf-8")

    assert result["diagnostics"]["matched_positive_count"] == 154
    assert result["diagnostics"]["age_gap_only_predictive_auc"] < 0.501
    assert comparison["facenet_tuned"]["roc_auc"] > comparison["facenet_frozen"]["roc_auc"]
    assert comparison["facenet_tuned"]["tar@far=0.01"] < comparison["facenet_frozen"]["tar@far=0.01"]
    for value in (
        comparison["facenet_frozen"]["roc_auc"],
        comparison["facenet_tuned"]["roc_auc"],
        *interval,
    ):
        assert _fmt3(value) in main
    assert (ROOT / "metrics/internal_endpoint_age_matched.manifest.json").is_file()


def test_endpoint_age_matched_fgnet_claim_tracks_subject_artifact() -> None:
    result = json.loads(
        (ROOT / "metrics/fgnet_endpoint_subject_stats.json").read_text(encoding="utf-8")
    )
    row = result["large_gap_25plus"]
    interval = row["subject_bootstrap"]
    construction = result["negative_construction"]
    main = MAIN.read_text(encoding="utf-8")

    assert result["protocol"] == "endpoint_age_matched"
    assert row["class_counts"] == {"positive": 215, "negative": 215}
    assert row["observed_age_gap"]["gap_only_auc"] == 0.5
    assert row["endpoint_age_only_auc"] == {"age_a": 0.5, "age_b": 0.5}
    assert construction["large_gap_positive_coverage"] == 1.0
    assert construction["all_negatives_have_different_subjects"]
    multiseed = json.loads(
        (ROOT / "metrics/fgnet_endpoint_multiseed.json").read_text(encoding="utf-8")
    )
    assert multiseed["seeds"] == [42, 1, 2]
    assert "Endpoint-age-matched protocol check (three seeds)" in main
    assert f"{multiseed['strata']['large_gap_25plus']['tuned_auc_mean']:.4f}" in main
    assert f"{multiseed['strata']['large_gap_25plus']['delta_auc_mean']:+.4f}" in main
    abstract = main.partition(r"\begin{abstract}")[2].partition(r"\end{abstract}")[0]
    assert "endpoint-age-matched FG-NET protocol" in abstract
    assert "0.8490" in abstract
    assert "original random-impostor protocol" in abstract
    for value in (
        interval["frozen_auc"],
        interval["tuned_auc"],
        interval["delta_auc"],
        *interval["delta_ci95"],
    ):
        assert _fmt3(value) in main
    assert (ROOT / "metrics/fgnet_endpoint_subject_stats.manifest.json").is_file()
    assert (ROOT / "metrics/fgnet_endpoint_multiseed.manifest.json").is_file()
