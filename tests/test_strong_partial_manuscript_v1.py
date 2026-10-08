"""Generated strong tables must remain verbatim and explicitly partial."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "latex/papers/journal-1-tbiom/en"
EVIDENCE = ROOT / "metrics/strong_fixed8_roc_presentation9_v1_20261008"


def test_presentation_outputs_are_bound_and_metrics_match_native_summary():
    manifest = json.loads((EVIDENCE / "presentation.manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((EVIDENCE / "summary.json").read_text(encoding="utf-8"))
    assert manifest["metrics"] == summary
    for name in ("summary.json", "overall_table.tex", "large_gap_25plus_table.tex"):
        path = EVIDENCE / name
        records = [record for record in manifest["outputs"]
                   if Path(record["path"]).name == name]
        assert len(records) == 1
        assert records[0]["bytes"] == path.stat().st_size
        assert records[0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_both_generated_strong_tables_inserted_verbatim():
    source = (PAPER / "supplement.tex").read_text(encoding="utf-8")
    block = source.split("% BEGIN GENERATED STRONG FIXED8 ROC V1", 1)[1].split(
        "% END GENERATED STRONG FIXED8 ROC V1", 1)[0]
    for stratum in ("overall", "large_gap_25plus"):
        table = (EVIDENCE / f"{stratum}_table.tex").read_text(encoding="utf-8").strip()
        assert table in block
        assert "not a seed mean or an ensemble" in table
        assert "No training-seed population inference" in table
        assert "no multiplicity correction" in table


def test_partial_scope_and_shared_counts_match_presentation():
    summary = json.loads((EVIDENCE / "summary.json").read_text(encoding="utf-8"))
    source = (PAPER / "supplement.tex").read_text(encoding="utf-8")
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert summary["evaluated_checkpoint_count"] == 9
    assert summary["expected_matrix_cells"] == 36
    assert summary["mechanism_complete"] is False
    assert summary["publication_ready"] is False
    assert summary["protocols"]["overall"] == {"n_pairs": 5308, "n_subjects": 82}
    assert summary["protocols"]["large_gap_25plus"] == {"n_pairs": 430, "n_subjects": 75}
    assert "they are not the full matrix" in source
    assert "from the same 650 images" in source
    assert "These strata share FG-NET" in source
    assert "identity independence remains\nunverified" in source
    assert "Partial Fixed8 AdaFace ROC-v2" in main
    assert "Neither result completes the planned fixed-budget" in main


def test_all_six_overall_operating_losses_reported_without_25plus_gain_claim():
    summary = json.loads((EVIDENCE / "summary.json").read_text(encoding="utf-8"))
    for name, cell in summary["cells"].items():
        if "_lr1e-06_" not in name:
            continue
        assert cell["overall"]["eer_interpolated"]["delta_ci95"][0] > 0
        for metric in ("tar@far=0.01", "tar@far=0.001"):
            assert cell["overall"][metric]["delta_ci95"][1] < 0
        for metric in cell["large_gap_25plus"].values():
            assert metric["delta_ci95"][0] <= 0 <= metric["delta_ci95"][1]
    source = (PAPER / "supplement.tex").read_text(encoding="utf-8")
    assert "all 25+ change intervals include zero for those six cells" in source


def test_high_lr_head_losses_and_seed2_low_far_uncertainty_are_reported():
    summary = json.loads((EVIDENCE / "summary.json").read_text(encoding="utf-8"))
    for seed in (42, 1, 2):
        cell = summary["cells"][f"random_head_lr1e-05_s{seed}"]["large_gap_25plus"]
        assert cell["roc_auc"]["delta_ci95"][1] < 0
        assert cell["eer_interpolated"]["delta_ci95"][0] > 0
        assert cell["tar@far=0.001"]["delta_ci95"][1] < 0
        lo, hi = cell["tar@far=0.01"]["delta_ci95"]
        if seed == 2:
            assert lo <= 0 <= hi
        else:
            assert hi < 0
    source = (PAPER / "supplement.tex").read_text(encoding="utf-8")
    assert "includes zero for seed2" in source
    assert "not intervals of seed means" in source
