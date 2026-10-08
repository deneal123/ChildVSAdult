"""Partial feature tables are traceable, denominator-consistent and scoped."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.render_age_baseline_points_v3 import table as age_table
from scripts.render_common_mechanism_v2 import strong_table, weak_table

ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = ROOT / "latex/papers/journal-1-tbiom/en/supplement.tex"
FEATURE = ROOT / "metrics/common_mechanism_presentation9_v2_20261008"
AGE = ROOT / "metrics/age_probe_absolute_presentation14_v3_full_20261008"
ROC = ROOT / "metrics/strong_fixed8_roc_presentation9_v1_20261008"


def bound_summary(directory):
    manifest = json.loads((directory / "presentation.manifest.json").read_text(encoding="utf-8"))
    for name in ("summary.json", *sorted(p.name for p in directory.glob("*.tex"))):
        path = directory / name
        records = [row for row in manifest["outputs"]
                   if Path(row["path"]).name == name]
        assert len(records) == 1
        recorded_path = Path(records[0]["path"])
        if not recorded_path.is_absolute():
            recorded_path = ROOT / recorded_path
        assert recorded_path.resolve() == path.resolve()
        assert records[0]["bytes"] == path.stat().st_size
        assert records[0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert summary == manifest["metrics"]
    return summary


@pytest.mark.parametrize("corruption", ["bytes", "path"])
def test_summary_guard_rejects_wrong_size_or_same_basename_elsewhere(tmp_path, corruption):
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    record = {"path": str(summary), "bytes": summary.stat().st_size,
              "sha256": hashlib.sha256(summary.read_bytes()).hexdigest()}
    if corruption == "bytes":
        record["bytes"] += 1
    else:
        record["path"] = str(tmp_path / "unrelated" / "summary.json")
    (tmp_path / "presentation.manifest.json").write_text(
        json.dumps({"outputs": [record], "metrics": {}}), encoding="utf-8")
    with pytest.raises(AssertionError):
        bound_summary(tmp_path)


@pytest.mark.parametrize("directory,filename,render,marker", [
    (FEATURE, "strong_table.tex", strong_table, "COMMON FEATURE V2"),
    (FEATURE, "weak_table.tex", weak_table, "COMMON FEATURE V2"),
    (AGE, "age_baselines_table.tex", age_table, "ABSOLUTE AGE POINTS V3"),
])
def test_native_tables_unchanged_except_explicit_tabular_fit_wrapper(directory, filename, render, marker):
    source = SUPPLEMENT.read_text(encoding="utf-8")
    block = source.split(f"% BEGIN GENERATED {marker}", 1)[1].split(
        f"% END GENERATED {marker}", 1)[0]
    # Format-only wrappers apply to tabular, never rewrite cells or captions.
    normalized = block.replace("\\resizebox{\\textwidth}{!}{%\n", "").replace(
        "\\end{tabular}%\n}\n", "\\end{tabular}\n")
    native = (directory / filename).read_text(encoding="utf-8")
    assert render(bound_summary(directory)) == native
    assert native.strip() in normalized


def test_person_age_changes_equal_absolute_points_not_image_weighted_points():
    feature, age = bound_summary(FEATURE), bound_summary(AGE)
    missing = set(feature["strong"]["diagnostics"]) - {
        key.removeprefix("strong/") for key in age["models"] if key.startswith("strong/")
    }
    assert missing == set()
    for family, diagnostics in (("strong", feature["strong"]["diagnostics"]),
                                ("weak", feature["weak"]["diagnostics"])):
        frozen = age["models"][family + "/frozen"]
        for name, row in diagnostics.items():
            tuned = age["models"][family + "/" + name]
            person_delta = tuned["mae_person"] - frozen["mae_person"]
            assert person_delta == pytest.approx(row["paired_age_error"]["delta_mae_person"], abs=1e-12)
    assert age["native_probe_records"] == 40
    assert age["unique_model_points"] == 14


def test_descriptive_interval_interpretation_matches_all_rows():
    summary = bound_summary(FEATURE)
    assert (summary["n_images"], summary["n_pairs"], summary["n_persons"]) == (650, 5308, 82)
    assert (summary["observed_strong_cells"], summary["expected_strong_matrix_cells"]) == (9, 36)
    assert summary["mechanism_complete"] is False
    assert summary["publication_ready"] is False
    for name, row in summary["strong"]["cells"].items():
        assert row["delta_ci95"][1] < 0
        lo, hi = summary["strong"]["diagnostics"][name]["paired_age_error"]["ci95"]
        if "_tail_" in name or name in {"random_head_lr1e-05_s1", "random_head_lr1e-05_s2"}:
            assert hi < 0
        else:
            assert lo <= 0 <= hi
    for name, row in summary["weak"]["diagnostics"].items():
        assert summary["weak"]["models"][name]["delta_ci95"][0] > 0
        lo, hi = row["paired_age_error"]["ci95"]
        assert lo <= 0 <= hi
    source = SUPPLEMENT.read_text(encoding="utf-8")
    for limit in ("Lower age MAE\nmeans easier age decoding, not age removal",
                  "does not prove\nabsence of age information", "no multiplicity correction",
                  "not human identity clearance", "unverified weak training history",
                  "not intervals of seed means", "the full matrix and\nbroader external checks remain open"):
        assert limit in source


def test_absolute_point_denominators_and_constant_claim_are_not_significance():
    summary = bound_summary(AGE)
    frozen = summary["models"]["strong/frozen"]
    for denominator in ("mae_image", "mae_person"):
        assert all(frozen[denominator] > row[denominator] for row in summary["baselines"].values())
    source = SUPPLEMENT.read_text(encoding="utf-8")
    assert "No new absolute-error interval or significance\ntest is supplied" in source
    assert "separates image-weighted and\nequal-person MAE" in source


def test_feature_age_and_roc_share_exact_declared_nine_cells():
    expected = {f"random_{scope}_lr1e-06_s{seed}"
                for scope in ("head", "tail") for seed in (42, 1, 2)}
    expected |= {f"random_head_lr1e-05_s{seed}" for seed in (42, 1, 2)}
    feature, age, roc = bound_summary(FEATURE), bound_summary(AGE), bound_summary(ROC)
    assert set(feature["strong"]["cells"]) == expected
    assert set(feature["strong"]["diagnostics"]) == expected
    assert {key.removeprefix("strong/") for key in age["models"]
            if key.startswith("strong/") and key != "strong/frozen"} == expected
    assert set(roc["cells"]) == expected
    source = SUPPLEMENT.read_text(encoding="utf-8")
    assert "describe nine AdaFace checkpoints" in source
    assert "the same nine strong checkpoints" in source
    assert "fourteen unique model points" in source
