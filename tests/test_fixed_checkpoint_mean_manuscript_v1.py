from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "metrics/fixed_checkpoint_means_presentation_v1_e362_20261009"
SUPPLEMENT = ROOT / "latex/papers/journal-1-tbiom/en/supplement.tex"


def test_both_accepted_native_tables_are_exactly_embedded():
    source = SUPPLEMENT.read_text(encoding="utf-8")
    section = source.split("% BEGIN GENERATED FIXED CHECKPOINT MEANS V1", 1)[1].split(
        "% END GENERATED FIXED CHECKPOINT MEANS V1", 1)[0]
    for stratum in ("overall", "large_gap_25plus"):
        assert (BASE / f"{stratum}_table.tex").read_text(encoding="utf-8").strip() in section
    assert section.count("Head,") == section.count("Tail,") == 20


def test_scopes_keep_twelve_means_separate_from_eleven_and_ten_snapshots():
    text = " ".join(SUPPLEMENT.read_text(encoding="utf-8").split())
    for phrase in ("twelve-checkpoint subset", "preceding eleven-cell per-checkpoint/feature snapshot",
                   "ten-history subset", "not a score ensemble", "SD is descriptive checkpoint variation",
                   "without multiplicity correction", "Both low-LR 25+ AUC-change intervals include zero"):
        assert phrase in text


def test_no_patch_marker_or_duplicate_labels_in_source():
    source = SUPPLEMENT.read_text(encoding="utf-8")
    assert not any(line.startswith("+\\") for line in source.splitlines())
    for label in ("sec:strong-fixed-means", "tab:strong-fixed-mean-overall", "tab:strong-fixed-mean-25plus"):
        assert source.count("\\label{" + label + "}") == 1
