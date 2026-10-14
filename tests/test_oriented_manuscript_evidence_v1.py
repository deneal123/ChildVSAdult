"""Regression link for current LOW/CROSS and fixed-permutation manuscript numbers.

Checks the bound summaries and presentation, not all original inputs or human labels.
"""

import json

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record


def result(folder, experiment):
    base = PROJECT_ROOT / "metrics" / folder
    summary = base / "summary.json"
    native = json.loads((base / "summary.manifest.json").read_text(encoding="utf-8"))
    assert native["experiment"] == experiment
    assert file_record(summary) in native["outputs"]
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data == native["metrics"]
    assert data["execution_complete"] is True
    return data


def documents():
    base = PROJECT_ROOT / "latex/papers/journal-1-tbiom/en"
    return tuple(
        (base / name).read_text(encoding="utf-8") for name in ("main.tex", "supplement.tex")
    )


def test_replacement_headline_matches_bound_metrics():
    data = result("oriented_cuda_fgnet_20261004", "oriented-cuda-controls-full-fgnet-roc-v2")
    main, supplement = documents()
    paragraph = main.split(r"\textbf{Within-source age-gap control.}", 1)[1].split("\n\n", 1)[0]
    metric = data["large_gap_25plus"]["metrics"]["roc_auc"]
    delta = metric["cross_minus_low"]
    for value in (metric["low"]["mean"], metric["cross"]["mean"], metric["frozen"]["point"]):
        assert f"{value:.4f}" in paragraph
    assert f"{delta['mean']:+.4f}" in paragraph
    lo, hi = delta["mean_checkpoint_ci95"]
    assert f"[{lo:+.4f},{hi:+.4f}]" in paragraph
    assert "its advantage is not established" in paragraph
    claim = supplement.split("The full-image-budget within-VK control", 1)[1].split(r"\\", 1)[0]
    assert f"{delta['mean']:+.4f}" in claim
    assert f"[{lo:+.4f},{hi:+.4f}]" in claim


def test_noise_headline_and_supplement_match_bound_metrics():
    data = result("oriented_noise_cuda_fgnet_20261004", "oriented-cuda-noise-full-fgnet-roc-v2")
    assert (data["direction"], data["low_role"], data["cross_role"], data["permutation_seed"]) == (
        "noise_minus_clean",
        "clean CROSS",
        "noisy CROSS",
        42,
    )
    main, supplement = documents()
    paragraph = main.split("Three fixed artificial identity permutations", 1)[1].split("\n\n", 1)[0]
    section = supplement.split(r"\subsection{Fixed-permutation supervision-noise sensitivity}", 1)[
        1
    ].split(r"\section{", 1)[0]
    additional = result("noise_perm12_cuda_fgnet_v2_e391_20261009", "oriented-cuda-noise-permutation-full-fgnet-roc-v2")
    aucs = [data["large_gap_25plus"]["metrics"]["roc_auc"], *[
        additional[f"permutation_seed{seed}"]["large_gap_25plus"]["metrics"]["roc_auc"] for seed in (1, 2)]]
    means = [metric["cross"]["mean"] for metric in aucs]
    deltas = [metric["cross_minus_low"]["mean"] for metric in aucs]
    assert f"{min(means):.4f}--{max(means):.4f}" in paragraph
    assert f"{min(deltas):.4f}" in paragraph and f"{max(deltas):.4f}" in paragraph
    assert all(metric["cross_minus_low"]["mean_checkpoint_ci95"][1] < 0 for metric in aucs)
    assert "each nominal paired subject CI excludes zero" in paragraph
    for subset in ("overall", "large_gap_25plus"):
        for metric in data[subset]["metrics"].values():
            # Overall secondary EER/TAR are not claimed in this subsection.
            if subset == "overall" and metric is not data[subset]["metrics"]["roc_auc"]:
                continue
            assert f"{metric['low']['mean']:.4f}" in section
            assert f"{metric['cross']['mean']:.4f}" in section
            delta = metric["cross_minus_low"]
            assert f"{delta['mean']:+.4f}" in section
            lo, hi = delta["mean_checkpoint_ci95"]
            assert f"[{lo:+.4f},{hi:+.4f}]" in section
    assert "not an estimate" in section
    assert "not population-level robustness across permutations" in section


def test_all_three_noise_arms_are_linked_to_full_metric_tables():
    old = result("oriented_noise_cuda_fgnet_20261004", "oriented-cuda-noise-full-fgnet-roc-v2")
    new = result("noise_perm12_cuda_fgnet_v2_e391_20261009", "oriented-cuda-noise-permutation-full-fgnet-roc-v2")
    _, supplement = documents()
    for stratum, label in (("overall", "overall"), ("large_gap_25plus", "25plus")):
        body = supplement.split(r"\label{tab:noise-permutations-" + label + "}", 1)[1].split(r"\end{table*}", 1)[0]
        rows = [line.strip().removesuffix(r"\\").strip().split(" & ") for line in body.splitlines()
            if line.startswith(("42 & ", "1 & ", "2 & "))]
        assert len(rows) == 15
        pos = 0
        for seed in (42, 1, 2):
            arm = old[stratum] if seed == 42 else new[f"permutation_seed{seed}"][stratum]
            for name in ("roc_auc", "eer_interpolated", "eer_discrete_minimax", "tar@far=0.01", "tar@far=0.001"):
                metric = arm["metrics"][name]
                delta = metric["cross_minus_low"]
                lo, hi = delta["mean_checkpoint_ci95"]
                row = [field.replace("$", "").strip() for field in rows[pos]]
                assert row[0] == str(seed)
                assert row[2:] == [f"{metric['frozen']['point']:.4f}",
                    f"{metric['low']['mean']:.4f}\\pm{metric['low']['sd']:.4f}",
                    f"{metric['cross']['mean']:.4f}\\pm{metric['cross']['sd']:.4f}",
                    f"{delta['mean']:+.4f}", f"[{lo:+.4f},{hi:+.4f}]"]
                pos += 1


def test_noise_low_far_interval_limitations_are_not_overstated():
    data = result("noise_perm12_cuda_fgnet_v2_e391_20261009", "oriented-cuda-noise-permutation-full-fgnet-roc-v2")
    _, supplement = documents()
    section = supplement.split(r"\subsection{Fixed-permutation supervision-noise sensitivity}", 1)[1].split(r"\section{", 1)[0]
    first = data["permutation_seed1"]["large_gap_25plus"]["metrics"]
    second = data["permutation_seed2"]["large_gap_25plus"]["metrics"]
    assert first["tar@far=0.001"]["cross_minus_low"]["mean_checkpoint_ci95"][1] < 0
    assert second["tar@far=0.001"]["cross_minus_low"]["mean_checkpoint_ci95"][1] == 0
    assert "only permutation 1 excludes zero nominally" in section
    assert "not consistent evidence of operating-point deterioration across arms" in section
    assert "not a population of training seeds or permutations" in section


def test_replacement_table_cells_match_bound_metrics():
    data = result("oriented_cuda_fgnet_20261004", "oriented-cuda-controls-full-fgnet-roc-v2")
    _, supplement = documents()
    table = supplement.split(r"\label{tab:oriented-source}", 1)[1].split(r"\end{table}", 1)[0]
    rows = [
        line.split("&") for line in table.splitlines() if line.startswith(("Overall &", "25+ &"))
    ]
    specifications = [
        ("overall", "roc_auc"),
        ("large_gap_25plus", "roc_auc"),
        ("large_gap_25plus", "eer_interpolated"),
        ("large_gap_25plus", "tar@far=0.01"),
        ("large_gap_25plus", "tar@far=0.001"),
    ]
    assert len(rows) == len(specifications)

    def short(value, signed=False):
        return (f"{value:+.4f}" if signed else f"{value:.4f}").replace("0.", ".")

    for cells, (subset, key) in zip(rows, specifications, strict=True):
        metric = data[subset]["metrics"][key]
        assert cells[2].strip() == short(metric["frozen"]["point"])
        assert cells[3].strip() == short(metric["low"]["mean"])
        assert cells[4].strip() == short(metric["cross"]["mean"])
        delta = metric["cross_minus_low"]
        lo, hi = delta["mean_checkpoint_ci95"]
        expected = f"{short(delta['mean'], True)} [{short(lo, True)},{short(hi, True)}]"
        assert cells[5].replace("$", "").removesuffix(r" \\").strip() == expected
