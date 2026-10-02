"""Render the corrected primary endpoint table from checksum-verified artifacts.

This validates evaluation provenance, not the absent legacy training manifests.
Use --check to detect a stale generated block in the English master manuscript.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record

BEGIN = "% BEGIN GENERATED FGNET ENDPOINT TABLE"
END = "% END GENERATED FGNET ENDPOINT TABLE"


def verified_payload(path: Path, root: Path, seen: set[Path] | None = None) -> dict:
    """Check output binding and every input, recursively checking input manifests."""
    seen = set() if seen is None else seen
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if file_record(path) not in manifest.get("outputs", []):
        raise ValueError(f"output checksum mismatch: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("metrics") != payload:
        raise ValueError(f"manifest metrics mismatch: {path.name}")
    if manifest_path in seen:
        return payload
    seen.add(manifest_path)
    for record in manifest.get("inputs", []):
        source = Path(record["path"])
        source = source if source.is_absolute() else root / source
        if file_record(source) != record:
            raise ValueError(f"input checksum mismatch: {source.name}")
        if source.name.endswith(".manifest.json"):
            result_path = source.with_name(source.name.replace(".manifest.json", ".json"))
            verified_payload(result_path, root, seen)
    return payload


def render_table(summary: dict) -> str:
    if summary.get("protocol") != "endpoint_age_matched":
        raise ValueError("refusing a legacy negative protocol")
    seeds = summary["seeds"]
    if len(seeds) != 3 or set(seeds) != {42, 1, 2}:
        raise ValueError("expected three distinct training seeds")
    rows = []
    for key, label in (("large_gap_25plus", "25+ years (primary)"), ("overall", "Overall")):
        row = summary["strata"][key]
        intervals = row["subject_bootstrap_by_seed"]
        if set(intervals) != {str(seed) for seed in seeds}:
            raise ValueError("missing per-seed subject interval")
        reference = intervals["42"]
        if any(
            item["n_pairs"] != reference["n_pairs"]
            or item["n_subjects"] != reference["n_subjects"]
            for item in intervals.values()
        ):
            raise ValueError("seed evaluations must use the same endpoint sample")
        low, high = reference["delta_ci95"]
        rows.append(
            f"{label} & {reference['n_pairs']} / {reference['n_subjects']} & "
            f"{row['frozen_auc']:.4f} & {row['tuned_auc_mean']:.4f} "
            f"$\\pm$ {row['tuned_auc_std_across_seeds']:.4f} & "
            f"${row['delta_auc_mean']:+.4f}\\pm"
            f"{row['delta_auc_std_across_seeds']:.4f}$ & "
            f"$[{low:+.4f},{high:+.4f}]$ \\\\"
        )
    return "\n".join([
        BEGIN,
        r"\begin{table*}[!t]",
        r"\caption{Corrected FG-NET endpoint-age-matched evaluation (FaceNet). Means and standard deviations cover three training seeds; the paired subject-bootstrap interval is for seed 42, not for the seed mean. All seeds reuse the same subjects. Legacy checkpoint training manifests are absent; train--benchmark identity overlap remains under review.}",
        r"\label{tab:endpoint}",
        r"\centering\footnotesize",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Stratum & Pairs / subjects & Frozen AUC & Tuned AUC & Gain & Gain 95\% CI (s42) \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        END,
    ]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "metrics/fgnet_endpoint_multiseed.json")
    parser.add_argument("--check", type=Path, help="check generated block without editing manuscript")
    args = parser.parse_args()
    table = render_table(verified_payload(args.summary, PROJECT_ROOT))
    if args.check:
        text = args.check.read_text(encoding="utf-8")
        if text.count(BEGIN) != 1 or text.count(END) != 1:
            raise SystemExit("missing or duplicate generated endpoint table block")
        actual = text[text.index(BEGIN):text.index(END) + len(END)] + "\n"
        if actual != table:
            raise SystemExit("stale generated endpoint table; rerender and update the master")
        print("Endpoint table matches verified evaluation artifacts; training provenance remains open.")
    else:
        print(table, end="")


if __name__ == "__main__":
    main()
