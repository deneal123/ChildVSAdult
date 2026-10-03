"""Generate internal ROC-v2 sensitivity table from a fully bound aggregate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.render_fgnet_evidence import interval, number, verified_result


def render(payload):
    if payload["metric_version"] != "empirical-roc-v2" or payload["deployment_calibration"]:
        raise ValueError("explicit ROC-v2 non-deployment protocol required")
    bootstrap = payload["bootstrap"]
    if not bootstrap["threshold_reselection"] or bootstrap["valid"] < 2:
        raise ValueError("conditional resampled ROC thresholds required")
    labels = (("roc_auc", "ROC-AUC"), ("eer_interpolated", "Interpolated EER"),
              ("eer_discrete_minimax", "Discrete minimax EER"),
              ("tar@far=0.01", r"TAR@FAR=1\%"), ("tar@far=0.001", r"TAR@FAR=0.1\%"))
    rows = []
    for key, name in (("original_tolerance", "Tolerance 2y"), ("exact_age_subset", "Exact ages")):
        subset = payload["subsets"][key]
        if subset["n_positive"] != subset["n_negative"]:
            raise ValueError("balanced whole matched blocks required")
        for metric, label in labels:
            values = subset["metrics"][metric]
            delta = values["tuned_minus_frozen"]
            rows.append(" & ".join((name, label, number(values["frozen"]["point"]), number(values["tuned"]["point"]),
                                   number(delta["point"], signed=True), interval(delta["ci95"]))) + r" \\")
    original, exact = (payload["subsets"][key] for key in ("original_tolerance", "exact_age_subset"))
    return "\n".join(("% BEGIN GENERATED INTERNAL ROC V2", r"\begin{table*}[!t]",
        r"\caption{Source-bound CPU FaceNet internal sensitivity: "
        + f"{original['n_positive']} positive/negative matched blocks and {original['recorded_people']} recorded persons; "
        + f"exact-age subset {exact['n_positive']} blocks/{exact['recorded_people']} persons. "
        + "Exact blocks were selected by endpoint ages before scoring, without re-mining. "
        + "Joint recorded-person dyad bootstrap uses the same positive/negative block weight, conditional on one fixed seed-42 checkpoint; "
        + "missed-merge independence and training provenance remain unverified. TAR maximizes the attainable empirical ROC at FAR no greater than target, "
        + "accepting tied scores together; thresholds are reselected in each draw, not calibrated for deployment. "
        + "Interpolated ROC EER and attainable discrete minimax EER are separate definitions. Negative EER deltas favor tuning; "
        + "Interpret changes using their reported intervals, which are not multiplicity-corrected.}",
        r"\label{tab:internal-roc-v2}", r"\centering\footnotesize", r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
        r"Subset & Metric & Frozen & Tuned & Tuned$-$frozen & Paired delta 95\% CI \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}", "% END GENERATED INTERNAL ROC V2")) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=PROJECT_ROOT / "metrics/internal_metrics_v2_20261003/summary.json")
    args = parser.parse_args()
    paths = [args.result, args.result.with_suffix(".manifest.json"), Path(__file__), PROJECT_ROOT / "scripts/render_fgnet_evidence.py"]
    before = [file_record(path) for path in paths]
    table = render(verified_result(args.result, PROJECT_ROOT))
    target = args.result.parent / "internal_table.tex"
    if target.exists():
        raise FileExistsError("fresh table output required")
    if before != [file_record(path) for path in paths]:
        raise ValueError("render inputs changed")
    target.write_text(table, encoding="utf-8")
    presentation = args.result.parent / "presentation.manifest.json"
    write_experiment_manifest(presentation, experiment="internal-roc-v2-table-presentation",
        parameters={"seed": None, "table_label": "tab:internal-roc-v2"}, metrics={"publication_ready": False}, inputs=paths, outputs=[target])
    if json.loads(presentation.read_text(encoding="utf-8"))["inputs"] != before:
        presentation.unlink()
        raise ValueError("render inputs changed while writing manifest")
    print(table)


if __name__ == "__main__":
    main()
