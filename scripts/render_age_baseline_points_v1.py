"""Bound absolute age-probe/constant points; not confidence intervals or a mechanism."""

import argparse
import json
import math
import re
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs
from scripts.run_oriented_campaign import paths_from, verified

EXPERIMENT = "native-person-age-baselines-v2"
CELL = re.compile(r"random_(head|tail)_lr1e-06_s(42|1|2)\Z")
EXPECTED = {"strong/frozen", "weak/frozen"} | {
    f"strong/random_{scope}_lr1e-06_s{seed}" for scope in ("head", "tail") for seed in (42, 1, 2)
} | {f"weak/tuned_seed{seed}" for seed in (42, 1, 2)}


def number(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError("finite nonnegative MAE required")
    return value


def key_for(name):
    parts = name.split("/")
    if len(parts) != 3:
        raise ValueError("canonical native probe key required")
    _, family, model = parts
    if CELL.fullmatch(family) and model in {"frozen_age_probe", "tuned_age_probe"}:
        return "strong/" + ("frozen" if model == "frozen_age_probe" else family)
    key = family + "/" + model
    if key not in EXPECTED:
        raise ValueError("unrecognized native model key")
    return key


def collect(natives):
    models, reference, physical = {}, None, 0
    for native in natives:
        if (native.get("experiment") != EXPERIMENT
                or native["metrics"].get("execution_complete") is not True
                or native["metrics"].get("publication_ready") is not False
                or native["parameters"] != dict(device="cpu", constants="equal-person fit-only mean/median")):
            raise ValueError("completed native constant-baseline evidence required")
        for name, raw in native["metrics"]["probes"].items():
            key = key_for(name)
            physical += 1
            point = dict(mae_image=number(raw["probe_mae_image"]),
                         mae_person=number(raw["probe_mae_person"]))
            if set(raw["baselines"]) != {"mean", "median"}:
                raise ValueError("both fit-only constant references required")
            current = {}
            for method, row in raw["baselines"].items():
                if set(row) != {"mae_image", "mae_person"}:
                    raise ValueError("separate image and person denominators required")
                current[method] = {field: number(row[field]) for field in ("mae_image", "mae_person")}
            if reference is not None and current != reference:
                raise ValueError("shared constant-reference points differ")
            if key in models and models[key] != point:
                raise ValueError("duplicate model point differs")
            reference, models[key] = current, point
    if set(models) != EXPECTED:
        raise ValueError("all six initial strong and three weak checkpoints plus frozen references required")
    return dict(models=models, baselines=reference, native_probe_records=physical,
                unique_model_points=len(models), strong_tuned_checkpoints=6, weak_tuned_checkpoints=3,
                publication_ready=False, mechanism_complete=False,
                scope="absolute point estimates in years; image-weighted and equal-person MAE "
                      "are different denominators; deduplicated model points, not independent replications; "
                      "no new CI or significance test, age removal, mechanism, controlled causal "
                      "family comparison or identity-independence clearance")


def label(key):
    family, model = key.split("/")
    if model == "frozen":
        return family.capitalize() + " frozen"
    if family == "weak":
        return "Weak, seed " + model.removeprefix("tuned_seed")
    scope, seed = CELL.fullmatch(model).groups()
    return f"Strong {scope}, seed {seed}"


def table(payload):
    rows = [f"Fit-only {name} & {row['mae_image']:.4f} & {row['mae_person']:.4f} " + r"\\"
            for name, row in payload["baselines"].items()]
    rows += [f"{label(key)} & {row['mae_image']:.4f} & {row['mae_person']:.4f} " + r"\\"
             for key, row in sorted(payload["models"].items())]
    return "\n".join((
        r"\begin{table}[!t]",
        r"\caption{Absolute age-probe and fit-only constant MAE points (years). Strong rows cover "
        "only the initial six random/head--tail/LR $10^{-6}$ checkpoints. Weak rows use different "
        "architecture/preprocessing and unverified training history, not a controlled causal family "
        "comparison. Repeated physical probe records are deduplicated, not independent replications. "
        "Image-weighted and equal-person MAE are separate denominators; no new interval or "
        "significance test, age-removal claim, mechanism or identity-independence clearance.}",
        r"\label{tab:age-probe-absolute-points}", r"\centering\scriptsize",
        r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
        r"Model/reference & Image-weighted MAE & Equal-person MAE \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    )) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output root required")
    bindings = [path.resolve() for path in args.bindings]
    if len(set(bindings)) != len(bindings):
        raise ValueError("duplicate native binding")
    initial_bindings = [file_record(path) for path in bindings]
    natives = [verified(path, EXPERIMENT) for path in bindings]
    if initial_bindings != [file_record(path) for path in bindings]:
        raise RuntimeError("native binding changed during verification")
    inputs = sorted({Path(__file__).resolve(), *bindings,
                     PROJECT_ROOT / "scripts/run_oriented_campaign.py",
                     PROJECT_ROOT / "scripts/run_restricted_matched_campaign.py",
                     PROJECT_ROOT / "scripts/evaluate_oriented_cuda_v1.py",
                     PROJECT_ROOT / "src/age_gap/common/manifest.py",
                     *(path for native in natives for path in paths_from(native))})
    before = [file_record(path) for path in inputs]
    payload = collect(natives)
    rendered = table(payload)
    if before != [file_record(path) for path in inputs]:
        raise RuntimeError("age presentation inputs changed")
    args.out.mkdir(parents=True)
    summary, tex = args.out / "summary.json", args.out / "age_baselines_table.tex"
    summary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tex.write_text(rendered, encoding="utf-8")
    target = args.out / "presentation.manifest.json"
    write_experiment_manifest(target, experiment="absolute-age-probe-points-presentation-v1",
                              parameters=dict(seed=None, inference="none; absolute points only"),
                              metrics=payload, inputs=inputs, outputs=[summary, tex])
    try:
        validate_written_inputs(target, before)
        if before != [file_record(path) for path in inputs]:
            raise RuntimeError("age presentation changed after publication")
    except Exception:
        target.unlink(missing_ok=True)  # Only this invocation's new manifest.
        raise
    print("rendered11 unique model points; no significance or mechanism claim", flush=True)


if __name__ == "__main__":
    main()
