"""Render declared absolute age-probe/constant points for canonical strong cells.

Version 3 of the absolute age presentation. It keeps the exact v1 scalar schema
(absolute MAE image-weighted and equal-person points plus the shared fit-only
constant references) but replaces the hardcoded six-cell strong set with an
explicit ``--expected-cells`` declaration over the canonical 36-cell naming::

    (random|lookalike) x (head|tail|full) x lr{1e-06,1e-05} x seed{42,1,2}

The declared set is enforced with exact membership: every declared cell must be
observed and no undeclared canonical cell may appear. Counts are derived from the
declared set, so partial coverage is reported as partial. This is descriptive
point estimation only: no averaging, no ensemble, no new interval or significance
test, no age-removal claim, no mechanism claim and no identity clearance.

``scripts/render_age_baseline_points_v1.py`` is immutable; this module neither
imports nor mutates it. The prerequisite verifier is stdlib-only and preserves the
exact ``run_restricted_matched_campaign.verify_records`` semantics, so this module
never imports Torch or OpenCV.
"""

import argparse
import json
import math
import re
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.common_pair_linkage_light_v1 import validate_written_inputs

EXPERIMENT = "native-person-age-baselines-v2"
PARAMETERS = dict(device="cpu", constants="equal-person fit-only mean/median")
PRESENTATION = "absolute-age-probe-points-presentation-v3"
CELL = re.compile(r"(random|lookalike)_(head|tail|full)_lr(1e-06|1e-05)_s(42|1|2)\Z")
NEGATIVES = ("random", "lookalike")
SCOPES = ("head", "tail", "full")
LRS = ("1e-06", "1e-05")
SEEDS = (42, 1, 2)
CANONICAL_CELLS = frozenset(
    f"{negative}_{scope}_lr{lr}_s{seed}"
    for negative in NEGATIVES
    for scope in SCOPES
    for lr in LRS
    for seed in SEEDS
)
LR_LABEL = {"1e-06": "10^{-6}", "1e-05": "10^{-5}"}
REFERENCE_KEYS = frozenset({"strong/frozen", "weak/frozen"})
WEAK_KEYS = frozenset(f"weak/tuned_seed{seed}" for seed in SEEDS)
WEAK_MODEL_FILES = frozenset({"frozen_age_probe", "tuned_age_probe"})
SCOPE = (
    "absolute point estimates in years for explicitly declared canonical strong cells; "
    "image-weighted and equal-person MAE are different denominators; deduplicated model "
    "points, not independent replications; no new CI or significance test, age removal, "
    "mechanism, controlled causal family comparison or identity-independence clearance"
)


def number(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError("finite nonnegative MAE required")
    return value


def expected_cells(values):
    """Reject malformed or duplicated explicit strong-cell declarations."""
    if not values:
        raise ValueError("at least one expected canonical strong cell required")
    declared, seen = [], set()
    for value in values:
        if not isinstance(value, str) or not CELL.fullmatch(value):
            raise ValueError(f"malformed canonical strong cell: {value!r}")
        if value in seen:
            raise ValueError(f"duplicate expected strong cell: {value}")
        seen.add(value)
        declared.append(value)
    return declared


def key_for(name):
    parts = name.split("/")
    if len(parts) != 3:
        raise ValueError("canonical native probe key required")
    _, family, model = parts
    if CELL.fullmatch(family) and model in WEAK_MODEL_FILES:
        return "strong/" + ("frozen" if model == "frozen_age_probe" else family)
    key = family + "/" + model
    if key not in REFERENCE_KEYS | WEAK_KEYS and not (family == "strong" and CELL.fullmatch(model)):
        raise ValueError("unrecognized native model key")
    return key


def verify_records(records, root=PROJECT_ROOT):
    """Stdlib-only port of run_restricted_matched_campaign.verify_records."""
    if not records:
        raise ValueError("nonempty declared input records required")
    for record in records:
        path = Path(record["path"])
        if not path.is_absolute():
            path = root / path
        try:
            if file_record(path) != record:
                raise ValueError("declared input size/checksum changed")
        except OSError:
            raise ValueError("declared input is unavailable") from None


def verified(path, experiment):
    native = json.loads(path.read_text(encoding="utf-8"))
    if native.get("experiment") != experiment:
        raise ValueError("wrong prerequisite type")
    verify_records(native["inputs"] + native["outputs"])
    summary_path = path.parent / "summary.json"
    if file_record(summary_path) not in native["outputs"]:
        raise ValueError("native summary output must be bound")
    if json.loads(summary_path.read_text(encoding="utf-8")) != native["metrics"]:
        raise ValueError("native summary disagrees with manifest metrics")
    return native


def paths_from(native):
    return [
        Path(record["path"]) if Path(record["path"]).is_absolute() else PROJECT_ROOT / record["path"]
        for record in native["inputs"] + native["outputs"]
    ]


def collect(natives, declared):
    declared = expected_cells(declared)
    models, reference, physical = {}, None, 0
    for native in natives:
        if (native.get("experiment") != EXPERIMENT
                or native["metrics"].get("execution_complete") is not True
                or native["metrics"].get("publication_ready") is not False
                or native["parameters"] != PARAMETERS):
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
    observed = {key.split("/", 1)[1] for key in models
                if key.startswith("strong/") and key != "strong/frozen"}
    unexpected, missing = sorted(observed - set(declared)), sorted(set(declared) - observed)
    if unexpected or missing:
        raise ValueError(f"unexpected strong cells {unexpected}; missing strong cells {missing}")
    expected = REFERENCE_KEYS | WEAK_KEYS | {f"strong/{cell}" for cell in declared}
    if set(models) != expected:
        raise ValueError("exact declared strong cells plus frozen/weak references required")
    weak_tuned = sum(1 for key in models if key.startswith("weak/tuned_seed"))
    return dict(models=models, baselines=reference, native_probe_records=physical,
                unique_model_points=len(models), expected_cells=sorted(declared),
                expected_strong_cells=len(declared), observed_strong_cells=len(observed),
                strong_tuned_checkpoints=len(observed), weak_tuned_checkpoints=weak_tuned,
                canonical_matrix_cells=len(CANONICAL_CELLS),
                matrix_complete=observed == CANONICAL_CELLS,
                publication_ready=False, mechanism_complete=False, scope=SCOPE)


def label(key):
    family, model = key.split("/")
    if model == "frozen":
        return family.capitalize() + " frozen"
    if family == "weak":
        return "Weak, seed " + model.removeprefix("tuned_seed")
    negative, scope, lr, seed = CELL.fullmatch(model).groups()
    return f"Strong {negative} {scope}, LR ${LR_LABEL[lr]}$, seed {seed}"


def table(payload):
    rows = [f"Fit-only {name} & {row['mae_image']:.4f} & {row['mae_person']:.4f} " + r"\\"
            for name, row in payload["baselines"].items()]
    rows += [f"{label(key)} & {row['mae_image']:.4f} & {row['mae_person']:.4f} " + r"\\"
             for key, row in sorted(payload["models"].items())]
    coverage = (f"Strong rows cover exactly the {payload['expected_strong_cells']} explicitly declared "
                f"cells of the {payload['canonical_matrix_cells']}-cell (random|lookalike) x "
                "(head|tail|full) x LR $10^{-6}$/$10^{-5}$ x seed {42,1,2} matrix; every remaining "
                "cell is not rendered. " +
                ("All declared matrix cells are covered; this is not a mechanism conclusion."
                 if payload["matrix_complete"] else "The matrix stays incomplete."))
    return "\n".join((
        r"\begin{table}[!t]",
        r"\caption{Absolute age-probe and fit-only constant MAE points (years). " + coverage +
        " Weak rows use different architecture/preprocessing and unverified training history, not a "
        "controlled causal family comparison. Repeated physical probe records are deduplicated, not "
        "independent replications. Image-weighted and equal-person MAE are separate denominators; no "
        "new interval or significance test, age-removal claim, mechanism or identity-independence "
        "clearance.}",
        r"\label{tab:age-probe-absolute-points}", r"\centering\scriptsize",
        r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
        r"Model/reference & Image-weighted MAE & Equal-person MAE \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    )) + "\n"


def run(bindings, out, declared):
    declared = expected_cells(declared)
    bindings = [Path(path).resolve() for path in bindings]
    if not bindings:
        raise ValueError("nonempty native bindings required")
    if out.exists():
        raise FileExistsError("fresh output root required")
    if len(set(bindings)) != len(bindings):
        raise ValueError("duplicate native binding")
    initial = [file_record(path) for path in bindings]
    natives = [verified(path, EXPERIMENT) for path in bindings]
    if initial != [file_record(path) for path in bindings]:
        raise RuntimeError("native binding changed during verification")
    inputs = sorted({
        Path(__file__).resolve(), *bindings,
        PROJECT_ROOT / "scripts/common_pair_linkage_light_v1.py",
        PROJECT_ROOT / "scripts/render_age_baseline_points_v1.py",
        PROJECT_ROOT / "scripts/run_oriented_campaign.py",
        PROJECT_ROOT / "scripts/run_restricted_matched_campaign.py",
        PROJECT_ROOT / "scripts/evaluate_oriented_cuda_v1.py",
        PROJECT_ROOT / "src/age_gap/common/manifest.py",
        *(path for native in natives for path in paths_from(native)),
    })
    before = [file_record(path) for path in inputs]
    payload = collect(natives, declared)
    rendered = table(payload)
    if before != [file_record(path) for path in inputs]:
        raise RuntimeError("age presentation inputs changed")
    out.mkdir(parents=True)
    summary, tex = out / "summary.json", out / "age_baselines_table.tex"
    summary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tex.write_text(rendered, encoding="utf-8")
    target = out / "presentation.manifest.json"
    try:
        write_experiment_manifest(
            target, experiment=PRESENTATION,
            parameters=dict(seed=None, inference="none; absolute points only",
                            expected_cells=sorted(declared)),
            metrics=payload, inputs=inputs, outputs=[summary, tex])
        validate_written_inputs(target, before)
        if before != [file_record(path) for path in inputs]:
            raise RuntimeError("age presentation changed after publication")
    except BaseException:
        # Fresh output root: only this invocation can own this completion marker.
        if target.exists():
            target.unlink()
        raise
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-cells", nargs="+", required=True,
                        help="explicit canonical strong cells, e.g. random_head_lr1e-06_s42")
    args = parser.parse_args()
    declared = expected_cells(args.expected_cells)
    bindings = [path.resolve() for path in args.bindings]
    payload = run(bindings, args.out, declared)
    print(f"rendered {payload['strong_tuned_checkpoints']} strong and "
          f"{payload['weak_tuned_checkpoints']} weak checkpoints; no significance or mechanism claim",
          flush=True)


if __name__ == "__main__":
    main()
