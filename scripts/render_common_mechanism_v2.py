"""Render bound per-checkpoint common-mechanism evidence; never claim a mechanism.

Descriptive presentation of explicitly selected completed ``common-image-index-mechanism-v1``
bindings (one canonical strong matrix cell each) plus the single deduplicated weak
reference family.  Frozen strong/weak references and the whole weak tuned family are
identical across bindings, so they are bound once and rendered once.  Every field is
derived from the already hash-verified native metrics; nothing is handwritten.  This is
*not* a mechanism conclusion even at all36cells, not a seed mean or an ensemble, not causal
and not training-seed population inference: lower age MAE means easier age decoding,
not age removal.
"""

import argparse
import copy
import json
import math
import re
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.audit_common_pair_protocol_v3 import audit as audit_shared_pairs
from scripts.common_pair_linkage_light_v1 import validate_written_inputs
from scripts.run_oriented_campaign import paths_from, verified

EXPERIMENT = "common-image-index-mechanism-v1"
SEPARATION_VERSION = "fixed-pair-subject-separability-v1"
STRONG_CELL = re.compile(r"(random|lookalike)_(head|tail|full)_lr(1e-06|1e-05)_s(42|1|2)\Z")
LR_LABEL = {"1e-06": "10^{-6}", "1e-05": "10^{-5}"}
WEAK_TUNED = ("tuned_seed42", "tuned_seed1", "tuned_seed2")
WEAK_MODELS = ("frozen", *WEAK_TUNED)
N_IMAGES = 650
N_PAIRS = 5308
N_PERSONS = 82
RESAMPLES = 2000
BOOTSTRAP_SEED = 0
EXPECTED_STRONG_MATRIX_CELLS = 36
DEPENDENCIES = (
    "scripts/audit_common_pair_protocol_v3.py",
    "scripts/common_pair_linkage_light_v1.py",
    "scripts/run_common_mechanism_v1.py",
    "scripts/identity_separability_v1.py",
    "scripts/mechanism_diagnostics_v1.py",
    "scripts/run_mechanism_diagnostics_v1.py",
    "scripts/run_oriented_campaign.py",
    "scripts/run_restricted_matched_campaign.py",
    "scripts/evaluate_oriented_cuda_v1.py",
    "src/age_gap/common/manifest.py",
)
LIMITATION = (
    "per-checkpoint conditional descriptive evidence only; incomplete 36-cell strong matrix; "
    "no average, ensemble or seed-mean; no training-seed population inference, causal claim or "
    "mechanism decision; lower age MAE is easier age decoding, not age removal; weak training "
    "history and train-benchmark independence unverified; fixed OOF age predictions, "
    "no probe refitting or multiplicity correction; different architecture/preprocessing "
    "families, not a controlled causal weak-versus-strong comparison"
)


def numeric(value, low, high):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError("finite in-range metric required")
    return value


def interval(value, low, high):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("ordered two-endpoint interval required")
    lo, hi = numeric(value[0], low, high), numeric(value[1], low, high)
    if lo > hi:
        raise ValueError("ordered two-endpoint interval required")
    return list(value)


def count(value, expected):
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError("expected bound integer count required")
    return value


def checked_sep_model(model):
    numeric(model["negative_mean"], -1, 1)
    numeric(model["positive_mean"], -1, 1)
    numeric(model["negative_variance"], 0, 4)
    numeric(model["positive_variance"], 0, 4)
    numeric(model["standardized_mean_difference"], -math.inf, math.inf)
    interval(model["ci95"], -math.inf, math.inf)
    numeric(model["delta_vs_frozen"], -math.inf, math.inf)
    interval(model["delta_ci95"], -math.inf, math.inf)
    return model


def checked_frozen(model):
    checked_sep_model(model)
    if model["delta_vs_frozen"] != 0 or list(model["delta_ci95"]) != [0, 0]:
        raise ValueError("frozen reference must carry a zero self-delta")
    return model


def checked_diagnostics(entry):
    representation = entry["representation"]
    numeric(representation["linear_cka"], 0, 1)
    numeric(representation["mean_cosine_drift"], 0, 2)
    count(representation["n_images"], N_IMAGES)
    age = entry["paired_age_error"]
    numeric(age["delta_mae_person"], -math.inf, math.inf)
    interval(age["ci95"], -math.inf, math.inf)
    count(age["n_persons"], N_PERSONS)
    count(age["resamples"], RESAMPLES)
    count(age["seed"], BOOTSTRAP_SEED)
    return entry


def checked_separability(sep, models):
    if (sep["status"] != "ok" or sep["version"] != SEPARATION_VERSION
            or sep["positive_weight"] != "owner multiplicity once"
            or sep["negative_weight"] != "endpoint multiplicity product"):
        raise ValueError("intact fixed-pair subject separability protocol required")
    for key, expected in (("n_pairs", N_PAIRS), ("n_persons", N_PERSONS),
                          ("requested_draws", RESAMPLES), ("valid_draws", RESAMPLES),
                          ("seed", BOOTSTRAP_SEED)):
        count(sep[key], expected)
    if not isinstance(sep["models"], dict) or set(sep["models"]) != set(models):
        raise ValueError("exact frozen plus tuned separability models required")
    return sep


def checked_delta(tuned, frozen):
    if not math.isclose(tuned["standardized_mean_difference"]
                        - frozen["standardized_mean_difference"],
                        tuned["delta_vs_frozen"], abs_tol=1e-9):
        raise ValueError("paired tuned delta inconsistent with frozen reference")


def require_common_parameters(parameters):
    checks = (
        parameters.get("device") == "cpu",
        not isinstance(parameters.get("age_alpha"), bool) and parameters.get("age_alpha") == 1,
        parameters.get("protocol") == "completed strong bound image-index pairs",
    )
    if not all(checks):
        raise ValueError("shared common-mechanism CPU protocol required")
    for key, expected in (("bootstrap_seed", BOOTSTRAP_SEED), ("resamples", RESAMPLES),
                          ("blas_threads", 1), ("age_folds", 5), ("age_fold_seed", 42)):
        try:
            count(parameters.get(key), expected)
        except ValueError:
            raise ValueError("shared common-mechanism CPU protocol required") from None


def collect(natives):
    """Validate already hash-verified native dictionaries; does not verify files.

    Strong cells are keyed by their single canonical tuned checkpoint.  The strong
    frozen reference, the weak separability family and the weak diagnostics family must
    be structurally identical across every binding, so each is rendered once.
    """
    cells, diagnostics, frozen = {}, {}, None
    weak_evidence = None
    for native in natives:
        if native.get("experiment") != EXPERIMENT:
            raise ValueError("wrong prerequisite experiment")
        metrics = native["metrics"]
        if (metrics.get("execution_complete") is not True
                or metrics.get("publication_ready") is not False
                or "no mechanism decision" not in str(metrics.get("limitation", ""))):
            raise ValueError("completed non-publication common-mechanism evidence required")
        count(metrics.get("n_images"), N_IMAGES)
        count(metrics.get("n_pairs"), N_PAIRS)
        require_common_parameters(native["parameters"])
        results = metrics["results"]
        strong = results["strong"]
        tuned_names = [name for name in strong["separability"]["models"] if name != "frozen"]
        if len(tuned_names) != 1 or not STRONG_CELL.fullmatch(tuned_names[0]):
            raise ValueError("one unique canonical strong tuned checkpoint per binding required")
        name = tuned_names[0]
        if sorted(strong["diagnostics"]) != [name]:
            raise ValueError("diagnostics must bind exactly the tuned strong checkpoint")
        sep = checked_separability(strong["separability"], ("frozen", name))
        checked_frozen(sep["models"]["frozen"])
        checked_sep_model(sep["models"][name])
        checked_delta(sep["models"][name], sep["models"]["frozen"])
        if frozen is not None and frozen != sep["models"]["frozen"]:
            raise ValueError("shared strong frozen reference differs across bindings")
        frozen = sep["models"]["frozen"]
        if name in cells:
            raise ValueError("unique strong tuned checkpoint per binding required")
        cells[name] = sep["models"][name]
        diagnostics[name] = checked_diagnostics(strong["diagnostics"][name])
        weak = results["weak"]
        weak_sep = checked_separability(weak["separability"], WEAK_MODELS)
        checked_frozen(weak_sep["models"]["frozen"])
        for tuned in WEAK_TUNED:
            checked_sep_model(weak_sep["models"][tuned])
            checked_delta(weak_sep["models"][tuned], weak_sep["models"]["frozen"])
        if sorted(weak["diagnostics"]) != sorted(WEAK_TUNED):
            raise ValueError("weak diagnostics must bind exactly the three tuned seeds")
        for tuned in WEAK_TUNED:
            checked_diagnostics(weak["diagnostics"][tuned])
        current = dict(models=weak_sep["models"], diagnostics=weak["diagnostics"])
        if weak_evidence is not None and weak_evidence != current:
            raise ValueError("shared weak reference differs across bindings")
        weak_evidence = current
    if not cells:
        raise ValueError("at least one completed strong tuned checkpoint required")
    return dict(
        strong=dict(frozen=frozen, cells=cells, diagnostics=diagnostics),
        weak=copy.deepcopy(weak_evidence),
        n_images=N_IMAGES, n_pairs=N_PAIRS, n_persons=N_PERSONS,
        resamples=RESAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
        observed_strong_cells=len(cells),
        expected_strong_matrix_cells=EXPECTED_STRONG_MATRIX_CELLS,
        evaluated_strong_cells=len(cells),
        mechanism_complete=False, publication_ready=False,
        limitation=LIMITATION,
    )


def strong_label(name):
    arm, scope, lr, seed = STRONG_CELL.fullmatch(name).groups()
    return f"{arm}, {scope}, LR ${LR_LABEL[lr]}$, seed {seed}"


def weak_label(name):
    return f"tuned, seed {name.removeprefix('tuned_seed')}"


def _row(label, sep, entry):
    representation, age = entry["representation"], entry["paired_age_error"]
    age_lo, age_hi = age["ci95"]
    delta_lo, delta_hi = sep["delta_ci95"]
    return (f"{label} & {representation['linear_cka']:.4f} "
            f"& {representation['mean_cosine_drift']:.4f} "
            f"& {age['delta_mae_person']:+.4f} & $[{age_lo:+.4f},{age_hi:+.4f}]$ "
            f"& {sep['standardized_mean_difference']:.4f} "
            f"& {sep['delta_vs_frozen']:+.4f} & $[{delta_lo:+.4f},{delta_hi:+.4f}]$ "
            + r"\\")


HEADER = (
    r"Checkpoint & CKA & Cos.\ drift & Age $\Delta$MAE "
    r"& Age $\Delta$MAE 95\% CI & Sep.\ SMD & Sep.\ $\Delta$SMD "
    r"& Sep.\ $\Delta$95\% CI \\"
)


def strong_table(payload):
    """One row per strong tuned checkpoint; frozen reference is shared and in the caption."""
    strong = payload["strong"]
    frozen = strong["frozen"]
    frozen_lo, frozen_hi = frozen["ci95"]
    rows = [_row(strong_label(name), strong["cells"][name], strong["diagnostics"][name])
            for name in sorted(strong["cells"])]
    count_observed = payload["observed_strong_cells"]
    count_expected = payload["expected_strong_matrix_cells"]
    return "\n".join((
        r"\begin{table*}[!t]",
        rf"\caption{{Strong arm: {count_observed} of {count_expected} expected tuned checkpoints, "
        "each conditional on its own fixed checkpoint (descriptive, not the full matrix). "
        rf"Frozen reference separation SMD is {frozen['standardized_mean_difference']:.4f} "
        rf"(95\% CI $[{frozen_lo:.4f},{frozen_hi:.4f}]$), the same bound reference in every row. "
        "$\\Delta$MAE and SMD deltas are tuned minus frozen; each interval is the paired "
        "subject/endpoint bootstrap interval for this checkpoint, not a seed mean and not "
        "an ensemble. "
        "Age intervals hold OOF probe predictions fixed, with no probe refitting. "
        "Lower age MAE means easier age decoding, not age removal. "
        "Descriptive only: no mechanism decision, no training-seed population inference, no "
        "identity-independence clearance and no multiplicity correction; not deployment-calibrated.}",
        r"\label{tab:common-mechanism-strong}",
        r"\centering\scriptsize",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule", HEADER, r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular}", r"\end{table*}",
    )) + "\n"


def weak_table(payload):
    """The weak reference family is deduplicated across bindings and rendered once."""
    weak = payload["weak"]
    models, diagnostics = weak["models"], weak["diagnostics"]
    frozen = models["frozen"]
    frozen_lo, frozen_hi = frozen["ci95"]
    rows = [_row(weak_label(name), models[name], diagnostics[name]) for name in WEAK_TUNED]
    return "\n".join((
        r"\begin{table*}[!t]",
        r"\caption{Weak arm: the same bound weak tuned family appears in every binding and is "
        "rendered once (deduplicated, not averaged or ensembled). "
        rf"Frozen reference separation SMD is {frozen['standardized_mean_difference']:.4f} "
        rf"(95\% CI $[{frozen_lo:.4f},{frozen_hi:.4f}]$). "
        "$\\Delta$MAE and SMD deltas are tuned minus frozen; intervals are conditional paired "
        "subject bootstrap intervals, not a seed mean and not an ensemble. Lower age MAE means "
        "easier age decoding, "
        "not age removal. Descriptive only: no mechanism decision, no training-seed population "
        "inference and no identity-independence clearance; weak training history unverified. "
        "Age intervals hold OOF predictions fixed, without probe refitting or multiplicity "
        "correction. Different architecture/preprocessing, not a causal family comparison.}",
        r"\label{tab:common-mechanism-weak}",
        r"\centering\scriptsize",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule", HEADER, r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular}", r"\end{table*}",
    )) + "\n"


def run(bindings, out):
    """Validate bindings, render both tables and publish a bound presentation manifest."""
    if out.exists():
        raise FileExistsError("fresh presentation root required")
    bindings = [path.resolve() for path in bindings]
    binding_before = [file_record(path) for path in bindings]
    natives = [verified(path, EXPERIMENT) for path in bindings]
    pair_audit, selected_paths = audit_shared_pairs(bindings)
    if binding_before != [file_record(path) for path in bindings]:
        raise RuntimeError("binding changed during native verification")
    inputs = sorted(set([
        Path(__file__), *bindings,
        *(PROJECT_ROOT / name for name in DEPENDENCIES),
        *(path for native in natives for path in paths_from(native)),
        *selected_paths,
    ]))
    before = [file_record(path) for path in inputs]
    payload = collect(natives)
    if (pair_audit.get("exact_pair_arrays_equal") is not True
            or pair_audit.get("n_images") != N_IMAGES or pair_audit.get("n_pairs") != N_PAIRS
            or pair_audit.get("checked_cells") != sorted(payload["strong"]["cells"])):
        raise ValueError("exact common pair audit must cover every rendered strong cell")
    payload["pair_linkage_scope"] = "selected records and exact arrays; not human identity clearance"
    rendered = {"strong_table.tex": strong_table(payload), "weak_table.tex": weak_table(payload)}
    if before != [file_record(path) for path in inputs]:
        raise RuntimeError("presentation ancestry changed")
    out.mkdir(parents=True)
    summary = out / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    outputs = [summary]
    for name, content in rendered.items():
        path = out / name
        path.write_text(content, encoding="utf-8")
        outputs.append(path)
    target = out / "presentation.manifest.json"
    try:
        write_experiment_manifest(
            target, experiment="common-mechanism-per-checkpoint-presentation-v2",
            parameters=dict(seed=None, aggregation="none; per-checkpoint only",
                            weak_reference="deduplicated across bindings; rendered once"),
            metrics=payload, inputs=inputs, outputs=outputs)
        validate_written_inputs(target, before)
        if before != [file_record(path) for path in inputs]:
            raise RuntimeError("presentation ancestry changed after publication")
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
    args = parser.parse_args()
    payload = run(args.bindings, args.out)
    print(f"rendered {payload['observed_strong_cells']} strong and 1 deduplicated weak family; "
          "no mechanism claim", flush=True)


if __name__ == "__main__":
    main()
