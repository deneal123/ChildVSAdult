"""Pure fresh-family LaTeX formatting; does not accept or publish native artifacts."""

from scripts.fresh_common_presentation_contract_v1 import WEAK_KEYS, collect
from scripts.render_common_mechanism_v2 import HEADER, _row, strong_label


def family_table(payload, family):
    if family not in {"strong", "weak"}:
        raise ValueError("explicit model family required")
    entry = payload["results"][family]
    models = entry["separability"]["models"]
    keys = payload["strong_cells"] if family == "strong" else WEAK_KEYS[1:]
    frozen = models["frozen"]
    rows = []
    for key in keys:
        label = (
            strong_label(key)
            if family == "strong"
            else f"fresh tuned, seed {key.removeprefix('tuned_s')}"
        )
        rows.append(_row(label, models[key], entry["diagnostics"][key]))
    scope = (
        "Strong arm: one explicitly selected fixed checkpoint, not the full 36-cell matrix. "
        if family == "strong"
        else "Fresh weak fixed8/head/b16 arm: three actual checkpoints shown separately, "
        "not averaged or ensembled and not recovery of the legacy weak headline. "
    )
    low, high = frozen["ci95"]
    caption = (
        scope
        + f"Frozen separation SMD is {frozen['standardized_mean_difference']:.4f} "
        + rf"(95\% CI $[{low:.4f},{high:.4f}]$). "
        + r"$\Delta$MAE and $\Delta$SMD are tuned minus frozen. Intervals are conditional "
        "paired subject/endpoint bootstrap intervals, not training-seed population intervals. "
        "Age intervals hold OOF probe predictions fixed, without refitting within draws. "
        "Lower age MAE means easier age decoding, not age removal. Different model architectures "
        "and preprocessing: descriptive, not controlled causal weak-versus-strong evidence. "
        "No mechanism decision, multiplicity correction, deployment calibration or human "
        "train--benchmark identity clearance."
    )
    return (
        "\n".join(
            (
                r"\begin{table*}[!t]",
                "\\caption{" + caption + "}",
                "\\label{tab:fresh-common-" + family + "}",
                r"\centering\scriptsize",
                r"\begin{tabular}{@{}lrrrrrrr@{}}",
                r"\toprule",
                HEADER,
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
            )
        )
        + "\n"
    )


def render(native):
    """Return validated payload and two strings; original-byte gates are upstream."""
    payload = collect(native)
    return payload, {
        f"fresh_{family}_table.tex": family_table(payload, family) for family in ("strong", "weak")
    }
