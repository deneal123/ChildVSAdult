"""Pure TeX projection of accepted fixed-checkpoint means, not native replay."""
import math

GROUPS = ("random_head_lr1e-06", "random_head_lr1e-05",
          "random_tail_lr1e-06", "random_tail_lr1e-05")
METRICS = ("roc_auc", "eer_interpolated", "eer_discrete_minimax", "tar@far=0.01", "tar@far=0.001")
LABELS = ("ROC-AUC", "Interpolated EER", "Discrete minimax EER", "TAR@FAR=1\\%", "TAR@FAR=0.1\\%")


def number(value):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("finite native float required")
    return f"{value:.4f}"


def table(summary, stratum):
    if stratum not in ("overall", "large_gap_25plus"):
        raise ValueError("declared stratum required")
    if summary.get("publication_ready") is not False or summary.get("full_matrix_complete") is not False:
        raise ValueError("partial-matrix scope must remain explicit")
    if set(summary["groups"]) != set(GROUPS):
        raise ValueError("exact four completed configurations required")
    rows = []
    for group in GROUPS:
        cell = summary["groups"][group][stratum]
        if cell["fixed_seed_list"] != [42, 1, 2] or any(type(x) is not int for x in cell["fixed_seed_list"]):
            raise ValueError("exact integer fixed seeds required")
        if cell["publication_ready"] is not False or cell["training_identity_independence"] != "unverified":
            raise ValueError("identity/publication gate remains open")
        if cell["bootstrap"]["shared_weights_across_models"] is not True:
            raise ValueError("joint weights required")
        if set(cell["aggregate"]) != set(METRICS) or set(cell["frozen"]) != set(METRICS):
            raise ValueError("exact ROC-v2 metric set required")
        scope = "Head" if "_head_" in group else "Tail"
        rate = "10^{-6}" if "1e-06" in group else "10^{-5}"
        for name, label in zip(METRICS, LABELS, strict=True):
            value = cell["aggregate"][name]
            lo, hi = value["delta_ci95"]
            number(lo)
            number(hi)
            if lo > hi or value["sd"] < 0:
                raise ValueError("ordered CI/nonnegative descriptive SD required")
            fields = [f"{scope}, $\\mathrm{{LR}}={rate}$", label,
                      number(cell["frozen"][name]["point"]),
                      f"{number(value['mean'])}$\\pm${number(value['sd'])}",
                      f"{value['mean_checkpoint_delta']:+.4f}",
                      f"[{lo:+.4f}, {hi:+.4f}]"]
            number(value["mean_checkpoint_delta"])
            number(hi)
            rows.append(" & ".join(fields) + r" \\")
    return "\n".join([
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Configuration & Metric & Frozen & Mean$\pm$SD & $\Delta$ & Paired mean-$\Delta$ 95\% CI \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", "",
    ])
