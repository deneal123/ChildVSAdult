"""CLI: publication-quality figures (IEEE T-BIOM) from the curated-data tables.

Numbers are hardcoded constants mirroring the paper's tables (docs/Results.md and
the paper draft), so the figures reproduce without re-running any experiment.
Output: vector PDF (for LaTeX) plus 300-dpi PNG (preview) in latex/figures/.

Styling targets IEEE / Q1 conventions:
  * column-width sizing so text renders at its intended size (no down-scaling);
  * serif (Times) text with embedded fonts (pdf.fonttype 42);
  * NO in-figure titles -- the LaTeX \\caption is the title;
  * a colour-blind-safe palette (Okabe-Ito) backed by distinct markers,
    line-styles and hatching, so the figures also read correctly in grayscale.

    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display

import matplotlib.pyplot as plt  # noqa: E402

from age_gap.common.io import resolve_path  # noqa: E402

OUT = resolve_path("latex", "figures")

# Okabe-Ito colour-blind-safe palette.
_FROZEN = "#9A9A9A"  # neutral grey  -- frozen baseline
_TUNED = "#0072B2"   # blue          -- +pairs
_ACCENT = "#D55E00"  # vermillion    -- synthetic / SOTA-objective / harmful
_GREEN = "#009E73"   # bluish green   -- spare

_CW = 3.4  # IEEE single-column width (inches)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.4,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(str(OUT / f"{name}.{ext}"))
    plt.close(fig)
    print(f"  -> {name}.pdf / .png")


def fig_scaling() -> None:
    """Data-scaling law (curated): metrics vs #train identities (early plateau)."""
    n = [1500, 3750, 7499, 14998]
    fgnet = [0.845, 0.853, 0.864, 0.848]
    our = [0.814, 0.842, 0.848, 0.838]
    fig, ax = plt.subplots(figsize=(_CW, 2.5))
    ax.plot(n, fgnet, "-o", color=_TUNED, label="FG-NET large-gap (external)", zorder=3)
    ax.plot(n, our, "--s", color=_ACCENT, label="our.25+ (internal)", zorder=3)
    ax.axhline(0.736, ls=":", color=_TUNED, lw=1.0, alpha=0.7)
    ax.axhline(0.640, ls=":", color=_ACCENT, lw=1.0, alpha=0.7)
    ax.text(n[-1], 0.741, "frozen", va="bottom", ha="right", color=_TUNED, fontsize=6.5)
    ax.text(n[-1], 0.635, "frozen", va="top", ha="right", color=_ACCENT, fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("Number of training identities (log)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.61, 0.885)
    ax.annotate(
        "~96% of the gain\nat 10% of identities",
        xy=(n[0], fgnet[0]),
        xytext=(n[0] * 1.22, 0.795),
        fontsize=6.8,
        arrowprops={"arrowstyle": "->", "lw": 0.7},
    )
    ax.legend(loc="lower right")
    _save(fig, "fig_scaling")


def fig_fairness() -> None:
    """Apparent-demographic audit (curated): all strata improve; strongest for 0-17."""
    strata = ["overall", "F", "M", "0-17", "18-29", "30-44", "45+"]
    frozen = [0.836, 0.840, 0.838, 0.750, 0.872, 0.822, 0.859]
    tuned = [0.916, 0.923, 0.897, 0.880, 0.936, 0.893, 0.893]
    x = range(len(strata))
    fig, ax = plt.subplots(figsize=(_CW, 2.6))
    w = 0.4
    ax.bar([i - w / 2 for i in x], frozen, w, color=_FROZEN, edgecolor="black", lw=0.4, label="frozen")
    ax.bar(
        [i + w / 2 for i in x], tuned, w, color=_TUNED, edgecolor="black", lw=0.4,
        hatch="////", label="+pairs",
    )
    for i in x:
        ax.text(i, tuned[i] + 0.004, f"+{tuned[i] - frozen[i]:.2f}", ha="center", va="bottom", fontsize=6.0)
    ax.axvspan(2.5, 3.5, color=_ACCENT, alpha=0.07, zorder=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(strata)
    ax.set_ylim(0.70, 1.0)
    ax.set_ylabel("ROC-AUC (test)")
    ax.set_xlabel("Stratum (apparent gender | apparent age of anchor)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.2, handletextpad=0.5)
    _save(fig, "fig_fairness")


def fig_headroom() -> None:
    """Headroom: +pairs gain on FG-NET large-gap vs frozen backbone strength (curated)."""
    bb = ["FaceNet", "ArcFace-r50", "AdaFace-IR50", "ArcFace-r100", "AdaFace-IR101"]
    frozen = [0.736, 0.764, 0.917, 0.954, 0.957]
    gain = [0.112, 0.041, 0.007, -0.039, -0.026]
    offs = [(0, 8), (0, 8), (6, 8), (0, -13), (0, 9)]
    fig, ax = plt.subplots(figsize=(_CW, 2.5))
    ax.fill_between([0.70, 1.0], 0, 0.13, color=_TUNED, alpha=0.06, zorder=0)
    ax.axhline(0, color="black", lw=0.7, zorder=1)
    ax.plot(frozen, gain, "-o", color=_TUNED, zorder=3)
    for f, g, name, off in zip(frozen, gain, bb, offs, strict=True):
        ax.annotate(name, (f, g), textcoords="offset points", xytext=off, fontsize=6.3, ha="center")
    ax.set_xlabel("Frozen FG-NET large-gap (backbone strength)")
    ax.set_ylabel(r"Gain from +pairs ($\Delta$ AUC)")
    ax.set_xlim(0.71, 0.99)
    ax.set_ylim(-0.075, 0.15)
    _save(fig, "fig_headroom")


def fig_shortcut() -> None:
    """Age shortcut (curated): frozen collapses under age-matched negatives; +pairs restores."""
    labels = ["frozen\n(random neg.)", "frozen\n(age-matched)", "+pairs\n(age-matched)"]
    vals = [0.640, 0.517, 0.838]
    colors = [_FROZEN, _ACCENT, _TUNED]
    hatches = ["", "xxx", "////"]
    fig, ax = plt.subplots(figsize=(2.95, 2.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", lw=0.4, width=0.62)
    for b, h in zip(bars, hatches, strict=True):
        b.set_hatch(h)
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.0)
    ax.axhline(0.5, ls=":", color="black", lw=0.9, alpha=0.6)
    ax.text(2.46, 0.506, "chance", va="bottom", ha="right", fontsize=6.3, alpha=0.7)
    ax.set_ylim(0.47, 0.90)
    ax.set_ylabel("our.25+ ROC-AUC (FaceNet)")
    _save(fig, "fig_shortcut")


def fig_sota() -> None:
    """Objective comparison (curated): ArcFace classification vs contrastive coincide cross-age."""
    metrics = ["FG-NET\nlarge-gap", "our.25+", "LFW", "AgeDB-30\nROC"]
    frozen = [0.736, 0.640, 0.969, 0.953]
    pairs = [0.848, 0.838, 0.946, 0.945]
    arcface = [0.851, 0.868, 0.952, 0.947]
    x = range(len(metrics))
    fig, ax = plt.subplots(figsize=(_CW, 2.6))
    w = 0.27
    ax.bar([i - w for i in x], frozen, w, color=_FROZEN, edgecolor="black", lw=0.4, label="frozen")
    ax.bar(
        list(x), pairs, w, color=_TUNED, edgecolor="black", lw=0.4, hatch="////",
        label="+pairs (contrastive)",
    )
    ax.bar(
        [i + w for i in x], arcface, w, color=_ACCENT, edgecolor="black", lw=0.4, hatch="xxx",
        label="+ArcFace (SOTA obj.)",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.60, 1.0)
    ax.set_ylabel("ROC-AUC / accuracy")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=0.9, handletextpad=0.4)
    _save(fig, "fig_sota")


def main() -> None:
    print(f"Figures -> {OUT}")
    fig_scaling()
    fig_fairness()
    fig_headroom()
    fig_shortcut()
    fig_sota()
    print("Done (5 figures used by the paper).")


if __name__ == "__main__":
    main()
