"""Render the manuscript pipeline/yield diagram from the audited funnel JSON."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from age_gap.common.io import PROJECT_ROOT, data_path


def main() -> None:
    metrics = json.loads(data_path("metrics_dir", "data_funnel.json").read_text(encoding="utf-8"))
    funnel = metrics["funnel"]
    boxes = [
        ("Public then/now\nposts", f"{funnel['posts']:,} posts\n{funnel['photos']:,} photos"),
        ("RetinaFace +\nalignment", f"{funnel['face_records']:,} face records\n{funnel['usable_faces']:,} usable"),
        ("Caption +\nintegrity audit", "regex + versioned LLM\naudit pack: 5 x 400"),
        ("Person clustering\n+ dedup", f"cos >= 0.85 / 0.97\n{funnel['identity_groups']:,} persons"),
        ("Person-disjoint\nprotocol", f"{funnel['positive_pairs']:,} positive +\n{funnel['negative_pairs']:,} negative pairs"),
    ]
    fig, ax = plt.subplots(figsize=(10.0, 2.05))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for index, (title, detail) in enumerate(boxes):
        x = index * 2.0 + 0.08
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.35), 1.62, 1.25,
                boxstyle="round,pad=0.05,rounding_size=0.08",
                linewidth=1.1, edgecolor="#17324D",
                facecolor="#EDF3F8" if index % 2 == 0 else "#F9F1E8",
            )
        )
        ax.text(x + 0.81, 1.28, title, ha="center", va="center", fontsize=8.0, weight="bold")
        ax.text(x + 0.81, 0.75, detail, ha="center", va="center", fontsize=7.5)
        if index < len(boxes) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 1.65, 0.98), (x + 1.96, 0.98), arrowstyle="-|>",
                    mutation_scale=10, linewidth=1.0, color="#17324D",
                )
            )
    destination = PROJECT_ROOT / "latex" / "shared" / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "png"]:
        fig.savefig(destination / f"fig_pipeline.{extension}", bbox_inches="tight", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
