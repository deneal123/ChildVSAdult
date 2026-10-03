"""Render the manuscript pipeline/yield diagram from the audited funnel JSON."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from age_gap.common.io import PROJECT_ROOT, data_path
from age_gap.common.manifest import write_experiment_manifest


def load_bound_funnel(path: Path) -> dict[str, int]:
    """Verify the aggregate output binding, not private-input provenance or labels."""
    payload = path.read_bytes()
    metrics = json.loads(payload)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", [])
    if len(outputs) != 1 or outputs[0].get("bytes") != len(payload) or outputs[0].get(
        "sha256"
    ) != hashlib.sha256(payload).hexdigest():
        raise ValueError("Funnel aggregate checksum/size binding is invalid")
    if manifest.get("metrics") != metrics:
        raise ValueError("Funnel manifest metrics differ from the aggregate")
    funnel = metrics["funnel"]
    required = (
        "posts", "photos", "face_records", "usable_faces", "rejected_face_records",
        "curated_faces", "identity_groups", "person_clusters", "positive_pairs",
        "negative_pairs", "final_pairs",
    )
    if any(type(funnel.get(key)) is not int or funnel[key] < 0 for key in required):
        raise ValueError("Funnel counts must be nonnegative integers")
    if funnel["face_records"] != funnel["usable_faces"] + funnel["rejected_face_records"]:
        raise ValueError("Face-record accounting is inconsistent")
    if funnel["final_pairs"] != funnel["positive_pairs"] + funnel["negative_pairs"]:
        raise ValueError("Pair accounting is inconsistent")
    if funnel["curated_faces"] > funnel["usable_faces"]:
        raise ValueError("Curated faces exceed usable faces")
    return funnel


def pipeline_boxes(funnel: dict[str, int]) -> list[tuple[str, str]]:
    """Keep caption grouping/merging before integrity pruning; no human-audit yield."""
    return [
        ("Public then/now\nposts", f"{funnel['posts']:,} posts\n{funnel['photos']:,} photos"),
        ("RetinaFace +\nalignment", f"{funnel['face_records']:,} face records\n{funnel['usable_faces']:,} usable"),
        ("Caption groups +\nperson merging", f"regex / cached LLM\ncos >= 0.85\n{funnel['person_clusters']:,} mapped clusters"),
        ("Integrity prune +\ndeduplication", f"cos >= 0.97\n{funnel['curated_faces']:,} faces\n{funnel['identity_groups']:,} retained groups"),
        ("Recorded-person\nsplit / evaluation", f"{funnel['positive_pairs']:,} positive +\n{funnel['negative_pairs']:,} negative pairs"),
    ]


def main() -> None:
    source = Path(data_path("metrics_dir", "data_funnel.json"))
    bound_inputs = [source, source.with_suffix(".manifest.json"), Path(__file__)]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in bound_inputs]
    funnel = load_bound_funnel(source)
    boxes = pipeline_boxes(funnel)
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
    # Detect changed aggregate/manifest before declaring the rendered output bound.
    after = [hashlib.sha256(path.read_bytes()).hexdigest() for path in bound_inputs]
    if after != before or load_bound_funnel(source) != funnel:
        raise ValueError("Presentation inputs changed during rendering")
    write_experiment_manifest(
        Path(data_path("metrics_dir", "pipeline_figure.manifest.json")),
        experiment="pipeline-figure-presentation",
        parameters={"scope": "aggregate presentation, not historical preprocessing replay"},
        metrics={"funnel": funnel},
        inputs=bound_inputs,
        outputs=[destination / "fig_pipeline.pdf", destination / "fig_pipeline.png"],
    )


if __name__ == "__main__":
    main()
