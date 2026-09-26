"""Recompute the exploratory frozen-vs-tuned backbone headroom table.

This binds every manuscript value to the exact frozen weight inventory, tuned
checkpoint and shared evaluation suite. It is intentionally an exploratory
one-seed audit; the separate strong-backbone study supplies the confirmatory
three-seed mechanism matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.common.manifest import write_experiment_manifest
from age_gap.evaluation.external_suite import eval_all
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import load_finetuned

BACKBONES = (
    ("facenet", "bb_facenet_seed42.pt"),
    ("arcface_r50_casia", "bb_arcface_r50_casia_pairs.pt"),
    ("adaface_ir50", "bb_adaface_ir50_pairs.pt"),
    ("arcface_r100", "bb_arcface_r100_pairs.pt"),
    ("adaface_ir101", "bb_adaface_ir101_pairs.pt"),
)
METRICS = ("fgnet.large_gap", "our.25+")


def main() -> None:
    device = torch_device()
    models = Path(str(data_path("models_dir")))
    rows: dict[str, dict] = {}
    checkpoints: list[Path] = []

    for backbone_name, checkpoint_name in BACKBONES:
        checkpoint = models / checkpoint_name
        checkpoints.append(checkpoint)
        frozen_model = make_backbone(backbone_name, pretrained=True).to(device).eval()
        frozen = eval_all(frozen_model, device)
        del frozen_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tuned_model = load_finetuned(checkpoint, device)
        tuned = eval_all(tuned_model, device)
        del tuned_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rows[backbone_name] = {
            "checkpoint": checkpoint_name,
            "frozen": {key: float(frozen[key]) for key in METRICS},
            "tuned": {key: float(tuned[key]) for key in METRICS},
            "delta": {key: float(tuned[key] - frozen[key]) for key in METRICS},
        }
        print(
            backbone_name,
            " ".join(
                f"{key}={frozen[key]:.4f}->{tuned[key]:.4f}"
                for key in METRICS
            ),
            flush=True,
        )

    payload = {
        "protocol": "exploratory one-seed frozen-vs-pairs comparison",
        "metrics": list(METRICS),
        "rows": rows,
    }
    output = Path(str(data_path("metrics_dir", "headroom_audit.json")))
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    external = Path(str(data_path("data_dir", "external")))
    write_experiment_manifest(
        output.with_suffix(".manifest.json"),
        experiment="exploratory-backbone-headroom-audit",
        parameters={
            "backbones": [name for name, _ in BACKBONES],
            "metrics": list(METRICS),
            "scope": "exploratory one-seed; not confirmatory mechanism evidence",
        },
        metrics=payload,
        inputs=[
            Path(str(data_path("metrics_dir", "model_inventory.json"))),
            Path(str(data_path("data_dir", "processed", "pairs.jsonl"))),
            *checkpoints,
            *[
                path
                for path in (
                    external / "fgnet_crops.npz",
                    external / "lfw_aligned.npz",
                    external / "agedb_30.bin",
                    external / "calfw.bin",
                )
                if path.is_file()
            ],
        ],
        outputs=[output],
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
