"""CLI: age-leakage probe (E20) — frozen vs +pairs vs +disentangle.

Декодируемость apparent-возраста из identity-эмбеддинга (линейный probe) + identity overall AUC.
genderage берётся из кэша (CPU); энкод — на GPU. См. evaluation/age_leakage.py.

    uv run python scripts/age_leakage.py
"""

from __future__ import annotations

import json
from pathlib import Path

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, read_jsonl
from age_gap.common.manifest import write_experiment_manifest
from age_gap.common.schemas import Pair
from age_gap.evaluation.age_leakage import print_leakage, summarize
from age_gap.evaluation.fairness import _age_band, _encode_faces, compute_face_attributes
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import load_finetuned


def main() -> None:
    device = torch_device()
    models = data_path("models_dir")
    bb = "facenet"
    split = "test"

    pairs_file = str(data_path("data_dir", "processed", "pairs.jsonl"))
    pairs = [Pair.from_dict(r) for r in read_jsonl(pairs_file) if r.get("split") == split]
    face_ids = sorted({p.face_a for p in pairs} | {p.face_b for p in pairs})

    # apparent-возраст -> бакет (genderage из кэша; CPU).
    attrs = compute_face_attributes(face_ids, device="cpu")
    face_bucket = {fid: _age_band(age) for fid, (_g, age) in attrs.items()}

    specs = [
        ("frozen", make_backbone(bb, pretrained=True).to(device).eval()),
        ("+pairs", load_finetuned(Path(str(models / f"bb_{bb}_seed42.pt")), device)),
        ("+disentangle", load_finetuned(Path(str(models / f"bb_{bb}_disentangle.pt")), device)),
    ]
    rows = []
    for name, backbone in specs:
        emb = _encode_faces(backbone, device, face_ids)
        rows.append(summarize(name, emb, face_bucket, pairs))
    print_leakage(rows)
    output = Path(str(data_path("metrics_dir", "age_leakage.json")))
    payload = {
        "backbone": bb,
        "split": split,
        "probe_seeds": [42, 1, 2],
        "probe_train_fraction": 0.7,
        "age_bands": ["0-17", "18-29", "30-44", "45+"],
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    attribute_cache = Path(str(data_path("data_dir", "interim", "face_genderage.jsonl")))
    model_inventory = Path(str(data_path("metrics_dir", "model_inventory.json")))
    write_experiment_manifest(
        output.with_suffix(".manifest.json"),
        experiment="apparent-age-leakage-linear-probe",
        parameters={
            "backbone": bb,
            "split": split,
            "probe_seeds": [42, 1, 2],
            "probe_train_fraction": 0.7,
            "probe_model": "standardized multinomial logistic regression",
            "attribute_source": "InsightFace buffalo_l genderage.onnx",
        },
        metrics=payload,
        inputs=[
            Path(pairs_file),
            attribute_cache,
            model_inventory,
            Path(str(models / f"bb_{bb}_seed42.pt")),
            Path(str(models / f"bb_{bb}_disentangle.pt")),
        ],
        outputs=[output],
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
