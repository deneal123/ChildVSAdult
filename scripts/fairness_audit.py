"""CLI: демографический аудит прироста +pairs по полу/возрасту (Q1, FR-этика).

Считает apparent пол+возраст (insightface genderage, CPU) для лиц test-сплита, затем сравнивает
frozen vs дообученный backbone по стратам: важно, что прирост не достаётся одной группе за счёт
другой. См. evaluation/fairness.py.

    uv run python scripts/fairness_audit.py --backbone facenet --tuned models/bb_facenet_seed42.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, read_jsonl
from age_gap.evaluation.fairness import compute_face_attributes, print_audit, stratified_audit
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import load_finetuned


def main() -> None:
    parser = argparse.ArgumentParser(description="Demographic fairness audit (gain by gender/age)")
    parser.add_argument("--backbone", default="facenet")
    parser.add_argument(
        "--tuned", default=None, help="чекпойнт дообученного (default bb_<bb>_seed42.pt)"
    )
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    device = torch_device()
    models = data_path("models_dir")
    bb = args.backbone
    tuned_ckpt = Path(args.tuned) if args.tuned else Path(str(models / f"bb_{bb}_seed42.pt"))

    # Лица сплита (для которых считаем apparent атрибуты).
    pairs_file = str(data_path("data_dir", "processed", "pairs.jsonl"))
    face_ids = sorted(
        {
            f
            for r in read_jsonl(pairs_file)
            if r.get("split") == args.split
            for f in (r["face_a"], r["face_b"])
        }
    )

    # Фаза 1: apparent пол+возраст (CPU onnx — не конфликтует с torch-CUDA), резюмируемо.
    attrs = compute_face_attributes(face_ids, device="cpu")

    # Фаза 2: torch-бэкбоны на GPU.
    frozen = make_backbone(bb, pretrained=True).to(device).eval()
    tuned = load_finetuned(tuned_ckpt, device)

    result = stratified_audit(frozen, tuned, device, attrs, split=args.split)
    print_audit(result)


if __name__ == "__main__":
    main()
