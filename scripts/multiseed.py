"""CLI: статистическая значимость headline-результата (несколько сидов) — для Q1.

Дообучает backbone на нескольких сидах, считает mean±std ключевых метрик (frozen — детерминирован,
оценивается один раз). Даёт доверительный разброс выигрыша +pairs.

    uv run python scripts/multiseed.py --backbone facenet --seeds 42 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.common.manifest import write_experiment_manifest
from age_gap.evaluation.external_suite import METRICS, eval_all
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import finetune, load_finetuned


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed significance of +pairs gain")
    parser.add_argument("--backbone", default="facenet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    parser.add_argument(
        "--reuse", action="store_true", help="не переобучать: оценить готовые bb_<bb>_seed<s>.pt"
    )
    args = parser.parse_args()

    device = torch_device()
    models = data_path("models_dir")
    bb = args.backbone

    frozen = eval_all(make_backbone(bb, pretrained=True).to(device).eval(), device)

    runs: list[dict[str, float]] = []
    checkpoints: list[Path] = []
    for s in args.seeds:
        ck = Path(str(models / f"bb_{bb}_seed{s}.pt"))
        if not (args.reuse and ck.exists()):
            ck = finetune(
                epochs=args.epochs,
                lr=args.lr,
                trainable_scope="head",
                backbone_name=bb,
                seed=s,
                ckpt_out=ck,
            )
        checkpoints.append(ck)
        runs.append(eval_all(load_finetuned(ck, device), device))

    # ASCII-вывод (Windows-консоль cp1251 не кодирует Δ).
    print(f"\n{bb}: frozen vs +pairs (seeds={args.seeds})")
    print(f"{'metric':<18}{'frozen':>10}{'+pairs_mean':>14}{'std':>9}{'gain':>9}")
    aggregates: dict[str, dict[str, float | int | list[float]]] = {}
    for m in METRICS:
        if m not in frozen:
            continue
        vals = np.array([r.get(m, float("nan")) for r in runs], dtype=float)
        mean, std = float(np.nanmean(vals)), float(np.nanstd(vals))
        aggregates[m] = {
            "frozen": float(frozen[m]),
            "mean": mean,
            "std": std,
            "gain": mean - float(frozen[m]),
            "n": int(np.isfinite(vals).sum()),
            "values": vals.tolist(),
        }
        print(f"{m:<18}{frozen[m]:>10.4f}{mean:>14.4f}{std:>9.4f}{mean - frozen[m]:>+9.4f}")

    payload = {
        "backbone": bb,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "seeds": args.seeds,
        "frozen": frozen,
        "runs": {str(seed): row for seed, row in zip(args.seeds, runs, strict=True)},
        "aggregates": aggregates,
    }
    output = Path(str(data_path("metrics_dir", f"multiseed_{bb}_e{args.epochs}.json")))
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    external = Path(str(data_path("data_dir", "external")))
    inputs = [
        Path(str(data_path("data_dir", "processed", "pairs.jsonl"))),
        Path(str(data_path("metrics_dir", "model_inventory.json"))),
        *checkpoints,
        *[
            path
            for path in (
                external / "lfw_aligned.npz",
                external / "agedb_30.bin",
                external / "calfw.bin",
                external / "fgnet_crops.npz",
                external / "cacd_vs_aligned.npz",
            )
            if path.is_file()
        ],
    ]
    write_experiment_manifest(
        output.with_suffix(".manifest.json"),
        experiment="main-effect-multiseed-external-suite",
        parameters={
            "backbone": bb,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "trainable_scope": "head",
            "seeds": args.seeds,
            "reused_checkpoints": args.reuse,
        },
        metrics=payload,
        inputs=inputs,
        outputs=[output],
    )
    print(f"machine-readable summary -> {output}")


if __name__ == "__main__":
    main()
