"""CLI: честный эксперимент «помогают ли наши данные».

Слабый backbone (facenet casia-webface): zero-shot на бенчмарке -> дообучение на наших
парах -> бенчмарк снова. Печатает Δ.

Бенчмарк: LFW (sklearn, автозагрузка) + любые insightface .bin из data/external/
(agedb_30.bin/calfw.bin — cross-age) + наш held-out cross-age test.

    uv run python scripts/finetune_facenet.py --epochs 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.evaluation.benchmark_external import evaluate_bin, evaluate_lfw
from age_gap.models.facenet import FaceNetBackbone
from age_gap.training.finetune import evaluate_our_split, finetune, load_finetuned


def _eval_all(backbone, device) -> dict[str, dict]:
    res: dict[str, dict] = {}
    res["LFW"] = evaluate_lfw(backbone, device)
    ext_dir = Path(str(data_path("data_dir", "external")))
    for bin_path in sorted(ext_dir.glob("*.bin")):
        res[bin_path.stem] = evaluate_bin(backbone, device, bin_path)
    res["our_test"] = evaluate_our_split(backbone, device)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen vs fine-tuned facenet on benchmarks")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--trainable",
        choices=["head", "tail", "full"],
        default="tail",
        help="Какие слои размораживать: head (мягко) | tail | full",
    )
    parser.add_argument(
        "--ckpt", default=None, help="Путь чекпойнта (по умолч. facenet_finetuned.pt)"
    )
    args = parser.parse_args()

    device = torch_device()
    ckpt_out = Path(args.ckpt) if args.ckpt else None

    print("== Zero-shot (frozen casia-webface) ==")
    zero = _eval_all(FaceNetBackbone(pretrained="casia-webface").to(device).eval(), device)

    ckpt = finetune(
        epochs=args.epochs, lr=args.lr, trainable_scope=args.trainable, ckpt_out=ckpt_out
    )
    after = _eval_all(load_finetuned(ckpt, device), device)

    # Сводная таблица Δ.
    def g(d, k, m):
        return d.get(k, {}).get(m)

    print(f"\n{'benchmark':<12}{'metric':<16}{'frozen':>10}{'finetuned':>11}{'Δ':>9}")
    rows = [
        ("LFW", "accuracy_10fold"),
        ("LFW", "roc_auc"),
        ("our_test", "overall_auc"),
        ("our_test", "large_gap_auc"),
    ]
    for bench in sorted(
        k for k in zero if k.endswith("lfw") or k in ("agedb_30", "calfw", "cplfw")
    ):
        rows.append((bench, "accuracy_10fold"))
        rows.append((bench, "roc_auc"))
    for bench, metric in rows:
        b, a = g(zero, bench, metric), g(after, bench, metric)
        if b is None or a is None:
            continue
        print(f"{bench:<12}{metric:<16}{b:>10.4f}{a:>11.4f}{a - b:>+9.4f}")


if __name__ == "__main__":
    main()
