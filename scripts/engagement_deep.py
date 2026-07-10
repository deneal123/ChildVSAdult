"""GPU-модели «симпатии на показ»: bi-encoder + cross-encoder против CPU-бустинга.

    uv run python scripts/engagement_deep.py                       # then/now
    ENV_FOR_DYNACONF=natural uv run python scripts/engagement_deep.py
    ... --arch cross --epochs 10 --batch 24

Тот же таргет y_rate (остаток ставки-на-показ) и та же GroupKFold(person_id), что у
scripts/engagement_train.py, поэтому числа прямо сопоставимы. Пишет metrics/engagement_deep.json.
Веса не сохраняются; кеш патчей лица — под data*/cache (в .gitignore).
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from age_gap.common.io import data_path
from age_gap.common.logging import get_logger
from age_gap.engagement import deep
from age_gap.engagement import model as mdl
from age_gap.engagement import target as tgt

log = get_logger(__name__)

FULL = ["meta", "face", "domain", "emb"]


def _load(platform: str) -> pd.DataFrame:
    p = data_path("data_dir", "processed", f"engagement_features_{platform}.parquet")
    if not p.exists():
        raise FileNotFoundError(f"Нет {p}. Запустите scripts/engagement_build.py")
    return pd.read_parquet(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="vk")
    ap.add_argument("--arch", choices=["bi", "cross", "both"], default="both")
    ap.add_argument("--folds", type=int, default=mdl.N_SPLITS)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--quick", action="store_true", help="3 фолда, 4 эпохи — быстрая проверка")
    args = ap.parse_args()

    df = _load(args.platform)
    df, tinfo = tgt.add_targets(df)
    splits = mdl.make_splits(df, n_splits=3 if args.quick else args.folds)
    mdl.assert_no_group_leakage(df, splits)

    # ГЛАВНЫЙ таргет — тот же, что в engagement_train: остаток e_rate после экзогенного охвата.
    y, _ = tgt.oof_reach_residual(df, splits, variant="A", target="e_rate")

    device = deep.pick_device()
    log.info("device=%s | posts=%d persons=%d", device, len(df), df["person_id"].nunique())

    rows: list[dict] = []
    # CPU-бустинг на ИДЕНТИЧНЫХ y/сплитах — базовая линия для сравнения.
    hgb = mdl.evaluate(df, y, splits, FULL, "y_rate", "hgb").aggregate()
    hgb["device"] = "cpu"
    rows.append(hgb)
    log.info("HGB(cpu)   spearman=%.4f", hgb["oof_overall"]["spearman"])

    archs = ["bi", "cross"] if args.arch == "both" else [args.arch]
    epochs = 4 if args.quick else args.epochs
    for a in archs:
        res = deep.evaluate_deep(df, y, splits, a, device, epochs=epochs, batch=args.batch, lr=args.lr).aggregate()
        res["device"] = str(device)
        rows.append(res)
        log.info("%-16s spearman=%.4f", res["model"], res["oof_overall"]["spearman"])

    out = {
        "platform": args.platform,
        "n_posts": int(len(df)),
        "n_persons": int(df["person_id"].nunique()),
        "device": str(device),
        "text_model": deep.TEXT_MODEL,
        "vision_model": deep.VISION_MODEL,
        "vision_trained": False,
        "target_info": {"e_rate_vs_views": tinfo["spearman_e_rate_vs_log_views"]},
        "models": rows,
    }
    dst = data_path("metrics_dir", "engagement_deep.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print(json.dumps({
        "device": str(device),
        "comparison": {r["model"]: round(r["oof_overall"]["spearman"], 4) for r in rows},
        "ndcg@50": {r["model"]: round(r["oof_overall"]["ndcg@50"], 4) for r in rows},
        "metrics": str(dst),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
