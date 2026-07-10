"""Обучение и честная валидация «симпатии сверх охвата».

    uv run python scripts/engagement_train.py
    uv run python scripts/engagement_train.py --quick        # меньше итераций/фолдов

Пишет:
    metrics/engagement_results.json          — все метрики, ablation, негативный контроль
    data/processed/engagement_oof.parquet    — OOF-предсказания для ранжирования в отчёте
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from age_gap.common.io import data_path
from age_gap.common.logging import get_logger
from age_gap.engagement import cluster as clu
from age_gap.engagement import model as mdl
from age_gap.engagement import target as tgt

log = get_logger(__name__)

LADDER = [
    (["meta"], "meta"),
    (["meta", "face"], "meta+face"),
    (["meta", "face", "domain"], "meta+face+domain"),
    (["meta", "face", "domain", "emb"], "meta+face+domain+emb"),
]
FULL = ["meta", "face", "domain", "emb"]


def _load(platform: str) -> pd.DataFrame:
    p = data_path("data_dir", "processed", f"engagement_features_{platform}.parquet")
    if not p.exists():
        alt = p.with_suffix(".csv.gz")
        if alt.exists():
            return pd.read_csv(alt)
        raise FileNotFoundError(f"Нет {p}. Запустите scripts/engagement_build.py")
    return pd.read_parquet(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="vk")
    ap.add_argument("--folds", type=int, default=mdl.N_SPLITS)
    ap.add_argument("--pca", type=int, default=mdl.N_PCA)
    ap.add_argument("--shuffle-target", action="store_true", help="только негативный контроль")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    df = _load(args.platform)
    df, tinfo = tgt.add_targets(df)
    log.info("Таблица: %s", df.shape)

    splits = mdl.make_splits(df, n_splits=3 if args.quick else args.folds)
    mdl.assert_no_group_leakage(df, splits)  # жёсткая проверка: person не пересекается

    results: dict[str, object] = {
        "platform": args.platform,
        "n_posts": int(len(df)),
        "n_persons": int(df["person_id"].nunique()),
        "targets_info": tinfo,
        "guardrail": "GroupKFold(person_id); PCA эмбеддингов фитится только на train",
    }

    # ---- 1. Сколько дисперсии съедает ОХВАТ -----------------------------------------
    resid_a, stats_a = tgt.oof_reach_residual(df, splits, variant="A")
    resid_b, stats_b = tgt.oof_reach_residual(df, splits, variant="B")
    df["y_symp_A"], df["y_symp_B"] = resid_a, resid_b
    results["reach"] = {"A_exogenous_only": stats_a, "B_also_controls_views": stats_b}

    if args.shuffle_target:
        y = tgt.shuffle_within_bucket(df, df["y_symp_A"].to_numpy())
        r = mdl.evaluate(df, y, splits, FULL, "y_symp_A_SHUFFLED", "hgb", args.pca)
        print(json.dumps(r.aggregate(), ensure_ascii=False, indent=2))
        return

    # ---- 2. Ablation-лестница на главном таргете ------------------------------------
    y_a = df["y_symp_A"].to_numpy()
    ablation = []
    oof_full = None
    for blocks, name in LADDER:
        r = mdl.evaluate(df, y_a, splits, blocks, "y_symp_A", "hgb", args.pca)
        ablation.append({"level": name, **r.aggregate()})
        log.info("ablation %-22s spearman=%.4f r2=%.4f", name,
                 r.aggregate()["oof_overall"]["spearman"], r.aggregate()["oof_overall"]["r2"])
        if blocks == FULL:
            oof_full = r
    results["ablation_y_symp_A"] = ablation

    # ---- 3. Другие таргеты и baseline'ы --------------------------------------------
    models: list[dict] = []
    for target_col, tname in [("e_raw", "e_raw (сырой композит)"),
                              ("y_symp_B", "y_symp_B (контроль views)"),
                              ("y_pct", "y_pct (перцентиль в бакете)")]:
        r = mdl.evaluate(df, df[target_col].to_numpy(), splits, FULL, tname, "hgb", args.pca)
        models.append(r.aggregate())
    for mname in ["ridge", "dummy"]:
        r = mdl.evaluate(df, y_a, splits, FULL, "y_symp_A", mname, args.pca)
        models.append(r.aggregate())
    results["models"] = models

    # ---- 4. Негативный контроль -----------------------------------------------------
    y_shuf = tgt.shuffle_within_bucket(df, y_a.copy())
    r_shuf = mdl.evaluate(df, y_shuf, splits, FULL, "y_symp_A_SHUFFLED", "hgb", args.pca)
    results["negative_control"] = r_shuf.aggregate()
    log.info("негативный контроль: spearman=%.4f (ожидаем ~0)",
             r_shuf.aggregate()["oof_overall"]["spearman"])

    # ---- 5. Важность признаков ------------------------------------------------------
    results["permutation_importance"] = mdl.permutation_importance_oof(
        df, y_a, splits, FULL, n_pca=args.pca, n_repeats=3 if args.quick else 5
    )

    # ---- 6. Кластеры (без обучения) --------------------------------------------------
    results["clusters"] = clu.cluster_archetypes(df, y_a)
    labels = results["clusters"].pop("labels")

    # ---- 7. OOF-предсказания для ранжирования в отчёте --------------------------------
    assert oof_full is not None
    out = pd.DataFrame({
        "post_id": df["post_id"], "person_id": df["person_id"], "owner_id": df["owner_id"],
        "cluster": labels,
        "likes": df["likes"], "comments": df["comments"], "reposts": df["reposts"],
        "views": df["views"], "e_raw": df["e_raw"],
        "y_symp_A": y_a, "pred_symp_A": oof_full.oof_pred,
        "n_photos": df["n_photos"], "n_adult_usable": df["n_adult_usable"],
        "has_child_photo": df["has_child_photo"], "age_gap_label": df["age_gap_label"],
        "age_est_median": df["age_est_median"], "share_female_adult": df["share_female_adult"],
        "ts_unix": df["ts_unix"], "post_age_days": df["post_age_days"],
    })
    oof_path: Path = data_path("data_dir", "processed", "engagement_oof.parquet")
    try:
        out.to_parquet(oof_path, index=False)
    except Exception:  # noqa: BLE001
        oof_path = oof_path.with_suffix(".csv.gz")
        out.to_csv(oof_path, index=False, compression="gzip")
    results["oof_path"] = str(oof_path)

    dst = data_path("metrics_dir", "engagement_results.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    o = oof_full.aggregate()["oof_overall"]
    print(json.dumps({
        "reach_R2_A": stats_a["reach_r2_oof"],
        "reach_R2_B": stats_b["reach_r2_oof"],
        "content_on_residual_A": {k: o[k] for k in ["r2", "spearman", "ndcg@50", "lift@50"]},
        "negative_control_spearman": r_shuf.aggregate()["oof_overall"]["spearman"],
        "metrics": str(dst),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
