"""Сборка post-level таблицы признаков для задачи «сила симпатии аудитории».

    uv run python scripts/engagement_build.py --platform vk

Пишет:
    data/processed/engagement_features_vk.parquet
    metrics/engagement_build.json   (статистика guardrails: сколько исключено и почему)
"""

from __future__ import annotations

import argparse
import json

from age_gap.common.io import data_path
from age_gap.common.logging import get_logger
from age_gap.engagement.features import build_table, feature_columns, save_table

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", choices=["vk", "reddit"], default="vk")
    ap.add_argument("--no-embeddings", action="store_true", help="не подмешивать ArcFace-эмбеддинги")
    args = ap.parse_args()

    df, stats = build_table(platform=args.platform, with_embeddings=not args.no_embeddings)
    path = save_table(df, platform=args.platform)

    blocks = feature_columns(df)
    payload = {
        "platform": args.platform,
        "table": {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "path": str(path)},
        "feature_blocks": {k: len(v) for k, v in blocks.items()},
        "guardrails": stats.as_dict(),
    }
    out = data_path("metrics_dir", "engagement_build.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    log.info("Готово: %s", out)


if __name__ == "__main__":
    main()
