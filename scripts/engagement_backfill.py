"""Дозабор engagement-счётчиков (лайки/комментарии/репосты/показы) для собранных постов.

Примеры:
    uv run python scripts/engagement_backfill.py --platform vk --limit 300   # дымовой прогон
    uv run python scripts/engagement_backfill.py --platform vk --force       # полный обход
    uv run python scripts/engagement_backfill.py --platform reddit           # 396 постов

VK идёт через wall.get (wall.getById недоступен сервисным токенам, VK error 1051).
Reddit требует REDDIT_CLIENT_ID/SECRET; без них плечо отключается (status=disabled).
"""

from __future__ import annotations

import argparse

from age_gap.common.logging import get_logger
from age_gap.engagement.backfill import backfill_reddit, backfill_vk, summarize

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", choices=["vk", "reddit", "both"], default="vk")
    ap.add_argument("--limit", type=int, default=None, help="макс. постов на сообщество (VK, дымовой прогон)")
    ap.add_argument("--force", action="store_true", help="перезаписать существующий файл")
    args = ap.parse_args()

    results: dict[str, object] = {}
    if args.platform in ("vk", "both"):
        results["vk"] = backfill_vk(limit_per_owner=args.limit, force=args.force)
    if args.platform in ("reddit", "both"):
        results["reddit"] = backfill_reddit(force=args.force)

    print(summarize(results))


if __name__ == "__main__":
    main()
