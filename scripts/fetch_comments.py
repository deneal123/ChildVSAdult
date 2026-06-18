"""CLI: дозагрузить комментарии постов и извлечь apparent-age (вспом. сигнал).

Комментарии — слабая вспомогательная supervision (apparent age), не основной сигнал
идентичности. Пишет data/processed/apparent_ages.jsonl: {post_id, n_comments, apparent_ages}.

    uv run python scripts/fetch_comments.py --limit 200
"""

from __future__ import annotations

import argparse
import re

from age_gap.common.io import data_path, read_jsonl, write_jsonl
from age_gap.common.logging import get_logger
from age_gap.datasets.apparent_age import collect_post_apparent_ages
from age_gap.parsing.vk_client import VKClient

log = get_logger(__name__)

_PID_RE = re.compile(r"vk_(-?\d+)_(\d+)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch comments -> apparent-age signal")
    parser.add_argument("--limit", type=int, default=200, help="Сколько постов обработать")
    parser.add_argument("--posts", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    posts_file = args.posts or str(data_path("data_dir", "raw", "posts.jsonl"))
    out_file = args.out or str(data_path("data_dir", "processed", "apparent_ages.jsonl"))

    client = VKClient()
    rows = []
    n_with = 0
    for i, post in enumerate(read_jsonl(posts_file)):
        if i >= args.limit:
            break
        m = _PID_RE.match(post["post_id"])
        if not m:
            continue
        comments = client.get_comments(int(m.group(1)), int(m.group(2)))
        ages = collect_post_apparent_ages(comments)
        if ages:
            n_with += 1
        rows.append(
            {"post_id": post["post_id"], "n_comments": len(comments), "apparent_ages": ages}
        )

    write_jsonl(out_file, rows)
    total = sum(len(r["apparent_ages"]) for r in rows)
    print(f"Готово: постов={len(rows)}, с apparent-age={n_with}, всего меток={total} -> {out_file}")


if __name__ == "__main__":
    main()
