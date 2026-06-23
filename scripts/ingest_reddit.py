"""CLI: ингест мультифото-постов Reddit «then/now» (не-VK источник супервизии).

Примеры:
    # весь сабреддит (top+new+hot, дедуп по id) -> data/raw/posts_reddit.jsonl
    uv run python scripts/ingest_reddit.py --subreddit PastAndPresentPics --limit 1000

    # проверка парсинга на одном посте (ничего не сохраняет, только качает его фото)
    uv run python scripts/ingest_reddit.py --post https://www.reddit.com/r/PastAndPresentPics/comments/1ucjmoj/then_vs_now/
"""

from __future__ import annotations

import argparse

from age_gap.parsing.reddit_ingest import ingest_reddit_post, ingest_subreddit


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Reddit then/now multi-photo posts")
    ap.add_argument("--subreddit", default="PastAndPresentPics")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--sorts", nargs="+", default=["top", "new", "hot"])
    ap.add_argument("--time-filter", default="all", help="для sort=top: all|year|month|week")
    ap.add_argument("--append", action="store_true", help="дозапись к posts.jsonl активного root")
    ap.add_argument("--posts-out", default=None, help="путь posts.jsonl (по умолчанию <data_dir>/raw/posts.jsonl)")
    ap.add_argument("--images-dir", default=None, help="каталог изображений (по умолчанию <data_dir>/raw/images)")
    ap.add_argument("--post", default=None, help="ингест одного поста по ссылке (проверка парсера)")
    args = ap.parse_args()

    if args.post:
        p = ingest_reddit_post(args.post, images_dir=args.images_dir)
        if p:
            got = sum(1 for ph in p.photos if ph.local_path)
            print(f"{p.post_id}: фото={len(p.photos)} (скачано {got}) | caption={p.caption[:90]!r}")
            for ph in p.photos:
                print(f"  - {ph.photo_id} order={ph.order} local={ph.local_path}")
        return

    posts = ingest_subreddit(
        subreddit=args.subreddit,
        limit=args.limit,
        sorts=tuple(args.sorts),
        time_filter=args.time_filter,
        append=args.append,
        posts_out=args.posts_out,
        images_dir=args.images_dir,
    )
    multi = sum(1 for p in posts if len(p.photos) >= 2)
    print(f"Готово: {len(posts)} постов r/{args.subreddit} (мультифото: {multi})")


if __name__ == "__main__":
    main()
