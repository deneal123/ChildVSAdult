"""CLI: ингест мультифото-постов Reddit «then/now» (не-VK источник супервизии).

Два режима:
  --mode api  (по умолчанию): официальный/публичный ``.json``. Нужен OAuth ИЛИ резидентский
              IP; с датацентра и многих IP Reddit отдаёт 403.
  --mode html: скрейпинг HTML ``old.reddit.com`` (обходит блок .json). Работает через
              резидентский прокси (PROXY_* в .env). Рекомендуется, если .json даёт 403.

Примеры:
    # HTML-скрейпинг всего сабреддита (рекоменд. при 403 на .json)
    ENV_FOR_DYNACONF=reddit uv run python scripts/ingest_reddit.py --mode html --subreddit PastAndPresentPics

    # проверка одного поста
    uv run python scripts/ingest_reddit.py --mode html --post https://www.reddit.com/r/PastAndPresentPics/comments/1ucjmoj/then_vs_now/
"""

from __future__ import annotations

import argparse

from age_gap.parsing.reddit_ingest import ingest_reddit_post, ingest_subreddit


def _show(p) -> None:
    got = sum(1 for ph in p.photos if ph.local_path)
    print(f"{p.post_id}: фото={len(p.photos)} (скачано {got}) | caption={p.caption[:90]!r}")
    for ph in p.photos:
        print(f"  - {ph.photo_id} order={ph.order} local={ph.local_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Reddit then/now multi-photo posts")
    ap.add_argument("--mode", choices=["api", "html"], default="api",
                    help="api=.json (OAuth/residential); html=old.reddit HTML (обход 403)")
    ap.add_argument("--subreddit", default="PastAndPresentPics")
    ap.add_argument("--limit", type=int, default=1000, help="api: максимум постов на sort")
    ap.add_argument("--pages", type=int, default=10, help="html: страниц листинга на sort")
    ap.add_argument("--sorts", nargs="+", default=["top", "new", "hot"])
    ap.add_argument("--time-filter", default="all", help="api top: all|year|month|week")
    ap.add_argument("--append", action="store_true", help="дозапись к posts.jsonl активного root")
    ap.add_argument("--posts-out", default=None, help="путь posts.jsonl (по умолчанию <data_dir>/raw/posts.jsonl)")
    ap.add_argument("--images-dir", default=None, help="каталог изображений (по умолчанию <data_dir>/raw/images)")
    ap.add_argument("--post", default=None, help="ингест одного поста по ссылке (проверка парсера)")
    args = ap.parse_args()

    if args.post:
        if args.mode == "html":
            from age_gap.parsing.reddit_scrape import scrape_post

            p = scrape_post(args.post)
        else:
            p = ingest_reddit_post(args.post, images_dir=args.images_dir)
        if p:
            _show(p)
        else:
            print("Пусто (нет изображений или доступ заблокирован).")
        return

    if args.mode == "html":
        from age_gap.parsing.reddit_scrape import scrape_subreddit

        posts = scrape_subreddit(
            subreddit=args.subreddit,
            sorts=tuple(args.sorts),
            pages=args.pages,
            posts_out=args.posts_out,
            images_dir=args.images_dir,
            append=args.append,
        )
    else:
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
    print(f"Готово [{args.mode}]: {len(posts)} постов r/{args.subreddit} (мультифото: {multi})")


if __name__ == "__main__":
    main()
