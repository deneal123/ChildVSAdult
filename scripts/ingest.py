"""CLI: парсинг постов VK и загрузка изображений.

Два режима:
    1) Стена сообщества (wall.get, работает с сервисным токеном) — основной:
        uv run python scripts/ingest.py --owner -77072632 --count 100
    2) Конкретные посты по ссылкам (wall.getById, нужен НЕ сервисный токен):
        uv run python scripts/ingest.py --links docs/info.md
"""

from __future__ import annotations

import argparse

from age_gap.parsing.ingest import ingest, ingest_wall


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest VK posts -> RawPost JSONL + images")
    parser.add_argument("--owner", default=None, help="owner_id стены (сообщество — отрицательный)")
    parser.add_argument("--count", type=int, default=100, help="Сколько последних постов забрать")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--links",
        default=None,
        help="Файл со ссылками/ID постов (wall.getById). Альтернатива --owner.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Дозапись к posts.jsonl с дедупом (расширение датасета новой стеной).",
    )
    args = parser.parse_args()

    if args.owner:
        posts = ingest_wall(args.owner, count=args.count, offset=args.offset, append=args.append)
    elif args.links:
        posts = ingest(args.links)
    else:
        parser.error("Укажите --owner (стена) или --links (конкретные посты)")

    print(f"Готово: {len(posts)} постов, {sum(len(p.photos) for p in posts)} фото.")


if __name__ == "__main__":
    main()
