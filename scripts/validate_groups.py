"""CLI: LLM-валидация целостности групп — single-identity vs мульти-человек/коллаж/мем (Part 2b).

Проходит по подписям мультифото-постов, классифицирует каждый (GigaChat, конкурентно),
кэширует (резюмируемо) и печатает распределение категорий. По ``--apply`` проставляет
``identity_review`` в группы (по source_post_id) — шумные (multi_person/collage/meme) помечаются.

    uv run python scripts/validate_groups.py                 # LLM-проход + отчёт
    uv run python scripts/validate_groups.py --limit 200      # проба
    uv run python scripts/validate_groups.py --apply          # + пометить группы
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path

from age_gap.common.io import append_jsonl, data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.common.schemas import IdentityGroup, RawPost
from age_gap.datasets.group_validator import (
    NOISY,
    SYSTEM_PROMPT,
    build_user_message,
    parse_validation,
)

log = get_logger(__name__)


def _cache_path() -> Path:
    return Path(resolve_path(str(data_path("data_dir", "interim", "group_validation_cache.jsonl"))))


def _load_posts(multi_only: bool, limit: int | None) -> list[RawPost]:
    out: list[RawPost] = []
    for row in read_jsonl(str(data_path("data_dir", "raw", "posts.jsonl"))):
        p = RawPost.from_dict(row)
        if not (p.caption or "").strip():
            continue
        if multi_only and len(p.photos) < 2:
            continue
        out.append(p)
        if limit and len(out) >= limit:
            break
    return out


async def _validate_async(posts: list[RawPost], cache_path: Path, concurrency: int) -> None:
    from age_gap.infrastructure.llm.client import get_client

    connector = get_client()
    sem = asyncio.Semaphore(concurrency)

    async def one(post: RawPost) -> tuple[RawPost, dict | None]:
        async with sem:
            messages = [
                ("system", SYSTEM_PROMPT),
                ("user", build_user_message((post.caption or "").strip(), len(post.photos))),
            ]
            try:
                content = await connector.chat(messages)
                result = parse_validation(content)
            except Exception as exc:  # noqa: BLE001 — единичный сбой не валит прогон
                log.warning("LLM fail %s: %s", post.post_id, exc)
                result = None
            return post, result

    def _write(post: RawPost, result: dict | None) -> None:
        append_jsonl(
            cache_path,
            {
                "post_id": post.post_id,
                "caption": post.caption,
                "n_photos": len(post.photos),
                **(result or {"category": "unknown", "confidence": 0.0, "reason": "llm_fail"}),
            },
        )

    if not posts:
        return
    post0, res0 = await one(posts[0])  # тёплый старт (OAuth без гонки)
    _write(post0, res0)
    chunk = 200
    done = 1
    for i in range(1, len(posts), chunk):
        batch = posts[i : i + chunk]
        results = await asyncio.gather(*(one(p) for p in batch))
        for post, result in results:
            _write(post, result)
        done += len(batch)
        log.info("LLM-валидация: %d/%d", done, len(posts))


def _report(cache_path: Path) -> Counter:
    cats: Counter = Counter()
    for row in read_jsonl(cache_path):
        cats[row.get("category", "unknown")] += 1
    return cats


def _apply(cache_path: Path) -> dict[str, int]:
    """Проставить identity_review группам по source_post_id; пометить шумные."""
    cat_by_post = {row["post_id"]: row.get("category", "unknown") for row in read_jsonl(cache_path)}
    groups_file = Path(
        resolve_path(str(data_path("data_dir", "processed", "identity_groups.jsonl")))
    )
    groups = [IdentityGroup.from_dict(r) for r in read_jsonl(str(groups_file))]
    stats = {"matched": 0, "noisy": 0}
    for g in groups:
        cat = cat_by_post.get(g.source_post_id)
        if cat is None:
            continue
        g.identity_review = cat
        stats["matched"] += 1
        if cat in NOISY:
            stats["noisy"] += 1
    backup = groups_file.with_suffix(".jsonl.pre_validate_bak")
    groups_file.replace(backup)
    from age_gap.common.io import write_jsonl

    write_jsonl(str(groups_file), (g.to_dict() for g in groups))
    log.info(
        "identity_review проставлен: matched=%d, noisy=%d (бэкап -> %s)",
        stats["matched"],
        stats["noisy"],
        backup.name,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM group-integrity validation (single vs noise)")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="включая одно-фото посты")
    parser.add_argument("--apply", action="store_true", help="проставить identity_review в группы")
    parser.add_argument("--skip-llm", action="store_true", help="только отчёт/apply из кэша")
    args = parser.parse_args()

    cache_path = _cache_path()
    if not args.skip_llm:
        posts = _load_posts(multi_only=not args.all, limit=args.limit)
        cached = {row.get("post_id") for row in read_jsonl(cache_path)}
        todo = [p for p in posts if p.post_id not in cached]
        log.info("Постов к LLM-валидации: %d (в кэше %d)", len(todo), len(cached))
        if todo:
            asyncio.run(_validate_async(todo, cache_path, args.concurrency))

    cats = _report(cache_path)
    total = sum(cats.values())
    print("\n=== категории постов (целостность групп) ===")
    for cat, n in cats.most_common():
        print(f"  {cat:<14} {n:>6} ({100 * n / total:.1f}%)")
    print(f"  всего {total}")

    if args.apply:
        _apply(cache_path)


if __name__ == "__main__":
    main()
