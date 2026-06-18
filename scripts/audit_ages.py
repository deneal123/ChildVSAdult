"""CLI: аудит и исправление парсинга возраста через GigaChat.

Проходит по ВСЕМ подписям постов, извлекает возраст LLM-ом (GigaChat, конкурентно), кэширует
результат (резюмируемо) и сравнивает с regex — печатает отчёт о расхождениях. По флагу
``--apply`` пересобирает identity_groups на LLM-возрастах (через CachedAgeExtractor, без
повторных сетевых вызовов), исправляя существующие данные.

    uv run python scripts/audit_ages.py                 # построить LLM-кэш + отчёт
    uv run python scripts/audit_ages.py --limit 200      # проба на 200 постах
    uv run python scripts/audit_ages.py --apply          # + пересобрать группы на LLM-возрастах

Концурентность по умолчанию 10 (под лимит токена GigaChat). Кэш и отчёт — UTF-8.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from age_gap.common.io import append_jsonl, data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.common.schemas import AgeLabel, RawPost
from age_gap.datasets.age_anchors import RegexAgeExtractor
from age_gap.datasets.llm_age_extractor import (
    SYSTEM_PROMPT,
    CachedAgeExtractor,
    _build_user_message,
    parse_llm_ages,
)

log = get_logger(__name__)


def _cache_path() -> Path:
    return Path(resolve_path(str(data_path("data_dir", "interim", "llm_age_cache.jsonl"))))


def _load_posts(posts_file: str | None, multi_only: bool, limit: int | None) -> list[RawPost]:
    posts_file = posts_file or str(data_path("data_dir", "raw", "posts.jsonl"))
    out: list[RawPost] = []
    for row in read_jsonl(posts_file):
        p = RawPost.from_dict(row)
        if not (p.caption or "").strip():
            continue
        if multi_only and len(p.photos) < 2:
            continue
        out.append(p)
        if limit and len(out) >= limit:
            break
    return out


async def _audit_async(posts: list[RawPost], cache_path: Path, concurrency: int) -> None:
    """Конкурентный LLM-проход; результаты дозаписываются чанками (резюмируемо)."""
    from age_gap.infrastructure.llm.client import get_client

    connector = get_client()
    sem = asyncio.Semaphore(concurrency)

    async def one(post: RawPost) -> tuple[RawPost, list[AgeLabel]]:
        async with sem:
            messages = [
                ("system", SYSTEM_PROMPT),
                ("user", _build_user_message((post.caption or "").strip(), len(post.photos))),
            ]
            try:
                content = await connector.chat(messages)
                labels = parse_llm_ages(content)
            except Exception as exc:  # noqa: BLE001 — единичный сбой не валит весь прогон
                log.warning("LLM fail %s: %s", post.post_id, exc)
                labels = []
            return post, labels

    def _write(post: RawPost, labels: list[AgeLabel]) -> None:
        append_jsonl(
            cache_path,
            {
                "post_id": post.post_id,
                "caption": post.caption,
                "n_photos": len(post.photos),
                "labels": [lbl.to_dict() for lbl in labels],
            },
        )

    if not posts:
        return
    # Тёплый старт: первый вызов отдельно (получение OAuth-токена без гонки при concurrency>1).
    post0, labels0 = await one(posts[0])
    _write(post0, labels0)

    chunk = 200
    done = 1
    for i in range(1, len(posts), chunk):
        batch = posts[i : i + chunk]
        results = await asyncio.gather(*(one(p) for p in batch))
        for post, labels in results:
            _write(post, labels)
        done += len(batch)
        log.info("LLM-аудит: %d/%d", done, len(posts))


def _report(cache_path: Path, report_path: Path) -> dict[str, int]:
    """Сравнить LLM-кэш с regex; вернуть статистику и записать расхождения (UTF-8)."""
    rx = RegexAgeExtractor()
    stats = {
        "total": 0,
        "agree_empty": 0,
        "agree_ages": 0,  # одинаковый набор возрастов
        "llm_filled": 0,  # regex пуст, LLM нашёл
        "llm_missed": 0,  # LLM пуст, regex нашёл
        "age_mismatch": 0,  # оба непусты, наборы возрастов разные
    }
    mismatches: list[dict] = []
    for row in read_jsonl(cache_path):
        stats["total"] += 1
        cap = row.get("caption") or ""
        n = row.get("n_photos")
        llm = [AgeLabel.from_dict(d) for d in row.get("labels", [])]
        rgx = rx.extract(cap, n)
        la = sorted(lbl.age for lbl in llm if lbl.age is not None)
        ra = sorted(lbl.age for lbl in rgx if lbl.age is not None)
        if not la and not ra:
            stats["agree_empty"] += 1
        elif not ra and la:
            stats["llm_filled"] += 1
            mismatches.append({"kind": "llm_filled", "caption": cap, "n": n, "regex": ra, "llm": la})
        elif ra and not la:
            stats["llm_missed"] += 1
            mismatches.append({"kind": "llm_missed", "caption": cap, "n": n, "regex": ra, "llm": la})
        elif la == ra:
            stats["agree_ages"] += 1
        else:
            stats["age_mismatch"] += 1
            mismatches.append(
                {"kind": "age_mismatch", "caption": cap, "n": n, "regex": ra, "llm": la}
            )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        for m in mismatches:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    return stats


def _apply(cache_path: Path) -> int:
    """Пересобрать identity_groups на LLM-возрастах (clean rebuild через CachedAgeExtractor)."""
    from age_gap.datasets.identity_groups import build_groups

    groups_out = Path(resolve_path(str(data_path("data_dir", "processed", "identity_groups.jsonl"))))
    if groups_out.exists():  # бэкап старых групп и чистая пересборка
        backup = groups_out.with_suffix(".jsonl.regex_bak")
        groups_out.replace(backup)
        log.info("Старые группы -> %s; пересобираю на LLM-возрастах", backup)
    extractor = CachedAgeExtractor.from_file(str(cache_path))
    new = build_groups(extractor=extractor, resume=False)
    log.info("Пересобрано групп: %d (cache hits=%d, misses=%d)", new, extractor.hits, extractor.misses)
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit/fix age parsing via GigaChat")
    parser.add_argument("--posts", default=None)
    parser.add_argument("--concurrency", type=int, default=10, help="параллельных запросов (лимит токена)")
    parser.add_argument("--limit", type=int, default=None, help="ограничить число постов (проба)")
    parser.add_argument("--all", action="store_true", help="включая одно-фото посты (по умолч. только >=2)")
    parser.add_argument("--apply", action="store_true", help="пересобрать группы на LLM-возрастах")
    parser.add_argument("--skip-audit", action="store_true", help="не звать LLM, только отчёт/apply из кэша")
    args = parser.parse_args()

    cache_path = _cache_path()
    if not args.skip_audit:
        posts = _load_posts(args.posts, multi_only=not args.all, limit=args.limit)
        cached = {row.get("post_id") for row in read_jsonl(cache_path)}
        todo = [p for p in posts if p.post_id not in cached]
        log.info("Постов к LLM-аудиту: %d (уже в кэше %d)", len(todo), len(cached))
        if todo:
            asyncio.run(_audit_async(todo, cache_path, args.concurrency))

    report_path = Path(resolve_path(str(data_path("data_dir", "interim", "age_audit_report.jsonl"))))
    stats = _report(cache_path, report_path)
    print("\n=== LLM vs regex (парсинг возраста) ===")
    for k, v in stats.items():
        print(f"  {k:<14} {v}")
    print(f"отчёт о расхождениях -> {report_path}")

    if args.apply:
        _apply(cache_path)


if __name__ == "__main__":
    main()
