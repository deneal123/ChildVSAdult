"""Дозабор счётчиков вовлечённости (engagement) для уже собранных постов.

Зачем: в ``data/raw/posts.jsonl`` есть likes/reposts/views, но НЕТ комментариев
(массив пуст у всех постов) и нет точного времени публикации (только YYYY-MM-DD).
Час/день недели — сильные предикторы охвата, поэтому их надо вернуть.

VK: используем ``wall.get`` (а НЕ ``wall.getById`` — он недоступен сервисным токенам,
VK error 1051; корпус и собирался через wall.get). Каждый item отдаёт
``likes.count``, ``comments.count``, ``reposts.count``, ``views.count`` и unix ``date``.

Reddit: ``/api/info`` батчами по 100 fullname (``t3_<id>``) — отдаёт ``score``,
``num_comments``, ``upvote_ratio``, ``num_crossposts``, ``created_utc``.
Требует REDDIT_CLIENT_ID/SECRET; при их отсутствии плечо просто отключается.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from age_gap.common.io import data_path, ensure_parent, read_jsonl, resolve_path, write_jsonl
from age_gap.common.logging import get_logger

log = get_logger(__name__)

VK_PAGE = 100
REDDIT_BATCH = 100


def vk_engagement_path() -> Path:
    return data_path("data_dir", "interim", "post_engagement.jsonl")


def reddit_engagement_path() -> Path:
    # Явный литерал: не зависим от ENV_FOR_DYNACONF при смешанном (vk+reddit) прогоне.
    return resolve_path("data_reddit", "interim", "post_engagement.jsonl")


def _count(node: Any) -> int | None:
    """VK отдаёт счётчики как {"count": N}; отсутствие поля -> None."""
    if isinstance(node, dict) and "count" in node:
        return int(node["count"])
    return None


def vk_owner_ids(posts_path: Path) -> dict[str, int]:
    """Сообщества и число постов корпуса: post_id = ``vk_{owner_id}_{id}``."""
    owners: dict[str, int] = {}
    for row in read_jsonl(posts_path):
        parts = str(row.get("post_id", "")).split("_")
        if len(parts) == 3 and parts[0] == "vk":
            owners[parts[1]] = owners.get(parts[1], 0) + 1
    return owners


def _normalize_vk_item(item: dict[str, Any]) -> dict[str, Any] | None:
    pid = item.get("id")
    owner = item.get("owner_id", item.get("from_id"))
    if pid is None or owner is None:
        return None
    attachments = item.get("attachments") or []
    n_photo = sum(1 for a in attachments if a.get("type") == "photo")
    return {
        "post_id": f"vk_{owner}_{pid}",
        "platform": "vk",
        "owner_id": int(owner),
        "ts_unix": int(item.get("date") or 0) or None,
        "likes": _count(item.get("likes")) or 0,
        "comments": _count(item.get("comments")),
        "reposts": _count(item.get("reposts")) or 0,
        "views": _count(item.get("views")),
        "is_pinned": int(bool(item.get("is_pinned"))),
        "marked_as_ads": int(bool(item.get("marked_as_ads"))),
        "is_repost": int(bool(item.get("copy_history"))),
        "n_attach_photos": n_photo,
        "text_len": len(item.get("text") or ""),
    }


def _wall_pages(client: Any, owner_id: str, limit: int | None) -> Iterator[dict[str, Any]]:
    """Постранично обойти стену сообщества. Возвращает нормализованные записи."""
    offset = 0
    total: int | None = None
    seen = 0
    while True:
        resp = client._call("wall.get", {"owner_id": owner_id, "count": VK_PAGE, "offset": offset})
        if not isinstance(resp, dict):
            break
        if total is None:
            total = int(resp.get("count", 0))
            log.info("VK стена %s: всего постов на стене %s", owner_id, total)
        items = resp.get("items") or []
        if not items:
            break
        for item in items:
            row = _normalize_vk_item(item)
            if row is not None:
                yield row
                seen += 1
                if limit is not None and seen >= limit:
                    return
        offset += VK_PAGE
        if total and offset >= total:
            break


def backfill_vk(limit_per_owner: int | None = None, force: bool = False) -> dict[str, Any]:
    """Собрать engagement по всем сообществам корпуса -> data/interim/post_engagement.jsonl."""
    out = vk_engagement_path()
    if out.exists() and not force:
        log.info("VK backfill уже есть: %s (--force чтобы перезаписать)", out)
        rows = list(read_jsonl(out))
        return {"status": "skipped", "rows": len(rows), "path": str(out)}

    from age_gap.parsing.vk_client import VKClient  # ленивый импорт: нужен токен

    posts_path = data_path("data_dir", "raw", "posts.jsonl")
    owners = vk_owner_ids(posts_path)
    log.info("Сообщества корпуса: %s", owners)

    client = VKClient()
    collected: dict[str, dict[str, Any]] = {}
    for owner in owners:
        n_before = len(collected)
        for row in _wall_pages(client, owner, limit_per_owner):
            collected[row["post_id"]] = row  # dedupe (закреплённый пост дублируется)
        log.info("owner %s: +%d записей", owner, len(collected) - n_before)

    rows = list(collected.values())
    write_jsonl(out, rows)

    corpus_ids = {r["post_id"] for r in read_jsonl(posts_path)}
    matched = len(corpus_ids & set(collected))
    with_comments = sum(1 for r in rows if r.get("comments") is not None)
    stats = {
        "status": "ok",
        "rows": len(rows),
        "corpus_posts": len(corpus_ids),
        "matched_corpus_posts": matched,
        "coverage": round(matched / max(1, len(corpus_ids)), 4),
        "with_comments": with_comments,
        "path": str(out),
    }
    log.info("VK backfill: %s", stats)
    return stats


def _reddit_fullnames(posts_path: Path) -> list[str]:
    names = []
    for row in read_jsonl(posts_path):
        pid = str(row.get("post_id", ""))
        if pid.startswith("reddit_"):
            names.append("t3_" + pid[len("reddit_") :])
    return names


def _normalize_reddit_child(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "post_id": f"reddit_{data.get('id')}",
        "platform": "reddit",
        "owner_id": str(data.get("subreddit") or ""),
        "ts_unix": int(data.get("created_utc") or 0) or None,
        "likes": int(data.get("score") or 0),  # НЕТТО-голос (up-down), не «лайк»
        "comments": int(data.get("num_comments") or 0),
        "reposts": int(data.get("num_crossposts") or 0),
        "views": None,  # Reddit не отдаёт показы
        "upvote_ratio": data.get("upvote_ratio"),
        "is_pinned": int(bool(data.get("stickied"))),
        "marked_as_ads": 0,
        "is_repost": 0,
        "n_attach_photos": 0,
        "text_len": len((data.get("title") or "") + (data.get("selftext") or "")),
    }


def backfill_reddit(force: bool = False) -> dict[str, Any]:
    """Дозабрать score/num_comments для 396 Reddit-постов. Без кредов — graceful skip."""
    out = reddit_engagement_path()
    if out.exists() and not force:
        rows = list(read_jsonl(out))
        return {"status": "skipped", "rows": len(rows), "path": str(out)}

    posts_path = resolve_path("data_reddit", "raw", "posts.jsonl")
    if not posts_path.exists():
        return {"status": "no_data", "reason": f"нет {posts_path}"}

    try:
        from age_gap.parsing.reddit_client import RedditClient, load_config

        cfg = load_config()
        if not (cfg.client_id and cfg.client_secret):
            raise RuntimeError("REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET не заданы")
        client = RedditClient(cfg)
    except Exception as exc:  # noqa: BLE001 — плечо необязательное, отключаемся мягко
        log.warning("Reddit backfill отключён: %s", exc)
        return {"status": "disabled", "reason": str(exc)}

    names = _reddit_fullnames(posts_path)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(names), REDDIT_BATCH):
        batch = names[start : start + REDDIT_BATCH]
        try:
            resp = client._get("/api/info", {"id": ",".join(batch)})
        except Exception as exc:  # noqa: BLE001
            log.warning("Reddit /api/info батч %d упал: %s", start // REDDIT_BATCH, exc)
            continue
        children = (resp or {}).get("data", {}).get("children", [])
        for ch in children:
            if ch.get("kind") == "t3":
                rows.append(_normalize_reddit_child(ch.get("data", {})))

    if not rows:
        return {"status": "empty", "reason": "API не вернул постов", "requested": len(names)}

    ensure_parent(out)
    write_jsonl(out, rows)
    nonzero = sum(1 for r in rows if r["likes"] or r["comments"])
    stats = {
        "status": "ok",
        "rows": len(rows),
        "requested": len(names),
        "nonzero_engagement": nonzero,
        "path": str(out),
    }
    log.info("Reddit backfill: %s", stats)
    return stats


def load_engagement(platform: str = "vk") -> dict[str, dict[str, Any]]:
    """post_id -> engagement-запись (для feature-сборки)."""
    path = vk_engagement_path() if platform == "vk" else reddit_engagement_path()
    return {r["post_id"]: r for r in read_jsonl(path)}


def summarize(stats: dict[str, Any]) -> str:
    return json.dumps(stats, ensure_ascii=False, indent=2)
