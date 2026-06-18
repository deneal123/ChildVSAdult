"""Парсинг постов VK в RawPost + загрузка изображений.

Поток (TODO §12 Phase 2):
    ссылки на посты -> wall.getById -> нормализация в RawPost -> скачивание фото ->
    сохранение data/raw/posts.jsonl + data/raw/images/<post_id>/<photo_id>.jpg

Для каждого поста заполняются provenance-поля (SKILL §9.4).
"""

from __future__ import annotations

import datetime as _dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from age_gap.common.io import append_jsonl, data_path, read_jsonl, write_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import Comment, Photo, Provenance, RawPost
from age_gap.parsing.vk_client import VKClient, best_photo_url, parse_wall_id

log = get_logger(__name__)


def read_links(path: Path | str) -> list[str]:
    """Прочитать ссылки/ID постов из текстового файла (по одному в строке).

    Совместимо с docs/info.md: пустые строки и строки без ``wall`` игнорируются.
    """
    links: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or "wall" not in line:
            continue
        try:
            parse_wall_id(line)
        except ValueError:
            continue
        links.append(line)
    return links


def _unix_to_date(ts: int | None) -> str | None:
    if not ts:
        return None
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).date().isoformat()


def normalize_post(item: dict[str, Any], origin_url: str | None, collection_date: str) -> RawPost:
    owner_id = item.get("owner_id") or item.get("from_id")
    post_inner_id = item.get("id")
    post_id = f"vk_{owner_id}_{post_inner_id}"

    photos: list[Photo] = []
    for order, att in enumerate(item.get("attachments", [])):
        if att.get("type") != "photo":
            continue
        ph = att["photo"]
        photos.append(
            Photo(
                photo_id=f"{ph.get('owner_id')}_{ph.get('id')}",
                url=best_photo_url(ph),
                order=order,
            )
        )

    comments = [
        Comment(
            comment_id=str(c.get("id")), text=c.get("text", ""), likes_count=_count(c.get("likes"))
        )
        for c in item.get("_comments", [])
    ]

    return RawPost(
        post_id=post_id,
        source="vk_public",
        url=origin_url or f"https://vk.com/wall{owner_id}_{post_inner_id}",
        caption=item.get("text", ""),
        date=_unix_to_date(item.get("date")),
        photos=photos,
        comments=comments,
        likes_count=_count(item.get("likes")),
        reposts_count=_count(item.get("reposts")),
        views_count=_count(item.get("views")) or None,
        provenance=Provenance(
            source="vk_public",
            origin_url=origin_url,
            collection_date=collection_date,
            license_or_consent_status="public_post_review_required",
            allowed_use="research",
            retention_policy="review_required",
            deletion_status="active",
            review_status="auto",
        ),
    )


def _count(obj: Any) -> int:
    if isinstance(obj, dict):
        return int(obj.get("count", 0))
    if isinstance(obj, int):
        return obj
    return 0


def download_image(session: requests.Session, url: str, dest: Path, timeout: float = 12.0) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with session.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with dest.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
        return True
    except requests.RequestException as exc:
        log.warning("Не удалось скачать %s: %s", url, exc)
        return False


# Параллельная загрузка фото: узкое место — битые VK-CDN ссылки с таймаутами.
DOWNLOAD_WORKERS = 16
DOWNLOAD_TIMEOUT = 12.0


def _materialize(
    items: list[dict[str, Any]],
    url_by_wall_id: dict[str, str],
    collection_date: str,
    images_dir: Path,
) -> list[RawPost]:
    """Нормализовать VK-элементы в RawPost и СКАЧАТЬ их изображения конкурентно (пул потоков)."""
    posts: list[RawPost] = []
    jobs: list[tuple[Photo, str, Path]] = []  # (photo, url, dest)
    base_dir = data_path("data_dir").parent

    for item in items:
        wall_id = f"{item.get('owner_id')}_{item.get('id')}"
        post = normalize_post(item, url_by_wall_id.get(wall_id), collection_date)
        for photo in post.photos:
            if photo.url:
                dest = images_dir / post.post_id / f"{photo.photo_id}.jpg"
                jobs.append((photo, photo.url, dest))
        posts.append(post)

    session = requests.Session()

    def _fetch(job: tuple[Photo, str, Path]) -> None:
        photo, url, dest = job
        if download_image(session, url, dest, timeout=DOWNLOAD_TIMEOUT):
            photo.local_path = str(dest.relative_to(base_dir))

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        list(pool.map(_fetch, jobs))

    ok = sum(1 for p in posts for ph in p.photos if ph.local_path)
    log.info(
        "Постов %d, фото-задач %d, скачано %d (workers=%d)",
        len(posts),
        len(jobs),
        ok,
        DOWNLOAD_WORKERS,
    )
    return posts


def ingest(
    links_file: Path | str,
    posts_out: Path | None = None,
    images_dir: Path | None = None,
) -> list[RawPost]:
    """Ingest по списку конкретных постов (wall.getById).

    Требует токен, поддерживающий wall.getById (НЕ сервисный — см. ingest_wall).
    """
    posts_out = posts_out or data_path("data_dir", "raw", "posts.jsonl")
    images_dir = images_dir or data_path("data_dir", "raw", "images")
    collection_date = _dt.datetime.now(tz=_dt.timezone.utc).date().isoformat()

    raw_links = read_links(links_file)
    if not raw_links:
        log.warning("В %s не найдено валидных ссылок на посты", links_file)
        return []

    wall_ids = [parse_wall_id(link) for link in raw_links]
    url_by_wall_id = dict(zip(wall_ids, raw_links, strict=False))

    client = VKClient()
    items = client.get_posts(wall_ids)
    log.info("Получено постов: %d из %d запрошенных", len(items), len(wall_ids))

    posts = _materialize(items, url_by_wall_id, collection_date, Path(images_dir))
    n = write_jsonl(posts_out, (p.to_dict() for p in posts))
    log.info("Сохранено %d постов в %s", n, posts_out)
    return posts


def _item_post_id(item: dict[str, Any]) -> str:
    owner_id = item.get("owner_id") or item.get("from_id")
    return f"vk_{owner_id}_{item.get('id')}"


def ingest_wall(
    owner_id: str,
    count: int = 100,
    offset: int = 0,
    posts_out: Path | None = None,
    images_dir: Path | None = None,
    append: bool = False,
) -> list[RawPost]:
    """Ingest последних постов со стены сообщества/пользователя (wall.get).

    Работает с сервисным токеном. owner_id сообщества — отрицательный («-77072632»).
    append=True — ДОЗАПИСЬ к существующему posts.jsonl с дедупом по post_id (не стирает
    ранее собранные паблики и не перекачивает уже скачанные посты — для расширения датасета
    несколькими стенами).
    """
    posts_out = posts_out or data_path("data_dir", "raw", "posts.jsonl")
    images_dir = images_dir or data_path("data_dir", "raw", "images")
    collection_date = _dt.datetime.now(tz=_dt.timezone.utc).date().isoformat()

    client = VKClient()
    items = client.get_wall(owner_id, count=count, offset=offset)
    log.info("Получено постов со стены %s: %d", owner_id, len(items))

    if append:
        existing = {row.get("post_id") for row in read_jsonl(posts_out)}
        before = len(items)
        items = [it for it in items if _item_post_id(it) not in existing]
        log.info("Дедуп: новых постов %d из %d (уже было %d)", len(items), before, len(existing))
        # Материализуем и дозаписываем ЧАНКАМИ — длинная загрузка (десятки тыс. постов)
        # переживает обрыв: прогресс персистится после каждого чанка, повтор догонит остаток.
        all_posts: list[RawPost] = []
        chunk = 200
        for start in range(0, len(items), chunk):
            posts = _materialize(items[start : start + chunk], {}, collection_date, Path(images_dir))
            for p in posts:
                append_jsonl(posts_out, p.to_dict())
            all_posts.extend(posts)
            log.info("Дозаписано %d/%d постов", len(all_posts), len(items))
        return all_posts

    posts = _materialize(items, {}, collection_date, Path(images_dir))
    n = write_jsonl(posts_out, (p.to_dict() for p in posts))
    log.info("Сохранено %d постов в %s", n, posts_out)
    return posts
