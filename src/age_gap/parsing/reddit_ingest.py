"""Ингест мультифото-постов Reddit в RawPost + загрузка изображений (не-VK источник).

Поток (зеркало VK-ингеста):
    листинг сабреддита (.json) -> нормализация в RawPost -> скачивание фото ->
    <data_dir>/raw/posts.jsonl + <data_dir>/raw/images/<post_id>/<photo_id>.jpg

Чтобы не затирать VK, запускайте под ОТДЕЛЬНЫМ data-root через reddit-окружение
(``ENV_FOR_DYNACONF=reddit`` → ``data_reddit/``; см. settings.toml). post_id префиксуется
``reddit_`` (не пересекается с ``vk_``); возраст извлекается downstream из caption (заголовка)
той же LLM. Запись идёт в стандартный ``posts.jsonl`` под активным data_dir, поэтому downstream
(build_groups → cluster_persons → dedup → build_pairs → split) работает без изменений.

    ENV_FOR_DYNACONF=reddit uv run python scripts/ingest_reddit.py --subreddit PastAndPresentPics
"""

from __future__ import annotations

import datetime as _dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from age_gap.common.io import append_jsonl, data_path, read_jsonl, write_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import Photo, Provenance, RawPost
from age_gap.parsing.ingest import DOWNLOAD_TIMEOUT, DOWNLOAD_WORKERS, download_image
from age_gap.parsing.reddit_client import BASE_URL, RedditClient, load_proxies, post_image_urls

log = get_logger(__name__)


def _unix_to_date(ts: float | None) -> str | None:
    if not ts:
        return None
    return _dt.datetime.fromtimestamp(float(ts), tz=_dt.timezone.utc).date().isoformat()


def normalize_reddit_post(data: dict[str, Any], collection_date: str) -> RawPost:
    rid = str(data.get("id"))
    post_id = f"reddit_{rid}"
    permalink = data.get("permalink") or f"/r/{data.get('subreddit')}/comments/{rid}/"
    url = BASE_URL + permalink

    # caption = заголовок (+ selftext, если есть): именно здесь у then/now-постов возраст/годы.
    title = (data.get("title") or "").strip()
    selftext = (data.get("selftext") or "").strip()
    caption = f"{title}\n{selftext}".strip()

    photos = [
        Photo(photo_id=f"{rid}_{pid}", url=purl, order=order)
        for order, (pid, purl) in enumerate(post_image_urls(data))
    ]

    return RawPost(
        post_id=post_id,
        source="reddit_public",
        url=url,
        caption=caption,
        date=_unix_to_date(data.get("created_utc")),
        photos=photos,
        comments=[],  # возраст у then/now-постов в заголовке; комментарии не тянем
        likes_count=int(data.get("score", 0) or 0),
        reposts_count=int(data.get("num_crossposts", 0) or 0),
        views_count=None,
        provenance=Provenance(
            source="reddit_public",
            origin_url=url,
            collection_date=collection_date,
            license_or_consent_status="public_post_review_required",
            allowed_use="research",
            retention_policy="review_required",
            deletion_status="active",
            review_status="auto",
        ),
    )


def _materialize(
    posts_data: list[dict[str, Any]],
    collection_date: str,
    images_dir: Path,
    user_agent: str,
) -> list[RawPost]:
    """Нормализовать посты Reddit в RawPost и конкурентно скачать их изображения."""
    posts: list[RawPost] = []
    jobs: list[tuple[Photo, str, Path]] = []
    base_dir = data_path("data_dir").parent

    for data in posts_data:
        post = normalize_reddit_post(data, collection_date)
        if not post.photos:  # текстовые/видео/внешние посты без изображений — пропускаем
            continue
        for photo in post.photos:
            if photo.url:
                jobs.append((photo, photo.url, images_dir / post.post_id / f"{photo.photo_id}.jpg"))
        posts.append(post)

    session = requests.Session()
    session.headers["User-Agent"] = user_agent  # i.redd.it отдаёт без UA, но не мешает
    proxies = load_proxies()
    if proxies:
        session.proxies.update(proxies)  # качаем картинки через тот же прокси (единый exit-IP)

    def _fetch(job: tuple[Photo, str, Path]) -> None:
        photo, url, dest = job
        if download_image(session, url, dest, timeout=DOWNLOAD_TIMEOUT):
            photo.local_path = str(dest.relative_to(base_dir))

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        list(pool.map(_fetch, jobs))

    ok = sum(1 for p in posts for ph in p.photos if ph.local_path)
    multi = sum(1 for p in posts if len(p.photos) >= 2)
    log.info(
        "Постов с фото %d (мультифото %d), фото-задач %d, скачано %d",
        len(posts), multi, len(jobs), ok,
    )
    return posts


def ingest_subreddit(
    subreddit: str = "PastAndPresentPics",
    limit: int = 1000,
    sorts: tuple[str, ...] = ("top", "new", "hot"),
    time_filter: str = "all",
    posts_out: Path | None = None,
    images_dir: Path | None = None,
    append: bool = False,
) -> list[RawPost]:
    """Собрать посты сабреддита по нескольким сортировкам (с дедупом по id) и материализовать.

    Reddit ограничивает листинг ~1000 постами на sort; объединение top/new/hot расширяет охват.
    append=True --- дозапись к существующему posts_reddit.jsonl с дедупом по post_id.
    """
    posts_out = posts_out or data_path("data_dir", "raw", "posts.jsonl")
    images_dir = Path(images_dir or data_path("data_dir", "raw", "images"))
    collection_date = _dt.datetime.now(tz=_dt.timezone.utc).date().isoformat()

    # Защита от случайной записи поверх VK-данных (запускать reddit-ингест под data_reddit/).
    if not append and Path(posts_out).exists():
        first = next(iter(read_jsonl(posts_out)), None)
        if first is not None and first.get("source") != "reddit_public":
            raise RuntimeError(
                f"{posts_out} содержит не-reddit посты (вероятно VK). Запускайте reddit-ингест "
                "под ENV_FOR_DYNACONF=reddit (отдельный data_reddit/) или задайте свой --posts-out."
            )

    client = RedditClient()
    seen_ids: set[str] = set()
    raw: list[dict[str, Any]] = []
    for sort in sorts:
        for d in client.get_subreddit(subreddit, sort=sort, limit=limit, time_filter=time_filter):
            rid = str(d.get("id"))
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                raw.append(d)
    log.info("r/%s: уникальных постов %d из сортировок %s", subreddit, len(raw), list(sorts))

    if append:
        existing = {row.get("post_id") for row in read_jsonl(posts_out)}
        raw = [d for d in raw if f"reddit_{d.get('id')}" not in existing]
        log.info("Дедуп с существующими: к материализации %d новых", len(raw))

    posts = _materialize(raw, collection_date, images_dir, client.user_agent)
    if append:
        for p in posts:
            append_jsonl(posts_out, p.to_dict())
    else:
        write_jsonl(posts_out, (p.to_dict() for p in posts))
    log.info("Сохранено %d постов Reddit в %s", len(posts), posts_out)
    return posts


def ingest_reddit_post(
    url_or_permalink: str,
    posts_out: Path | None = None,
    images_dir: Path | None = None,
) -> RawPost | None:
    """Ингест одного поста по ссылке (удобно для проверки парсинга на примере)."""
    images_dir = Path(images_dir or data_path("data_dir", "raw", "images"))
    collection_date = _dt.datetime.now(tz=_dt.timezone.utc).date().isoformat()
    client = RedditClient()
    data = client.get_post(url_or_permalink)
    posts = _materialize([data], collection_date, images_dir, client.user_agent)
    if not posts:
        log.warning("В посте %s не найдено изображений", url_or_permalink)
        return None
    if posts_out:
        write_jsonl(posts_out, (p.to_dict() for p in posts))
    return posts[0]
