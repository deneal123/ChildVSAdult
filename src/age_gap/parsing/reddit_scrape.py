"""HTML-скрейпер old.reddit.com — обход заблокированного .json API.

Reddit отдаёт 403 на неаутентифицированный ``.json`` (даже с резидентных IP), НО HTML-страницы
``old.reddit.com`` грузятся (особенно через резидентский прокси). Здесь мы парсим листинг
сабреддита (permalink'и + заголовки) и страницы постов (URL изображений галереи в нормальном
разрешении), формируя те же ``RawPost``, что и API-путь. Прокси берётся из env (PROXY_*).

    ENV_FOR_DYNACONF=reddit uv run python scripts/ingest_reddit.py --mode html --subreddit PastAndPresentPics
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from age_gap.common.io import append_jsonl, data_path, read_jsonl, write_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import Photo, Provenance, RawPost
from age_gap.parsing.ingest import DOWNLOAD_TIMEOUT, DOWNLOAD_WORKERS
from age_gap.parsing.reddit_client import BASE_URL, BROWSER_HEADERS, BROWSER_UA, load_proxies

log = get_logger(__name__)
OLD = "https://old.reddit.com"
_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
# Заголовки, проходящие CDN-челлендж redd.it ("Please wait for verification"):
# image-Accept + Sec-Fetch-Dest:image + Referer на страницу поста.
IMG_HEADERS = {
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Sec-Fetch-Dest": "image",
    "Sec-Fetch-Mode": "no-cors",
    "Sec-Fetch-Site": "cross-site",
}
# URL-сорты old.reddit: hot (пусто), new, top за всё время.
SORTS: dict[str, str] = {"hot": "", "new": "new/", "top": "top/?sort=top&t=all"}


def _stem(url: str) -> str:
    """id картинки из redd.it-URL (часть имени файла до расширения)."""
    name = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0]


class RedditScraper:
    def __init__(self, delay: float = 2.5) -> None:
        self.session = requests.Session()
        self.session.headers.update(BROWSER_HEADERS)
        self.session.headers["User-Agent"] = BROWSER_UA
        px = load_proxies()
        if px:
            self.session.proxies.update(px)
            log.info("scraper через прокси %s:%s", _env_host(), _env_port())
        self.delay = delay

    def _get(self, url: str, retries: int = 5) -> str | None:
        for attempt in range(1, retries + 1):
            wait = self.delay * attempt
            try:
                r = self.session.get(url, timeout=40)
                if r.status_code == 429:  # rate limit: ждать заметно дольше
                    wait = max(wait, 12 * attempt)
                    raise requests.RequestException("HTTP 429")
                if r.status_code == 403:
                    raise requests.RequestException("HTTP 403")
                r.raise_for_status()
                if r.text.strip():
                    return r.text
                raise requests.RequestException("empty body")
            except requests.RequestException as exc:
                log.warning("scrape GET %d/%d (%s): %s", attempt, retries, url, exc)
                time.sleep(wait)
        return None

    def listing(self, subreddit: str, sort: str = "hot", pages: int = 10) -> list[dict[str, Any]]:
        """Собрать посты листинга (постранично по next-кнопке old.reddit)."""
        out: list[dict[str, Any]] = []
        url = f"{OLD}/r/{subreddit}/{SORTS.get(sort, '')}"
        seen: set[str] = set()
        for _ in range(pages):
            html = self._get(url)
            if not html:
                break
            for post in _parse_listing(html):
                if post["id"] not in seen:
                    seen.add(post["id"])
                    out.append(post)
            nb = re.search(r'class="next-button"[^>]*>\s*<a[^>]+href="([^"]+)"', html)
            if not nb:
                break
            url = _html.unescape(nb.group(1))
            time.sleep(self.delay)
        log.info("r/%s/%s: собрано %d постов", subreddit, sort, len(out))
        return out

    def post_images(self, post: dict[str, Any]) -> list[tuple[str, str]]:
        """URL изображений поста: галерея -> страница поста; одиночное -> data-url."""
        durl = post.get("data_url") or ""
        if "/gallery/" not in durl and durl.lower().split("?")[0].endswith(_IMG_EXT):
            return [(_stem(durl), durl)]  # одиночное изображение (i.redd.it)
        html = self._get(OLD + post["permalink"])
        if not html:
            return []
        # gallery-item ссылки несут лучшее доступное разрешение со своей подписью
        hrefs = [_html.unescape(u) for u in re.findall(r'class="gallery-item[^"]*"[^>]+href="([^"]+)"', html)]
        if not hrefs:  # резерв: лучший preview по каждому id картинки
            hrefs = _best_previews(html)
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for u in hrefs:
            sid = _stem(u)
            if sid not in seen:
                seen.add(sid)
                out.append((sid, u))
        return out


def _env_host() -> str:
    from age_gap.parsing.reddit_client import _env

    return _env("PROXY_HOST")


def _env_port() -> str:
    from age_gap.parsing.reddit_client import _env

    return _env("PROXY_PORT")


def _parse_listing(html: str) -> list[dict[str, Any]]:
    """Извлечь посты (t3) из HTML листинга old.reddit.

    Матчим thing-div по ``data-fullname`` (атрибуты идут в произвольном порядке: class раньше id),
    затем тянем data-url/permalink/timestamp из того же тега.
    """
    posts: list[dict[str, Any]] = []
    for m in re.finditer(r'<div\b([^>]*\bdata-fullname="(t3_\w+)"[^>]*)>', html):
        attrs, fid = m.group(1), m.group(2)
        if _attr(attrs, "promoted") == "true":
            continue  # реклама
        chunk = html[m.end() : m.end() + 4000]
        tm = re.search(r'<a[^>]*class="[^"]*\btitle\b[^"]*"[^>]*>([^<]+)</a>', chunk)
        posts.append(
            {
                "id": fid.split("_", 1)[-1],
                "permalink": _attr(attrs, "permalink") or "",
                "data_url": _attr(attrs, "url") or "",
                "title": _html.unescape(tm.group(1)) if tm else "",
                "ts": _attr(attrs, "timestamp"),
            }
        )
    return posts


def _attr(tag: str, name: str) -> str | None:
    m = re.search(rf'data-{name}="([^"]*)"', tag)
    return _html.unescape(m.group(1)) if m else None


def _best_previews(html: str) -> list[str]:
    """Резерв: по каждому id картинки взять URL максимальной ширины из всех preview-ссылок."""
    best: dict[str, tuple[int, str]] = {}
    for u in re.findall(r'https://preview\.redd\.it/[^\s"\'<>\\]+', html):
        u = _html.unescape(u)
        sid = _stem(u)
        w = int((re.search(r"width=(\d+)", u) or [0, 0])[1] or 0)
        if sid not in best or w > best[sid][0]:
            best[sid] = (w, u)
    return [v[1] for v in best.values()]


def _ts_to_date(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        return _dt.datetime.fromtimestamp(int(ts) / 1000.0, tz=_dt.timezone.utc).date().isoformat()
    except (ValueError, OverflowError):
        return None


def _to_rawpost(post: dict[str, Any], images: list[tuple[str, str]], collection_date: str) -> RawPost:
    rid = post["id"]
    url = BASE_URL + post["permalink"]
    photos = [Photo(photo_id=f"{rid}_{sid}", url=u, order=i) for i, (sid, u) in enumerate(images)]
    return RawPost(
        post_id=f"reddit_{rid}",
        source="reddit_public",
        url=url,
        caption=post.get("title", ""),
        date=_ts_to_date(post.get("ts")),
        photos=photos,
        comments=[],
        likes_count=0,
        reposts_count=0,
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


def scrape_subreddit(
    subreddit: str = "PastAndPresentPics",
    sorts: tuple[str, ...] = ("hot", "new", "top"),
    pages: int = 10,
    max_posts: int = 400,
    posts_out: Path | None = None,
    images_dir: Path | None = None,
    append: bool = False,
) -> list[RawPost]:
    """Сбор через HTML old.reddit: листинг -> страницы постов -> RawPost + загрузка фото."""
    posts_out = posts_out or data_path("data_dir", "raw", "posts.jsonl")
    images_dir = Path(images_dir or data_path("data_dir", "raw", "images"))
    collection_date = _dt.datetime.now(tz=_dt.timezone.utc).date().isoformat()
    if not append and Path(posts_out).exists():
        first = next(iter(read_jsonl(posts_out)), None)
        if first is not None and first.get("source") != "reddit_public":
            raise RuntimeError(
                f"{posts_out} содержит не-reddit посты — запускайте под ENV_FOR_DYNACONF=reddit "
                "(отдельный data_reddit/) или задайте свой --posts-out."
            )

    sc = RedditScraper()
    listed: dict[str, dict[str, Any]] = {}
    for sort in sorts:
        for p in sc.listing(subreddit, sort=sort, pages=pages):
            listed.setdefault(p["id"], p)
    if append:
        existing = {row.get("post_id") for row in read_jsonl(posts_out)}
        listed = {k: v for k, v in listed.items() if f"reddit_{k}" not in existing}
    items = list(listed.values())
    if max_posts and len(items) > max_posts:
        items = items[:max_posts]  # кап: меньше post-page запросов -> меньше 429
    log.info("Уникальных постов %d; к разбору: %d", len(listed), len(items))

    rawposts: list[RawPost] = []
    for post in items:
        imgs = sc.post_images(post)
        if imgs:
            rawposts.append(_to_rawpost(post, imgs, collection_date))
        time.sleep(sc.delay)

    _download(rawposts, images_dir, sc.session)
    if append:
        for p in rawposts:
            append_jsonl(posts_out, p.to_dict())
    else:
        write_jsonl(posts_out, (p.to_dict() for p in rawposts))
    multi = sum(1 for p in rawposts if len(p.photos) >= 2)
    log.info("Сохранено %d постов (мультифото %d) -> %s", len(rawposts), multi, posts_out)
    return rawposts


def _download(posts: list[RawPost], images_dir: Path, session: requests.Session) -> None:
    base_dir = data_path("data_dir").parent
    jobs: list[tuple[Photo, str, Path, str]] = []
    for p in posts:
        referer = p.url.replace("https://www.reddit.com", OLD)  # страница поста = Referer
        for ph in p.photos:
            if ph.url:
                jobs.append((ph, ph.url, images_dir / p.post_id / f"{ph.photo_id}.jpg", referer))

    def _fetch(job: tuple[Photo, str, Path, str]) -> None:
        photo, url, dest, referer = job
        headers = {**IMG_HEADERS, "Referer": referer}
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            r = session.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT, stream=True)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                with dest.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                photo.local_path = str(dest.relative_to(base_dir))
            else:
                log.warning("img skip %s: %s %s", url[:60], r.status_code, r.headers.get("content-type"))
        except requests.RequestException as exc:
            log.warning("img dl err %s: %s", url[:60], exc)

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        list(pool.map(_fetch, jobs))
    ok = sum(1 for p in posts for ph in p.photos if ph.local_path)
    log.info("Скачано изображений: %d из %d", ok, len(jobs))


def scrape_post(url_or_permalink: str) -> RawPost | None:
    """Разобрать один пост по ссылке (проверка скрейпера)."""
    from age_gap.parsing.reddit_client import permalink_path

    sc = RedditScraper()
    permalink = permalink_path(url_or_permalink)
    html = sc._get(OLD + permalink)
    if not html:
        return None
    tm = re.search(r'<a[^>]*class="[^"]*\btitle\b[^"]*"[^>]*>([^<]+)</a>', html)
    post = {
        "id": permalink.strip("/").split("/")[3] if len(permalink.strip("/").split("/")) > 3 else permalink,
        "permalink": permalink,
        "title": _html.unescape(tm.group(1)) if tm else "",
        "data_url": "/gallery/",  # форсируем разбор страницы поста на изображения
        "ts": None,
    }
    imgs = sc.post_images(post)
    if not imgs:
        return None
    rp = _to_rawpost(post, imgs, _dt.datetime.now(tz=_dt.timezone.utc).date().isoformat())
    _download([rp], Path(data_path("data_dir", "raw", "images")), sc.session)
    return rp
