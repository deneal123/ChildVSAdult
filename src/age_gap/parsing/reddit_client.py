"""Тонкий клиент к Reddit для сбора мультифото-постов «then/now» (не-VK источник).

Два режима, OAuth НЕОБЯЗАТЕЛЕН:

1. **Публичный скрейпинг (по умолчанию, без креденшелов).** Ходим на публичный ``.json``
   с браузерными заголовками и фоллбэком хостов ``www.reddit.com`` -> ``old.reddit.com``,
   вежливым rate limit и бэкоффом. С резидентного (домашнего) IP это обычно работает; с
   дата-центровых IP Reddit часто отдаёт 403 ``Blocked``.
2. **OAuth2 (фоллбэк для заблокированных IP).** Зарегистрируйте «script»-приложение на
   https://www.reddit.com/prefs/apps и положите в ``src/age_gap/settings/.env``:

       REDDIT_CLIENT_ID=...
       REDDIT_CLIENT_SECRET=...
       REDDIT_USERNAME=...            # опционально (password-grant); иначе client_credentials
       REDDIT_PASSWORD=...
       REDDIT_USER_AGENT=script:age-gap:0.1 (by /u/<username>)

   При наличии client_id/secret клиент аутентифицируется и ходит на ``oauth.reddit.com``.

Сначала проверьте публичный режим одним постом: ``scripts/ingest_reddit.py --post <url>``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests

from age_gap.common.logging import get_logger
from age_gap.settings import settings

log = get_logger(__name__)

BASE_URL = "https://www.reddit.com"
OAUTH_URL = "https://oauth.reddit.com"
TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")
# Браузерный UA для публичного отката; описательный UA для OAuth (требование Reddit).
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
OAUTH_UA = "script:age-gap:0.1 (cross-age research)"
# Публичный (без OAuth) скрейпинг .json: пробуем несколько хостов, маскируемся под браузер.
PUBLIC_HOSTS = ("https://www.reddit.com", "https://old.reddit.com")
BROWSER_HEADERS = {
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}


@dataclass
class RedditConfig:
    client_id: str = ""
    client_secret: str = ""
    username: str = ""
    password: str = ""
    oauth_user_agent: str = OAUTH_UA
    public_user_agent: str = BROWSER_UA
    min_interval_sec: float = 1.1  # ~1 запрос/сек
    max_retries: int = 4
    timeout_sec: float = 30.0


def _env(key: str) -> str:
    return os.getenv(key) or str(getattr(settings, key, "") or "")


def load_config() -> RedditConfig:
    cfg = RedditConfig(
        client_id=_env("REDDIT_CLIENT_ID"),
        client_secret=_env("REDDIT_CLIENT_SECRET"),
        username=_env("REDDIT_USERNAME"),
        password=_env("REDDIT_PASSWORD"),
    )
    ua = _env("REDDIT_USER_AGENT")
    if ua:
        cfg.oauth_user_agent = ua
    return cfg


def permalink_path(url_or_permalink: str) -> str:
    """Достать путь permalink (``/r/.../comments/.../``) из полной ссылки или голого пути."""
    s = url_or_permalink.strip().split("?")[0].split("#")[0]
    for host in ("https://www.reddit.com", "https://old.reddit.com", "https://reddit.com"):
        if s.startswith(host):
            s = s[len(host) :]
            break
    if not s.startswith("/"):
        s = "/" + s
    return s.rstrip("/") + "/"


class RedditClient:
    def __init__(self, config: RedditConfig | None = None) -> None:
        self.config = config or load_config()
        self._session = requests.Session()
        self._token: str | None = None
        self._last_call = 0.0
        self._public_hosts = list(PUBLIC_HOSTS)
        if self.config.client_id and self.config.client_secret:
            self._authenticate()
            self._api_base = OAUTH_URL
            self._suffix = ""
            self.user_agent = self.config.oauth_user_agent
        else:
            self._api_base = self._public_hosts[0]
            self._suffix = ".json"
            self.user_agent = self.config.public_user_agent
            self._session.headers.update(BROWSER_HEADERS)  # маскируемся под браузер
            log.warning(
                "Reddit без OAuth: скрейпинг публичного .json (хосты www->old, браузерные "
                "заголовки, бэкофф). С дата-центровых IP часто 403; с резидентного обычно работает."
            )
        self._session.headers["User-Agent"] = self.user_agent

    def _authenticate(self) -> None:
        auth = requests.auth.HTTPBasicAuth(self.config.client_id, self.config.client_secret)
        if self.config.username and self.config.password:
            data = {
                "grant_type": "password",
                "username": self.config.username,
                "password": self.config.password,
            }
        else:
            data = {"grant_type": "client_credentials"}
        resp = requests.post(
            TOKEN_URL,
            auth=auth,
            data=data,
            headers={"User-Agent": self.config.oauth_user_agent},
            timeout=self.config.timeout_sec,
        )
        resp.raise_for_status()
        self._token = resp.json()["access_token"]
        log.info("Reddit OAuth: токен получен (grant=%s)", data["grant_type"])

    def _throttle(self) -> None:
        wait = self.config.min_interval_sec - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        hosts = [self._api_base] if self._token else self._public_hosts
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        last_err: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            for host in hosts:  # публичный режим: www -> old при 403/блоке
                self._throttle()
                url = f"{host}{path}{self._suffix}"
                try:
                    resp = self._session.get(
                        url, params=params, headers=headers, timeout=self.config.timeout_sec
                    )
                    if resp.status_code in (403, 429):
                        raise requests.RequestException(f"HTTP {resp.status_code} ({host})")
                    resp.raise_for_status()
                    return resp.json()
                except (requests.RequestException, ValueError) as exc:
                    last_err = exc
                    log.warning(
                        "Reddit GET попытка %d/%d (%s): %s",
                        attempt, self.config.max_retries, url, exc,
                    )
            time.sleep(self.config.min_interval_sec * 2 * attempt)
        raise RuntimeError(
            f"Reddit GET не удался после {self.config.max_retries} попыток по {hosts}: {last_err}"
        )

    def get_subreddit(
        self,
        subreddit: str,
        sort: str = "new",
        limit: int = 1000,
        time_filter: str = "all",
    ) -> list[dict[str, Any]]:
        """Постранично собрать посты (t3) листинга сабреддита (~1000 максимум на один sort)."""
        out: list[dict[str, Any]] = []
        after: str | None = None
        path = f"/r/{subreddit}/{sort}"
        while len(out) < limit:
            params: dict[str, Any] = {"limit": 100, "raw_json": 1}
            if after:
                params["after"] = after
            if sort == "top":
                params["t"] = time_filter
            data = self._get(path, params)
            children = data.get("data", {}).get("children", [])
            page = [c["data"] for c in children if c.get("kind") == "t3"]
            if not page:
                break
            out.extend(page)
            after = data.get("data", {}).get("after")
            log.info("r/%s/%s: +%d (всего %d)", subreddit, sort, len(page), len(out))
            if not after:
                break
        return out[:limit]

    def get_post(self, url_or_permalink: str) -> dict[str, Any]:
        """Получить data одного поста (t3) по ссылке/permalink."""
        data = self._get(permalink_path(url_or_permalink), {"raw_json": 1})
        listing = data[0] if isinstance(data, list) else data
        return listing["data"]["children"][0]["data"]


def post_image_urls(data: dict[str, Any]) -> list[tuple[str, str]]:
    """Извлечь упорядоченные (photo_id, url) изображений поста.

    Поддержано: gallery-посты (несколько фото --- основной мультифото-сигнал) и одиночные
    image-посты. Внешние хостинги (imgur и т.п.) и видео пропускаются; коллажи «два лица в одном
    кадре» дадут одно изображение (downstream-детектор разберётся с числом лиц).
    """
    out: list[tuple[str, str]] = []
    # 1) Gallery: упорядоченный gallery_data.items + media_metadata[mid].s.u
    if data.get("is_gallery") and isinstance(data.get("media_metadata"), dict):
        meta = data["media_metadata"]
        for it in data.get("gallery_data", {}).get("items", []):
            mid = it.get("media_id")
            m = meta.get(mid) if mid else None
            if not m or m.get("status") != "valid":
                continue
            src = m.get("s", {})
            url = src.get("u") or src.get("gif")
            if url:
                out.append((str(mid), url))
        if out:
            return out
    # 2) Одиночное изображение (i.redd.it / прямой image-URL)
    url = data.get("url_overridden_by_dest") or data.get("url") or ""
    is_image = data.get("post_hint") == "image" or url.lower().split("?")[0].endswith(_IMAGE_EXT)
    if url and is_image:
        out.append((f"{data.get('id')}_0", url))
        return out
    # 3) Резерв: preview-источник
    for i, im in enumerate(data.get("preview", {}).get("images", [])):
        u = im.get("source", {}).get("u")
        if u:
            out.append((f"{data.get('id')}_p{i}", u))
    return out
