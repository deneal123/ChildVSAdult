"""Тонкий клиент к VK API для получения постов со стены.

Используется метод ``wall.getById`` (получение конкретных постов по ``owner_id_post_id``).
Токен берётся из окружения (``.env`` уже загружается в settings.config).

Конвейер работает только с публичными постами (SKILL §4/§9): приватные/скрытые записи
не запрашиваются. Соблюдается rate limit и логируются ошибки (TODO §12 Phase 2).
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from age_gap.common.logging import get_logger
from age_gap.settings import settings

log = get_logger(__name__)

API_URL = "https://api.vk.com/method/{method}"
API_VERSION = "5.199"

# https://vk.ru/wall-77072632_452850 -> owner_id=-77072632, post_id=452850
_WALL_RE = re.compile(r"wall(-?\d+)_(\d+)")


@dataclass
class VKConfig:
    token: str
    api_version: str = API_VERSION
    min_interval_sec: float = 0.34  # ~3 запроса/сек
    max_retries: int = 3
    timeout_sec: float = 30.0


def parse_wall_id(url_or_id: str) -> str:
    """Извлечь ``owner_post`` из ссылки или принять уже готовый ``-77072632_452850``."""
    s = url_or_id.strip()
    m = _WALL_RE.search(s)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    if re.fullmatch(r"-?\d+_\d+", s):
        return s
    raise ValueError(f"Не удалось распознать VK wall id: {url_or_id!r}")


def load_config() -> VKConfig:
    token = os.getenv("VK_TOKEN") or str(getattr(settings, "VK_TOKEN", "") or "")
    if not token:
        raise RuntimeError(
            "VK_TOKEN не задан. Укажите его в src/age_gap/settings/.env (VK_TOKEN=...)."
        )
    return VKConfig(token=token)


class VKClient:
    def __init__(self, config: VKConfig | None = None) -> None:
        self.config = config or load_config()
        self._session = requests.Session()
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        wait = self.config.min_interval_sec - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = {
            **params,
            "access_token": self.config.token,
            "v": self.config.api_version,
        }
        last_err: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.get(
                    API_URL.format(method=method),
                    params=payload,
                    timeout=self.config.timeout_sec,
                )
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_err = exc
                log.warning(
                    "VK %s попытка %d/%d: %s", method, attempt, self.config.max_retries, exc
                )
                time.sleep(self.config.min_interval_sec * attempt)
                continue

            if "error" in data:
                err = data["error"]
                code = err.get("error_code")
                msg = err.get("error_msg")
                # 6 = Too many requests per second — повторяем с паузой.
                if code == 6 and attempt < self.config.max_retries:
                    log.warning("VK rate limit (code 6), пауза и повтор")
                    time.sleep(self.config.min_interval_sec * (attempt + 1))
                    continue
                raise RuntimeError(f"VK API error {code}: {msg}")

            return data["response"]

        raise RuntimeError(
            f"VK {method} не удался после {self.config.max_retries} попыток: {last_err}"
        )

    def get_posts(self, wall_ids: list[str]) -> list[dict[str, Any]]:
        """Получить посты по списку ``owner_post`` через wall.getById (батч до 100).

        Внимание: wall.getById недоступен для сервисных токенов (VK error 1051).
        Для сервисного токена используйте get_wall().
        """
        items: list[dict[str, Any]] = []
        for start in range(0, len(wall_ids), 100):
            batch = wall_ids[start : start + 100]
            response = self._call("wall.getById", {"posts": ",".join(batch)})
            # v>=5.101: {"items": [...]}; старые версии: просто список.
            if isinstance(response, dict):
                items.extend(response.get("items", []))
            elif isinstance(response, list):
                items.extend(response)
        return items

    def get_wall(self, owner_id: str, count: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Получить посты со стены через wall.get (работает с сервисным токеном).

        owner_id для сообщества — отрицательный («-77072632»). VK отдаёт максимум 100
        постов за вызов, поэтому при count>100 идёт постраничный обход.
        """
        items: list[dict[str, Any]] = []
        remaining = count
        cur = offset
        total: int | None = None
        while remaining > 0:
            batch = min(100, remaining)
            response = self._call("wall.get", {"owner_id": owner_id, "count": batch, "offset": cur})
            if not isinstance(response, dict):
                break
            if total is None:
                total = int(response.get("count", 0))
            page = response.get("items", [])
            if not page:  # прерываемся только на ПУСТОЙ странице (короткая — норма: удалённые посты)
                break
            items.extend(page)
            cur += batch  # сдвигаемся на полный шаг: VK может вернуть <batch из-за удалённых записей
            remaining -= len(page)
            if total and cur >= total:  # дошли до конца стены
                break
        return items

    def get_comments(self, owner_id: int, post_id: int, count: int = 100) -> list[dict[str, Any]]:
        """Комментарии к посту (wall.getComments). Возвращает список {id, text, likes...}.

        Комментарии — НЕ основной сигнал идентичности; используются как слабая
        вспомогательная supervision для apparent-age (SKILL §8.4).
        """
        try:
            response = self._call(
                "wall.getComments",
                {"owner_id": owner_id, "post_id": post_id, "count": min(100, count), "sort": "asc"},
            )
        except RuntimeError as exc:  # закрытые комментарии и т.п. — не валим ingest
            log.warning("Комментарии %s_%s недоступны: %s", owner_id, post_id, exc)
            return []
        return response.get("items", []) if isinstance(response, dict) else []


def best_photo_url(photo: dict[str, Any]) -> str | None:
    """Выбрать URL максимального размера из вложения-фото VK."""
    sizes = photo.get("sizes") or []
    if not sizes:
        return photo.get("url")
    best = max(sizes, key=lambda s: int(s.get("width", 0)) * int(s.get("height", 0)))
    return best.get("url")
