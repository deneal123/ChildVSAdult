"""Единый устойчивый ретрай для ЛЮБЫХ запросов к GigaChat (chat / achat_raw / embeddings).

Повторяет только транзиентные ошибки (429 rate-limit, 5xx, таймауты/сетевые сбои) с экспоненциальной
задержкой + джиттер, уважая заголовок ``Retry-After``. Перманентные ошибки (400/401/403/404/422) и
413 (его обрабатывает сплит-логика эмбеддингов) — НЕ повторяет, пробрасывает сразу.

Используется централизованно в GigaChatConnector, поэтому ретраем покрыты все эксперименты, где
GigaChat применяется напрямую (classic, rag, rag-openai, а также эмбеддинги ретрива для rag-langgraph).
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

import httpx
from loguru import logger

T = TypeVar("T")

# Транзиентные HTTP-статусы (413 СПЕЦИАЛЬНО исключён — его делит/усекает _aembed_batch).
_TRANSIENT_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 6  # суммарно попыток (1 основная + 5 повторов)
    base_delay: float = 1.0  # стартовая задержка, сек
    max_delay: float = 30.0  # потолок задержки, сек
    multiplier: float = 2.0  # экспонента
    jitter: float = 0.5  # доля случайного разброса сверху (0..1)
    statuses: frozenset = field(default_factory=lambda: _TRANSIENT_STATUSES)


def _status_of(exc: BaseException) -> int | None:
    st = getattr(exc, "status_code", None)
    if isinstance(st, int):
        return st
    resp = getattr(exc, "response", None)
    st = getattr(resp, "status_code", None) if resp is not None else None
    return st if isinstance(st, int) else None


def _retry_after_seconds(exc: BaseException) -> float | None:
    headers = getattr(exc, "headers", None)
    if headers is None:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    try:
        raw = headers.get("retry-after")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def is_transient(exc: BaseException, policy: RetryPolicy) -> bool:
    """Стоит ли повторять запрос при данной ошибке."""
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True  # сетевые сбои/таймауты
    status = _status_of(exc)
    if status is not None:
        return status in policy.statuses
    return False


def _delay(exc: BaseException, attempt: int, policy: RetryPolicy) -> float:
    after = _retry_after_seconds(exc)
    if after is not None:
        return min(after, policy.max_delay)
    base = min(policy.base_delay * (policy.multiplier**attempt), policy.max_delay)
    return base * (1.0 + random.uniform(0.0, policy.jitter))


async def retry_async(
    factory: Callable[[], Awaitable[T]], policy: RetryPolicy, *, label: str = "gigachat"
) -> T:
    """Выполнить ``factory()`` с повторами на транзиентных ошибках.

    ``factory`` ОБЯЗАН возвращать свежую корутину при каждом вызове (корутину нельзя await дважды).
    """
    attempt = 0
    while True:
        try:
            return await factory()
        except Exception as exc:  # noqa: BLE001 — классифицируем ниже
            if attempt >= policy.max_attempts - 1 or not is_transient(exc, policy):
                raise
            delay = _delay(exc, attempt, policy)
            logger.warning(
                "{}: транзиентная ошибка {} (status={}), повтор {}/{} через {:.1f}s",
                label,
                type(exc).__name__,
                _status_of(exc),
                attempt + 1,
                policy.max_attempts,
                delay,
            )
            await asyncio.sleep(delay)
            attempt += 1


def policy_from_settings(cfg: object | None) -> RetryPolicy:
    """Строит RetryPolicy из секции настроек ([gigachat.retry]) с разумными дефолтами."""
    if cfg is None:
        return RetryPolicy()

    def g(key: str, default: float) -> float:
        return getattr(cfg, key, default)

    return RetryPolicy(
        max_attempts=int(g("max_attempts", 6)),
        base_delay=float(g("base_delay", 1.0)),
        max_delay=float(g("max_delay", 30.0)),
        multiplier=float(g("multiplier", 2.0)),
        jitter=float(g("jitter", 0.5)),
    )
