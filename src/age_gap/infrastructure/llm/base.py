from __future__ import annotations

import asyncio
import threading
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Минимальный контракт источника эмбеддингов.

    Используется для внедрения зависимостей (DI) в FastAPI и для подмены на
    детерминированный dummy в тестах/офлайн-режиме.
    """

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        """Синхронный батчевый эмбеддинг (обёртка над :meth:`aembeddings`)."""
        ...

    async def aembeddings(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        max_concurrency: int | None = None,
    ) -> list[list[float]]:
        """Асинхронный эмбеддинг пакетами с конкурентной отправкой запросов."""
        ...


class _BackgroundLoop:
    """Единственный фоновый event loop для исполнения корутин из sync-кода.

    Все корутины исполняются в одном и том же loop, поэтому async-клиент httpx
    внутри GigaChat остаётся привязанным к одному loop (нет ошибок
    "event loop is closed" при повторных вызовах из синхронного кода).

    Это НЕ параллелизм на потоках: конкурентность достигается внутри loop
    через ``asyncio`` (один поток-хост, много одновременных задач).
    """

    _loop: asyncio.AbstractEventLoop | None = None
    _thread: threading.Thread | None = None
    _lock = threading.Lock()

    @classmethod
    def _ensure(cls) -> asyncio.AbstractEventLoop:
        if cls._loop is not None and cls._loop.is_running():
            return cls._loop
        with cls._lock:
            if cls._loop is not None and cls._loop.is_running():
                return cls._loop
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                name="kspz-embedder-loop",
                daemon=True,
            )
            thread.start()
            cls._loop = loop
            cls._thread = thread
            return loop

    @classmethod
    def run(cls, coro) -> object:
        loop = cls._ensure()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()


def run_coro_blocking(coro):
    """Выполнить корутину из синхронного кода и вернуть результат.

    Безопасно вызывать как вне event loop, так и из работающего loop
    (например, из обработчика FastAPI, отданного в threadpool): корутина
    всегда исполняется в выделенном фоновом loop.
    """
    return _BackgroundLoop.run(coro)
