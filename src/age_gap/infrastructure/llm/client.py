from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from gigachat import GigaChat
from gigachat.exceptions import ResponseError

try:  # в части версий gigachat есть отдельный класс, в части — нет
    from gigachat.exceptions import RequestEntityTooLargeError
except ImportError:  # pragma: no cover - зависит от версии SDK

    class RequestEntityTooLargeError(Exception):  # type: ignore[no-redef]
        """Заглушка для версий gigachat без отдельного класса 413."""


from age_gap.infrastructure.llm.base import run_coro_blocking
from age_gap.infrastructure.llm.retry import RetryPolicy, policy_from_settings, retry_async
from age_gap.settings import settings


def _is_too_large(exc: Exception) -> bool:
    """Эвристически определяет ошибку 413 (Request Entity Too Large)."""
    if "TooLarge" in type(exc).__name__:
        return True
    if any(isinstance(arg, int) and arg == 413 for arg in getattr(exc, "args", ())):
        return True
    return "413" in str(exc)


def split_text_chunks(text: str, limit: int) -> list[str]:
    """Бьёт текст на куски ≤ limit символов по границам слов (для chunk+mean-pool длинных запросов).

    Короткий текст → [text]. Длинный набирается по словам до лимита; «слово» длиннее лимита (без
    пробелов) режется жёстко. Полное покрытие без потери хвоста (раньше текст усекался до 512 симв.).
    """
    text = str(text)
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    cur = ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > limit:
            chunks.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        chunks.append(cur)
    out: list[str] = []
    for c in chunks:  # очень длинное «слово» без пробелов — режем жёстко
        while len(c) > limit:
            out.append(c[:limit])
            c = c[limit:]
        if c:
            out.append(c)
    return out or [text[:limit]]


def _extract_vectors(response: Any, expected: int) -> list[list[float]]:
    """Достаёт список векторов из ответа GigaChat embeddings."""
    data = getattr(response, "data", response)
    if not data:
        raise RuntimeError("GigaChat embeddings вернул пустой ответ")

    vectors: list[list[float]] = []
    for item in data:
        if hasattr(item, "embedding"):
            vectors.append(list(item.embedding))
        elif isinstance(item, dict) and "embedding" in item:
            vectors.append(list(item["embedding"]))
        else:
            vectors.append(list(item))

    if len(vectors) != expected:
        raise RuntimeError(f"GigaChat embeddings вернул {len(vectors)} векторов вместо {expected}")
    return vectors


class GigaChatConnector:
    def __init__(
        self,
        client_secret: str,
        cert_path: str,
        verify_ssl: bool,
        api_endpoint: str,
        auth_endpoint: str,
        model_name: str,
        scope: str,
        temperature: float = 1e-7,
        max_gen_tokens: int = 2048,
        max_embed_tokens: int = 514,
        top_p: float = 0.0,
        n: int = 1,
        stream: bool = False,
        repetition_penalty: float = 1,
        profanity_check: bool = False,
        embed_batch_size: int = 16,
        embed_max_concurrency: int = 8,
        embed_chunk_chars: int = 1200,
        retry_policy: RetryPolicy | None = None,
    ):
        self.model_name = model_name
        self.retry_policy = retry_policy or RetryPolicy()
        # относительный cert_path резолвим от папки settings (переносимо между машинами)
        cert = Path(cert_path)
        if not cert.is_absolute():
            from age_gap.settings.config import BASE_DIR

            cert = BASE_DIR / cert
        self.cert_path = str(cert.resolve())

        self.temperature = temperature
        self.verify_ssl = verify_ssl
        self.max_gen_tokens = max_gen_tokens
        self.max_embed_tokens = max_embed_tokens
        self.top_p = top_p
        self.n = n
        self.stream = stream
        self.repetition_penalty = repetition_penalty
        self.profanity_check = profanity_check
        self.embed_batch_size = max(1, int(embed_batch_size))
        self.embed_max_concurrency = max(1, int(embed_max_concurrency))
        self.embed_chunk_chars = max(64, int(embed_chunk_chars))  # размер куска для chunk+mean-pool
        # Подпись кеша: cm<chars> — chunk-mean инвалидирует старый кеш усечённых на 512 симв. эмбеддингов.
        self.cache_signature = f"gigachat:{model_name}:embeddings:cm{self.embed_chunk_chars}"

        self.client = GigaChat(
            credentials=client_secret,
            verify_ssl=verify_ssl,
            scope=scope,
            auth_url=auth_endpoint,
            base_url=api_endpoint,
            ca_bundle_file=self.cert_path,
            timeout=300,
        )

    def get_query_params(self, llm_config: dict[str, Any] | None = None) -> dict:
        """Формирует параметры для запроса к LLM."""
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_gen_tokens": self.max_gen_tokens,
            "max_embed_tokens": self.max_embed_tokens,
            "top_p": self.top_p,
            "n": self.n,
            "stream": self.stream,
            "repetition_penalty": self.repetition_penalty,
            "profanity_check": self.profanity_check,
        }
        if llm_config:
            params.update(llm_config)
        return params

    async def chat(self, messages, llm_config: dict[str, Any] | None = None) -> str:
        """Отправляет запрос в chat API."""
        params = self.get_query_params(llm_config)

        formatted_messages = [{"role": role, "content": content} for role, content in messages]

        chat_payload = {
            "model": params["model"],
            "messages": formatted_messages,
            "temperature": params["temperature"],
            "max_tokens": params["max_gen_tokens"],
            "top_p": params["top_p"],
            "n": params["n"],
            "stream": params["stream"],
            "repetition_penalty": params["repetition_penalty"],
            "profanity_check": params["profanity_check"],
        }

        response = await retry_async(
            lambda: self.client.achat(payload=chat_payload),
            self.retry_policy,
            label="gigachat.chat",
        )
        return response.choices[0].message.content

    async def achat_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        functions: list[dict[str, Any]] | None = None,
        llm_config: dict[str, Any] | None = None,
    ):
        """Низкоуровневый chat с function-calling: messages уже в формате GigaChat
        (role/content/name/function_call), functions — список схем. Возвращает сырой ответ
        (response.choices[0].message + usage). Используется агентным rag-openai адаптером."""
        params = self.get_query_params(llm_config)
        payload: dict[str, Any] = {
            "model": params["model"],
            "messages": messages,
            "temperature": params["temperature"],
            "max_tokens": params["max_gen_tokens"],
            "top_p": params["top_p"],
            "repetition_penalty": params["repetition_penalty"],
            "profanity_check": params["profanity_check"],
        }
        if functions:
            payload["functions"] = functions
        return await retry_async(
            lambda: self.client.achat(payload=payload),
            self.retry_policy,
            label="gigachat.achat_raw",
        )

    # ------------------------------------------------------------------
    # Эмбеддинги
    # ------------------------------------------------------------------
    def _truncate(self, text: str, limit: int | None = None) -> str:
        limit = limit or self.max_embed_tokens
        return text[:limit] if len(text) > limit else text

    def _chunks(self, text: str) -> list[str]:
        """Куски ≤ embed_chunk_chars для mean-pool длинных запросов (раньше усечение до 512 симв.)."""
        return split_text_chunks(str(text), self.embed_chunk_chars)

    async def _aembed_once(self, texts: list[str]) -> list[list[float]]:
        """Один запрос эмбеддингов через единый ретрай (429/5xx/сеть). 413 не повторяется —
        его ловит _aembed_batch (деление/усечение)."""

        async def _call() -> list[list[float]]:
            response = await self.client.aembeddings(texts, model="Embeddings")
            return _extract_vectors(response, expected=len(texts))

        return await retry_async(_call, self.retry_policy, label="gigachat.embeddings")

    async def _aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Эмбеддит один пакет; при 413 рекурсивно делит пакет/усекает текст."""
        if not texts:
            return []
        try:
            return await self._aembed_once(texts)
        except (RequestEntityTooLargeError, ResponseError) as exc:
            if not _is_too_large(exc):
                raise
            if len(texts) == 1:
                shorter = self._truncate(texts[0], max(1, self.max_embed_tokens // 2))
                return await self._aembed_once([shorter])
            mid = len(texts) // 2
            left, right = await asyncio.gather(
                self._aembed_batch(texts[:mid]),
                self._aembed_batch(texts[mid:]),
            )
            return left + right

    async def aembeddings(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        max_concurrency: int | None = None,
    ) -> list[list[float]]:
        """Асинхронно эмбеддит список текстов пакетами и конкурентно.

        Первый пакет выполняется отдельно (тёплый старт: получение OAuth-токена и
        соединения), затем остальные — конкурентно. Порядок сохраняется.
        """
        if not texts:
            return []

        batch_size = batch_size or self.embed_batch_size
        max_concurrency = max_concurrency or self.embed_max_concurrency

        # каждый текст → 1+ кусок ≤ embed_chunk_chars; эмбеддим все куски, потом усредняем по тексту
        chunk_lists = [self._chunks(text) for text in texts]
        counts = [len(cl) for cl in chunk_lists]
        prepared = [c for cl in chunk_lists for c in cl]
        batches = [prepared[i : i + batch_size] for i in range(0, len(prepared), batch_size)]
        results: list[list[list[float]]] = [[] for _ in batches]

        # Тёплый старт: первый пакет один (избегаем гонки за токеном при concurrency>1).
        results[0] = await self._aembed_batch(batches[0])

        if len(batches) > 1:
            semaphore = asyncio.Semaphore(max_concurrency)

            async def run_one(index: int, batch: list[str]) -> None:
                async with semaphore:
                    results[index] = await self._aembed_batch(batch)

            await asyncio.gather(*(run_one(i, batches[i]) for i in range(1, len(batches))))

        flat: list[list[float]] = []
        for chunk in results:
            flat.extend(chunk)

        # mean-pool кусков обратно в один вектор на исходный текст. Без L2-нормы: при k=1
        # mean = сам вектор (короткие тексты ведут себя как раньше), downstream (PathIndex) нормализует.
        out: list[list[float]] = []
        pos = 0
        for k in counts:
            vecs = np.asarray(flat[pos : pos + k], dtype=float)
            pos += k
            out.append(vecs.mean(axis=0).tolist())
        return out

    def embeddings(
        self, texts: list[str], llm_config: dict[str, Any] | None = None
    ) -> list[list[float]]:
        """Синхронная обёртка над :meth:`aembeddings` (батчи + конкурентность)."""
        if not texts:
            return []
        return run_coro_blocking(self.aembeddings(list(texts)))


_client: GigaChatConnector | None = None


def get_client() -> GigaChatConnector:
    """Ленивый синглтон GigaChat-коннектора.

    Создаётся при первом обращении, а не на импорте: импорт модуля не требует секрета
    и сети (важно для офлайн-тестов и для частей пайплайна без LLM).
    """
    global _client
    if _client is None:
        secret = settings.get("GIGACHAT_SECRET")
        if not secret:
            raise RuntimeError("GIGACHAT_SECRET не задан. Укажите его в src/age_gap/settings/.env")
        embed_cfg = getattr(settings.gigachat, "embeddings", None)
        retry_cfg = getattr(settings.gigachat, "retry", None)
        _client = GigaChatConnector(
            client_secret=secret,
            cert_path=settings.gigachat.cert_path,
            verify_ssl=settings.gigachat.verify_ssl,
            api_endpoint=settings.gigachat.api_endpoint,
            auth_endpoint=settings.gigachat.auth_endpoint,
            model_name=settings.gigachat.model_name,
            scope=settings.gigachat.scope,
            temperature=settings.gigachat.llm_params.temperature,
            top_p=settings.gigachat.llm_params.top_p,
            max_gen_tokens=settings.gigachat.llm_params.max_gen_tokens,
            max_embed_tokens=settings.gigachat.llm_params.max_embed_tokens,
            repetition_penalty=settings.gigachat.llm_params.repetition_penalty,
            profanity_check=settings.gigachat.llm_params.profanity_check,
            embed_batch_size=getattr(embed_cfg, "batch_size", 16) if embed_cfg else 16,
            embed_max_concurrency=getattr(embed_cfg, "max_concurrency", 8) if embed_cfg else 8,
            retry_policy=policy_from_settings(retry_cfg),
        )
    return _client
