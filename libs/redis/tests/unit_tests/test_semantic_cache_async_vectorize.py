"""Regression test: RedisSemanticCache.aupdate() must not block the event loop.

`aupdate()` previously called the synchronous `_vectorize_prompt()` instead
of `await self.cache._avectorize_prompt()`. Since real embedding calls are
network-bound, this silently turned "async" cache updates into blocking
calls that stall the whole event loop for the duration of the embedding
request - exactly the problem async support is meant to avoid. Its sibling
method, `alookup()`, already awaited the async variant correctly.

This test asserts the async vectorize path is used (and the sync one is
not) by tracking which method is actually invoked, rather than comparing
output values (which would be identical for both paths and wouldn't catch
the regression).
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.outputs import Generation

from langchain_redis import RedisSemanticCache


class SpyRedisVLSemanticCache:
    """Mimics redisvl's SemanticCache but records which vectorize path is used."""

    def __init__(self) -> None:
        self.data: Dict[tuple, List[Dict[str, Any]]] = {}
        self.distance_threshold: float = 0.2
        self.index = Mock()
        self.index.name = "test_index"
        self.sync_vectorize_calls = 0
        self.async_vectorize_calls = 0

    def _vectorize_prompt(self, prompt: str) -> List[float]:
        self.sync_vectorize_calls += 1
        return [0.1, 0.2, 0.3]

    async def _avectorize_prompt(self, prompt: str) -> List[float]:
        self.async_vectorize_calls += 1
        return [0.1, 0.2, 0.3]

    async def acheck(self, vector: List[float]) -> List[Dict[str, Any]]:
        return []

    def store(
        self,
        prompt: str,
        response: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data[tuple(vector)] = [{"response": response, "metadata": metadata}]

    async def astore(
        self,
        prompt: str,
        response: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.store(prompt, response, vector, metadata)


class TestSemanticCacheAsyncVectorize:
    @pytest.fixture
    def mock_embeddings(self) -> Mock:
        embeddings = Mock(spec=Embeddings)
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        return embeddings

    @pytest.fixture
    def spy_cache(self, mock_embeddings: Mock) -> SpyRedisVLSemanticCache:
        spy = SpyRedisVLSemanticCache()
        with patch("langchain_redis.cache.RedisVLSemanticCache", return_value=spy):
            self.cache = RedisSemanticCache(
                embeddings=mock_embeddings, redis_url="redis://localhost:6379"
            )
        return spy

    @pytest.mark.asyncio
    async def test_aupdate_uses_async_vectorize_path(
        self, spy_cache: SpyRedisVLSemanticCache
    ) -> None:
        """aupdate() must call _avectorize_prompt, never the sync _vectorize_prompt."""
        await self.cache.aupdate(
            "test prompt", "test_llm", [Generation(text="test response")]
        )

        assert spy_cache.async_vectorize_calls == 1, (
            "aupdate() must use the async _avectorize_prompt path"
        )
        assert spy_cache.sync_vectorize_calls == 0, (
            "aupdate() must not call the blocking sync _vectorize_prompt"
        )

    @pytest.mark.asyncio
    async def test_alookup_uses_async_vectorize_path(
        self, spy_cache: SpyRedisVLSemanticCache
    ) -> None:
        """Sanity check: alookup() already did this correctly before the fix."""

        await self.cache.alookup("test prompt", "test_llm")

        assert spy_cache.async_vectorize_calls == 1
        assert spy_cache.sync_vectorize_calls == 0

    def test_update_uses_sync_vectorize_path(
        self, spy_cache: SpyRedisVLSemanticCache
    ) -> None:
        """Sanity check: the sync update() should keep using the sync path."""
        self.cache.update("test prompt", "test_llm", [Generation(text="test response")])

        assert spy_cache.sync_vectorize_calls == 1
        assert spy_cache.async_vectorize_calls == 0
