from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from langchain_core.embeddings import Embeddings
from langchain_core.outputs import Generation

from langchain_redis import LangCacheSemanticCache


class DummyLangCacheSemanticCache:
    def __init__(
        self,
        name: str = "langcache",
        server_url: str = "https://api.langcache.com",
        cache_id: str = "",
        api_key: str = "",
        ttl: Optional[int] = None,
        **_: object,
    ) -> None:
        self.name = name
        self.server_url = server_url
        self.cache_id = cache_id
        self.api_key = api_key
        self.ttl = ttl
        self.data: Dict[tuple[str, str], List[Dict[str, Any]]] = {}

    def store(
        self,
        prompt: str,
        response: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, str]] = None,
        filters: Optional[Dict[str, str]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        llm = (metadata or {}).get("llm_string", "")
        self.data[(prompt, llm)] = [{"response": response, "metadata": metadata or {}}]
        return "entry-id"

    def check(
        self,
        prompt: Optional[str] = None,
        vector: Optional[List[float]] = None,
        num_results: int = 1,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[object] = None,
        distance_threshold: Optional[float] = None,
        attributes: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        llm = (attributes or {}).get("llm_string", "")
        return self.data.get((prompt or "", llm), [])[:num_results]

    def clear(self) -> None:
        self.data.clear()

    async def astore(self, **kwargs: Any) -> str:
        return self.store(**kwargs)

    async def acheck(self, **kwargs: Any) -> List[Dict[str, Any]]:
        return self.check(**kwargs)

    async def aclear(self) -> None:
        self.clear()


def test_langcache_semantic_cache_update_lookup_and_clear() -> None:
    with patch(
        "redisvl.extensions.cache.llm.LangCacheSemanticCache",
        DummyLangCacheSemanticCache,
    ):
        embeddings = MagicMock(spec=Embeddings)
        cache = LangCacheSemanticCache(
            embeddings=embeddings,
            cache_id="test-cache-id",
            api_key="test-api-key",
        )

        prompt = "test prompt"
        llm_string = "test_llm"
        return_val = [Generation(text="test response")]

        cache.update(prompt, llm_string, return_val)
        result = cache.lookup(prompt, llm_string)

        assert result is not None
        assert len(result) == 1
        assert result[0].text == "test response"

        cache.clear()
        assert cache.lookup(prompt, llm_string) is None


def test_langcache_semantic_cache_name_prefix_mapping() -> None:
    with patch(
        "redisvl.extensions.cache.llm.LangCacheSemanticCache",
        DummyLangCacheSemanticCache,
    ):
        embeddings = MagicMock(spec=Embeddings)

        c1 = LangCacheSemanticCache(
            embeddings=embeddings,
            name="cache_name",
            prefix="tenant1",
            cache_id="test-cache-id",
            api_key="test-api-key",
        )
        assert c1.name() == "cache_name:tenant1"

        c2 = LangCacheSemanticCache(
            embeddings=embeddings,
            prefix="tenant2",
            cache_id="test-cache-id",
            api_key="test-api-key",
        )
        assert c2.name() == "tenant2"

        c3 = LangCacheSemanticCache(
            embeddings=embeddings,
            name="my_custom",
            cache_id="test-cache-id",
            api_key="test-api-key",
        )
        assert c3.name() == "my_custom"

        c4 = LangCacheSemanticCache(
            embeddings=embeddings,
            cache_id="test-cache-id",
            api_key="test-api-key",
        )
        assert c4.name() == "llmcache"
