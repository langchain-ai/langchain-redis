import importlib
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from langchain_core.outputs import Generation

import langchain_redis.cache as cache_module
from langchain_redis import LangCacheSemanticCache


class DummyLangCacheSemanticCache:
    def __init__(
        self,
        name: str = "langcache",
        server_url: str = "https://api.langcache.com",
        cache_id: str = "",
        api_key: str = "",
        ttl: Optional[int] = None,
        use_exact_search: bool = True,
        use_semantic_search: bool = True,
        distance_scale: str = "normalized",
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
        cache = LangCacheSemanticCache(
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


def test_langcache_semantic_cache_name_defaulting() -> None:
    with patch(
        "redisvl.extensions.cache.llm.LangCacheSemanticCache",
        DummyLangCacheSemanticCache,
    ):
        c1 = LangCacheSemanticCache(
            name="cache_name",
            cache_id="test-cache-id",
            api_key="test-api-key",
        )
        assert c1.name() == "cache_name"

        c2 = LangCacheSemanticCache(
            cache_id="test-cache-id",
            api_key="test-api-key",
        )
        assert c2.name() == "llmcache"


def test_langcache_semantic_cache_default_server_url(monkeypatch: Any) -> None:
    """When LANGCACHE_SERVER_URL is not set, use the managed default endpoint."""

    # Ensure the env var is not set before reloading the module
    monkeypatch.delenv("LANGCACHE_SERVER_URL", raising=False)

    # Reload to re-evaluate the module-level default that reads the environment
    importlib.reload(cache_module)

    with patch(
        "redisvl.extensions.cache.llm.LangCacheSemanticCache",
        DummyLangCacheSemanticCache,
    ):
        cache = cache_module.LangCacheSemanticCache(
            cache_id="test-cache-id",
            api_key="test-api-key",
        )

        assert cache.cache.server_url == "https://aws-us-east-1.langcache.redis.io"


def test_langcache_semantic_cache_server_url_from_env(monkeypatch: Any) -> None:
    """LANGCACHE_SERVER_URL environment variable should override the default."""

    env_url = "https://example.langcache.internal"
    monkeypatch.setenv("LANGCACHE_SERVER_URL", env_url)

    # Reload to re-evaluate the module-level default that reads the environment
    importlib.reload(cache_module)

    with patch(
        "redisvl.extensions.cache.llm.LangCacheSemanticCache",
        DummyLangCacheSemanticCache,
    ):
        cache = cache_module.LangCacheSemanticCache(
            cache_id="test-cache-id",
            api_key="test-api-key",
        )

        assert cache.cache.server_url == env_url
