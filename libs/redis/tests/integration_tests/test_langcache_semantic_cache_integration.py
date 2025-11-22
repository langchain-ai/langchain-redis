"""Integration tests for LangCacheSemanticCache against the LangCache managed service.

These tests exercise the real LangCache API using two configured caches:
- One with attributes configured
- One without attributes configured

Env vars (injected via CI or your shell):
- LANGCACHE_WITH_ATTRIBUTES_API_KEY
- LANGCACHE_WITH_ATTRIBUTES_CACHE_ID
- LANGCACHE_WITH_ATTRIBUTES_URL
- LANGCACHE_NO_ATTRIBUTES_API_KEY
- LANGCACHE_NO_ATTRIBUTES_CACHE_ID
- LANGCACHE_NO_ATTRIBUTES_URL

If any of the required variables for a given cache are missing, the
corresponding tests will be skipped.
"""

from __future__ import annotations

import os
from typing import Dict

import dotenv
import pytest
from langchain_core.outputs import Generation

from langchain_redis import LangCacheSemanticCache

try:  # Optional direct redisvl client for debugging comparisons
    from redisvl.extensions.cache.llm import (  # type: ignore
        LangCacheSemanticCache as RedisVLLangCacheSemanticCache,
    )
except Exception:  # pragma: no cover - optional dependency path
    RedisVLLangCacheSemanticCache = None


REQUIRED_WITH_ATTRS_VARS = (
    "LANGCACHE_WITH_ATTRIBUTES_API_KEY",
    "LANGCACHE_WITH_ATTRIBUTES_CACHE_ID",
    "LANGCACHE_WITH_ATTRIBUTES_URL",
)

REQUIRED_NO_ATTRS_VARS = (
    "LANGCACHE_NO_ATTRIBUTES_API_KEY",
    "LANGCACHE_NO_ATTRIBUTES_CACHE_ID",
    "LANGCACHE_NO_ATTRIBUTES_URL",
)

dotenv.load_dotenv()


def _require_env_vars(var_names: tuple[str, ...]) -> Dict[str, str]:
    """Return a mapping of required env vars or skip tests if any are missing."""

    missing = [name for name in var_names if not os.getenv(name)]
    if missing:
        pytest.skip(
            "Missing required LangCache env vars: "
            f"{', '.join(missing)}. "
            "Set them locally or in CI secrets to run these tests.",
        )

    return {name: os.environ[name] for name in var_names}


@pytest.fixture
def langcache_with_attrs() -> LangCacheSemanticCache:
    """LangCacheSemanticCache bound to a cache with attributes configured."""

    env = _require_env_vars(REQUIRED_WITH_ATTRS_VARS)

    return LangCacheSemanticCache(
        name="langchain_redis_with_attributes",
        server_url=env["LANGCACHE_WITH_ATTRIBUTES_URL"],
        cache_id=env["LANGCACHE_WITH_ATTRIBUTES_CACHE_ID"],
        api_key=env["LANGCACHE_WITH_ATTRIBUTES_API_KEY"],
        ttl=60,
        distance_threshold=0.2,
    )


@pytest.fixture
def langcache_no_attrs() -> LangCacheSemanticCache:
    """LangCacheSemanticCache bound to a cache with *no* attributes configured."""

    env = _require_env_vars(REQUIRED_NO_ATTRS_VARS)

    return LangCacheSemanticCache(
        name="langchain_redis_no_attributes",
        server_url=env["LANGCACHE_NO_ATTRIBUTES_URL"],
        cache_id=env["LANGCACHE_NO_ATTRIBUTES_CACHE_ID"],
        api_key=env["LANGCACHE_NO_ATTRIBUTES_API_KEY"],
        ttl=60,
        distance_threshold=0.2,
    )


@pytest.fixture
def rv_langcache_with_attrs() -> "RedisVLLangCacheSemanticCache":
    """Direct redisvl LangCacheSemanticCache bound to attrs-enabled cache.

    This bypasses the langchain_redis adapter and talks to LangCache via the
    redisvl integration directly, without using any llm_string attribute
    filtering. These tests help isolate whether the underlying cache instance
    can successfully store and retrieve entries when attributes are not used
    as filters.
    """

    if RedisVLLangCacheSemanticCache is None:
        pytest.skip("redisvl LangCacheSemanticCache is not available")

    env = _require_env_vars(REQUIRED_WITH_ATTRS_VARS)

    return RedisVLLangCacheSemanticCache(
        name="redisvl_with_attributes_direct",
        server_url=env["LANGCACHE_WITH_ATTRIBUTES_URL"],
        cache_id=env["LANGCACHE_WITH_ATTRIBUTES_CACHE_ID"],
        api_key=env["LANGCACHE_WITH_ATTRIBUTES_API_KEY"],
        ttl=60,
    )


@pytest.mark.requires_api_keys
class TestLangCacheSemanticCacheIntegrationWithAttributes:
    def test_update_and_lookup_roundtrip(
        self, langcache_with_attrs: LangCacheSemanticCache
    ) -> None:
        """Basic sync round-trip using the managed LangCache service."""

        prompt = "What is Redis?"
        llm_string = "langchain-redis/tests:with-attributes:sync"
        result = [Generation(text="Redis is an in-memory data store.")]

        langcache_with_attrs.update(prompt, llm_string, result)
        hits = langcache_with_attrs.lookup(prompt, llm_string)

        assert hits is not None
        assert len(hits) == 1
        assert hits[0].text == "Redis is an in-memory data store."

    @pytest.mark.asyncio
    async def test_async_update_and_lookup_roundtrip(
        self, langcache_with_attrs: LangCacheSemanticCache
    ) -> None:
        """Async round-trip using the managed LangCache service."""

        prompt = "What is Redis (async)?"
        llm_string = "langchain-redis/tests:with-attributes:async"
        result = [Generation(text="Redis is an in-memory data store (async).")]

        await langcache_with_attrs.aupdate(prompt, llm_string, result)
        hits = await langcache_with_attrs.alookup(prompt, llm_string)

        assert hits is not None
        assert len(hits) == 1
        assert hits[0].text == "Redis is an in-memory data store (async)."

    def test_clear_removes_entries(
        self, langcache_with_attrs: LangCacheSemanticCache
    ) -> None:
        """clear() should remove entries from the underlying cache."""

        prompt = "Delete me (sync)"
        llm_string = "langchain-redis/tests:with-attributes:clear-sync"
        result = [Generation(text="You should not see this after clear().")]

        langcache_with_attrs.update(prompt, llm_string, result)
        assert langcache_with_attrs.lookup(prompt, llm_string) is not None

        langcache_with_attrs.clear()
        assert langcache_with_attrs.lookup(prompt, llm_string) is None

    @pytest.mark.asyncio
    async def test_aclear_removes_entries(
        self, langcache_with_attrs: LangCacheSemanticCache
    ) -> None:
        """aclear() should remove entries from the underlying cache."""

        prompt = "Delete me (async)"
        llm_string = "langchain-redis/tests:with-attributes:clear-async"
        result = [Generation(text="You should not see this after aclear().")]

        await langcache_with_attrs.aupdate(prompt, llm_string, result)
        assert await langcache_with_attrs.alookup(prompt, llm_string) is not None

        await langcache_with_attrs.aclear()
        assert await langcache_with_attrs.alookup(prompt, llm_string) is None

    @pytest.mark.requires_api_keys
    class TestLangCacheSemanticCacheIntegrationWithoutAttributes:
        def test_update_errors_when_no_attributes_configured(
            self, langcache_no_attrs: LangCacheSemanticCache
        ) -> None:
            """update() should raise if attributes are not configured on the cache."""

            prompt = "Attributes not configured (sync)"
            llm_string = "langchain-redis/tests:no-attributes:sync"

            with pytest.raises(RuntimeError) as exc:
                langcache_no_attrs.update(
                    prompt,
                    llm_string,
                    [Generation(text="This should not be stored.")],
                )

            assert (
                "attributes are not configured for this cache" in str(exc.value).lower()
            )

    def test_lookup_errors_when_no_attributes_configured(
        self, langcache_no_attrs: LangCacheSemanticCache
    ) -> None:
        """lookup() should raise if attributes are not configured on the cache."""

        prompt = "Attributes not configured (lookup)"
        llm_string = "langchain-redis/tests:no-attributes:lookup"

        with pytest.raises(RuntimeError) as exc:
            _ = langcache_no_attrs.lookup(prompt, llm_string)

        assert "attributes are not configured for this cache" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_async_update_and_lookup_error_when_no_attributes_configured(
        self, langcache_no_attrs: LangCacheSemanticCache
    ) -> None:
        """Async variants should also raise when attributes are not configured."""

        prompt = "Attributes not configured (async)"
        llm_string = "langchain-redis/tests:no-attributes:async"

        with pytest.raises(RuntimeError) as exc_update:
            await langcache_no_attrs.aupdate(
                prompt,
                llm_string,
                [Generation(text="This should not be stored (async).")],
            )

        assert (
            "attributes are not configured for this cache"
            in str(exc_update.value).lower()
        )

        with pytest.raises(RuntimeError) as exc_lookup:
            await langcache_no_attrs.alookup(prompt, llm_string)

        assert (
            "attributes are not configured for this cache"
            in str(exc_lookup.value).lower()
        )
