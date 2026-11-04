"""Unit tests: RedisSemanticCache must scope entries by `llm_string`.

RedisVL derives a cache entry's key from the prompt and its `filters` only, and
`check()` returns a single nearest hit by default. Passing `llm_string` solely as
metadata therefore let entries for different models overwrite each other, and
let a closer hit from another model hide a valid one. The `llm_string` is now
stored as a tag filter, which separates the keys and is applied by the search.
"""

import hashlib
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.outputs import Generation
from redisvl.extensions.cache.llm.schema import (  # type: ignore[import]
    SemanticCacheIndexSchema,
)
from redisvl.query.filter import Tag  # type: ignore[import]

from langchain_redis import RedisSemanticCache

LLM_STRING_TAG = {"name": "llm_string_hash", "type": "tag"}


def _llm_string_hash(llm_string: str) -> str:
    return hashlib.md5(llm_string.encode()).hexdigest()


def _schema(dims: int = 3, with_tag: bool = True) -> SemanticCacheIndexSchema:
    schema = SemanticCacheIndexSchema.from_params("idx", "idx", dims, "float32")
    if with_tag:
        schema.add_field(LLM_STRING_TAG)
    return schema


def _existing_index(schema: SemanticCacheIndexSchema) -> Any:
    return patch(
        "langchain_redis.cache.SearchIndex.from_existing",
        return_value=Mock(schema=schema),
    )


@pytest.fixture
def mock_embeddings() -> Mock:
    embeddings = Mock(spec=Embeddings)
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    return embeddings


@pytest.fixture
def redisvl_cache_class() -> Generator[MagicMock, None, None]:
    with patch("langchain_redis.cache.RedisVLSemanticCache") as cache_class:
        cache_class.return_value.acheck = AsyncMock(return_value=[])
        cache_class.return_value.astore = AsyncMock()
        cache_class.return_value._avectorize_prompt = AsyncMock(
            return_value=[0.1, 0.2, 0.3]
        )
        yield cache_class


@pytest.fixture
def cache(mock_embeddings: Mock, redisvl_cache_class: MagicMock) -> RedisSemanticCache:
    return RedisSemanticCache(
        embeddings=mock_embeddings, redis_url="redis://localhost:6379"
    )


class TestSemanticCacheLLMStringScoping:
    def test_index_declares_llm_string_tag_field(
        self, cache: RedisSemanticCache, redisvl_cache_class: MagicMock
    ) -> None:
        kwargs = redisvl_cache_class.call_args.kwargs
        assert kwargs["filterable_fields"] == [LLM_STRING_TAG]

    def test_update_stores_entry_under_llm_string_hash(
        self, cache: RedisSemanticCache
    ) -> None:
        cache.update("prompt", "llm_a", [Generation(text="response")])

        kwargs = cache.cache.store.call_args.kwargs
        assert kwargs["filters"] == {"llm_string_hash": _llm_string_hash("llm_a")}
        assert kwargs["metadata"] == {"llm_string": "llm_a"}

    def test_lookup_filters_by_llm_string_hash(self, cache: RedisSemanticCache) -> None:
        cache.cache.check.return_value = []

        cache.lookup("prompt", "llm_a")

        expected = Tag("llm_string_hash") == _llm_string_hash("llm_a")
        kwargs = cache.cache.check.call_args.kwargs
        assert str(kwargs["filter_expression"]) == str(expected)

    async def test_aupdate_stores_entry_under_llm_string_hash(
        self, cache: RedisSemanticCache
    ) -> None:
        await cache.aupdate("prompt", "llm_a", [Generation(text="response")])

        kwargs = cache.cache.astore.call_args.kwargs
        assert kwargs["filters"] == {"llm_string_hash": _llm_string_hash("llm_a")}

    async def test_alookup_filters_by_llm_string_hash(
        self, cache: RedisSemanticCache
    ) -> None:
        await cache.alookup("prompt", "llm_a")

        expected = Tag("llm_string_hash") == _llm_string_hash("llm_a")
        kwargs = cache.cache.acheck.call_args.kwargs
        assert str(kwargs["filter_expression"]) == str(expected)


class TestSemanticCacheLegacyIndexDetection:
    def test_legacy_index_raises_migration_error(
        self, mock_embeddings: Mock, redisvl_cache_class: MagicMock
    ) -> None:
        expected_cache = MagicMock()
        redisvl_cache_class.side_effect = [ValueError("mismatch"), expected_cache]

        with patch.object(
            RedisSemanticCache, "_is_legacy_index", return_value=True
        ) as is_legacy_index:
            with pytest.raises(ValueError, match="migrate_index"):
                RedisSemanticCache(embeddings=mock_embeddings)

        is_legacy_index.assert_called_once_with(expected_cache.index)
        assert redisvl_cache_class.call_args.kwargs["create_index"] is False

    def test_other_schema_mismatch_is_reraised(
        self, mock_embeddings: Mock, redisvl_cache_class: MagicMock
    ) -> None:
        redisvl_cache_class.side_effect = [ValueError("mismatch"), MagicMock()]

        with patch.object(RedisSemanticCache, "_is_legacy_index", return_value=False):
            with pytest.raises(ValueError, match="mismatch"):
                RedisSemanticCache(embeddings=mock_embeddings)

    @pytest.mark.parametrize(
        ("existing_schema", "expected"),
        [
            pytest.param(_schema(with_tag=False), True, id="legacy"),
            pytest.param(_schema(), False, id="already_upgraded"),
            pytest.param(_schema(dims=4, with_tag=False), False, id="other_change"),
        ],
    )
    def test_is_legacy_index(
        self,
        cache: RedisSemanticCache,
        existing_schema: SemanticCacheIndexSchema,
        expected: bool,
    ) -> None:
        index = Mock(schema=_schema())
        index.name = "idx"
        index.exists.return_value = True

        with _existing_index(existing_schema):
            assert cache._is_legacy_index(index) is expected

    def test_missing_index_is_not_legacy(self, cache: RedisSemanticCache) -> None:
        index = Mock()
        index.exists.return_value = False

        with patch("langchain_redis.cache.SearchIndex.from_existing") as from_existing:
            assert cache._is_legacy_index(index) is False
        from_existing.assert_not_called()


class TestSemanticCacheMigrateIndex:
    @pytest.fixture
    def client(self) -> MagicMock:
        return MagicMock()

    def test_adds_tag_field_without_scanning_by_default(
        self, client: MagicMock
    ) -> None:
        with _existing_index(_schema(with_tag=False)):
            result = RedisSemanticCache.migrate_index(redis_client=client, name="idx")

        assert result == {"backfilled": 0, "skipped": 0}
        client.execute_command.assert_called_once_with(
            "FT.ALTER",
            "idx",
            "SKIPINITIALSCAN",
            "SCHEMA",
            "ADD",
            "llm_string_hash",
            "TAG",
        )
        client.scan_iter.assert_not_called()

    def test_does_not_alter_upgraded_index(self, client: MagicMock) -> None:
        with _existing_index(_schema()):
            RedisSemanticCache.migrate_index(redis_client=client, name="idx")

        client.execute_command.assert_not_called()

    def test_refuses_other_indexes(self, client: MagicMock) -> None:
        other_schema = SemanticCacheIndexSchema.from_params("idx", "idx", 3, "float32")
        other_schema.add_field({"name": "other", "type": "tag"})

        with _existing_index(other_schema):
            with pytest.raises(ValueError, match="not a semantic cache index"):
                RedisSemanticCache.migrate_index(redis_client=client, name="idx")

        client.execute_command.assert_not_called()
