"""Integration tests: RedisSemanticCache must isolate entries by `llm_string`.

The same prompt is routinely sent to several models (routers, fallback chains,
calls that differ only in temperature). Each `(prompt, llm_string)` pair must
get its own cache entry, and a lookup must only ever be answered by an entry
written for the same `llm_string`.
"""

import hashlib
import json
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.outputs import Generation
from redis import Redis
from redis.client import Pipeline
from redisvl.exceptions import RedisSearchError  # type: ignore[import]
from redisvl.extensions.cache.llm import (  # type: ignore[import]
    SemanticCache as RedisVLSemanticCache,
)
from redisvl.index import SearchIndex  # type: ignore[import]
from redisvl.redis.utils import array_to_buffer, hashify  # type: ignore[import]
from ulid import ULID

from langchain_redis import RedisSemanticCache
from langchain_redis.cache import EmbeddingsVectorizer

PROMPT = "What is the capital of France?"
NEAR_PROMPT = "Tell me the capital city of France."

# Fixed vectors so cosine distances are known exactly: PROMPT is the query,
# NEAR_PROMPT sits ~0.005 away from it, well within the default threshold.
VECTORS: Dict[str, List[float]] = {
    PROMPT: [1.0, 0.0, 0.0],
    NEAR_PROMPT: [1.0, 0.1, 0.0],
}


class FixedVectorEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return VECTORS.get(text, [0.0, 0.0, 1.0])


@pytest.fixture
def cache_name() -> str:
    return f"semcache_llm_string_{ULID()}"


@pytest.fixture
def semantic_cache(
    redis_url: str, cache_name: str
) -> Generator[RedisSemanticCache, None, None]:
    cache = RedisSemanticCache(
        embeddings=FixedVectorEmbeddings(),
        redis_url=redis_url,
        name=cache_name,
    )
    try:
        yield cache
    finally:
        cache.clear()
        cache.cache.delete()


class TestSemanticCacheLLMStringIsolation:
    def test_same_prompt_for_different_llms_does_not_overwrite(
        self, semantic_cache: RedisSemanticCache
    ) -> None:
        semantic_cache.update(PROMPT, "llm_a", [Generation(text="answer from a")])
        semantic_cache.update(PROMPT, "llm_b", [Generation(text="answer from b")])

        result_a = semantic_cache.lookup(PROMPT, "llm_a")
        result_b = semantic_cache.lookup(PROMPT, "llm_b")

        assert result_a is not None and result_a[0].text == "answer from a"
        assert result_b is not None and result_b[0].text == "answer from b"

    def test_same_prompt_and_llm_overwrites(
        self, semantic_cache: RedisSemanticCache
    ) -> None:
        semantic_cache.update(PROMPT, "llm_a", [Generation(text="old")])
        semantic_cache.update(PROMPT, "llm_a", [Generation(text="new")])

        result = semantic_cache.lookup(PROMPT, "llm_a")

        assert result is not None and result[0].text == "new"

    def test_lookup_ignores_closer_entry_from_another_llm(
        self, semantic_cache: RedisSemanticCache
    ) -> None:
        semantic_cache.update(NEAR_PROMPT, "llm_a", [Generation(text="from a")])
        # An exact match for the query, but written by a different model.
        semantic_cache.update(PROMPT, "llm_b", [Generation(text="from b")])

        result = semantic_cache.lookup(PROMPT, "llm_a")

        assert result is not None and result[0].text == "from a"

    def test_lookup_misses_when_only_another_llm_has_entry(
        self, semantic_cache: RedisSemanticCache
    ) -> None:
        semantic_cache.update(PROMPT, "llm_b", [Generation(text="from b")])

        assert semantic_cache.lookup(PROMPT, "llm_a") is None

    async def test_async_same_prompt_for_different_llms_does_not_overwrite(
        self, semantic_cache: RedisSemanticCache
    ) -> None:
        await semantic_cache.aupdate(PROMPT, "llm_a", [Generation(text="from a")])
        await semantic_cache.aupdate(PROMPT, "llm_b", [Generation(text="from b")])

        result_a = await semantic_cache.alookup(PROMPT, "llm_a")
        result_b = await semantic_cache.alookup(PROMPT, "llm_b")

        assert result_a is not None and result_a[0].text == "from a"
        assert result_b is not None and result_b[0].text == "from b"

    async def test_async_lookup_ignores_closer_entry_from_another_llm(
        self, semantic_cache: RedisSemanticCache
    ) -> None:
        await semantic_cache.aupdate(NEAR_PROMPT, "llm_a", [Generation(text="from a")])
        await semantic_cache.aupdate(PROMPT, "llm_b", [Generation(text="from b")])

        result = await semantic_cache.alookup(PROMPT, "llm_a")

        assert result is not None and result[0].text == "from a"


def _llm_string_hash(llm_string: str) -> str:
    return hashlib.md5(llm_string.encode()).hexdigest()


def _new_key(cache_name: str, prompt: str, llm_string: str) -> str:
    entry_id = hashify(prompt, {"llm_string_hash": _llm_string_hash(llm_string)})
    return f"{cache_name}:{entry_id}"


class TestSemanticCacheLegacyIndexMigration:
    """Indexes created before `llm_string` filtering lack the tag field."""

    @pytest.fixture
    def client(self, redis_url: str) -> Redis:
        return Redis.from_url(redis_url)

    @pytest.fixture
    def legacy(
        self, client: Redis, cache_name: str
    ) -> Generator[RedisVLSemanticCache, None, None]:
        # Mirrors how RedisSemanticCache built its index before the fix.
        cache = RedisVLSemanticCache(
            vectorizer=EmbeddingsVectorizer(embeddings=FixedVectorEmbeddings()),
            redis_client=client,
            name=cache_name,
            distance_threshold=0.2,
        )
        try:
            yield cache
        finally:
            # The index may have been upgraded, so clean up without schema checks.
            client.ft(cache_name).dropindex()
            cache.clear()

    @staticmethod
    def _store(
        legacy: RedisVLSemanticCache, llm_string: str = "llm_a", **kwargs: Any
    ) -> str:
        return legacy.store(
            prompt=PROMPT,
            response=json.dumps([dumps(Generation(text="legacy response"))]),
            vector=VECTORS[PROMPT],
            metadata={"llm_string": llm_string},
            **kwargs,
        )

    @staticmethod
    def _cache(client: Redis, cache_name: str) -> RedisSemanticCache:
        return RedisSemanticCache(
            embeddings=FixedVectorEmbeddings(), redis_client=client, name=cache_name
        )

    def test_legacy_index_raises_until_migrated(
        self, client: Redis, legacy: RedisVLSemanticCache, cache_name: str
    ) -> None:
        key = self._store(legacy)

        with pytest.raises(ValueError, match="migrate_index"):
            self._cache(client, cache_name)

        index = SearchIndex.from_existing(cache_name, redis_client=client)
        assert "llm_string_hash" not in index.schema.field_names
        assert client.exists(key)

    def test_mismatched_legacy_index_still_raises(
        self, client: Redis, legacy: RedisVLSemanticCache, cache_name: str
    ) -> None:
        # A different embedding size can't be fixed by migrating.
        class OtherDimsEmbeddings(FixedVectorEmbeddings):
            def embed_query(self, text: str) -> List[float]:
                return [1.0, 0.0, 0.0, 0.0]

        with pytest.raises(ValueError, match="does not match"):
            RedisSemanticCache(
                embeddings=OtherDimsEmbeddings(), redis_client=client, name=cache_name
            )

    def test_migrate_keeps_keys_and_starts_cold(
        self, client: Redis, legacy: RedisVLSemanticCache, cache_name: str
    ) -> None:
        key = self._store(legacy)

        result = RedisSemanticCache.migrate_index(redis_client=client, name=cache_name)

        assert result == {"backfilled": 0, "skipped": 0}
        assert client.exists(key)
        assert b"llm_string_hash" not in client.hkeys(key)

        cache = self._cache(client, cache_name)
        assert cache.lookup(PROMPT, "llm_a") is None
        cache.update(PROMPT, "llm_a", [Generation(text="from a")])
        cache.update(PROMPT, "llm_b", [Generation(text="from b")])
        result_a = cache.lookup(PROMPT, "llm_a")
        result_b = cache.lookup(PROMPT, "llm_b")
        assert result_a is not None and result_a[0].text == "from a"
        assert result_b is not None and result_b[0].text == "from b"

    def test_migrate_does_not_reindex_existing_entries(
        self, client: Redis, legacy: RedisVLSemanticCache, cache_name: str
    ) -> None:
        self._store(legacy)
        max_doc_id = client.ft(cache_name).info()["max_doc_id"]

        RedisSemanticCache.migrate_index(redis_client=client, name=cache_name)

        info = client.ft(cache_name).info()
        assert info["max_doc_id"] == max_doc_id
        assert int(info["indexing"]) == 0

    def test_migrated_cache_can_be_cleared(
        self, client: Redis, legacy: RedisVLSemanticCache, cache_name: str
    ) -> None:
        key = self._store(legacy)
        RedisSemanticCache.migrate_index(redis_client=client, name=cache_name)

        self._cache(client, cache_name).clear()

        assert not client.exists(key)

    def test_migrate_resolves_name_and_prefix_like_the_constructor(
        self, client: Redis, cache_name: str
    ) -> None:
        prefix = f"tenant_{ULID()}"
        full_name = f"{cache_name}:{prefix}"
        legacy = RedisVLSemanticCache(
            vectorizer=EmbeddingsVectorizer(embeddings=FixedVectorEmbeddings()),
            redis_client=client,
            name=full_name,
        )
        try:
            RedisSemanticCache.migrate_index(
                redis_client=client, name=cache_name, prefix=prefix
            )
            RedisSemanticCache(
                embeddings=FixedVectorEmbeddings(),
                redis_client=client,
                name=cache_name,
                prefix=prefix,
            )
        finally:
            client.ft(full_name).dropindex()
            legacy.clear()

    def test_migrate_can_be_rerun(
        self, client: Redis, legacy: RedisVLSemanticCache, cache_name: str
    ) -> None:
        RedisSemanticCache.migrate_index(redis_client=client, name=cache_name)
        RedisSemanticCache.migrate_index(redis_client=client, name=cache_name)

        self._cache(client, cache_name)

    def test_migrate_missing_index_raises(self, client: Redis, cache_name: str) -> None:
        with pytest.raises(RedisSearchError):
            RedisSemanticCache.migrate_index(redis_client=client, name=cache_name)

    def test_migrate_refuses_other_indexes(
        self, client: Redis, cache_name: str
    ) -> None:
        schema = {
            "index": {"name": cache_name, "prefix": cache_name},
            "fields": [{"name": "text", "type": "text"}],
        }
        index = SearchIndex.from_dict(schema, redis_client=client)
        index.create()
        try:
            with pytest.raises(ValueError, match="not a semantic cache index"):
                RedisSemanticCache.migrate_index(redis_client=client, name=cache_name)

            existing = SearchIndex.from_existing(cache_name, redis_client=client)
            assert "llm_string_hash" not in existing.schema.field_names
        finally:
            index.delete()

    def test_backfill_moves_entry_and_preserves_fields(
        self, client: Redis, legacy: RedisVLSemanticCache, cache_name: str
    ) -> None:
        key = self._store(legacy)
        client.hset(key, "custom_field", "custom value")
        before = client.hgetall(key)

        result = RedisSemanticCache.migrate_index(
            redis_client=client, name=cache_name, backfill=True
        )

        assert result == {"backfilled": 1, "skipped": 0}
        assert not client.exists(key)

        new_key = _new_key(cache_name, PROMPT, "llm_a")
        after = client.hgetall(new_key)
        assert after.pop(b"llm_string_hash") == _llm_string_hash("llm_a").encode()
        assert after.pop(b"entry_id") == new_key.split(":")[-1].encode()
        before.pop(b"entry_id")
        assert after == before

        cache = self._cache(client, cache_name)
        hit = cache.lookup(PROMPT, "llm_a")
        assert hit is not None and hit[0].text == "legacy response"
        assert cache.lookup(PROMPT, "llm_b") is None

    def test_backfilled_entry_is_overwritten_by_the_write_path(
        self, client: Redis, legacy: RedisVLSemanticCache, cache_name: str
    ) -> None:
        self._store(legacy)
        RedisSemanticCache.migrate_index(
            redis_client=client, name=cache_name, backfill=True
        )

        cache = self._cache(client, cache_name)
        cache.update(PROMPT, "llm_a", [Generation(text="new")])

        assert len(list(client.scan_iter(match=f"{cache_name}:*"))) == 1
        hit = cache.lookup(PROMPT, "llm_a")
        assert hit is not None and hit[0].text == "new"

    @pytest.mark.parametrize(
        "ttl",
        [
            pytest.param(None, id="no_expiration"),
            pytest.param(1000, id="with_expiration"),
        ],
    )
    def test_backfill_preserves_expiration(
        self,
        client: Redis,
        legacy: RedisVLSemanticCache,
        cache_name: str,
        ttl: Optional[int],
    ) -> None:
        key = self._store(legacy, ttl=ttl)
        expire_at = client.pexpiretime(key)

        RedisSemanticCache.migrate_index(
            redis_client=client, name=cache_name, backfill=True
        )

        new_key = _new_key(cache_name, PROMPT, "llm_a")
        assert client.pexpiretime(new_key) == expire_at

    @pytest.mark.parametrize("protocol", [2, 3])
    def test_backfill_with_decoding_client_keeps_binary_vector(
        self,
        redis_url: str,
        client: Redis,
        legacy: RedisVLSemanticCache,
        cache_name: str,
        protocol: int,
    ) -> None:
        self._store(legacy)
        decoding_client = Redis.from_url(
            redis_url, decode_responses=True, protocol=protocol
        )

        result = RedisSemanticCache.migrate_index(
            redis_client=decoding_client, name=cache_name, backfill=True
        )

        assert result == {"backfilled": 1, "skipped": 0}
        new_key = _new_key(cache_name, PROMPT, "llm_a")
        assert client.hget(new_key, "prompt_vector") == array_to_buffer(
            VECTORS[PROMPT], "float32"
        )

    @pytest.mark.parametrize(
        "metadata",
        [
            pytest.param(None, id="missing"),
            pytest.param("not json", id="malformed"),
            pytest.param("[]", id="not_a_mapping"),
            pytest.param("{}", id="no_llm_string"),
            pytest.param('{"llm_string": 1}', id="llm_string_not_a_string"),
        ],
    )
    def test_backfill_skips_unreadable_entries(
        self,
        client: Redis,
        legacy: RedisVLSemanticCache,
        cache_name: str,
        metadata: Any,
    ) -> None:
        key = self._store(legacy)
        if metadata is None:
            client.hdel(key, "metadata")
        else:
            client.hset(key, "metadata", metadata)

        result = RedisSemanticCache.migrate_index(
            redis_client=client, name=cache_name, backfill=True
        )

        assert result == {"backfilled": 0, "skipped": 1}
        assert client.exists(key)
        assert len(list(client.scan_iter(match=f"{cache_name}:*"))) == 1

    def test_backfill_after_cold_migration(
        self, client: Redis, legacy: RedisVLSemanticCache, cache_name: str
    ) -> None:
        self._store(legacy)
        RedisSemanticCache.migrate_index(redis_client=client, name=cache_name)

        result = RedisSemanticCache.migrate_index(
            redis_client=client, name=cache_name, backfill=True
        )

        assert result == {"backfilled": 1, "skipped": 0}
        hit = self._cache(client, cache_name).lookup(PROMPT, "llm_a")
        assert hit is not None and hit[0].text == "legacy response"

    def test_backfill_keeps_source_when_write_fails_and_can_be_rerun(
        self, client: Redis, legacy: RedisVLSemanticCache, cache_name: str
    ) -> None:
        key = self._store(legacy)

        with patch.object(Pipeline, "execute", side_effect=ConnectionError):
            with pytest.raises(ConnectionError):
                RedisSemanticCache.migrate_index(
                    redis_client=client, name=cache_name, backfill=True
                )

        new_key = _new_key(cache_name, PROMPT, "llm_a")
        assert client.exists(key)
        assert not client.exists(new_key)

        result = RedisSemanticCache.migrate_index(
            redis_client=client, name=cache_name, backfill=True
        )

        assert result == {"backfilled": 1, "skipped": 0}
        assert not client.exists(key)
        assert client.exists(new_key)

    def test_backfill_rerun_after_interruption_before_source_removal(
        self, client: Redis, legacy: RedisVLSemanticCache, cache_name: str
    ) -> None:
        key = self._store(legacy, ttl=1000)

        with patch.object(Redis, "delete", side_effect=ConnectionError):
            with pytest.raises(ConnectionError):
                RedisSemanticCache.migrate_index(
                    redis_client=client, name=cache_name, backfill=True
                )

        new_key = _new_key(cache_name, PROMPT, "llm_a")
        expire_at = client.pexpiretime(new_key)
        assert client.exists(key)
        assert expire_at == client.pexpiretime(key)

        result = RedisSemanticCache.migrate_index(
            redis_client=client, name=cache_name, backfill=True
        )

        assert result == {"backfilled": 1, "skipped": 0}
        assert not client.exists(key)
        assert client.pexpiretime(new_key) == expire_at

    def test_backfill_does_not_replace_newer_entry(
        self, client: Redis, legacy: RedisVLSemanticCache, cache_name: str
    ) -> None:
        key = self._store(legacy)
        RedisSemanticCache.migrate_index(redis_client=client, name=cache_name)
        cache = self._cache(client, cache_name)
        cache.update(PROMPT, "llm_a", [Generation(text="newer")])

        RedisSemanticCache.migrate_index(
            redis_client=client, name=cache_name, backfill=True
        )

        assert not client.exists(key)
        hit = cache.lookup(PROMPT, "llm_a")
        assert hit is not None and hit[0].text == "newer"

    def test_backfill_ignores_cache_sharing_the_key_pattern(
        self, client: Redis, legacy: RedisVLSemanticCache, cache_name: str
    ) -> None:
        other_key = f"{cache_name}:tenant:{hashify(PROMPT)}"
        client.hset(
            other_key,
            mapping={
                "entry_id": hashify(PROMPT),
                "prompt": PROMPT,
                "metadata": json.dumps({"llm_string": "llm_a"}),
            },
        )

        result = RedisSemanticCache.migrate_index(
            redis_client=client, name=cache_name, backfill=True
        )

        assert result == {"backfilled": 0, "skipped": 0}
        assert client.exists(other_key)
