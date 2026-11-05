"""Redis cache implementation for LangChain."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import quote

import numpy as np
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from pydantic import ConfigDict, Field
from redis import Redis
from redis.client import NEVER_DECODE
from redis.commands.json.path import Path
from redis.exceptions import ResponseError
from redis.utils import str_if_bytes
from redisvl.extensions.cache.llm import (  # type: ignore[import]
    SemanticCache as RedisVLSemanticCache,
)
from redisvl.index import SearchIndex  # type: ignore[import]
from redisvl.query.filter import Tag  # type: ignore[import]
from redisvl.redis.utils import hashify  # type: ignore[import]
from redisvl.schema.fields import VectorDataType  # type: ignore[import]
from redisvl.utils.vectorize import BaseVectorizer  # type: ignore[import]

from langchain_redis.version import __full_lib_name__

# Default cache name used across semantic caches
_DEFAULT_CACHE_NAME = "llmcache"

# Default cache prefix used across semantic caches
_DEFAULT_CACHE_PREFIX = "llmcache"

# Tag field that scopes semantic cache entries to the LLM that produced them
_LLM_STRING_HASH_FIELD_NAME = "llm_string_hash"
_LLM_STRING_HASH_FIELD = {"name": _LLM_STRING_HASH_FIELD_NAME, "type": "tag"}

# Fields of a semantic cache index created before entries were scoped by LLM
_LEGACY_INDEX_FIELD_NAMES = {
    "prompt",
    "response",
    "inserted_at",
    "updated_at",
    "prompt_vector",
}

# Default LangCache server URL; prefer the URL provided for your cache.
# Can be overridden via the LANGCACHE_SERVER_URL environment variable.
_DEFAULT_LANGCACHE_SERVER_URL = os.environ.get(
    "LANGCACHE_SERVER_URL", "https://aws-us-east-1.langcache.redis.io"
)

if TYPE_CHECKING:  # pragma: no cover
    from redisvl.extensions.cache.llm import (
        LangCacheSemanticCache as TC_LangCacheSemanticCache,
    )
else:
    TC_LangCacheSemanticCache = Any  # type: ignore[misc]


logger = logging.getLogger(__name__)


def _hash_llm_string(llm_string: str) -> str:
    """Hash an `llm_string` into a value that is safe to use as a tag.

    Raw `llm_string` values are long and contain characters that tag fields
    treat as separators or require escaping, so the hex digest is used instead.
    """
    return hashlib.md5(llm_string.encode()).hexdigest()


class EmbeddingsVectorizer(BaseVectorizer):
    embeddings: Embeddings = Field(...)
    model: str = Field(default="custom_embeddings")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, embeddings: Embeddings):
        dims = len(embeddings.embed_query("test"))
        super().__init__(model="custom_embeddings", dims=dims, embeddings=embeddings)

    def encode(
        self,
        texts: Union[str, List[str]],
        dtype: Union[str, VectorDataType],
        **kwargs: Any,
    ) -> np.ndarray:
        if isinstance(dtype, VectorDataType):
            dtype = dtype.value.lower()
        if isinstance(texts, str):
            return np.array(self.embeddings.embed_query(texts), dtype=dtype)
        return np.array(self.embeddings.embed_documents(texts), dtype=dtype)

    def embed(
        self,
        text: str,
        dtype: Union[str, VectorDataType] = "float32",
        **kwargs: Any,
    ) -> List[float]:
        return self.encode(text, dtype, **kwargs).tolist()

    def embed_many(
        self,
        texts: List[str],
        dtype: Union[str, VectorDataType] = "float32",
        **kwargs: Any,
    ) -> List[List[float]]:
        return self.encode(texts, dtype, **kwargs).tolist()

    async def aembed(
        self,
        text: str,
        dtype: Union[str, VectorDataType] = "float32",
        **kwargs: Any,
    ) -> List[float]:
        return await asyncio.to_thread(self.embed, text, dtype, **kwargs)

    async def aembed_many(
        self,
        texts: List[str],
        dtype: Union[str, VectorDataType] = "float32",
        **kwargs: Any,
    ) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_many, texts, dtype, **kwargs)


class RedisCache(BaseCache):
    """Redis cache implementation for LangChain.

    This class provides a Redis-based caching mechanism for LangChain, allowing
    storage and retrieval of language model responses.

    Attributes:
        redis (Redis): The Redis client instance.
        ttl (Optional[int]): Time-to-live for cache entries in seconds.

            If `None`, entries don't expire.
        prefix (Optional[str]): Prefix for all keys stored in Redis.

    Args:
        redis_url (str): The URL of the Redis instance to connect to.
        ttl (Optional[int]): Time-to-live for cache entries in seconds.
        prefix (Optional[str]): Prefix for all keys stored in Redis.
        redis_client (Optional[Redis]): An existing Redis client instance.

            If provided, `redis_url` is ignored.

    Example:
        ```python
        from langchain_redis import RedisCache
        from langchain_core.globals import set_llm_cache

        # Create a Redis cache instance
        redis_cache = RedisCache(redis_url="redis://localhost:6379", ttl=3600)

        # Set it as the global LLM cache
        set_llm_cache(redis_cache)

        # Now, when you use an LLM, it will automatically use this cache
        ```

    Note:
        - This cache implementation uses Redis JSON capabilities to store
            structured data.
        - The cache key is created using MD5 hashes of the prompt and LLM string.
        - If TTL is set, cache entries will automatically expire
            after the specified duration.
        - The prefix can be used to namespace cache entries.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl: Optional[int] = None,
        prefix: Optional[str] = "redis",
        redis_client: Optional[Redis] = None,
    ):
        if redis_client is not None:
            self.redis = redis_client
        elif redis_url.startswith("redis+sentinel://"):
            # For Sentinel URLs, use RedisVL's connection factory
            from redisvl.redis.connection import (  # type: ignore[import-untyped]
                RedisConnectionFactory,
            )

            self.redis = RedisConnectionFactory.get_redis_connection(
                redis_url=redis_url
            )
        else:
            self.redis = Redis.from_url(redis_url)

        try:
            self.redis.client_setinfo("LIB-NAME", __full_lib_name__)  # type: ignore
        except ResponseError:
            # Fall back to a simple log echo
            self.redis.echo(__full_lib_name__)
        self.ttl = ttl
        self.prefix = prefix

    def _key(self, prompt: str, llm_string: str) -> str:
        """Create a key for the cache."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        llm_string_hash = hashlib.md5(llm_string.encode()).hexdigest()
        return f"{self.prefix}:{prompt_hash}:{llm_string_hash}"

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up the result of a previous language model call in the Redis cache.

        This method checks if there's a cached result for the given prompt and language
        model combination.

        Args:
            prompt (str): The input prompt for which to look up the cached result.
            llm_string (str): A string representation of the language model and
                its parameters.

        Returns:
            The cached result if found, or `None` if not present in the cache.

                The result is typically a list of `Generation` objects.

        Example:
            ```python
            cache = RedisCache(redis_url="redis://localhost:6379")
            prompt = "What is the capital of France?"
            llm_string = "openai/gpt-3.5-turbo"

            result = cache.lookup(prompt, llm_string)
            if result:
                print("Cache hit:", result[0].text)
            else:
                print("Cache miss")
            ```

        Note:
            - The method uses an MD5 hash of the `prompt` and `llm_string` to create
                the cache key.
            - The cached value is stored as JSON and parsed back into a list of
                `Generation` objects.
            - If the key is missing or the stored value is empty, `None` is
                returned.
            - This method is typically called internally by LangChain, but can be used
                directly for manual cache interactions.
        """
        key = self._key(prompt, llm_string)
        result = self.redis.json().get(key)
        if not result:
            return None
        if isinstance(result, list):
            # `allowed_objects="core"` matches the current default explicitly, so
            # this keeps existing behavior stable when langchain-core changes its
            # default (see LangChainPendingDeprecationWarning) instead of silently
            # picking up a narrower allowlist on a future langchain-core upgrade.
            return [loads(json.dumps(gen), allowed_objects="core") for gen in result]
        # Entries written before multi-generation support stored a single
        # Generation as the JSON root instead of a list.
        return [loads(json.dumps(result), allowed_objects="core")]

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update the cache with a new result for a given prompt and language model.

        This method stores a new result in the Redis cache for the specified prompt and
        language model combination.

        Args:
            prompt (str): The input prompt associated with the result.
            llm_string (str): A string representation of the language model
                and its parameters.
            return_val (RETURN_VAL_TYPE): The result to be cached, typically a list
                of `Generation` objects.

        Example:
            ```python
            from langchain_core.outputs import Generation

            cache = RedisCache(redis_url="redis://localhost:6379", ttl=3600)
            prompt = "What is the capital of France?"
            llm_string = "openai/gpt-3.5-turbo"
            result = [Generation(text="The capital of France is Paris.")]

            cache.update(prompt, llm_string, result)
            ```

        Note:
            - The method uses an MD5 hash of the `prompt` and `llm_string` to create the
                cache key.
            - The result is stored as JSON in Redis.
            - If a TTL (Time To Live) was specified when initializing the cache,
                it will be applied to this entry.
            - This method is typically called internally by LangChain after a language
                model generates a response, but it can be used directly
                for manual cache updates.
            - If the cache already contains an entry for this `prompt` and `llm_string`,
                it will be overwritten.
        """
        key = self._key(prompt, llm_string)
        json_value = [json.loads(dumps(gen)) for gen in return_val]
        self.redis.json().set(key, Path.root_path(), json_value)
        if self.ttl is not None:
            self.redis.expire(key, self.ttl)

    def clear(self, **kwargs: Any) -> None:
        """Clear all entries in the Redis cache that match the cache prefix.

        This method removes all cache entries that start with the specified prefix.

        Args:
            **kwargs: Additional keyword arguments.

                Currently not used, but included for potential future extensions.

        Example:
            ```python
            from langchain_core.outputs import Generation

            cache = RedisCache(
                redis_url="redis://localhost:6379",
                prefix="my_cache"
            )

            # Add some entries to the cache
            cache.update("prompt1", "llm1", [Generation(text="Result 1")])
            cache.update("prompt2", "llm2", [Generation(text="Result 2")])

            # Clear all entries
            cache.clear()

            # After this, all entries with keys starting with "my_cache:"
            # will be removed
            ```

        Note:
            - This method uses Redis SCAN to iterate over keys, which is safe
                for large datasets.
            - It scans with a `COUNT` hint of 100 and deletes each batch of
                matching keys.
            - Only keys that start with the specified prefix (default is `'redis:'`)
                will be deleted.
            - This operation is irreversible. Make sure you want to clear all cached
                data before calling this method.
            - If no keys match the prefix, the method will complete without any errors.
        """
        cursor = 0
        pipe = self.redis.pipeline()
        while True:
            try:
                cursor, keys = self.redis.scan(
                    cursor, match=f"{self.prefix}:*", count=100
                )  # type: ignore[misc]
                if keys:
                    pipe.delete(*keys)
                    pipe.execute()

                if cursor == 0:
                    break
            finally:
                pipe.reset()


class RedisSemanticCache(BaseCache):
    """Redis-based semantic cache implementation for LangChain.

    This class provides a semantic caching mechanism using Redis and vector similarity
    search. It allows for storing and retrieving language model responses based on the
    semantic similarity of prompts, rather than exact string matching.

    Attributes:
        redis (Redis): The Redis client instance.
        embeddings (Embeddings): The embedding function to use for encoding prompts.
        cache (RedisVLSemanticCache): The underlying RedisVL semantic cache instance.

    Args:
        embeddings (Embeddings): The embedding function to use for encoding prompts.
        redis_url (str): The URL of the Redis instance to connect to.
        distance_threshold (float): The maximum distance for considering a cache hit.
        ttl (Optional[int]): Time-to-live for cache entries in seconds.
        name (Optional[str]): Name for the cache index.

            Defaults to `'llmcache'`.
        prefix (Optional[str]): Prefix for all keys stored in Redis.

            Defaults to `'llmcache'`. A non-default `prefix` becomes the cache's
            index name and key prefix, combined as `'{name}:{prefix}'` when a
            different non-default `name` is also given.
        redis_client (Optional[Redis]): An existing Redis client instance.

            If provided, `redis_url` is ignored.

    Example:
        ```python
        from langchain_redis import RedisSemanticCache
        from langchain_openai import OpenAIEmbeddings
        from langchain_core.globals import set_llm_cache

        embeddings = OpenAIEmbeddings()
        semantic_cache = RedisSemanticCache(
            embeddings=embeddings,
            redis_url="redis://localhost:6379",
            distance_threshold=0.1
        )

        set_llm_cache(semantic_cache)

        # Now, when you use an LLM, it will automatically use this semantic cache
        ```

    Note:
        - This cache uses vector similarity search to find semantically similar prompts.
        - The distance_threshold determines how similar a prompt must be to trigger
            a cache hit.
        - Lowering the distance_threshold increases precision but may reduce cache hits.
        - The cache uses the RedisVL library for efficient vector storage and retrieval.
        - Semantic caching can be more flexible than exact matching, allowing cache hits
            for prompts that are semantically similar but not identical.
        - Entries are scoped by `llm_string`, so different models (or the same
            model with different parameters) never share or overwrite entries.
        - An existing index created by an earlier version raises a `ValueError`
            until it is upgraded with `migrate_index()`.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        redis_url: str = "redis://localhost:6379",
        distance_threshold: float = 0.2,
        ttl: Optional[int] = None,
        name: Optional[str] = _DEFAULT_CACHE_NAME,
        prefix: Optional[str] = _DEFAULT_CACHE_PREFIX,
        redis_client: Optional[Redis] = None,
    ):
        self.redis = self._get_redis_client(redis_url, redis_client)
        self.embeddings = embeddings
        self.prefix = prefix
        vectorizer = EmbeddingsVectorizer(embeddings=self.embeddings)

        cache_kwargs: Dict[str, Any] = {
            "vectorizer": vectorizer,
            "redis_client": self.redis,
            "distance_threshold": distance_threshold,
            "ttl": ttl,
            "name": self._get_cache_name(name, prefix),
            "filterable_fields": [_LLM_STRING_HASH_FIELD],
        }
        try:
            self.cache = RedisVLSemanticCache(**cache_kwargs)
        except ValueError as e:
            # Built only to read the expected schema; it issues no index commands.
            index = RedisVLSemanticCache(create_index=False, **cache_kwargs).index
            if not self._is_legacy_index(index):
                raise
            raise ValueError(
                f"Semantic cache index '{index.name}' was created by an earlier "
                "version of langchain-redis and cannot scope entries by llm_string. "
                "Upgrade it by calling RedisSemanticCache.migrate_index() with the "
                "same connection, name and prefix."
            ) from e

    @staticmethod
    def _get_redis_client(redis_url: str, redis_client: Optional[Redis]) -> Redis:
        """Return `redis_client` if provided, otherwise connect to `redis_url`."""
        if redis_client is not None:
            return redis_client
        if redis_url.startswith("redis+sentinel://"):
            # For Sentinel URLs, use RedisVL's connection factory
            from redisvl.redis.connection import (  # type: ignore[import-untyped]
                RedisConnectionFactory,
            )

            return RedisConnectionFactory.get_redis_connection(redis_url=redis_url)
        return Redis.from_url(redis_url)

    @staticmethod
    def _get_cache_name(name: Optional[str], prefix: Optional[str]) -> Optional[str]:
        """Map `name` and `prefix` to the name of the RedisVL cache."""
        # RedisVL's SemanticCache uses 'name' as the prefix for keys.
        # To support the 'prefix' parameter for multi-tenant isolation,
        # we need to map it appropriately:
        # - If both name and prefix are provided and different, combine them
        # - If only prefix is provided (and differs from default), use it
        # - Otherwise use name (maintains backward compatibility)
        if prefix and prefix != _DEFAULT_CACHE_PREFIX:
            if name and name != _DEFAULT_CACHE_NAME and name != prefix:
                # Both are provided and different: combine them
                return f"{name}:{prefix}"
            # Only prefix is meaningfully set: use it
            return prefix
        return name

    def _is_legacy_index(self, index: SearchIndex) -> bool:
        """Check whether `index` exists with the schema used before entries were
        scoped by `llm_string`, i.e. the current schema minus the tag field."""
        if not index.exists():
            return False

        existing = SearchIndex.from_existing(index.name, redis_client=self.redis)
        if _LLM_STRING_HASH_FIELD_NAME in existing.schema.field_names:
            return False

        existing.schema.add_field(_LLM_STRING_HASH_FIELD)
        return existing.schema.to_dict() == index.schema.to_dict()

    @classmethod
    def migrate_index(
        cls,
        redis_url: str = "redis://localhost:6379",
        name: Optional[str] = _DEFAULT_CACHE_NAME,
        prefix: Optional[str] = _DEFAULT_CACHE_PREFIX,
        redis_client: Optional[Redis] = None,
        backfill: bool = False,
    ) -> Dict[str, int]:
        """Upgrade a semantic cache index created by an earlier version.

        This method adds the `llm_string` tag field to the existing index and keeps
        its keys. By default the entries themselves are not touched, so the cache
        starts cold: entries cached before the upgrade are no longer returned, but
        still expire via TTL and are removed by `clear()`.

        With `backfill=True`, each of those entries is also moved to the key the
        current version would write it to, which keeps the cache warm.

        Args:
            redis_url (str): The URL of the Redis instance to connect to.
            name (Optional[str]): Name of the cache index, as passed to
                `RedisSemanticCache`.
            prefix (Optional[str]): Prefix for the keys, as passed to
                `RedisSemanticCache`.
            redis_client (Optional[Redis]): An existing Redis client instance.

                If provided, `redis_url` is ignored.
            backfill (bool): Whether to also migrate the existing entries.

                This scans the whole cache.

        Returns:
            The number of entries that were `'backfilled'`, and of entries `'skipped'`
                because their stored prompt or `llm_string` could not be read.

                Both are `0` unless `backfill` is `True`.

        Raises:
            ValueError: If the index is not a semantic cache index.
            RedisSearchError: If the index does not exist.

        Example:
            ```python
            RedisSemanticCache.migrate_index(
                redis_url="redis://localhost:6379",
                name="my_semantic_cache",
                backfill=True,
            )
            ```

        Note:
            - Stop every application using the cache first: versions before and after
                the upgrade cannot share a migrated index.
            - It is safe to run this method again, including after an interruption or
                to backfill an index that was already upgraded.
            - Backfilling keeps all the fields of an entry, except for the new tag and
                `entry_id`, and its original expiration deadline.
            - An entry is only removed after its replacement is written. If the
                replacement already exists, it is kept and the entry is removed.
            - Skipped entries are left in place.
            - Entries that an earlier version already overwrote cannot be recovered.
        """
        client = cls._get_redis_client(redis_url, redis_client)
        cache_name = cls._get_cache_name(name, prefix)
        index = SearchIndex.from_existing(cache_name, redis_client=client)

        field_names = set(index.schema.field_names)
        if field_names - {_LLM_STRING_HASH_FIELD_NAME} != _LEGACY_INDEX_FIELD_NAMES:
            raise ValueError(f"Index '{cache_name}' is not a semantic cache index.")

        # Legacy entries don't have the field, so skip the scan that would make
        # Redis re-index all of them, vectors included.
        if _LLM_STRING_HASH_FIELD_NAME not in field_names:
            client.execute_command(
                "FT.ALTER",
                cache_name,
                "SKIPINITIALSCAN",
                "SCHEMA",
                "ADD",
                _LLM_STRING_HASH_FIELD_NAME,
                "TAG",
            )

        if not backfill:
            return {"backfilled": 0, "skipped": 0}
        return cls._backfill_entries(index, client)

    @classmethod
    def _backfill_entries(cls, index: SearchIndex, client: Redis) -> Dict[str, int]:
        """Move the entries without the `llm_string` tag to their current keys."""
        result = {"backfilled": 0, "skipped": 0}
        for key in client.scan_iter(match=index.key("*"), _type="HASH", count=100):
            # prompt_vector is binary, which a decode_responses client can't read.
            raw_fields = client.execute_command("HGETALL", key, **{NEVER_DECODE: True})
            fields = {str_if_bytes(field): val for field, val in raw_fields.items()}

            # Another cache whose name extends this one shares the key pattern.
            entry_id = fields.get("entry_id", b"").decode(errors="replace")
            if str_if_bytes(key) != index.key(entry_id):
                continue
            if _LLM_STRING_HASH_FIELD_NAME in fields:
                continue

            parsed = cls._parse_legacy_entry(fields)
            if parsed is None:
                result["skipped"] += 1
                continue

            expire_at = client.pexpiretime(key)
            if expire_at == -2:  # Expired since it was read
                continue

            prompt, llm_string = parsed
            filters = {_LLM_STRING_HASH_FIELD_NAME: _hash_llm_string(llm_string)}
            new_entry_id = hashify(prompt, filters)
            new_key = index.key(new_entry_id)

            # A replacement is either complete or absent, thanks to the transaction:
            # one that exists comes from an interrupted run or a newer write.
            if not client.exists(new_key):
                new_fields: Dict[Any, Any] = {
                    **fields,
                    **filters,
                    "entry_id": new_entry_id,
                }
                pipe = client.pipeline(transaction=True)
                pipe.hset(new_key, mapping=new_fields)
                if expire_at != -1:  # -1 means the entry never expires
                    pipe.pexpireat(new_key, expire_at)
                pipe.execute()
            client.delete(key)
            result["backfilled"] += 1

        if result["skipped"]:
            logger.warning(
                "Skipped %d entries of semantic cache index '%s' without a readable "
                "prompt or llm_string. They are no longer returned; call clear() to "
                "remove them.",
                result["skipped"],
                index.name,
            )
        return result

    @staticmethod
    def _parse_legacy_entry(fields: Dict[str, bytes]) -> Optional[Tuple[str, str]]:
        """Read the prompt and `llm_string` of an entry, or `None` if unreadable."""
        try:
            prompt = fields["prompt"].decode()
            llm_string = json.loads(fields["metadata"])["llm_string"]
        except (KeyError, TypeError, ValueError):
            return None
        return (prompt, llm_string) if isinstance(llm_string, str) else None

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up the result of a previous language model call in the Redis semantic
        cache.

        This method checks if there's a cached result for a semantically similar prompt
        and the same language model combination.

        Args:
            prompt (str): The input prompt for which to look up the cached result.
            llm_string (str): A string representation of the language model
                and its parameters.

        Returns:
            The cached result if a semantically similar prompt is found, or `None` if no
                suitable match is present in the cache.

                The result is typically a list containing a single `Generation` object.

        Example:
            ```python
            from langchain_openai import OpenAIEmbeddings
            cache = RedisSemanticCache(
                embeddings=OpenAIEmbeddings(),
                redis_url="redis://localhost:6379"
            )
            prompt = "What's the capital city of France?"
            llm_string = "openai/gpt-3.5-turbo"

            result = cache.lookup(prompt, llm_string)
            if result:
                print("Semantic cache hit:", result[0].text)
            else:
                print("Semantic cache miss")
            ```

        Note:
            - This method uses vector similarity search to find semantically
                similar prompts.
            - The prompt is embedded using the provided embedding function.
            - The method checks for cached results within the distance
                threshold specified during cache initialization.
            - If multiple results are within the threshold, the most similar
                one is returned.
            - Only entries stored for the same `llm_string` are considered, so
                a closer match cached for a different language model never hides
                a valid one.
            - This method is typically called internally by LangChain, but can
                be used directly for manual cache interactions.
            - Unlike exact matching, this may return results for prompts that
                are semantically similar but not identical to the input.
        """
        vector = self.cache._vectorize_prompt(prompt)
        results = self.cache.check(
            vector=vector,
            filter_expression=(
                Tag(_LLM_STRING_HASH_FIELD_NAME) == _hash_llm_string(llm_string)
            ),
        )

        if results:
            for result in results:
                if result.get("metadata", {}).get("llm_string") == llm_string:
                    try:
                        return [
                            loads(gen_str, allowed_objects="core")
                            for gen_str in json.loads(result.get("response"))
                        ]
                    except (json.JSONDecodeError, TypeError):
                        return None
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update the semantic cache with a new result for a given prompt and
        language model.

        This method stores a new result in the Redis semantic cache for the
        specified prompt and language model combination, using vector embedding
        for semantic similarity.

        Args:
            prompt (str): The input prompt associated with the result.
            llm_string (str): A string representation of the language model
                and its parameters.
            return_val (RETURN_VAL_TYPE): The result to be cached, typically a list
                containing a single `Generation` object.

        Example:
            ```python
            from langchain_core.outputs import Generation
            from langchain_openai import OpenAIEmbeddings

            cache = RedisSemanticCache(
                embeddings=OpenAIEmbeddings(),
                redis_url="redis://localhost:6379"
            )
            prompt = "What is the capital of France?"
            llm_string = "openai/gpt-3.5-turbo"
            result = [Generation(text="The capital of France is Paris.")]

            cache.update(prompt, llm_string, result)
            ```

        Note:
            - The method uses the provided embedding function to convert the prompt
                into a vector.
            - The vector, along with the `prompt`, `llm_string`, and result, is stored
                in the Redis cache.
            - If a TTL (Time To Live) was specified when initializing the cache, it will
                be applied to this entry.
            - This method is typically called internally by LangChain after a language
                model generates a response, but it can be used directly for manual
                cache updates.
            - Unlike exact matching caches, this allows for semantic similarity
                lookups later.
            - Entries are keyed by `prompt` and `llm_string`: updating the same
                pair overwrites its entry, while the same prompt for a different
                `llm_string` is stored as a separate entry.
            - If the cache already contains very similar entries, this will add a
                new entry rather than overwriting.
            - The effectiveness of the cache depends on the quality of the embedding
                function used.
        """
        serialized_response = json.dumps([dumps(gen) for gen in return_val])
        vector = self.cache._vectorize_prompt(prompt)

        self.cache.store(
            prompt=prompt,
            response=serialized_response,
            vector=vector,
            metadata={"llm_string": llm_string},
            filters={_LLM_STRING_HASH_FIELD_NAME: _hash_llm_string(llm_string)},
        )

    def clear(self, **kwargs: Any) -> None:
        """Clear all entries in the Redis semantic cache.

        This method removes all cache entries from the semantic cache.

        Args:
            **kwargs: Additional keyword arguments.

                Currently not used, but included for potential future extensions.

        Example:
            ```python
            from langchain_core.outputs import Generation
            from langchain_openai import OpenAIEmbeddings

            cache = RedisSemanticCache(
                embeddings=OpenAIEmbeddings(),
                redis_url="redis://localhost:6379",
                name="my_semantic_cache"
            )

            # Add some entries to the cache
            cache.update(
                "What is the capital of France?",
                "llm1",
                [Generation(text="Paris")]
            )
            cache.update(
                "Who wrote Romeo and Juliet?",
                "llm2",
                [Generation(text="Shakespeare")]
            )

            # Clear all entries
            cache.clear()

            # After this, all entries in the semantic cache will be removed
            ```

        Note:
            - This method clears all entries in the semantic cache, regardless of their
                content or similarity.
            - It uses the underlying cache implementation's clear method, which
                efficiently removes all entries.
            - This operation is irreversible. Make sure you want to clear all cached
                data before calling this method.
            - After clearing, the cache will be empty, but the index structure is
                maintained and ready for new entries.
            - This method is useful for resetting the cache or clearing out old data,
                especially if the nature of the queries or the embedding model has
                changed significantly.
        """
        self.cache.clear()

    def _key(self, prompt: str, llm_string: str) -> str:
        """Create a key for the cache."""

        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        llm_string_hash = hashlib.md5(llm_string.encode()).hexdigest()
        return f"{self.prefix}:{prompt_hash}:{llm_string_hash}"

    def name(self) -> str:
        """Get the name of the semantic cache index.

        This method returns the name of the index used for the semantic cache in Redis.

        Returns:
            str: The name of the semantic cache index.

        Example:
            ```python
            from langchain_openai import OpenAIEmbeddings

            cache = RedisSemanticCache(
                embeddings=OpenAIEmbeddings(),
                redis_url="redis://localhost:6379",
                name="my_custom_cache"
            )

            index_name = cache.name()
            print(f"The semantic cache is using index: {index_name}")
            ```

        Note:
            - The index name is set during the initialization of the
                `RedisSemanticCache`.
            - This is the `name` passed at initialization (default
                `'llmcache'`), unless a non-default `prefix` was also given;
                see the `prefix` argument.
            - This name is used internally to identify and manage the semantic cache
                in Redis.
            - Knowing the index name can be useful for debugging or for direct
                interactions with the Redis database outside of this cache interface.
        """
        return self.cache.index.name

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Async look up based on `prompt` and `llm_string`.

        Async version of `lookup`: checks for a cached result for a semantically
        similar prompt and the same `llm_string`.

        Args:
            prompt: String representation of the prompt.

                In the case of a Chat model, the prompt is a non-trivial
                serialization of the prompt into the language model.
            llm_string: String representation of the LLM configuration.

                This is used to capture the invocation parameters of the LLM
                (e.g., model name, temperature, stop tokens, max tokens, etc.).
                These invocation parameters are serialized into a string
                representation.

        Returns:
            On a cache miss, return `None`. On a cache hit, return the cached value.
                The cached value is a list of `Generation` objects (or subclasses).
        """
        vector = await self.cache._avectorize_prompt(prompt)
        results = await self.cache.acheck(
            vector=vector,
            filter_expression=(
                Tag(_LLM_STRING_HASH_FIELD_NAME) == _hash_llm_string(llm_string)
            ),
        )

        if results:
            for result in results:
                if result.get("metadata", {}).get("llm_string") == llm_string:
                    try:
                        return [
                            loads(gen_str, allowed_objects="core")
                            for gen_str in json.loads(result.get("response"))
                        ]
                    except (json.JSONDecodeError, TypeError):
                        return None
        return None

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        """Async update cache based on `prompt` and `llm_string`.

        Async version of `update`: embeds the `prompt` and stores it with the
        result, using `llm_string` as metadata.

        Args:
            prompt: String representation of the prompt.

                In the case of a Chat model, the prompt is a non-trivial
                serialization of the prompt into the language model.
            llm_string: String representation of the LLM configuration.

                This is used to capture the invocation parameters of the LLM
                (e.g., model name, temperature, stop tokens, max tokens, etc.).
                These invocation parameters are serialized into a string
                representation.
            return_val: The value to be cached.

                The value is a list of `Generation` objects (or subclasses).
        """
        serialized_response = json.dumps([dumps(gen) for gen in return_val])
        vector = await self.cache._avectorize_prompt(prompt)

        await self.cache.astore(
            prompt=prompt,
            response=serialized_response,
            vector=vector,
            metadata={"llm_string": llm_string},
            filters={_LLM_STRING_HASH_FIELD_NAME: _hash_llm_string(llm_string)},
        )

    async def aclear(self, **kwargs: Any) -> None:
        """Async clear cache that can take additional keyword arguments."""
        await self.cache.aclear()


class LangCacheSemanticCache(BaseCache):
    """Managed LangCache-backed semantic cache.

    This uses `redisvl.extensions.cache.llm.LangCacheSemanticCache` (a wrapper over the
    managed LangCache API). The optional dependency `langcache` must be installed at
    runtime when this class is used.

    Install with either `pip install 'langchain-redis[langcache]'` or
    `pip install langcache`.

    Args:
        distance_threshold (float): Maximum distance for semantic matches.
        ttl (Optional[int]): Cache TTL in seconds.

            If `None`, entries do not expire.
        name (Optional[str]): Cache name used by LangCache.

            Defaults to `'llmcache'`.
        server_url (Optional[str]): LangCache API endpoint.

            If not set, uses the `LANGCACHE_SERVER_URL` environment variable,
            read when `langchain_redis.cache` is imported, or a default managed
            endpoint if the variable is unset. Set the variable before importing
            `langchain_redis`, or pass `server_url` explicitly. Prefer the server
            URL provided for your cache.
        api_key (Optional[str]): Required API key for LangCache authentication.
        cache_id (Optional[str]): Required LangCache instance identifier.
        use_exact_search (bool): Enable exact match search.
        use_semantic_search (bool): Enable semantic search.
        distance_scale (Literal["normalized", "redis"]): Distance scaling mode.
        **kwargs: Additional options forwarded to the LangCache wrapper.

    Example:
        ```python
        from langchain_redis import LangCacheSemanticCache

        cache = LangCacheSemanticCache(
            cache_id="your-cache-id",
            api_key="your-api-key",
            name="mycache",
            ttl=3600,
        )
        ```

    Note:
        - Embeddings are computed server-side in LangCache; client-side embeddings
            are not used.
        - Per-entry TTL is ignored; cache-level TTL applies if set.

    Raises:
        ValueError: If `cache_id` or `api_key` is not provided.
        ImportError: If the `langcache` package is not installed.
    """

    # Hint for type checkers/IDEs; defined at class scope
    cache: TC_LangCacheSemanticCache

    def __init__(
        self,
        distance_threshold: float = 0.2,
        ttl: Optional[int] = None,
        name: Optional[str] = _DEFAULT_CACHE_NAME,
        *,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_id: Optional[str] = None,
        use_exact_search: bool = True,
        use_semantic_search: bool = True,
        distance_scale: Literal["normalized", "redis"] = "normalized",
        **kwargs: Any,
    ):
        if not cache_id:
            raise ValueError("cache_id is required for LangCacheSemanticCache")
        if not api_key:
            raise ValueError("api_key is required for LangCacheSemanticCache")

        self._cache_name = name or _DEFAULT_CACHE_NAME
        self.ttl = ttl
        self._distance_threshold = distance_threshold

        try:
            from redisvl.extensions.cache.llm import (
                LangCacheSemanticCache as RVLLangCacheSemanticCache,
            )
        except ImportError as e:
            # Check if this is a missing langcache dependency or outdated redisvl
            error_msg = str(e).lower()
            if "langcache" in error_msg:
                raise ImportError(
                    "LangCacheSemanticCache requires the langcache package. "
                    "Install it with: pip install langcache "
                    "or pip install 'langchain-redis[langcache]'"
                ) from e
            else:
                raise ImportError(
                    "LangCacheSemanticCache requires redisvl>=0.11.0. "
                    "Update redisvl with: pip install --upgrade redisvl"
                ) from e

        # Instantiate the LangCache wrapper; it will validate cache_id/api_key
        self.cache: Any = RVLLangCacheSemanticCache(
            name=self._cache_name,
            server_url=server_url or _DEFAULT_LANGCACHE_SERVER_URL,
            cache_id=cache_id,
            api_key=api_key,
            ttl=ttl,
            use_exact_search=use_exact_search,
            use_semantic_search=use_semantic_search,
            distance_scale=distance_scale,
            **kwargs,
        )

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up using LangCache's check API."""
        results = self.cache.check(
            prompt=prompt,
            num_results=1,
            distance_threshold=self._distance_threshold,
            attributes={"llm_string": quote(llm_string)},
        )
        return self._process_lookup_results(results)

    def _process_lookup_results(
        self, results: List[Dict[str, Any]]
    ) -> Optional[RETURN_VAL_TYPE]:
        # The underlying LangCache service already filters by the (encoded)
        # llm_string attribute, and we always request at most one result.
        # If we have a hit, we can trust that it corresponds to the
        # requested llm_string and simply deserialize the response payload.
        if not results:
            return None

        first = results[0]
        try:
            return [
                loads(s, allowed_objects="core")
                for s in json.loads(first.get("response", "[]"))
            ]
        except (json.JSONDecodeError, TypeError):
            return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Store using LangCache's store API via redisvl wrapper."""
        serialized_response = json.dumps([dumps(gen) for gen in return_val])
        # LangCacheSemanticCache ignores per-entry TTL; it uses cache-level TTL if set
        self.cache.store(
            prompt=prompt,
            response=serialized_response,
            metadata={"llm_string": quote(llm_string)},
            ttl=self.ttl,
        )

    def clear(self, **kwargs: Any) -> None:
        """Clear all entries via the wrapper's clear API."""
        self.cache.clear()

    def name(self) -> str:
        return self._cache_name

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Async look up through LangCache's acheck API."""
        results = await self.cache.acheck(
            prompt=prompt,
            num_results=1,
            distance_threshold=self._distance_threshold,
            attributes={"llm_string": quote(llm_string)},
        )
        return self._process_lookup_results(results)

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        """Async store using LangCache's astore API via redisvl wrapper."""
        serialized_response = json.dumps([dumps(gen) for gen in return_val])
        await self.cache.astore(
            prompt=prompt,
            response=serialized_response,
            metadata={"llm_string": quote(llm_string)},
            ttl=self.ttl,
        )

    async def aclear(self, **kwargs: Any) -> None:
        """Async clear via the wrapper's aclear API."""
        await self.cache.aclear()
