"""Redis cache implementation for LangChain."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, List, Optional, Union

import numpy as np
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from pydantic.v1 import Field as FieldV1
from redis import Redis
from redis.commands.json.path import Path
from redis.exceptions import ResponseError
from redisvl.extensions.llmcache import (  # type: ignore[import]
    SemanticCache as RedisVLSemanticCache,
)
from redisvl.schema.fields import VectorDataType  # type: ignore[import]
from redisvl.utils.vectorize import BaseVectorizer  # type: ignore[import]

from langchain_redis.version import __full_lib_name__


class EmbeddingsVectorizer(BaseVectorizer):
    # BaseVectorizer is a pydantic.v1.BaseModel so we need to use pydantic.v1.Field.
    embeddings: Embeddings = FieldV1(...)
    model: str = FieldV1(default="custom_embeddings")

    class Config:
        arbitrary_types_allowed = True

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
                             If None, entries don't expire.
        prefix (Optional[str]): Prefix for all keys stored in Redis.

    Args:
        redis_url (str): The URL of the Redis instance to connect to.
                         Defaults to "redis://localhost:6379".
        ttl (Optional[int]): Time-to-live for cache entries in seconds.
                             Defaults to None (no expiration).
        prefix (Optional[str]): Prefix for all keys stored in Redis.
                                Defaults to "redis".
        redis (Optional[Redis]): An existing Redis client instance.
                                 If provided, redis_url is ignored.

    Example:
        .. code-block:: python

            from langchain_redis import RedisCache
            from langchain_core.globals import set_llm_cache

            # Create a Redis cache instance
            redis_cache = RedisCache(redis_url="redis://localhost:6379", ttl=3600)

            # Set it as the global LLM cache
            set_llm_cache(redis_cache)

            # Now, when you use an LLM, it will automatically use this cache

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
        self.redis = redis_client or Redis.from_url(redis_url)
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
            Optional[RETURN_VAL_TYPE]: The cached result if found, or None if not
                                       present in the cache.
            The result is typically a list containing a single Generation object.

        Example:
            .. code-block:: python

                cache = RedisCache(redis_url="redis://localhost:6379")
                prompt = "What is the capital of France?"
                llm_string = "openai/gpt-3.5-turbo"

                result = cache.lookup(prompt, llm_string)
                if result:
                    print("Cache hit:", result[0].text)
                else:
                    print("Cache miss")

        Note:
            - The method uses an MD5 hash of the prompt and llm_string to create
              the cache key.
            - The cached value is stored as JSON and parsed back into a
              Generation object.
            - If the key exists but the value is None or cannot be parsed,
              None is returned.
            - This method is typically called internally by LangChain, but can be used
              directly for manual cache interactions.
        """
        key = self._key(prompt, llm_string)
        result = self.redis.json().get(key)
        if result:
            return [loads(json.dumps(result))]
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update the cache with a new result for a given prompt and language model.

        This method stores a new result in the Redis cache for the specified prompt and
        language model combination.

        Args:
            prompt (str): The input prompt associated with the result.
            llm_string (str): A string representation of the language model
                              and its parameters.
            return_val (RETURN_VAL_TYPE): The result to be cached, typically a list
                                          containing a single Generation object.

        Returns:
            None

        Example:
            .. code-block:: python

                from langchain_core.outputs import Generation

                cache = RedisCache(redis_url="redis://localhost:6379", ttl=3600)
                prompt = "What is the capital of France?"
                llm_string = "openai/gpt-3.5-turbo"
                result = [Generation(text="The capital of France is Paris.")]

                cache.update(prompt, llm_string, result)

        Note:
            - The method uses an MD5 hash of the prompt and llm_string to create the
              cache key.
            - The result is stored as JSON in Redis.
            - If a TTL (Time To Live) was specified when initializing the cache,
              it will be applied to this entry.
            - This method is typically called internally by LangChain after a language
              model generates a response, but it can be used directly
              for manual cache updates.
            - If the cache already contains an entry for this prompt and llm_string,
              it will be overwritten.
        """
        key = self._key(prompt, llm_string)
        json_value = json.loads(dumps(return_val[0]))
        self.redis.json().set(key, Path.root_path(), json_value)
        if self.ttl is not None:
            self.redis.expire(key, self.ttl)

    def clear(self, **kwargs: Any) -> None:
        """Clear all entries in the Redis cache that match the cache prefix.

        This method removes all cache entries that start with the specified prefix.

        Args:
            **kwargs: Additional keyword arguments. Currently not used, but included
                    for potential future extensions.

        Returns:
            None

        Example:
            .. code-block:: python

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

        Note:
            - This method uses Redis SCAN to iterate over keys, which is safe
              for large datasets.
            - It deletes keys in batches of 100 to optimize performance.
            - Only keys that start with the specified prefix (default is "redis:")
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
                         Defaults to "redis://localhost:6379".
        distance_threshold (float): The maximum distance for considering a cache hit.
                                    Defaults to 0.2.
        ttl (Optional[int]): Time-to-live for cache entries in seconds.
                             Defaults to None (no expiration).
        name (Optional[str]): Name for the cache index. Defaults to "llmcache".
        prefix (Optional[str]): Prefix for all keys stored in Redis.
                                Defaults to "llmcache".
        redis (Optional[Redis]): An existing Redis client instance.
                                 If provided, redis_url is ignored.

    Example:
        .. code-block:: python

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

    Note:
        - This cache uses vector similarity search to find semantically similar prompts.
        - The distance_threshold determines how similar a prompt must be to trigger
          a cache hit.
        - Lowering the distance_threshold increases precision but may reduce cache hits.
        - The cache uses the RedisVL library for efficient vector storage and retrieval.
        - Semantic caching can be more flexible than exact matching, allowing cache hits
          for prompts that are semantically similar but not identical.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        redis_url: str = "redis://localhost:6379",
        distance_threshold: float = 0.2,
        ttl: Optional[int] = None,
        name: Optional[str] = "llmcache",
        prefix: Optional[str] = "llmcache",
        redis_client: Optional[Redis] = None,
    ):
        self.redis = redis_client or Redis.from_url(redis_url)
        self.embeddings = embeddings
        self.prefix = prefix
        vectorizer = EmbeddingsVectorizer(embeddings=self.embeddings)

        self.cache = RedisVLSemanticCache(
            vectorizer=vectorizer,
            redis_client=self.redis,
            distance_threshold=distance_threshold,
            ttl=ttl,
            name=name,
            prefix=prefix,
        )

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up the result of a previous language model call in the
           Redis semantic cache.

        This method checks if there's a cached result for a semantically similar prompt
        and the same language model combination.

        Args:
            prompt (str): The input prompt for which to look up the cached result.
            llm_string (str): A string representation of the language model
                              and its parameters.

        Returns:
            Optional[RETURN_VAL_TYPE]: The cached result if a semantically similar
                                       prompt is found,
            or None if no suitable match is present in the cache. The result
            is typically a list containing a single Generation object.

        Example:
            .. code-block:: python

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

        Note:
            - This method uses vector similarity search to find semantically
              similar prompts.
            - The prompt is embedded using the provided embedding function.
            - The method checks for cached results within the distance
              threshold specified during cache initialization.
            - If multiple results are within the threshold, the most similar
              one is returned.
            - The llm_string is used to ensure the cached result is from the
              same language model.
            - This method is typically called internally by LangChain, but can
              be used directly for manual cache interactions.
            - Unlike exact matching, this may return results for prompts that
              are semantically similar but not identical to the input.
        """
        vector = self.cache._vectorize_prompt(prompt)
        results = self.cache.check(vector=vector)

        if results:
            for result in results:
                if result.get("metadata", {}).get("llm_string") == llm_string:
                    try:
                        return [
                            loads(gen_str)
                            for gen_str in json.loads(result.get("response"))
                        ]
                    except (json.JSONDecodeError, TypeError):
                        return None
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update the semantic cache with a new result for a given prompt
           and language model.

        This method stores a new result in the Redis semantic cache for the
        specified prompt and language model combination, using vector embedding
        for semantic similarity.

        Args:
            prompt (str): The input prompt associated with the result.
            llm_string (str): A string representation of the language model
                              and its parameters.
            return_val (RETURN_VAL_TYPE): The result to be cached, typically a list
                                          containing a single Generation object.

        Returns:
            None

        Example:
            .. code-block:: python

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

        Note:
            - The method uses the provided embedding function to convert the prompt
              into a vector.
            - The vector, along with the prompt, llm_string, and result, is stored in
              the Redis cache.
            - If a TTL (Time To Live) was specified when initializing the cache, it will
              be applied to this entry.
            - This method is typically called internally by LangChain after a language
              model generates a response, but it can be used directly for manual
              cache updates.
            - Unlike exact matching caches, this allows for semantic similarity
              lookups later.
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
        )

    def clear(self, **kwargs: Any) -> None:
        """Clear all entries in the Redis semantic cache.

        This method removes all cache entries from the semantic cache.

        Args:
            **kwargs: Additional keyword arguments. Currently not used, but included
                    for potential future extensions.

        Returns:
            None

        Example:
            .. code-block:: python

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
            .. code-block:: python

                from langchain_openai import OpenAIEmbeddings

                cache = RedisSemanticCache(
                    embeddings=OpenAIEmbeddings(),
                    redis_url="redis://localhost:6379",
                    name="my_custom_cache"
                )

                index_name = cache.name()
                print(f"The semantic cache is using index: {index_name}")

        Note:
            - The index name is set during the initialization of the RedisSemanticCache.
            - If no custom name was provided during initialization,
              a default name is used.
            - This name is used internally to identify and manage the semantic cache
              in Redis.
            - Knowing the index name can be useful for debugging or for direct
              interactions with the Redis database outside of this cache interface.
        """
        return self.cache.index.name

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Async look up based on prompt and llm_string.

        A cache implementation is expected to generate a key from the 2-tuple
        of prompt and llm_string (e.g., by concatenating them with a delimiter).

        Args:
            prompt: a string representation of the prompt.
                In the case of a Chat model, the prompt is a non-trivial
                serialization of the prompt into the language model.
            llm_string: A string representation of the LLM configuration.
                This is used to capture the invocation parameters of the LLM
                (e.g., model name, temperature, stop tokens, max tokens, etc.).
                These invocation parameters are serialized into a string
                representation.

        Returns:
            On a cache miss, return None. On a cache hit, return the cached value.
            The cached value is a list of Generations (or subclasses).
        """
        vector = await self.cache._avectorize_prompt(prompt)
        results = await self.cache.acheck(vector=vector)

        if results:
            for result in results:
                if result.get("metadata", {}).get("llm_string") == llm_string:
                    try:
                        return [
                            loads(gen_str)
                            for gen_str in json.loads(result.get("response"))
                        ]
                    except (json.JSONDecodeError, TypeError):
                        return None
        return None

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        """Async update cache based on prompt and llm_string.

        The prompt and llm_string are used to generate a key for the cache.
        The key should match that of the look up method.

        Args:
            prompt: a string representation of the prompt.
                In the case of a Chat model, the prompt is a non-trivial
                serialization of the prompt into the language model.
            llm_string: A string representation of the LLM configuration.
                This is used to capture the invocation parameters of the LLM
                (e.g., model name, temperature, stop tokens, max tokens, etc.).
                These invocation parameters are serialized into a string
                representation.
            return_val: The value to be cached. The value is a list of Generations
                (or subclasses).
        """
        serialized_response = json.dumps([dumps(gen) for gen in return_val])
        vector = self.cache._vectorize_prompt(prompt)

        await self.cache.astore(
            prompt=prompt,
            response=serialized_response,
            vector=vector,
            metadata={"llm_string": llm_string},
        )

    async def aclear(self, **kwargs: Any) -> None:
        """Async clear cache that can take additional keyword arguments."""
        await self.cache.aclear()
