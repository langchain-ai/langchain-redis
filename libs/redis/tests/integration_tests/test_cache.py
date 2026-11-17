import asyncio
from itertools import cycle
from typing import AsyncGenerator, Generator, List

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import FakeListLLM, GenericFakeChatModel
from langchain_core.load.dump import dumps
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs import ChatGeneration, Generation
from redis import Redis
from ulid import ULID

from langchain_redis import RedisCache, RedisSemanticCache

# Import our patch for OpenAI embeddings
from tests.integration_tests.embed_patch import get_embeddings_for_tests


def random_string() -> str:
    return str(ULID())


class DummyEmbeddings(Embeddings):
    def __init__(self, dims: int = 3):
        self.dims = dims

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * self.dims for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * self.dims


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(size=768)


@pytest.fixture
def openai_embeddings() -> Embeddings:
    """Get appropriately configured embeddings for testing."""
    return get_embeddings_for_tests()


@pytest.fixture(autouse=True)
def reset_llm_cache() -> Generator[None, None, None]:
    """Reset the global LLM cache that tests and fixtures set via set_llm_cache."""
    yield
    set_llm_cache(None)


@pytest.fixture
def redis_cache(redis_url: str) -> Generator[RedisCache, None, None]:
    cache = RedisCache(redis_url=redis_url, ttl=3600)  # Set TTL to 1 hour
    set_llm_cache(cache)
    try:
        yield cache
    finally:
        cache.clear()


@pytest.fixture
async def async_redis_cache(redis_url: str) -> AsyncGenerator[RedisCache, None]:
    cache = RedisCache(redis_url=redis_url, ttl=3600)  # Set TTL to 1 hour
    set_llm_cache(cache)
    try:
        yield cache
    finally:
        await cache.aclear()


@pytest.fixture(scope="function")
def redis_semantic_cache(
    openai_embeddings: Embeddings, redis_url: str
) -> Generator[RedisSemanticCache, None, None]:
    cache = RedisSemanticCache(
        name=f"semcache_{str(ULID())}",
        redis_url=redis_url,
        embeddings=openai_embeddings,
    )
    try:
        yield cache
    finally:
        cache.clear()


class TestRedisCacheBasicIntegration:
    def test_redis_cache(self, redis_cache: RedisCache) -> None:
        redis_cache.update(
            "test_prompt", "test_llm", [Generation(text="test_response")]
        )
        result = redis_cache.lookup("test_prompt", "test_llm")
        assert result is not None
        assert len(result) == 1
        assert result[0].text == "test_response"
        redis_cache.clear()
        assert redis_cache.lookup("test_prompt", "test_llm") is None

    def test_clear_leaves_another_cache_alone(self, redis_url: str) -> None:
        # A prefix carrying a glob metacharacter used to be handed to SCAN as a
        # pattern: `cache[1]` became a character class matching `cache1`, so
        # clear() emptied the other cache and left its own entries in place.
        mine = RedisCache(redis_url=redis_url, prefix=f"cache[1]{random_string()}")
        assert mine.prefix is not None
        theirs = RedisCache(redis_url=redis_url, prefix=mine.prefix.replace("[1]", "1"))

        my_key = mine._key("prompt", "llm")
        their_key = theirs._key("prompt", "llm")
        mine.redis.set(my_key, "mine")
        theirs.redis.set(their_key, "theirs")

        mine.clear()

        assert mine.redis.exists(my_key) == 0
        assert theirs.redis.exists(their_key) == 1
        theirs.clear()

    def test_clear_escapes_the_encoded_prefix(self, redis_url: str) -> None:
        # Redis matches the pattern against raw key bytes, so escaping the
        # prefix text and letting redis-py encode the result puts a NUL between
        # the backslash and the bracket under utf-16, and the escape is lost.
        client = Redis.from_url(redis_url, encoding="utf-16")
        cache = RedisCache(redis_client=client, prefix=f"cache[1]{random_string()}")

        key = cache._key("prompt", "llm")
        client.set(key, "mine")

        cache.clear()

        assert client.exists(key) == 0
        client.close()

    def test_redis_cache_ttl(self, redis_cache: RedisCache) -> None:
        llm = FakeListLLM(cache=redis_cache, responses=["foo", "bar"])
        prompt = random_string()
        redis_cache.update(prompt, str(llm), [Generation(text="test response")])

        # Check that the TTL is set
        key = redis_cache._key(prompt, str(llm))
        ttl_result = redis_cache.redis.ttl(key)
        if asyncio.iscoroutine(ttl_result):
            ttl = asyncio.get_event_loop().run_until_complete(ttl_result)
        else:
            ttl = ttl_result
        assert 0 < ttl <= 3600

    @pytest.mark.asyncio
    async def test_async_redis_cache(self, async_redis_cache: RedisCache) -> None:
        llm = FakeListLLM(cache=async_redis_cache, responses=["foo", "bar"])
        prompt = random_string()
        await async_redis_cache.aupdate(
            prompt, str(llm), [Generation(text="async test")]
        )

        result = await async_redis_cache.alookup(prompt, str(llm))
        assert result == [Generation(text="async test")]

    @pytest.mark.asyncio
    async def test_async_redis_cache_clear(self, async_redis_cache: RedisCache) -> None:
        llm = FakeListLLM(cache=async_redis_cache, responses=["foo", "bar"])
        prompt = random_string()
        await async_redis_cache.aupdate(
            prompt, str(llm), [Generation(text="async test")]
        )

        await async_redis_cache.aclear()

        result = await async_redis_cache.alookup(prompt, str(llm))
        assert result is None

    def test_redis_cache_chat(self, redis_cache: RedisCache) -> None:
        responses = cycle(
            [
                AIMessage(content="Hello from cache"),
                AIMessage(content="How are you from cache"),
            ]
        )
        chat_model = GenericFakeChatModel(messages=responses)

        human_message1 = HumanMessage(content="Hello")
        human_message2 = HumanMessage(content="How are you?")
        prompt1: List[BaseMessage] = [human_message1]
        prompt2: List[BaseMessage] = [human_message2]

        # First call should generate a response and cache it
        result1 = chat_model.generate([prompt1])

        # Instead of comparing the entire LLMResult, let's check specific parts
        assert len(result1.generations) == 1
        assert len(result1.generations[0]) == 1
        assert isinstance(result1.generations[0][0], ChatGeneration)
        assert result1.generations[0][0].message.content == "Hello from cache"

        # Cache the result manually (since GenericFakeChatModel doesn't use the cache
        # internally)
        redis_cache.update(dumps(prompt1), str(chat_model), result1.generations[0])

        # Second call with the same prompt should hit the cache
        cached_result = redis_cache.lookup(dumps(prompt1), str(chat_model))
        assert cached_result is not None
        assert isinstance(cached_result[0], ChatGeneration)
        assert cached_result[0].message.content == "Hello from cache"

        # Call with a different prompt should generate a new response
        result3 = chat_model.generate([prompt2])

        # Check specific parts of the new result
        assert len(result3.generations) == 1
        assert len(result3.generations[0]) == 1
        assert isinstance(result3.generations[0][0], ChatGeneration)
        assert result3.generations[0][0].message.content == "How are you from cache"

        # Cache the new result
        redis_cache.update(dumps(prompt2), str(chat_model), result3.generations[0])

        # Verify that both prompts are in the cache
        cached_result1 = redis_cache.lookup(dumps(prompt1), str(chat_model))
        cached_result2 = redis_cache.lookup(dumps(prompt2), str(chat_model))
        assert cached_result1 is not None and cached_result2 is not None
        assert isinstance(cached_result1[0], ChatGeneration)
        assert isinstance(cached_result2[0], ChatGeneration)
        assert cached_result1[0].message.content == "Hello from cache"
        assert cached_result2[0].message.content == "How are you from cache"


def test_redis_cache_with_preconfigured_client(redis_url: str) -> None:
    redis_client = Redis.from_url(redis_url)
    cache = RedisCache(redis_client=redis_client)

    cache.update("test_prompt", "test_llm", [Generation(text="test_response")])
    result = cache.lookup("test_prompt", "test_llm")

    assert result is not None
    assert len(result) == 1
    assert result[0].text == "test_response"

    cache.clear()


def test_redis_semantic_cache_with_preconfigured_client(
    openai_embeddings: Embeddings, redis_url: str
) -> None:
    redis_client = Redis.from_url(redis_url)
    cache = RedisSemanticCache(
        embeddings=openai_embeddings,
        redis_client=redis_client,
    )

    prompt = "What is the capital of France?"
    llm_string = "test_llm"
    cache.update(prompt, llm_string, [Generation(text="Paris")])

    result = cache.lookup(prompt, llm_string)
    assert result is not None
    assert len(result) == 1
    assert result[0].text == "Paris"

    cache.clear()


def test_set_llm(openai_embeddings: Embeddings, redis_url: str) -> None:
    # Initialize RedisSemanticCache with custom settings
    custom_semantic_cache = RedisSemanticCache(
        redis_url=redis_url,
        embeddings=openai_embeddings,
        distance_threshold=0.1,  # Stricter similarity threshold
        ttl=3600,  # 1 hour TTL
        name=f"custom_cache_{str(ULID())}",  # Custom cache name
    )

    # Use FakeListLLM instead of ChatOpenAI to avoid API issues
    responses = ["Jupiter is the largest planet in our solar system."]
    llm = FakeListLLM(responses=responses)

    # Test the custom semantic cache
    set_llm_cache(custom_semantic_cache)
    try:
        test_prompt = "What's the largest planet in our solar system?"
        result = llm.invoke(test_prompt)

        # Try a slightly different query that should hit the semantic cache
        similar_test_prompt = "Which planet is the biggest in the solar system?"
        similar_result = llm.invoke(similar_test_prompt)

        # With semantic similarity, both should return the same result
        assert result == similar_result
    finally:
        custom_semantic_cache.cache.delete()
