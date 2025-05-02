import os

import pytest
from langchain_core.outputs import Generation

from langchain_redis import RedisSemanticCache
from tests.integration_tests.embed_patch import get_embeddings_for_tests


@pytest.mark.asyncio
async def test_async_semantic_cache_update_and_lookup(redis_url: str) -> None:
    # Use a unique name for this test
    cache_name = f"test_cache_{os.urandom(8).hex()}"
    # Use patched embeddings
    embeddings = get_embeddings_for_tests()
    cache = RedisSemanticCache(
        embeddings=embeddings,
        redis_url=redis_url,
        name=cache_name,
    )

    # Make sure to clear before starting
    await cache.aclear()

    # Perform async update
    prompt = "What is the capital of France?"
    response = "Paris"
    llm_string = "test_llm"
    await cache.aupdate(prompt, llm_string, [Generation(text=response)])

    result = await cache.alookup(prompt, llm_string)
    assert result is not None
    assert len(result) == 1
    assert result[0].text == "Paris"


@pytest.mark.asyncio
async def test_redis_cache(redis_url: str) -> None:
    # Use a unique name for this test
    cache_name = f"test_cache_{os.urandom(8).hex()}"
    # Use patched embeddings
    embeddings = get_embeddings_for_tests()
    cache = RedisSemanticCache(
        embeddings=embeddings,
        redis_url=redis_url,
        name=cache_name,
    )

    # Make sure to clear before starting
    await cache.aclear()

    # Now update and lookup
    await cache.aupdate("test_prompt", "test_llm", [Generation(text="test_response")])
    result = await cache.alookup("test_prompt", "test_llm")
    assert result is not None
    assert len(result) == 1
    assert result[0].text == "test_response"

    # Test the clear functionality
    await cache.aclear()
    assert await cache.alookup("test_prompt", "test_llm") is None
