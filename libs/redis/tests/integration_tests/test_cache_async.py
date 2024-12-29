import pytest
from langchain_core.outputs import Generation
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_redis import RedisSemanticCache


@pytest.mark.asyncio
async def test_async_semantic_cache_update_and_lookup(redis_url: str) -> None:
    # Use OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    cache = RedisSemanticCache(
        embeddings=embeddings,
        redis_url=redis_url,
    )

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
    # Use OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    cache = RedisSemanticCache(
        embeddings=embeddings,
        redis_url=redis_url,
    )
    await cache.aupdate("test_prompt", "test_llm", [Generation(text="test_response")])
    result = await cache.alookup("test_prompt", "test_llm")
    assert result is not None
    assert len(result) == 1
    assert result[0].text == "test_response"
    await cache.aclear()
    assert await cache.alookup("test_prompt", "test_llm") is None
