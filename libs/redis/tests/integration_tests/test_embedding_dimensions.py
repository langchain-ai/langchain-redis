"""Integration tests for embeddings that don't fit the index's vector field."""

from typing import Any, Dict, Generator
from uuid import uuid4

import pytest
from redis import Redis

from langchain_redis import RedisVectorStore
from tests.integration_tests.embed_patch import FixedEmbeddings

DIMS = 4


@pytest.fixture
def index_name(redis_url: str) -> Generator[str, None, None]:
    name = f"embedding_dims_{uuid4().hex[:8]}"
    yield name
    client = Redis.from_url(redis_url)
    try:
        client.ft(name).dropindex(delete_documents=True)
    finally:
        client.close()


@pytest.fixture(params=["hash", "json"])
def config(
    request: pytest.FixtureRequest, redis_url: str, index_name: str
) -> Dict[str, Any]:
    return {
        "redis_url": redis_url,
        "index_name": index_name,
        "storage_type": request.param,
    }


def _stored_keys(redis_url: str, index_name: str) -> int:
    client = Redis.from_url(redis_url)
    try:
        return len(list(client.scan_iter(f"{index_name}:*")))
    finally:
        client.close()


def test_other_model_is_rejected_and_nothing_is_written(
    config: Dict[str, Any], redis_url: str, index_name: str
) -> None:
    store = RedisVectorStore(FixedEmbeddings(dimensions=DIMS), **config)
    store.add_texts(["alpha", "beta"])
    other_model = RedisVectorStore(FixedEmbeddings(dimensions=DIMS * 2), **config)

    with pytest.raises(ValueError, match="Embedding 0 has 8 dimensions"):
        other_model.add_texts(["gamma", "delta"])

    info = store.index.info()
    assert int(info["num_docs"]) == 2
    assert int(info["hash_indexing_failures"]) == 0
    assert _stored_keys(redis_url, index_name) == 2


def test_new_index_with_wrong_embedding_dimensions_setting(
    config: Dict[str, Any], redis_url: str, index_name: str
) -> None:
    """`embedding_dimensions` that disagrees with the model is caught on a new
    index too, before the first document is written."""
    store = RedisVectorStore(
        FixedEmbeddings(dimensions=DIMS * 2), embedding_dimensions=DIMS, **config
    )

    with pytest.raises(ValueError, match="8 dimensions.*stores 4-dimensional"):
        store.add_texts(["alpha"])

    assert _stored_keys(redis_url, index_name) == 0
