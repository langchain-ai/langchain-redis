"""Integration tests for embeddings that don't fit the index's vector field."""

from typing import Any, Dict, Generator, List
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings
from redis import Redis

from langchain_redis import RedisVectorStore

DIMS = 4


class SizedEmbeddings(Embeddings):
    """One-hot vectors of a given size, so every text has a distinct neighbor."""

    def __init__(self, size: int) -> None:
        self.size = size

    def _vector(self, text: str) -> List[float]:
        vector = [0.0] * self.size
        vector[sum(map(ord, text)) % self.size] = 1.0
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


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


@pytest.fixture
def store(config: Dict[str, Any]) -> RedisVectorStore:
    """A store whose index holds DIMS-dimensional vectors and two documents."""
    store = RedisVectorStore(SizedEmbeddings(DIMS), **config)
    store.add_texts(["alpha", "beta"])
    return store


def _stored_keys(redis_url: str, index_name: str) -> int:
    client = Redis.from_url(redis_url)
    try:
        return len(list(client.scan_iter(f"{index_name}:*")))
    finally:
        client.close()


def test_other_model_is_rejected_and_nothing_is_written(
    store: RedisVectorStore, config: Dict[str, Any], redis_url: str, index_name: str
) -> None:
    other_model = RedisVectorStore(SizedEmbeddings(DIMS * 2), **config)

    with pytest.raises(ValueError, match=r"Embedding 0 has 8 dimensions"):
        other_model.add_texts(["gamma", "delta"])

    info = store.index.info()
    assert int(info["num_docs"]) == 2
    assert int(info["hash_indexing_failures"]) == 0
    assert _stored_keys(redis_url, index_name) == 2


def test_writes_that_fit_the_index_still_work_after_a_rejection(
    store: RedisVectorStore, config: Dict[str, Any]
) -> None:
    other_model = RedisVectorStore(SizedEmbeddings(DIMS * 2), **config)
    with pytest.raises(ValueError):
        other_model.add_texts(["gamma"])

    store.add_texts(["delta"])

    found = {doc.page_content for doc in store.similarity_search("alpha", k=5)}
    assert found == {"alpha", "beta", "delta"}


def test_from_texts_rejects_another_model(
    store: RedisVectorStore, config: Dict[str, Any], redis_url: str, index_name: str
) -> None:
    with pytest.raises(ValueError, match="stores 4-dimensional vectors"):
        RedisVectorStore.from_texts(["gamma"], SizedEmbeddings(DIMS * 2), **config)

    assert int(store.index.info()["num_docs"]) == 2
    assert _stored_keys(redis_url, index_name) == 2


def test_new_index_with_wrong_embedding_dimensions_setting(
    config: Dict[str, Any], redis_url: str, index_name: str
) -> None:
    """`embedding_dimensions` that disagrees with the model is caught on a new
    index too, before the first document is written."""
    store = RedisVectorStore(
        SizedEmbeddings(DIMS * 2), embedding_dimensions=DIMS, **config
    )

    with pytest.raises(ValueError, match="8 dimensions.*stores 4-dimensional"):
        store.add_texts(["alpha"])

    assert _stored_keys(redis_url, index_name) == 0
