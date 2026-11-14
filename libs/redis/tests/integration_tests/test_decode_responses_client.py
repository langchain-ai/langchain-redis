from typing import Generator, List
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings
from redis import Redis

from langchain_redis import RedisConfig, RedisVectorStore

DIMS = 4
VECTOR_WHOSE_FLOAT32_BYTES_ARE_NOT_UTF8 = [0.1] * DIMS


class ConstantEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [VECTOR_WHOSE_FLOAT32_BYTES_ARE_NOT_UTF8 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return VECTOR_WHOSE_FLOAT32_BYTES_ARE_NOT_UTF8


@pytest.fixture
def decoding_client(redis_url: str) -> Generator[Redis, None, None]:
    client = Redis.from_url(redis_url, decode_responses=True)
    yield client
    client.close()


@pytest.fixture(params=["hash", "json"])
def store(
    request: pytest.FixtureRequest, decoding_client: Redis
) -> Generator[RedisVectorStore, None, None]:
    config = RedisConfig(
        index_name=f"decode_responses_{uuid4().hex[:8]}",
        redis_client=decoding_client,
        embedding_dimensions=DIMS,
        storage_type=request.param,
    )
    vector_store = RedisVectorStore(ConstantEmbeddings(), config=config)
    vector_store.add_texts(
        ["foo", "bar"],
        metadatas=[{"source": "first"}, {"source": "second"}],
        ids=["a", "b"],
    )
    yield vector_store
    vector_store.index.delete(drop=True)


@pytest.fixture
def store_of_another_index_on_the_same_prefix(
    store: RedisVectorStore, decoding_client: Redis
) -> Generator[RedisVectorStore, None, None]:
    config = RedisConfig(
        index_name=f"{store.config.index_name}_other",
        key_prefix=store.config.index_name,
        redis_client=decoding_client,
        embedding_dimensions=DIMS,
        storage_type=store.config.storage_type,
    )
    vector_store = RedisVectorStore(ConstantEmbeddings(), config=config)
    yield vector_store
    vector_store.index.delete(drop=False)


def _searched_contents(vector_store: RedisVectorStore) -> List[str]:
    return sorted(doc.page_content for doc in vector_store.similarity_search("foo"))


def test_get_by_ids_returns_the_stored_documents(store: RedisVectorStore) -> None:
    documents = store.get_by_ids(["a", "b"])

    assert {doc.id: (doc.page_content, doc.metadata) for doc in documents} == {
        "a": ("foo", {"source": "first"}),
        "b": ("bar", {"source": "second"}),
    }


def test_add_texts_overwrites_an_existing_id(store: RedisVectorStore) -> None:
    store.add_texts(["foo updated"], ids=["a"])

    assert _searched_contents(store) == ["bar", "foo updated"]


def test_add_texts_refuses_an_id_owned_by_another_index(
    store: RedisVectorStore,
    store_of_another_index_on_the_same_prefix: RedisVectorStore,
) -> None:
    with pytest.raises(ValueError, match="index ownership could not be verified"):
        store_of_another_index_on_the_same_prefix.add_texts(["hijacked"], ids=["a"])

    assert _searched_contents(store) == ["bar", "foo"]


def test_delete_removes_the_document(store: RedisVectorStore) -> None:
    assert store.delete(["b"]) is True

    assert _searched_contents(store) == ["foo"]


def test_delete_keeps_a_document_owned_by_another_index(
    store: RedisVectorStore,
    store_of_another_index_on_the_same_prefix: RedisVectorStore,
) -> None:
    assert store_of_another_index_on_the_same_prefix.delete(["a"]) is False

    assert _searched_contents(store) == ["bar", "foo"]


def test_similarity_search_return_all_returns_the_documents(
    store: RedisVectorStore,
) -> None:
    documents = store.similarity_search("foo", return_all=True)

    assert sorted(doc.page_content for doc in documents) == ["bar", "foo"]


def test_similarity_search_with_score_return_all_returns_the_documents(
    store: RedisVectorStore,
) -> None:
    results = store.similarity_search_with_score("foo", return_all=True)

    assert sorted(doc.page_content for doc, _ in results) == ["bar", "foo"]
