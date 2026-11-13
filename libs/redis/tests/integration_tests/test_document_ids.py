"""Integration tests for caller-supplied document ids."""

from typing import Any, Dict, Generator, List, Tuple, cast
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from redis import Redis

from langchain_redis import RedisVectorStore

DIMS = 4


class ConstantEmbeddings(Embeddings):
    """All texts share one vector; these tests look documents up by id."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * DIMS for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * DIMS


@pytest.fixture
def index_name(redis_url: str) -> Generator[str, None, None]:
    name = f"document_ids_{uuid4().hex[:8]}"
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
        "embedding_dimensions": DIMS,
        "storage_type": request.param,
    }


def _contents_by_id(store: RedisVectorStore, ids: List[str]) -> Dict[str, str]:
    return {doc.id: doc.page_content for doc in store.get_by_ids(ids) if doc.id}


def _mixed_batch(doc_id: str) -> List[Document]:
    """One document with an id, followed by two without."""
    return [
        Document(page_content=doc_id, id=doc_id),
        Document(page_content=f"{doc_id} first without id"),
        Document(page_content=f"{doc_id} second without id"),
    ]


def _assert_stored(
    store: RedisVectorStore, batch: List[Document], ids: List[str]
) -> None:
    assert ids[0] == batch[0].id
    assert len(set(ids)) == len(batch)
    assert _contents_by_id(store, ids) == {
        id_: doc.page_content for id_, doc in zip(ids, batch)
    }


def test_from_documents_uses_document_ids(config: Dict[str, Any]) -> None:
    docs = [
        Document(page_content="foo", id="doc-1"),
        Document(page_content="bar", id="doc-2"),
    ]

    result = RedisVectorStore.from_documents(
        docs, ConstantEmbeddings(), return_keys=True, **config
    )
    store, keys = cast(Tuple[RedisVectorStore, List[str]], result)

    assert keys == ["doc-1", "doc-2"]
    assert _contents_by_id(store, keys) == {"doc-1": "foo", "doc-2": "bar"}


def test_from_texts_uses_ids(config: Dict[str, Any]) -> None:
    store = RedisVectorStore.from_texts(
        ["foo", "bar"], ConstantEmbeddings(), ids=["doc-1", "doc-2"], **config
    )

    assert _contents_by_id(store, ["doc-1", "doc-2"]) == {
        "doc-1": "foo",
        "doc-2": "bar",
    }


def test_from_texts_prefers_keys_over_ids(config: Dict[str, Any]) -> None:
    store = RedisVectorStore.from_texts(
        ["foo"], ConstantEmbeddings(), keys=["key-1"], ids=["doc-1"], **config
    )

    assert _contents_by_id(store, ["key-1", "doc-1"]) == {"key-1": "foo"}


@pytest.mark.parametrize("argument", ["ids", "keys"])
def test_from_documents_prefers_explicit_ids_over_document_ids(
    config: Dict[str, Any], argument: str
) -> None:
    config[argument] = ["explicit-1"]

    store = RedisVectorStore.from_documents(
        [Document(page_content="foo", id="doc-1")], ConstantEmbeddings(), **config
    )

    assert _contents_by_id(store, ["explicit-1", "doc-1"]) == {"explicit-1": "foo"}


def test_from_documents_with_same_ids_updates_documents(
    config: Dict[str, Any],
) -> None:
    RedisVectorStore.from_documents(
        [Document(page_content="v1", id="doc-1")], ConstantEmbeddings(), **config
    )

    store = RedisVectorStore.from_documents(
        [Document(page_content="v2", id="doc-1")], ConstantEmbeddings(), **config
    )

    assert [doc.page_content for doc in store.similarity_search("v2", k=10)] == ["v2"]


def test_add_documents_generates_ids_for_documents_without_one(
    config: Dict[str, Any],
) -> None:
    store = RedisVectorStore(ConstantEmbeddings(), **config)
    first, second = _mixed_batch("doc-1"), _mixed_batch("doc-2")

    first_ids = store.add_documents(first)
    second_ids = store.add_documents(second)

    # The second batch must not overwrite the first one's documents.
    _assert_stored(store, first, first_ids)
    _assert_stored(store, second, second_ids)


def test_from_documents_generates_ids_for_documents_without_one(
    config: Dict[str, Any],
) -> None:
    batch = _mixed_batch("doc-1")

    result = RedisVectorStore.from_documents(
        batch, ConstantEmbeddings(), return_keys=True, **config
    )
    store, keys = cast(Tuple[RedisVectorStore, List[str]], result)

    _assert_stored(store, batch, keys)


async def test_afrom_documents_uses_document_ids(config: Dict[str, Any]) -> None:
    store = await RedisVectorStore.afrom_documents(
        [Document(page_content="foo", id="doc-1")], ConstantEmbeddings(), **config
    )

    assert _contents_by_id(store, ["doc-1"]) == {"doc-1": "foo"}


async def test_afrom_texts_uses_ids(config: Dict[str, Any]) -> None:
    store = await RedisVectorStore.afrom_texts(
        ["foo", "bar"], ConstantEmbeddings(), ids=["doc-1", "doc-2"], **config
    )

    assert _contents_by_id(store, ["doc-1", "doc-2"]) == {
        "doc-1": "foo",
        "doc-2": "bar",
    }


async def test_aadd_documents_generates_ids_for_documents_without_one(
    config: Dict[str, Any],
) -> None:
    store = RedisVectorStore(ConstantEmbeddings(), **config)
    batch = _mixed_batch("doc-1")

    ids = await store.aadd_documents(batch)

    _assert_stored(store, batch, ids)
