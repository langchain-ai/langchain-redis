"""Integration tests for caller-supplied document ids."""

from typing import Any, Dict, Generator, List, Tuple, cast
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from redis import Redis

from langchain_redis import RedisVectorStore

DIMS = 4
STORAGE_TYPES = ["hash", "json"]


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


def _config(redis_url: str, index_name: str, storage_type: str) -> Dict[str, Any]:
    return {
        "redis_url": redis_url,
        "index_name": index_name,
        "embedding_dimensions": DIMS,
        "storage_type": storage_type,
    }


def _contents_by_id(store: RedisVectorStore, ids: List[str]) -> Dict[str, str]:
    return {doc.id: doc.page_content for doc in store.get_by_ids(ids) if doc.id}


@pytest.mark.parametrize("storage_type", STORAGE_TYPES)
def test_from_documents_uses_document_ids(
    redis_url: str, index_name: str, storage_type: str
) -> None:
    docs = [
        Document(page_content="foo", id="doc-1"),
        Document(page_content="bar", id="doc-2"),
    ]

    result = RedisVectorStore.from_documents(
        docs,
        ConstantEmbeddings(),
        return_keys=True,
        **_config(redis_url, index_name, storage_type),
    )
    store, keys = cast(Tuple[RedisVectorStore, List[str]], result)

    assert keys == ["doc-1", "doc-2"]
    assert _contents_by_id(store, ["doc-1", "doc-2"]) == {
        "doc-1": "foo",
        "doc-2": "bar",
    }


@pytest.mark.parametrize("storage_type", STORAGE_TYPES)
def test_from_texts_uses_ids(
    redis_url: str, index_name: str, storage_type: str
) -> None:
    store = RedisVectorStore.from_texts(
        ["foo", "bar"],
        ConstantEmbeddings(),
        ids=["doc-1", "doc-2"],
        **_config(redis_url, index_name, storage_type),
    )

    assert _contents_by_id(store, ["doc-1", "doc-2"]) == {
        "doc-1": "foo",
        "doc-2": "bar",
    }


@pytest.mark.parametrize("storage_type", STORAGE_TYPES)
def test_from_texts_prefers_keys_over_ids(
    redis_url: str, index_name: str, storage_type: str
) -> None:
    store = RedisVectorStore.from_texts(
        ["foo"],
        ConstantEmbeddings(),
        keys=["key-1"],
        ids=["doc-1"],
        **_config(redis_url, index_name, storage_type),
    )

    assert _contents_by_id(store, ["key-1", "doc-1"]) == {"key-1": "foo"}


@pytest.mark.parametrize("storage_type", STORAGE_TYPES)
@pytest.mark.parametrize("argument", ["ids", "keys"])
def test_from_documents_prefers_explicit_ids_over_document_ids(
    redis_url: str, index_name: str, storage_type: str, argument: str
) -> None:
    config = _config(redis_url, index_name, storage_type)
    config[argument] = ["explicit-1"]

    store = RedisVectorStore.from_documents(
        [Document(page_content="foo", id="doc-1")], ConstantEmbeddings(), **config
    )

    assert _contents_by_id(store, ["explicit-1", "doc-1"]) == {"explicit-1": "foo"}


@pytest.mark.parametrize("storage_type", STORAGE_TYPES)
def test_from_documents_with_same_ids_updates_documents(
    redis_url: str, index_name: str, storage_type: str
) -> None:
    config = _config(redis_url, index_name, storage_type)
    RedisVectorStore.from_documents(
        [Document(page_content="v1", id="doc-1")], ConstantEmbeddings(), **config
    )

    store = RedisVectorStore.from_documents(
        [Document(page_content="v2", id="doc-1")], ConstantEmbeddings(), **config
    )

    assert [doc.page_content for doc in store.similarity_search("v2", k=10)] == ["v2"]


@pytest.mark.parametrize("storage_type", STORAGE_TYPES)
def test_add_documents_generates_ids_for_documents_without_one(
    redis_url: str, index_name: str, storage_type: str
) -> None:
    store = RedisVectorStore(
        ConstantEmbeddings(), **_config(redis_url, index_name, storage_type)
    )

    first = store.add_documents(
        [
            Document(page_content="a", id="doc-1"),
            Document(page_content="b"),
            Document(page_content="c"),
        ]
    )
    second = store.add_documents(
        [
            Document(page_content="d", id="doc-2"),
            Document(page_content="e"),
            Document(page_content="f"),
        ]
    )

    assert first[0] == "doc-1"
    assert second[0] == "doc-2"
    assert len(set(first + second)) == 6
    assert "None" not in first + second
    assert sorted(_contents_by_id(store, first + second).values()) == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    ]


@pytest.mark.parametrize("storage_type", STORAGE_TYPES)
def test_from_documents_generates_ids_for_documents_without_one(
    redis_url: str, index_name: str, storage_type: str
) -> None:
    result = RedisVectorStore.from_documents(
        [
            Document(page_content="a", id="doc-1"),
            Document(page_content="b"),
            Document(page_content="c"),
        ],
        ConstantEmbeddings(),
        return_keys=True,
        **_config(redis_url, index_name, storage_type),
    )
    store, keys = cast(Tuple[RedisVectorStore, List[str]], result)

    assert keys[0] == "doc-1"
    assert len(set(keys)) == 3
    assert "None" not in keys
    assert sorted(_contents_by_id(store, keys).values()) == ["a", "b", "c"]


@pytest.mark.parametrize("storage_type", STORAGE_TYPES)
async def test_afrom_documents_uses_document_ids(
    redis_url: str, index_name: str, storage_type: str
) -> None:
    store = await RedisVectorStore.afrom_documents(
        [
            Document(page_content="foo", id="doc-1"),
            Document(page_content="bar"),
        ],
        ConstantEmbeddings(),
        **_config(redis_url, index_name, storage_type),
    )

    assert _contents_by_id(store, ["doc-1"]) == {"doc-1": "foo"}
    assert len(store.similarity_search("bar", k=10)) == 2


@pytest.mark.parametrize("storage_type", STORAGE_TYPES)
async def test_afrom_texts_uses_ids(
    redis_url: str, index_name: str, storage_type: str
) -> None:
    store = await RedisVectorStore.afrom_texts(
        ["foo", "bar"],
        ConstantEmbeddings(),
        ids=["doc-1", "doc-2"],
        **_config(redis_url, index_name, storage_type),
    )

    assert _contents_by_id(store, ["doc-1", "doc-2"]) == {
        "doc-1": "foo",
        "doc-2": "bar",
    }


@pytest.mark.parametrize("storage_type", STORAGE_TYPES)
async def test_aadd_documents_generates_ids_for_documents_without_one(
    redis_url: str, index_name: str, storage_type: str
) -> None:
    store = RedisVectorStore(
        ConstantEmbeddings(), **_config(redis_url, index_name, storage_type)
    )

    ids = await store.aadd_documents(
        [
            Document(page_content="a", id="doc-1"),
            Document(page_content="b"),
            Document(page_content="c"),
        ]
    )

    assert ids[0] == "doc-1"
    assert len(set(ids)) == 3
    assert "None" not in ids
    assert sorted(_contents_by_id(store, ids).values()) == ["a", "b", "c"]
