"""add_texts() must refuse embeddings that don't fit the index's vector field.

Redis stores a record whose vector has the wrong size but leaves it out of the
index, so without this check the write looks successful.
"""

from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from redisvl.schema import IndexSchema  # type: ignore[import]

from langchain_redis import RedisConfig, RedisVectorStore

INDEX_NAME = "dims_unit"
DIMS = 4


class SizedEmbeddings(Embeddings):
    """Returns one vector per text, with the size given for that text."""

    def __init__(self, *sizes: int) -> None:
        self.sizes = sizes

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sizes = self.sizes or (DIMS,)
        return [[0.1] * sizes[i % len(sizes)] for i, _ in enumerate(texts)]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * (self.sizes[0] if self.sizes else DIMS)


def _schema(dims: int, storage_type: str) -> IndexSchema:
    return RedisConfig(
        index_name=INDEX_NAME,
        redis_url="redis://localhost",
        embedding_dimensions=dims,
        storage_type=storage_type,
    ).to_index_schema()


class FakeIndex:
    """Stands in for a SearchIndex and records what would be written."""

    last_instance: Optional["FakeIndex"] = None
    live_schema: Optional[IndexSchema] = None

    def __init__(self, schema: IndexSchema, **kwargs: Any) -> None:
        self.schema = schema
        self.name = schema.index.name
        self.client = MagicMock()
        self.loaded: List[Dict[str, Any]] = []
        type(self).last_instance = self

    @classmethod
    def from_existing(cls, name: str, **kwargs: Any) -> "FakeIndex":
        # What Redis reports for an index created earlier, possibly with
        # another vector size than the one the store was configured with.
        assert cls.last_instance is not None
        if cls.live_schema is not None:
            cls.last_instance.schema = cls.live_schema
        return cls.last_instance

    def create(self, overwrite: bool = False) -> None:
        pass

    def key(self, document_id: str) -> str:
        return f"{self.name}:{document_id}"

    def load(self, data: Any, keys: Optional[List[str]] = None, **kwargs: Any) -> Any:
        self.loaded = list(data)
        return keys or [self.key(str(i)) for i in range(len(self.loaded))]


@pytest.fixture(autouse=True)
def fake_index() -> Iterator[None]:
    FakeIndex.last_instance = None
    FakeIndex.live_schema = None
    with patch("langchain_redis.vectorstores.SearchIndex", FakeIndex):
        yield


@pytest.fixture(params=["hash", "json"])
def storage_type(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _store(embeddings: Embeddings, storage_type: str) -> RedisVectorStore:
    return RedisVectorStore(
        embeddings,
        index_name=INDEX_NAME,
        redis_url="redis://localhost",
        embedding_dimensions=DIMS,
        storage_type=storage_type,
    )


def _written() -> List[Dict[str, Any]]:
    assert FakeIndex.last_instance is not None
    return FakeIndex.last_instance.loaded


def test_matching_embeddings_are_written(storage_type: str) -> None:
    store = _store(SizedEmbeddings(DIMS), storage_type)

    store.add_texts(["a", "b"])

    assert len(_written()) == 2


def test_wrong_size_is_rejected_and_nothing_is_written(storage_type: str) -> None:
    store = _store(SizedEmbeddings(DIMS * 2), storage_type)

    with pytest.raises(ValueError, match=r"Embedding 0 has 8 dimensions") as exc:
        store.add_texts(["a", "b"])

    assert f"index '{INDEX_NAME}' stores 4-dimensional vectors" in str(exc.value)
    assert _written() == []


def test_one_wrong_embedding_rejects_the_whole_batch(storage_type: str) -> None:
    store = _store(SizedEmbeddings(DIMS, DIMS, DIMS + 1), storage_type)

    with pytest.raises(ValueError, match=r"Embedding 2 has 5 dimensions"):
        store.add_texts(["a", "b", "c"])

    assert _written() == []


def test_documents_and_from_texts_go_through_the_check(storage_type: str) -> None:
    store = _store(SizedEmbeddings(DIMS + 1), storage_type)

    with pytest.raises(ValueError, match="5 dimensions"):
        store.add_documents([Document(page_content="a")])
    with pytest.raises(ValueError, match="5 dimensions"):
        RedisVectorStore.from_texts(
            ["a"],
            SizedEmbeddings(DIMS + 1),
            index_name=INDEX_NAME,
            redis_url="redis://localhost",
            embedding_dimensions=DIMS,
            storage_type=storage_type,
        )

    assert _written() == []


def test_live_index_wins_over_the_configured_size(storage_type: str) -> None:
    """The issue's case: the index already holds 4-dimensional vectors while the
    new model, and so the store's config, say 8."""
    FakeIndex.live_schema = _schema(DIMS, storage_type)
    store = RedisVectorStore(
        SizedEmbeddings(DIMS * 2),
        index_name=INDEX_NAME,
        redis_url="redis://localhost",
        embedding_dimensions=DIMS * 2,
        storage_type=storage_type,
    )
    assert store.config.embedding_dimensions == DIMS * 2

    with pytest.raises(ValueError, match="8 dimensions.*stores 4-dimensional"):
        store.add_texts(["a"])

    assert _written() == []


def test_empty_batch_is_not_affected(storage_type: str) -> None:
    store = _store(SizedEmbeddings(DIMS + 1), storage_type)

    assert store.add_texts([]) == []
