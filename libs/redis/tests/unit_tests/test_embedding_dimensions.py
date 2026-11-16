"""add_texts() must refuse embeddings that don't fit the index's vector field.

Redis stores a record whose vector has the wrong size but leaves it out of the
index, so without this check the write looks successful.
"""

from typing import Any, Dict, Iterator
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from redisvl.schema import IndexSchema  # type: ignore[import]

from langchain_redis import RedisConfig, RedisVectorStore

INDEX_NAME = "dims_unit"
DIMS = 4


@pytest.fixture(params=["hash", "json"])
def storage_type(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def index(storage_type: str) -> Iterator[MagicMock]:
    """A search index that already holds DIMS-dimensional vectors."""
    schema: IndexSchema = RedisConfig(
        index_name=INDEX_NAME,
        redis_url="redis://localhost",
        embedding_dimensions=DIMS,
        storage_type=storage_type,
    ).to_index_schema()
    with patch("langchain_redis.vectorstores.SearchIndex") as search_index:
        index = search_index.return_value
        search_index.from_existing.return_value = index
        index.schema = schema
        index.name = INDEX_NAME
        index.key.side_effect = lambda document_id: f"{INDEX_NAME}:{document_id}"
        index.load.side_effect = lambda records, **kwargs: [
            index.key(str(i)) for i in range(len(records))
        ]
        yield index


def _embeddings(*sizes: int) -> Embeddings:
    """Embeddings whose n-th vector has the n-th size, repeating the sizes."""
    embeddings = MagicMock(spec=Embeddings)
    embeddings.embed_documents.side_effect = lambda texts: [
        [0.1] * sizes[i % len(sizes)] for i in range(len(texts))
    ]
    return embeddings


def _store_kwargs(storage_type: str, dims: int = DIMS) -> Dict[str, Any]:
    return {
        "index_name": INDEX_NAME,
        "redis_url": "redis://localhost",
        "embedding_dimensions": dims,
        "storage_type": storage_type,
    }


def test_matching_embeddings_are_written(index: MagicMock, storage_type: str) -> None:
    store = RedisVectorStore(_embeddings(DIMS), **_store_kwargs(storage_type))

    store.add_texts(["a", "b"])

    index.load.assert_called_once()
    assert len(index.load.call_args.args[0]) == 2


@pytest.mark.parametrize(
    ("sizes", "configured_dims", "position"),
    [
        pytest.param((2 * DIMS, 2 * DIMS), DIMS, 0, id="wrong-size"),
        pytest.param((DIMS, DIMS, DIMS + 1), DIMS, 2, id="one-bad-embedding"),
        # The index holds DIMS while the new model, and so the config, say 8
        pytest.param((2 * DIMS,), 2 * DIMS, 0, id="live-index-wins-over-config"),
    ],
)
def test_mismatched_embedding_is_rejected_and_nothing_is_written(
    index: MagicMock,
    storage_type: str,
    sizes: tuple,
    configured_dims: int,
    position: int,
) -> None:
    store = RedisVectorStore(
        _embeddings(*sizes), **_store_kwargs(storage_type, configured_dims)
    )
    assert store.config.embedding_dimensions == configured_dims

    with pytest.raises(
        ValueError,
        match=(
            rf"Embedding {position} has {sizes[position]} dimensions.*"
            rf"index '{INDEX_NAME}' stores {DIMS}-dimensional vectors"
        ),
    ):
        store.add_texts(["text"] * len(sizes))

    index.load.assert_not_called()


def test_add_documents_and_from_texts_go_through_the_check(
    index: MagicMock, storage_type: str
) -> None:
    embeddings = _embeddings(DIMS + 1)
    store = RedisVectorStore(embeddings, **_store_kwargs(storage_type))

    with pytest.raises(ValueError, match="5 dimensions"):
        store.add_documents([Document(page_content="a")])
    with pytest.raises(ValueError, match="5 dimensions"):
        RedisVectorStore.from_texts(["a"], embeddings, **_store_kwargs(storage_type))

    index.load.assert_not_called()
