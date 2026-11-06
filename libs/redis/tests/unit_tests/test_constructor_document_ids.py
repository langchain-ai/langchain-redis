"""Caller-supplied document ids must survive the `RedisVectorStore` constructors.

`VectorStore.from_documents` passes `Document.id` on as `ids`, and
`VectorStore.from_texts` takes `ids` directly, but `RedisVectorStore` overrides
both and used to drop the ids on the floor: the records were written under
generated ULIDs, so `get_by_ids()` and `delete()` could not reach them and
re-ingesting the same documents duplicated them instead of updating them.

`add_documents` has a sharper version of the same problem. The base class sends
an id of None for every document without one as soon as one document in the
batch has one, and those used to be keyed as the literal "<prefix>:None", so a
mixed batch overwrote itself down to a single record. See issue #224.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from langchain_redis import RedisVectorStore

_CONFIG_ATTRS: Dict[str, Any] = {
    "index_name": "test_index",
    "key_prefix": "p",
    "primary_prefix": "p",
    "embedding_dimensions": 3,
    "content_field": "text",
    "embedding_field": "embedding",
    "index_schema": None,
    "schema_path": None,
    "from_existing": False,
    "storage_type": "JSON",
    "metadata_schema": None,
    "vector_datatype": "FLOAT32",
    "default_tag_separator": ",",
    "distance_metric": "COSINE",
    "indexing_algorithm": "FLAT",
}


def _configure(mock_config: MagicMock) -> None:
    """Make both `RedisConfig()` and `RedisConfig.from_kwargs()` usable."""
    for stub in (mock_config.return_value, mock_config.from_kwargs.return_value):
        for name, value in _CONFIG_ATTRS.items():
            setattr(stub, name, value)
        stub.redis.return_value = MagicMock()


def _index() -> MagicMock:
    index = MagicMock()
    index.key.side_effect = lambda document_id: f"p:{document_id}"
    index.load.side_effect = lambda *args, **kwargs: list(kwargs.get("keys") or [])
    index.schema.fields.values.return_value = []
    index.schema.fields.get.return_value = None
    return index


def _embeddings() -> MagicMock:
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    embeddings.embed_documents.side_effect = lambda texts: [
        [0.1, 0.2, 0.3] for _ in texts
    ]
    return embeddings


def _written_keys(index: MagicMock) -> Optional[List[str]]:
    """The keys handed to the one `SearchIndex.load` call."""
    assert index.load.call_count == 1
    return index.load.call_args.kwargs.get("keys")


def test_from_documents_writes_each_document_under_its_own_id() -> None:
    with (
        patch("langchain_redis.vectorstores.RedisConfig") as mock_config,
        patch("langchain_redis.vectorstores.SearchIndex") as mock_search_index,
    ):
        _configure(mock_config)
        index = _index()
        mock_search_index.return_value = index
        mock_search_index.from_dict.return_value = index
        mock_search_index.from_existing.return_value = index

        RedisVectorStore.from_documents(
            [
                Document(page_content="first", id="doc-1"),
                Document(page_content="second", id="doc-2"),
            ],
            _embeddings(),
        )

        assert _written_keys(index) == ["p:doc-1", "p:doc-2"]


def test_from_documents_leaves_keys_generated_when_no_document_has_an_id() -> None:
    """With no ids anywhere, RedisVL still has to do the key generation."""
    with (
        patch("langchain_redis.vectorstores.RedisConfig") as mock_config,
        patch("langchain_redis.vectorstores.SearchIndex") as mock_search_index,
    ):
        _configure(mock_config)
        index = _index()
        mock_search_index.return_value = index
        mock_search_index.from_dict.return_value = index
        mock_search_index.from_existing.return_value = index

        RedisVectorStore.from_documents(
            [Document(page_content="first"), Document(page_content="second")],
            _embeddings(),
        )

        assert _written_keys(index) is None


def test_from_documents_prefers_an_explicit_ids_argument() -> None:
    with (
        patch("langchain_redis.vectorstores.RedisConfig") as mock_config,
        patch("langchain_redis.vectorstores.SearchIndex") as mock_search_index,
    ):
        _configure(mock_config)
        index = _index()
        mock_search_index.return_value = index
        mock_search_index.from_dict.return_value = index
        mock_search_index.from_existing.return_value = index

        RedisVectorStore.from_documents(
            [Document(page_content="first", id="ignored")],
            _embeddings(),
            ids=["chosen"],
        )

        assert _written_keys(index) == ["p:chosen"]


def test_from_texts_forwards_its_ids_argument() -> None:
    with (
        patch("langchain_redis.vectorstores.RedisConfig") as mock_config,
        patch("langchain_redis.vectorstores.SearchIndex") as mock_search_index,
    ):
        _configure(mock_config)
        index = _index()
        mock_search_index.return_value = index
        mock_search_index.from_dict.return_value = index
        mock_search_index.from_existing.return_value = index

        RedisVectorStore.from_texts(
            ["first", "second"], _embeddings(), ids=["doc-1", "doc-2"]
        )

        assert _written_keys(index) == ["p:doc-1", "p:doc-2"]


def test_add_documents_keys_an_id_less_document_apart_from_the_rest() -> None:
    """A mixed batch must not collapse onto one "<prefix>:None" key."""
    with (
        patch("langchain_redis.vectorstores.RedisConfig") as mock_config,
        patch("langchain_redis.vectorstores.SearchIndex") as mock_search_index,
    ):
        _configure(mock_config)
        index = _index()
        mock_search_index.return_value = index
        mock_search_index.from_dict.return_value = index
        mock_search_index.from_existing.return_value = index

        store = RedisVectorStore(embeddings=_embeddings())
        with patch.object(store, "_validate_write_ownership"):
            store.add_documents(
                [
                    Document(page_content="a", id="doc-1"),
                    Document(page_content="b"),
                    Document(page_content="c"),
                ]
            )

        keys = _written_keys(index)
        assert keys is not None
        assert keys[0] == "p:doc-1"
        assert len(set(keys)) == 3
        assert "p:None" not in keys


async def test_afrom_documents_also_writes_each_document_under_its_own_id() -> None:
    """`afrom_documents` has no override here, so it inherits the fixed path."""
    with (
        patch("langchain_redis.vectorstores.RedisConfig") as mock_config,
        patch("langchain_redis.vectorstores.SearchIndex") as mock_search_index,
    ):
        _configure(mock_config)
        index = _index()
        mock_search_index.return_value = index
        mock_search_index.from_dict.return_value = index
        mock_search_index.from_existing.return_value = index

        await RedisVectorStore.afrom_documents(
            [Document(page_content="first", id="doc-1")], _embeddings()
        )

        assert _written_keys(index) == ["p:doc-1"]
