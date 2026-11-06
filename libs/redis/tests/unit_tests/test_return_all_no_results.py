"""`return_all=True` searches that match nothing must not fetch zero keys."""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings
from redis.exceptions import ResponseError  # type: ignore[import]
from redisvl.schema import FieldTypes  # type: ignore[import]

from langchain_redis import RedisVectorStore

INDEX_NAME = "return_all_unit"
REDIS_URL = "redis://localhost"
DIMS = 3


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]


class FakeJson:
    """The `JSON.MGET` arity, which is what makes an empty fetch a failure.

    `JSON.MGET key [key ...] path` takes one or more keys, so Redis answers an
    empty key list with an error rather than an empty reply.
    """

    calls: List[List[str]] = []

    def mget(self, keys: List[str], path: str) -> List[Dict[str, Any]]:
        type(self).calls.append(list(keys))
        if not keys:
            raise ResponseError("wrong number of arguments for 'json.mget' command")
        return [{"text": "stored text"} for _ in keys]


class FakeSearchIndex:
    """A search index whose query() returns whatever rows the test set."""

    rows: List[Dict[str, Any]] = []
    last_instance: Optional["FakeSearchIndex"] = None

    def __init__(self, schema: Optional[Dict[str, Any]] = None, **kwargs: Any):
        field_specs = (schema or {}).get("fields", [])
        self.schema = SimpleNamespace(
            fields={
                spec["name"]: SimpleNamespace(
                    name=spec["name"], type=FieldTypes(spec["type"])
                )
                for spec in field_specs
            }
        )
        self.name = (schema or {}).get("index", {}).get("name", INDEX_NAME)
        self.client = SimpleNamespace(
            json=lambda: FakeJson(),
            info=lambda section=None: {"redis_version": "8.4.0"},
        )
        type(self).last_instance = self

    @classmethod
    def from_dict(cls, schema: Dict[str, Any], **kwargs: Any) -> "FakeSearchIndex":
        return cls(schema=schema)

    @classmethod
    def from_existing(cls, name: str, **kwargs: Any) -> "FakeSearchIndex":
        assert cls.last_instance is not None
        return cls.last_instance

    def create(self, overwrite: bool = False) -> None:
        pass

    def query(self, query: Any) -> List[Dict[str, Any]]:
        return list(type(self).rows)


@pytest.fixture
def store() -> RedisVectorStore:
    with patch("langchain_redis.vectorstores.SearchIndex", FakeSearchIndex):
        return RedisVectorStore(
            MockEmbeddings(),
            index_name=INDEX_NAME,
            redis_url=REDIS_URL,
            storage_type="json",
            embedding_dimensions=DIMS,
        )


@pytest.fixture(autouse=True)
def reset_fakes() -> None:
    FakeSearchIndex.rows = []
    FakeJson.calls = []


def test_with_score_return_all_json_handles_no_matches(
    store: RedisVectorStore,
) -> None:
    """An empty result set returns no documents instead of fetching no keys.

    Any search that matches nothing reaches here: an empty index, a filter or
    a distance_threshold that excludes every document.
    """
    assert store.similarity_search_with_score("hello", return_all=True) == []
    assert FakeJson.calls == []


def test_no_score_return_all_json_handles_no_matches(
    store: RedisVectorStore,
) -> None:
    """The sibling without scores behaves the same, as it already did."""
    assert store.similarity_search_by_vector([0.1, 0.2, 0.3], return_all=True) == []
    assert FakeJson.calls == []


def test_with_score_return_all_json_still_fetches_matches(
    store: RedisVectorStore,
) -> None:
    """The guard leaves the fetch alone when the search does match."""
    FakeSearchIndex.rows = [{"id": "doc:1", "vector_distance": "0.25"}]

    results = store.similarity_search_with_score("hello", return_all=True)

    assert FakeJson.calls == [["doc:1"]]
    assert len(results) == 1
    document, score = results[0]
    assert document.page_content == "stored text"
    assert score == pytest.approx(0.25)
