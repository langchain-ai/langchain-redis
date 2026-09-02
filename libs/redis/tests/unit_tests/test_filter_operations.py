"""Unit tests for filter-based delete and metadata update on RedisVectorStore."""

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings
from redisvl.query.filter import Tag  # type: ignore[import]

from langchain_redis import RedisVectorStore

INDEX_NAME = "filter_ops_unit"
REDIS_URL = "redis://localhost"
TEAM_FIELD = "team"
TEAM_VALUE = "alpha"
USER_FILTER = Tag(TEAM_FIELD) == TEAM_VALUE
NEW_VALUES = {"status": "archived"}
EXISTING_METADATA = {TEAM_FIELD: TEAM_VALUE, "status": "draft", "doc_id": "doc1"}
MATCHING_ROWS = [
    {"id": "filter_ops_unit:doc1", "_metadata_json": json.dumps(EXISTING_METADATA)}
]


class FakePipeline:
    def __init__(self, index: "FakeBulkIndex") -> None:
        self.index = index

    def __enter__(self) -> "FakePipeline":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def hset(self, key: str, mapping: Dict[str, Any]) -> None:
        self.index.hset_calls.append((key, mapping))

    def execute(self) -> List[int]:
        return [1 for _ in self.index.hset_calls]


class FakeRedisClient:
    def __init__(self, index: "FakeBulkIndex") -> None:
        self.index = index

    def pipeline(self, transaction: bool = False) -> FakePipeline:
        return FakePipeline(self.index)


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]


class FakeBulkIndex:
    """Captures bulk-operation calls and returns a canned BulkResult."""

    bulk_result = SimpleNamespace(matched=3, processed=3, completed=True)
    last_instance: Optional["FakeBulkIndex"] = None

    def __init__(self, schema: Optional[Dict[str, Any]] = None, **kwargs: Any):
        field_specs = (schema or {}).get("fields", [])
        self.schema = SimpleNamespace(
            fields={
                spec["name"]: SimpleNamespace(name=spec["name"]) for spec in field_specs
            }
        )
        self.captured_filter: Any = None
        self.captured_kwargs: Dict[str, Any] = {}
        self.captured_query: Any = None
        self.hset_calls: List[tuple[str, Dict[str, Any]]] = []
        self.client = FakeRedisClient(self)
        type(self).last_instance = self

    @classmethod
    def from_dict(cls, schema: Dict[str, Any], **kwargs: Any) -> "FakeBulkIndex":
        return cls(schema=schema)

    def create(self, overwrite: bool = False) -> None:
        pass

    def drop_by_filter(self, filter_expression: Any, **kwargs: Any) -> Any:
        self.captured_filter = filter_expression
        self.captured_kwargs = kwargs
        return type(self).bulk_result

    def paginate(self, query: Any, page_size: int) -> List[List[Dict[str, Any]]]:
        self.captured_query = query
        return [MATCHING_ROWS]


@pytest.fixture
def store() -> RedisVectorStore:
    with patch("langchain_redis.vectorstores.SearchIndex", FakeBulkIndex):
        return RedisVectorStore(
            MockEmbeddings(), index_name=INDEX_NAME, redis_url=REDIS_URL
        )


@pytest.fixture(autouse=True)
def reset_fake_index() -> None:
    FakeBulkIndex.bulk_result = SimpleNamespace(matched=3, processed=3, completed=True)


def _fake(store: RedisVectorStore) -> FakeBulkIndex:
    return store.index  # type: ignore[return-value]


def test_delete_rejects_ids_and_filter_together(store: RedisVectorStore) -> None:
    """ids and filter are mutually exclusive delete selectors."""
    with pytest.raises(ValueError, match="not both"):
        store.delete(ids=["doc1"], filter=USER_FILTER)


def test_delete_without_ids_or_filter_returns_false(store: RedisVectorStore) -> None:
    """The pre-existing no-op contract of delete() is preserved."""
    assert store.delete() is False


@pytest.mark.parametrize(
    "processed,expected", [(3, True), (0, False)], ids=["deleted", "no-match"]
)
def test_delete_with_filter_reports_whether_documents_removed(
    store: RedisVectorStore, processed: int, expected: bool
) -> None:
    """delete(filter=...) returns True iff at least one document was deleted."""
    FakeBulkIndex.bulk_result = SimpleNamespace(
        matched=processed, processed=processed, completed=True
    )
    assert store.delete(filter=USER_FILTER) is expected


def test_delete_by_filter_scopes_to_index_name(store: RedisVectorStore) -> None:
    """The user filter is AND-combined with the _index_name guard.

    Without this, a filter delete on an index sharing a key_prefix would
    destroy sibling indexes' documents.
    """
    store.delete_by_filter(USER_FILTER)
    captured = str(_fake(store).captured_filter)
    assert "_index_name" in captured
    assert TEAM_FIELD in captured


def test_delete_by_filter_dry_run_counts_without_deleting(
    store: RedisVectorStore,
) -> None:
    """dry_run=True forwards to redisvl and reports the matched count."""
    FakeBulkIndex.bulk_result = SimpleNamespace(matched=7, processed=0, completed=True)
    count = store.delete_by_filter(USER_FILTER, dry_run=True)
    assert count == 7
    assert _fake(store).captured_kwargs["dry_run"] is True


def test_delete_by_filter_requires_filter(store: RedisVectorStore) -> None:
    """A None filter is refused instead of silently deleting the index."""
    with pytest.raises(ValueError):
        store.delete_by_filter(None)  # type: ignore[arg-type]


def test_update_metadata_by_filter_syncs_metadata_json(
    store: RedisVectorStore,
) -> None:
    """Values are written to fields and merged into _metadata_json."""
    count = store.update_metadata_by_filter(USER_FILTER, NEW_VALUES)
    fake = _fake(store)
    assert count == len(MATCHING_ROWS)
    assert "_index_name" in str(fake.captured_query.filter)

    [(key, mapping)] = fake.hset_calls
    assert key == MATCHING_ROWS[0]["id"]
    assert mapping["status"] == NEW_VALUES["status"]
    assert json.loads(mapping["_metadata_json"]) == {
        **EXISTING_METADATA,
        **NEW_VALUES,
    }


@pytest.mark.parametrize("operation", ["delete", "update"])
def test_bulk_filter_operations_reject_raw_string_filters(
    store: RedisVectorStore, operation: str
) -> None:
    """Raw strings are refused because destructive filters must be safely scoped."""
    with pytest.raises(ValueError, match="FilterExpression"):
        if operation == "delete":
            store.delete_by_filter("@team:{alpha}")  # type: ignore[arg-type]
        else:
            store.update_metadata_by_filter(  # type: ignore[arg-type]
                "@team:{alpha}", NEW_VALUES
            )


def test_update_metadata_by_filter_rejects_protected_fields(
    store: RedisVectorStore,
) -> None:
    """Metadata updates may not corrupt content, embeddings, or internal fields."""
    with pytest.raises(ValueError, match="protected fields"):
        store.update_metadata_by_filter(USER_FILTER, {"_metadata_json": "{}"})
