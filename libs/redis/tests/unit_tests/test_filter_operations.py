"""Unit tests for filter-based deletion on RedisVectorStore."""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings
from redisvl.query.filter import (  # type: ignore[import]
    FilterExpression,
    Num,
    Tag,
)

from langchain_redis import RedisVectorStore

INDEX_NAME = "filter_ops_unit"
REDIS_URL = "redis://localhost"
TEAM_FIELD = "team"
TEAM_VALUE = "alpha"
USER_FILTER = Tag(TEAM_FIELD) == TEAM_VALUE
_MATCH_ALL_ERROR = "refuses filters that match all documents"
_UNINITIALIZED_FILTER_ERROR = "specific, initialized filter expression"


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


@pytest.mark.parametrize(
    ("unsafe_filter", "expected_error"),
    [
        pytest.param(
            FilterExpression("*"),
            _MATCH_ALL_ERROR,
            id="explicit-match-all",
        ),
        pytest.param(
            FilterExpression("   *   "),
            _MATCH_ALL_ERROR,
            id="whitespace-match-all",
        ),
        pytest.param(
            FilterExpression("(*)"),
            _MATCH_ALL_ERROR,
            id="parenthesized-match-all",
        ),
        pytest.param(
            FilterExpression("(((*)))"),
            _MATCH_ALL_ERROR,
            id="nested-match-all",
        ),
        pytest.param(
            Tag(TEAM_FIELD) == "",
            _MATCH_ALL_ERROR,
            id="empty-tag",
        ),
        pytest.param(
            Tag(TEAM_FIELD) == [],
            _MATCH_ALL_ERROR,
            id="empty-tag-list",
        ),
        pytest.param(
            Tag(TEAM_FIELD) != "",
            _MATCH_ALL_ERROR,
            id="not-empty-tag",
        ),
        pytest.param(
            Num("priority") == None,  # noqa: E711
            _MATCH_ALL_ERROR,
            id="none-number",
        ),
        pytest.param(
            FilterExpression(),
            _UNINITIALIZED_FILTER_ERROR,
            id="uninitialized",
        ),
    ],
)
def test_delete_by_filter_rejects_unsafe_filters(
    store: RedisVectorStore,
    unsafe_filter: FilterExpression,
    expected_error: str,
) -> None:
    """Unsafe filters are rejected through the intended validation path."""
    with pytest.raises(ValueError, match=expected_error):
        store.delete_by_filter(unsafe_filter)

    assert _fake(store).captured_filter is None


def test_delete_by_filter_allows_field_wildcard(
    store: RedisVectorStore,
) -> None:
    """A field-specific wildcard remains valid because it is not global `*`."""
    store.delete_by_filter(Tag(TEAM_FIELD) % "*")

    captured = str(_fake(store).captured_filter)
    assert f"@{TEAM_FIELD}:{{*}}" in captured
    assert "_index_name" in captured


def test_delete_by_filter_rejects_raw_string_filters(
    store: RedisVectorStore,
) -> None:
    """Raw strings are refused because destructive filters must be safely scoped."""
    with pytest.raises(ValueError, match="FilterExpression"):
        store.delete_by_filter("@team:{alpha}")  # type: ignore[arg-type]
