"""Unit tests for filter-based deletion on RedisVectorStore."""

import json
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings
from redisvl.query.filter import (  # type: ignore[import]
    FilterExpression,
    Num,
    Tag,
)
from redisvl.redis.utils import hashify  # type: ignore[import]
from redisvl.schema import FieldTypes  # type: ignore[import]

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
    live_schema: Any = None
    live_schema_error: Optional[Exception] = None

    def __init__(self, schema: Optional[Dict[str, Any]] = None, **kwargs: Any):
        field_specs = (schema or {}).get("fields", [])
        self.schema: Any = SimpleNamespace(
            fields={
                spec["name"]: SimpleNamespace(
                    name=spec["name"],
                    type=FieldTypes(spec["type"]),
                    attrs=SimpleNamespace(
                        **{
                            "no_index": False,
                            "separator": ",",
                            **spec.get("attrs", {}),
                        }
                    ),
                )
                for spec in field_specs
            }
        )
        self.name = (schema or {}).get("index", {}).get("name", INDEX_NAME)
        self.client = object()
        self.captured_filter: Any = None
        self.captured_kwargs: Dict[str, Any] = {}
        self.loaded_records: List[Dict[str, Any]] = []
        type(self).last_instance = self

    @classmethod
    def from_dict(cls, schema: Dict[str, Any], **kwargs: Any) -> "FakeBulkIndex":
        return cls(schema=schema)

    @classmethod
    def from_existing(cls, name: str, redis_client: Any = None, **kwargs: Any) -> Any:
        if cls.live_schema_error is not None:
            raise cls.live_schema_error
        assert cls.last_instance is not None
        assert name == cls.last_instance.name
        assert redis_client is cls.last_instance.client
        if cls.live_schema is not None:
            return SimpleNamespace(schema=cls.live_schema)
        return cls.last_instance

    def create(self, overwrite: bool = False) -> None:
        pass

    def drop_by_filter(self, filter_expression: Any, **kwargs: Any) -> Any:
        self.captured_filter = filter_expression
        self.captured_kwargs = kwargs
        return type(self).bulk_result

    def load(
        self, data: Any, keys: Optional[List[str]] = None, **kwargs: Any
    ) -> List[str]:
        self.loaded_records = list(data)
        return keys or [f"{self.name}:generated"]


class BrokenSchema:
    """Schema double that fails whenever callers inspect its fields."""

    @property
    def fields(self) -> Dict[str, Any]:
        raise RuntimeError("schema unavailable")


@pytest.fixture
def store() -> Iterator[RedisVectorStore]:
    with patch("langchain_redis.vectorstores.SearchIndex", FakeBulkIndex):
        yield RedisVectorStore(
            MockEmbeddings(), index_name=INDEX_NAME, redis_url=REDIS_URL
        )


@pytest.fixture(autouse=True)
def reset_fake_index() -> None:
    FakeBulkIndex.bulk_result = SimpleNamespace(matched=3, processed=3, completed=True)
    FakeBulkIndex.last_instance = None
    FakeBulkIndex.live_schema = None
    FakeBulkIndex.live_schema_error = None


def _fake(store: RedisVectorStore) -> FakeBulkIndex:
    return store.index  # type: ignore[return-value]


def _schema_with_index_marker(
    field_type: Optional[FieldTypes],
    *,
    no_index: bool = False,
    separator: str = ",",
) -> Any:
    fields = {}
    if field_type is not None:
        fields["_index_name"] = SimpleNamespace(
            name="_index_name",
            type=field_type,
            attrs=SimpleNamespace(no_index=no_index, separator=separator),
        )
    return SimpleNamespace(fields=fields)


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
    assert hashify(INDEX_NAME) in captured
    assert INDEX_NAME not in captured
    assert TEAM_FIELD in captured


def test_generated_index_name_field_is_case_sensitive_tag(
    store: RedisVectorStore,
) -> None:
    """Generated indexes use an exact TAG marker for destructive scoping."""
    field = _fake(store).schema.fields["_index_name"]
    assert field.type == FieldTypes.TAG
    assert field.attrs.case_sensitive is True


def test_tag_index_marker_uses_search_index_name(
    store: RedisVectorStore,
) -> None:
    """TAG ownership follows RedisVL's index identity, not stale config."""
    fake = _fake(store)
    fake.name = "live-index-name"

    store.delete_by_filter(USER_FILTER)

    captured = str(fake.captured_filter)
    assert hashify(fake.name) in captured
    assert hashify(INDEX_NAME) not in captured
    assert store._index_name_value(FieldTypes.TEXT) == INDEX_NAME


def test_add_texts_protects_tag_index_marker_from_metadata(
    store: RedisVectorStore,
) -> None:
    """Caller metadata cannot replace the indexed ownership marker."""
    fake = _fake(store)
    fake.name = "live-index-name"
    caller_marker = hashify("sibling-index")

    store.add_texts(
        ["protected document"],
        metadatas=[{"_index_name": caller_marker, TEAM_FIELD: TEAM_VALUE}],
        keys=["protected"],
    )

    record = fake.loaded_records[0]
    assert record["_index_name"] == hashify(fake.name)
    assert json.loads(record["_metadata_json"])["_index_name"] == caller_marker


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


@pytest.mark.parametrize("dry_run", [False, True], ids=["delete", "dry-run"])
@pytest.mark.parametrize(
    "field_type",
    [
        pytest.param(None, id="missing"),
        pytest.param(FieldTypes.TEXT, id="text"),
        pytest.param(FieldTypes.NUMERIC, id="numeric"),
    ],
)
def test_delete_by_filter_requires_tag_index_marker(
    store: RedisVectorStore,
    field_type: Optional[FieldTypes],
    dry_run: bool,
) -> None:
    """Deletion fails closed when exact index ownership cannot be proven."""
    fake = _fake(store)
    assert fake.schema.fields["_index_name"].type == FieldTypes.TAG
    FakeBulkIndex.live_schema = _schema_with_index_marker(field_type)

    with pytest.raises(ValueError, match="requires an '_index_name' TAG field"):
        store.delete_by_filter(USER_FILTER, dry_run=dry_run)

    assert fake.captured_filter is None


def test_delete_by_filter_fails_closed_when_live_schema_inspection_fails(
    store: RedisVectorStore,
) -> None:
    """Schema errors are surfaced before RedisVL receives a destructive call."""
    fake = _fake(store)
    FakeBulkIndex.live_schema = BrokenSchema()

    with pytest.raises(ValueError, match="could not inspect the live index schema"):
        store.delete_by_filter(USER_FILTER)

    assert fake.captured_filter is None


def test_delete_by_filter_fails_closed_when_live_schema_lookup_fails(
    store: RedisVectorStore,
) -> None:
    """A failed Redis schema lookup cannot fall back to the local schema."""
    fake = _fake(store)
    FakeBulkIndex.live_schema_error = RuntimeError("FT.INFO unavailable")

    with pytest.raises(ValueError, match="could not inspect the live index schema"):
        store.delete_by_filter(USER_FILTER)

    assert fake.captured_filter is None


def test_delete_by_filter_accepts_live_tag_schema_when_local_schema_is_stale(
    store: RedisVectorStore,
) -> None:
    """The live TAG schema is authoritative when the local schema is stale."""
    fake = _fake(store)
    fake.schema.fields["_index_name"].type = FieldTypes.TEXT
    FakeBulkIndex.live_schema = _schema_with_index_marker(FieldTypes.TAG)

    assert store.delete_by_filter(USER_FILTER) == 3
    assert hashify(INDEX_NAME) in str(fake.captured_filter)


@pytest.mark.parametrize(
    ("schema", "expected_error"),
    [
        pytest.param(
            _schema_with_index_marker(FieldTypes.TAG, no_index=True),
            "TAG field to be indexed",
            id="not-indexed",
        ),
        pytest.param(
            _schema_with_index_marker(
                FieldTypes.TAG,
                separator=hashify(INDEX_NAME)[0],
            ),
            "separator because it splits",
            id="marker-splitting-separator",
        ),
    ],
)
def test_delete_by_filter_rejects_unusable_tag_index_marker(
    store: RedisVectorStore,
    schema: Any,
    expected_error: str,
) -> None:
    """A TAG marker must be indexed and store the hash as one tag."""
    fake = _fake(store)
    FakeBulkIndex.live_schema = schema

    with pytest.raises(ValueError, match=expected_error):
        store.delete_by_filter(USER_FILTER)

    assert fake.captured_filter is None


def test_read_filter_supports_existing_text_index_marker(
    store: RedisVectorStore,
) -> None:
    """Read queries retain compatibility with existing TEXT marker schemas."""
    _fake(store).schema.fields["_index_name"].type = FieldTypes.TEXT

    scoped = store._with_index_name_filter(USER_FILTER)

    assert INDEX_NAME in str(scoped)
    assert hashify(INDEX_NAME) not in str(scoped)


def test_read_filter_tolerates_missing_index_marker(store: RedisVectorStore) -> None:
    """Custom schemas without the internal marker remain searchable."""
    _fake(store).schema.fields.pop("_index_name")

    assert store._with_index_name_filter(USER_FILTER) is USER_FILTER


def test_read_filter_tolerates_schema_inspection_failure(
    store: RedisVectorStore,
) -> None:
    """Read scoping preserves its existing best-effort failure policy."""
    _fake(store).schema = BrokenSchema()

    assert store._with_index_name_filter(USER_FILTER) is USER_FILTER


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
