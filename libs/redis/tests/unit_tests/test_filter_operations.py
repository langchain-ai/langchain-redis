"""Unit tests for filter-based deletion on RedisVectorStore."""

import json
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.embeddings import Embeddings
from redisvl.index.index import BulkResult  # type: ignore[import]
from redisvl.query.filter import (  # type: ignore[import]
    FilterExpression,
    Num,
    Tag,
)
from redisvl.redis.utils import hashify  # type: ignore[import]
from redisvl.schema import FieldTypes, StorageType  # type: ignore[import]

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

    bulk_result = BulkResult(matched=3, processed=3)
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

    def key(self, document_id: str) -> str:
        return f"{self.name}:{document_id}"

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
    FakeBulkIndex.bulk_result = BulkResult(matched=3, processed=3)
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


@pytest.mark.parametrize(
    "kwargs",
    [pytest.param({}, id="omitted"), pytest.param({"filter": None}, id="none")],
)
def test_delete_without_selector_returns_false(
    store: RedisVectorStore, kwargs: Dict[str, Any]
) -> None:
    """The pre-existing no-op contract of delete() is preserved."""
    assert store.delete(**kwargs) is False


@pytest.mark.parametrize(
    "ids",
    [
        pytest.param(None, id="omitted"),
        pytest.param([], id="empty"),
        pytest.param(["doc1"], id="populated"),
    ],
)
def test_delete_rejects_filter_without_mutating(
    store: RedisVectorStore, ids: Optional[List[str]]
) -> None:
    """A filter never activates either deletion path through delete()."""
    with patch.object(store.index, "drop_keys", create=True) as drop_keys:
        with patch.object(store, "delete_by_filter") as delete_by_filter:
            with pytest.raises(ValueError, match="delete_by_filter"):
                store.delete(ids=ids, filter=USER_FILTER)

    drop_keys.assert_not_called()
    delete_by_filter.assert_not_called()


def test_direct_id_operations_reject_foreign_index_records(
    store: RedisVectorStore,
) -> None:
    """Direct reads and deletes enforce the top-level ownership marker."""
    fake = _fake(store)
    own_marker = hashify(fake.name)
    ids = ["owned", "foreign", "unmarked", "missing"]
    records = [
        {
            "text": "owned document",
            "_index_name": own_marker,
            "_metadata_json": "{}",
        },
        {
            "text": "foreign document",
            "_index_name": hashify("sibling-index"),
            "_metadata_json": json.dumps({"_index_name": own_marker}),
        },
        {"text": "unmarked document", "_metadata_json": "{}"},
        None,
    ]

    with patch.object(store, "_fetch_records_by_keys", return_value=records):
        assert [doc.id for doc in store.get_by_ids(ids)] == ["owned"]
        with patch.object(fake, "drop_keys", return_value=1, create=True) as drop_keys:
            assert store.delete(ids) is True

    drop_keys.assert_called_once_with([store._redis_keys(["owned"])[0]])


@pytest.mark.parametrize(
    ("field_type", "record_marker"),
    [
        pytest.param(FieldTypes.TAG, hashify(INDEX_NAME), id="tag"),
        pytest.param(FieldTypes.TEXT, INDEX_NAME, id="legacy-text"),
    ],
)
def test_direct_id_operations_preserve_marker_schema_compatibility(
    store: RedisVectorStore,
    field_type: FieldTypes,
    record_marker: str,
) -> None:
    """TAG and legacy TEXT markers support exact direct-ID ownership checks."""
    fake = _fake(store)
    fake.schema.fields["_index_name"].type = field_type
    record = {
        "text": "document",
        "_index_name": record_marker,
        "_metadata_json": "{}",
    }

    with patch.object(store, "_fetch_records_by_keys", return_value=[record]):
        assert store.add_texts(["updated document"], keys=["doc"]) == ["doc"]
        assert [doc.id for doc in store.get_by_ids(["doc"])] == ["doc"]
        with patch.object(fake, "drop_keys", return_value=1, create=True):
            assert store.delete(["doc"]) is True


@pytest.mark.parametrize(
    "records",
    [
        pytest.param(
            [{"_index_name": hashify("sibling-index")}],
            id="foreign-owner",
        ),
        pytest.param([{}], id="unmarked-existing-record"),
        pytest.param(
            [
                None,
                {"_index_name": hashify(INDEX_NAME)},
                {"_index_name": hashify("sibling-index")},
            ],
            id="mixed-batch",
        ),
    ],
)
def test_add_texts_rejects_foreign_existing_key_before_load(
    store: RedisVectorStore,
    records: List[Optional[Dict[str, Any]]],
) -> None:
    """No explicit-key records are written when any owner is foreign."""
    fake = _fake(store)
    keys = [f"doc-{index}" for index in range(len(records))]

    with patch.object(store, "_fetch_records_by_keys", return_value=records):
        with patch.object(fake, "load", wraps=fake.load) as load:
            with pytest.raises(ValueError, match="ownership could not be verified"):
                store.add_texts(["document"] * len(keys), keys=keys)

    load.assert_not_called()


@pytest.mark.parametrize(
    "field_type",
    [pytest.param(None, id="missing"), pytest.param(FieldTypes.NUMERIC, id="numeric")],
)
def test_add_texts_preserves_markerless_schema_behavior(
    store: RedisVectorStore,
    field_type: Optional[FieldTypes],
) -> None:
    """Schemas without a supported ownership marker retain explicit writes."""
    fake = _fake(store)
    if field_type is None:
        fake.schema.fields.pop("_index_name")
    else:
        fake.schema.fields["_index_name"].type = field_type

    with patch.object(store, "_fetch_records_by_keys") as fetch_records:
        assert store.add_texts(["document"], keys=["doc"]) == ["doc"]

    fetch_records.assert_not_called()


def test_add_texts_uses_live_marker_schema_when_local_schema_is_stale(
    store: RedisVectorStore,
) -> None:
    """A stale markerless schema cannot bypass live ownership checks."""
    fake = _fake(store)
    fake.schema.fields.pop("_index_name")
    FakeBulkIndex.live_schema = _schema_with_index_marker(FieldTypes.TAG)

    with patch.object(
        store,
        "_fetch_records_by_keys",
        return_value=[{"_index_name": hashify("sibling-index")}],
    ):
        with patch.object(fake, "load", wraps=fake.load) as load:
            with pytest.raises(ValueError, match="ownership could not be verified"):
                store.add_texts(["document"], keys=["doc"])

    load.assert_not_called()


@pytest.mark.parametrize(
    "field_type",
    [pytest.param(None, id="missing"), pytest.param(FieldTypes.NUMERIC, id="numeric")],
)
def test_direct_id_operations_reject_unverifiable_marker_schema(
    store: RedisVectorStore,
    field_type: Optional[FieldTypes],
) -> None:
    """Missing or unsupported markers cannot authorize direct-ID access."""
    fake = _fake(store)
    if field_type is None:
        fake.schema.fields.pop("_index_name")
    else:
        fake.schema.fields["_index_name"].type = field_type

    with patch.object(store, "_fetch_records_by_keys") as fetch_records:
        with patch.object(fake, "drop_keys", create=True) as drop_keys:
            with pytest.raises(ValueError, match="requires an '_index_name'"):
                store.get_by_ids(["doc"])
            with pytest.raises(ValueError, match="requires an '_index_name'"):
                store.delete(["doc"])

    fetch_records.assert_not_called()
    drop_keys.assert_not_called()


def test_direct_id_operations_fail_closed_on_schema_error(
    store: RedisVectorStore,
) -> None:
    """Schema inspection failures cannot authorize direct-ID access."""
    fake = _fake(store)
    fake.schema = BrokenSchema()

    with patch.object(store, "_fetch_records_by_keys") as fetch_records:
        with patch.object(fake, "drop_keys", create=True) as drop_keys:
            with pytest.raises(ValueError, match="could not inspect the index schema"):
                store.get_by_ids(["doc"])
            with pytest.raises(ValueError, match="could not inspect the index schema"):
                store.delete(["doc"])

    fetch_records.assert_not_called()
    drop_keys.assert_not_called()


def test_empty_direct_id_read_remains_a_noop(store: RedisVectorStore) -> None:
    """An empty ID collection needs no ownership decision."""
    _fake(store).schema = BrokenSchema()

    with patch.object(store, "_fetch_records_by_keys") as fetch_records:
        assert store.get_by_ids([]) == []

    fetch_records.assert_not_called()


def test_fetch_records_by_keys_uses_single_key_json_commands(
    store: RedisVectorStore,
) -> None:
    """JSON ownership reads remain safe across Redis Cluster hash slots."""
    client = MagicMock()
    pipe = client.pipeline.return_value.__enter__.return_value
    json_pipe = pipe.json.return_value
    pipe.execute.return_value = [
        {b"text": b"owned", b"_index_name": b"marker"},
        None,
    ]
    _fake(store).client = client
    store.config.storage_type = StorageType.JSON.value

    assert store._fetch_records_by_keys(["prefix:a", "prefix:b"]) == [
        {"text": "owned", "_index_name": "marker"},
        None,
    ]
    client.pipeline.assert_called_once_with(transaction=False)
    assert json_pipe.get.call_args_list == [
        call("prefix:a", "."),
        call("prefix:b", "."),
    ]
    json_pipe.mget.assert_not_called()


def test_fetch_records_by_keys_uses_non_transactional_hash_pipeline(
    store: RedisVectorStore,
) -> None:
    """HASH ownership reads remain safe across Redis Cluster hash slots."""
    client = MagicMock()
    pipe = client.pipeline.return_value.__enter__.return_value
    pipe.execute.return_value = [{b"text": b"owned"}, {}]
    _fake(store).client = client

    assert store._fetch_records_by_keys(["prefix:a", "prefix:b"]) == [
        {"text": "owned"},
        None,
    ]
    client.pipeline.assert_called_once_with(transaction=False)
    assert pipe.hgetall.call_args_list == [call("prefix:a"), call("prefix:b")]


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


def test_index_marker_uses_search_index_name(
    store: RedisVectorStore,
) -> None:
    """Ownership follows RedisVL's live index identity, not stale config."""
    fake = _fake(store)
    fake.name = "live-index-name"

    store.delete_by_filter(USER_FILTER)

    captured = str(fake.captured_filter)
    assert hashify(fake.name) in captured
    assert hashify(INDEX_NAME) not in captured
    assert store._index_name_value(FieldTypes.TEXT) == fake.name


def test_legacy_text_write_ownership_ignores_stale_config(
    store: RedisVectorStore,
) -> None:
    """Stale config cannot authorize a write to another live index."""
    fake = _fake(store)
    fake.name = "live-index-name"
    fake.schema.fields["_index_name"].type = FieldTypes.TEXT
    store.config.index_name = "sibling-index"

    with patch.object(
        store,
        "_fetch_records_by_keys",
        return_value=[{"_index_name": "sibling-index"}],
    ):
        with patch.object(fake, "load", wraps=fake.load) as load:
            with pytest.raises(ValueError, match="ownership could not be verified"):
                store.add_texts(["document"], keys=["doc"])

    load.assert_not_called()


def test_add_texts_protects_tag_index_marker_from_metadata(
    store: RedisVectorStore,
) -> None:
    """Caller metadata cannot replace the indexed ownership marker."""
    fake = _fake(store)
    fake.name = "live-index-name"
    caller_marker = hashify("sibling-index")

    with patch.object(store, "_fetch_records_by_keys", return_value=[None]):
        store.add_texts(
            ["protected document"],
            metadatas=[{"_index_name": caller_marker, TEAM_FIELD: TEAM_VALUE}],
            keys=["protected"],
        )

    record = fake.loaded_records[0]
    assert record["_index_name"] == hashify(fake.name)
    assert json.loads(record["_metadata_json"])["_index_name"] == caller_marker


def test_delete_by_filter_dry_run_returns_bulk_result(
    store: RedisVectorStore,
) -> None:
    """dry_run=True preserves RedisVL's complete result."""
    expected = BulkResult(matched=7, processed=7, dry_run=True)
    FakeBulkIndex.bulk_result = expected

    result = store.delete_by_filter(USER_FILTER, dry_run=True)

    assert result is expected
    assert result.matched == 7
    assert result.processed == 7
    assert result.completed is True
    assert result.dry_run is True
    assert _fake(store).captured_kwargs["dry_run"] is True


def test_delete_by_filter_preserves_incomplete_bulk_result(
    store: RedisVectorStore,
) -> None:
    """Callers can detect when RedisVL stops before deleting every match."""
    expected = BulkResult(matched=10, processed=6, completed=False)
    FakeBulkIndex.bulk_result = expected

    result = store.delete_by_filter(USER_FILTER)

    assert result is expected
    assert result.matched == 10
    assert result.processed == 6
    assert result.completed is False
    assert result.dry_run is False


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

    result = store.delete_by_filter(USER_FILTER)
    assert result.processed == 3
    assert result.completed is True
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


def test_read_filter_rejects_existing_text_index_marker(
    store: RedisVectorStore,
) -> None:
    """Tokenized TEXT markers cannot provide exact search isolation."""
    _fake(store).schema.fields["_index_name"].type = FieldTypes.TEXT

    with pytest.raises(ValueError, match="requires an exact '_index_name' TAG"):
        store._with_index_name_filter(USER_FILTER)


@pytest.mark.parametrize(
    "marker_type",
    [pytest.param(None, id="missing"), pytest.param(FieldTypes.NUMERIC, id="numeric")],
)
def test_read_filter_rejects_missing_or_unsupported_index_marker(
    store: RedisVectorStore,
    marker_type: Optional[FieldTypes],
) -> None:
    """Searches fail closed when the schema cannot express ownership."""
    fake = _fake(store)
    if marker_type is None:
        fake.schema.fields.pop("_index_name")
    else:
        fake.schema.fields["_index_name"].type = marker_type

    with pytest.raises(ValueError, match="requires an exact '_index_name' TAG"):
        store._with_index_name_filter(USER_FILTER)


def test_read_filter_rejects_schema_inspection_failure(
    store: RedisVectorStore,
) -> None:
    """Schema inspection failures cannot silently remove search scoping."""
    _fake(store).schema = BrokenSchema()

    with pytest.raises(ValueError, match="could not inspect the index schema"):
        store._with_index_name_filter(USER_FILTER)


def test_read_filter_scopes_raw_string(store: RedisVectorStore) -> None:
    """A complete raw-string expression is grouped before ownership scoping."""
    raw_filter = "@team:{alpha}|@team:{beta}"

    scoped = str(store._with_index_name_filter(raw_filter))

    assert f"({raw_filter})" in scoped
    assert hashify(INDEX_NAME) in scoped


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
