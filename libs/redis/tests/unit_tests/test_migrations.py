from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest
from redisvl.schema import IndexSchema  # type: ignore[import]

from langchain_redis.migrations import (
    _assert_no_overlapping_prefixes,
    _schema_action,
    _target_schema,
    _validate_source_schema,
    migrate_index_ownership,
)

INDEX_NAME = "migration-test"


def _schema(
    marker: Optional[Dict[str, Any]], storage_type: str = "hash"
) -> IndexSchema:
    fields: List[Dict[str, Any]] = [{"name": "text", "type": "text"}]
    if marker is not None:
        fields.append(marker)
    return IndexSchema.from_dict(
        {
            "index": {
                "name": INDEX_NAME,
                "prefix": "migration:",
                "storage_type": storage_type,
            },
            "fields": fields,
        }
    )


def _attribute(
    identifier: str,
    field_type: str,
    *options: Any,
    alias: Optional[str] = None,
) -> List[Any]:
    return [
        "identifier",
        identifier,
        "attribute",
        alias or identifier,
        "type",
        field_type,
        *options,
    ]


def _info(*attributes: List[Any]) -> Dict[str, Any]:
    return {
        "index_name": INDEX_NAME,
        "index_definition": {"key_type": "HASH", "prefixes": ["migration:"]},
        "attributes": [
            _attribute("text", "TEXT", "WEIGHT", 1.0),
            *attributes,
            _attribute("_index_name", "TEXT", "WEIGHT", 1.0),
        ],
    }


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        pytest.param(None, "alter", id="missing"),
        pytest.param({"name": "_index_name", "type": "text"}, "rebuild", id="text"),
        pytest.param({"name": "_index_name", "type": "tag"}, "rebuild", id="tag"),
        pytest.param(
            {
                "name": "_index_name",
                "type": "tag",
                "attrs": {"case_sensitive": True, "no_index": True},
            },
            "rebuild",
            id="unindexed-tag",
        ),
        pytest.param(
            {
                "name": "_index_name",
                "type": "tag",
                "attrs": {"case_sensitive": True},
            },
            "none",
            id="compatible-tag",
        ),
    ],
)
def test_schema_action(marker: Optional[Dict[str, Any]], expected: str) -> None:
    assert _schema_action(_schema(marker), INDEX_NAME) == expected


def test_json_marker_must_use_the_root_ownership_path() -> None:
    schema = _schema(
        {
            "name": "_index_name",
            "path": "$.metadata.owner",
            "type": "tag",
            "attrs": {"case_sensitive": True},
        },
        storage_type="json",
    )

    with pytest.raises(ValueError, match="Nonstandard"):
        _schema_action(schema, INDEX_NAME)


def test_target_schema_replaces_legacy_marker_without_changing_other_fields() -> None:
    source = _schema({"name": "_index_name", "type": "text"}, "json")

    target = _target_schema(source)

    assert target.index == source.index
    assert target.fields["text"] == source.fields["text"]
    marker = target.fields["_index_name"]
    assert marker.type.value == "tag"
    assert marker.attrs.case_sensitive is True


def test_target_schema_preserves_declared_hash_field_alias() -> None:
    source = _schema({"name": "_index_name", "type": "text"})
    field = source.fields.pop("text")
    field.name, field.path = "display", "text"
    source.fields["display"] = field

    target = _target_schema(source)

    assert target.fields["display"] == field
    assert target.fields["display"] is not field


@pytest.mark.parametrize(
    "mismatch",
    ["missing", "extra", "alias", "type", "field-options", "stopwords"],
)
def test_source_schema_must_match_reliable_live_definition(mismatch: str) -> None:
    source = _schema({"name": "_index_name", "type": "text"})
    info = _info()
    if mismatch == "missing":
        source.fields.pop("text")
    elif mismatch == "extra":
        source.add_field({"name": "extra", "type": "text"})
    elif mismatch == "alias":
        field = source.fields.pop("text")
        field.name, field.path = "display", "text"
        source.fields["display"] = field
    elif mismatch == "type":
        source.fields.pop("text")
        source.add_field({"name": "text", "type": "tag"})
    elif mismatch == "field-options":
        source.fields["text"].attrs.weight = 2
    else:
        source.index.stopwords = ["custom"]
        info["stopwords_list"] = ["different"]

    with pytest.raises(ValueError, match="source_schema does not match"):
        _validate_source_schema(info, source)


def test_source_schema_vector_tuning_remains_authoritative() -> None:
    source = _schema({"name": "_index_name", "type": "text"})
    source.add_field(
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "algorithm": "hnsw",
                "dims": 4,
                "datatype": "float32",
                "distance_metric": "cosine",
                "ef_runtime": 75,
                "epsilon": 0.2,
            },
        }
    )
    _validate_source_schema(_info(_attribute("embedding", "VECTOR")), source)


@pytest.mark.parametrize(
    ("target_prefix", "other_prefix", "key_type", "rejects"),
    [
        pytest.param("app:", "app:b:", "HASH", True, id="target-is-broader"),
        pytest.param("app:b:", "app:", "HASH", True, id="target-is-narrower"),
        pytest.param("app:", "app:", "HASH", True, id="identical"),
        pytest.param("app:a:", "app:b:", "HASH", False, id="disjoint"),
        pytest.param("app:", "app:b:", "JSON", False, id="different-storage"),
    ],
)
def test_overlapping_index_prefixes_are_rejected(
    target_prefix: str,
    other_prefix: str,
    key_type: str,
    rejects: bool,
) -> None:
    index = Mock()
    index.name = INDEX_NAME
    index.schema = _schema(None)
    index.schema.index.prefix = target_prefix
    index.listall.return_value = [INDEX_NAME, "other-index"]
    index.client.ft.return_value.info.return_value = {
        "index_definition": {"key_type": key_type, "prefixes": [other_prefix]}
    }

    if rejects:
        with pytest.raises(ValueError, match="overlaps"):
            _assert_no_overlapping_prefixes(index)
    else:
        _assert_no_overlapping_prefixes(index)


def test_resp2_index_definition_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = _info()
    info["index_definition"] = [
        "key_type",
        "HASH",
        "prefixes",
        ["migration:"],
    ]
    index = Mock()
    index.name = INDEX_NAME
    index.query.return_value = 0
    index.listall.return_value = [INDEX_NAME]
    client = Mock()
    client.ft.return_value.info.return_value = info
    monkeypatch.setattr(
        "langchain_redis.migrations.SearchIndex", lambda **kwargs: index
    )
    monkeypatch.setattr("langchain_redis.migrations._index_keys", lambda *args: [])

    result = migrate_index_ownership(
        INDEX_NAME,
        redis_client=client,
        source_schema=_schema({"name": "_index_name", "type": "text"}),
    )

    assert result.schema_action == "rebuild"


def test_preflight_rejects_incomplete_cursor_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = Mock()
    index.name = INDEX_NAME
    index.query.return_value = 2
    index.listall.return_value = [INDEX_NAME]
    client = Mock()
    client.ft.return_value.info.return_value = {
        "index_name": INDEX_NAME,
        "percent_indexed": 1,
        "index_definition": {"key_type": "HASH", "prefixes": ["migration:"]},
        "attributes": [],
    }
    monkeypatch.setattr(
        "langchain_redis.migrations.SearchIndex", lambda **kwargs: index
    )
    monkeypatch.setattr(
        "langchain_redis.migrations._index_keys", lambda *args: ["migration:one"]
    )

    with pytest.raises(RuntimeError, match="different number of documents"):
        migrate_index_ownership(INDEX_NAME, redis_client=client, dry_run=False)

    client.pipeline.assert_not_called()
    client.ft.return_value.alter_schema_add.assert_not_called()
    index.update_by_filter.assert_not_called()


@pytest.mark.parametrize("becomes_ready", [True, False], ids=["ready", "timeout"])
def test_backfill_waits_for_index_readiness(
    monkeypatch: pytest.MonkeyPatch, becomes_ready: bool
) -> None:
    index = Mock()
    index.name = INDEX_NAME
    index.query.return_value = 1
    index.listall.return_value = [INDEX_NAME]
    building = {"indexing": 1, "percent_indexed": 0.5}
    ready = {"indexing": 0, "percent_indexed": 1}
    index.info.side_effect = [building, ready] if becomes_ready else [building] * 60
    index.update_by_filter.return_value.processed = 1
    client = Mock()
    client.ft.return_value.info.return_value = {
        "index_name": INDEX_NAME,
        "percent_indexed": 1,
        "index_definition": {"key_type": "HASH", "prefixes": ["migration:"]},
        "attributes": [],
    }
    client.pipeline.return_value.execute.return_value = [None]
    sleep = Mock()
    monkeypatch.setattr(
        "langchain_redis.migrations.SearchIndex", lambda **kwargs: index
    )
    monkeypatch.setattr(
        "langchain_redis.migrations._index_keys", lambda *args: ["migration:one"]
    )
    monkeypatch.setattr("langchain_redis.migrations.sleep", sleep)

    if becomes_ready:
        result = migrate_index_ownership(INDEX_NAME, redis_client=client, dry_run=False)
        assert (result.updated, result.completed) == (1, True)
        index.update_by_filter.assert_called_once()
        sleep.assert_called_once_with(1)
    else:
        with pytest.raises(TimeoutError, match="still rebuilding"):
            migrate_index_ownership(INDEX_NAME, redis_client=client, dry_run=False)
        index.update_by_filter.assert_not_called()
        assert sleep.call_count == 60
