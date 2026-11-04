from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest
from redisvl.schema import IndexSchema  # type: ignore[import]

from langchain_redis.migrations import (
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


@pytest.mark.parametrize(
    "attribute",
    [
        ["identifier", "text", "attribute", "display", "type", "TEXT"],
        {"identifier": "text", "attribute": "display", "type": "TEXT"},
    ],
    ids=["list-response", "mapping-response"],
)
def test_target_schema_preserves_hash_field_alias(attribute: Any) -> None:
    source = _schema({"name": "_index_name", "type": "text"})

    target = _target_schema(source, [attribute])

    field = target.fields["display"]
    assert (field.path, field.name) == ("text", "display")
    assert field.attrs == source.fields["text"].attrs


@pytest.mark.parametrize(
    ("section", "changes"),
    [
        pytest.param("index", {"name": "other"}, id="name"),
        pytest.param("index", {"prefix": "other:"}, id="prefix"),
        pytest.param("index", {"storage_type": "json"}, id="storage"),
        pytest.param("index", {"stopwords": ["custom"]}, id="stopwords"),
        pytest.param("field", {"name": "other", "type": "text"}, id="fields"),
        pytest.param(
            "field",
            {"name": "text", "type": "text", "attrs": {"weight": 2}},
            id="field-options",
        ),
    ],
)
def test_source_schema_must_match_live_definition(
    section: str, changes: Dict[str, Any]
) -> None:
    live = _target_schema(_schema({"name": "_index_name", "type": "text"}))
    definition = live.to_dict()
    if section == "index":
        definition["index"].update(changes)
    else:
        definition["fields"][0] = changes

    with pytest.raises(ValueError, match="source_schema does not match"):
        _validate_source_schema(live, IndexSchema.from_dict(definition))


@pytest.mark.parametrize("becomes_ready", [True, False], ids=["ready", "timeout"])
def test_backfill_waits_for_index_readiness(
    monkeypatch: pytest.MonkeyPatch, becomes_ready: bool
) -> None:
    index = Mock(schema=_schema(None))
    building = {"indexing": 1, "percent_indexed": 0.5}
    ready = {"indexing": 0, "percent_indexed": 1}
    index.info.side_effect = [
        {"index_name": INDEX_NAME, "percent_indexed": 1, "attributes": []},
        *([building, ready] if becomes_ready else [building] * 60),
    ]
    index.update_by_filter.return_value.processed = 1
    client = Mock()
    client.pipeline.return_value.execute.return_value = [None]
    sleep = Mock()
    monkeypatch.setattr(
        "langchain_redis.migrations.SearchIndex.from_existing", lambda **kwargs: index
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
