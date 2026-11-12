"""Unit tests for RedisConfig's one-schema-source rule with the `schema` alias."""

from typing import Any, Dict, List

import pytest
from redisvl.schema import IndexSchema  # type: ignore[import]

from langchain_redis import RedisConfig

INDEX_NAME = "schema_options_unit"
KEY_SEPARATOR = "-"
METADATA_SCHEMA: List[Dict[str, Any]] = [{"name": "color", "type": "tag"}]
SCHEMA_OPTIONS_ERROR = "Only one of 'index_schema', 'schema_path', or 'metadata_schema'"


def _schema() -> IndexSchema:
    return IndexSchema.from_dict(
        {
            "index": {
                "name": INDEX_NAME,
                "prefix": INDEX_NAME,
                "key_separator": KEY_SEPARATOR,
                "storage_type": "json",
            },
            "fields": [{"name": "text", "type": "text"}],
        }
    )


def test_schema_alias_with_metadata_schema_raises() -> None:
    with pytest.raises(ValueError, match=SCHEMA_OPTIONS_ERROR):
        RedisConfig(schema=_schema(), metadata_schema=METADATA_SCHEMA)


def test_from_schema_with_metadata_schema_raises() -> None:
    with pytest.raises(ValueError, match=SCHEMA_OPTIONS_ERROR):
        RedisConfig.from_schema(_schema(), metadata_schema=METADATA_SCHEMA)


def test_from_schema_with_empty_metadata_schema_raises() -> None:
    with pytest.raises(ValueError, match=SCHEMA_OPTIONS_ERROR):
        RedisConfig.from_schema(_schema(), metadata_schema=[])


def test_schema_alias_with_schema_path_raises() -> None:
    with pytest.raises(ValueError, match=SCHEMA_OPTIONS_ERROR):
        RedisConfig(schema=_schema(), schema_path="schema.yaml")


@pytest.mark.parametrize(
    "build",
    [
        lambda schema: RedisConfig(schema=schema),
        lambda schema: RedisConfig.from_schema(schema),
    ],
    ids=["schema_alias", "from_schema"],
)
def test_schema_alone_is_accepted(build: Any) -> None:
    schema = _schema()

    config = build(schema)

    assert config.index_schema is schema
    assert config.index_name == INDEX_NAME
    assert config.key_prefix == INDEX_NAME
    assert config.key_separator == KEY_SEPARATOR
    assert config.storage_type == "json"
    assert config.to_index_schema() is schema
