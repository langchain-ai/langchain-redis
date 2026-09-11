"""Unit tests for multi-prefix indexes and index-level stopwords."""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings
from redisvl.redis.utils import hashify  # type: ignore[import]
from redisvl.schema import FieldTypes, IndexSchema, StorageType  # type: ignore[import]

from langchain_redis import RedisConfig, RedisVectorStore

DIMS = 4
INDEX_NAME = "prefix_unit"
REDIS_URL = "redis://localhost"
PREFIX_A = "tenant_a"
PREFIX_B = "tenant_b"
PREFIXES = [PREFIX_A, PREFIX_B]
CUSTOM_KEY_SEPARATOR = "|"
CUSTOM_STOPWORDS = ["the", "a"]


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * DIMS for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * DIMS


class CapturingIndex:
    """Records the schema dict and the keys passed to drop_keys."""

    last_schema: Optional[Dict[str, Any]] = None
    last_instance: Optional["CapturingIndex"] = None

    def __init__(self) -> None:
        self.dropped_keys: List[str] = []
        self.name = INDEX_NAME
        self.client = object()

    @classmethod
    def from_dict(cls, schema: Dict[str, Any], **kwargs: Any) -> "CapturingIndex":
        cls.last_schema = schema
        instance = cls()
        field_specs = schema.get("fields", [])
        instance.schema = SimpleNamespace(  # type: ignore[attr-defined]
            fields={
                spec["name"]: SimpleNamespace(
                    name=spec["name"], type=FieldTypes(spec["type"])
                )
                for spec in field_specs
            }
        )
        cls.last_instance = instance
        return instance

    @classmethod
    def from_existing(cls, name: str, **kwargs: Any) -> "CapturingIndex":
        assert cls.last_instance is not None
        return cls.last_instance

    def create(self, overwrite: bool = False) -> None:
        pass

    def key(self, id_: str) -> str:
        assert self.last_schema is not None
        prefix = self.last_schema["index"]["prefix"]
        if isinstance(prefix, list):
            prefix = prefix[0]
        return f"{prefix.rstrip(':')}:{id_}"

    def drop_keys(self, keys: List[str]) -> int:
        self.dropped_keys = keys
        return len(keys)


def _make_store(**config_kwargs: Any) -> RedisVectorStore:
    with patch("langchain_redis.vectorstores.SearchIndex", CapturingIndex):
        return RedisVectorStore(
            MockEmbeddings(),
            index_name=INDEX_NAME,
            redis_url=REDIS_URL,
            **config_kwargs,
        )


@pytest.mark.parametrize(
    "key_prefix,expected",
    [(PREFIXES, PREFIX_A), (PREFIX_A, PREFIX_A), (None, INDEX_NAME)],
    ids=["list", "string", "default"],
)
def test_primary_prefix_for_list_string_and_default(
    key_prefix: Optional[Union[str, List[str]]], expected: str
) -> None:
    """primary_prefix is the first list element, the string itself, or the
    index name when no prefix was configured."""
    config = RedisConfig(
        index_name=INDEX_NAME, key_prefix=key_prefix, embedding_dimensions=DIMS
    )
    assert config.primary_prefix == expected


def test_custom_key_separator_reaches_generated_schema() -> None:
    """Custom separators bypass the colon-only legacy prefix format."""
    store = _make_store(
        key_prefix=PREFIX_A,
        key_separator=CUSTOM_KEY_SEPARATOR,
    )

    assert store.config.key_separator == CUSTOM_KEY_SEPARATOR
    assert CapturingIndex.last_schema is not None
    assert CapturingIndex.last_schema["index"]["prefix"] == PREFIX_A
    assert CapturingIndex.last_schema["index"]["key_separator"] == CUSTOM_KEY_SEPARATOR


def test_redisvl_schema_key_separator_is_authoritative() -> None:
    """A complete RedisVL schema does not require duplicate configuration."""
    schema = IndexSchema.from_dict(
        {
            "index": {
                "name": INDEX_NAME,
                "prefix": PREFIX_A,
                "key_separator": CUSTOM_KEY_SEPARATOR,
            },
            "fields": [{"name": "text", "type": "text"}],
        }
    )

    config = RedisConfig(
        schema=schema,
        key_separator="/",
        embedding_dimensions=DIMS,
    )

    assert config.key_separator == CUSTOM_KEY_SEPARATOR


def test_generated_schema_preserves_list_prefix() -> None:
    """A prefix list reaches the generated schema as a list, spanning all
    namespaces at query time."""
    config = RedisConfig(key_prefix=PREFIXES, embedding_dimensions=DIMS)
    schema_index = config.to_index_schema().to_dict()["index"]
    assert schema_index["prefix"] == PREFIXES
    assert schema_index["key_separator"] == ":"


def test_schema_list_prefix_uses_first_prefix_for_keys() -> None:
    """Configs built from a multi-prefix redisvl schema construct keys from
    the first prefix.

    Regression: this used to stringify the whole list into keys like
    "['a', 'b']:doc_id".
    """
    schema = IndexSchema.from_dict(
        {
            "index": {"name": INDEX_NAME, "prefix": PREFIXES},
            "fields": [{"name": "text", "type": "text"}],
        }
    )
    config = RedisConfig(schema=schema, embedding_dimensions=DIMS)
    assert config.primary_prefix == PREFIX_A


@pytest.mark.parametrize(
    ("live_prefix", "legacy_key_format", "expected_prefix"),
    [
        pytest.param("live:", True, "live", id="legacy"),
        pytest.param("live", False, "live", id="modern"),
        pytest.param(["live_a:", "live_b:"], True, ["live_a", "live_b"], id="list"),
    ],
)
def test_live_schema_settings_replace_stale_config(
    live_prefix: Union[str, List[str]],
    legacy_key_format: bool,
    expected_prefix: Union[str, List[str]],
) -> None:
    """Storage and prefix settings come from the schema Redis retained."""
    schema = IndexSchema.from_dict(
        {
            "index": {
                "name": INDEX_NAME,
                "prefix": live_prefix,
                "storage_type": "json",
            },
            "fields": [{"name": "text", "type": "text"}],
        }
    )
    store = object.__new__(RedisVectorStore)
    store.config = RedisConfig(
        index_name=INDEX_NAME,
        key_prefix="stale",
        storage_type="hash",
        embedding_dimensions=DIMS,
        legacy_key_format=legacy_key_format,
    )
    store._index = SimpleNamespace(schema=schema)

    store._sync_config_with_live_index()

    assert store.config.storage_type == StorageType.JSON.value
    assert store.config.key_prefix == expected_prefix


@pytest.mark.parametrize(
    "legacy_key_format,expected_prefix",
    [
        (False, PREFIXES),
        (True, [f"{PREFIX_A}:", f"{PREFIX_B}:"]),
    ],
    ids=["modern-prefix", "legacy-prefix"],
)
def test_inline_schema_formats_list_prefixes(
    legacy_key_format: bool, expected_prefix: List[str]
) -> None:
    """The schema spans every prefix while id operations use the primary one."""
    store = _make_store(key_prefix=PREFIXES, legacy_key_format=legacy_key_format)

    assert CapturingIndex.last_schema is not None
    assert CapturingIndex.last_schema["index"]["prefix"] == expected_prefix

    with patch.object(
        store,
        "_fetch_records_by_keys",
        return_value=[{"_index_name": hashify(store.index.name)}],
    ):
        store.delete(ids=["doc1"])
    assert store.index.dropped_keys == [f"{PREFIX_A}:doc1"]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "stopwords,expected",
    [([], []), (CUSTOM_STOPWORDS, CUSTOM_STOPWORDS), (None, None)],
    ids=["disabled", "custom", "server-default"],
)
def test_stopword_states_in_generated_schema(
    stopwords: Optional[List[str]], expected: Optional[List[str]]
) -> None:
    """None keeps server defaults, [] disables (STOPWORDS 0), a list replaces."""
    config = RedisConfig(embedding_dimensions=DIMS, stopwords=stopwords)
    schema = config.to_index_schema()
    assert schema.index.stopwords == expected


def test_inline_schema_forwards_stopwords() -> None:
    """The store's inline schema path forwards stopwords like to_index_schema."""
    _make_store(stopwords=[])
    assert CapturingIndex.last_schema is not None
    assert CapturingIndex.last_schema["index"]["stopwords"] == []
