"""Unit tests for multi-prefix indexes and index-level stopwords."""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings
from redisvl.schema import IndexSchema  # type: ignore[import]

from langchain_redis import RedisConfig, RedisVectorStore

DIMS = 4
INDEX_NAME = "prefix_unit"
REDIS_URL = "redis://localhost"
PREFIX_A = "tenant_a"
PREFIX_B = "tenant_b"
PREFIXES = [PREFIX_A, PREFIX_B]
CUSTOM_STOPWORDS = ["the", "a"]


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * DIMS for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * DIMS


class CapturingIndex:
    """Records the schema dict and the keys passed to drop_keys."""

    last_schema: Optional[Dict[str, Any]] = None

    def __init__(self) -> None:
        self.dropped_keys: List[str] = []

    @classmethod
    def from_dict(cls, schema: Dict[str, Any], **kwargs: Any) -> "CapturingIndex":
        cls.last_schema = schema
        instance = cls()
        field_specs = schema.get("fields", [])
        instance.schema = SimpleNamespace(  # type: ignore[attr-defined]
            fields={
                spec["name"]: SimpleNamespace(name=spec["name"]) for spec in field_specs
            }
        )
        return instance

    def create(self, overwrite: bool = False) -> None:
        pass

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


def test_generated_schema_preserves_list_prefix() -> None:
    """A prefix list reaches the generated schema as a list, spanning all
    namespaces at query time."""
    config = RedisConfig(key_prefix=PREFIXES, embedding_dimensions=DIMS)
    schema_index = config.to_index_schema().to_dict()["index"]
    assert schema_index["prefix"] == PREFIXES


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
