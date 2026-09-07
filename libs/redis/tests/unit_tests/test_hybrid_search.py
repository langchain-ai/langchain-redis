"""Unit tests for hybrid and full-text search on RedisVectorStore."""

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings
from redisvl.query import (  # type: ignore[import]
    AggregateHybridQuery,
    HybridQuery,
    TextQuery,
)
from redisvl.query.filter import Tag  # type: ignore[import]
from redisvl.redis.utils import hashify  # type: ignore[import]
from redisvl.schema import FieldTypes  # type: ignore[import]

from langchain_redis import RedisVectorStore

QUERY = "hello"
INDEX_NAME = "hybrid_unit"
REDIS_URL = "redis://localhost"
CATEGORY_FIELD = "category"
CATEGORY_VALUE = "pets"
METADATA = {CATEGORY_FIELD: CATEGORY_VALUE}
HYBRID_SCORE = 0.75


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]


class FakeSearchIndex:
    """Captures the query object passed to query() and returns canned rows."""

    redis_version = "8.4.0"
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
            info=lambda section=None: {"redis_version": type(self).redis_version}
        )
        self.captured_query: Any = None
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
        self.captured_query = query
        return list(type(self).rows)


@pytest.fixture
def store() -> RedisVectorStore:
    with patch("langchain_redis.vectorstores.SearchIndex", FakeSearchIndex):
        return RedisVectorStore(
            MockEmbeddings(), index_name=INDEX_NAME, redis_url=REDIS_URL
        )


@pytest.fixture(autouse=True)
def reset_fake_index() -> None:
    FakeSearchIndex.redis_version = "8.4.0"
    FakeSearchIndex.rows = [
        {
            "text": QUERY,
            "_metadata_json": json.dumps(METADATA),
            "hybrid_score": str(HYBRID_SCORE),
            "text_score": "0.5",
            "vector_similarity": "0.9",
        }
    ]


def _captured(store: RedisVectorStore) -> Any:
    return store.index.captured_query  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "method,expected_query_class",
    [("ft_hybrid", HybridQuery), ("aggregate", AggregateHybridQuery)],
)
def test_explicit_method_builds_matching_query(
    store: RedisVectorStore, method: str, expected_query_class: type
) -> None:
    """An explicit method= dispatches to the matching redisvl query class."""
    store.hybrid_search(QUERY, method=method)
    assert isinstance(_captured(store), expected_query_class)


@pytest.mark.parametrize(
    "server_version,expected_query_class,expected_kwargs",
    [
        (
            "8.4.0",
            HybridQuery,
            {"combination_method": "LINEAR", "linear_alpha": pytest.approx(0.3)},
        ),
        ("8.2.1", AggregateHybridQuery, {"alpha": 0.7}),
    ],
)
def test_auto_uses_linear_fusion_across_engines(
    store: RedisVectorStore,
    server_version: str,
    expected_query_class: type,
    expected_kwargs: Dict[str, Any],
) -> None:
    """The default call preserves LINEAR semantics across both engines."""
    FakeSearchIndex.redis_version = server_version
    target = f"langchain_redis.vectorstores.{expected_query_class.__name__}"
    with patch(target, wraps=expected_query_class) as query_class:
        store.hybrid_search(QUERY)

    assert isinstance(_captured(store), expected_query_class)
    for name, expected in expected_kwargs.items():
        assert query_class.call_args.kwargs[name] == expected


def test_hybrid_options_are_case_insensitive(store: RedisVectorStore) -> None:
    """Engine and fusion names are normalized before reaching RedisVL."""
    with patch(
        "langchain_redis.vectorstores.HybridQuery", wraps=HybridQuery
    ) as query_class:
        store.hybrid_search(
            QUERY,
            method="FT_HYBRID",
            combination_method="linear",
            alpha=0.5,
        )

    assert query_class.call_args.kwargs["combination_method"] == "LINEAR"
    assert query_class.call_args.kwargs["linear_alpha"] == 0.5


def test_rrf_omits_linear_alpha(store: RedisVectorStore) -> None:
    """RRF reaches RedisVL without a meaningless LINEAR weight."""
    with patch(
        "langchain_redis.vectorstores.HybridQuery", wraps=HybridQuery
    ) as query_class:
        store.hybrid_search(QUERY, method="ft_hybrid", combination_method="rrf")

    assert query_class.call_args.kwargs["combination_method"] == "RRF"
    assert "linear_alpha" not in query_class.call_args.kwargs


def test_explicit_ft_hybrid_on_old_server_raises(store: RedisVectorStore) -> None:
    """Explicit method='ft_hybrid' on a pre-8.4 server raises a clear ValueError."""
    FakeSearchIndex.redis_version = "8.0.0"
    with pytest.raises(ValueError, match="8.4"):
        store.hybrid_search(QUERY, method="ft_hybrid")


def test_unknown_method_raises(store: RedisVectorStore) -> None:
    """An unrecognized method raises ValueError instead of silently defaulting."""
    with pytest.raises(ValueError, match="Unknown hybrid search method"):
        store.hybrid_search(QUERY, method="bm42")


def test_unknown_combination_method_raises_before_embedding(
    store: RedisVectorStore,
) -> None:
    """Invalid fusion methods fail locally before doing embedding work."""
    with patch.object(store.embeddings, "embed_query") as embed_query:
        with pytest.raises(ValueError, match="Unknown combination method"):
            store.hybrid_search(QUERY, combination_method="weighted")

    embed_query.assert_not_called()


@pytest.mark.parametrize(
    "alpha",
    [-1.0, 0.0, 1.0, 2.0, pytest.param(float("nan"), id="nan")],
)
def test_linear_rejects_invalid_alpha(store: RedisVectorStore, alpha: float) -> None:
    """LINEAR weights must keep both text and vector signals active."""
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        store.hybrid_search(QUERY, combination_method="LINEAR", alpha=alpha)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_rrf_rejects_explicit_alpha(store: RedisVectorStore, alpha: float) -> None:
    """RRF rejects alpha even when the supplied value is falsy."""
    with pytest.raises(ValueError, match="alpha cannot be used with RRF"):
        store.hybrid_search(
            QUERY,
            method="ft_hybrid",
            combination_method="RRF",
            alpha=alpha,
        )


@pytest.mark.parametrize("method", ["auto", "aggregate"])
def test_rrf_rejects_aggregate_engine(store: RedisVectorStore, method: str) -> None:
    """RRF never silently changes to LINEAR through the aggregate fallback."""
    FakeSearchIndex.redis_version = "8.2.0"
    with pytest.raises(ValueError, match="RRF"):
        store.hybrid_search(QUERY, method=method, combination_method="RRF")


def test_index_name_filter_injected_when_field_exists(
    store: RedisVectorStore,
) -> None:
    """User filters are AND-combined with the _index_name shared-prefix guard.

    The default schema includes _index_name; the guard must be applied both
    when a user filter is given and when no filter is passed at all.
    """
    combined = store._with_index_name_filter(Tag(CATEGORY_FIELD) == CATEGORY_VALUE)
    assert "_index_name" in str(combined)
    assert hashify(INDEX_NAME) in str(combined)
    assert CATEGORY_FIELD in str(combined)

    default = store._with_index_name_filter(None)
    assert "_index_name" in str(default)


def test_hybrid_search_with_score_extracts_hybrid_score(
    store: RedisVectorStore,
) -> None:
    """The unified hybrid_score field is parsed to a float and paired per doc."""
    results = store.hybrid_search_with_score(QUERY)
    assert len(results) == 1
    doc, score = results[0]
    assert score == HYBRID_SCORE
    assert doc.page_content == QUERY
    assert doc.metadata == METADATA


def test_score_fields_do_not_leak_into_metadata(store: RedisVectorStore) -> None:
    """Score fields are stripped from rows before metadata extraction.

    When a row has no _metadata_json, metadata is scraped from the remaining
    row fields — the hybrid score fields must not end up in it.
    """
    FakeSearchIndex.rows = [
        {"text": QUERY, CATEGORY_FIELD: CATEGORY_VALUE, "hybrid_score": "0.4"}
    ]
    [(doc, score)] = store.hybrid_search_with_score(QUERY)
    assert score == 0.4
    assert doc.metadata == METADATA


def test_full_text_search_builds_text_query(store: RedisVectorStore) -> None:
    """full_text_search dispatches to TextQuery, accepting weighted fields."""
    docs = store.full_text_search(QUERY, text_fields={"title": 5.0, "text": 1.0})
    assert isinstance(_captured(store), TextQuery)
    assert docs[0].page_content == QUERY
