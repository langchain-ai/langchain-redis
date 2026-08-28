"""Unit tests for vector-field tuning attributes (RedisConfig.vector_attrs)."""

from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError
from redisvl.schema import IndexSchema  # type: ignore[import]

from langchain_redis import RedisConfig, RedisVectorStore

DIMS = 4
HNSW_ATTRS = {"m": 8, "ef_construction": 100, "ef_runtime": 20}
SVS_ATTRS = {"graph_max_degree": 32, "search_window_size": 10}
CONFIG_OWNED_ATTR = "dims"

MINIMAL_INDEX_SCHEMA = IndexSchema.from_dict(
    {
        "index": {"name": "minimal", "prefix": "minimal"},
        "fields": [{"name": "text", "type": "text"}],
    }
)


def _generated_vector_attrs(config: RedisConfig) -> Dict[str, Any]:
    """The vector field attrs of the generated schema, as a plain dict."""
    schema_dict = config.to_index_schema().to_dict()
    (vector_field,) = [
        field for field in schema_dict["fields"] if field["type"] == "vector"
    ]
    return vector_field["attrs"]


def test_hnsw_attrs_land_in_generated_schema() -> None:
    """HNSW tuning attributes are merged into the generated vector field."""
    config = RedisConfig(
        embedding_dimensions=DIMS,
        indexing_algorithm="HNSW",
        vector_attrs=HNSW_ATTRS,
    )
    attrs = _generated_vector_attrs(config)
    assert attrs["dims"] == DIMS
    for name, value in HNSW_ATTRS.items():
        assert attrs[name] == value


def test_svs_vamana_attrs_with_compression_are_valid() -> None:
    """SVS-VAMANA attributes, including LeanVec compression, build a schema."""
    config = RedisConfig(
        embedding_dimensions=DIMS,
        indexing_algorithm="SVS-VAMANA",
        vector_attrs={**SVS_ATTRS, "compression": "LeanVec4x8", "reduce": 2},
    )
    attrs = _generated_vector_attrs(config)
    assert attrs["graph_max_degree"] == SVS_ATTRS["graph_max_degree"]
    # redisvl normalizes the compression enum to lowercase on serialization
    assert str(attrs["compression"]).lower() == "leanvec4x8"
    assert attrs["reduce"] == 2


def test_config_owned_attr_collision_raises() -> None:
    """vector_attrs may not override attrs owned by dedicated config fields."""
    with pytest.raises(ValidationError, match="cannot override"):
        RedisConfig(embedding_dimensions=DIMS, vector_attrs={CONFIG_OWNED_ATTR: 8})


@pytest.mark.parametrize(
    "schema_kwargs",
    [
        {"schema_path": "schema.yaml"},
        {"index_schema": MINIMAL_INDEX_SCHEMA},
        {"from_existing": True},
    ],
    ids=["schema_path", "index_schema", "from_existing"],
)
def test_vector_attrs_rejected_with_full_schema(schema_kwargs: Dict[str, Any]) -> None:
    """vector_attrs only applies when creating generated schemas."""
    with pytest.raises(ValidationError, match="vector_attrs"):
        RedisConfig(vector_attrs=HNSW_ATTRS, **schema_kwargs)


def test_invalid_svs_datatype_rejected_by_redisvl() -> None:
    """Attr value validation is redisvl's: SVS-VAMANA rejects FLOAT64."""
    config = RedisConfig(
        embedding_dimensions=DIMS,
        indexing_algorithm="SVS-VAMANA",
        vector_datatype="FLOAT64",
        vector_attrs=SVS_ATTRS,
    )
    with pytest.raises(Exception, match="FLOAT16"):
        config.to_index_schema()


def test_no_vector_attrs_leaves_schema_unchanged() -> None:
    """Without vector_attrs, no tuning attributes appear in the generated schema.

    redisvl's serialization may add its own defaults (e.g. index_missing), so
    this pins that the config-owned attrs are present and no tuning keys leak.
    """
    config = RedisConfig(embedding_dimensions=DIMS)
    attrs = _generated_vector_attrs(config)
    assert {"dims", "distance_metric", "algorithm", "datatype"} <= set(attrs)
    assert not set(HNSW_ATTRS) & set(attrs)


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * DIMS for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * DIMS


class SchemaCapturingIndex:
    """Records the schema dict the vector store builds for its index."""

    last_schema: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, schema: Dict[str, Any], **kwargs: Any) -> "SchemaCapturingIndex":
        cls.last_schema = schema
        return cls()

    def create(self, overwrite: bool = False) -> None:
        pass


def test_store_default_schema_path_merges_vector_attrs() -> None:
    """The store's inline schema construction also merges vector_attrs.

    RedisVectorStore builds its default schema in vectorstores.py, separately
    from RedisConfig.to_index_schema(); both paths must honor vector_attrs.
    """
    with patch("langchain_redis.vectorstores.SearchIndex", SchemaCapturingIndex):
        RedisVectorStore(
            MockEmbeddings(),
            index_name="vector_attrs_unit",
            redis_url="redis://localhost",
            indexing_algorithm="HNSW",
            vector_attrs=HNSW_ATTRS,
        )

    assert SchemaCapturingIndex.last_schema is not None
    (vector_field,) = [
        field
        for field in SchemaCapturingIndex.last_schema["fields"]
        if field["type"] == "vector"
    ]
    for name, value in HNSW_ATTRS.items():
        assert vector_field["attrs"][name] == value
