"""Integration tests for vector-field tuning attributes on real indexes."""

from typing import Generator, List
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings

from langchain_redis import RedisVectorStore

DIMS = 4
TARGET_DOC = "target document"
OTHER_DOC = "unrelated document"
TEXTS = [TARGET_DOC, OTHER_DOC]
QUERY = "find the target"

_VECTORS = {
    TARGET_DOC: [1.0, 0.0, 0.0, 0.0],
    OTHER_DOC: [0.0, 1.0, 0.0, 0.0],
    QUERY: [1.0, 0.0, 0.0, 0.0],
}
_DEFAULT_VECTOR = [0.0, 0.0, 1.0, 0.0]

HNSW_ATTRS = {"m": 8, "ef_construction": 100, "ef_runtime": 20}
SVS_ATTRS = {"graph_max_degree": 32, "search_window_size": 10, "compression": "LVQ8"}


class KeywordEmbeddings(Embeddings):
    """Deterministic embeddings mapping known texts to fixed vectors."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [_VECTORS.get(text, _DEFAULT_VECTOR) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return _VECTORS.get(text, _DEFAULT_VECTOR)


@pytest.fixture
def svs_server(redis_server_version: tuple) -> None:
    """Skip the test when the server lacks SVS-VAMANA (Redis < 8.2)."""
    if redis_server_version < (8, 2):
        pytest.skip("SVS-VAMANA requires Redis >= 8.2")


def _round_trip_store(
    redis_url: str, **config_kwargs: object
) -> Generator[RedisVectorStore, None, None]:
    store = RedisVectorStore(
        KeywordEmbeddings(),
        index_name=f"vector_attrs_{uuid4().hex[:8]}",
        redis_url=redis_url,
        **config_kwargs,  # type: ignore[arg-type]
    )
    try:
        store.add_texts(TEXTS)
        yield store
    finally:
        store.index.delete(drop=True)


def test_hnsw_with_custom_attrs_round_trip(redis_url: str) -> None:
    """An HNSW index with tuned attrs is created server-side and searchable."""
    for store in _round_trip_store(
        redis_url, indexing_algorithm="HNSW", vector_attrs=HNSW_ATTRS
    ):
        docs = store.similarity_search(QUERY, k=1)
        assert docs[0].page_content == TARGET_DOC
        assert "hnsw" in str(store.index.info()).lower()


def test_svs_vamana_round_trip(redis_url: str, svs_server: None) -> None:
    """An SVS-VAMANA index with compression is created and searchable (8.2+)."""
    for store in _round_trip_store(
        redis_url, indexing_algorithm="SVS-VAMANA", vector_attrs=SVS_ATTRS
    ):
        docs = store.similarity_search(QUERY, k=1)
        assert docs[0].page_content == TARGET_DOC
        assert "vamana" in str(store.index.info()).lower()
