"""Integration tests for relevance scores and score-threshold retrieval."""

from typing import Generator, List
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings
from redisvl.query.filter import Tag  # type: ignore[import]

from langchain_redis import RedisVectorStore

DIMS = 4
QUERY = "query"
CLOSE_DOC = "same direction as the query"
MID_DOC = "partly aligned with the query"
FAR_DOC = "orthogonal to the query"

# Unit-length vectors with known cosine similarity to the query: 1.0, 0.6 and 0.0.
_VECTORS = {
    QUERY: [1.0, 0.0, 0.0, 0.0],
    CLOSE_DOC: [1.0, 0.0, 0.0, 0.0],
    MID_DOC: [0.6, 0.8, 0.0, 0.0],
    FAR_DOC: [0.0, 1.0, 0.0, 0.0],
}
EXPECTED = {CLOSE_DOC: 1.0, MID_DOC: 0.6, FAR_DOC: 0.0}


class KeywordEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [_VECTORS[text] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return _VECTORS[text]


def _store(
    redis_url: str, distance_metric: str, storage_type: str = "hash"
) -> Generator[RedisVectorStore, None, None]:
    store = RedisVectorStore(
        KeywordEmbeddings(),
        index_name=f"relevance_{uuid4().hex[:8]}",
        redis_url=redis_url,
        embedding_dimensions=DIMS,
        distance_metric=distance_metric,
        storage_type=storage_type,
    )
    try:
        store.add_texts([CLOSE_DOC, MID_DOC, FAR_DOC])
        yield store
    finally:
        store.index.delete(drop=True)


@pytest.mark.parametrize("storage_type", ["hash", "json"])
@pytest.mark.parametrize("distance_metric", ["COSINE", "L2", "IP"])
def test_relevance_scores_are_cosine_similarity(
    redis_url: str, distance_metric: str, storage_type: str
) -> None:
    """Every metric yields the same score on normalized embeddings."""
    for store in _store(redis_url, distance_metric, storage_type):
        results = store.similarity_search_with_relevance_scores(QUERY, k=3)

        assert {doc.page_content: score for doc, score in results} == pytest.approx(
            EXPECTED, abs=1e-4
        )
        # Best match first, and identical vectors score exactly 1 on every metric.
        assert [doc.page_content for doc, _ in results] == [
            CLOSE_DOC,
            MID_DOC,
            FAR_DOC,
        ]


@pytest.mark.parametrize("distance_metric", ["COSINE", "L2", "IP"])
def test_score_threshold_retriever(redis_url: str, distance_metric: str) -> None:
    """`similarity_score_threshold` drops documents below the threshold."""
    for store in _store(redis_url, distance_metric):
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5, "k": 3},
        )

        docs = retriever.invoke(QUERY)

        assert [doc.page_content for doc in docs] == [CLOSE_DOC, MID_DOC]


@pytest.mark.parametrize("distance_metric", ["COSINE", "L2", "IP"])
async def test_score_threshold_retriever_async(
    redis_url: str, distance_metric: str
) -> None:
    for store in _store(redis_url, distance_metric):
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5, "k": 3},
        )

        docs = await retriever.ainvoke(QUERY)

        assert [doc.page_content for doc in docs] == [CLOSE_DOC, MID_DOC]


def test_score_threshold_respects_metadata_filter(redis_url: str) -> None:
    store = RedisVectorStore(
        KeywordEmbeddings(),
        index_name=f"relevance_{uuid4().hex[:8]}",
        redis_url=redis_url,
        embedding_dimensions=DIMS,
        metadata_schema=[{"name": "topic", "type": "tag"}],
    )
    try:
        store.add_texts(
            [CLOSE_DOC, MID_DOC],
            metadatas=[{"topic": "a"}, {"topic": "b"}],
        )

        results = store.similarity_search_with_relevance_scores(
            QUERY, k=2, filter=Tag("topic") == "b", score_threshold=0.5
        )

        assert [doc.page_content for doc, _ in results] == [MID_DOC]
    finally:
        store.index.delete(drop=True)


def test_reopened_index_uses_live_distance_metric(redis_url: str) -> None:
    """A reopened L2 index keeps scoring as L2, not as the config default."""
    for store in _store(redis_url, "L2"):
        reopened = RedisVectorStore.from_existing_index(
            index_name=store.index.name,
            embedding=KeywordEmbeddings(),
            redis_url=redis_url,
            embedding_dimensions=DIMS,
        )

        # The reopened config falls back to the default metric...
        assert reopened.config.distance_metric == "COSINE"

        results = reopened.similarity_search_with_relevance_scores(QUERY, k=3)

        # ...but the scores still follow the index's real metric.
        assert {doc.page_content: score for doc, score in results} == pytest.approx(
            EXPECTED, abs=1e-4
        )
