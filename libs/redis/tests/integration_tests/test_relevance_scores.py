"""Integration tests for relevance scores and score-threshold retrieval."""

from typing import Any, Dict, Generator, List, Sequence
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

# Unit-length vectors with cosine similarity to the query of 1.0, 0.6 and 0.0.
_VECTORS = {
    QUERY: [1.0, 0.0, 0.0, 0.0],
    CLOSE_DOC: [1.0, 0.0, 0.0, 0.0],
    MID_DOC: [0.6, 0.8, 0.0, 0.0],
    FAR_DOC: [0.0, 1.0, 0.0, 0.0],
}

# What Redis reports for those vectors. L2 is the squared Euclidean distance.
RAW_DISTANCES: Dict[str, Dict[str, float]] = {
    "COSINE": {CLOSE_DOC: 0.0, MID_DOC: 0.4, FAR_DOC: 1.0},
    "L2": {CLOSE_DOC: 0.0, MID_DOC: 0.8, FAR_DOC: 2.0},
}
# RedisVL's normalization: (2 - d) / 2 for COSINE and 1 / (1 + d) for L2.
RELEVANCE: Dict[str, Dict[str, float]] = {
    "COSINE": {CLOSE_DOC: 1.0, MID_DOC: 0.8, FAR_DOC: 0.5},
    "L2": {CLOSE_DOC: 1.0, MID_DOC: 1 / 1.8, FAR_DOC: 1 / 3},
}
# A score_threshold that keeps the close and mid documents and drops the far one.
SCORE_THRESHOLD = {"COSINE": 0.7, "L2": 0.5}
# A raw distance_threshold of 0.5 keeps the documents at or under that distance.
# Read as a normalized similarity instead, it would be a radius of 1.0.
DISTANCE_THRESHOLD_HITS = {"COSINE": {CLOSE_DOC, MID_DOC}, "L2": {CLOSE_DOC}}

SUPPORTED_METRICS = ["COSINE", "L2"]


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


def _by_content(results: Sequence[Any]) -> Dict[str, float]:
    return {doc.page_content: score for doc, score in results}


@pytest.mark.parametrize("storage_type", ["hash", "json"])
@pytest.mark.parametrize("distance_metric", SUPPORTED_METRICS)
def test_relevance_scores_use_redisvl_normalization(
    redis_url: str, distance_metric: str, storage_type: str
) -> None:
    for store in _store(redis_url, distance_metric, storage_type):
        results = store.similarity_search_with_relevance_scores(QUERY, k=3)

        assert _by_content(results) == pytest.approx(RELEVANCE[distance_metric])
        # Scores are in [0, 1] and the closest document comes first.
        assert [doc.page_content for doc, _ in results] == [
            CLOSE_DOC,
            MID_DOC,
            FAR_DOC,
        ]
        assert all(0.0 <= score <= 1.0 for _, score in results)


@pytest.mark.parametrize("distance_metric", SUPPORTED_METRICS)
def test_similarity_search_with_score_keeps_raw_distances(
    redis_url: str, distance_metric: str
) -> None:
    for store in _store(redis_url, distance_metric):
        store.similarity_search_with_relevance_scores(QUERY, k=3)

        raw = store.similarity_search_with_score(QUERY, k=3)

        assert _by_content(raw) == pytest.approx(RAW_DISTANCES[distance_metric])


@pytest.mark.parametrize("distance_metric", SUPPORTED_METRICS)
def test_score_threshold_retriever(redis_url: str, distance_metric: str) -> None:
    for store in _store(redis_url, distance_metric):
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": SCORE_THRESHOLD[distance_metric], "k": 3},
        )

        docs = retriever.invoke(QUERY)

        assert [doc.page_content for doc in docs] == [CLOSE_DOC, MID_DOC]


@pytest.mark.parametrize("distance_metric", SUPPORTED_METRICS)
async def test_relevance_scores_async(redis_url: str, distance_metric: str) -> None:
    for store in _store(redis_url, distance_metric):
        results = await store.asimilarity_search_with_relevance_scores(QUERY, k=3)

        assert _by_content(results) == pytest.approx(RELEVANCE[distance_metric])


@pytest.mark.parametrize("distance_metric", SUPPORTED_METRICS)
async def test_score_threshold_retriever_async(
    redis_url: str, distance_metric: str
) -> None:
    for store in _store(redis_url, distance_metric):
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": SCORE_THRESHOLD[distance_metric], "k": 3},
        )

        docs = await retriever.ainvoke(QUERY)

        assert [doc.page_content for doc in docs] == [CLOSE_DOC, MID_DOC]


@pytest.mark.parametrize("distance_metric", SUPPORTED_METRICS)
def test_distance_threshold_stays_a_raw_distance(
    redis_url: str, distance_metric: str
) -> None:
    for store in _store(redis_url, distance_metric):
        results = store.similarity_search_with_relevance_scores(
            QUERY, k=3, distance_threshold=0.5
        )
        raw = store.similarity_search_with_score(QUERY, k=3, distance_threshold=0.5)

        expected_hits = DISTANCE_THRESHOLD_HITS[distance_metric]
        assert set(_by_content(results)) == expected_hits
        assert set(_by_content(raw)) == expected_hits
        # The hits that remain still carry normalized scores.
        assert _by_content(results) == pytest.approx(
            {doc: RELEVANCE[distance_metric][doc] for doc in expected_hits}
        )


@pytest.mark.parametrize("distance_metric", SUPPORTED_METRICS)
async def test_distance_threshold_stays_a_raw_distance_async(
    redis_url: str, distance_metric: str
) -> None:
    for store in _store(redis_url, distance_metric):
        results = await store.asimilarity_search_with_relevance_scores(
            QUERY, k=3, distance_threshold=0.5
        )

        assert set(_by_content(results)) == DISTANCE_THRESHOLD_HITS[distance_metric]


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
            QUERY, k=2, filter=Tag("topic") == "b", score_threshold=0.7
        )

        assert [doc.page_content for doc, _ in results] == [MID_DOC]
    finally:
        store.index.delete(drop=True)


def test_ip_index_rejects_relevance_scores(redis_url: str) -> None:
    """RedisVL doesn't normalize IP, so it can't be returned as a relevance score."""
    for store in _store(redis_url, "IP"):
        with pytest.raises(ValueError, match="IP"):
            store.similarity_search_with_relevance_scores(QUERY, k=3)

        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5},
        )
        with pytest.raises(ValueError, match="IP"):
            retriever.invoke(QUERY)

        # Raw distances are still available.
        raw = store.similarity_search_with_score(QUERY, k=3)
        assert _by_content(raw) == pytest.approx(
            {CLOSE_DOC: 0.0, MID_DOC: 0.4, FAR_DOC: 1.0}
        )


async def test_ip_index_rejects_relevance_scores_async(redis_url: str) -> None:
    for store in _store(redis_url, "IP"):
        with pytest.raises(ValueError, match="IP"):
            await store.asimilarity_search_with_relevance_scores(QUERY, k=3)


def test_reopened_index_uses_live_distance_metric(redis_url: str) -> None:
    """A reopened index keeps its own metric, not the config default."""
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
        assert _by_content(results) == pytest.approx(RELEVANCE["L2"])


def test_reopened_ip_index_still_rejects_relevance_scores(redis_url: str) -> None:
    for store in _store(redis_url, "IP"):
        reopened = RedisVectorStore.from_existing_index(
            index_name=store.index.name,
            embedding=KeywordEmbeddings(),
            redis_url=redis_url,
            embedding_dimensions=DIMS,
        )

        assert reopened.config.distance_metric == "COSINE"
        with pytest.raises(ValueError, match="IP"):
            reopened.similarity_search_with_relevance_scores(QUERY, k=3)
