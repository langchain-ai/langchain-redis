"""Integration tests for relevance scores and score-threshold retrieval.

Also covers the `(Document, score)` contract of the score methods: passing
the deprecated `with_vectors` must not turn their results into triples, and
MMR must still get the stored vectors it re-ranks with (issue #233).
"""

import math
import warnings
from contextlib import contextmanager
from typing import Any, ContextManager, Dict, Generator, Iterator, List, Sequence
from uuid import uuid4

import pytest
from langchain_core._api import LangChainDeprecationWarning
from langchain_core.embeddings import Embeddings
from redisvl.query.filter import Tag  # type: ignore[import]

from langchain_redis import RedisVectorStore

DIMS = 4
QUERY = "query"
CLOSE_DOC = "same direction as the query"
MID_DOC = "partly aligned with the query"
FAR_DOC = "orthogonal to the query"

# Documents for the MMR test, searched with the same QUERY. A plain top-2
# search returns MMR_CLOSE and MMR_NEAR_DUPLICATE. MMR with lambda_mult=0.5
# swaps the near duplicate for MMR_DIVERSE, which it only gets right if it
# receives the real stored vectors.
MMR_CLOSE = "close to the query, off-axis"
MMR_NEAR_DUPLICATE = "near duplicate of the off-axis document"
MMR_DIVERSE = "further from the query, in a different direction"
NO_MATCH_QUERY = "orthogonal to every document"

DEFAULT_TEXTS = [CLOSE_DOC, MID_DOC, FAR_DOC]
MMR_TEXTS = [MMR_CLOSE, MMR_NEAR_DUPLICATE, MMR_DIVERSE]


def _unit(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


_VECTORS = {
    QUERY: [1.0, 0.0, 0.0, 0.0],
    # Unit-length vectors with cosine similarity to the query of 1.0, 0.6 and 0.0.
    CLOSE_DOC: [1.0, 0.0, 0.0, 0.0],
    MID_DOC: [0.6, 0.8, 0.0, 0.0],
    FAR_DOC: [0.0, 1.0, 0.0, 0.0],
    # MMR documents. Each test stores only its own texts.
    MMR_CLOSE: _unit([0.8, 0.6, 0.0, 0.0]),
    MMR_NEAR_DUPLICATE: _unit([0.79, 0.6, 0.1, 0.0]),
    MMR_DIVERSE: _unit([0.7, -0.714, 0.0, 0.0]),
    NO_MATCH_QUERY: [0.0, 0.0, 0.0, 1.0],
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

# Passing the deprecated `with_vectors` must not change what these methods
# return: (Document, score) pairs, not triples (issue #233).
EXTRA_KWARGS = pytest.mark.parametrize(
    "extra_kwargs",
    [
        pytest.param({}, id="default"),
        pytest.param({"with_vectors": True}, id="with_vectors"),
        pytest.param({"with_vectors": False}, id="with_vectors_false"),
    ],
)


class KeywordEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [_VECTORS[text] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return _VECTORS[text]


def _store(
    redis_url: str,
    distance_metric: str,
    storage_type: str = "hash",
    texts: Sequence[str] = DEFAULT_TEXTS,
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
        store.add_texts(list(texts))
        yield store
    finally:
        store.index.delete(drop=True)


def _by_content(results: Sequence[Any]) -> Dict[str, float]:
    return {doc.page_content: score for doc, score in results}


@contextmanager
def _no_deprecation_warnings() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", LangChainDeprecationWarning)
        yield


def _expect_deprecation(extra_kwargs: Dict[str, Any]) -> ContextManager[Any]:
    """Expect the `with_vectors` warning if it was passed, else no deprecation."""
    if "with_vectors" in extra_kwargs:
        return pytest.warns(LangChainDeprecationWarning, match="with_vectors")
    return _no_deprecation_warnings()


@EXTRA_KWARGS
@pytest.mark.parametrize("storage_type", ["hash", "json"])
@pytest.mark.parametrize("distance_metric", SUPPORTED_METRICS)
def test_relevance_scores_use_redisvl_normalization(
    redis_url: str,
    distance_metric: str,
    storage_type: str,
    extra_kwargs: Dict[str, Any],
) -> None:
    for store in _store(redis_url, distance_metric, storage_type):
        with _expect_deprecation(extra_kwargs):
            results = store.similarity_search_with_relevance_scores(
                QUERY, k=3, **extra_kwargs
            )

        assert _by_content(results) == pytest.approx(RELEVANCE[distance_metric])
        # Scores are in [0, 1] and the closest document comes first.
        assert [doc.page_content for doc, _ in results] == [
            CLOSE_DOC,
            MID_DOC,
            FAR_DOC,
        ]
        assert all(0.0 <= score <= 1.0 for _, score in results)


@EXTRA_KWARGS
@pytest.mark.parametrize("distance_metric", SUPPORTED_METRICS)
def test_similarity_search_with_score_keeps_raw_distances(
    redis_url: str, distance_metric: str, extra_kwargs: Dict[str, Any]
) -> None:
    for store in _store(redis_url, distance_metric):
        store.similarity_search_with_relevance_scores(QUERY, k=3)

        with _expect_deprecation(extra_kwargs):
            raw = store.similarity_search_with_score(QUERY, k=3, **extra_kwargs)
            raw_by_vector = store.similarity_search_with_score_by_vector(
                _VECTORS[QUERY], k=3, **extra_kwargs
            )

        assert _by_content(raw) == pytest.approx(RAW_DISTANCES[distance_metric])
        assert _by_content(raw_by_vector) == pytest.approx(
            RAW_DISTANCES[distance_metric]
        )


@EXTRA_KWARGS
@pytest.mark.parametrize("distance_metric", SUPPORTED_METRICS)
def test_score_threshold_retriever(
    redis_url: str, distance_metric: str, extra_kwargs: Dict[str, Any]
) -> None:
    for store in _store(redis_url, distance_metric):
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": SCORE_THRESHOLD[distance_metric],
                "k": 3,
                **extra_kwargs,
            },
        )

        with _expect_deprecation(extra_kwargs):
            docs = retriever.invoke(QUERY)

        assert [doc.page_content for doc in docs] == [CLOSE_DOC, MID_DOC]


@EXTRA_KWARGS
@pytest.mark.parametrize("distance_metric", SUPPORTED_METRICS)
async def test_relevance_scores_async(
    redis_url: str, distance_metric: str, extra_kwargs: Dict[str, Any]
) -> None:
    for store in _store(redis_url, distance_metric):
        with _expect_deprecation(extra_kwargs):
            results = await store.asimilarity_search_with_relevance_scores(
                QUERY, k=3, **extra_kwargs
            )

        assert _by_content(results) == pytest.approx(RELEVANCE[distance_metric])


@EXTRA_KWARGS
@pytest.mark.parametrize("distance_metric", SUPPORTED_METRICS)
async def test_score_threshold_retriever_async(
    redis_url: str, distance_metric: str, extra_kwargs: Dict[str, Any]
) -> None:
    for store in _store(redis_url, distance_metric):
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": SCORE_THRESHOLD[distance_metric],
                "k": 3,
                **extra_kwargs,
            },
        )

        with _expect_deprecation(extra_kwargs):
            docs = await retriever.ainvoke(QUERY)

        assert [doc.page_content for doc in docs] == [CLOSE_DOC, MID_DOC]


@pytest.mark.parametrize(
    "extra_kwargs",
    [
        pytest.param({}, id="default"),
        # Deprecated: warns and has no effect, instead of raising TypeError.
        pytest.param({"with_vectors": True}, id="with_vectors"),
    ],
)
@pytest.mark.parametrize(
    "return_all", [False, True], ids=["indexed_fields", "return_all"]
)
@pytest.mark.parametrize("storage_type", ["hash", "json"])
def test_mmr_reranks_with_stored_vectors(
    redis_url: str,
    storage_type: str,
    return_all: bool,
    extra_kwargs: Dict[str, Any],
) -> None:
    for store in _store(redis_url, "COSINE", storage_type, texts=MMR_TEXTS):
        plain = store.similarity_search(QUERY, k=2, return_all=return_all)
        assert {doc.page_content for doc in plain} == {MMR_CLOSE, MMR_NEAR_DUPLICATE}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mmr = store.max_marginal_relevance_search(
                QUERY, k=2, fetch_k=3, return_all=return_all, **extra_kwargs
            )
            mmr_by_vector = store.max_marginal_relevance_search_by_vector(
                _VECTORS[QUERY],
                k=2,
                fetch_k=3,
                return_all=return_all,
                **extra_kwargs,
            )
            # Every document is at cosine distance 1.0, so none is a candidate.
            no_candidates = store.max_marginal_relevance_search(
                NO_MATCH_QUERY,
                k=2,
                fetch_k=3,
                distance_threshold=0.01,
                return_all=return_all,
                **extra_kwargs,
            )

        assert [doc.page_content for doc in mmr] == [MMR_CLOSE, MMR_DIVERSE]
        assert [doc.page_content for doc in mmr_by_vector] == [MMR_CLOSE, MMR_DIVERSE]
        assert no_candidates == []
        # One deprecation warning per call that passed `with_vectors`, else none.
        deprecations = [
            w for w in caught if issubclass(w.category, LangChainDeprecationWarning)
        ]
        assert len(deprecations) == (3 if extra_kwargs else 0)


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


@pytest.mark.parametrize(
    "distance_metric, distance_threshold, score_threshold, expected_hits",
    [
        # The raw-distance cutoff is the stricter one: on its own it drops
        # FAR_DOC, while a score_threshold this low would keep every document.
        pytest.param("COSINE", 0.6, 0.4, {CLOSE_DOC, MID_DOC}, id="cosine-distance"),
        pytest.param("L2", 1.0, 0.2, {CLOSE_DOC, MID_DOC}, id="l2-distance"),
        # The relevance cutoff is the stricter one: on its own the distance
        # cutoff would keep MID_DOC, but its score is under the threshold.
        pytest.param("COSINE", 0.6, 0.9, {CLOSE_DOC}, id="cosine-score"),
        pytest.param("L2", 1.0, 0.7, {CLOSE_DOC}, id="l2-score"),
    ],
)
def test_distance_and_score_thresholds_combine(
    redis_url: str,
    distance_metric: str,
    distance_threshold: float,
    score_threshold: float,
    expected_hits: set,
) -> None:
    """Each cutoff is applied: distance as a raw distance, score as a relevance."""
    for store in _store(redis_url, distance_metric):
        results = store.similarity_search_with_relevance_scores(
            QUERY,
            k=3,
            distance_threshold=distance_threshold,
            score_threshold=score_threshold,
        )

        assert set(_by_content(results)) == expected_hits
        assert _by_content(results) == pytest.approx(
            {doc: RELEVANCE[distance_metric][doc] for doc in expected_hits}
        )


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

        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5},
        )
        with pytest.raises(ValueError, match="IP"):
            await retriever.ainvoke(QUERY)


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
