"""Integration tests for hybrid and full-text search on RedisVectorStore.

Uses deterministic embeddings crafted so that the best full-text match and
the best vector match for the test query are different documents. This makes
the effect of the text/vector weighting (`alpha`) observable.
"""

from typing import Any, Generator, List
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings
from redisvl.query.filter import Tag  # type: ignore[import]

from langchain_redis import RedisVectorStore

TEXT_WINNER = "needle appears in the text body"
VECTOR_WINNER = "semantic neighbor without the search term"
TITLE_WINNER = "infrastructure guide without the search term"
FILTERED_TECH_DOC = "analytics pipeline reference"

TEXT_WINNER_ID = "text_winner"
VECTOR_WINNER_ID = "vector_winner"
TITLE_WINNER_ID = "title_winner"
FILTERED_TECH_DOC_ID = "filtered_tech_doc"

DOC_ID_FIELD = "doc_id"
CATEGORY_FIELD = "category"
TITLE_FIELD = "title"
PETS_CATEGORY = "pets"
TECH_CATEGORY = "tech"

TEXTS = [TEXT_WINNER, VECTOR_WINNER, TITLE_WINNER, FILTERED_TECH_DOC]
METADATAS = [
    {
        DOC_ID_FIELD: TEXT_WINNER_ID,
        CATEGORY_FIELD: PETS_CATEGORY,
        TITLE_FIELD: "body match",
    },
    {
        DOC_ID_FIELD: VECTOR_WINNER_ID,
        CATEGORY_FIELD: PETS_CATEGORY,
        TITLE_FIELD: "vector match",
    },
    {
        DOC_ID_FIELD: TITLE_WINNER_ID,
        CATEGORY_FIELD: TECH_CATEGORY,
        TITLE_FIELD: "needle",
    },
    {
        DOC_ID_FIELD: FILTERED_TECH_DOC_ID,
        CATEGORY_FIELD: TECH_CATEGORY,
        TITLE_FIELD: "other",
    },
]
METADATA_SCHEMA = [
    {"name": DOC_ID_FIELD, "type": "tag"},
    {"name": CATEGORY_FIELD, "type": "tag"},
    {"name": TITLE_FIELD, "type": "text"},
]

# The query text matches TEXT_WINNER, but its embedding is identical to
# VECTOR_WINNER's vector: full-text search favors one doc, vector search the other.
QUERY = "needle"

_VECTORS = {
    # Make TEXT_WINNER the second-best vector hit, so RRF does not depend on
    # Redis tie-breaking among unrelated orthogonal vectors.
    TEXT_WINNER: [0.2, 0.8, 0.0, 0.0],
    VECTOR_WINNER: [0.0, 1.0, 0.0, 0.0],
    TITLE_WINNER: [1.0, 0.0, 0.0, 0.0],
    FILTERED_TECH_DOC: [1.0, 0.0, 0.0, 0.0],
    QUERY: [0.0, 1.0, 0.0, 0.0],
}
_DEFAULT_VECTOR = [0.5, 0.5, 0.5, 0.5]


class KeywordEmbeddings(Embeddings):
    """Deterministic embeddings mapping known texts to fixed vectors."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [_VECTORS.get(text, _DEFAULT_VECTOR) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return _VECTORS.get(text, _DEFAULT_VECTOR)


def _make_store(redis_url: str, **config_kwargs: Any) -> RedisVectorStore:
    store = RedisVectorStore(
        KeywordEmbeddings(),
        index_name=f"hybrid_test_{uuid4().hex[:8]}",
        redis_url=redis_url,
        metadata_schema=METADATA_SCHEMA,
        **config_kwargs,
    )
    store.add_texts(TEXTS, metadatas=METADATAS)
    return store


@pytest.fixture
def store(redis_url: str) -> Generator[RedisVectorStore, None, None]:
    store = _make_store(redis_url)
    yield store
    store.index.delete(drop=True)


@pytest.fixture
def ft_hybrid_server(redis_server_version: tuple) -> None:
    """Skip the test when the server lacks FT.HYBRID (Redis < 8.4)."""
    if redis_server_version < (8, 4):
        pytest.skip("FT.HYBRID requires Redis >= 8.4")


@pytest.mark.parametrize(
    "alpha,expected_winner",
    [(1.0, VECTOR_WINNER_ID), (0.0, TEXT_WINNER_ID)],
    ids=["vector-only", "text-only"],
)
def test_aggregate_alpha_extremes_pick_matching_winner(
    store: RedisVectorStore, alpha: float, expected_winner: str
) -> None:
    """alpha fully weights one signal: 1.0 ranks the vector winner first,
    0.0 the BM25 winner."""
    docs = store.hybrid_search(QUERY, k=2, method="aggregate", alpha=alpha)
    assert docs[0].metadata[DOC_ID_FIELD] == expected_winner


def test_aggregate_balanced_alpha_surfaces_both(store: RedisVectorStore) -> None:
    """A balanced alpha surfaces both winners, with scores in descending order."""
    results = store.hybrid_search_with_score(QUERY, k=2, method="aggregate", alpha=0.5)
    doc_ids = {doc.metadata[DOC_ID_FIELD] for doc, _ in results}
    assert doc_ids == {TEXT_WINNER_ID, VECTOR_WINNER_ID}
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_search_filter_narrows_results(store: RedisVectorStore) -> None:
    """A tag FilterExpression restricts hybrid results to matching documents."""
    docs = store.hybrid_search(
        QUERY, k=4, method="aggregate", filter=Tag(CATEGORY_FIELD) == PETS_CATEGORY
    )
    assert docs
    assert {doc.metadata[DOC_ID_FIELD] for doc in docs} <= {
        TEXT_WINNER_ID,
        VECTOR_WINNER_ID,
    }


@pytest.mark.parametrize(
    "fusion_kwargs",
    [
        {"combination_method": "RRF"},
        {"combination_method": "LINEAR", "alpha": 0.5},
    ],
    ids=["rrf", "linear"],
)
def test_ft_hybrid_fusion_surfaces_text_and_vector_matches(
    store: RedisVectorStore, ft_hybrid_server: None, fusion_kwargs: dict
) -> None:
    """FT.HYBRID surfaces both winners under RRF and LINEAR fusion (Redis 8.4+).

    The LINEAR case also exercises the alpha -> 1-alpha translation into
    FT.HYBRID's text-weighted linear_alpha.
    """
    results = store.hybrid_search_with_score(
        QUERY, k=2, method="ft_hybrid", **fusion_kwargs
    )
    doc_ids = {doc.metadata[DOC_ID_FIELD] for doc, _ in results}
    assert doc_ids == {TEXT_WINNER_ID, VECTOR_WINNER_ID}


def test_auto_method_works_on_any_supported_server(
    store: RedisVectorStore,
) -> None:
    """The default call (method='auto') returns correct results end-to-end.

    Runs against whichever real server the suite uses, exercising the INFO
    version probe on genuine server output — the one thing mocks can't cover.
    """
    docs = store.hybrid_search(QUERY, k=2)
    assert {doc.metadata[DOC_ID_FIELD] for doc in docs} == {
        TEXT_WINNER_ID,
        VECTOR_WINNER_ID,
    }


def test_hybrid_search_on_json_storage(redis_url: str) -> None:
    """Hybrid search works against JSON storage, not just hash."""
    store = _make_store(redis_url, storage_type="json")
    try:
        docs = store.hybrid_search(QUERY, k=2, method="aggregate", alpha=0.5)
        assert {doc.metadata[DOC_ID_FIELD] for doc in docs} == {
            TEXT_WINNER_ID,
            VECTOR_WINNER_ID,
        }
    finally:
        store.index.delete(drop=True)


def test_full_text_search_matches_terms(store: RedisVectorStore) -> None:
    """BM25 full-text search returns term-matching docs with their metadata."""
    docs = store.full_text_search("needle", k=4)
    assert [doc.metadata[DOC_ID_FIELD] for doc in docs] == [TEXT_WINNER_ID]
    assert docs[0].metadata[CATEGORY_FIELD] == PETS_CATEGORY


def test_full_text_search_field_weights(store: RedisVectorStore) -> None:
    """text_fields weighting redirects matching to the specified field.

    Only TITLE_WINNER has "needle" in its title, so a title-weighted search
    must return it rather than the content-field match.
    """
    docs = store.full_text_search("needle", k=4, text_fields={TITLE_FIELD: 2.0})
    assert [doc.metadata[DOC_ID_FIELD] for doc in docs] == [TITLE_WINNER_ID]


def test_full_text_search_respects_filter(store: RedisVectorStore) -> None:
    """Full-text results are restricted by a tag FilterExpression."""
    docs = store.full_text_search(
        "infrastructure analytics",
        k=4,
        filter=Tag(CATEGORY_FIELD) == TECH_CATEGORY,
    )
    assert docs
    assert {doc.metadata[DOC_ID_FIELD] for doc in docs} <= {
        TITLE_WINNER_ID,
        FILTERED_TECH_DOC_ID,
    }
