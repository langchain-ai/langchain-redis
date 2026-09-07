"""Integration tests for filter-based deletion."""

from typing import Any, List, Set
from uuid import uuid4

from langchain_core.embeddings import Embeddings
from redisvl.query.filter import Tag  # type: ignore[import]

from langchain_redis import RedisVectorStore

DIMS = 4
TEAM_FIELD = "team"
DOC_ID_FIELD = "doc_id"
TEAM_A = "team_a"
TEAM_B = "team_b"
GHOST_TEAM = "team_ghost"

TEAM_A_DOC_IDS = ["a1", "a2", "a3", "a4"]
TEAM_B_DOC_IDS = ["b1", "b2"]
ALL_DOC_IDS = TEAM_A_DOC_IDS + TEAM_B_DOC_IDS

METADATA_SCHEMA = [
    {"name": TEAM_FIELD, "type": "tag"},
    {"name": DOC_ID_FIELD, "type": "tag"},
]
QUERY = "any query"


class ConstantEmbeddings(Embeddings):
    """All texts share one vector; these tests assert sets, not rankings."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * DIMS for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * DIMS


def _make_store(redis_url: str, **config_kwargs: Any) -> RedisVectorStore:
    store = RedisVectorStore(
        ConstantEmbeddings(),
        index_name=f"filter_ops_{uuid4().hex[:8]}",
        redis_url=redis_url,
        metadata_schema=METADATA_SCHEMA,
        **config_kwargs,
    )
    texts = [f"document {doc_id}" for doc_id in ALL_DOC_IDS]
    metadatas = [
        {
            DOC_ID_FIELD: doc_id,
            TEAM_FIELD: TEAM_A if doc_id in TEAM_A_DOC_IDS else TEAM_B,
        }
        for doc_id in ALL_DOC_IDS
    ]
    store.add_texts(texts, metadatas=metadatas)
    return store


def _remaining_doc_ids(store: RedisVectorStore) -> Set[str]:
    docs = store.similarity_search(QUERY, k=20)
    return {doc.metadata[DOC_ID_FIELD] for doc in docs}


def test_delete_by_filter_removes_only_matching(redis_url: str) -> None:
    """Filter delete removes exactly the matching docs; delete(filter=) works too."""
    store = _make_store(redis_url)
    try:
        deleted = store.delete_by_filter(Tag(TEAM_FIELD) == TEAM_A)
        assert deleted == len(TEAM_A_DOC_IDS)
        assert _remaining_doc_ids(store) == set(TEAM_B_DOC_IDS)

        assert store.delete(filter=Tag(TEAM_FIELD) == TEAM_B) is True
        assert _remaining_doc_ids(store) == set()
    finally:
        store.index.delete(drop=True)


def test_dry_run_counts_without_deleting(redis_url: str) -> None:
    """dry_run reports the would-be count and leaves every document in place."""
    store = _make_store(redis_url)
    try:
        would_delete = store.delete_by_filter(Tag(TEAM_FIELD) == TEAM_A, dry_run=True)
        assert would_delete == len(TEAM_A_DOC_IDS)
        assert _remaining_doc_ids(store) == set(ALL_DOC_IDS)
    finally:
        store.index.delete(drop=True)


def test_no_match_returns_zero_and_false(redis_url: str) -> None:
    """A filter matching nothing deletes nothing and reports it truthfully."""
    store = _make_store(redis_url)
    try:
        assert store.delete_by_filter(Tag(TEAM_FIELD) == GHOST_TEAM) == 0
        assert store.delete(filter=Tag(TEAM_FIELD) == GHOST_TEAM) is False
        assert _remaining_doc_ids(store) == set(ALL_DOC_IDS)
    finally:
        store.index.delete(drop=True)


def test_shared_prefix_indexes_are_isolated(redis_url: str) -> None:
    """A filter delete on one index never touches a sibling sharing its prefix."""
    shared_prefix = f"shared_{uuid4().hex[:8]}"
    store_a = _make_store(redis_url, key_prefix=shared_prefix)
    store_b = _make_store(redis_url, key_prefix=shared_prefix)
    try:
        deleted = store_a.delete_by_filter(Tag(TEAM_FIELD) == TEAM_A)
        assert deleted == len(TEAM_A_DOC_IDS)

        assert _remaining_doc_ids(store_a) == set(TEAM_B_DOC_IDS)
        # store_b's documents, including its TEAM_A ones, must be untouched
        assert _remaining_doc_ids(store_b) == set(ALL_DOC_IDS)
    finally:
        store_a.index.delete(drop=True)
        store_b.index.delete(drop=True)


def test_delete_by_filter_on_json_storage(redis_url: str) -> None:
    """Filter deletion works against JSON storage, not just hash."""
    store = _make_store(redis_url, storage_type="json")
    try:
        deleted = store.delete_by_filter(Tag(TEAM_FIELD) == TEAM_A)
        assert deleted == len(TEAM_A_DOC_IDS)
        assert _remaining_doc_ids(store) == set(TEAM_B_DOC_IDS)
    finally:
        store.index.delete(drop=True)
