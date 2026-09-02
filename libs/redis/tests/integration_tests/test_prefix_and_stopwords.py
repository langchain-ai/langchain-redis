"""Integration tests for multi-prefix indexes, index-level stopwords, and
wildcard tag filters."""

import json
from typing import Any, List, Set, cast
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings
from redisvl.query.filter import Tag  # type: ignore[import]
from redisvl.redis.utils import array_to_buffer  # type: ignore[import]

from langchain_redis import RedisVectorStore

DIMS = 4
DOC_ID_FIELD = "doc_id"
CATEGORY_FIELD = "category"
METADATA_SCHEMA = [
    {"name": DOC_ID_FIELD, "type": "tag"},
    {"name": CATEGORY_FIELD, "type": "tag"},
]
QUERY = "any query"


class ConstantEmbeddings(Embeddings):
    """All texts share one vector; these tests assert sets, not rankings."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * DIMS for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * DIMS


def _make_store(redis_url: str, **config_kwargs: Any) -> RedisVectorStore:
    return RedisVectorStore(
        ConstantEmbeddings(),
        index_name=f"prefix_test_{uuid4().hex[:8]}",
        redis_url=redis_url,
        metadata_schema=METADATA_SCHEMA,
        embedding_dimensions=DIMS,
        **config_kwargs,
    )


def _doc_ids(store: RedisVectorStore, **search_kwargs: Any) -> Set[str]:
    docs = store.similarity_search(QUERY, k=20, **search_kwargs)
    return {doc.metadata[DOC_ID_FIELD] for doc in docs}


def test_multi_prefix_index_spans_namespaces(redis_url: str) -> None:
    """Searches cover every configured prefix; writes land under the first."""
    run_id = uuid4().hex[:8]
    prefix_a, prefix_b = f"tenant_a_{run_id}", f"tenant_b_{run_id}"
    store = _make_store(
        redis_url, key_prefix=[prefix_a, prefix_b], legacy_key_format=False
    )
    try:
        store.add_texts(
            ["first doc", "second doc"],
            metadatas=[
                {DOC_ID_FIELD: "written_1", CATEGORY_FIELD: "native"},
                {DOC_ID_FIELD: "written_2", CATEGORY_FIELD: "native"},
            ],
        )
        # Plant a conforming document under the second prefix, as an outside
        # writer (e.g. another application) would.
        store.index.load(
            [
                {
                    "text": "planted doc",
                    "embedding": array_to_buffer([0.1] * DIMS, "float32"),
                    "_index_name": store.config.index_name,
                    "_metadata_json": json.dumps(
                        {DOC_ID_FIELD: "planted", CATEGORY_FIELD: "external"}
                    ),
                    DOC_ID_FIELD: "planted",
                    CATEGORY_FIELD: "external",
                }
            ],
            keys=[f"{prefix_b}:planted"],
        )

        assert _doc_ids(store) == {"written_1", "written_2", "planted"}

        client = store.config.redis()
        keys_a = cast(List[Any], client.keys(f"{prefix_a}:*"))
        keys_b = cast(List[Any], client.keys(f"{prefix_b}:*"))
        assert len(keys_a) == 2
        assert len(keys_b) == 1
    finally:
        store.index.delete(drop=True)


def test_stopwords_disabled_reaches_server(redis_url: str) -> None:
    """stopwords=[] creates the index with STOPWORDS 0 server-side."""
    store = _make_store(redis_url, stopwords=[])
    try:
        info = store.config.redis().ft(store.config.index_name).info()
        assert info["stopwords_list"] == []
    finally:
        store.index.delete(drop=True)


@pytest.mark.xfail(
    reason="redisvl joins text queries and filters with a literal ' AND ' token, "
    "which is only parseable because 'and' is a default stopword; on a "
    "STOPWORDS-0 index filtered text queries match nothing. Remove this marker "
    "once the redisvl fix ships.",
    strict=True,
)
def test_stopwords_disabled_makes_stopwords_searchable(redis_url: str) -> None:
    """stopwords=[] makes default stopword terms discriminating search terms."""
    store = _make_store(redis_url, stopwords=[])
    try:
        store.add_texts(
            ["the quick brown fox", "lazy dog"],
            metadatas=[
                {DOC_ID_FIELD: "with_the", CATEGORY_FIELD: "misc"},
                {DOC_ID_FIELD: "without_the", CATEGORY_FIELD: "misc"},
            ],
        )
        found = store.full_text_search("the", k=5, stopwords=None)
        assert [doc.metadata[DOC_ID_FIELD] for doc in found] == ["with_the"]
    finally:
        store.index.delete(drop=True)


def test_custom_stopwords_make_default_stopwords_searchable(redis_url: str) -> None:
    """A custom stopword list replaces the defaults: "the" becomes searchable.

    Client-side stopword stripping is disabled in the queries so the test
    isolates the index-level behavior. ("and" stays in the custom list to
    keep this test independent of the upstream ' AND '-join bug.)
    """
    texts = ["the quick brown fox", "lazy dog"]
    metadatas = [
        {DOC_ID_FIELD: "with_the", CATEGORY_FIELD: "misc"},
        {DOC_ID_FIELD: "without_the", CATEGORY_FIELD: "misc"},
    ]

    default_store = _make_store(redis_url)
    custom_store = _make_store(redis_url, stopwords=["and"])
    try:
        default_store.add_texts(texts, metadatas=metadatas)
        custom_store.add_texts(texts, metadatas=metadatas)

        # Custom stopwords: "the" is indexed and discriminates between docs.
        found = custom_store.full_text_search("the", k=5, stopwords=None)
        assert [doc.metadata[DOC_ID_FIELD] for doc in found] == ["with_the"]

        # Default stopwords: "the" is dropped from the query, which then
        # cannot discriminate — it degrades to a match-all within the index.
        blind = default_store.full_text_search("the", k=5, stopwords=None)
        assert {doc.metadata[DOC_ID_FIELD] for doc in blind} == {
            "with_the",
            "without_the",
        }
    finally:
        default_store.index.delete(drop=True)
        custom_store.index.delete(drop=True)


def test_wildcard_tag_filter_narrows_search(redis_url: str) -> None:
    """Tag wildcard patterns (Tag % "elec*") pass through to the server."""
    store = _make_store(redis_url)
    try:
        store.add_texts(
            ["tv", "toaster", "shovel"],
            metadatas=[
                {DOC_ID_FIELD: "tv", CATEGORY_FIELD: "electronics"},
                {DOC_ID_FIELD: "toaster", CATEGORY_FIELD: "electrical"},
                {DOC_ID_FIELD: "shovel", CATEGORY_FIELD: "garden"},
            ],
        )
        matched = _doc_ids(store, filter=Tag(CATEGORY_FIELD) % "elec*")
        assert matched == {"tv", "toaster"}
    finally:
        store.index.delete(drop=True)
