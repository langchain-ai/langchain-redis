"""Integration tests for filter-based deletion."""

from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings
from redisvl.query.filter import FilterExpression, Tag  # type: ignore[import]
from redisvl.redis.utils import hashify  # type: ignore[import]
from redisvl.schema import FieldTypes, IndexSchema  # type: ignore[import]

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


def _make_store(
    redis_url: str,
    *,
    index_name: Optional[str] = None,
    doc_id_prefix: str = "",
    **config_kwargs: Any,
) -> RedisVectorStore:
    store = RedisVectorStore(
        ConstantEmbeddings(),
        index_name=index_name or f"filter_ops_{uuid4().hex[:8]}",
        redis_url=redis_url,
        metadata_schema=METADATA_SCHEMA,
        **config_kwargs,
    )
    doc_ids = [f"{doc_id_prefix}{doc_id}" for doc_id in ALL_DOC_IDS]
    texts = [f"document {doc_id}" for doc_id in doc_ids]
    metadatas = [
        {
            DOC_ID_FIELD: doc_id,
            TEAM_FIELD: TEAM_A if position < len(TEAM_A_DOC_IDS) else TEAM_B,
        }
        for position, doc_id in enumerate(doc_ids)
    ]
    store.add_texts(texts, metadatas=metadatas)
    return store


def _make_custom_marker_store(
    redis_url: str,
    marker_type: Optional[str],
    storage_type: str = "hash",
) -> RedisVectorStore:
    """Create a store with a missing or legacy internal index marker."""
    index_name = f"filter_ops_custom_{uuid4().hex[:8]}"
    fields: List[Dict[str, Any]] = [
        {"name": "text", "type": "text"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "dims": DIMS,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32",
            },
        },
        {"name": "_metadata_json", "type": "text"},
        *METADATA_SCHEMA,
    ]
    if marker_type is not None:
        fields.append({"name": "_index_name", "type": marker_type})

    schema = IndexSchema.from_dict(
        {
            "index": {
                "name": index_name,
                "prefix": index_name,
                "storage_type": storage_type,
            },
            "fields": fields,
        }
    )
    store = RedisVectorStore(
        ConstantEmbeddings(),
        schema=schema,
        redis_url=redis_url,
    )
    store.add_texts(
        ["document custom"],
        metadatas=[{DOC_ID_FIELD: "custom", TEAM_FIELD: TEAM_A}],
        keys=["custom"],
    )
    return store


def _remaining_doc_ids(store: RedisVectorStore) -> Set[str]:
    docs = store.similarity_search(QUERY, k=20)
    return {doc.metadata[DOC_ID_FIELD] for doc in docs}


@pytest.mark.parametrize("storage_type", ["hash", "json"])
def test_delete_by_filter_removes_only_matching(
    redis_url: str, storage_type: str
) -> None:
    """Filter deletion removes exactly the matching documents."""
    store = _make_store(redis_url, storage_type=storage_type)
    try:
        deleted = store.delete_by_filter(Tag(TEAM_FIELD) == TEAM_A)
        assert deleted == len(TEAM_A_DOC_IDS)
        assert _remaining_doc_ids(store) == set(TEAM_B_DOC_IDS)
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


def test_no_match_returns_zero(redis_url: str) -> None:
    """An explicit filter matching nothing deletes nothing and returns zero."""
    store = _make_store(redis_url)
    try:
        assert store.delete_by_filter(Tag(TEAM_FIELD) == GHOST_TEAM) == 0
        assert _remaining_doc_ids(store) == set(ALL_DOC_IDS)
    finally:
        store.index.delete(drop=True)


def test_delete_filter_overload_raises_without_deleting(redis_url: str) -> None:
    """The ambiguous delete overload fails without changing Redis data."""
    store = _make_store(redis_url)
    try:
        before = _remaining_doc_ids(store)

        with pytest.raises(ValueError, match="delete_by_filter"):
            store.delete(filter=Tag(TEAM_FIELD) == TEAM_A)

        assert _remaining_doc_ids(store) == before
    finally:
        store.index.delete(drop=True)


@pytest.mark.parametrize(
    ("storage_type", "sibling_name"),
    [
        pytest.param("hash", lambda name: f"{name}-archive", id="hash-hyphen"),
        pytest.param("hash", str.upper, id="hash-case"),
        pytest.param("hash", lambda name: f"{name},archive", id="hash-tag-separator"),
        pytest.param("hash", lambda name: f"{name}|archive", id="hash-tag-operator"),
        pytest.param("hash", lambda name: f"{name} archive", id="hash-space"),
        pytest.param("hash", lambda name: f"{name}-東京", id="hash-unicode"),
        pytest.param("json", lambda name: f"{name}|archive", id="json-tag-operator"),
    ],
)
def test_shared_prefix_indexes_are_exactly_isolated(
    redis_url: str,
    storage_type: str,
    sibling_name: Callable[[str], str],
) -> None:
    """Index-name syntax cannot broaden a filter deletion into a sibling."""
    shared_prefix = f"shared_{uuid4().hex[:8]}"
    index_name = f"scope_{uuid4().hex[:8]}"
    store_a = _make_store(
        redis_url,
        index_name=index_name,
        doc_id_prefix="a-",
        key_prefix=shared_prefix,
        storage_type=storage_type,
    )
    store_b = _make_store(
        redis_url,
        index_name=sibling_name(index_name),
        doc_id_prefix="b-",
        key_prefix=shared_prefix,
        storage_type=storage_type,
    )
    try:
        store_a_ids = {f"a-{doc_id}" for doc_id in ALL_DOC_IDS}
        store_b_ids = {f"b-{doc_id}" for doc_id in ALL_DOC_IDS}
        assert _remaining_doc_ids(store_a) == store_a_ids
        assert _remaining_doc_ids(store_b) == store_b_ids

        deleted = store_a.delete_by_filter(Tag(TEAM_FIELD) == TEAM_A)
        assert deleted == len(TEAM_A_DOC_IDS)

        assert _remaining_doc_ids(store_a) == {
            f"a-{doc_id}" for doc_id in TEAM_B_DOC_IDS
        }
        # store_b's documents, including its TEAM_A ones, must be untouched
        assert _remaining_doc_ids(store_b) == store_b_ids
    finally:
        store_a.index.delete(drop=True)
        store_b.index.delete(drop=True)


@pytest.mark.parametrize("storage_type", ["hash", "json"])
def test_shared_prefix_direct_ids_are_owned_by_their_index(
    redis_url: str,
    storage_type: str,
) -> None:
    """Direct ID reads and deletes cannot cross a shared-prefix boundary."""
    shared_prefix = f"shared_ids_{uuid4().hex[:8]}"
    store_a = RedisVectorStore(
        ConstantEmbeddings(),
        index_name=f"direct_a_{uuid4().hex[:8]}",
        key_prefix=shared_prefix,
        redis_url=redis_url,
        metadata_schema=METADATA_SCHEMA,
        storage_type=storage_type,
    )
    store_b = RedisVectorStore(
        ConstantEmbeddings(),
        index_name=f"direct_b_{uuid4().hex[:8]}",
        key_prefix=shared_prefix,
        redis_url=redis_url,
        metadata_schema=METADATA_SCHEMA,
        storage_type=storage_type,
    )
    owned_id = "owned"
    foreign_id = "foreign"
    try:
        store_a.add_texts(
            ["owned document"],
            metadatas=[{DOC_ID_FIELD: owned_id, TEAM_FIELD: TEAM_A}],
            keys=[owned_id],
        )
        store_b.add_texts(
            ["foreign document"],
            metadatas=[{DOC_ID_FIELD: foreign_id, TEAM_FIELD: TEAM_B}],
            keys=[foreign_id],
        )

        assert [doc.id for doc in store_a.get_by_ids([owned_id, foreign_id])] == [
            owned_id
        ]
        assert store_a.delete([foreign_id]) is False
        assert [doc.id for doc in store_b.get_by_ids([foreign_id])] == [foreign_id]

        assert store_a.delete([owned_id, foreign_id]) is True
        assert store_a.get_by_ids([owned_id, foreign_id]) == []
        assert [doc.id for doc in store_b.get_by_ids([foreign_id])] == [foreign_id]
    finally:
        store_a.index.delete(drop=True)
        store_b.index.delete(drop=True)


def test_shared_prefix_deletion_uses_search_index_name(
    redis_url: str,
) -> None:
    """A stale config name cannot redirect deletion to a sibling index."""
    shared_prefix = f"shared_identity_{uuid4().hex[:8]}"
    store_a = _make_store(
        redis_url,
        index_name=f"identity_a_{uuid4().hex[:8]}",
        doc_id_prefix="a-",
        key_prefix=shared_prefix,
    )
    store_b = _make_store(
        redis_url,
        index_name=f"identity_b_{uuid4().hex[:8]}",
        doc_id_prefix="b-",
        key_prefix=shared_prefix,
    )
    try:
        store_a.config.index_name = store_b.index.name

        deleted = store_a.delete_by_filter(Tag(TEAM_FIELD) == TEAM_A)

        assert deleted == len(TEAM_A_DOC_IDS)
        assert _remaining_doc_ids(store_a) == {
            f"a-{doc_id}" for doc_id in TEAM_B_DOC_IDS
        }
        assert _remaining_doc_ids(store_b) == {f"b-{doc_id}" for doc_id in ALL_DOC_IDS}
    finally:
        store_a.index.delete(drop=True)
        store_b.index.delete(drop=True)


@pytest.mark.parametrize("storage_type", ["hash", "json"])
def test_metadata_cannot_override_shared_prefix_ownership_marker(
    redis_url: str,
    storage_type: str,
) -> None:
    """Conflicting metadata cannot transfer a document to a sibling index."""
    shared_prefix = f"shared_metadata_{uuid4().hex[:8]}"
    store_a = _make_store(
        redis_url,
        index_name=f"metadata_a_{uuid4().hex[:8]}",
        doc_id_prefix="a-",
        key_prefix=shared_prefix,
        storage_type=storage_type,
    )
    store_b = _make_store(
        redis_url,
        index_name=f"metadata_b_{uuid4().hex[:8]}",
        doc_id_prefix="b-",
        key_prefix=shared_prefix,
        storage_type=storage_type,
    )
    caller_marker = hashify(store_b.index.name)
    protected_doc_id = "protected"
    try:
        store_a.add_texts(
            ["protected document"],
            metadatas=[
                {
                    "_index_name": caller_marker,
                    DOC_ID_FIELD: protected_doc_id,
                    TEAM_FIELD: TEAM_A,
                }
            ],
            keys=[protected_doc_id],
        )

        docs = store_a.similarity_search(
            QUERY, k=1, filter=Tag(DOC_ID_FIELD) == protected_doc_id
        )
        assert len(docs) == 1
        assert docs[0].metadata["_index_name"] == caller_marker

        assert store_b.delete_by_filter(Tag(DOC_ID_FIELD) == protected_doc_id) == 0
        assert store_a.delete_by_filter(Tag(DOC_ID_FIELD) == protected_doc_id) == 1
    finally:
        store_a.index.delete(drop=True)
        store_b.index.delete(drop=True)


@pytest.mark.parametrize(
    "marker_type",
    [pytest.param(None, id="missing"), pytest.param("text", id="legacy-text")],
)
def test_incompatible_marker_schema_is_searchable_but_not_filter_deletable(
    redis_url: str,
    marker_type: Optional[str],
) -> None:
    """Existing/custom schemas retain reads while destructive filters fail closed."""
    store = _make_custom_marker_store(redis_url, marker_type)
    try:
        assert _remaining_doc_ids(store) == {"custom"}

        with pytest.raises(ValueError, match="requires an '_index_name' TAG field"):
            store.delete_by_filter(Tag(TEAM_FIELD) == TEAM_A)

        assert _remaining_doc_ids(store) == {"custom"}
        assert [doc.id for doc in store.get_by_ids(["custom"])] == ["custom"]
        assert store.delete(["custom"]) is True
        assert store.get_by_ids(["custom"]) == []
    finally:
        store.index.delete(drop=True)


@pytest.mark.parametrize("storage_type", ["hash", "json"])
def test_reopened_legacy_index_uses_live_schema(
    redis_url: str,
    storage_type: str,
) -> None:
    """Normal construction rehydrates a retained legacy TEXT schema."""
    legacy_store = _make_custom_marker_store(redis_url, "text", storage_type)
    reopened_store = RedisVectorStore(
        ConstantEmbeddings(),
        index_name=legacy_store.index.name,
        redis_url=redis_url,
        metadata_schema=METADATA_SCHEMA,
        storage_type=storage_type,
    )
    try:
        assert reopened_store.index.schema.fields["_index_name"].type == FieldTypes.TEXT
        assert _remaining_doc_ids(reopened_store) == {"custom"}

        reopened_store.add_texts(
            ["document new"],
            metadatas=[{DOC_ID_FIELD: "new", TEAM_FIELD: TEAM_B}],
        )
        assert _remaining_doc_ids(reopened_store) == {"custom", "new"}

        with pytest.raises(ValueError, match="requires an '_index_name' TAG field"):
            reopened_store.delete_by_filter(Tag(TEAM_FIELD) == TEAM_A)

        assert _remaining_doc_ids(reopened_store) == {"custom", "new"}
        assert [doc.id for doc in reopened_store.get_by_ids(["custom"])] == ["custom"]
        assert reopened_store.delete(["custom"]) is True
        assert reopened_store.get_by_ids(["custom"]) == []
        assert _remaining_doc_ids(reopened_store) == {"new"}
    finally:
        legacy_store.index.delete(drop=True)


def test_raw_tag_marker_requires_migration_before_filter_deletion(
    redis_url: str,
) -> None:
    """Raw legacy TAG values fail safely until rewritten to the hashed marker."""
    store = _make_custom_marker_store(redis_url, "tag")
    raw_id = "custom"
    current_id = "current"
    raw_key = f"{store.config.primary_prefix}:{raw_id}"
    try:
        store.add_texts(
            ["document current"],
            metadatas=[{DOC_ID_FIELD: current_id, TEAM_FIELD: TEAM_A}],
            keys=[current_id],
        )
        store.index.client.hset(raw_key, "_index_name", store.config.index_name)

        assert [doc.id for doc in store.get_by_ids([raw_id, current_id])] == [
            current_id
        ]
        assert store.delete([raw_id]) is False
        assert store.index.client.exists(raw_key) == 1

        assert store.delete_by_filter(Tag(TEAM_FIELD) == TEAM_A) == 1
        assert store.get_by_ids([raw_id, current_id]) == []

        store.index.client.hset(raw_key, "_index_name", hashify(store.index.name))
        assert store.delete_by_filter(Tag(TEAM_FIELD) == TEAM_A) == 1
        assert store.get_by_ids([raw_id]) == []
    finally:
        store.index.delete(drop=True)


@pytest.mark.parametrize("legacy_key_format", [True, False], ids=["legacy", "modern"])
def test_reopened_json_index_uses_live_storage_and_prefix(
    redis_url: str,
    legacy_key_format: bool,
) -> None:
    """A reopened index does not retain stale HASH or key-prefix defaults."""
    index_name = f"filter_ops_reopen_{uuid4().hex[:8]}"
    key_prefix = f"filter_docs_{uuid4().hex[:8]}"
    original_store = RedisVectorStore(
        ConstantEmbeddings(),
        index_name=index_name,
        key_prefix=key_prefix,
        redis_url=redis_url,
        metadata_schema=METADATA_SCHEMA,
        storage_type="json",
        legacy_key_format=legacy_key_format,
    )
    try:
        original_ids = original_store.add_texts(
            ["document original"],
            metadatas=[{DOC_ID_FIELD: "original", TEAM_FIELD: TEAM_A}],
        )
        reopened_store = RedisVectorStore.from_existing_index(
            index_name=index_name,
            embedding=ConstantEmbeddings(),
            redis_url=redis_url,
            legacy_key_format=legacy_key_format,
        )
        assert reopened_store.config.storage_type == "json"
        assert reopened_store.config.key_prefix == key_prefix
        assert [
            doc.page_content for doc in reopened_store.get_by_ids(original_ids)
        ] == ["document original"]

        new_ids = reopened_store.add_texts(
            ["document new"],
            metadatas=[{DOC_ID_FIELD: "new", TEAM_FIELD: TEAM_B}],
            keys=["new"],
        )
        assert new_ids == ["new"]
        assert [doc.page_content for doc in reopened_store.get_by_ids(new_ids)] == [
            "document new"
        ]
        assert reopened_store.delete(ids=new_ids) is True
        assert reopened_store.get_by_ids(new_ids) == []
    finally:
        original_store.index.delete(drop=True)


def test_delete_by_filter_works_after_reopening_generated_index(
    redis_url: str,
) -> None:
    """A reopened store accepts the exact TAG marker reported by Redis."""
    store = _make_store(redis_url)
    reopened_store = RedisVectorStore.from_existing_index(
        index_name=store.config.index_name,
        embedding=ConstantEmbeddings(),
        redis_url=redis_url,
    )
    try:
        deleted = reopened_store.delete_by_filter(Tag(TEAM_FIELD) == TEAM_A)

        assert deleted == len(TEAM_A_DOC_IDS)
        assert _remaining_doc_ids(reopened_store) == set(TEAM_B_DOC_IDS)
    finally:
        store.index.delete(drop=True)


def test_rejected_match_all_filter_leaves_documents_untouched(
    redis_url: str,
) -> None:
    """The local guard rejects match-all before RedisVL sees a scoped filter."""
    store = _make_store(redis_url)
    try:
        with pytest.raises(ValueError, match="refuses filters that match all"):
            store.delete_by_filter(FilterExpression("*"))

        assert _remaining_doc_ids(store) == set(ALL_DOC_IDS)
    finally:
        store.index.delete(drop=True)
