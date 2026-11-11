from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings
from redis import Redis
from redis.commands.search.query import Query
from redisvl.index import SearchIndex  # type: ignore[import]
from redisvl.redis.utils import convert_bytes, hashify  # type: ignore[import]
from redisvl.schema import IndexSchema  # type: ignore[import]

from langchain_redis import RedisConfig, RedisVectorStore, migrate_index_ownership

DIMS = 4


class ConstantEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * DIMS for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * DIMS


@pytest.fixture
def embeddings() -> Embeddings:
    return ConstantEmbeddings()


@pytest.fixture
def client(redis_url: str) -> Iterator[Redis]:
    with Redis.from_url(redis_url, decode_responses=False) as connection:
        yield connection


@pytest.fixture
def make_legacy_store(
    client: Redis, embeddings: Embeddings
) -> Iterator[Callable[..., RedisVectorStore]]:
    stores: List[RedisVectorStore] = []

    def create(
        storage_type: str = "hash",
        marker_type: Optional[str] = None,
        ids: Sequence[str] = ("one", "two"),
        vector_attrs: Optional[Dict[str, Any]] = None,
    ) -> RedisVectorStore:
        name = f"ownership_migration_{uuid4().hex[:8]}"
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
                    **(vector_attrs or {}),
                },
            },
            {"name": "_metadata_json", "type": "text"},
            {"name": "doc_id", "type": "tag"},
        ]
        if marker_type is not None:
            fields.append({"name": "_index_name", "type": marker_type})
        schema = IndexSchema.from_dict(
            {
                "index": {
                    "name": name,
                    "prefix": name,
                    "storage_type": storage_type,
                },
                "fields": fields,
            }
        )
        store = RedisVectorStore(embeddings, schema=schema, redis_client=client)
        stores.append(store)
        store.add_texts(
            [f"document {doc_id}" for doc_id in ids],
            metadatas=[{"doc_id": doc_id} for doc_id in ids],
            keys=list(ids),
        )
        return store

    try:
        yield create
    finally:
        for store in stores:
            if store.index.exists():
                store.index.delete(drop=True)


def _read_record(client: Redis, store: RedisVectorStore, key: str) -> Dict[str, Any]:
    record = (
        client.json().get(key)
        if store.index.schema.index.storage_type.value == "json"
        else client.hgetall(key)
    )
    return convert_bytes(record)


def test_migration_supports_resp2(
    redis_url: str,
    make_legacy_store: Callable[..., RedisVectorStore],
) -> None:
    store = make_legacy_store(ids=("one",))
    # Exercise RESP2 normalization while checking another index's disjoint prefix.
    make_legacy_store(ids=("other",))
    key = store.index.key("one")

    with Redis.from_url(redis_url, protocol=2) as client:
        preview = migrate_index_ownership(store.index.name, redis_client=client)
        result = migrate_index_ownership(
            store.index.name, redis_client=client, dry_run=False
        )

        assert (preview.schema_action, preview.would_update) == ("alter", 1)
        assert (result.updated, result.completed) == (1, True)
        assert client.hget(key, "_index_name") == hashify(store.index.name).encode()


@pytest.mark.parametrize("dry_run", [True, False], ids=["preview", "apply"])
def test_overlapping_prefixes_are_rejected_before_mutation(
    client: Redis, dry_run: bool
) -> None:
    run_id = uuid4().hex[:8]
    prefix = f"ownership_overlap_{run_id}:"
    schemas = [
        IndexSchema.from_dict(
            {
                "index": {"name": f"overlap_{run_id}_{suffix}", "prefix": key_prefix},
                "fields": [{"name": "text", "type": "text"}],
            }
        )
        for suffix, key_prefix in (("broad", prefix), ("nested", f"{prefix}b:"))
    ]
    indexes = [SearchIndex(schema=schema, redis_client=client) for schema in schemas]
    keys = [f"{prefix}a:one", f"{prefix}b:one"]
    try:
        for index in indexes:
            index.create()
        client.hset(keys[0], mapping={"text": "broad"})
        client.hset(keys[1], mapping={"text": "nested"})
        before_attributes = [
            convert_bytes(client.ft(index.name).info())["attributes"]
            for index in indexes
        ]
        before_documents = [client.hgetall(key) for key in keys]

        with pytest.raises(ValueError, match="overlaps"):
            migrate_index_ownership(
                indexes[0].name, redis_client=client, dry_run=dry_run
            )

        assert [
            convert_bytes(client.ft(index.name).info())["attributes"]
            for index in indexes
        ] == before_attributes
        assert [client.hgetall(key) for key in keys] == before_documents
    finally:
        for index in indexes:
            if index.exists():
                index.delete(drop=False)
        client.delete(*keys)


@pytest.mark.parametrize(
    ("storage_type", "marker_type", "action"),
    [
        pytest.param("hash", None, "alter", id="missing-hash"),
        pytest.param("json", None, "alter", id="missing-json"),
        pytest.param("json", "text", "rebuild", id="legacy-text-json"),
    ],
)
def test_migration_preserves_documents_and_is_rerunnable(
    client: Redis,
    embeddings: Embeddings,
    make_legacy_store: Callable[..., RedisVectorStore],
    storage_type: str,
    marker_type: Optional[str],
    action: str,
) -> None:
    store = make_legacy_store(storage_type, marker_type)
    keys = [store.index.key(doc_id) for doc_id in ("one", "two")]
    for key in keys:
        client.expire(key, 300)
    before = [_read_record(client, store, key) for key in keys]
    ttls_before = [client.pttl(key) for key in keys]
    schema_before = SearchIndex.from_existing(
        store.index.name, redis_client=client
    ).schema.to_dict()
    source = store.config.to_index_schema()

    preview = migrate_index_ownership(
        store.index.name, redis_client=client, source_schema=source
    )
    assert preview.dry_run
    assert (preview.schema_action, preview.would_update, preview.updated) == (
        action,
        2,
        0,
    )
    assert [_read_record(client, store, key) for key in keys] == before
    assert (
        SearchIndex.from_existing(
            store.index.name, redis_client=client
        ).schema.to_dict()
        == schema_before
    )

    result = migrate_index_ownership(
        store.index.name,
        redis_client=client,
        source_schema=source,
        dry_run=False,
        batch_size=1,
    )
    assert (result.schema_action, result.updated, result.completed) == (action, 2, True)
    after = [_read_record(client, store, key) for key in keys]
    for old, new, ttl_before, key in zip(before, after, ttls_before, keys):
        assert new["_index_name"] == hashify(store.index.name)
        assert {k: v for k, v in new.items() if k != "_index_name"} == {
            k: v for k, v in old.items() if k != "_index_name"
        }
        ttl_after = client.pttl(key)
        assert 0 < ttl_after <= ttl_before

    live = RedisVectorStore.from_existing_index(
        store.index.name, embeddings, redis_client=client
    )
    assert {
        doc.metadata["doc_id"] for doc in live.similarity_search("document", k=10)
    } == {
        "one",
        "two",
    }

    rerun = migrate_index_ownership(
        store.index.name, redis_client=client, dry_run=False
    )
    assert (
        rerun.schema_action,
        rerun.would_update,
        rerun.updated,
        rerun.completed,
    ) == (
        "none",
        0,
        0,
        True,
    )
    assert [_read_record(client, store, key) for key in keys] == after


def test_expiry_after_preflight_reports_incomplete_and_allows_retry(
    client: Redis,
    make_legacy_store: Callable[..., RedisVectorStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_legacy_store()
    expired, remaining = [store.index.key(doc_id) for doc_id in ("one", "two")]
    update_by_filter = SearchIndex.update_by_filter

    def expire_before_update(index: SearchIndex, *args: Any, **kwargs: Any) -> Any:
        # A past timestamp expires the key immediately, without a real wait.
        client.pexpireat(expired, 1)
        return update_by_filter(index, *args, **kwargs)

    monkeypatch.setattr(SearchIndex, "update_by_filter", expire_before_update)
    result = migrate_index_ownership(
        store.index.name, redis_client=client, dry_run=False
    )
    assert (result.would_update, result.updated, result.completed) == (2, 1, False)
    assert client.exists(expired) == 0
    assert client.hget(remaining, "_index_name") == hashify(store.index.name).encode()

    retry = migrate_index_ownership(
        store.index.name, redis_client=client, dry_run=False
    )
    assert (retry.would_update, retry.updated, retry.completed) == (0, 0, True)


def test_json_backfill_handles_null_and_skips_already_migrated_documents(
    client: Redis, make_legacy_store: Callable[..., RedisVectorStore]
) -> None:
    store = make_legacy_store(storage_type="json", marker_type="text")
    null_key, migrated_key = [store.index.key(doc_id) for doc_id in ("one", "two")]
    marker = hashify(store.index.name)
    client.json().set(null_key, "$._index_name", None)
    client.json().set(migrated_key, "$._index_name", marker)

    source = store.config.to_index_schema()
    preview = migrate_index_ownership(
        store.index.name, redis_client=client, source_schema=source
    )
    assert preview.would_update == 1
    result = migrate_index_ownership(
        store.index.name, redis_client=client, source_schema=source, dry_run=False
    )
    assert (result.updated, result.completed) == (1, True)
    assert client.json().get(null_key, "$._index_name") == [marker]
    assert client.json().get(migrated_key, "$._index_name") == [marker]


def test_migration_rejects_conflicting_ownership_before_mutation(
    client: Redis, make_legacy_store: Callable[..., RedisVectorStore]
) -> None:
    store = make_legacy_store(marker_type="text", ids=("one",))
    key = store.index.key("one")
    client.hset(key, "_index_name", hashify("another-index"))
    before = client.hgetall(key)
    attributes = convert_bytes(client.ft(store.index.name).info())["attributes"]

    with pytest.raises(ValueError, match="another index's marker"):
        migrate_index_ownership(
            store.index.name,
            redis_client=client,
            source_schema=store.config.to_index_schema(),
            dry_run=False,
        )
    assert client.hgetall(key) == before
    assert convert_bytes(client.ft(store.index.name).info())["attributes"] == attributes


def test_incomplete_preflight_cursor_is_rejected_before_mutation(
    client: Redis,
    make_legacy_store: Callable[..., RedisVectorStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_legacy_store(ids=("one", "two"))
    keys = [store.index.key(doc_id) for doc_id in ("one", "two")]
    client.hset(keys[1], "_index_name", hashify("another-index"))
    before = [client.hgetall(key) for key in keys]
    attributes = convert_bytes(client.ft(store.index.name).info())["attributes"]
    monkeypatch.setattr(
        "langchain_redis.migrations._index_keys", lambda *args: [keys[0]]
    )

    with pytest.raises(RuntimeError, match="different number of documents"):
        migrate_index_ownership(
            store.index.name,
            redis_client=client,
            dry_run=False,
        )

    assert convert_bytes(client.ft(store.index.name).info())["attributes"] == attributes
    assert [client.hgetall(key) for key in keys] == before


@pytest.mark.parametrize(
    ("alias_template", "error"),
    [
        pytest.param(
            "{name}", "cannot be supplied as its own alias", id="index-name"
        ),
        pytest.param("{name}-current", "does not point", id="missing"),
        pytest.param("", "must be non-empty", id="empty"),
    ],
)
@pytest.mark.parametrize("dry_run", [True, False], ids=["preview", "apply"])
def test_rebuild_rejects_invalid_alias_before_mutation(
    client: Redis,
    make_legacy_store: Callable[..., RedisVectorStore],
    dry_run: bool,
    alias_template: str,
    error: str,
) -> None:
    store = make_legacy_store(storage_type="json", marker_type="text", ids=("one",))
    name, key = store.index.name, store.index.key("one")
    alias = f"{name}_current"
    client.ft(name).aliasadd(alias)
    attributes = convert_bytes(client.ft(name).info())["attributes"]
    before = client.json().get(key)

    with pytest.raises(ValueError, match=error):
        migrate_index_ownership(
            name,
            redis_client=client,
            source_schema=store.config.to_index_schema(),
            aliases=[alias_template.format(name=name)],
            dry_run=dry_run,
        )

    assert convert_bytes(client.ft(name).info())["attributes"] == attributes
    assert client.json().get(key) == before
    assert convert_bytes(client.ft(alias).info())["index_name"] == name


def test_rebuild_restores_supplied_alias(
    client: Redis, make_legacy_store: Callable[..., RedisVectorStore]
) -> None:
    store = make_legacy_store(storage_type="json", marker_type="text")
    alias = f"{store.index.name}_current"
    client.ft(store.index.name).aliasadd(alias)

    result = migrate_index_ownership(
        store.index.name,
        redis_client=client,
        source_schema=store.config.to_index_schema(),
        aliases=[alias],
        dry_run=False,
    )
    assert result.completed
    assert convert_bytes(client.ft(alias).info())["index_name"] == store.index.name


def test_rerun_restores_alias_missing_after_rebuild(
    client: Redis, make_legacy_store: Callable[..., RedisVectorStore]
) -> None:
    store = make_legacy_store(marker_type="text")
    name = store.index.name
    alias = f"{name}_current"
    client.ft(name).aliasadd(alias)

    # Recreate the index without restoring its alias, matching an interruption
    # between the migration's rebuild and alias restoration steps.
    target = store.config.to_index_schema().model_copy(deep=True)
    target.fields.pop("_index_name")
    target.add_field(
        {"name": "_index_name", "type": "tag", "attrs": {"case_sensitive": True}}
    )
    SearchIndex(schema=target, redis_client=client).create(overwrite=True, drop=False)

    result = migrate_index_ownership(
        name,
        redis_client=client,
        aliases=[alias],
        dry_run=False,
    )

    assert (result.schema_action, result.updated, result.completed) == (
        "none",
        2,
        True,
    )
    assert convert_bytes(client.ft(alias).info())["index_name"] == name


def test_alias_pointing_to_another_index_is_rejected(
    client: Redis, make_legacy_store: Callable[..., RedisVectorStore]
) -> None:
    store = make_legacy_store(marker_type="text", ids=("one",))
    other = make_legacy_store(ids=("other",))
    alias = f"{store.index.name}_current"
    client.ft(other.index.name).aliasadd(alias)

    with pytest.raises(ValueError, match="points to another index"):
        migrate_index_ownership(
            store.index.name,
            redis_client=client,
            source_schema=store.config.to_index_schema(),
            aliases=[alias],
            dry_run=False,
        )

    assert convert_bytes(client.ft(alias).info())["index_name"] == other.index.name


def test_rebuild_preserves_hash_field_aliases(
    client: Redis, make_legacy_store: Callable[..., RedisVectorStore]
) -> None:
    store = make_legacy_store(marker_type="text", ids=("one",))
    name, key = store.index.name, store.index.key("one")
    client.execute_command(
        "FT.ALTER",
        name,
        "SCHEMA",
        "ADD",
        "title",
        "AS",
        "display",
        "TEXT",
        "WEIGHT",
        2,
        "NOSTEM",
        "SORTABLE",
    )
    client.hset(key, "title", "hello")
    source = store.config.to_index_schema().model_copy(deep=True)
    source.add_field(
        {
            "name": "display",
            "type": "text",
            "attrs": {"weight": 2, "no_stem": True, "sortable": True},
        }
    )
    source.fields["display"].path = "title"
    before = client.hgetall(key)
    attributes = convert_bytes(client.ft(name).info())["attributes"]
    query = Query("@display:hello").no_content().dialect(2)
    assert [doc.id for doc in client.ft(name).search(query).docs] == [key]

    preview = migrate_index_ownership(name, redis_client=client, source_schema=source)
    assert preview.schema_action == "rebuild"
    assert convert_bytes(client.ft(name).info())["attributes"] == attributes
    assert client.hgetall(key) == before

    result = migrate_index_ownership(
        name, redis_client=client, source_schema=source, dry_run=False
    )
    assert (result.updated, result.completed) == (1, True)
    after_attributes = convert_bytes(client.ft(name).info())["attributes"]
    for attribute in attributes:
        identifier = (
            attribute["identifier"] if isinstance(attribute, dict) else attribute[1]
        )
        if identifier != "_index_name":
            assert attribute in after_attributes
    assert [doc.id for doc in client.ft(name).search(query).docs] == [key]
    assert client.hgetall(key) == {**before, b"_index_name": hashify(name).encode()}


def test_generated_config_supplies_rebuild_schema(
    client: Redis, embeddings: Embeddings
) -> None:
    name = f"ownership_generated_{uuid4().hex[:8]}"
    config = RedisConfig(
        index_name=name,
        key_prefix=name,
        redis_client=client,
        embedding_dimensions=DIMS,
        indexing_algorithm="HNSW",
        vector_attrs={"ef_runtime": 75, "epsilon": 0.2},
        legacy_key_format=False,
    )
    store = RedisVectorStore(embeddings, config=config)
    try:
        store.add_texts(["document one"], keys=["one"])
        key = store.index.key("one")
        legacy = config.to_index_schema().model_copy(deep=True)
        legacy.fields.pop("_index_name")
        legacy.add_field({"name": "_index_name", "type": "text"})
        SearchIndex(schema=legacy, redis_client=client).create(
            overwrite=True, drop=False
        )
        client.hset(key, "_index_name", name)

        mismatched = config.model_copy(update={"embedding_dimensions": DIMS * 2})
        with pytest.raises(ValueError, match="source.*match"):
            migrate_index_ownership(name, redis_client=client, source_config=mismatched)

        preview = migrate_index_ownership(
            name, redis_client=client, source_config=config
        )
        result = migrate_index_ownership(
            name, redis_client=client, source_config=config, dry_run=False
        )

        assert (preview.schema_action, preview.would_update) == ("rebuild", 1)
        assert (result.updated, result.completed) == (1, True)
        assert client.hget(key, "_index_name") == hashify(name).encode()
    finally:
        if store.index.exists():
            store.index.delete(drop=True)


@pytest.mark.parametrize("storage_type", ["hash", "json"])
@pytest.mark.parametrize(
    ("algorithm", "tuning"),
    [
        pytest.param("flat", {}, id="flat"),
        pytest.param("hnsw", {"ef_runtime": 75, "epsilon": 0.2}, id="hnsw"),
        pytest.param(
            "svs-vamana",
            {
                "compression": "LVQ8",
                "training_threshold": 2048,
                "search_window_size": 35,
                "epsilon": 0.2,
            },
            id="svs",
        ),
    ],
)
def test_vector_rebuild_requires_original_schema_and_preserves_tuning(
    client: Redis,
    make_legacy_store: Callable[..., RedisVectorStore],
    monkeypatch: pytest.MonkeyPatch,
    redis_server_version: tuple,
    storage_type: str,
    algorithm: str,
    tuning: Dict[str, Any],
) -> None:
    if algorithm == "svs-vamana" and redis_server_version < (8, 2):
        pytest.skip("SVS-VAMANA requires Redis >= 8.2")
    store = make_legacy_store(
        storage_type=storage_type,
        marker_type="text",
        ids=("one",),
        vector_attrs={"algorithm": algorithm, **tuning},
    )
    name, key = store.index.name, store.index.key("one")
    source = store.config.to_index_schema().model_copy(deep=True)
    source_before = source.to_dict()
    before = _read_record(client, store, key)
    attributes = convert_bytes(client.ft(name).info())["attributes"]

    with pytest.raises(ValueError, match="source_schema or source_config"):
        migrate_index_ownership(name, redis_client=client, dry_run=False)
    mismatched = source.model_copy(deep=True)
    mismatched.index.prefix = "wrong:"
    with pytest.raises(ValueError, match="source.*match"):
        migrate_index_ownership(
            name, redis_client=client, source_schema=mismatched, dry_run=False
        )
    assert convert_bytes(client.ft(name).info())["attributes"] == attributes
    assert _read_record(client, store, key) == before

    # FT.INFO/RedisVL cannot round-trip all tuning; inspect the replacement command.
    creates = []
    execute = client.execute_command

    def capture_create(*args: Any, **kwargs: Any) -> Any:
        if args[0] == "FT.CREATE":
            creates.append(args)
        return execute(*args, **kwargs)

    monkeypatch.setattr(client, "execute_command", capture_create)
    result = migrate_index_ownership(
        name, redis_client=client, source_schema=source, dry_run=False
    )
    assert (result.updated, result.completed) == (1, True)
    assert len(creates) == 1
    for option, expected in tuning.items():
        assert creates[0][creates[0].index(option.upper()) + 1] == expected
    assert source.to_dict() == source_before
    assert _read_record(client, store, key) == {**before, "_index_name": hashify(name)}


@pytest.mark.parametrize("dry_run", [True, False], ids=["preview", "apply"])
def test_indexall_rebuild_is_rejected_before_mutation(
    client: Redis, dry_run: bool
) -> None:
    name = f"ownership_indexall_{uuid4().hex[:8]}"
    key = f"{name}:one"
    client.execute_command(
        "FT.CREATE",
        name,
        "ON",
        "HASH",
        "PREFIX",
        1,
        f"{name}:",
        "INDEXALL",
        "ENABLE",
        "SCHEMA",
        "text",
        "TEXT",
        "_index_name",
        "TEXT",
    )
    try:
        client.hset(key, mapping={"text": "hello", "_index_name": name})
        before = client.hgetall(key)
        info = convert_bytes(client.ft(name).info())

        with pytest.raises(ValueError, match="INDEXALL enabled"):
            migrate_index_ownership(name, redis_client=client, dry_run=dry_run)

        after = convert_bytes(client.ft(name).info())
        assert after["index_definition"] == info["index_definition"]
        assert after["attributes"] == info["attributes"]
        assert client.hgetall(key) == before
    finally:
        client.ft(name).dropindex(delete_documents=True)
