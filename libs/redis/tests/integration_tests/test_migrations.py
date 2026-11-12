from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, cast
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings
from redis import Redis
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redisvl.exceptions import RedisSearchError  # type: ignore[import]
from redisvl.index import SearchIndex  # type: ignore[import]
from redisvl.migration.utils import wait_for_index_ready  # type: ignore[import]
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
        extra_fields: Optional[List[Dict[str, Any]]] = None,
    ) -> RedisVectorStore:
        name = f"ownership_migration_{uuid4().hex[:8]}"
        fields: List[Dict[str, Any]] = [
            *(extra_fields or []),
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


@pytest.mark.parametrize("storage_type", ["hash", "json"])
@pytest.mark.parametrize("legacy_responses", [True, False])
def test_migration_supports_resp2(
    redis_url: str,
    make_legacy_store: Callable[..., RedisVectorStore],
    storage_type: str,
    legacy_responses: bool,
) -> None:
    store = make_legacy_store(storage_type, ids=("one",))
    # Exercise RESP2 normalization while checking another index's disjoint prefix.
    make_legacy_store(ids=("other",))
    key = store.index.key("one")

    with Redis.from_url(
        redis_url, protocol=2, legacy_responses=legacy_responses
    ) as client:
        preview = migrate_index_ownership(
            store.index.name, source_config=store.config, redis_client=client
        )
        result = migrate_index_ownership(
            store.index.name,
            source_config=store.config,
            redis_client=client,
            dry_run=False,
        )

        assert (preview.schema_action, preview.would_update) == ("alter", 1)
        assert (result.updated, result.completed) == (1, True)
        if storage_type == "json":
            assert client.json().get(key, "$._index_name") == [
                hashify(store.index.name)
            ]
        else:
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


@pytest.mark.parametrize("source_kind", ["schema", "config"])
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
    source_kind: str,
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
    source: Dict[str, Any] = (
        {"source_schema": store.config.to_index_schema()}
        if source_kind == "schema"
        else {"source_config": store.config}
    )

    preview = migrate_index_ownership(
        store.index.name, redis_client=client, **source, aliases=[]
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
        **source,
        aliases=[],
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
        store.index.name, redis_client=client, **source, dry_run=False
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


@pytest.mark.parametrize("marker_type", [None, "text", "compatible-tag"])
@pytest.mark.parametrize("invalid_source", ["missing", "both", "prefix", "fields"])
@pytest.mark.parametrize("dry_run", [True, False], ids=["preview", "apply"])
def test_invalid_source_is_rejected_before_mutation(
    client: Redis,
    make_legacy_store: Callable[..., RedisVectorStore],
    marker_type: Optional[str],
    invalid_source: str,
    dry_run: bool,
) -> None:
    store = make_legacy_store(
        marker_type=None if marker_type == "compatible-tag" else marker_type,
        ids=("one",),
    )
    source = store.config.to_index_schema().model_copy(deep=True)
    if marker_type == "compatible-tag":
        source.add_field(
            {"name": "_index_name", "type": "tag", "attrs": {"case_sensitive": True}}
        )
        client.ft(store.index.name).alter_schema_add(
            source.fields["_index_name"].as_redis_field()
        )
    key = store.index.key("one")
    before = client.hgetall(key)
    attributes = convert_bytes(client.ft(store.index.name).info())["attributes"]
    supplied: Dict[str, Any] = {"source_schema": source}
    if invalid_source == "missing":
        supplied = {}
    elif invalid_source == "both":
        supplied["source_config"] = store.config
    elif invalid_source == "prefix":
        source.index.prefix = "wrong:"
    else:
        source.fields.pop("doc_id")

    with pytest.raises(ValueError):
        migrate_index_ownership(
            store.index.name,
            redis_client=client,
            **supplied,
            aliases=[],
            dry_run=dry_run,
        )

    assert convert_bytes(client.ft(store.index.name).info())["attributes"] == attributes
    assert client.hgetall(key) == before


@pytest.mark.parametrize("dry_run", [True, False], ids=["preview", "apply"])
def test_aliases_rejects_a_bare_string_before_mutation(
    client: Redis,
    make_legacy_store: Callable[..., RedisVectorStore],
    dry_run: bool,
) -> None:
    store = make_legacy_store(ids=("one",))
    name, key = store.index.name, store.index.key("one")
    migrate_index_ownership(
        name, source_config=store.config, redis_client=client, dry_run=False
    )
    attributes = convert_bytes(client.ft(name).info())["attributes"]
    before = client.hgetall(key)

    with pytest.raises(ValueError, match="iterable of names"):
        migrate_index_ownership(
            name,
            source_config=store.config,
            redis_client=client,
            aliases=f"{name}_current",  # type: ignore[arg-type]
            dry_run=dry_run,
        )

    assert convert_bytes(client.ft(name).info())["attributes"] == attributes
    assert client.hgetall(key) == before


@pytest.mark.parametrize("legacy_responses", [True, False])
@pytest.mark.parametrize(
    "root",
    [
        pytest.param(["precious", "list", 1], id="array"),
        pytest.param("precious string", id="string"),
        pytest.param(42, id="number"),
        pytest.param(True, id="boolean"),
        pytest.param(None, id="null"),
    ],
)
@pytest.mark.parametrize("dry_run", [True, False], ids=["preview", "apply"])
def test_json_nonobject_roots_are_rejected_before_mutation(
    client: Redis,
    redis_url: str,
    make_legacy_store: Callable[..., RedisVectorStore],
    root: Any,
    dry_run: bool,
    legacy_responses: bool,
) -> None:
    store = make_legacy_store("json", marker_type="text")
    name = store.index.name
    keys = [store.index.key(doc_id) for doc_id in ("one", "two")]
    client.json().set(keys[1], "$", root)
    before = [client.json().get(key) for key in keys]
    attributes = convert_bytes(client.ft(name).info())["attributes"]
    assert {doc.id for doc in client.ft(name).search(Query("*").no_content()).docs} == (
        set(keys)
    )

    with (
        Redis.from_url(
            redis_url, protocol=2, legacy_responses=legacy_responses
        ) as migration_client,
        pytest.raises(ValueError) as exc,
    ):
        migrate_index_ownership(
            name,
            redis_client=migration_client,
            source_schema=store.config.to_index_schema(),
            aliases=[],
            dry_run=dry_run,
            batch_size=1,
        )

    assert keys[1] in str(exc.value)
    assert convert_bytes(client.ft(name).info())["attributes"] == attributes
    assert client.exists(*keys) == len(keys)
    assert [client.json().get(key) for key in keys] == before


@pytest.mark.parametrize("unseen_reason", ["invalid-marker", "skipped-initial-scan"])
def test_rebuild_does_not_claim_documents_missing_from_preflight(
    client: Redis,
    make_legacy_store: Callable[..., RedisVectorStore],
    unseen_reason: str,
) -> None:
    store = make_legacy_store("json", marker_type="text", ids=("one",))
    name = store.index.name
    checked, unseen = [store.index.key(doc_id) for doc_id in ("one", "unseen")]
    source = store.config.to_index_schema()
    document = cast(Dict[str, Any], client.json().get(checked))
    # TEXT rejects a boolean marker; TAG accepts it after the rebuild.
    document["_index_name"] = (
        True if unseen_reason == "invalid-marker" else "other_index"
    )
    client.json().set(unseen, "$", document)
    if unseen_reason == "skipped-initial-scan":
        client.ft(name).dropindex(delete_documents=False)
        client.ft(name).create_index(
            source.redis_fields,
            definition=IndexDefinition(prefix=[name], index_type=IndexType.JSON),
            skip_initial_scan=True,
        )
        # Only this post-creation write is picked up by the old index.
        client.json().set(checked, "$._index_name", name)
    query = Query("*").no_content()
    assert [doc.id for doc in client.ft(name).search(query).docs] == [checked]

    result = migrate_index_ownership(
        name,
        redis_client=client,
        source_schema=source,
        aliases=[],
        dry_run=False,
    )

    assert (result.would_update, result.updated, result.completed) == (1, 1, True)
    assert client.json().get(checked, "$._index_name") == [hashify(name)]
    assert client.json().get(unseen) == document
    wait_for_index_ready(store.index, timeout_seconds=5, poll_interval_seconds=0.05)
    assert {doc.id for doc in client.ft(name).search(query).docs} == {checked, unseen}


@pytest.mark.parametrize("storage_type", ["hash", "json"])
def test_expiry_after_preflight_reports_incomplete_and_allows_retry(
    client: Redis,
    make_legacy_store: Callable[..., RedisVectorStore],
    monkeypatch: pytest.MonkeyPatch,
    storage_type: str,
) -> None:
    store = make_legacy_store(storage_type)
    expired, remaining = [store.index.key(doc_id) for doc_id in ("one", "two")]
    apply_update_batch = SearchIndex._apply_update_batch

    def expire_before_update(index: SearchIndex, *args: Any, **kwargs: Any) -> Any:
        # A past timestamp expires the key immediately, without a real wait.
        client.pexpireat(expired, 1)
        return apply_update_batch(index, *args, **kwargs)

    monkeypatch.setattr(SearchIndex, "_apply_update_batch", expire_before_update)
    result = migrate_index_ownership(
        store.index.name, source_config=store.config, redis_client=client, dry_run=False
    )
    assert (result.would_update, result.updated, result.completed) == (2, 1, False)
    assert client.exists(expired) == 0
    assert _read_record(client, store, remaining)["_index_name"] == hashify(
        store.index.name
    )

    retry = migrate_index_ownership(
        store.index.name, source_config=store.config, redis_client=client, dry_run=False
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
        store.index.name, redis_client=client, source_schema=source, aliases=[]
    )
    assert preview.would_update == 1
    result = migrate_index_ownership(
        store.index.name,
        redis_client=client,
        source_schema=source,
        aliases=[],
        dry_run=False,
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
            source_config=store.config,
            redis_client=client,
            dry_run=False,
        )

    assert convert_bytes(client.ft(store.index.name).info())["attributes"] == attributes
    assert [client.hgetall(key) for key in keys] == before


@pytest.mark.parametrize(
    ("alias_template", "error"),
    [
        pytest.param(None, "require aliases explicitly", id="omitted"),
        pytest.param("{name}", "cannot be supplied as its own alias", id="index-name"),
        pytest.param("{name}-current", "does not point", id="missing"),
        pytest.param("", "must be non-empty", id="empty"),
        pytest.param(b"{name}", None, id="bytes-index-name"),
    ],
)
@pytest.mark.parametrize("dry_run", [True, False], ids=["preview", "apply"])
def test_rebuild_rejects_invalid_alias_before_mutation(
    client: Redis,
    make_legacy_store: Callable[..., RedisVectorStore],
    dry_run: bool,
    alias_template: Optional[str | bytes],
    error: Optional[str],
) -> None:
    store = make_legacy_store(storage_type="json", marker_type="text", ids=("one",))
    name, key = store.index.name, store.index.key("one")
    alias = f"{name}_current"
    client.ft(name).aliasadd(alias)
    attributes = convert_bytes(client.ft(name).info())["attributes"]
    before = client.json().get(key)
    aliases: Optional[List[Any]] = None
    if alias_template is not None:
        aliases = [
            alias_template.replace(b"{name}", name.encode())
            if isinstance(alias_template, bytes)
            else alias_template.format(name=name)
        ]

    with pytest.raises(ValueError, match=error):
        migrate_index_ownership(
            name,
            redis_client=client,
            source_schema=store.config.to_index_schema(),
            aliases=aliases,
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
    rebuilt_index = SearchIndex(schema=target, redis_client=client)
    rebuilt_index.create(overwrite=True, drop=False)
    wait_for_index_ready(
        rebuilt_index,
        timeout_seconds=5,
        poll_interval_seconds=0.05,
    )

    result = migrate_index_ownership(
        name,
        source_config=store.config,
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


@pytest.mark.parametrize("storage_type", ["hash", "json"])
def test_rebuild_preserves_field_aliases_sharing_a_path(
    client: Redis,
    make_legacy_store: Callable[..., RedisVectorStore],
    storage_type: str,
) -> None:
    store = make_legacy_store(storage_type, marker_type="text", ids=("one",))
    name, key = store.index.name, store.index.key("one")
    path = "$.title" if storage_type == "json" else "title"
    client.execute_command(
        "FT.ALTER",
        name,
        "SCHEMA",
        "ADD",
        path,
        "AS",
        "display",
        "TEXT",
        "WEIGHT",
        2,
        "NOSTEM",
        "SORTABLE",
        path,
        "AS",
        "display_exact",
        "TAG",
        "SEPARATOR",
        ",",
    )
    if storage_type == "json":
        client.json().set(key, path, "hello world")
    else:
        client.hset(key, "title", "hello world")
    source = store.config.to_index_schema().model_copy(deep=True)
    source.add_field(
        {
            "name": "display",
            "type": "text",
            "attrs": {"weight": 2, "no_stem": True, "sortable": True},
        }
    )
    source.add_field({"name": "display_exact", "type": "tag"})
    # Both aliases index the same value, with distinct TEXT and TAG options.
    for alias in ("display", "display_exact"):
        source.fields[alias].path = path
    before = _read_record(client, store, key)
    attributes = convert_bytes(client.ft(name).info())["attributes"]
    queries = [
        Query(expression).no_content().dialect(2)
        for expression in ("@display:hello", "@display_exact:{hello world}")
    ]
    for query in queries:
        assert [doc.id for doc in client.ft(name).search(query).docs] == [key]

    preview = migrate_index_ownership(
        name, redis_client=client, source_schema=source, aliases=[]
    )
    assert preview.schema_action == "rebuild"
    assert convert_bytes(client.ft(name).info())["attributes"] == attributes
    assert _read_record(client, store, key) == before

    result = migrate_index_ownership(
        name,
        redis_client=client,
        source_schema=source,
        aliases=[],
        dry_run=False,
    )
    assert (result.updated, result.completed) == (1, True)
    after_attributes = convert_bytes(client.ft(name).info())["attributes"]
    for attribute in attributes:
        identifier = (
            attribute["identifier"] if isinstance(attribute, dict) else attribute[1]
        )
        if identifier not in ("_index_name", "$._index_name"):
            assert attribute in after_attributes
    for query in queries:
        assert [doc.id for doc in client.ft(name).search(query).docs] == [key]
    assert _read_record(client, store, key) == {**before, "_index_name": hashify(name)}


@pytest.mark.parametrize("storage_type", ["hash", "json"])
@pytest.mark.parametrize(
    ("marker_type", "index_missing", "supplied_index_missing", "refused"),
    [
        ("text", False, True, True),
        ("text", True, False, True),
        ("text", True, True, True),
        (None, True, True, False),
        ("tag", True, True, False),
    ],
)
@pytest.mark.parametrize("dry_run", [True, False], ids=["preview", "apply"])
def test_vector_indexmissing_migration_paths(
    client: Redis,
    make_legacy_store: Callable[..., RedisVectorStore],
    storage_type: str,
    marker_type: Optional[str],
    index_missing: bool,
    supplied_index_missing: bool,
    refused: bool,
    dry_run: bool,
) -> None:
    store = make_legacy_store(storage_type, marker_type=marker_type, ids=("one",))
    source = store.config.to_index_schema().model_copy(deep=True)
    if marker_type == "tag":
        source.fields["_index_name"].attrs.case_sensitive = True
    if index_missing:
        # RedisVL currently serializes this vector flag as a parameter; use
        # redis-py to create the live index with the proper field-level flag.
        redis_fields = source.redis_fields
        redis_fields[list(source.fields).index("embedding")].append_arg("INDEXMISSING")
        client.ft(store.index.name).dropindex(delete_documents=False)
        client.ft(store.index.name).create_index(
            redis_fields,
            definition=IndexDefinition(
                prefix=[store.index.name], index_type=IndexType[storage_type.upper()]
            ),
        )
        wait_for_index_ready(store.index, timeout_seconds=5, poll_interval_seconds=0.05)
    source.fields["embedding"].attrs.index_missing = supplied_index_missing
    key = store.index.key("one")
    if marker_type == "tag":
        # Exercise a marker-only backfill rather than an already-migrated no-op.
        if storage_type == "json":
            client.json().set(key, "$._index_name", store.index.name)
        else:
            client.hset(key, "_index_name", store.index.name)
    before = _read_record(client, store, key)
    attributes = convert_bytes(client.ft(store.index.name).info())["attributes"]

    def migrate() -> Any:
        return migrate_index_ownership(
            store.index.name,
            redis_client=client,
            source_schema=source,
            aliases=[],
            dry_run=dry_run,
        )

    if refused:
        with pytest.raises(ValueError):
            migrate()
        assert (
            convert_bytes(client.ft(store.index.name).info())["attributes"]
            == attributes
        )
        assert _read_record(client, store, key) == before
    else:
        result = migrate()
        assert result.schema_action == ("alter" if marker_type is None else "none")
        assert (result.updated, result.completed) == (0 if dry_run else 1, True)
        assert _read_record(client, store, key) == (
            before if dry_run else {**before, "_index_name": hashify(store.index.name)}
        )


@pytest.mark.parametrize("storage_type", ["hash", "json"])
@pytest.mark.parametrize(
    ("field_type", "attrs", "values", "expression", "expected_ids"),
    [
        pytest.param(
            "text",
            {"phonetic_matcher": "dm:en"},
            ("Jon", "Bob"),
            "@value:John",
            ("one",),
            id="phonetic",
        ),
        *[
            pytest.param(
                "numeric",
                {"sortable": True, "unf": unf},
                (10, 2),
                "@value:[0 20]",
                ("two", "one"),
                id=f"numeric-unf-{unf}",
            )
            for unf in (False, True)
        ],
    ],
)
def test_rebuild_preserves_nonvector_field_options(
    client: Redis,
    make_legacy_store: Callable[..., RedisVectorStore],
    storage_type: str,
    field_type: str,
    attrs: Dict[str, Any],
    values: Sequence[Any],
    expression: str,
    expected_ids: Sequence[str],
) -> None:
    store = make_legacy_store(
        storage_type,
        marker_type="text",
        extra_fields=[{"name": "value", "type": field_type, "attrs": attrs}],
    )
    source = store.config.to_index_schema().model_copy(deep=True)
    keys = [store.index.key(doc_id) for doc_id in ("one", "two")]
    for key, value in zip(keys, values, strict=True):
        if storage_type == "json":
            client.json().set(key, "$.value", value)
        else:
            client.hset(key, "value", value)
    before = [_read_record(client, store, key) for key in keys]
    query = Query(expression).no_content().dialect(2)
    if field_type == "numeric":
        query.sort_by("value")
    expected_keys = [store.index.key(doc_id) for doc_id in expected_ids]
    assert [doc.id for doc in client.ft(store.index.name).search(query).docs] == (
        expected_keys
    )

    result = migrate_index_ownership(
        store.index.name,
        redis_client=client,
        source_schema=source,
        aliases=[],
        dry_run=False,
    )

    assert result.completed
    assert [doc.id for doc in client.ft(store.index.name).search(query).docs] == (
        expected_keys
    )
    assert [_read_record(client, store, key) for key in keys] == [
        {**record, "_index_name": hashify(store.index.name)} for record in before
    ]


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
            name, redis_client=client, source_config=config, aliases=[]
        )
        result = migrate_index_ownership(
            name,
            redis_client=client,
            source_config=config,
            aliases=[],
            dry_run=False,
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
def test_vector_rebuild_preserves_supplied_tuning(
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
    # FT.INFO/RedisVL cannot round-trip all tuning; inspect the replacement command.
    creates = []
    execute = client.execute_command

    def capture_create(*args: Any, **kwargs: Any) -> Any:
        if args[0] == "FT.CREATE":
            creates.append(args)
        return execute(*args, **kwargs)

    monkeypatch.setattr(client, "execute_command", capture_create)
    result = migrate_index_ownership(
        name,
        redis_client=client,
        source_schema=source,
        aliases=[],
        dry_run=False,
    )
    assert (result.updated, result.completed) == (1, True)
    assert len(creates) == 2
    for create in creates:
        for option, expected in tuning.items():
            assert create[create.index(option.upper()) + 1] == expected
    assert source.to_dict() == source_before
    assert _read_record(client, store, key) == {**before, "_index_name": hashify(name)}


@pytest.mark.parametrize("dry_run", [True, False], ids=["preview", "apply"])
def test_invalid_rebuild_schema_does_not_drop_live_index(
    client: Redis,
    make_legacy_store: Callable[..., RedisVectorStore],
    dry_run: bool,
) -> None:
    store = make_legacy_store(
        marker_type="text",
        ids=("one",),
        vector_attrs={"algorithm": "hnsw"},
    )
    name, key = store.index.name, store.index.key("one")
    alias = f"{name}_current"
    client.ft(name).aliasadd(alias)
    source = store.config.to_index_schema().model_copy(deep=True)
    source.fields["embedding"].attrs.m = 0
    attributes = convert_bytes(client.ft(name).info())["attributes"]
    before = _read_record(client, store, key)
    indexes = set(convert_bytes(client.execute_command("FT._LIST")))

    with pytest.raises(RedisSearchError, match="M.*outside acceptable bounds"):
        migrate_index_ownership(
            name,
            redis_client=client,
            source_schema=source,
            aliases=[alias],
            dry_run=dry_run,
        )

    assert set(convert_bytes(client.execute_command("FT._LIST"))) == indexes
    assert convert_bytes(client.ft(name).info())["attributes"] == attributes
    assert convert_bytes(client.ft(alias).info())["index_name"] == name
    assert _read_record(client, store, key) == before


@pytest.mark.parametrize("dry_run", [True, False], ids=["preview", "apply"])
def test_rebuild_probe_uses_live_prefix_with_key_scoped_acl(
    redis_url: str,
    client: Redis,
    make_legacy_store: Callable[..., RedisVectorStore],
    monkeypatch: pytest.MonkeyPatch,
    dry_run: bool,
) -> None:
    # Test probe permissions independently of access to unrelated index definitions.
    monkeypatch.setattr(
        "langchain_redis.migrations._assert_no_overlapping_prefixes", lambda index: None
    )
    store = make_legacy_store(marker_type="text", ids=("one",))
    source = store.config.to_index_schema()
    live_prefix = source.index.prefix
    if not isinstance(live_prefix, str):
        live_prefix = live_prefix[0]
    username = f"migration_{uuid4().hex[:8]}"
    password = uuid4().hex
    indexes = set(convert_bytes(client.execute_command("FT._LIST")))
    client.execute_command(
        "ACL",
        "SETUSER",
        username,
        "reset",
        "on",
        f">{password}",
        f"~{live_prefix}*",
        "+@all",
    )
    try:
        with Redis.from_url(
            redis_url,
            username=username,
            password=password,
            decode_responses=False,
        ) as restricted_client:
            result = migrate_index_ownership(
                store.index.name,
                redis_client=restricted_client,
                source_schema=source,
                aliases=[],
                dry_run=dry_run,
            )
    finally:
        client.execute_command("ACL", "DELUSER", username)

    assert (result.schema_action, result.completed) == ("rebuild", True)
    assert set(convert_bytes(client.execute_command("FT._LIST"))) == indexes


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
