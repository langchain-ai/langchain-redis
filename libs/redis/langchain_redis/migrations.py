"""Migrate legacy RedisVectorStore ownership fields and stored markers."""

from dataclasses import dataclass
from time import sleep
from typing import Any, Dict, Iterable, List, Literal, Optional, cast

import redis.exceptions
from redis import Redis
from redis.commands.search.aggregation import AggregateRequest, Cursor
from redisvl.index import SearchIndex  # type: ignore[import]
from redisvl.query.filter import Tag  # type: ignore[import]
from redisvl.redis.connection import convert_index_info_to_schema  # type: ignore[import]
from redisvl.redis.utils import convert_bytes, hashify  # type: ignore[import]
from redisvl.schema import FieldTypes, IndexSchema, StorageType  # type: ignore[import]

_FIELD = "_index_name"


@dataclass(frozen=True)
class IndexOwnershipMigrationResult:
    """Schema operation and marker counts for a preview or applied migration."""

    schema_action: Literal["none", "alter", "rebuild"]
    dry_run: bool
    would_update: int
    updated: int
    completed: bool


def migrate_index_ownership(
    index_name: str,
    *,
    redis_client: Redis,
    source_schema: Optional[IndexSchema] = None,
    dry_run: bool = True,
    aliases: Iterable[str] = (),
    batch_size: int = 500,
) -> IndexOwnershipMigrationResult:
    """Upgrade a legacy index and backfill its hashed ownership marker.

    Preview by default; pass ``dry_run=False`` to apply. The index must have a
    dedicated document namespace: all indexed documents must belong to it.
    Missing, null and raw index-name markers are backfilled; already-hashed
    markers are left alone. Conflicting markers are refused before mutation.

    Pause all application/background writers and back up the schema before applying.
    Missing fields use FT.ALTER; legacy TEXT or incompatible TAG fields require
    recreating the index definition, causing query downtime during reindexing.
    Document keys, embeddings and TTLs are retained. Supply every existing alias
    in ``aliases`` so it can be restored after a rebuild. Custom FT.CREATE options
    and nonstandard ownership paths require manual migration.

    Every rebuild requires the complete, current deployment ``source_schema``.
    It is authoritative: only its ownership field is replaced. The caller must
    include all fields, aliases and vector tuning; schemas inferred from FT.INFO
    or generated from an incomplete config are not sufficient. The index name,
    storage type and prefixes are checked against the live index before mutation.
    Rebuilding an index with INDEXALL enabled requires manual migration.

    The operation is not transactional and has no automatic rollback. Re-run
    marker backfills after interruption. ``completed`` reports marker writes,
    not whether Redis has finished reindexing after the writes. Schema changes
    check indexing readiness up to 60 times, once per second, before the bulk update.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    # Inspect the live definition before planning either schema or data changes.
    client = cast(Any, redis_client)
    info = convert_bytes(client.ft(index_name).info())
    if info["index_name"] != index_name:
        raise ValueError("Use the concrete index name, not an alias.")
    if float(info.get("percent_indexed", 1)) < 1:
        raise ValueError("Wait for the source index to finish indexing first.")
    schema = _ownership_schema(info)
    index = SearchIndex(schema=schema, redis_client=client)
    action = _schema_action(schema, index_name)
    target = _target_schema(schema)
    if action == "rebuild":
        # IndexSchema cannot represent these FT.CREATE options.
        index_definition = info.get("index_definition", {})
        if str(index_definition.get("indexes_all", "false")).lower() == "true":
            raise ValueError("Indexes with INDEXALL enabled require manual migration.")
        supported = {"key_type", "prefixes", "default_score", "indexes_all"}
        if (
            info.get("index_options")
            or set(index_definition) - supported
            or float(index_definition.get("default_score", 1)) != 1
        ):
            raise ValueError("Custom index options require manual schema migration.")
        if source_schema is None:
            raise ValueError("Rebuilds require the complete original source_schema.")
        prefixes = source_schema.index.prefix
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        if (
            source_schema.index.name != index_name
            or source_schema.index.storage_type != schema.index.storage_type
            or prefixes != index_definition["prefixes"]
        ):
            raise ValueError(
                "source_schema must match the live index name, "
                "storage type and prefixes."
            )
        # Use the operator's declaration directly; FT.INFO is not a schema export.
        target = _target_schema(source_schema)

    # Preflight every marker before any writes. With application writers paused,
    # this establishes that the bulk backfill will not claim another index's data.
    keys = _index_keys(index, batch_size)
    storage = schema.index.storage_type.value
    marker = hashify(index_name)
    would_update = 0
    for offset in range(0, len(keys), batch_size):
        batch = keys[offset : offset + batch_size]
        pipe = client.pipeline(transaction=False)
        for key in batch:
            if storage == "json":
                pipe.json().get(key, "$._index_name")
            else:
                pipe.hget(key, _FIELD)
        for value in convert_bytes(pipe.execute()):
            if storage == "json":
                # JSONPath reads wrap values in a list; missing/null both become None.
                value = value[0] if value else None
            if value is None or value == index_name:
                would_update += 1
            elif value != marker:
                raise ValueError("A document carries another index's marker.")

    # Validate supplied index aliases before a rebuild can remove their bindings.
    aliases = tuple(aliases)
    for alias in aliases:
        if alias == index_name:
            raise ValueError("An index cannot be supplied as its own alias.")
        if convert_bytes(client.ft(alias).info())["index_name"] != index_name:
            raise ValueError(f"Alias {alias!r} does not point to {index_name!r}.")

    updated = 0
    if not dry_run:
        if action == "rebuild":
            # Redis cannot change an existing field's type/options in place.
            # drop=False retains document keys and TTLs while replacing the index.
            SearchIndex(schema=target, redis_client=client).create(
                overwrite=True, drop=False
            )
            for alias in aliases:
                client.ft(index_name).aliasupdate(alias)
        elif action == "alter":
            client.ft(index_name).alter_schema_add(
                target.fields[_FIELD].as_redis_field()
            )
        if would_update:
            # Bulk updates select through the index, so a rebuild/ALTER must
            # finish indexing the retained documents before selection starts.
            for _ in range(60):
                status = index.info()
                if (
                    not int(status["indexing"])
                    and float(status["percent_indexed"]) >= 1
                ):
                    break
                sleep(1)
            else:
                raise TimeoutError("Index is still rebuilding; retry migration later.")
            # After preflight, non-hashed markers are legacy or missing/null.
            # This negative filter includes missing fields without INDEXMISSING.
            # RedisVL handles HASH/JSON partial writes and skips expired documents.
            result = index.update_by_filter(
                Tag(_FIELD) != marker,
                {_FIELD: marker},
                batch_size=batch_size,
            )
            updated = result.processed

    # Fewer writes than planned (for example, from TTL expiry) means incomplete.
    return IndexOwnershipMigrationResult(
        schema_action=action,
        dry_run=dry_run,
        would_update=would_update,
        updated=updated,
        completed=dry_run or updated == would_update,
    )


def _schema_action(
    schema: IndexSchema, index_name: str
) -> Literal["none", "alter", "rebuild"]:
    """Choose between adding the field, retaining it, or rebuilding its definition."""
    field = schema.fields.get(_FIELD)
    if field is None:
        return "alter"
    expected_path = (
        f"$.{_FIELD}" if schema.index.storage_type == StorageType.JSON else None
    )
    if field.path != expected_path or field.type not in (
        FieldTypes.TEXT,
        FieldTypes.TAG,
    ):
        raise ValueError(
            "Nonstandard ownership fields require manual schema migration."
        )
    attrs = field.attrs
    # Exact ownership matching needs an indexed, case-sensitive, unsplit hash.
    if (
        field.type == FieldTypes.TAG
        and getattr(attrs, "case_sensitive", False)
        and not getattr(attrs, "no_index", False)
        and getattr(attrs, "separator", ",") not in hashify(index_name)
    ):
        return "none"
    return "rebuild"


def _ownership_schema(info: Dict[str, Any]) -> IndexSchema:
    """Inspect only the live ownership field, without reconstructing vector settings."""
    path = "$._index_name" if info["index_definition"]["key_type"] == "JSON" else _FIELD
    attributes = []
    for attribute in info["attributes"]:
        identifier, alias = (
            (attribute["identifier"], attribute["attribute"])
            if isinstance(attribute, dict)
            else (attribute[1], attribute[3])
        )
        if identifier == path or alias == _FIELD:
            if identifier != path or alias != _FIELD:
                raise ValueError(
                    "Nonstandard ownership fields require manual migration."
                )
            attributes.append(attribute)
    return IndexSchema.from_dict(
        convert_index_info_to_schema({**info, "attributes": attributes})
    )


def _target_schema(source: IndexSchema) -> IndexSchema:
    """Replace the ownership definition while retaining unrelated fields and options."""
    target = source.model_copy(deep=True)
    target.fields.pop(_FIELD, None)
    target.add_field({"name": _FIELD, "type": "tag", "attrs": {"case_sensitive": True}})
    return target


def _index_keys(index: SearchIndex, batch_size: int) -> List[str]:
    """Collect indexed keys with a cursor to avoid search-result pagination limits."""
    client = cast(Any, index.client)
    request = AggregateRequest("*").load("__key").cursor(count=batch_size).dialect(2)
    cursor_id = 0
    keys = set()  # A cursor may return a document more than once.
    try:
        result = client.ft(index.name).aggregate(request)
        while True:
            cursor_id = result.cursor.cid if result.cursor else 0
            for row in convert_bytes(result.rows):
                key = (
                    row["__key"]
                    if isinstance(row, dict)
                    else row[row.index("__key") + 1]
                )
                keys.add(key)
            if not cursor_id:
                break
            result = client.ft(index.name).aggregate(Cursor(cursor_id))
    finally:
        if cursor_id:
            # Release the server cursor if iteration ended before it was exhausted.
            try:
                client.execute_command("FT.CURSOR", "DEL", index.name, cursor_id)
            except redis.exceptions.RedisError:
                pass
    return sorted(keys)
