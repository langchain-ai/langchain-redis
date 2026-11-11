"""Migrate legacy RedisVectorStore ownership fields and stored markers."""

from dataclasses import dataclass
from time import sleep
from typing import Any, Dict, Iterable, List, Literal, Optional, cast
from uuid import uuid4

import redis.exceptions
from redis import Redis
from redis.commands.search.aggregation import AggregateRequest, Cursor
from redisvl.exceptions import (  # type: ignore[import]
    RedisSearchError,
    _is_missing_index_error,
)
from redisvl.index import SearchIndex  # type: ignore[import]
from redisvl.query import CountQuery  # type: ignore[import]
from redisvl.query.filter import Tag  # type: ignore[import]
from redisvl.redis.connection import (  # type: ignore[import]
    convert_index_info_to_schema,
    normalize_index_definition,
    normalize_index_fields,
)
from redisvl.redis.utils import convert_bytes, hashify  # type: ignore[import]
from redisvl.schema import FieldTypes, IndexSchema, StorageType  # type: ignore[import]

from langchain_redis.config import RedisConfig

_FIELD = "_index_name"
_SCHEMA_MISMATCH = "Migration source does not match the live index definition."


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
    source_config: Optional[RedisConfig] = None,
    dry_run: bool = True,
    aliases: Optional[Iterable[str]] = None,
    batch_size: int = 500,
) -> IndexOwnershipMigrationResult:
    """Upgrade a legacy index and backfill its hashed ownership marker.

    Preview by default; pass ``dry_run=False`` to apply. The index must have a
    dedicated document namespace: all indexed documents must belong to it.
    Missing, null and raw index-name markers are backfilled; already-hashed
    markers are left alone. Conflicting markers are refused before mutation.

    Stop all application/background writers and back up the schema before applying.
    After applying, restart the writers or recreate every RedisVectorStore instance
    so no process continues writing markers using its cached legacy schema.
    Missing fields use FT.ALTER; legacy TEXT or incompatible TAG fields require
    recreating the index definition, causing query downtime during reindexing.
    Document keys, embeddings and TTLs are retained. Supply every desired alias
    in ``aliases`` so it can be restored after a rebuild or interrupted run.
    Rebuilds require this argument explicitly; use an empty iterable when the
    index has no aliases.
    Custom FT.CREATE options and nonstandard ownership paths require manual
    migration. Indexes with overlapping prefixes of the same storage type cannot
    be migrated automatically; recreate them with disjoint prefixes instead.
    For both previews and applied migrations, credentials must permit FT._LIST
    and FT.INFO on every index, including those outside the target key prefix,
    so overlap checks can inspect all index definitions.

    Every rebuild requires either the complete deployment ``source_schema`` or the
    original ``source_config`` used for a generated schema. It is authoritative:
    only its ownership field is replaced. The index name, storage type, prefixes,
    fields, explicit stopwords and reported core vector settings are checked against
    the live index before mutation. Vector tuning is not inferred or compared with
    FT.INFO; it remains authoritative from the supplied source. Rebuilding an index
    with INDEXALL enabled requires manual migration.

    The operation is not transactional and has no automatic rollback. Re-run
    marker backfills after interruption. ``completed`` reports marker writes,
    not whether Redis has finished reindexing after the writes. Schema changes
    check indexing readiness up to 60 times, once per second, before the bulk update.
    Rebuild previews create and remove an empty temporary index to confirm that
    Redis accepts the supplied schema before the live index can be replaced.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    # Inspect the live definition before planning either schema or data changes.
    client = cast(Any, redis_client)
    info = convert_bytes(client.ft(index_name).info())
    info["index_definition"] = normalize_index_definition(info)
    if info["index_name"] != index_name:
        raise ValueError("Use the concrete index name, not an alias.")
    if float(info.get("percent_indexed", 1)) < 1:
        raise ValueError("Wait for the source index to finish indexing first.")
    schema = _ownership_schema(info)
    index = SearchIndex(schema=schema, redis_client=client)
    _assert_no_overlapping_prefixes(index)
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
        source = _resolve_source_schema(action, source_schema, source_config)
        assert source is not None
        prefixes = source.index.prefix
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        if (
            source.index.name != index_name
            or source.index.storage_type != schema.index.storage_type
            or prefixes != index_definition["prefixes"]
        ):
            raise ValueError(
                "The migration source must match the live index name, "
                "storage type and prefixes."
            )
        _validate_source_schema(info, source)
        # Use the operator's declaration directly; FT.INFO is not a schema export.
        target = _target_schema(source)

    # Preflight every marker before any writes. With application writers stopped,
    # this establishes that the bulk backfill will not claim another index's data.
    expected = cast(int, index.query(CountQuery("*")))
    keys = _index_keys(index, batch_size)
    if len(keys) != expected:
        raise RuntimeError(
            "Preflight returned a different number of documents than expected; "
            "retry migration."
        )
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

    # Refuse aliases owned by another index, but allow interrupted runs to
    # recreate aliases that disappeared during a rebuild.
    if action == "rebuild" and aliases is None:
        raise ValueError(
            "Rebuilds require aliases explicitly; use aliases=[] when there are none."
        )
    if isinstance(aliases, (str, bytes)):
        raise ValueError("aliases must be an iterable of names, not a string.")
    aliases = tuple(aliases or ())
    for alias in aliases:
        if not alias:
            raise ValueError("Alias names must be non-empty.")
        if alias == index_name:
            raise ValueError("An index cannot be supplied as its own alias.")
        try:
            alias_index = index.info(alias)["index_name"]
        except RedisSearchError as exc:
            if not _is_missing_index_error(exc):
                raise
            if action != "none":
                raise ValueError(
                    f"Alias {alias!r} does not point to {index_name!r}."
                ) from exc
        else:
            if alias_index != index_name:
                raise ValueError(f"Alias {alias!r} points to another index.")

    if action == "rebuild":
        _validate_rebuild_target(target, client)

    updated = 0
    if not dry_run:
        if action == "rebuild":
            # Redis cannot change an existing field's type/options in place.
            # drop=False retains document keys and TTLs while replacing the index.
            SearchIndex(schema=target, redis_client=client).create(
                overwrite=True, drop=False
            )
        elif action == "alter":
            client.ft(index_name).alter_schema_add(
                target.fields[_FIELD].as_redis_field()
            )
        for alias in aliases:
            client.ft(index_name).aliasupdate(alias)
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


def _resolve_source_schema(
    action: Literal["none", "alter", "rebuild"],
    source_schema: Optional[IndexSchema],
    source_config: Optional[RedisConfig],
) -> Optional[IndexSchema]:
    """Resolve rebuild input to one authoritative schema."""
    if source_schema is not None and source_config is not None:
        raise ValueError("Provide source_schema or source_config, not both.")
    if action != "rebuild":
        return None
    if source_schema is not None:
        return source_schema
    if source_config is not None:
        if (
            source_config.index_schema is None
            and source_config.schema_path is None
            and source_config.embedding_dimensions is None
        ):
            raise ValueError(
                "source_config.embedding_dimensions is required for generated schemas."
            )
        return source_config.to_index_schema()
    raise ValueError("Rebuilds require the original source_schema or source_config.")


def _assert_no_overlapping_prefixes(index: SearchIndex) -> None:
    """Refuse namespaces that another index of the same storage type can match."""
    target = index.schema.index
    target_prefixes = (
        target.prefix if isinstance(target.prefix, list) else [target.prefix]
    )
    client = cast(Any, index.client)

    for other_name in index.listall():
        if other_name == index.name:
            continue
        other_info = convert_bytes(client.ft(other_name).info())
        other = normalize_index_definition(other_info)
        if other["key_type"].lower() != target.storage_type.value:
            continue
        for target_prefix in target_prefixes:
            for other_prefix in other["prefixes"]:
                if target_prefix.startswith(other_prefix) or other_prefix.startswith(
                    target_prefix
                ):
                    raise ValueError(
                        f"Index prefix {target_prefix!r} overlaps index "
                        f"{other_name!r} prefix {other_prefix!r}; automatic "
                        "migration requires disjoint prefixes."
                    )


def _attribute_identity(attribute: Any) -> tuple[str, str, str]:
    """Read identifier, alias and type directly from a raw FT.INFO attribute."""
    metadata = _attribute_metadata(attribute)
    return (
        str(metadata["identifier"]),
        str(metadata["attribute"]),
        str(metadata["type"]).upper(),
    )


def _schema_field_identity(field: Any) -> tuple[str, str, str]:
    """Return a supplied field's identifier, alias and type."""
    field_type = getattr(field.type, "value", field.type)
    return field.path or field.name, field.name, str(field_type).upper()


def _attribute_metadata(attribute: Any) -> Dict[str, Any]:
    """Normalize RESP2/RESP3 FT.INFO field metadata and key casing."""
    normalized = normalize_index_fields({"attributes": [attribute]})[0]
    return {str(key).lower(): value for key, value in normalized.items()}


def _vector_core_settings(attribute: Any) -> Dict[str, Any]:
    """Return only the core vector settings that FT.INFO explicitly reports."""
    metadata = _attribute_metadata(attribute)
    aliases = {
        "algorithm": "algorithm",
        "data_type": "datatype",
        "datatype": "datatype",
        "dim": "dims",
        "dims": "dims",
        "distance_metric": "distance_metric",
    }
    settings = {}
    for live_name, source_name in aliases.items():
        if live_name not in metadata:
            continue
        value = metadata[live_name]
        settings[source_name] = (
            int(value)
            if source_name == "dims"
            else str(getattr(value, "value", value)).upper()
        )
    return settings


def _schema_mismatch(component: str, live: Any, supplied: Any) -> ValueError:
    """Describe the live and supplied values that prevent a safe rebuild."""
    return ValueError(
        f"{_SCHEMA_MISMATCH} {component} differs: live={live!r}, supplied={supplied!r}."
    )


def _validate_source_schema(info: Dict[str, Any], source: IndexSchema) -> None:
    """Compare reliable structural metadata and reported core vector settings."""
    live_stopwords = info.get("stopwords_list")
    source_stopwords = source.index.stopwords
    if ("stopwords_list" in info) != (source_stopwords is not None) or (
        source_stopwords is not None
        and sorted(live_stopwords or []) != sorted(source_stopwords)
    ):
        raise _schema_mismatch("Stopwords", live_stopwords, source_stopwords)

    live = []
    for attribute in info["attributes"]:
        identity = _attribute_identity(attribute)
        if identity[1] != _FIELD:
            live.append((identity, attribute))
    supplied = {
        field.name: (_schema_field_identity(field), field)
        for field in source.fields.values()
        if field.name != _FIELD
    }
    live_identities = sorted(identity for identity, _ in live)
    supplied_identities = sorted(identity for identity, _ in supplied.values())
    if live_identities != supplied_identities:
        raise _schema_mismatch("Fields", live_identities, supplied_identities)

    nonvector = [attribute for identity, attribute in live if identity[2] != "VECTOR"]
    definition = convert_index_info_to_schema({**info, "attributes": nonvector})
    # Preserve distinct aliases for fields sharing the same stored value.
    for field, attribute in zip(definition["fields"], nonvector, strict=True):
        field["name"] = _attribute_identity(attribute)[1]
    live_fields = IndexSchema.from_dict(definition).fields
    for (_, alias, field_type), attribute in live:
        if field_type == "VECTOR":
            live_core = _vector_core_settings(attribute)
            source_attrs = supplied[alias][1].attrs
            expected_core = {
                name: (
                    int(value)
                    if name == "dims"
                    else str(getattr(value, "value", value)).upper()
                )
                for name in live_core
                if (value := getattr(source_attrs, name, None)) is not None
            }
            if live_core != expected_core:
                raise _schema_mismatch(
                    f"Vector field {alias!r} core settings",
                    live_core,
                    expected_core,
                )
            continue
        live_attrs = live_fields[alias].attrs.model_dump()
        expected_attrs = supplied[alias][1].attrs.model_dump()
        for options in (live_attrs, expected_attrs):
            if field_type == "TEXT":
                # FT.INFO omits the phonetic matcher; retain the supplied setting.
                options.pop("phonetic_matcher", None)
                if source.index.storage_type == StorageType.JSON:
                    # Redis adds UNF automatically to SORTABLE JSON text fields.
                    options.pop("unf", None)
            elif field_type == "NUMERIC" and options.get("sortable"):
                # SORTABLE numeric fields always have UNF, even if not requested.
                options["unf"] = True
        if live_attrs != expected_attrs:
            raise _schema_mismatch(
                f"Field {alias!r} options", live_attrs, expected_attrs
            )


def _ownership_schema(info: Dict[str, Any]) -> IndexSchema:
    """Inspect only the live ownership field, without reconstructing vector settings."""
    path = "$._index_name" if info["index_definition"]["key_type"] == "JSON" else _FIELD
    attributes = []
    for attribute in info["attributes"]:
        identifier, alias, _ = _attribute_identity(attribute)
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


def _validate_rebuild_target(target: IndexSchema, client: Any) -> None:
    """Confirm Redis accepts a replacement schema before dropping the live index."""
    probe_schema = target.model_copy(deep=True)
    probe_name = f"__ownership_migration_probe_{uuid4().hex}"
    probe_schema.index.name = probe_name
    live_prefix = target.index.prefix
    if not isinstance(live_prefix, str):
        live_prefix = live_prefix[0]
    probe_schema.index.prefix = f"{live_prefix}{probe_name}:"
    probe = SearchIndex(schema=probe_schema, redis_client=client)
    try:
        probe.create()
    finally:
        if probe.exists():
            probe.delete(drop=False)


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
