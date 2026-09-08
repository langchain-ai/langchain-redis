"""Redis vector store."""

from __future__ import annotations

import ast
import json
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from redisvl.index import SearchIndex  # type: ignore[import]
from redisvl.query import (  # type: ignore[import]
    AggregateHybridQuery,
    HybridQuery,
    RangeQuery,
    TextQuery,
    VectorQuery,
)
from redisvl.query.filter import FilterExpression, Tag, Text  # type: ignore[import]
from redisvl.redis.utils import (  # type: ignore[import]
    array_to_buffer,
    buffer_to_array,
    convert_bytes,
    hashify,
)
from redisvl.schema import FieldTypes, IndexSchema, StorageType  # type: ignore[import]

from langchain_redis.config import RedisConfig
from langchain_redis.version import __lib_name__

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]
_INDEX_NAME_FIELD = "_index_name"


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    try:
        import simsimd as simd  # type: ignore

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = 1 - simd.cdist(X, Y, metric="cosine")
        if isinstance(Z, float):
            return np.array([Z])
        return np.array(Z)
    except ImportError:
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: List[np.ndarray],
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance.

    Maximal marginal relevance optimizes for similarity to the query AND diversity
    among selected documents.

    Args:
        query_embedding: Embedding of the query text.
        embedding_list: List of embeddings to select from.
        lambda_mult: Number between `0` and `1` that determines the degree
            of diversity among the results, where `0` corresponds to
            maximum diversity and `1` to minimum diversity.
        k: Number of results to return.

    Returns:
        List of indices of selected embeddings.

    Example:
        ```python
        from langchain_redis import RedisVectorStore
        from langchain_openai import OpenAIEmbeddings
        import numpy as np

        embeddings = OpenAIEmbeddings()
        vector_store = RedisVectorStore(
            embeddings=embeddings,
            index_name="langchain-demo",
            redis_url="redis://localhost:6379",
        )

        query = "What is the capital of France?"
        query_embedding = embeddings.embed_query(query)

        # Assuming you have a list of document embeddings
        doc_embeddings = [embeddings.embed_query(doc) for doc in documents]

        selected_indices = vector_store.maximal_marginal_relevance(
            query_embedding=np.array(query_embedding),
            embedding_list=[np.array(emb) for emb in doc_embeddings],
            lambda_mult=0.5,
            k=2
        )

        for idx in selected_indices:
            print(f"Selected document: {documents[idx]}")
        ```
    """
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


class RedisVectorStore(VectorStore):
    """Redis vector store integration.

    Setup:
        Install `langchain-redis` and running the Redis docker container.

        ```bash
        pip install -qU langchain-redis
        docker run -d --name redis -p 6379:6379 redis
        ```

    Key init args — indexing params:
        index_name: str
            Name of the index to create.
        embedding: Embeddings
            Embedding function to use.
        distance_metric: str
            Distance metric to use for similarity search. Default is `'COSINE'`.
        indexing_algorithm: str
            Indexing algorithm to use: `'FLAT'`, `'HNSW'` or `'SVS-VAMANA'`.
            Default is `'FLAT'`.
        vector_datatype: str
            Data type of the vector. Default is `'FLOAT32'`.
        vector_attrs: Optional[Dict[str, Any]]
            Algorithm-specific tuning attributes for the vector field, e.g.
            `{"ef_runtime": 20}` for HNSW or `{"compression": "LVQ8"}` for
            SVS-VAMANA. See `RedisConfig.vector_attrs`.

    Key init args — client params:
        redis_url: Optional[str]
            URL of the Redis instance to connect to.
        redis_client: Optional[Redis]
            Pre-existing Redis connection.
        ttl: Optional[int]
            Time-to-live for the Redis keys.

    Instantiate:
        ```python
        from langchain_redis import RedisVectorStore
        from langchain_openai import OpenAIEmbeddings

        vector_store = RedisVectorStore(
            embeddings=OpenAIEmbeddings(),
            index_name="langchain-demo",
            redis_url="redis://localhost:6379",
        )
        ```

    You can also connect to an existing Redis instance by passing in a
    pre-existing Redis connection via the `redis_client` argument.

    !!! example "Instantiate from existing connection"

        ```python
        from langchain_redis import RedisVectorStore
        from langchain_openai import OpenAIEmbeddings
        from redis import Redis

        redis_client = Redis.from_url("redis://localhost:6379")

        store = RedisVectorStore(
            embeddings=OpenAIEmbeddings(),
            index_name="langchain-demo",
            redis_client=redis_client
        )
        ```

    !!! example "Add Documents"

        ```python
        from langchain_core.documents import Document

        document_1 = Document(page_content="foo", metadata={"baz": "bar"})
        document_2 = Document(page_content="bar", metadata={"foo": "baz"})
        document_3 = Document(page_content="to be deleted")

        documents = [document_1, document_2, document_3]
        ids = ["1", "2", "3"]
        vector_store.add_documents(documents=documents, ids=ids)
        ```

    !!! example Delete Documents

        ```python
        vector_store.delete(ids=["3"])
        ```

    !!! example "Search"

        ```python
        results = vector_store.similarity_search(query="foo", k=1)
        for doc in results:
            print(f"* {doc.page_content} [{doc.metadata}]")
        ```

        ```python
        * foo [{'baz': 'bar'}]
        ```

    !!! example "Search with filter"

        ```python
        from redisvl.query.filter import Tag

        results = vector_store.similarity_search(
            query="foo",
            k=1,
            filter=Tag("baz") == "bar"
        )
        for doc in results:
            print(f"* {doc.page_content} [{doc.metadata}]")
        ```

        ```python
        * foo [{'baz': 'bar'}]
        ```

    !!! example "Search with score"

        ```python
        results = vector_store.similarity_search_with_score(query="foo", k=1)
        for doc, score in results:
            print(f"* [SIM={score:.3f}] {doc.page_content} [{doc.metadata}]")
        ```

        ```python
        * [SIM=0.916] foo [{'baz': 'bar'}]
        ```

    !!! example "Use as Retriever"

        ```python
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
        )
        retriever.get_relevant_documents("foo")
        ```

        ```python
        [Document(page_content='foo', metadata={'baz': 'bar'})]
        ```
    """

    def __init__(
        self,
        embeddings: Embeddings,
        config: Optional[RedisConfig] = None,
        ttl: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize the `RedisVectorStore`.

        Args:
            embeddings: The `Embeddings` instance used for this store.
            config: Optional `RedisConfig` object.

                If not provided, a new one will be created from kwargs.
            ttl: Optional time-to-live for Redis keys.
            **kwargs: Additional keyword arguments for `RedisConfig` if
                config is not provided.
        """
        # 1. Load or create the Redis configuration
        self.config = config or RedisConfig(**kwargs)

        # 2. Store embeddings and TTL
        self._embeddings = embeddings
        self.ttl = ttl

        # Lazily-probed FT.HYBRID support (None until first hybrid search)
        self._ft_hybrid_support: Optional[bool] = None

        # 3. Determine embedding dimensions if not explicitly set
        if self.config.embedding_dimensions is None:
            sample_text = "The quick brown fox jumps over the lazy dog"
            self.config.embedding_dimensions = len(
                self._embeddings.embed_query(sample_text)
            )

        # 4. Initialize the index based on config settings
        redis_client = self.config.redis()
        if self.config.index_schema:
            # Create index from the provided schema
            self._index = SearchIndex(
                schema=self.config.index_schema,
                redis_client=redis_client,
                lib_name=__lib_name__,
            )
            self._index.create(overwrite=False)
        elif self.config.schema_path:
            # Create index from a YAML schema file
            self._index = SearchIndex.from_yaml(
                self.config.schema_path,
                redis_client=redis_client,
                lib_name=__lib_name__,
            )
            self._index.create(overwrite=False)
        elif self.config.from_existing and self.config.index_name:
            # Create index from an existing index configuration
            self._index = SearchIndex.from_existing(
                name=self.config.index_name,
                redis_client=redis_client,
                lib_name=__lib_name__,
            )
            self._index.create(overwrite=False)
        else:
            # Build a default schema if no schema or path is provided
            modified_metadata_schema = []
            if self.config.metadata_schema is not None:
                for field in self.config.metadata_schema:
                    if field["type"] == "tag":
                        # Ensure a default separator is present
                        if "attrs" not in field or "separator" not in field.get(
                            "attrs", {}
                        ):
                            updated_field = field.copy()
                            updated_field.setdefault("attrs", {})["separator"] = (
                                self.config.default_tag_separator
                            )
                            modified_metadata_schema.append(updated_field)
                        else:
                            modified_metadata_schema.append(field)
                    else:
                        modified_metadata_schema.append(field)

            # Handle key prefix format
            # Issue #78: The trailing ":" was added historically, creating double
            # colons in keys (e.g., "prefix::doc_id"). For backward compatibility,
            # this behavior is maintained by default via legacy_key_format flag.
            # Set legacy_key_format=False for correct single-colon format.
            # Note: key_prefix is always set by validator, so it's never None here
            key_prefix = self.config.key_prefix or self.config.index_name
            prefix: Union[str, List[str]]
            if isinstance(key_prefix, list):
                # Multi-prefix index: searches span every prefix; writes use
                # the first one (config.primary_prefix)
                if self.config.legacy_key_format:
                    prefix = [f"{key_pref}:" for key_pref in key_prefix]
                else:
                    prefix = list(key_prefix)
            elif self.config.legacy_key_format:
                # Legacy format: adds trailing ":" (creates "prefix::doc_id")
                prefix = f"{key_prefix}:"
            else:
                # Correct format: no trailing ":" (creates "prefix:doc_id")
                prefix = key_prefix

            index_info: Dict[str, Any] = {
                "name": self.config.index_name,
                "prefix": prefix,
                "storage_type": self.config.storage_type,
            }
            if self.config.stopwords is not None:
                index_info["stopwords"] = self.config.stopwords

            self._index = SearchIndex.from_dict(
                {
                    "index": index_info,
                    "fields": [
                        {"name": self.config.content_field, "type": "text"},
                        {
                            "name": self.config.embedding_field,
                            "type": "vector",
                            "attrs": {
                                "dims": self.config.embedding_dimensions,
                                "distance_metric": self.config.distance_metric,
                                "algorithm": self.config.indexing_algorithm,
                                "datatype": self.config.vector_datatype,
                                **(self.config.vector_attrs or {}),
                            },
                        },
                        {
                            "name": _INDEX_NAME_FIELD,
                            "type": "tag",
                            "attrs": {"case_sensitive": True},
                        },
                        {"name": "_metadata_json", "type": "text"},
                        *modified_metadata_schema,
                    ],
                },
                redis_client=redis_client,
                lib_name=__lib_name__,
            )
            self._index.create(overwrite=False)

        # create(overwrite=False) leaves the configured schema unchanged when
        # the index already exists. Rehydrate it so reads and writes use the
        # schema that Redis actually retained.
        if not self.config.from_existing:
            self._index = SearchIndex.from_existing(
                name=self._index.name,
                redis_client=self._index.client,
                lib_name=__lib_name__,
            )
        self._sync_config_with_live_index()

    def _sync_config_with_live_index(self) -> None:
        """Synchronize schema-owned runtime settings with the live index."""
        schema = getattr(self._index, "schema", None)
        if not isinstance(schema, IndexSchema):
            # Some SearchIndex-compatible test doubles do not expose a complete
            # RedisVL schema. A real SearchIndex always does.
            return

        self.config.storage_type = schema.index.storage_type.value

        def logical_prefix(prefix: str) -> str:
            # Generated legacy schemas include the key separator in the index
            # prefix. Keep RedisConfig's logical prefix separator-free so IDs
            # that already include the legacy leading colon still round-trip.
            if self.config.legacy_key_format and prefix.endswith(":"):
                return prefix[:-1]
            return prefix

        live_prefix = schema.index.prefix
        if isinstance(live_prefix, list):
            self.config.key_prefix = [logical_prefix(prefix) for prefix in live_prefix]
        else:
            self.config.key_prefix = logical_prefix(live_prefix)

    @property
    def index(self) -> SearchIndex:
        return self._index

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    @property
    def key_prefix(self) -> Optional[Union[str, List[str]]]:
        return self.config.key_prefix

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add text documents to the vector store.

        Args:
            texts: Iterable of strings to add to the vector store.
            metadatas: Optional list of metadata dicts associated with the texts.
            keys: Optional list of keys to associate with the documents.
            **kwargs: Additional keyword arguments.

                Common kwargs include:

                - `ids`: Optional list of ids to associate with the documents.
                - `refresh_indices`: Whether to refresh the Redis indices
                    after adding the texts.
                    - Defaults to `True`.
                - `create_index_if_not_exists`: Whether to create the Redis
                index if it doesn't already exist.
                    - Defaults to `True`.
                - `batch_size`: Optional. Number of texts to add to the
                index at a time.
                    - Defaults to `1000`.

        Returns:
            List of ids from adding the texts into the vector store.

        Raises:
            ValueError: If `metadatas` is provided and its length does not
                match the number of `texts`.
            ValueError: If `keys` is provided and its length does not
                match the number of `texts`.
            ValueError: If `ids` is provided (via `kwargs`) and its length
                does not match the number of `texts`.

        Example:
            ```python
            from langchain_redis import RedisVectorStore
            from langchain_openai import OpenAIEmbeddings

            vector_store = RedisVectorStore(
                embeddings=OpenAIEmbeddings(),
                index_name="langchain-demo",
                redis_url="redis://localhost:6379",
            )

            texts = [
                "The quick brown fox jumps over the lazy dog",
                "Hello world",
                "Machine learning is fascinating"
            ]
            metadatas = [
                {"source": "book", "page": 1},
                {"source": "greeting", "language": "english"},
                {"source": "article", "topic": "AI"}
            ]

            ids = vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                batch_size=2
            )

            print(f"Added documents with ids: {ids}")
            ```

        Note:
            - If `metadatas` is provided, it must have the same length as `texts`.
            - If `keys` is provided, it must have the same length as `texts`.
            - The `batch_size` parameter can be used to control the number of
                documents added in each batch, which can be useful for managing
                memory usage when adding a large number of documents.
        """

        # Convert texts to a list if it's not already
        texts_list = list(texts)

        # If keys is not provided but ids exists in kwargs, use ids as keys
        if keys is None and "ids" in kwargs:
            keys = kwargs["ids"]

        # Validate lengths of metadatas and keys if provided
        if metadatas and len(metadatas) != len(texts_list):
            raise ValueError(
                "The length of 'metadatas' must match the number of 'texts'."
            )
        if keys and len(keys) != len(texts_list):
            raise ValueError("The length of 'keys' must match the number of 'texts'.")

        # If keys is None but ids is provided in kwargs, use ids as keys
        if keys is None and "ids" in kwargs and kwargs["ids"] is not None:
            ids = kwargs["ids"]
            if len(ids) != len(texts_list):
                raise ValueError(
                    "The length of 'ids' must match the number of 'texts'."
                )
            keys = ids

        # Generate embeddings for all texts
        document_embeddings = self._embeddings.embed_documents(texts_list)

        # Check if schema has _index_name and _metadata_json fields
        index_name_field = self._index.schema.fields.get(_INDEX_NAME_FIELD)
        has_metadata_json_field = any(
            field.name == "_metadata_json"
            for field in self._index.schema.fields.values()
        )

        # Build records to load to SearchIndex
        records = []
        for text, embedding, metadata in zip(
            texts_list,
            document_embeddings,
            metadatas or [{}] * len(texts_list),
        ):
            record = {
                self.config.content_field: text,
                self.config.embedding_field: (
                    embedding
                    if self.config.storage_type == StorageType.JSON.value
                    else array_to_buffer(embedding, dtype=self.config.vector_datatype)
                ),
            }

            # Only add _metadata_json if the field exists in the schema
            if has_metadata_json_field:
                metadata_json = json.dumps(metadata)
                record["_metadata_json"] = metadata_json
            for field_name, field_value in metadata.items():
                # _index_name is reserved for the internal ownership marker. The
                # caller's value remains available through _metadata_json.
                if field_name == _INDEX_NAME_FIELD:
                    continue
                # Skip empty values
                if field_value is None:
                    continue
                # Convert lists to tag strings with separator
                elif isinstance(field_value, list):
                    record[field_name] = self.config.default_tag_separator.join(
                        field_value
                    )
                else:
                    record[field_name] = field_value

            # Assign the protected marker after user metadata so it cannot be
            # overridden by a conflicting metadata field.
            if index_name_field is not None:
                record[_INDEX_NAME_FIELD] = self._index_name_value(
                    index_name_field.type
                )
            records.append(record)

        # Load records into the index
        primary_prefix = self.config.primary_prefix
        if keys:
            # Already have key_prefix in index definition (with ending colon).
            # New documents are always written under the primary prefix.
            record_keys = [f"{primary_prefix}:{key}" for key in keys]
            result = self._index.load(records, keys=record_keys, ttl=self.ttl)
        else:
            result = self._index.load(records, ttl=self.ttl)

        if result is None:
            return []

        # `SearchIndex.load` returns the full Redis keys it wrote, prefix
        # included. `delete()` and `get_by_ids()` take bare ids and add that
        # same prefix themselves, so strip it back off here. Otherwise the ids
        # this method returns can't be passed to either of those without
        # ending up double-prefixed.
        if primary_prefix:
            full_prefix = f"{primary_prefix}:"
            return [
                key[len(full_prefix) :] if key.startswith(full_prefix) else key
                for key in result
            ]
        return list(result)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        config: Optional[RedisConfig] = None,
        keys: Optional[List[str]] = None,
        return_keys: bool = False,
        **kwargs: Any,
    ) -> RedisVectorStore:
        """Create a `RedisVectorStore` from a list of texts.

        Args:
            texts: List of texts to add to the vector store.
            embedding: Embedding function to use for encoding the texts.
            metadatas: Optional list of metadata dicts associated with the texts.
            config: Optional `RedisConfig` object. If not provided, one will be created
                from kwargs.
            keys: Optional list of keys to associate with the documents.
            return_keys: Whether to return the keys of the added documents.
            **kwargs: Additional keyword arguments to pass to `RedisConfig` if config is
                not provided.

                Common kwargs include:

                - `index_name`: Name of the Redis index to create.
                - `redis_url`: URL of the Redis instance to connect to.
                - `distance_metric`: Distance metric to use for similarity search.
                    - Default is `'COSINE'`.
                - `indexing_algorithm`: Indexing algorithm to use.
                    - Default is `'FLAT'`.

        Returns:
            RedisVectorStore: A new `RedisVectorStore` instance with the texts added.

        Example:
            ```python
            from langchain_redis import RedisVectorStore
            from langchain_openai import OpenAIEmbeddings

            texts = [
                "The quick brown fox jumps over the lazy dog",
                "Hello world",
                "Machine learning is fascinating"
            ]
            metadatas = [
                {"source": "book", "page": 1},
                {"source": "greeting", "language": "english"},
                {"source": "article", "topic": "AI"}
            ]

            embeddings = OpenAIEmbeddings()

            vector_store = RedisVectorStore.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas,
                index_name="langchain-demo",
                redis_url="redis://localhost:6379",
                distance_metric="COSINE"
            )

            # Now you can use the vector_store for similarity search
            results = vector_store.similarity_search("AI and machine learning", k=1)
            print(results[0].page_content)
            ```

        Note:
            - This method creates a new `RedisVectorStore` instance and adds the
                provided texts to it.
            - If `metadatas` is provided, it must have the same length as `texts`.
            - If `keys` is provided, it must have the same length as `texts`.
            - The `return_keys` parameter determines whether the method returns just the
                `RedisVectorStore` instance or a tuple of
                `(RedisVectorStore, List[str])` where the second element is the list of
                keys for the added documents.
        """
        config = config or RedisConfig.from_kwargs(**kwargs)

        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]

        vector_store = cls(embeddings=embedding, config=config, **kwargs)
        out_keys = vector_store.add_texts(texts, metadatas, keys)  # type: ignore

        if return_keys:
            return cast(RedisVectorStore, (vector_store, out_keys))
        else:
            return vector_store

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        config: Optional[RedisConfig] = None,
        return_keys: bool = False,
        **kwargs: Any,
    ) -> RedisVectorStore:
        """Create a `RedisVectorStore` from a list of `Document` objects.

        Args:
            documents: List of `Document` objects to add to the vector store.
            embedding: `Embeddings` object to use for encoding the documents.
            config: Optional `RedisConfig` object.

                If not provided, one will be created from kwargs.
            return_keys: Whether to return the keys of the added documents.
            **kwargs: Additional keyword arguments to pass to `RedisConfig` if config
                is not provided.

                Common kwargs include:

                - `index_name`: Name of the Redis index to create.
                - `redis_url`: URL of the Redis instance to connect to.
                - `distance_metric`: Distance metric to use for similarity search.
                    - Default is `'COSINE'`.
                - indexing_algorithm: Indexing algorithm to use.
                    - Default is `'FLAT'`.

        Returns:
            RedisVectorStore: A new `RedisVectorStore` instance with the documents
                added.

        Example:
            ```python
            from langchain_redis import RedisVectorStore
            from langchain_openai import OpenAIEmbeddings
            from langchain_core.documents import Document

            documents = [
                Document(
                    page_content="The quick brown fox",
                    metadata={"animal": "fox"}
                ),
                Document(
                    page_content="jumps over the lazy dog",
                    metadata={"animal": "dog"}
                )
            ]

            embeddings = OpenAIEmbeddings()

            vector_store = RedisVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                index_name="animal-docs",
                redis_url="redis://localhost:6379"
            )

            # Now you can use the vector_store for similarity search
            results = vector_store.similarity_search("quick animal", k=1)
            print(results[0].page_content)
            ```

        Note:
            - This method creates a new `RedisVectorStore` instance and adds the
                provided documents to it.
            - The method extracts the text content and metadata from
                each `Document` object.
            - If a `RedisConfig` object is not provided, one will be created using
                the additional kwargs passed to this method.
            - The embedding function is used to convert the document text into vector
                representations for efficient similarity search.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        config = config or RedisConfig.from_kwargs(**kwargs)

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            config=config,
            return_keys=return_keys,
            **kwargs,
        )

    @classmethod
    def from_existing_index(
        cls,
        index_name: str,
        embedding: Embeddings,
        **kwargs: Any,
    ) -> RedisVectorStore:
        """Create a `RedisVectorStore` from an existing Redis Search Index.

        This method allows you to connect to an already existing index in Redis,
        which can be useful for continuing work with previously created indexes
        or for connecting to indexes created outside of this client.

        Args:
            index_name: Name of the existing index to use.
            embedding: Embedding function to use for encoding queries.
            **kwargs: Additional keyword arguments to pass to `RedisConfig`.

                Common kwargs include:

                - `redis_url`: URL of the Redis instance to connect to.
                - `redis_client`: Pre-existing Redis client to use.
                - `vector_query_field`: Name of the field containing the vector
                    representations.
                - `content_field`: Name of the field containing the document content.

        Returns:
            RedisVectorStore: A new `RedisVectorStore` instance connected to the
                existing index.

        Example:
            ```python
            from langchain_redis import RedisVectorStore
            from langchain_openai import OpenAIEmbeddings
            from redis import Redis

            embeddings = OpenAIEmbeddings()

            # Connect to an existing index
            vector_store = RedisVectorStore.from_existing_index(
                index_name="my-existing-index",
                embedding=embeddings,
                redis_url="redis://localhost:6379",
                vector_query_field="embedding",
                content_field="text"
            )

            # Now you can use the vector_store for similarity search
            results = vector_store.similarity_search("AI and machine learning", k=1)
            print(results[0].page_content)
            ```

        Note:
            - This method assumes that the index already exists in Redis.
            - The embedding function provided should be compatible with the embeddings
                stored in the existing index.
            - If you're using custom field names for vectors or content in your
                existing index, make sure to specify them using `vector_query_field` and
                `content_field` respectively.
            - This method is useful for scenarios where you want to reuse an
                existing index, such as when the index was created by another process
                or when you want to use the same index across different sessions
                or applications.
        """
        config = RedisConfig.from_kwargs(**kwargs)
        config.index_name = index_name
        config.from_existing = True

        return RedisVectorStore(embedding, config=config, **kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents by id.

        Args:
            ids: Optional list of ids of the documents to delete.
            **kwargs: Additional keyword arguments. A non-`None` `filter` is
                rejected; use `delete_by_filter()` for filter-based deletion.

        Returns:
            Optional[bool]: `True` if one or more documents were deleted,
                `False` otherwise.

        Raises:
            ValueError: If a non-`None` `filter` argument is provided.

        Example:
            ```python
            vector_store.delete(ids=["doc1", "doc2", "doc3"])
            ```

        Note:
            - If `ids` is omitted or empty, the method returns `False`.
            - When the index defines a recognized `_index_name` marker, only
                records whose stored marker matches this index are deleted. The
                ownership check and deletion are currently separate operations.
            - The ids path uses RedisVL's `drop_keys`; keys are constructed by
                prefixing each id with the configured primary key prefix.
            - Use `delete_by_filter()` for explicit filter deletion, exact
                counts, and dry-run support.
        """
        filter = kwargs.get("filter")
        if filter is not None:
            raise ValueError(
                "delete(filter=...) is not supported. Use delete_by_filter() "
                "for filter-based deletion."
            )
        if ids and len(ids) > 0:
            keys = self._redis_keys(ids)
            expected_marker = self._index_name_marker_for_delete()
            records = self._fetch_records_by_keys(keys)
            keys = [
                key
                for key, record in zip(keys, records)
                if record and self._record_belongs_to_index(record, expected_marker)
            ]
            if not keys:
                return False
            # Always return True if we delete at least one key
            # This matches the behavior expected by the tests
            return self._index.drop_keys(keys) > 0
        else:
            return False

    def delete_by_filter(
        self,
        filter: FilterExpression,
        *,
        dry_run: bool = False,
        batch_size: Optional[int] = None,
    ) -> int:
        """Delete every document in this index matching a filter expression.

        Args:
            filter: RedisVL `FilterExpression` selecting the documents to
                delete — the same filter builder style accepted by
                `similarity_search`, including wildcard tag patterns like
                `Tag("source") % "docs-v1*"`. Raw filter strings are not
                accepted for mutating operations because they cannot be
                safely combined with the internal index-scoping filter.
            dry_run: If `True`, nothing is deleted; the return value is the
                number of documents that would be deleted.
            batch_size: Optional number of documents to resolve and delete
                per round-trip.

        Returns:
            int: The number of documents deleted (or matched, for a dry run).

        Example:
            ```python
            from redisvl.query.filter import Tag

            # Preview a purge, then run it
            would_delete = vector_store.delete_by_filter(
                Tag("tenant_id") == "acme", dry_run=True
            )
            deleted = vector_store.delete_by_filter(Tag("tenant_id") == "acme")
            ```

        Note:
            - Generated schemas use an exact `_index_name` TAG marker to keep
                indexes sharing a `key_prefix` from deleting each other's data.
                Existing or custom schemas without that TAG marker are refused;
                recreate or migrate the index before using filter deletion.
                Custom TAG markers containing raw index names must also be
                reindexed with the current hashed marker values; changing only
                the field type is not sufficient. Legacy TEXT markers remain
                readable, but cannot be used for filter deletion.
            - Use RedisVL filter builders such as `Tag`, `Num`, or `Text`
                instead of raw RediSearch filter strings.
            - Filters that render to Redis's global match-all expression are
                refused; use `index.clear()` for an intentional full-index
                operation.
        """
        scoped_filter = self._prepare_bulk_filter(filter, "delete_by_filter")
        bulk_kwargs: Dict[str, Any] = {"dry_run": dry_run}
        if batch_size is not None:
            bulk_kwargs["batch_size"] = batch_size
        result = self._index.drop_by_filter(scoped_filter, **bulk_kwargs)
        return result.matched if dry_run else result.processed

    def _prepare_bulk_filter(
        self,
        filter: Optional[FilterExpression],
        operation: str,
    ) -> FilterExpression:
        """Validate and scope a destructive bulk-operation filter.

        Filter deletion requires a RedisVL `FilterExpression` so the user's
        filter can be combined with the internal `_index_name` guard before
        any mutation is sent to Redis. Filters that render to Redis's global
        match-all expression (`*`) are intentionally rejected as a safety
        guardrail; use `index.clear()` for an intentional full-index operation.
        """
        if filter is None:
            raise ValueError(
                f"{operation} requires a RedisVL FilterExpression. To delete "
                "specific documents use delete(ids=...)."
            )
        if not isinstance(filter, FilterExpression):
            raise ValueError(
                f"{operation} strictly requires a RedisVL FilterExpression. "
                "Use filter builders like Tag, Num, or Text instead of raw strings."
            )

        # Validate before adding the index guard; otherwise `*` becomes a scoped
        # expression and bypasses RedisVL's match-all protection.
        try:
            rendered_filter = str(filter).strip()
        except ValueError as exc:
            raise ValueError(
                f"{operation} requires a specific, initialized filter expression."
            ) from exc

        match_all_candidate = rendered_filter
        while match_all_candidate.startswith("(") and match_all_candidate.endswith(")"):
            match_all_candidate = match_all_candidate[1:-1].strip()
        if match_all_candidate in ("", "*"):
            raise ValueError(
                f"{operation} refuses filters that match all documents. "
                "Use index.clear() for an intentional full-index operation."
            )

        return filter & self._require_exact_index_name_filter()

    def _index_name_value(self, field_type: FieldTypes) -> str:
        """Return the stored and queried value for the index marker."""
        if field_type == FieldTypes.TAG:
            return hashify(self._index.name)
        return self.config.index_name

    def _index_name_marker_from_schema(self) -> Optional[str]:
        """Derive the ownership marker from the configured index schema."""
        field = self._index.schema.fields.get(_INDEX_NAME_FIELD)
        if field is None or field.type not in (FieldTypes.TAG, FieldTypes.TEXT):
            return None
        return self._index_name_value(field.type)

    def _index_name_marker_for_read(self) -> Optional[str]:
        """Return the ownership marker, tolerating legacy custom schemas."""
        try:
            return self._index_name_marker_from_schema()
        except Exception:
            return None

    def _index_name_marker_for_delete(self) -> Optional[str]:
        """Return the ownership marker or refuse deletion on inspection failure."""
        try:
            return self._index_name_marker_from_schema()
        except Exception as exc:
            raise ValueError(
                "delete() could not inspect the index schema; the deletion was refused."
            ) from exc

    @staticmethod
    def _record_belongs_to_index(
        record: Dict[str, Any], expected_marker: Optional[str]
    ) -> bool:
        """Return whether a fetched record belongs to the current index."""
        record_marker = record.get(_INDEX_NAME_FIELD)
        return expected_marker is None or record_marker == expected_marker

    def _redis_keys(self, ids: Sequence[str]) -> List[str]:
        """Return Redis keys for vector-store document IDs."""
        if self.config.primary_prefix:
            return [f"{self.config.primary_prefix}:{_id}" for _id in ids]
        return list(ids)

    def _fetch_records_by_keys(
        self, keys: Sequence[str]
    ) -> List[Optional[Dict[str, Any]]]:
        """Fetch raw HASH or JSON records for ownership validation."""
        redis = self.config.redis()
        if self.config.storage_type == StorageType.JSON.value:
            return cast(
                List[Optional[Dict[str, Any]]], redis.json().mget(list(keys), ".")
            )

        pipe = redis.pipeline()
        for key in keys:
            pipe.hgetall(key)
        return [convert_bytes(value) if value else None for value in pipe.execute()]

    def _build_index_name_filter(
        self,
        schema: IndexSchema,
        *,
        allow_text: bool,
    ) -> Optional[FilterExpression]:
        """Build an index namespace filter from the provided schema."""
        field = schema.fields.get(_INDEX_NAME_FIELD)
        if field is None:
            return None

        value = self._index_name_value(field.type)
        if field.type == FieldTypes.TAG:
            return Tag(_INDEX_NAME_FIELD) == value
        if allow_text and field.type == FieldTypes.TEXT:
            return Text(_INDEX_NAME_FIELD) == value
        return None

    def _live_index_schema(self) -> IndexSchema:
        """Fetch the schema currently installed in Redis."""
        return SearchIndex.from_existing(
            name=self._index.name,
            redis_client=self._index.client,
            lib_name=__lib_name__,
        ).schema

    def _require_exact_index_name_filter(self) -> FilterExpression:
        """Return an exact TAG scope or refuse filter-based deletion."""
        try:
            field = self._live_index_schema().fields.get(_INDEX_NAME_FIELD)
        except Exception as exc:
            raise ValueError(
                "delete_by_filter() could not inspect the live index schema; "
                "the deletion was refused."
            ) from exc

        if field is None or field.type != FieldTypes.TAG:
            raise ValueError(
                "delete_by_filter() requires an '_index_name' TAG field. "
                "Recreate or migrate this index before using filter deletion."
            )

        if field.attrs.no_index:
            raise ValueError(
                "delete_by_filter() requires the '_index_name' TAG field to be "
                "indexed. Recreate or migrate this index before using filter "
                "deletion."
            )

        marker = hashify(self._index.name)
        if field.attrs.separator and field.attrs.separator in marker:
            raise ValueError(
                "delete_by_filter() cannot safely use the '_index_name' TAG "
                "separator because it splits the index ownership marker. "
                "Recreate or migrate this index with a compatible separator."
            )

        return Tag(_INDEX_NAME_FIELD) == marker

    def _with_index_name_filter(
        self, filter: Optional[Union[str, FilterExpression]]
    ) -> Optional[Union[str, FilterExpression]]:
        """Restrict a filter to documents belonging to the current index.

        Reads retain best-effort compatibility with existing TEXT markers and
        custom schemas that do not define an `_index_name` field.
        """
        try:
            index_filter = self._build_index_name_filter(
                self._index.schema, allow_text=True
            )
        except Exception:
            return filter

        if index_filter is None:
            return filter
        if filter is None:
            return index_filter
        if isinstance(filter, FilterExpression):
            return filter & index_filter
        return filter

    def _query_builder(
        self,
        embedding: Union[List[float], bytes],
        k: int = 10,
        distance_threshold: Any = None,
        sort_by: Optional[str] = None,
        filter: Optional[Union[str, FilterExpression]] = None,
        return_fields: Optional[List[str]] = None,
    ) -> Union[VectorQuery, RangeQuery]:
        filter = self._with_index_name_filter(filter)
        if distance_threshold is None:
            return VectorQuery(
                vector=embedding,
                vector_field_name=self.config.embedding_field,
                return_fields=return_fields,
                num_results=k,
                filter_expression=filter,
                sort_by=sort_by,
            )
        else:
            return RangeQuery(
                vector=embedding,
                vector_field_name=self.config.embedding_field,
                return_fields=return_fields,
                num_results=k,
                filter_expression=filter,
                distance_threshold=distance_threshold,
                sort_by=sort_by,
            )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[FilterExpression] = None,
        sort_by: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of `Document` objects to return.
            filter: Optional `filter` expression to apply.
            sort_by: Optional `sort_by` expression to apply.
            **kwargs: Other keyword arguments.

                Common kwargs include:

                - `return_metadata`: Whether to return metadata.
                    - Defaults to `True`.
                - `distance_threshold`: Optional distance threshold for filtering
                    results.
                - `return_all`: Whether to return all data in the Hash/JSON including
                    non-indexed fields

        Returns:
            List of `Document` objects most similar to the query vector.
        """
        return_metadata = kwargs.get("return_metadata", True)
        distance_threshold = kwargs.get("distance_threshold", None)
        return_all = kwargs.get("return_all", False)

        return_fields = []

        if not return_all:
            return_fields = [self.config.content_field]
            if return_metadata:
                return_fields += [
                    field.name
                    for field in self._index.schema.fields.values()
                    if field.name
                    not in [self.config.embedding_field, self.config.content_field]
                ]

        query = self._query_builder(
            distance_threshold=distance_threshold,
            embedding=embedding,
            k=k,
            sort_by=sort_by,
            filter=filter,
            return_fields=return_fields,
        )

        results = self._index.query(query)

        if not return_all:
            return cast(
                List[Document], self._prepare_docs(return_all, results, return_metadata)
            )
        else:
            if self.config.storage_type == StorageType.HASH.value:
                # Fetch full hash data for each document
                if not results:
                    full_docs = []
                else:
                    with self._index.client.pipeline(transaction=False) as pipe:
                        for doc in results:
                            pipe.hgetall(doc["id"])
                        full_docs = convert_bytes(pipe.execute())

                return cast(
                    List[Document],
                    self._prepare_docs_full(
                        return_all, results, full_docs, return_metadata
                    ),
                )
            else:
                # Fetch full JSON data for each document
                if not results:
                    full_docs = []
                else:
                    with self._index.client.json().pipeline(transaction=False) as pipe:
                        for doc in results:
                            pipe.get(doc["id"], ".")
                        full_docs = pipe.execute()

                return cast(
                    List[Document],
                    self._prepare_docs_full(
                        return_all, results, full_docs, return_metadata
                    ),
                )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[FilterExpression] = None,
        sort_by: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of `Document` objects to return.
            filter: Optional `filter` expression to apply. Tag filters
                support wildcard patterns via the modulo operator, e.g.
                `Tag("category") % "elec*"`.
            sort_by: Optional `sort_by` expression to apply.
            **kwargs: Other keyword arguments to pass to the search function.

        Returns:
            List of `Document` objects most similar to the query.
        """
        embedding = self._embeddings.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, filter, sort_by, **kwargs)

    def _default_return_fields(self, return_metadata: bool) -> List[str]:
        """Return fields for search queries: content plus indexed metadata."""
        return_fields = [self.config.content_field]
        if return_metadata:
            return_fields += [
                field.name
                for field in self._index.schema.fields.values()
                if field.name
                not in [self.config.embedding_field, self.config.content_field]
            ]
        return return_fields

    def _supports_ft_hybrid(self) -> Optional[bool]:
        """Whether the server supports `FT.HYBRID` (Redis >= 8.4.0).

        Returns `None` when the server version cannot be determined. The
        result is cached on the instance after the first successful probe.
        """
        if self._ft_hybrid_support is None:
            try:
                info = self._index.client.info("server")
                version = str(info.get("redis_version", ""))
                major, minor = (int(part) for part in version.split(".")[:2])
                self._ft_hybrid_support = (major, minor) >= (8, 4)
            except Exception:
                return None
        return self._ft_hybrid_support

    def _resolve_hybrid_options(
        self,
        method: str,
        combination_method: str,
        alpha: Optional[float],
    ) -> Tuple[str, str, Optional[float]]:
        """Normalize and validate hybrid-search options and select an engine."""
        method = method.lower()
        combination_method = combination_method.upper()

        if method not in ("auto", "ft_hybrid", "aggregate"):
            raise ValueError(
                f"Unknown hybrid search method: {method!r}. "
                "Expected 'auto', 'ft_hybrid' or 'aggregate'."
            )
        if combination_method not in ("LINEAR", "RRF"):
            raise ValueError(
                f"Unknown combination method: {combination_method!r}. "
                "Expected 'LINEAR' or 'RRF'."
            )

        if combination_method == "LINEAR":
            alpha = 0.7 if alpha is None else alpha
            if not 0 < alpha < 1:
                raise ValueError(
                    "alpha must be strictly between 0 and 1 for LINEAR fusion."
                )
        elif alpha is not None:
            raise ValueError("alpha cannot be used with RRF fusion.")

        if method == "aggregate":
            if combination_method == "RRF":
                raise ValueError(
                    "RRF fusion is unavailable with method='aggregate'. "
                    "Use LINEAR or method='ft_hybrid'."
                )
            return method, combination_method, alpha

        supports_ft_hybrid = self._supports_ft_hybrid()
        if method == "ft_hybrid":
            if supports_ft_hybrid is False:
                raise ValueError(
                    "method='ft_hybrid' requires Redis >= 8.4.0 (the FT.HYBRID "
                    "command). Use method='aggregate' on older servers."
                )
            return method, combination_method, alpha

        if supports_ft_hybrid:
            return "ft_hybrid", combination_method, alpha
        if combination_method == "RRF":
            raise ValueError(
                "RRF fusion requires Redis >= 8.4.0 and cannot use the "
                "aggregate fallback."
            )
        return "aggregate", combination_method, alpha

    def hybrid_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[FilterExpression] = None,
        *,
        text_field: Optional[str] = None,
        method: str = "auto",
        combination_method: str = "LINEAR",
        alpha: Optional[float] = None,
        text_scorer: str = "BM25STD",
        text_weights: Optional[Dict[str, float]] = None,
        stopwords: Optional[Union[str, Set[str]]] = "english",
        return_metadata: bool = True,
    ) -> List[Tuple[Document, float]]:
        """Return docs scored by combined full-text and vector similarity.

        Args:
            query: Query text. It is used both for full-text scoring and,
                embedded, for vector similarity.
            k: Number of `Document` objects to return.
            filter: Optional `FilterExpression` to apply.
            text_field: Text field to search in. Defaults to the configured
                `content_field`.
            method: `'ft_hybrid'` uses the `FT.HYBRID` command (Redis 8.4+),
                `'aggregate'` uses an `FT.AGGREGATE`-based combination that
                works on any Redis with the Query Engine. `'auto'` (default)
                probes the server version and picks accordingly.
            combination_method: `'LINEAR'` (default) or `'RRF'` (reciprocal
                rank fusion). RRF requires native `FT.HYBRID` support.
            alpha: Weight of the vector similarity in linear combination:
                `score = alpha * vector_score + (1 - alpha) * text_score`.
                Defaults to 0.7 for LINEAR and must be strictly between 0 and
                1. Do not provide it with RRF.
            text_scorer: Full-text scoring algorithm (default `'BM25STD'`).
            text_weights: Optional per-word importance weights for the
                full-text part of the query.
            stopwords: Stopwords to strip from the query text client-side.
                Language string, set of words, or `None` to disable.
            return_metadata: Whether to return metadata with the documents.

        Returns:
            List of `(Document, score)` tuples, best first. Higher scores
            are better. Scores are combined hybrid scores — rank-based for
            `'RRF'`, weighted sums for linear combination — and are not
            comparable with the cosine distances returned by
            `similarity_search_with_score`.

        Example:
            ```python
            from redisvl.query.filter import Tag

            results = vector_store.hybrid_search_with_score(
                "durable message queue",
                k=5,
                filter=Tag("category") == "infra",
                alpha=0.5,
            )
            for doc, score in results:
                print(score, doc.page_content)
            ```
        """
        resolved, combination_method, alpha = self._resolve_hybrid_options(
            method, combination_method, alpha
        )

        embedding = self._embeddings.embed_query(query)
        filter = self._with_index_name_filter(filter)
        return_fields = self._default_return_fields(return_metadata)
        dtype = self.config.vector_datatype.lower()
        text_field = text_field or self.config.content_field

        query_kwargs: Dict[str, Any] = {
            "text": query,
            "text_field_name": text_field,
            "vector": embedding,
            "vector_field_name": self.config.embedding_field,
            "text_scorer": text_scorer,
            "filter_expression": filter,
            "dtype": dtype,
            "num_results": k,
            "return_fields": return_fields,
            "stopwords": stopwords,
            "text_weights": text_weights,
        }

        hybrid_query: Union[HybridQuery, AggregateHybridQuery]
        if resolved == "ft_hybrid":
            native_kwargs: Dict[str, Any] = {
                "combination_method": combination_method,
                "yield_combined_score_as": "hybrid_score",
            }
            if combination_method == "LINEAR":
                assert alpha is not None
                # Our alpha weights the vector score; FT.HYBRID's linear_alpha
                # weights the text score.
                native_kwargs["linear_alpha"] = 1 - alpha
            hybrid_query = HybridQuery(**query_kwargs, **native_kwargs)
        else:
            assert alpha is not None
            hybrid_query = AggregateHybridQuery(**query_kwargs, alpha=alpha)

        results = self._index.query(hybrid_query)

        docs_with_scores = []
        for res in results:
            score = float(res.get("hybrid_score", 0.0))
            doc_fields = {
                key: value
                for key, value in res.items()
                if key not in ("hybrid_score", "text_score", "vector_similarity")
            }
            doc = self._build_document_from_result(doc_fields)
            if not return_metadata:
                doc.metadata = {}
            docs_with_scores.append((doc, score))
        return docs_with_scores

    def hybrid_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[FilterExpression] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs ranked by combined full-text and vector similarity.

        Args:
            query: Query text, used for both full-text and vector scoring.
            k: Number of `Document` objects to return.
            filter: Optional `FilterExpression` to apply.
            **kwargs: See `hybrid_search_with_score` for the remaining
                keyword arguments (`method`, `combination_method`, `alpha`,
                `text_field`, `text_scorer`, `text_weights`, `stopwords`,
                `return_metadata`).

        Returns:
            List of `Document` objects, best first.
        """
        return [
            doc for doc, _ in self.hybrid_search_with_score(query, k, filter, **kwargs)
        ]

    def full_text_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[FilterExpression] = None,
        *,
        text_fields: Optional[Union[str, Dict[str, float]]] = None,
        text_scorer: str = "BM25STD",
        stopwords: Optional[Union[str, Set[str]]] = "english",
        return_metadata: bool = True,
    ) -> List[Document]:
        """Return docs matching the query by full-text relevance only.

        Args:
            query: Full-text query string.
            k: Number of `Document` objects to return.
            filter: Optional `FilterExpression` to apply.
            text_fields: Text field to search in, or a mapping of field
                names to weights (e.g. `{"title": 5.0, "text": 1.0}`).
                Defaults to the configured `content_field`.
            text_scorer: Full-text scoring algorithm (default `'BM25STD'`).
            stopwords: Stopwords to strip from the query text client-side.
                Language string, set of words, or `None` to disable.
            return_metadata: Whether to return metadata with the documents.

        Returns:
            List of `Document` objects, best first.
        """
        text_query = TextQuery(
            text=query,
            text_field_name=text_fields or self.config.content_field,
            text_scorer=text_scorer,
            filter_expression=self._with_index_name_filter(filter),
            return_fields=self._default_return_fields(return_metadata),
            num_results=k,
            return_score=False,
            stopwords=stopwords,
        )

        results = self._index.query(text_query)
        return cast(List[Document], self._prepare_docs(False, results, return_metadata))

    def _build_document_from_result(self, res: Dict[str, Any]) -> Document:
        """Build a `Document` object from a Redis search result."""
        # Get the document content
        content = res[self.config.content_field]

        # Process metadata - first try to use the JSON metadata
        metadata = {}
        if "_metadata_json" in res:
            try:
                # Try to parse the JSON metadata
                metadata = json.loads(res["_metadata_json"])
            except (json.JSONDecodeError, TypeError):
                # Fall back to extracting metadata fields from result directly
                metadata = self._extract_metadata_from_result(res)
        else:
            # Fall back to extracting metadata fields from result directly
            metadata = self._extract_metadata_from_result(res)

        return Document(page_content=content, metadata=metadata)

    def _extract_metadata_from_result(self, res: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata fields from a search result without `_metadata_json`."""
        metadata = {}
        # Extract all fields except for special ones
        for key, value in res.items():
            if (
                key != self.config.content_field
                and key != self.config.embedding_field
                and key != "_index_name"
                and not key.startswith("_")
            ):
                # Try to convert string numbers to their native types
                # This helps when comparing metadata in tests
                if isinstance(value, str):
                    try:
                        # Try to convert to int first
                        if value.isdigit():
                            metadata[key] = int(value)  # type: ignore
                        # Then try float
                        elif value.replace(".", "", 1).isdigit():
                            metadata[key] = float(value)  # type: ignore
                        else:
                            metadata[key] = value  # type: ignore
                    except (ValueError, TypeError):
                        metadata[key] = value  # type: ignore
                else:
                    metadata[key] = value
        return metadata

    def _prepare_docs(
        self,
        return_all: bool | None,
        results: List[Dict[str, Any]],
        return_metadata: bool,
        with_vectors: bool = False,
        with_scores: bool = False,
    ) -> Sequence[Any]:
        docs = []

        for res in results:
            # Use the _build_document_from_result method to ensure complete metadata
            # is properly reconstructed from the stored JSON
            lc_doc = self._build_document_from_result(res)

            # If return_metadata is False, clear the metadata
            if not return_metadata:
                lc_doc.metadata = {}

            if with_scores:
                vector_distance = float(res.get("vector_distance", 0))
                parsed = (lc_doc, vector_distance)  # type: ignore

            if with_vectors:
                vector = self.convert_vector(res)
                parsed = (lc_doc, vector_distance, vector)  # type: ignore

            if not with_scores and not with_vectors:
                parsed = lc_doc  # type: ignore

            docs.append(parsed)

        return docs

    def convert_vector(self, obj: dict) -> List[float]:
        vector = obj.get(self.config.embedding_field)
        if isinstance(vector, bytes):
            vector = buffer_to_array(vector, dtype=self.config.vector_datatype)  # type: ignore
        if isinstance(vector, str):
            vector = ast.literal_eval(vector)  # type: ignore

        return vector  # type: ignore[return-value]

    def _prepare_docs_full(
        self,
        return_all: bool | None,
        results: List[Dict[str, Any]],
        full_docs: List[Dict[str, Any]],
        return_metadata: bool,
        with_vectors: bool = False,
        with_scores: bool = False,
    ) -> Sequence[Any]:
        docs = []

        for fdoc, res in zip(full_docs, results):
            if fdoc is None:
                continue

            if not return_all:
                metadata = (
                    {
                        field.name: res[field.name]
                        for field in self._index.schema.fields.values()
                        if field.name
                        not in [
                            self.config.embedding_field,
                            self.config.content_field,
                            "_index_name",
                        ]
                    }
                    if return_metadata
                    else {}
                )
            else:
                metadata = {
                    k: v
                    for k, v in fdoc.items()
                    if (
                        k != self.config.content_field
                        and k != self.config.embedding_field
                    )
                }

            lc_doc = Document(
                id=res[self.config.id_field],
                page_content=fdoc[self.config.content_field],
                metadata=metadata,
            )

            if with_scores:
                vector_distance = float(res.get("vector_distance", 0))
                parsed = (lc_doc, vector_distance)  # type: ignore

            if with_vectors:
                vector = self.convert_vector(fdoc)
                parsed = (lc_doc, vector_distance, vector)  # type: ignore

            if not with_scores and not with_vectors:
                parsed = lc_doc  # type: ignore

            docs.append(parsed)

        return docs

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[FilterExpression] = None,
        sort_by: Optional[str] = None,
        **kwargs: Any,
    ) -> Sequence[Any]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of `Document` objects to return.
            filter: Optional `filter` expression to apply.
            sort_by: Optional `sort_by` expression to apply.
            **kwargs: Other keyword arguments.

                Common kwargs include:

                - `with_vectors`: Whether to return document vectors.
                    - Defaults to `False`.
                - `return_metadata`: Whether to return metadata.
                    - Defaults to `True`.
                - `distance_threshold`: Optional distance threshold for filtering
                    results.

        Returns:
            List of tuples of `Document` objects most similar to the query vector,
                score, and optionally the document vector.
        """
        with_vectors = kwargs.get("with_vectors", False)
        return_metadata = kwargs.get("return_metadata", True)
        distance_threshold = kwargs.get("distance_threshold")
        return_all = kwargs.get("return_all", False)

        return_fields = []

        if not return_all:
            return_fields = [self.config.content_field]
            if return_metadata:
                return_fields += [
                    field.name
                    for field in self._index.schema.fields.values()
                    if field.name
                    not in [self.config.embedding_field, self.config.content_field]
                ]

            if with_vectors:
                return_fields.append(self.config.embedding_field)

        query = self._query_builder(
            distance_threshold=distance_threshold,
            embedding=embedding,
            k=k,
            sort_by=sort_by,
            filter=filter,
            return_fields=return_fields,
        )

        if not return_all:
            if with_vectors:
                query.return_field(
                    self.config.embedding_field,
                    decode_field=(self.config.storage_type != StorageType.HASH.value),
                )

            results = self._index.query(query)

            docs_with_scores = self._prepare_docs(
                return_all,
                results,
                return_metadata,
                with_vectors=with_vectors,
                with_scores=True,
            )
        else:
            results = self._index.query(query)

            if self.config.storage_type == StorageType.HASH.value:
                # Fetch full hash data for each document
                pipe = self._index.client.pipeline()
                for doc in results:
                    pipe.hgetall(doc["id"])
                full_docs = convert_bytes(pipe.execute())

                docs_with_scores = self._prepare_docs_full(
                    return_all,
                    results,
                    full_docs,
                    return_metadata,
                    with_vectors=with_vectors,
                    with_scores=True,
                )
            else:
                # Fetch full JSON data for each document
                doc_ids = [doc["id"] for doc in results]
                full_docs = self._index.client.json().mget(doc_ids, ".")

                docs_with_scores = self._prepare_docs_full(
                    return_all,
                    results,
                    full_docs,
                    return_metadata,
                    with_vectors=with_vectors,
                    with_scores=True,
                )

        return docs_with_scores

    def similarity_search_with_score(  # type: ignore[override]
        self,
        query: str,
        k: int = 4,
        filter: Optional[FilterExpression] = None,
        sort_by: Optional[str] = None,
        **kwargs: Any,
    ) -> Sequence[Any]:
        r"""Return documents most similar to query string, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of `Document` objects to return.
            filter: Optional `filter` expression to apply to the query.
            sort_by: Optional `sort_by` expression to apply to the query.
            **kwargs: Other keyword arguments to pass to the search function.

                Common kwargs include:

                - `custom_query`: Optional callable that can be used
                    to customize the query.
                - `doc_builder`: Optional callable to customize `Document` creation.
                - `return_metadata`: Whether to return metadata.
                    - Defaults to `True`.
                - `distance_threshold`: Optional distance threshold for filtering
                    results.
                - `return_all`: Whether to return all data in the Hash/JSON including
                    non-indexed fields.
                    - Defaults to `False`.

        Returns:
            List of tuples of `(Document, score)` most similar to the query.

        Example:
            ```python
            from langchain_redis import RedisVectorStore
            from langchain_openai import OpenAIEmbeddings

            vector_store = RedisVectorStore(
                embeddings=OpenAIEmbeddings(),
                index_name="langchain-demo",
                redis_url="redis://localhost:6379",
            )

            results = vector_store.similarity_search_with_score(
                "What is machine learning?",
                k=2,
                filter=None
            )

            for doc, score in results:
                print(f"Score: {score}")
                print(f"Content: {doc.page_content}")
                print(f"Metadata: {doc.metadata}\n")
            ```

        Note:
            - The method returns scores along with documents. Lower scores indicate
                higher similarity.
            - The actual search is performed using the vector representation of the
                query, which is why an embedding function must be provided during
                initialization.
            - The `filter` parameter allows for additional filtering of results
                based on metadata.
            - If `return_all` is set to `True`, all fields stored in Redis will be
                returned, which may include non-indexed fields.
        """
        embedding = self._embeddings.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter,
            sort_by,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of `Document` objects to return.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree
                of diversity among the results with `0` corresponding
                to maximum diversity and `1` to minimum diversity.
            **kwargs: Other keyword arguments to pass to the search function.

        Returns:
            List of `Document` objects selected by maximal marginal relevance.
        """
        # Fetch top fetch_k documents based on similarity to the embedding
        docs_scores_embeddings = self.similarity_search_with_score_by_vector(
            embedding, k=fetch_k, with_vectors=True, **kwargs
        )

        # Extract documents and embeddings
        documents = []
        embeddings = []
        for item in docs_scores_embeddings:
            if len(item) == 3:
                doc, _, emb = item
                documents.append(doc)
                embeddings.append(emb)
            elif len(item) == 2:
                doc, _ = item
                documents.append(doc)

        # Perform MMR on the embeddings
        if embeddings:
            mmr_selected = maximal_marginal_relevance(
                np.array(embedding),
                embeddings,
                k=min(k, len(documents)),
                lambda_mult=lambda_mult,
            )

            # Return the selected documents based on MMR
            return [documents[i] for i in mmr_selected]
        else:
            return []

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of `Document` objects to return.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree
                of diversity among the results with `0` corresponding
                to maximum diversity and `1` to minimum diversity.
            **kwargs: Other keyword arguments to pass to the search function.

        Returns:
            List of `Document` objects selected by maximal marginal relevance.
        """
        query_embedding = self.embeddings.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            query_embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    def get_by_ids(self, ids: Sequence[str]) -> List[Document]:
        """Get documents by their IDs.

        The returned documents are expected to have the ID field set to the ID of the
        document in the vector store.

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs. Records owned by another index that shares the
        same key prefix are also omitted when the schema defines a recognized
        `_index_name` marker.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of ids to retrieve.

        Returns:
            List of `Document` objects.

        !!! version-added "Added in `langchain-redis` 0.1.2"
        """
        full_ids = self._redis_keys(ids)
        values = self._fetch_records_by_keys(full_ids)
        expected_marker = self._index_name_marker_for_read()
        documents = []
        for id_, value in zip(ids, values):
            if value is None or not value:
                continue
            doc = value
            if not self._record_belongs_to_index(doc, expected_marker):
                continue
            # Process metadata the same way we do in _build_document_from_result
            metadata = {}
            if "_metadata_json" in doc:
                try:
                    # Try to parse the JSON metadata
                    metadata = json.loads(doc["_metadata_json"])
                except (json.JSONDecodeError, TypeError):
                    # Fall back to extracting metadata fields from result directly
                    metadata = {
                        k: v
                        for k, v in doc.items()
                        if k != self.config.content_field
                        and k != self.config.embedding_field
                        and k != "_index_name"
                        and not k.startswith("_")
                    }
            else:
                # Fall back to extracting metadata fields from doc directly
                metadata = {
                    k: v
                    for k, v in doc.items()
                    if k != self.config.content_field
                    and k != self.config.embedding_field
                    and k != "_index_name"
                    and not k.startswith("_")
                }

            documents.append(
                Document(
                    id=id_,
                    page_content=doc[self.config.content_field],
                    metadata=metadata,
                )
            )
        return documents
