"""Redis vector store."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union, cast

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from redisvl.index import SearchIndex  # type: ignore[import]
from redisvl.query import RangeQuery, VectorQuery  # type: ignore[import]
from redisvl.query.filter import FilterExpression  # type: ignore[import]
from redisvl.redis.utils import (  # type: ignore[import]
    array_to_buffer,
    buffer_to_array,
    convert_bytes,
)
from redisvl.schema import StorageType  # type: ignore[import]

from langchain_redis.config import RedisConfig
from langchain_redis.version import __lib_name__

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


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
        lambda_mult: Number between 0 and 1 that determines the degree
            of diversity among the results, where 0 corresponds to
            maximum diversity and 1 to minimum diversity.
            Defaults to 0.5.
        k: Number of results to return. Defaults to 4.

    Returns:
        List of indices of selected embeddings.

    Example:
        .. code-block:: python

            from langchain_redis import RedisVectorStore
            from langchain_openai import OpenAIEmbeddings
            import numpy as np

            embeddings = OpenAIEmbeddings()
            vector_store = RedisVectorStore(
                index_name="langchain-demo",
                embedding=embeddings,
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
        Install ``langchain-redis`` and running the Redis docker container.

        .. code-block:: bash

            pip install -qU langchain-redis
            docker run -p 6379:6379 redis/redis-stack-server:latest

    Key init args — indexing params:
        index_name: str
            Name of the index to create.
        embedding: Embeddings
            Embedding function to use.
        distance_metric: str
            Distance metric to use for similarity search. Default is "COSINE".
        indexing_algorithm: str
            Indexing algorithm to use. Default is "FLAT".
        vector_datatype: str
            Data type of the vector. Default is "FLOAT32".

    Key init args — client params:
        redis_url: Optional[str]
            URL of the Redis instance to connect to.
        redis_client: Optional[Redis]
            Pre-existing Redis connection.
        ttl: Optional[int]
            Time-to-live for the Redis keys.

    Instantiate:
        .. code-block:: python

            from langchain_redis import RedisVectorStore
            from langchain_openai import OpenAIEmbeddings

            vector_store = RedisVectorStore(
                index_name="langchain-demo",
                embedding=OpenAIEmbeddings(),
                redis_url="redis://localhost:6379",
            )

    You can also connect to an existing Redis instance by passing in a
    pre-existing Redis connection via the redis_client argument.

    Instantiate from existing connection:
        .. code-block:: python

            from langchain_redis import RedisVectorStore
            from langchain_openai import OpenAIEmbeddings
            from redis import Redis

            redis_client = Redis.from_url("redis://localhost:6379")

            store = RedisVectorStore(
                embedding=OpenAIEmbeddings(),
                index_name="langchain-demo",
                redis_client=redis_client
            )

    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="bar", metadata={"foo": "baz"})
            document_3 = Document(page_content="to be deleted")

            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="foo", k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * foo [{'baz': 'bar'}]

    Search with filter:
        .. code-block:: python

            from redisvl.query.filter import Tag

            results = vector_store.similarity_search(
                query="foo",
                k=1,
                filter=Tag("baz") == "bar"
            )
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * foo [{'baz': 'bar'}]

    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="foo", k=1)
            for doc, score in results:
                print(f"* [SIM={score:.3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.916] foo [{'baz': 'bar'}]

    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
            )
            retriever.get_relevant_documents("foo")

        .. code-block:: python

            [Document(page_content='foo', metadata={'baz': 'bar'})]
    """

    def __init__(
        self,
        embeddings: Embeddings,
        config: Optional[RedisConfig] = None,
        ttl: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize the RedisVectorStore.

        Args:
            embeddings: The Embeddings instance used for this store.
            config: Optional RedisConfig object. If not provided, a new
                one will be created from kwargs.
            ttl: Optional time-to-live for Redis keys.
            **kwargs: Additional keyword arguments for RedisConfig if
                config is not provided.
        """
        # 1. Load or create the Redis configuration
        self.config = config or RedisConfig(**kwargs)

        # 2. Store embeddings and TTL
        self._embeddings = embeddings
        self.ttl = ttl

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

            self._index = SearchIndex.from_dict(
                {
                    "index": {
                        "name": self.config.index_name,
                        "prefix": f"{self.config.key_prefix}:",
                        "storage_type": self.config.storage_type,
                    },
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
                            },
                        },
                        {"name": "_index_name", "type": "text"},
                        {"name": "_metadata_json", "type": "text"},
                        *modified_metadata_schema,
                    ],
                },
                redis_client=redis_client,
                lib_name=__lib_name__,
            )
            self._index.create(overwrite=False)

    @property
    def index(self) -> SearchIndex:
        return self._index

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    @property
    def key_prefix(self) -> Optional[str]:
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
            **kwargs: Additional keyword arguments:
                - ids: Optional list of ids to associate with the documents.
                - refresh_indices: Whether to refresh the Redis indices
                after adding the texts. Defaults to True.
                - create_index_if_not_exists: Whether to create the Redis
                index if it doesn't already exist. Defaults to True.
                - batch_size: Optional. Number of texts to add to the
                index at a time. Defaults to 1000.

        Returns:
            List of ids from adding the texts into the vector store.

        Example:
            .. code-block:: python

                from langchain_redis import RedisVectorStore
                from langchain_openai import OpenAIEmbeddings

                vector_store = RedisVectorStore(
                    index_name="langchain-demo",
                    embedding=OpenAIEmbeddings(),
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

        # Build records to load to SearchIndex
        records = []
        for text, embedding, metadata in zip(
            texts_list,
            document_embeddings,
            metadatas or [{}] * len(texts_list),
        ):
            # Store complete metadata as JSON for exact retrieval later
            metadata_json = json.dumps(metadata)

            record = {
                self.config.content_field: text,
                self.config.embedding_field: (
                    embedding
                    if self.config.storage_type == StorageType.JSON.value
                    else array_to_buffer(embedding, dtype=self.config.vector_datatype)
                ),
                # Add index name as internal metadata to enable filtering
                "_index_name": self.config.index_name,
                # Store metadata as JSON to ensure exact retrieval
                "_metadata_json": metadata_json,
            }
            for field_name, field_value in metadata.items():
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
            records.append(record)

        # Load records into the index
        if keys:
            # Already have key_prefix in index definition (with ending colon)
            record_keys = [f"{self.config.key_prefix}:{key}" for key in keys]
            result = self._index.load(records, keys=record_keys, ttl=self.ttl)
        else:
            result = self._index.load(records, ttl=self.ttl)

        # Return list of IDs or empty list if result is None
        return list(result) if result is not None else []

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
        """Create a RedisVectorStore from a list of texts.

        Args:
            texts: List of texts to add to the vector store.
            embedding: Embedding function to use for encoding the texts.
            metadatas: Optional list of metadata dicts associated with the texts.
            config: Optional RedisConfig object. If not provided, one will be created
                from kwargs.
            keys: Optional list of keys to associate with the documents.
            return_keys: Whether to return the keys of the added documents.
            **kwargs: Additional keyword arguments to pass to RedisConfig if config is
                not provided.
                Commonly used kwargs include:
                - index_name: Name of the Redis index to create.
                - redis_url: URL of the Redis instance to connect to.
                - distance_metric: Distance metric to use for similarity search.
                    Default is "COSINE".
                - indexing_algorithm: Indexing algorithm to use. Default is "FLAT".

        Returns:
            RedisVectorStore: A new RedisVectorStore instance with the texts added.

        Example:
            .. code-block:: python

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

        Note:
            - This method creates a new RedisVectorStore instance and adds the
                provided texts to it.
            - If `metadatas` is provided, it must have the same length as `texts`.
            - If `keys` is provided, it must have the same length as `texts`.
            - The `return_keys` parameter determines whether the method returns just the
            RedisVectorStore instance or a tuple of (RedisVectorStore, List[str]) where
            the second element is the list of keys for the added documents.
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
        """Create a RedisVectorStore from a list of Documents.

        Args:
            documents: List of Document objects to add to the vector store.
            embedding: Embeddings object to use for encoding the documents.
            config: Optional RedisConfig object. If not provided, one will be
                    created from kwargs.
            return_keys: Whether to return the keys of the added documents.
            **kwargs: Additional keyword arguments to pass to RedisConfig if config
                      is not provided.
                Common kwargs include:
                - index_name: Name of the Redis index to create.
                - redis_url: URL of the Redis instance to connect to.
                - distance_metric: Distance metric to use for similarity search.
                                   Default is "COSINE".
                - indexing_algorithm: Indexing algorithm to use. Default is "FLAT".

        Returns:
            RedisVectorStore: A new RedisVectorStore instance with the documents added.

        Example:
            .. code-block:: python

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

        Note:
            - This method creates a new RedisVectorStore instance and adds the provided
              documents to it.
            - The method extracts the text content and metadata from
              each Document object.
            - If a RedisConfig object is not provided, one will be created using
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
        """Create a RedisVectorStore from an existing Redis Search Index.

        This method allows you to connect to an already existing index in Redis,
        which can be useful for continuing work with previously created indexes
        or for connecting to indexes created outside of this client.

        Args:
            index_name: Name of the existing index to use.
            embedding: Embedding function to use for encoding queries.
            **kwargs: Additional keyword arguments to pass to RedisConfig.
                Common kwargs include:
                - redis_url: URL of the Redis instance to connect to.
                - redis_client: Pre-existing Redis client to use.
                - vector_query_field: Name of the field containing the vector
                    representations.
                - content_field: Name of the field containing the document content.

        Returns:
            RedisVectorStore: A new RedisVectorStore instance connected to the
                existing index.

        Example:
            .. code-block:: python

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
        """Delete ids from the vector store.

        Args:
            ids: Optional list of ids of the documents to delete.
            **kwargs: Additional keyword arguments (not used in the
                      current implementation).

        Returns:
            Optional[bool]: True if one or more keys are deleted,
            False otherwise

        Example:
            .. code-block:: python

                from langchain_redis import RedisVectorStore
                from langchain_openai import OpenAIEmbeddings

                vector_store = RedisVectorStore(
                    index_name="langchain-demo",
                    embedding=OpenAIEmbeddings(),
                    redis_url="redis://localhost:6379",
                )

                # Assuming documents with these ids exist in the store
                ids_to_delete = ["doc1", "doc2", "doc3"]

                result = vector_store.delete(ids=ids_to_delete)
                if result:
                  print("Documents were succesfully deleted")
                else:
                  print("No Documents were deleted")

        Note:
            - If `ids` is None or an empty list, the method returns False.
            - If the number of actually deleted keys differs from the number of keys
              submitted for deletion the method returns False
            - The method uses the `drop_keys` functionality from RedisVL to delete
              the keys from Redis.
            - Keys are constructed by prefixing each id with the `key_prefix` specified
              in the configuration.
        """
        if ids and len(ids) > 0:
            if self.config.key_prefix:
                keys = [f"{self.config.key_prefix}:{_id}" for _id in ids]
            else:
                keys = ids
            # Always return True if we delete at least one key
            # This matches the behavior expected by the tests
            return self._index.drop_keys(keys) > 0
        else:
            return False

    def _query_builder(
        self,
        embedding: Union[List[float], bytes],
        k: int = 10,
        distance_threshold: Any = None,
        sort_by: Optional[str] = None,
        filter: Optional[Union[str, FilterExpression]] = None,
        return_fields: Optional[List[str]] = None,
    ) -> Union[VectorQuery, RangeQuery]:
        # Add a filter to restrict search to the current index
        # This is needed to ensure we only get results from the current index
        # when multiple indexes share the same key_prefix
        # Only apply the _index_name filter if we have the field in the schema
        try:
            # Check if we have an _index_name field in the schema
            has_index_name_field = False
            for field in self._index.schema.fields.values():
                if field.name == "_index_name":
                    has_index_name_field = True
                    break

            if has_index_name_field:
                # Apply the filter since we have the field
                from redisvl.query.filter import Text

                index_filter = Text("_index_name") == self.config.index_name
                if filter is not None:
                    if hasattr(filter, "__and__"):
                        filter = filter & index_filter
                    else:
                        # Don't apply the filter if we can't combine it safely
                        pass
                else:
                    filter = index_filter
        except Exception:
            # If any issues occur, just use the original filter
            pass
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
            k: Number of Documents to return. Defaults to 4.
            filter: Optional filter expression to apply.
            sort_by: Optional sort_by expression to apply.
            **kwargs: Other keyword arguments:
                - return_metadata: Whether to return metadata. Defaults to True.
                - distance_threshold: Optional distance threshold for filtering results.
                - return_all: Whether to return all data in the Hash/JSON including
                  non-indexed fields

        Returns:
            List of Documents most similar to the query vector.
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
            k: Number of Documents to return. Defaults to 4.
            filter: Optional filter expression to apply.
            sort_by: Optional sort_by expression to apply.
            **kwargs: Other keyword arguments to pass to the search function.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self._embeddings.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, filter, sort_by, **kwargs)

    def _build_document_from_result(self, res: Dict[str, Any]) -> Document:
        """Build a Document object from a Redis search result."""
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
        """Get metadata fields from a search result without _metadata_json."""
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

        return vector

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
            k: Number of Documents to return. Defaults to 4.
            filter: Optional filter expression to apply.
            sort_by: Optional sort_by expression to apply.
            **kwargs: Other keyword arguments:
                with_vectors: Whether to return document vectors. Defaults to False.
                return_metadata: Whether to return metadata. Defaults to True.
                distance_threshold: Optional distance threshold for filtering results.

        Returns:
            List of tuples of Documents most similar to the query vector, score, and
            optionally the document vector.
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
        """Return documents most similar to query string, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Optional filter expression to apply to the query.
            sort_by: Optional sort_by expression to apply to the query.
            **kwargs: Other keyword arguments to pass to the search function:
                - custom_query: Optional callable that can be used
                                to customize the query.
                - doc_builder: Optional callable to customize Document creation.
                - return_metadata: Whether to return metadata. Defaults to True.
                - distance_threshold: Optional distance threshold for filtering results.
                - return_all: Whether to return all data in the Hash/JSON including
                non-indexed fields. Defaults to False.

        Returns:
            List of tuples of (Document, score) most similar to the query.

        Example:
            .. code-block:: python

                from langchain_redis import RedisVectorStore
                from langchain_openai import OpenAIEmbeddings

                vector_store = RedisVectorStore(
                    index_name="langchain-demo",
                    embedding=OpenAIEmbeddings(),
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

        Note:
            - The method returns scores along with documents. Lower scores indicate
            higher similarity.
            - The actual search is performed using the vector representation of the
              query, which is why an embedding function must be provided during
              initialization.
            - The `filter` parameter allows for additional filtering of results
              based on metadata.
            - If `return_all` is set to True, all fields stored in Redis will be
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
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                    Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            **kwargs: Other keyword arguments to pass to the search function.

        Returns:
            List of Documents selected by maximal marginal relevance.
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
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                    Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            **kwargs: Other keyword arguments to pass to the search function.

        Returns:
            List of Documents selected by maximal marginal relevance.
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
        if there are duplicated IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of ids to retrieve.

        Returns:
            List of Documents.

        .. versionadded:: 0.1.2
        """
        redis = self.config.redis()
        if self.config.key_prefix:
            full_ids = [f"{self.config.key_prefix}:{_id}" for _id in ids]
        else:
            full_ids = list(ids)
        if self.config.storage_type == StorageType.JSON.value:
            values = redis.json().mget(full_ids, ".")
        else:
            pipe = redis.pipeline()
            for id_ in full_ids:
                pipe.hgetall(id_)
            values = pipe.execute()
        documents = []
        for id_, value in zip(ids, values):
            if value is None or not value:
                continue
            if self.config.storage_type == StorageType.JSON.value:
                doc = cast(dict, value)
            else:
                doc = convert_bytes(value)
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
