from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from redisvl.redis.utils import hashify  # type: ignore[import]
from redisvl.schema.fields import VectorDistanceMetric  # type: ignore[import]

from langchain_redis import RedisConfig, RedisVectorStore


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]


class MockField:
    def __init__(self, name: str, field_type: str, attrs: Optional[Any] = None) -> None:
        self.name = name
        self.type = field_type
        self.attrs = attrs or {}


class MockSchema:
    def __init__(self, fields: Dict[str, Dict[str, Any]]) -> None:
        self.fields = {
            name: MockField(name, details["type"], details.get("attrs"))
            for name, details in fields.items()
        }

    def values(self) -> Iterable[MockField]:
        return self.fields.values()


class MockStorage:
    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Any]] = {}

    def get(self, client: Any, doc_ids: List[str]) -> List[Dict[str, Any]]:
        return [self.data.get(doc_id, {}) for doc_id in doc_ids]


class MockSearchIndex:
    last_instance: Optional["MockSearchIndex"] = None

    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        redis_client: Optional[Any] = None,
        lib_name: Optional[str] = None,
    ) -> None:
        self.data: List[Dict[str, Any]] = []
        default_schema = {
            "text": {"type": "text"},
            "embedding": {
                "type": "vector",
                "attrs": {"dims": 3, "distance_metric": "cosine"},
            },
            "metadata": {"type": "text"},
            "_index_name": {"type": "tag"},
        }
        self.schema = MockSchema(
            schema["fields"] if schema and "fields" in schema else default_schema  # type: ignore
        )
        self.redis_client = redis_client or Mock()
        self.client = self.redis_client
        self.name = "test_index"
        self._storage = MockStorage()
        self.batch_size: Optional[int] = None
        type(self).last_instance = self

    def create(self, overwrite: bool = False) -> None:
        pass

    def load(
        self,
        documents: List[Dict[str, Any]],
        keys: Optional[List[str]] = None,
        ttl: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        self.batch_size = batch_size
        for i, doc in enumerate(documents):
            key = keys[i] if keys else f"key_{i}"
            self._storage.data[key] = doc
        self.data.extend(documents)
        return keys or [f"key_{i}" for i in range(len(documents))]

    def query(self, query: Any) -> List[Dict[str, Any]]:
        k = query._num_results if hasattr(query, "_num_results") else len(self.data)

        mock_response = [
            {
                "id": f"key_{i}",
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
                "vector_distance": 0.1,
            }
            for i, doc in enumerate(self.data[:k])
        ]

        if "embedding" in query._return_fields:
            if query._return_fields_decode_as["embedding"] is None:
                vec = b"\xcd\xccL>\xcd\xccL>\xcd\xccL>"  # type: ignore
            else:
                vec = [0.2, 0.2, 0.2]  # type: ignore

            mock_response = [{**m, "embedding": vec} for m in mock_response]

        return mock_response

    def drop_keys(self, keys: Union[str, List[str]]) -> int:
        self.data = [doc for i, doc in enumerate(self.data) if f"key_{i}" not in keys]
        for key in keys:
            self._storage.data.pop(key, None)
        return len(keys)

    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> "MockSearchIndex":
        return cls(schema=dict_data)

    @classmethod
    def from_existing(cls, name: str, **kwargs: Any) -> "MockSearchIndex":
        assert cls.last_instance is not None
        return cls.last_instance

    def key(self, id: str) -> str:
        return f"key:{id}"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MockSearchIndex":
        return cls()


class TestRedisVectorStore:
    @pytest.fixture
    def mock_embeddings(self) -> MockEmbeddings:
        return MockEmbeddings()

    @pytest.fixture
    def mock_config(self) -> RedisConfig:
        config = RedisConfig(
            index_name="test_index",
            redis_url="redis://localhost:6379",
            schema_path="mock_schema.yaml",
        )
        config.content_field = "text"
        config.embedding_field = "embedding"
        config.index_schema = {
            "fields": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "vector",
                    "attrs": {"dims": 3, "distance_metric": "cosine"},
                },
                "metadata": {"type": "text"},
                "_index_name": {"type": "tag"},
            }
        }
        return config

    @pytest.fixture
    def vector_store(
        self, mock_embeddings: MockEmbeddings, mock_config: RedisConfig
    ) -> RedisVectorStore:
        with patch("langchain_redis.vectorstores.SearchIndex", MockSearchIndex):
            return RedisVectorStore(embeddings=mock_embeddings, config=mock_config)

    @pytest.mark.parametrize(
        ("batch_size", "keys"),
        [
            pytest.param(None, None, id="default-batch-size"),
            pytest.param(1, None, id="minimum-batch-size"),
            pytest.param(2, ["doc1", "doc2"], id="explicit-keys"),
        ],
    )
    def test_add_texts(
        self,
        vector_store: RedisVectorStore,
        batch_size: Optional[int],
        keys: Optional[List[str]],
    ) -> None:
        texts = ["Hello, world!", "Test document"]
        metadatas = [{"source": "greeting"}, {"source": "test"}]
        with patch.object(vector_store, "_validate_write_ownership"):
            result = vector_store.add_texts(
                texts, metadatas, keys=keys, batch_size=batch_size
            )
        assert len(result) == 2
        if keys is None:
            assert all(key.startswith("key_") for key in result)
        else:
            assert result == keys
        index = MockSearchIndex.last_instance
        assert index is not None
        assert index.batch_size == batch_size

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_add_texts_rejects_invalid_batch_size(
        self, vector_store: RedisVectorStore, batch_size: int
    ) -> None:
        with patch.object(
            vector_store.embeddings, "embed_documents"
        ) as embed_documents:
            with pytest.raises(ValueError):
                vector_store.add_texts(["document"], batch_size=batch_size)

        embed_documents.assert_not_called()

    def test_add_texts_uses_each_tag_field_separator(
        self, vector_store: RedisVectorStore
    ) -> None:
        index = MockSearchIndex.last_instance
        assert index is not None
        index.schema.fields.update(
            {
                "comma_tags": MockField(
                    "comma_tags", "tag", SimpleNamespace(separator=",")
                ),
                "pipe_tags": MockField(
                    "pipe_tags", "tag", SimpleNamespace(separator="|")
                ),
            }
        )

        vector_store.add_texts(
            ["document"],
            metadatas=[
                {
                    "comma_tags": ["one", "two"],
                    "pipe_tags": ["one", "two"],
                }
            ],
        )

        assert index.data[-1]["comma_tags"] == "one,two"
        assert index.data[-1]["pipe_tags"] == "one|two"

    def test_similarity_search(self, vector_store: RedisVectorStore) -> None:
        vector_store.add_texts(["Hello, world!", "Test document"])
        results = vector_store.similarity_search("Hello", k=1)
        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert results[0].page_content in ["Hello, world!", "Test document"]

    def test_similarity_search_with_score(self, vector_store: RedisVectorStore) -> None:
        vector_store.add_texts(["Hello, world!", "Test document"])
        results = vector_store.similarity_search_with_score("Hello", k=1)
        assert len(results) == 1
        assert isinstance(results[0][0], Document)
        assert isinstance(results[0][1], float)

    def test_similarity_search_with_score_return_all_json_no_matches(
        self, vector_store: RedisVectorStore
    ) -> None:
        """A search that matched nothing sends no record fetch to Redis."""
        vector_store.config.storage_type = "json"
        client = Mock()

        with patch.object(vector_store._index, "client", client):
            results = vector_store.similarity_search_with_score(
                "Hello", return_all=True
            )

        assert results == []
        client.pipeline.assert_not_called()
        client.json().mget.assert_not_called()

    @pytest.mark.parametrize("storage_type", ["hash", "json"])
    @pytest.mark.parametrize(
        "search",
        [
            pytest.param(
                lambda store: store.similarity_search("Hello", k=2, return_all=True),
                id="similarity_search",
            ),
            pytest.param(
                lambda store: [
                    doc
                    for doc, _ in store.similarity_search_with_score(
                        "Hello", k=2, return_all=True
                    )
                ],
                id="similarity_search_with_score",
            ),
        ],
    )
    def test_return_all_skips_a_hit_whose_record_is_gone(
        self,
        vector_store: RedisVectorStore,
        storage_type: str,
        search: Callable[[RedisVectorStore], List[Document]],
    ) -> None:
        """A hit deleted between FT.SEARCH and the record fetch is skipped.

        `_fetch_records_by_keys` reports a missing record as `None` (an
        `HGETALL` of a deleted key returns `{}`), and both searches skip it
        instead of raising `KeyError` on the content field.
        """
        vector_store.config.storage_type = storage_type
        vector_store.add_texts(["Hello, world!", "Test document"])
        surviving = {"text": "Test document", "embedding": [0.2, 0.2, 0.2]}

        with patch.object(
            vector_store, "_fetch_records_by_keys", return_value=[None, surviving]
        ) as fetch_records:
            docs = search(vector_store)

        fetch_records.assert_called_once_with(["key_0", "key_1"])
        assert [doc.page_content for doc in docs] == ["Test document"]

    @staticmethod
    def _use_metric(metric: VectorDistanceMetric, field: str = "embedding") -> None:
        index = MockSearchIndex.last_instance
        assert index is not None
        index.schema.fields[field] = MockField(
            field, "vector", SimpleNamespace(distance_metric=metric)
        )

    @pytest.mark.parametrize(
        "metric, distance, expected",
        [
            # RedisVL normalizes COSINE as (2 - d) / 2 and L2 as 1 / (1 + d).
            (VectorDistanceMetric.COSINE, 0.0, 1.0),
            (VectorDistanceMetric.COSINE, 0.4, 0.8),
            (VectorDistanceMetric.COSINE, 2.0, 0.0),
            (VectorDistanceMetric.L2, 0.0, 1.0),
            (VectorDistanceMetric.L2, 0.8, 1 / 1.8),
            (VectorDistanceMetric.L2, 3.0, 0.25),
        ],
    )
    def test_relevance_score_fn_uses_redisvl_normalization(
        self,
        vector_store: RedisVectorStore,
        metric: VectorDistanceMetric,
        distance: float,
        expected: float,
    ) -> None:
        self._use_metric(metric)

        score_fn = vector_store._select_relevance_score_fn()

        assert score_fn(distance) == pytest.approx(expected)

    def test_relevance_score_fn_rejects_ip(
        self, vector_store: RedisVectorStore
    ) -> None:
        self._use_metric(VectorDistanceMetric.IP)

        with pytest.raises(ValueError, match="IP.*COSINE"):
            vector_store._select_relevance_score_fn()

    def test_relevance_score_fn_uses_live_index_metric(
        self, vector_store: RedisVectorStore
    ) -> None:
        """A reopened index only has the default metric in its config."""
        vector_store.config.distance_metric = "COSINE"

        self._use_metric(VectorDistanceMetric.L2)
        assert vector_store._select_relevance_score_fn()(0.8) == pytest.approx(1 / 1.8)

        self._use_metric(VectorDistanceMetric.IP)
        with pytest.raises(ValueError, match="IP"):
            vector_store._select_relevance_score_fn()

    @pytest.mark.parametrize(
        "default_field_metric, custom_field_metric",
        [
            (VectorDistanceMetric.COSINE, VectorDistanceMetric.L2),
            (VectorDistanceMetric.L2, VectorDistanceMetric.IP),
        ],
    )
    def test_relevance_score_fn_uses_configured_embedding_field(
        self,
        vector_store: RedisVectorStore,
        default_field_metric: VectorDistanceMetric,
        custom_field_metric: VectorDistanceMetric,
    ) -> None:
        """The metric comes from `config.embedding_field`, not from "embedding"."""
        vector_store.config.embedding_field = "custom_vector"
        vector_store.config.distance_metric = "COSINE"
        # The default field and the config both point at a different metric, so
        # a hardcoded "embedding" lookup or a read of the config would be wrong.
        self._use_metric(default_field_metric)
        self._use_metric(custom_field_metric, field="custom_vector")

        if custom_field_metric == VectorDistanceMetric.IP:
            with pytest.raises(ValueError, match="IP"):
                vector_store._select_relevance_score_fn()
        else:
            score_fn = vector_store._select_relevance_score_fn()
            # 1 / (1 + d) for L2. COSINE would give (2 - d) / 2 = 0.6.
            assert score_fn(0.8) == pytest.approx(1 / 1.8)

    def test_similarity_search_with_relevance_scores(
        self, vector_store: RedisVectorStore
    ) -> None:
        self._use_metric(VectorDistanceMetric.COSINE)
        vector_store.add_texts(["Hello, world!", "Test document"])

        results = vector_store.similarity_search_with_relevance_scores("Hello", k=2)
        assert [score for _, score in results] == pytest.approx([0.95, 0.95])

        # similarity_search_with_score keeps returning the raw distance.
        raw = vector_store.similarity_search_with_score("Hello", k=2)
        assert [distance for _, distance in raw] == pytest.approx([0.1, 0.1])

    @pytest.mark.parametrize(
        "score_threshold, expected_hits",
        [
            # The mock reports a raw distance of 0.1, which scores exactly 0.95.
            # score_threshold is inclusive, so a hit on the boundary is kept.
            (0.95, 2),
            (0.96, 0),
        ],
    )
    def test_score_threshold_is_inclusive(
        self,
        vector_store: RedisVectorStore,
        score_threshold: float,
        expected_hits: int,
    ) -> None:
        self._use_metric(VectorDistanceMetric.COSINE)
        vector_store.add_texts(["Hello, world!", "Test document"])

        results = vector_store.similarity_search_with_relevance_scores(
            "Hello", k=2, score_threshold=score_threshold
        )

        assert len(results) == expected_hits

    async def test_asimilarity_search_with_relevance_scores(
        self, vector_store: RedisVectorStore
    ) -> None:
        self._use_metric(VectorDistanceMetric.COSINE)
        vector_store.add_texts(["Hello, world!", "Test document"])

        results = await vector_store.asimilarity_search_with_relevance_scores(
            "Hello", k=2
        )

        assert [score for _, score in results] == pytest.approx([0.95, 0.95])

    async def test_asimilarity_search_with_relevance_scores_rejects_ip(
        self, vector_store: RedisVectorStore
    ) -> None:
        self._use_metric(VectorDistanceMetric.IP)
        vector_store.add_texts(["Hello, world!"])

        with pytest.raises(ValueError, match="IP"):
            await vector_store.asimilarity_search_with_relevance_scores("Hello", k=1)

    def test_max_marginal_relevance_search(
        self, vector_store: RedisVectorStore
    ) -> None:
        vector_store.add_texts(["Hello, world!", "Test document", "Another test"])
        results = vector_store.max_marginal_relevance_search("Hello", k=2, fetch_k=3)
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)

    def test_delete(self, vector_store: RedisVectorStore) -> None:
        keys = vector_store.add_texts(["Hello, world!", "Test document"])
        with patch.object(
            vector_store,
            "_fetch_records_by_keys",
            return_value=[
                {"text": "Hello, world!", "_index_name": hashify("test_index")},
                {"text": "Test document", "_index_name": hashify("test_index")},
            ],
        ):
            result = vector_store.delete(keys)
        assert result is True

    @patch("langchain_redis.vectorstores.RedisVectorStore.add_texts")
    def test_from_texts(
        self,
        mock_add_texts: Mock,
        mock_embeddings: MockEmbeddings,
        mock_config: RedisConfig,
    ) -> None:
        with patch("langchain_redis.vectorstores.SearchIndex", MockSearchIndex):
            texts = ["Hello, world!", "Test document"]
            metadatas = [{"source": "greeting"}, {"source": "test"}]
            RedisVectorStore.from_texts(
                texts,
                mock_embeddings,
                metadatas=metadatas,
                config=mock_config,
                batch_size=2,
            )
            mock_add_texts.assert_called_once_with(texts, metadatas, None, batch_size=2)

    @patch("langchain_redis.vectorstores.RedisVectorStore.add_texts")
    def test_from_documents(
        self,
        mock_add_texts: Mock,
        mock_embeddings: MockEmbeddings,
        mock_config: RedisConfig,
    ) -> None:
        with patch("langchain_redis.vectorstores.SearchIndex", MockSearchIndex):
            docs = [
                Document(page_content="Hello, world!", metadata={"source": "greeting"}),
                Document(page_content="Test document", metadata={"source": "test"}),
            ]
            RedisVectorStore.from_documents(
                docs, mock_embeddings, config=mock_config, batch_size=2
            )
            mock_add_texts.assert_called_once_with(
                ["Hello, world!", "Test document"],
                [{"source": "greeting"}, {"source": "test"}],
                None,
                batch_size=2,
            )
