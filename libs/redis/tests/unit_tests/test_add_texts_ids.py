"""Test that ids parameter in kwargs works correctly in add_texts."""

from unittest.mock import MagicMock, patch

from langchain_redis import RedisVectorStore


def test_add_texts_with_ids_in_kwargs() -> None:
    """Test that ids parameter in kwargs is used when keys is None."""
    # Create the complete patch setup
    with patch("langchain_redis.vectorstores.RedisConfig") as mock_config, patch(
        "langchain_redis.vectorstores.SearchIndex"
    ) as mock_search_index_class:
        # Setup mock embeddings
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        # Mock SearchIndex instance
        mock_index = MagicMock()
        mock_index.load.return_value = ["key1:id1", "key1:id2"]
        mock_index.schema.fields.values.return_value = []
        # Make the SearchIndex constructor return our mock
        mock_search_index_class.return_value = mock_index
        # Also mock the from_dict method
        mock_search_index_class.from_dict.return_value = mock_index

        # Setup config
        mock_config.return_value.index_name = "test_index"
        mock_config.return_value.key_prefix = "key1"
        mock_config.return_value.embedding_dimensions = 3
        mock_config.return_value.content_field = "text"
        mock_config.return_value.embedding_field = "embedding"
        mock_config.return_value.redis.return_value = MagicMock()
        mock_config.return_value.index_schema = None
        mock_config.return_value.schema_path = None
        mock_config.return_value.from_existing = False
        mock_config.return_value.storage_type = "JSON"
        mock_config.return_value.metadata_schema = None
        mock_config.return_value.vector_datatype = "FLOAT32"
        mock_config.return_value.default_tag_separator = ","
        mock_config.return_value.distance_metric = "COSINE"
        mock_config.return_value.indexing_algorithm = "FLAT"

        # Create vector store
        vector_store = RedisVectorStore(embeddings=mock_embeddings)

        # Test add_texts with ids in kwargs
        texts = ["text1", "text2"]
        ids = ["id1", "id2"]
        result = vector_store.add_texts(texts=texts, ids=ids)

        # Verify that index.load was called with the expected keys
        expected_keys = ["key1:id1", "key1:id2"]
        mock_index.load.assert_called_once()
        args, kwargs = mock_index.load.call_args
        assert kwargs["keys"] == expected_keys
        assert len(result) == 2


def test_add_texts_with_both_keys_and_ids() -> None:
    """Test that keys parameter takes precedence over ids in kwargs."""
    # Create the complete patch setup
    with patch("langchain_redis.vectorstores.RedisConfig") as mock_config, patch(
        "langchain_redis.vectorstores.SearchIndex"
    ) as mock_search_index_class:
        # Setup mock embeddings
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        # Mock SearchIndex instance
        mock_index = MagicMock()
        mock_index.load.return_value = ["key1:key1", "key1:key2"]
        mock_index.schema.fields.values.return_value = []
        # Make the SearchIndex constructor return our mock
        mock_search_index_class.return_value = mock_index
        # Also mock the from_dict method
        mock_search_index_class.from_dict.return_value = mock_index

        # Setup config
        mock_config.return_value.index_name = "test_index"
        mock_config.return_value.key_prefix = "key1"
        mock_config.return_value.embedding_dimensions = 3
        mock_config.return_value.content_field = "text"
        mock_config.return_value.embedding_field = "embedding"
        mock_config.return_value.redis.return_value = MagicMock()
        mock_config.return_value.index_schema = None
        mock_config.return_value.schema_path = None
        mock_config.return_value.from_existing = False
        mock_config.return_value.storage_type = "JSON"
        mock_config.return_value.metadata_schema = None
        mock_config.return_value.vector_datatype = "FLOAT32"
        mock_config.return_value.default_tag_separator = ","
        mock_config.return_value.distance_metric = "COSINE"
        mock_config.return_value.indexing_algorithm = "FLAT"

        # Create vector store
        vector_store = RedisVectorStore(embeddings=mock_embeddings)

        # Test add_texts with both keys and ids
        texts = ["text1", "text2"]
        keys = ["key1", "key2"]
        ids = ["id1", "id2"]
        result = vector_store.add_texts(texts=texts, keys=keys, ids=ids)

        # Verify that index.load was called with keys (not ids)
        expected_keys = ["key1:key1", "key1:key2"]
        mock_index.load.assert_called_once()
        args, kwargs = mock_index.load.call_args
        assert kwargs["keys"] == expected_keys
        assert len(result) == 2
