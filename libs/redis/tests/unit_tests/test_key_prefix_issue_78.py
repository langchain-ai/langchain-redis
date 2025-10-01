"""Tests for RedisVectorStore key prefix issue (#78)."""

from unittest.mock import MagicMock, patch

from langchain_redis import RedisConfig, RedisVectorStore


class TestKeyPrefixIssue78:
    """Test key prefix handling in RedisVectorStore."""

    @patch("langchain_redis.vectorstores.SearchIndex")
    def test_double_colon_in_generated_keys(self, mock_search_index: MagicMock) -> None:
        """Reproduce issue #78: double colon in keys due to added trailing colon.

        When SearchIndex.from_dict is called with prefix="myprefix:",
        RedisVL will generate keys like "myprefix::doc_id" (double colon)
        because it adds its own key_separator.
        """
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 768

        config = RedisConfig(
            index_name="test_index",
            key_prefix="test_prefix",
            embedding_dimensions=768,
        )

        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        _ = RedisVectorStore(embeddings=mock_embeddings, config=config)

        # Verify SearchIndex.from_dict was called
        assert mock_search_index.from_dict.called

        # Get the schema dict that was passed
        call_args = mock_search_index.from_dict.call_args
        schema_dict = call_args[0][0]  # First positional argument

        # The issue: prefix has trailing colon added
        assert schema_dict["index"]["prefix"] == "test_prefix:"

        # This will cause RedisVL to generate keys like:
        # "test_prefix::doc_id" (double colon)
        # because RedisVL adds key_separator ":" between prefix and document ID

    @patch("langchain_redis.vectorstores.SearchIndex")
    def test_key_prefix_with_default_index_name(
        self, mock_search_index: MagicMock
    ) -> None:
        """Test that key_prefix defaults to index_name and gets trailing colon."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 768

        config = RedisConfig(
            index_name="my_index",
            # key_prefix not specified, will default to index_name
            embedding_dimensions=768,
        )

        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        _ = RedisVectorStore(embeddings=mock_embeddings, config=config)

        call_args = mock_search_index.from_dict.call_args
        schema_dict = call_args[0][0]

        # key_prefix defaults to index_name, then gets ":" added
        assert schema_dict["index"]["prefix"] == "my_index:"
        # This creates keys like "my_index::doc_id"

    @patch("langchain_redis.vectorstores.SearchIndex")
    def test_prefix_already_has_colon(self, mock_search_index: MagicMock) -> None:
        """Test what happens if user provides prefix with trailing colon."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 768

        # User explicitly sets key_prefix with trailing colon
        config = RedisConfig(
            index_name="test_index",
            key_prefix="my_prefix:",  # Already has colon
            embedding_dimensions=768,
        )

        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        _ = RedisVectorStore(embeddings=mock_embeddings, config=config)

        call_args = mock_search_index.from_dict.call_args
        schema_dict = call_args[0][0]

        # Another colon is added, creating double colon
        assert schema_dict["index"]["prefix"] == "my_prefix::"
        # This creates keys like "my_prefix:::doc_id" (triple colon!)

    @patch("langchain_redis.vectorstores.SearchIndex")
    def test_using_provided_index_schema_no_double_colon(
        self, mock_search_index: MagicMock
    ) -> None:
        """Test that using index_schema directly avoids the double colon issue.

        When using index_schema, the from_dict path is not taken,
        so no trailing colon is added.
        """
        from redisvl.schema import IndexSchema  # type: ignore[import-untyped]

        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 768

        # User provides their own schema without trailing colon
        schema = IndexSchema.from_dict(
            {
                "index": {
                    "name": "test_index",
                    "prefix": "test_prefix",  # No trailing colon
                    "storage_type": "hash",
                },
                "fields": [
                    {"name": "text", "type": "text"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "dims": 768,
                            "distance_metric": "cosine",
                            "algorithm": "flat",
                            "datatype": "float32",
                        },
                    },
                ],
            }
        )

        config = RedisConfig(schema=schema, embedding_dimensions=768)

        mock_index_instance = MagicMock()
        mock_search_index.return_value = mock_index_instance

        _ = RedisVectorStore(embeddings=mock_embeddings, config=config)

        # Verify SearchIndex was called with the schema directly
        mock_search_index.assert_called_once()
        call_kwargs = mock_search_index.call_args[1]
        assert call_kwargs["schema"] == schema

        # The schema's prefix doesn't have trailing colon added
        assert schema.index.prefix == "test_prefix"
        # This will generate keys like "test_prefix:doc_id" (correct!)

    def test_actual_key_format_with_redisvl(self) -> None:
        """Document the actual key format generated by RedisVL."""
        from redisvl.schema import IndexSchema

        # Test with prefix WITH trailing colon (current buggy behavior)
        schema_with_colon = IndexSchema.from_dict(
            {
                "index": {
                    "name": "test",
                    "prefix": "myprefix:",  # Trailing colon
                    "storage_type": "hash",
                },
                "fields": [{"name": "text", "type": "text"}],
            }
        )

        # Test with prefix WITHOUT trailing colon (correct)
        schema_without_colon = IndexSchema.from_dict(
            {
                "index": {
                    "name": "test",
                    "prefix": "myprefix",  # No trailing colon
                    "storage_type": "hash",
                },
                "fields": [{"name": "text", "type": "text"}],
            }
        )

        # RedisVL generates keys as: prefix + key_separator + document_id
        doc_id = "abc123"

        # With trailing colon in prefix
        key_with_colon = (
            f"{schema_with_colon.index.prefix}"
            f"{schema_with_colon.index.key_separator}"
            f"{doc_id}"
        )
        assert key_with_colon == "myprefix::abc123"  # Double colon!

        # Without trailing colon in prefix
        key_without_colon = (
            f"{schema_without_colon.index.prefix}"
            f"{schema_without_colon.index.key_separator}"
            f"{doc_id}"
        )
        assert key_without_colon == "myprefix:abc123"  # Correct!

    @patch("langchain_redis.vectorstores.SearchIndex")
    def test_fix_with_legacy_format_true(self, mock_search_index: MagicMock) -> None:
        """Test that legacy_key_format=True maintains backward compatibility."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 768

        config = RedisConfig(
            index_name="test_index",
            key_prefix="test_prefix",
            embedding_dimensions=768,
            legacy_key_format=True,  # Maintain old behavior (default)
        )

        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        _ = RedisVectorStore(embeddings=mock_embeddings, config=config)

        call_args = mock_search_index.from_dict.call_args
        schema_dict = call_args[0][0]

        # With legacy format, trailing colon is still added
        assert schema_dict["index"]["prefix"] == "test_prefix:"
        # Keys will be "test_prefix::doc_id" (backward compatible)

    @patch("langchain_redis.vectorstores.SearchIndex")
    def test_fix_with_legacy_format_false(self, mock_search_index: MagicMock) -> None:
        """Test that legacy_key_format=False fixes the double colon issue."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 768

        config = RedisConfig(
            index_name="test_index",
            key_prefix="test_prefix",
            embedding_dimensions=768,
            legacy_key_format=False,  # Use correct format
        )

        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        _ = RedisVectorStore(embeddings=mock_embeddings, config=config)

        call_args = mock_search_index.from_dict.call_args
        schema_dict = call_args[0][0]

        # With legacy_key_format=False, no trailing colon added
        assert schema_dict["index"]["prefix"] == "test_prefix"
        # Keys will be "test_prefix:doc_id" (correct format!)

    @patch("langchain_redis.vectorstores.SearchIndex")
    def test_default_behavior_unchanged(self, mock_search_index: MagicMock) -> None:
        """Test that default behavior (no legacy_key_format specified) is unchanged."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 768

        config = RedisConfig(
            index_name="test_index",
            key_prefix="test_prefix",
            embedding_dimensions=768,
            # legacy_key_format not specified, defaults to True
        )

        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        _ = RedisVectorStore(embeddings=mock_embeddings, config=config)

        call_args = mock_search_index.from_dict.call_args
        schema_dict = call_args[0][0]

        # Default behavior: trailing colon is added (backward compatible)
        assert schema_dict["index"]["prefix"] == "test_prefix:"
