"""Tests for RedisSemanticCache prefix parameter (Issue #76)."""

from unittest.mock import MagicMock, patch

from langchain_redis import RedisSemanticCache


class TestRedisSemanticCachePrefix:
    """Test that prefix parameter works correctly in RedisSemanticCache."""

    @patch("langchain_redis.cache.EmbeddingsVectorizer")
    @patch("langchain_redis.cache.RedisVLSemanticCache")
    @patch("langchain_redis.cache.Redis")
    def test_prefix_parameter_fix_issue_76(
        self,
        mock_redis_class: MagicMock,
        mock_semantic_cache_class: MagicMock,
        mock_vectorizer_class: MagicMock,
    ) -> None:
        """Verify fix for issue #76: prefix parameter now works correctly.

        When both name and prefix are provided, they should be combined to create
        the cache name, allowing for both cache type naming and tenant isolation.
        """
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_redis_class.from_url.return_value = mock_redis_client
        mock_cache_instance = MagicMock()
        mock_semantic_cache_class.return_value = mock_cache_instance

        # Create a mock embeddings
        mock_embeddings = MagicMock()

        # User expects: when prefix="user_1", keys should include the prefix
        _ = RedisSemanticCache(
            embeddings=mock_embeddings,
            redis_url="redis://localhost:6379",
            name="langgraph_agent",
            prefix="user_1",  # This parameter should be used in key naming
            ttl=30,
        )

        # Check what was actually passed to RedisVLSemanticCache
        call_kwargs = mock_semantic_cache_class.call_args[1]

        # The fix: name should combine both parameters
        # Keys will be 'langgraph_agent:user_1:...'
        assert call_kwargs["name"] == "langgraph_agent:user_1"
        assert "prefix" not in call_kwargs  # prefix is no longer passed separately

    @patch("langchain_redis.cache.Redis")
    def test_actual_redis_vl_behavior(self, mock_redis_class: MagicMock) -> None:
        """Test actual RedisVL SemanticCache behavior with name vs prefix.

        This test shows that RedisVL's SemanticCache uses 'name' as the prefix
        and doesn't accept a separate 'prefix' parameter.
        """
        from redisvl.extensions.cache.llm import (  # type: ignore[import-untyped]
            SemanticCache,
        )

        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_redis_class.from_url.return_value = mock_redis_client

        # Create a RedisVL SemanticCache with a name
        cache = SemanticCache(name="test_name", redis_client=mock_redis_client)

        # The index should use 'test_name' as the prefix
        assert cache._index.schema.index.prefix == "test_name"
        assert cache._index.schema.index.name == "test_name"

    def test_expected_behavior_for_prefix(self) -> None:
        """Document the expected behavior for the prefix parameter.

        When a user provides:
        - name="langgraph_agent"
        - prefix="user_1"

        They expect keys to be stored as: "user_1:..." or "langgraph_agent:user_1:..."
        But currently, keys are stored as: "langgraph_agent:..."

        The prefix parameter should allow for multi-tenant isolation or
        user-specific namespacing within the same named cache.
        """
        # This is a documentation test - no assertions
        pass

    @patch("langchain_redis.cache.EmbeddingsVectorizer")
    @patch("langchain_redis.cache.RedisVLSemanticCache")
    @patch("langchain_redis.cache.Redis")
    def test_prefix_defaults_to_name_when_custom_name_provided(
        self,
        mock_redis_class: MagicMock,
        mock_semantic_cache_class: MagicMock,
        mock_vectorizer_class: MagicMock,
    ) -> None:
        """Test that when only custom name is provided, it's used (defaults ignored)."""
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_redis_class.from_url.return_value = mock_redis_client
        mock_cache_instance = MagicMock()
        mock_semantic_cache_class.return_value = mock_cache_instance

        mock_embeddings = MagicMock()

        _ = RedisSemanticCache(
            embeddings=mock_embeddings,
            redis_url="redis://localhost:6379",
            name="my_cache",
            # prefix not specified (defaults to "llmcache" but will be ignored)
            ttl=30,
        )

        call_kwargs = mock_semantic_cache_class.call_args[1]

        # name should be used as-is since prefix defaults to "llmcache"
        assert call_kwargs["name"] == "my_cache"
        assert "prefix" not in call_kwargs

    @patch("langchain_redis.cache.EmbeddingsVectorizer")
    @patch("langchain_redis.cache.RedisVLSemanticCache")
    @patch("langchain_redis.cache.Redis")
    def test_both_name_and_prefix_provided(
        self,
        mock_redis_class: MagicMock,
        mock_semantic_cache_class: MagicMock,
        mock_vectorizer_class: MagicMock,
    ) -> None:
        """Test behavior when both name and prefix are explicitly provided.

        This is the problematic case from issue #76.
        """
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_redis_class.from_url.return_value = mock_redis_client
        mock_cache_instance = MagicMock()
        mock_semantic_cache_class.return_value = mock_cache_instance

        mock_embeddings = MagicMock()

        _ = RedisSemanticCache(
            embeddings=mock_embeddings,
            redis_url="redis://localhost:6379",
            name="cache_name",
            prefix="custom_prefix",
            ttl=30,
        )

        call_kwargs = mock_semantic_cache_class.call_args[1]

        # With the fix: name should be combined as "cache_name:custom_prefix"
        assert call_kwargs["name"] == "cache_name:custom_prefix"
        # prefix should no longer be passed to RedisVL
        assert "prefix" not in call_kwargs

    @patch("langchain_redis.cache.EmbeddingsVectorizer")
    @patch("langchain_redis.cache.RedisVLSemanticCache")
    @patch("langchain_redis.cache.Redis")
    def test_only_prefix_provided_fix(
        self,
        mock_redis_class: MagicMock,
        mock_semantic_cache_class: MagicMock,
        mock_vectorizer_class: MagicMock,
    ) -> None:
        """Test that providing only prefix uses it as the cache name.

        This is the main fix for issue #76.
        """
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_redis_class.from_url.return_value = mock_redis_client
        mock_cache_instance = MagicMock()
        mock_semantic_cache_class.return_value = mock_cache_instance

        mock_embeddings = MagicMock()

        # User provides only prefix (name will default to "llmcache")
        _ = RedisSemanticCache(
            embeddings=mock_embeddings,
            redis_url="redis://localhost:6379",
            prefix="user_1",  # This should be used as the name
            ttl=30,
        )

        call_kwargs = mock_semantic_cache_class.call_args[1]

        # The fix: prefix should be used as the name
        assert call_kwargs["name"] == "user_1"
        assert "prefix" not in call_kwargs

    @patch("langchain_redis.cache.EmbeddingsVectorizer")
    @patch("langchain_redis.cache.RedisVLSemanticCache")
    @patch("langchain_redis.cache.Redis")
    def test_only_name_provided_backward_compat(
        self,
        mock_redis_class: MagicMock,
        mock_semantic_cache_class: MagicMock,
        mock_vectorizer_class: MagicMock,
    ) -> None:
        """Test backward compatibility when only name is provided."""
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_redis_class.from_url.return_value = mock_redis_client
        mock_cache_instance = MagicMock()
        mock_semantic_cache_class.return_value = mock_cache_instance

        mock_embeddings = MagicMock()

        # User provides only name (prefix will default to "llmcache")
        _ = RedisSemanticCache(
            embeddings=mock_embeddings,
            redis_url="redis://localhost:6379",
            name="my_custom_cache",
            ttl=30,
        )

        call_kwargs = mock_semantic_cache_class.call_args[1]

        # Backward compatibility: name should be used as-is
        assert call_kwargs["name"] == "my_custom_cache"
        assert "prefix" not in call_kwargs

    @patch("langchain_redis.cache.EmbeddingsVectorizer")
    @patch("langchain_redis.cache.RedisVLSemanticCache")
    @patch("langchain_redis.cache.Redis")
    def test_defaults_unchanged(
        self,
        mock_redis_class: MagicMock,
        mock_semantic_cache_class: MagicMock,
        mock_vectorizer_class: MagicMock,
    ) -> None:
        """Test that default behavior is unchanged."""
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_redis_class.from_url.return_value = mock_redis_client
        mock_cache_instance = MagicMock()
        mock_semantic_cache_class.return_value = mock_cache_instance

        mock_embeddings = MagicMock()

        # User doesn't provide name or prefix (both default to "llmcache")
        _ = RedisSemanticCache(
            embeddings=mock_embeddings,
            redis_url="redis://localhost:6379",
            ttl=30,
        )

        call_kwargs = mock_semantic_cache_class.call_args[1]

        # Default behavior: use default name
        assert call_kwargs["name"] == "llmcache"
        assert "prefix" not in call_kwargs
