"""Tests for Redis Sentinel support."""

from unittest.mock import MagicMock, patch

from langchain_redis import RedisCache, RedisChatMessageHistory, RedisConfig


class TestRedisSentinelURLDetection:
    """Test Sentinel URL detection in RedisConfig."""

    def test_is_sentinel_url_true(self) -> None:
        """Test that Sentinel URLs are correctly detected."""
        config = RedisConfig(
            redis_url="redis+sentinel://sentinel1:26379,sentinel2:26379/mymaster"
        )
        assert config.is_sentinel_url() is True

    def test_is_sentinel_url_false_redis(self) -> None:
        """Test that standard Redis URLs are not detected as Sentinel."""
        config = RedisConfig(redis_url="redis://localhost:6379")
        assert config.is_sentinel_url() is False

    def test_is_sentinel_url_false_rediss(self) -> None:
        """Test that Redis SSL URLs are not detected as Sentinel."""
        config = RedisConfig(redis_url="rediss://localhost:6379")
        assert config.is_sentinel_url() is False


class TestRedisConfigSentinelConnection:
    """Test RedisConfig creates proper connections for Sentinel."""

    @patch("redisvl.redis.connection.RedisConnectionFactory.get_redis_connection")
    def test_redis_method_uses_connection_factory_for_sentinel(
        self, mock_get_connection: MagicMock
    ) -> None:
        """Test RedisConfig.redis() uses RedisVL connection factory."""
        mock_redis_client = MagicMock()
        mock_get_connection.return_value = mock_redis_client

        config = RedisConfig(
            redis_url="redis+sentinel://sentinel1:26379,sentinel2:26379/mymaster"
        )
        client = config.redis()

        # Verify that RedisConnectionFactory was called
        mock_get_connection.assert_called_once_with(
            redis_url="redis+sentinel://sentinel1:26379,sentinel2:26379/mymaster"
        )
        assert client == mock_redis_client

    @patch("langchain_redis.config.Redis")
    def test_redis_method_uses_from_url_for_standard_redis(
        self, mock_redis: MagicMock
    ) -> None:
        """Test that RedisConfig.redis() uses Redis.from_url for standard URLs."""
        mock_redis_client = MagicMock()
        mock_redis.from_url.return_value = mock_redis_client

        config = RedisConfig(redis_url="redis://localhost:6379")
        client = config.redis()

        # Verify that Redis.from_url was called
        mock_redis.from_url.assert_called_once_with("redis://localhost:6379")
        assert client == mock_redis_client


class TestRedisCacheSentinelConnection:
    """Test RedisCache Sentinel connection support."""

    @patch("redisvl.redis.connection.RedisConnectionFactory.get_redis_connection")
    def test_redis_cache_uses_connection_factory_for_sentinel(
        self, mock_get_connection: MagicMock
    ) -> None:
        """Test that RedisCache uses RedisVL connection factory for Sentinel."""
        mock_redis_client = MagicMock()
        # Mock the client_setinfo method to avoid errors
        mock_redis_client.client_setinfo = MagicMock()
        mock_get_connection.return_value = mock_redis_client

        cache = RedisCache(
            redis_url="redis+sentinel://sentinel1:26379/mymaster", ttl=3600
        )

        # Verify that RedisConnectionFactory was called
        mock_get_connection.assert_called_once_with(
            redis_url="redis+sentinel://sentinel1:26379/mymaster"
        )
        assert cache.redis == mock_redis_client

    @patch("langchain_redis.cache.Redis")
    def test_redis_cache_uses_from_url_for_standard_redis(
        self, mock_redis: MagicMock
    ) -> None:
        """Test that RedisCache uses Redis.from_url for standard URLs."""
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_redis.from_url.return_value = mock_redis_client

        cache = RedisCache(redis_url="redis://localhost:6379", ttl=3600)

        # Verify that Redis.from_url was called
        mock_redis.from_url.assert_called_once_with("redis://localhost:6379")
        assert cache.redis == mock_redis_client


class TestRedisChatMessageHistorySentinelConnection:
    """Test RedisChatMessageHistory Sentinel connection support."""

    @patch("langchain_redis.chat_message_history.SearchIndex")
    @patch("redisvl.redis.connection.RedisConnectionFactory.get_redis_connection")
    def test_chat_history_uses_connection_factory_for_sentinel(
        self, mock_get_connection: MagicMock, mock_search_index: MagicMock
    ) -> None:
        """Test that RedisChatMessageHistory uses connection factory for Sentinel."""
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_redis_client.ft = MagicMock()
        mock_get_connection.return_value = mock_redis_client
        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_url="redis+sentinel://sentinel1:26379/mymaster",
        )

        # Verify that RedisConnectionFactory was called
        mock_get_connection.assert_called_once_with(
            redis_url="redis+sentinel://sentinel1:26379/mymaster"
        )
        assert history.redis_client == mock_redis_client

    @patch("langchain_redis.chat_message_history.SearchIndex")
    @patch("langchain_redis.chat_message_history.Redis")
    def test_chat_history_uses_from_url_for_standard_redis(
        self, mock_redis: MagicMock, mock_search_index: MagicMock
    ) -> None:
        """Test that RedisChatMessageHistory uses Redis.from_url for standard URLs."""
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_redis_client.ft = MagicMock()
        mock_redis.from_url.return_value = mock_redis_client
        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_url="redis://localhost:6379",
        )

        # Verify that Redis.from_url was called
        mock_redis.from_url.assert_called_once_with("redis://localhost:6379")
        assert history.redis_client == mock_redis_client
