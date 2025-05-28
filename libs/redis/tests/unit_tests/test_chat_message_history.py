from unittest.mock import patch

import pytest

from langchain_redis import RedisChatMessageHistory


class TestRedisChatMessageHistoryMinimal:
    """Minimal unit tests focusing on input validation and utility methods."""

    def test_session_id_validation_empty_string(self) -> None:
        """Test that empty session_id raises ValueError."""
        with patch("langchain_redis.chat_message_history.SearchIndex"), patch(
            "redis.Redis.from_url"
        ):
            with pytest.raises(
                ValueError, match="session_id must be a non-empty, valid string"
            ):
                RedisChatMessageHistory(session_id="")

    def test_session_id_validation_none(self) -> None:
        """Test that None session_id raises ValueError."""
        with patch("langchain_redis.chat_message_history.SearchIndex"), patch(
            "redis.Redis.from_url"
        ):
            with pytest.raises(
                ValueError, match="session_id must be a non-empty, valid string"
            ):
                RedisChatMessageHistory(session_id=None)  # type: ignore

    def test_id_property_returns_session_id(self) -> None:
        """Test that id property returns session_id."""
        with patch("langchain_redis.chat_message_history.SearchIndex"), patch(
            "redis.Redis.from_url"
        ):
            history = RedisChatMessageHistory(session_id="test_session")
            assert history.id == "test_session"

    def test_message_key_generation_with_provided_id(self) -> None:
        """Test message key generation with provided message_id."""
        with patch("langchain_redis.chat_message_history.SearchIndex"), patch(
            "redis.Redis.from_url"
        ):
            history = RedisChatMessageHistory(session_id="test_session")
            key = history._message_key("msg123")
            assert key == "chat:test_session:msg123"

    def test_message_key_generation_with_custom_prefix(self) -> None:
        """Test message key generation with custom key_prefix."""
        with patch("langchain_redis.chat_message_history.SearchIndex"), patch(
            "redis.Redis.from_url"
        ):
            history = RedisChatMessageHistory(
                session_id="test_session", key_prefix="custom:"
            )
            key = history._message_key("msg123")
            assert key == "custom:test_session:msg123"

    def test_message_key_generation_auto_id(self) -> None:
        """Test message key generation with auto-generated message_id."""
        with patch("langchain_redis.chat_message_history.SearchIndex"), patch(
            "redis.Redis.from_url"
        ):
            history = RedisChatMessageHistory(session_id="test_session")
            key = history._message_key()

            # Should have format: prefix:session:ulid
            parts = key.split(":")
            assert len(parts) == 3
            assert parts[0] == "chat"
            assert parts[1] == "test_session"
            assert len(parts[2]) > 0  # ULID should be generated

    def test_search_messages_empty_query_returns_empty_list(self) -> None:
        """Test that empty search query returns empty list without Redis calls."""
        with patch("langchain_redis.chat_message_history.SearchIndex"), patch(
            "redis.Redis.from_url"
        ):
            history = RedisChatMessageHistory(session_id="test_session")

            # These should return empty list immediately
            assert history.search_messages("") == []
            assert history.search_messages(None) == []  # type: ignore

    def test_default_parameters(self) -> None:
        """Test that default parameters are set correctly."""
        with patch("langchain_redis.chat_message_history.SearchIndex"), patch(
            "redis.Redis.from_url"
        ):
            history = RedisChatMessageHistory(session_id="test_session")

            assert history.session_id == "test_session"
            assert history.key_prefix == "chat:"
            assert history.ttl is None
            assert history.index_name == "idx:chat_history"

    def test_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        with patch("langchain_redis.chat_message_history.SearchIndex"), patch(
            "redis.Redis.from_url"
        ):
            history = RedisChatMessageHistory(
                session_id="custom_session",
                key_prefix="custom:",
                ttl=7200,
                index_name="custom_index",
            )

            assert history.session_id == "custom_session"
            assert history.key_prefix == "custom:"
            assert history.ttl == 7200
            assert history.index_name == "custom_index"
