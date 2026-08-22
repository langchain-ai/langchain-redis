from unittest.mock import patch

import pytest

from langchain_redis import RedisChatMessageHistory


class TestRedisChatMessageHistoryMinimal:
    """Minimal unit tests focusing on input validation and utility methods."""

    def test_session_id_validation_empty_string(self) -> None:
        """Test that empty session_id raises ValueError."""
        with (
            patch("langchain_redis.chat_message_history.SearchIndex"),
            patch("redis.Redis.from_url"),
        ):
            with pytest.raises(
                ValueError, match="session_id must be a non-empty, valid string"
            ):
                RedisChatMessageHistory(session_id="")

    def test_session_id_validation_none(self) -> None:
        """Test that None session_id raises ValueError."""
        with (
            patch("langchain_redis.chat_message_history.SearchIndex"),
            patch("redis.Redis.from_url"),
        ):
            with pytest.raises(
                ValueError, match="session_id must be a non-empty, valid string"
            ):
                RedisChatMessageHistory(session_id=None)  # type: ignore

    def test_id_property_returns_session_id(self) -> None:
        """Test that id property returns session_id."""
        with (
            patch("langchain_redis.chat_message_history.SearchIndex"),
            patch("redis.Redis.from_url"),
        ):
            history = RedisChatMessageHistory(session_id="test_session")
            assert history.id == "test_session"

    def test_message_key_generation_with_provided_id(self) -> None:
        """Test message key generation with provided message_id."""
        with (
            patch("langchain_redis.chat_message_history.SearchIndex"),
            patch("redis.Redis.from_url"),
        ):
            history = RedisChatMessageHistory(session_id="test_session")
            key = history._message_key("msg123")
            assert key == "chat:test_session:msg123"

    def test_message_key_generation_with_custom_prefix(self) -> None:
        """Test message key generation with custom key_prefix."""
        with (
            patch("langchain_redis.chat_message_history.SearchIndex"),
            patch("redis.Redis.from_url"),
        ):
            history = RedisChatMessageHistory(
                session_id="test_session", key_prefix="custom:"
            )
            key = history._message_key("msg123")
            assert key == "custom:test_session:msg123"

    def test_message_key_generation_auto_id(self) -> None:
        """Test message key generation with auto-generated message_id."""
        with (
            patch("langchain_redis.chat_message_history.SearchIndex"),
            patch("redis.Redis.from_url"),
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
        with (
            patch("langchain_redis.chat_message_history.SearchIndex"),
            patch("redis.Redis.from_url"),
        ):
            history = RedisChatMessageHistory(session_id="test_session")

            # These should return empty list immediately
            assert history.search_messages("") == []
            assert history.search_messages(None) == []  # type: ignore

    def test_default_parameters(self) -> None:
        """Test that default parameters are set correctly."""
        with (
            patch("langchain_redis.chat_message_history.SearchIndex"),
            patch("redis.Redis.from_url"),
        ):
            history = RedisChatMessageHistory(session_id="test_session")

            assert history.session_id == "test_session"
            assert history.key_prefix == "chat:"
            assert history.ttl is None
            assert history.index_name == "idx:chat_history"

    def test_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        with (
            patch("langchain_redis.chat_message_history.SearchIndex"),
            patch("redis.Redis.from_url"),
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

    def test_add_messages_uses_single_index_load_call(self) -> None:
        """add_messages should batch all messages into one index.load() call."""
        from langchain_core.messages import AIMessage, HumanMessage

        with (
            patch("langchain_redis.chat_message_history.SearchIndex"),
            patch("redis.Redis.from_url"),
        ):
            history = RedisChatMessageHistory(session_id="test_session")

            messages = [
                HumanMessage(content="Hello, AI!"),
                AIMessage(content="Hello! How can I assist you today?"),
            ]
            history.add_messages(messages)

            history.index.load.assert_called_once()
            _, kwargs = history.index.load.call_args
            assert len(kwargs["data"]) == 2
            assert len(kwargs["keys"]) == 2
            assert kwargs["data"][0]["data"]["content"] == "Hello, AI!"
            assert (
                kwargs["data"][1]["data"]["content"]
                == "Hello! How can I assist you today?"
            )
            assert kwargs["ttl"] == history.ttl

    def test_add_messages_empty_list_does_not_call_index(self) -> None:
        """add_messages with an empty sequence should be a no-op."""
        with (
            patch("langchain_redis.chat_message_history.SearchIndex"),
            patch("redis.Redis.from_url"),
        ):
            history = RedisChatMessageHistory(session_id="test_session")
            history.add_messages([])
            history.index.load.assert_not_called()

    def test_add_messages_none_entry_raises(self) -> None:
        """add_messages should reject a None entry instead of silently failing."""
        from langchain_core.messages import HumanMessage

        with (
            patch("langchain_redis.chat_message_history.SearchIndex"),
            patch("redis.Redis.from_url"),
        ):
            history = RedisChatMessageHistory(session_id="test_session")
            with pytest.raises(ValueError, match="Message cannot be None"):
                history.add_messages([HumanMessage(content="hi"), None])  # type: ignore
