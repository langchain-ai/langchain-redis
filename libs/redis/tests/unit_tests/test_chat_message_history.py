from typing import Any
from unittest.mock import patch

import pytest
from langchain_core._api import LangChainDeprecationWarning

from langchain_redis import RedisChatMessageHistory


def test_chat_message_history_deprecated() -> None:
    with (
        patch("langchain_redis.chat_message_history.SearchIndex"),
        patch("redis.Redis.from_url"),
        pytest.warns(
            LangChainDeprecationWarning,
            match=r"RedisChatMessageHistory.*0\.2\.6.*1\.0\.0.*short-term-memory",
        ),
    ):
        history = RedisChatMessageHistory(session_id="test_session")
    assert history.id == "test_session"
    assert "https://docs.langchain.com/oss/python/langchain/short-term-memory" in (
        RedisChatMessageHistory.__doc__ or ""
    )


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


class TestRedisChatMessageHistoryExistingIndex:
    """Creating an instance when the search index already exists.

    Under RESP2 (protocol 2) FT.INFO returns the index definition as a
    flat list, while under RESP3 (redis-py 8, protocol 3) it returns a
    dict. The constructor must read the key prefixes from both shapes
    instead of crashing on the second instance (issue #222).
    """

    def _history_with_index_info(self, index_info):  # type: ignore[no-untyped-def]
        """Build a history whose mocked index returns the given info."""
        with (
            patch(
                "langchain_redis.chat_message_history.SearchIndex"
            ) as mock_search_index,
            patch("redis.Redis.from_url"),
        ):
            mock_search_index.from_dict.return_value.info.return_value = index_info
            return RedisChatMessageHistory(session_id="test_session")

    @pytest.mark.parametrize(
        "index_definition, should_warn",
        [
            (
                {"key_type": "JSON", "prefixes": ["chat:"]},
                False,
            ),
            (
                {"key_type": "JSON", "prefixes": ["other:"]},
                True,
            ),
            (
                ["key_type", "JSON", "prefixes", ["chat:"]],
                False,
            ),
            (
                ["key_type", "JSON", "prefixes", ["other:"]],
                True,
            ),
        ],
        ids=[
            "resp3-dict-matching-prefix",
            "resp3-dict-different-prefix",
            "resp2-list-matching-prefix",
            "resp2-list-different-prefix",
        ],
    )
    def test_index_definition_prefix_warning(
        self,
        index_definition: dict[str, Any] | list[Any],
        should_warn: bool,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The prefix-mismatch warning fires only when the prefix differs.

        Covers both FT.INFO response shapes (RESP3 dict and RESP2 list)
        and both prefix outcomes (matching and different).
        """
        with caplog.at_level("WARNING", logger="langchain_redis.chat_message_history"):
            history = self._history_with_index_info(
                {
                    "index_name": "idx:chat_history",
                    "index_definition": index_definition,
                }
            )
        if should_warn:
            assert "already exists with different key prefix" in caplog.text
        else:
            assert history.id == "test_session"
            history.index.create.assert_called_once()
            assert "already exists with different key prefix" not in caplog.text
