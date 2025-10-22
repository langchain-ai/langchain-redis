"""Tests for ToolMessage KeyError issue (#51)."""

from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_redis import RedisChatMessageHistory


class TestToolMessageIssue51:
    """Test ToolMessage serialization and deserialization."""

    @patch("langchain_redis.chat_message_history.SearchIndex")
    def test_add_tool_message_stores_tool_call_id(
        self, mock_search_index: MagicMock
    ) -> None:
        """Test that adding a ToolMessage stores tool_call_id.

        This is the fix for issue #51 where `ToolMessage.tool_call_id`
        was not being stored, causing KeyError on deserialization.
        """
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_redis_client.ft = MagicMock()
        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_redis_client,
        )

        # Add a ToolMessage with tool_call_id
        tool_message = ToolMessage(
            content="Tool result", tool_call_id="call_123", status="success"
        )
        history.add_message(tool_message)

        # Verify that index.load was called
        assert mock_index_instance.load.called

        # Get the data that was passed to index.load
        call_args = mock_index_instance.load.call_args
        data = call_args[1]["data"][0]

        # Verify tool_call_id is in the stored data
        assert "tool_call_id" in data["data"], "tool_call_id must be stored"
        assert data["data"]["tool_call_id"] == "call_123"
        assert "status" in data["data"], "status must be stored"
        assert data["data"]["status"] == "success"

    @patch("langchain_redis.chat_message_history.SearchIndex")
    def test_add_regular_messages_without_tool_call_id(
        self, mock_search_index: MagicMock
    ) -> None:
        """Test that regular messages don't have tool_call_id added."""
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_redis_client.ft = MagicMock()
        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_redis_client,
        )

        # Add various message types
        history.add_message(HumanMessage(content="Hello"))
        call_args = mock_index_instance.load.call_args
        data = call_args[1]["data"][0]
        assert "tool_call_id" not in data["data"]

        history.add_message(AIMessage(content="Hi there"))
        call_args = mock_index_instance.load.call_args
        data = call_args[1]["data"][0]
        assert "tool_call_id" not in data["data"]

        history.add_message(SystemMessage(content="System message"))
        call_args = mock_index_instance.load.call_args
        data = call_args[1]["data"][0]
        assert "tool_call_id" not in data["data"]

    @patch("langchain_redis.chat_message_history.SearchIndex")
    def test_retrieve_tool_message_without_key_error(
        self, mock_search_index: MagicMock
    ) -> None:
        """Test that retrieving ToolMessage doesn't raise KeyError.

        This reproduces the original issue #51 where messages_from_dict
        would fail with KeyError: 'tool_call_id' when retrieving ToolMessages.
        """
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_redis_client.ft = MagicMock()
        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        # Mock the query to return a ToolMessage
        mock_index_instance.query.return_value = [
            {
                "type": "tool",
                "$.data": (
                    '{"content": "Tool result", "additional_kwargs": {}, '
                    '"type": "tool", "tool_call_id": "call_123", '
                    '"status": "success"}'
                ),
            }
        ]

        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_redis_client,
        )

        # This should not raise KeyError
        messages = history.messages

        assert len(messages) == 1
        assert isinstance(messages[0], ToolMessage)
        assert messages[0].content == "Tool result"
        assert messages[0].tool_call_id == "call_123"
        assert messages[0].status == "success"

    @patch("langchain_redis.chat_message_history.SearchIndex")
    def test_round_trip_tool_message(self, mock_search_index: MagicMock) -> None:
        """Test complete round-trip: add ToolMessage and retrieve it.

        This simulates the real-world scenario from issue #51 where
        a ToolMessage is added in one session and retrieved in a follow-up.
        """
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_redis_client.ft = MagicMock()
        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        # Create stored data that will be captured
        stored_data = []

        def capture_load(**kwargs: Any) -> None:
            stored_data.append(kwargs["data"][0])

        mock_index_instance.load.side_effect = capture_load

        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_redis_client,
        )

        # Add a ToolMessage
        tool_msg = ToolMessage(
            content="Search results", tool_call_id="call_456", status="success"
        )
        history.add_message(tool_msg)

        # Verify data was stored
        assert len(stored_data) == 1
        stored = stored_data[0]

        # Now mock the query to return what was stored
        import json

        mock_index_instance.query.return_value = [
            {"type": stored["type"], "$.data": json.dumps(stored["data"])}
        ]

        # Retrieve messages - should not raise KeyError
        messages = history.messages

        assert len(messages) == 1
        assert isinstance(messages[0], ToolMessage)
        assert messages[0].content == "Search results"
        assert messages[0].tool_call_id == "call_456"
        assert messages[0].status == "success"

    @patch("langchain_redis.chat_message_history.SearchIndex")
    def test_mixed_message_types_with_tool_message(
        self, mock_search_index: MagicMock
    ) -> None:
        """Test conversation with mixed message types including `ToolMessage`."""
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_redis_client.ft = MagicMock()
        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        stored_messages = []

        def capture_load(**kwargs: Any) -> None:
            stored_messages.append(kwargs["data"][0])

        mock_index_instance.load.side_effect = capture_load

        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_redis_client,
        )

        # Add a conversation with tool calls
        history.add_message(HumanMessage(content="Search for Python tutorials"))
        history.add_message(AIMessage(content="I'll search for that"))
        history.add_message(
            ToolMessage(
                content="Found 10 tutorials", tool_call_id="call_789", status="success"
            )
        )
        history.add_message(AIMessage(content="Here are the tutorials I found..."))

        # Verify all messages were stored
        assert len(stored_messages) == 4

        # Verify the ToolMessage has tool_call_id
        tool_msg_data = stored_messages[2]
        assert tool_msg_data["type"] == "tool"
        assert "tool_call_id" in tool_msg_data["data"]
        assert tool_msg_data["data"]["tool_call_id"] == "call_789"
        assert tool_msg_data["data"]["status"] == "success"

        # Mock query to return all messages
        import json

        mock_index_instance.query.return_value = [
            {"type": msg["type"], "$.data": json.dumps(msg["data"])}
            for msg in stored_messages
        ]

        # Retrieve all messages - should not raise KeyError
        messages = history.messages

        assert len(messages) == 4
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert isinstance(messages[2], ToolMessage)
        assert isinstance(messages[3], AIMessage)
        assert messages[2].tool_call_id == "call_789"
