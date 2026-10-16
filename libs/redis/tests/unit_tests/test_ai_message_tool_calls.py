"""Tests for AIMessage.tool_calls being lost on Redis round-trip.

Mirrors the structure of test_tool_message_issue_51.py, which covers the
sibling case of ToolMessage.tool_call_id / status not surviving a round-trip.
Here, an AIMessage's tool_calls (and invalid_tool_calls) were not being
stored at all, so any agent conversation persisted to Redis and then
reloaded would silently lose which tools the model had asked to call.
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_redis import RedisChatMessageHistory


class TestAIMessageToolCalls:
    """Test that AIMessage.tool_calls survive storage and retrieval."""

    @patch("langchain_redis.chat_message_history.SearchIndex")
    def test_add_ai_message_with_tool_calls_stores_tool_calls(
        self, mock_search_index: MagicMock
    ) -> None:
        """Adding an AIMessage with tool_calls must store them."""
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_redis_client,
        )

        tool_calls = [
            {
                "name": "get_weather",
                "args": {"city": "San Francisco"},
                "id": "call_123",
                "type": "tool_call",
            }
        ]
        ai_message = AIMessage(content="", tool_calls=tool_calls)
        history.add_message(ai_message)

        call_args = mock_index_instance.load.call_args
        data = call_args[1]["data"][0]

        assert "tool_calls" in data["data"], "tool_calls must be stored"
        assert data["data"]["tool_calls"] == tool_calls
        assert "invalid_tool_calls" in data["data"]
        assert data["data"]["invalid_tool_calls"] == []

    @patch("langchain_redis.chat_message_history.SearchIndex")
    def test_round_trip_ai_message_preserves_tool_calls(
        self, mock_search_index: MagicMock
    ) -> None:
        """Full round-trip: add an AIMessage with tool_calls, then retrieve it.

        Before this fix, tool_calls was reconstructed as an empty list,
        even though the original AIMessage had tool_calls set.
        """
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        stored_data = []

        def capture_load(**kwargs: Any) -> None:
            stored_data.append(kwargs["data"][0])

        mock_index_instance.load.side_effect = capture_load

        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_redis_client,
        )

        tool_calls = [
            {
                "name": "search",
                "args": {"query": "python tutorials"},
                "id": "call_456",
                "type": "tool_call",
            }
        ]
        ai_message = AIMessage(content="Let me search for that", tool_calls=tool_calls)
        history.add_message(ai_message)

        assert len(stored_data) == 1
        stored = stored_data[0]

        mock_index_instance.query.return_value = [
            {"type": stored["type"], "$.data": json.dumps(stored["data"])}
        ]

        messages = history.messages

        assert len(messages) == 1
        assert isinstance(messages[0], AIMessage)
        assert messages[0].content == "Let me search for that"
        assert messages[0].tool_calls == tool_calls

    @patch("langchain_redis.chat_message_history.SearchIndex")
    def test_ai_message_without_tool_calls_round_trips_with_empty_list(
        self, mock_search_index: MagicMock
    ) -> None:
        """A plain AIMessage (no tool calls) should round-trip with tool_calls=[]."""
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
        mock_index_instance = MagicMock()
        mock_search_index.from_dict.return_value = mock_index_instance

        stored_data = []

        def capture_load(**kwargs: Any) -> None:
            stored_data.append(kwargs["data"][0])

        mock_index_instance.load.side_effect = capture_load

        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_client=mock_redis_client,
        )

        history.add_message(AIMessage(content="Hi there"))

        stored = stored_data[0]
        mock_index_instance.query.return_value = [
            {"type": stored["type"], "$.data": json.dumps(stored["data"])}
        ]

        messages = history.messages
        assert isinstance(messages[0], AIMessage)
        assert messages[0].tool_calls == []

    @patch("langchain_redis.chat_message_history.SearchIndex")
    def test_full_tool_calling_conversation_round_trip(
        self, mock_search_index: MagicMock
    ) -> None:
        """End-to-end: Human -> AI (with tool_calls) -> Tool result round-trips."""
        mock_redis_client = MagicMock()
        mock_redis_client.client_setinfo = MagicMock()
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

        tool_calls = [
            {
                "name": "get_weather",
                "args": {"city": "SF"},
                "id": "call_789",
                "type": "tool_call",
            }
        ]
        history.add_message(HumanMessage(content="What's the weather in SF?"))
        history.add_message(AIMessage(content="", tool_calls=tool_calls))
        history.add_message(
            ToolMessage(content="Sunny, 72F", tool_call_id="call_789", status="success")
        )

        mock_index_instance.query.return_value = [
            {"type": msg["type"], "$.data": json.dumps(msg["data"])}
            for msg in stored_messages
        ]

        messages = history.messages

        assert len(messages) == 3
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert messages[1].tool_calls == tool_calls
        assert isinstance(messages[2], ToolMessage)
        assert messages[2].tool_call_id == "call_789"
