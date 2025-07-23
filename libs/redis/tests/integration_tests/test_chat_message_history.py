import asyncio
import time
from typing import Generator, List, Type, Union, cast

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from redis import Redis
from redis.commands.search.query import Query
from ulid import ULID

from langchain_redis import RedisChatMessageHistory


@pytest.fixture
def redis_client(redis_url: str) -> Redis:
    return Redis.from_url(redis_url)


@pytest.fixture
def chat_history(redis_url: str) -> Generator[RedisChatMessageHistory, None, None]:
    session_id = f"test_session_{str(ULID())}"
    history = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)
    history.clear()
    try:
        yield history
    finally:
        history.delete()


def test_add_and_retrieve_messages(chat_history: RedisChatMessageHistory) -> None:
    chat_history.add_message(HumanMessage(content="Hello, AI!"))
    chat_history.add_message(AIMessage(content="Hello, human!"))

    messages = chat_history.messages
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Hello, AI!"
    assert messages[1].content == "Hello, human!"


def test_clear_messages(chat_history: RedisChatMessageHistory) -> None:
    chat_history.add_message(HumanMessage(content="Test message"))
    assert len(chat_history.messages) == 1

    chat_history.clear()
    assert len(chat_history.messages) == 0


def test_add_multiple_messages(chat_history: RedisChatMessageHistory) -> None:
    messages = [
        HumanMessage(content="Message 1"),
        AIMessage(content="Response 1"),
        HumanMessage(content="Message 2"),
    ]
    for message in messages:
        chat_history.add_message(message)

    assert len(chat_history.messages) == 3
    assert [msg.content for msg in chat_history.messages] == [
        "Message 1",
        "Response 1",
        "Message 2",
    ]


def test_search_messages(chat_history: RedisChatMessageHistory) -> None:
    chat_history.add_message(HumanMessage(content="Hello, how are you?"))
    chat_history.add_message(AIMessage(content="I'm doing well, thank you!"))
    chat_history.add_message(HumanMessage(content="What's the weather like today?"))

    results = chat_history.search_messages("weather")

    assert len(results) == 1
    assert "weather" in results[0]["content"]

    # Test retrieving all messages
    all_messages = chat_history.messages
    assert len(all_messages) == 3
    assert all_messages[0].content == "Hello, how are you?"
    assert all_messages[1].content == "I'm doing well, thank you!"
    assert all_messages[2].content == "What's the weather like today?"


def test_length(chat_history: RedisChatMessageHistory) -> None:
    messages = [HumanMessage(content=f"Message {i}") for i in range(5)]
    for message in messages:
        chat_history.add_message(message)

    assert len(chat_history) == 5


def test_ttl(redis_url: str, redis_client: Redis) -> None:
    session_id = f"ttl_test_{str(ULID())}"
    chat_history = RedisChatMessageHistory(
        session_id=session_id, redis_url=redis_url, ttl=1
    )
    chat_history.add_message(HumanMessage(content="This message will expire"))

    # Check that the message was added
    assert len(chat_history.messages) == 1

    # Find the key for the added message
    query = Query(f"@session_id:{{{chat_history.id}}}")
    results = chat_history.redis_client.ft(chat_history.index_name).search(query)
    assert len(results.docs) == 1
    message_key = results.docs[0].id

    # Check TTL on the message key
    ttl_result = redis_client.ttl(message_key)
    if asyncio.iscoroutine(ttl_result):
        ttl = asyncio.get_event_loop().run_until_complete(ttl_result)
    else:
        ttl = ttl_result
    assert ttl > 0

    time.sleep(2)

    # Verify that the message has expired
    assert len(chat_history.messages) == 0

    # Verify that the key no longer exists
    assert redis_client.exists(message_key) == 0


def test_multiple_sessions(redis_url: str) -> None:
    session1 = f"ttl_test_{str(ULID())}"
    session2 = f"ttl_test_{str(ULID())}"
    history1 = RedisChatMessageHistory(session_id=session1, redis_url=redis_url)
    history2 = RedisChatMessageHistory(session_id=session2, redis_url=redis_url)

    history1.add_message(HumanMessage(content="Message for session 1"))
    history2.add_message(HumanMessage(content="Message for session 2"))

    assert len(history1.messages) == 1
    assert len(history2.messages) == 1
    assert history1.messages[0].content != history2.messages[0].content

    history1.clear()
    history2.clear()


def test_index_creation(redis_client: Redis, redis_url: str) -> None:
    session_id = f"index_test_{str(ULID())}"
    RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)
    index_info = redis_client.ft("idx:chat_history").info()
    assert index_info is not None
    assert index_info["index_name"] == "idx:chat_history"


@pytest.mark.parametrize("message_type", [HumanMessage, AIMessage, SystemMessage])
def test_different_message_types(
    chat_history: RedisChatMessageHistory,
    message_type: Type[Union[HumanMessage, AIMessage, SystemMessage]],
) -> None:
    message = message_type(content="Test content")
    chat_history.add_message(message)

    retrieved = chat_history.messages[-1]
    assert isinstance(retrieved, BaseMessage)
    assert isinstance(retrieved, message_type)
    assert retrieved.content == "Test content"

    # Use type casting to satisfy mypy
    typed_retrieved = cast(Union[HumanMessage, AIMessage, SystemMessage], retrieved)
    assert typed_retrieved.content == "Test content"


def test_large_number_of_messages(chat_history: RedisChatMessageHistory) -> None:
    large_number = 1000
    messages: List[BaseMessage] = [
        HumanMessage(content=f"Message {i}") for i in range(large_number)
    ]
    for message in messages:
        chat_history.add_message(message)

    retrieved_messages = chat_history.messages

    for i, message in enumerate(retrieved_messages):
        message_content = message.content
        expected_content = f"Message {i}"
        assert (
            message_content == expected_content
        ), f"Message at index {i} has content '{message_content}', \
            expected '{expected_content}'"

    assert (
        retrieved_messages[-1].content == f"Message {large_number - 1}"
    ), f"Last message content is '{retrieved_messages[-1].content}', \
        expected 'Message {large_number - 1}'"


def test_empty_messages(chat_history: RedisChatMessageHistory) -> None:
    assert len(chat_history.messages) == 0


def test_json_structure(
    redis_client: Redis, chat_history: RedisChatMessageHistory
) -> None:
    chat_history.add_message(HumanMessage(content="Test message"))

    # Find the key for the added message
    query = Query(f"@session_id:{{{chat_history.session_id}}}")
    results = chat_history.redis_client.ft(chat_history.index_name).search(query)
    assert len(results.docs) == 1, "Expected one message in the chat history"

    # Get the key of the first (and only) message
    message_key = results.docs[0].id

    # Retrieve the JSON data for this key
    json_data = redis_client.json().get(message_key)

    # Assert the structure of the JSON data
    assert "session_id" in json_data, "session_id should be present in the JSON data"
    assert "type" in json_data, "type should be present in the JSON data"
    assert "data" in json_data, "data should be present in the JSON data"
    assert "timestamp" in json_data, "timestamp should be present in the JSON data"

    # Check the content of the data field
    assert "content" in json_data["data"], "content should be present in the data field"
    assert (
        json_data["data"]["content"] == "Test message"
    ), "Content should match the added message"
    assert (
        json_data["data"]["type"] == "human"
    ), "Type should be 'human' for a HumanMessage"

    # Check the type at the root level
    assert (
        json_data["type"] == "human"
    ), "Type at root level should be 'human' for a HumanMessage"

    # Check the session_id
    assert (
        json_data["session_id"] == chat_history.session_id
    ), "session_id should match the chat history session_id"


def test_search_non_existent_message(chat_history: RedisChatMessageHistory) -> None:
    chat_history.add_message(HumanMessage(content="Hello, how are you?"))
    results = chat_history.search_messages("nonexistent")
    assert len(results) == 0


def test_add_message_to_existing_session(redis_url: str) -> None:
    session_id = f"existing_session_{str(ULID())}"
    history1 = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)
    history1.add_message(HumanMessage(content="First message"))

    history2 = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)
    history2.add_message(HumanMessage(content="Second message"))

    assert len(history1.messages) == 2
    assert len(history2.messages) == 2


def test_chat_history_with_preconfigured_client(redis_url: str) -> None:
    redis_client = Redis.from_url(redis_url)
    session_id = f"test_session_{str(ULID())}"
    history = RedisChatMessageHistory(session_id=session_id, redis_client=redis_client)

    history.add_message(HumanMessage(content="Hello, AI!"))
    history.add_message(AIMessage(content="Hello, human!"))

    messages = history.messages
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Hello, AI!"
    assert messages[1].content == "Hello, human!"

    history.clear()


def test_session_id_with_special_characters(redis_url: str) -> None:
    """Test that session IDs with special characters (like UUIDs with hyphens) work
    correctly."""
    # Use a UUID with hyphens - this would have caused syntax errors before the fix
    uuid_session_id = "550e8400-e29b-41d4-a716-446655440000"

    history = RedisChatMessageHistory(session_id=uuid_session_id, redis_url=redis_url)

    try:
        # Add messages - this should work without syntax errors
        history.add_message(HumanMessage(content="Hello with UUID session!"))
        history.add_message(AIMessage(content="Hello back!"))

        # Retrieve messages - this should work without syntax errors
        messages = history.messages
        assert len(messages) == 2
        assert messages[0].content == "Hello with UUID session!"
        assert messages[1].content == "Hello back!"

        # Test search functionality - this should work without syntax errors
        search_results = history.search_messages("UUID")
        assert len(search_results) == 1
        assert "UUID" in search_results[0]["content"]

        # Test length functionality - this should work without syntax errors
        assert len(history) == 2

        # Test clear functionality - this should work without syntax errors
        history.clear()
        assert len(history.messages) == 0

    finally:
        # Ensure cleanup even if test fails
        history.clear()


def test_timestamp_sorting(chat_history: RedisChatMessageHistory) -> None:
    """Test that messages are returned in correct timestamp order."""
    # Add messages with small delays to ensure different timestamps
    chat_history.add_message(HumanMessage(content="First message"))
    chat_history.add_message(AIMessage(content="Second message"))
    chat_history.add_message(HumanMessage(content="Third message"))
    chat_history.add_message(AIMessage(content="Fourth message"))

    # Retrieve messages and verify they're in chronological order
    messages = chat_history.messages
    assert len(messages) == 4

    # Check content order
    assert messages[0].content == "First message"
    assert messages[1].content == "Second message"
    assert messages[2].content == "Third message"
    assert messages[3].content == "Fourth message"

    # Check message types
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], HumanMessage)
    assert isinstance(messages[3], AIMessage)


def test_session_sensitive_clear(redis_url: str) -> None:
    """Test that clear() only clears messages for the current session."""
    session1_id = f"session1_{str(ULID())}"
    session2_id = f"session2_{str(ULID())}"

    history1 = RedisChatMessageHistory(session_id=session1_id, redis_url=redis_url)
    history2 = RedisChatMessageHistory(session_id=session2_id, redis_url=redis_url)

    try:
        # Add messages to both sessions
        history1.add_message(HumanMessage(content="Session 1 message 1"))
        history1.add_message(AIMessage(content="Session 1 message 2"))

        history2.add_message(HumanMessage(content="Session 2 message 1"))
        history2.add_message(AIMessage(content="Session 2 message 2"))

        # Verify both sessions have messages
        assert len(history1.messages) == 2
        assert len(history2.messages) == 2

        # Clear only session 1
        history1.clear()

        # Verify session 1 is empty but session 2 still has messages
        assert len(history1.messages) == 0
        assert len(history2.messages) == 2

        # Verify session 2 messages are intact
        session2_messages = history2.messages
        assert session2_messages[0].content == "Session 2 message 1"
        assert session2_messages[1].content == "Session 2 message 2"

    finally:
        # Clean up both sessions
        history1.clear()
        history2.clear()


def test_empty_session_id(redis_url: str) -> None:
    """Test behavior with empty session ID."""
    with pytest.raises(ValueError):
        RedisChatMessageHistory(session_id="", redis_url=redis_url)


def test_none_session_id(redis_url: str) -> None:
    """Test behavior with None session ID."""
    with pytest.raises(ValueError):
        RedisChatMessageHistory(session_id=None, redis_url=redis_url)  # type: ignore


def test_unicode_session_id(redis_url: str) -> None:
    """Test behavior with Unicode characters in session ID."""
    unicode_session_id = f"session_测试_🚀_{str(ULID())}"
    history = RedisChatMessageHistory(
        session_id=unicode_session_id, redis_url=redis_url
    )

    try:
        history.add_message(HumanMessage(content="Unicode session test"))
        messages = history.messages
        assert len(messages) == 1
        assert messages[0].content == "Unicode session test"
    finally:
        history.delete()


def test_empty_message_content(chat_history: RedisChatMessageHistory) -> None:
    """Test adding messages with empty content."""
    history = chat_history

    # Test empty string content
    history.add_message(HumanMessage(content=""))
    history.add_message(AIMessage(content=""))

    messages = history.messages
    assert len(messages) == 2
    assert messages[0].content == ""
    assert messages[1].content == ""


def test_very_large_message_content(chat_history: RedisChatMessageHistory) -> None:
    """Test adding messages with very large content."""
    large_content = "x" * 100000  # 100KB message
    chat_history.add_message(HumanMessage(content=large_content))

    messages = chat_history.messages
    assert len(messages) == 1
    assert messages[0].content == large_content


def test_unicode_message_content(chat_history: RedisChatMessageHistory) -> None:
    """Test messages with Unicode content including emojis."""
    unicode_content = "Hello 世界! 🌍🚀 Testing Unicode: αβγδε ñáéíóú"
    chat_history.add_message(HumanMessage(content=unicode_content))

    messages = chat_history.messages
    assert len(messages) == 1
    assert messages[0].content == unicode_content


def test_special_characters_in_content(chat_history: RedisChatMessageHistory) -> None:
    """Test messages with special characters that might break JSON or search."""
    special_content = 'Test with "quotes", {brackets}, [arrays], and \\ backslashes'
    chat_history.add_message(HumanMessage(content=special_content))
    messages = chat_history.messages
    assert len(messages) == 1
    assert messages[0].content == special_content


def test_message_with_additional_kwargs(chat_history: RedisChatMessageHistory) -> None:
    """Test messages with additional_kwargs are preserved."""
    message = HumanMessage(
        content="Test message",
        additional_kwargs={"custom_field": "custom_value", "number": 42},
    )
    chat_history.add_message(message)

    messages = chat_history.messages
    assert len(messages) == 1
    assert messages[0].content == "Test message"
    assert messages[0].additional_kwargs == {
        "custom_field": "custom_value",
        "number": 42,
    }


def test_search_case_insensitive(chat_history: RedisChatMessageHistory) -> None:
    """Test that search is case insensitive."""
    chat_history.add_message(HumanMessage(content="Hello World"))
    chat_history.add_message(HumanMessage(content="GOODBYE WORLD"))

    # Test different cases
    results = chat_history.search_messages("hello")
    assert len(results) == 1

    results = chat_history.search_messages("HELLO")
    assert len(results) == 1

    results = chat_history.search_messages("world")
    assert len(results) == 2

    results = chat_history.search_messages("WORLD")
    assert len(results) == 2


def test_search_with_limit_zero(chat_history: RedisChatMessageHistory) -> None:
    """Test search with limit=0."""
    chat_history.add_message(HumanMessage(content="Test message"))

    results = chat_history.search_messages("test", limit=0)
    assert len(results) == 0


def test_search_with_large_limit(chat_history: RedisChatMessageHistory) -> None:
    """Test search with very large limit."""
    for i in range(5):
        chat_history.add_message(HumanMessage(content=f"Test message {i}"))

    results = chat_history.search_messages("test", limit=1000)
    assert len(results) == 5


def test_invalid_redis_url() -> None:
    """Test behavior with invalid Redis URL."""
    with pytest.raises(Exception):  # Could be ConnectionError or similar
        history = RedisChatMessageHistory(
            session_id="test", redis_url="redis://invalid-host:6379"
        )
        # Try to use it to trigger connection
        history.add_message(HumanMessage(content="Test"))


def test_custom_key_prefix(redis_url: str) -> None:
    """Test custom key prefix functionality."""
    custom_prefix = "custom_chat:"
    session_id = f"test_{str(ULID())}"

    history = RedisChatMessageHistory(
        session_id=session_id, redis_url=redis_url, key_prefix=custom_prefix
    )

    try:
        history.add_message(HumanMessage(content="Test with custom prefix"))
        messages = history.messages
        assert len(messages) == 1
        assert messages[0].content == "Test with custom prefix"
    finally:
        history.delete()


def test_custom_index_name(redis_url: str) -> None:
    """Test custom index name functionality."""
    custom_index = "custom_chat_index"
    session_id = f"test_{str(ULID())}"

    history = RedisChatMessageHistory(
        session_id=session_id, redis_url=redis_url, index_name=custom_index
    )

    try:
        history.add_message(HumanMessage(content="Test with custom index"))
        messages = history.messages
        assert len(messages) == 1
        assert messages[0].content == "Test with custom index"

        # Verify the custom index was created
        redis_client = Redis.from_url(redis_url)
        index_info = redis_client.ft(custom_index).info()
        assert index_info["index_name"] == custom_index
    finally:
        history.delete()


def test_clear_empty_session(chat_history: RedisChatMessageHistory) -> None:
    """Test clearing an empty session doesn't cause errors."""
    # Should not raise any exceptions
    chat_history.clear()
    assert len(chat_history.messages) == 0


def test_clear_large_session(redis_url: str) -> None:
    """Test clearing a session with many messages."""
    session_id = f"large_clear_test_{str(ULID())}"
    history = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)

    try:
        # Add many messages
        for i in range(100):
            history.add_message(HumanMessage(content=f"Message {i}"))

        assert len(history.messages) == 100

        # Clear all messages
        history.clear()
        assert len(history.messages) == 0
    finally:
        history.delete()


def test_message_ordering_with_rapid_additions(redis_url: str) -> None:
    """Test message ordering when messages are added very rapidly."""
    session_id = f"rapid_test_{str(ULID())}"
    history = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)

    try:
        # Add messages very rapidly
        for i in range(20):
            history.add_message(HumanMessage(content=f"Rapid message {i}"))

        messages = history.messages
        assert len(messages) == 20

        # Verify ordering (should be chronological)
        for i, message in enumerate(messages):
            assert message.content == f"Rapid message {i}"
    finally:
        history.delete()


def test_ttl_edge_cases(redis_url: str) -> None:
    """Test TTL edge cases."""
    session_id = f"ttl_edge_test_{str(ULID())}"

    # Test TTL = 0 (should expire immediately)
    history_zero = RedisChatMessageHistory(
        session_id=f"{session_id}_zero", redis_url=redis_url, ttl=0
    )

    try:
        history_zero.add_message(HumanMessage(content="Should expire immediately"))
        # Message might already be expired
        time.sleep(0.1)
        messages = history_zero.messages
        # Could be 0 or 1 depending on timing
        assert len(messages) <= 1
    finally:
        history_zero.clear()

    # Test very large TTL
    history_large = RedisChatMessageHistory(
        session_id=f"{session_id}_large",
        redis_url=redis_url,
        ttl=2147483647,  # Max 32-bit int
    )

    try:
        history_large.add_message(HumanMessage(content="Long TTL message"))
        messages = history_large.messages
        assert len(messages) == 1
    finally:
        history_large.clear()


def test_id_property(chat_history: RedisChatMessageHistory) -> None:
    """Test the id property returns session_id."""
    assert chat_history.id == chat_history.session_id


def test_messages_property_empty(chat_history: RedisChatMessageHistory) -> None:
    """Test messages property when no messages exist."""
    messages = chat_history.messages
    assert isinstance(messages, list)
    assert len(messages) == 0


def test_len_with_cleared_session(chat_history: RedisChatMessageHistory) -> None:
    """Test __len__ after clearing session."""
    chat_history.add_message(HumanMessage(content="Test"))
    assert len(chat_history) == 1

    chat_history.clear()
    assert len(chat_history) == 0


def test_index_recreation_after_deletion(redis_url: str) -> None:
    """Test that index can be recreated if manually deleted."""
    session_id = f"index_recreation_test_{str(ULID())}"
    history = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)

    try:
        # Add a message to ensure index is working
        history.add_message(HumanMessage(content="Test message"))
        assert len(history.messages) == 1

        # Manually delete the index
        redis_client = Redis.from_url(redis_url)
        try:
            redis_client.ft(history.index_name).dropindex()
        except Exception:
            pass  # Index might not exist

        # Create new instance - should recreate index
        history2 = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)

        # Should be able to add messages again
        history2.add_message(HumanMessage(content="After recreation"))
        messages = history2.messages
        # Note: Original message might be gone since index was dropped
        assert len(messages) >= 1
        assert any("After recreation" in msg.content for msg in messages)

    finally:
        try:
            history.delete()
            history2.delete()
        except Exception:
            pass


def test_multiple_instances_same_session(redis_url: str) -> None:
    """Test multiple instances accessing the same session."""
    session_id = f"multi_instance_test_{str(ULID())}"

    history1 = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)
    history2 = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)

    try:
        # Add message with first instance
        history1.add_message(HumanMessage(content="From instance 1"))

        # Read with second instance
        messages = history2.messages
        assert len(messages) == 1
        assert messages[0].content == "From instance 1"

        # Add message with second instance
        history2.add_message(AIMessage(content="From instance 2"))

        # Read with first instance
        messages = history1.messages
        assert len(messages) == 2
        assert messages[0].content == "From instance 1"
        assert messages[1].content == "From instance 2"

    finally:
        history1.clear()


def test_search_with_empty_query(chat_history: RedisChatMessageHistory) -> None:
    """Test search with empty query string."""
    chat_history.add_message(HumanMessage(content="Test message"))

    # Empty query should return no results or all results depending on implementation
    results = chat_history.search_messages("")
    # This behavior might vary - just ensure it doesn't crash
    assert isinstance(results, list)
    assert results == []


def test_message_with_none_content() -> None:
    """Test that messages with None content are handled properly."""
    # This should raise an error during message creation, not in our code
    with pytest.raises((TypeError, ValueError)):
        HumanMessage(content=None)  # type: ignore


def test_redis_connection_recovery(redis_url: str) -> None:
    """Test behavior when Redis connection is temporarily lost."""
    session_id = f"connection_test_{str(ULID())}"
    history = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)

    try:
        # Add a message normally
        history.add_message(HumanMessage(content="Before connection issue"))
        assert len(history.messages) == 1

        # Simulate connection issue by closing the connection
        # Note: This is a basic test - real connection recovery would be more complex
        history.redis_client.connection_pool.disconnect()

        # Try to add another message - should work due to connection pooling
        history.add_message(HumanMessage(content="After connection issue"))
        messages = history.messages
        assert len(messages) == 2

    finally:
        history.delete()


def test_very_long_key_prefix(redis_url: str) -> None:
    """Test with very long key prefix."""
    long_prefix = "very_long_prefix_" * 10 + ":"  # ~170 characters
    session_id = f"test_{str(ULID())}"

    history = RedisChatMessageHistory(
        session_id=session_id, redis_url=redis_url, key_prefix=long_prefix
    )

    try:
        history.add_message(HumanMessage(content="Test with long prefix"))
        messages = history.messages
        assert len(messages) == 1
        assert messages[0].content == "Test with long prefix"
    finally:
        history.delete()


def test_special_characters_in_key_prefix(redis_url: str) -> None:
    """Test with special characters in key prefix."""
    special_prefix = "test-prefix_with.special@chars:"
    session_id = f"test_{str(ULID())}"

    history = RedisChatMessageHistory(
        session_id=session_id, redis_url=redis_url, key_prefix=special_prefix
    )

    try:
        history.add_message(HumanMessage(content="Test with special prefix"))
        messages = history.messages
        assert len(messages) == 1
        assert messages[0].content == "Test with special prefix"
    finally:
        history.delete()


def test_negative_ttl(redis_url: str) -> None:
    """Test behavior with negative TTL."""
    session_id = f"negative_ttl_test_{str(ULID())}"

    # Negative TTL should either be rejected or treated as no TTL
    try:
        history = RedisChatMessageHistory(
            session_id=session_id, redis_url=redis_url, ttl=-1
        )

        history.add_message(HumanMessage(content="Negative TTL test"))
        messages = history.messages
        # Should either work (treating -1 as no TTL) or fail gracefully
        assert isinstance(messages, list)

    except (ValueError, TypeError):
        # It's acceptable to reject negative TTL
        pass
    finally:
        history.delete()


def test_pagination_in_clear_method(redis_url: str) -> None:
    """Test that the pagination in clear() method works correctly."""
    session_id = f"pagination_clear_test_{str(ULID())}"
    history = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)

    try:
        # Add more messages than the page size (50)
        for i in range(75):
            history.add_message(HumanMessage(content=f"Message {i}"))

        assert len(history.messages) == 75

        # Clear should handle pagination correctly
        history.clear()
        assert len(history.messages) == 0

    finally:
        history.delete()


def test_json_serialization_edge_cases(chat_history: RedisChatMessageHistory) -> None:
    """Test edge cases in JSON serialization."""
    # Test with content that might cause JSON issues
    problematic_content = """{"nested": "json", "array": [1, 2, 3], "null": null}"""

    chat_history.add_message(HumanMessage(content=problematic_content))

    messages = chat_history.messages
    assert len(messages) == 1
    assert messages[0].content == problematic_content


def test_timestamp_precision(redis_url: str) -> None:
    """Test timestamp precision and uniqueness."""
    session_id = f"timestamp_test_{str(ULID())}"
    history = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)

    try:
        # Add messages in very quick succession
        import time

        start_time = time.time()

        for i in range(10):
            history.add_message(HumanMessage(content=f"Timestamp test {i}"))

        end_time = time.time()

        messages = history.messages
        assert len(messages) == 10

        # Verify all messages are in order
        for i, message in enumerate(messages):
            assert message.content == f"Timestamp test {i}"

        # Test took less than a second but all messages should be ordered
        assert end_time - start_time < 1.0

    finally:
        history.delete()


def test_key_prefix_isolation_with_overwrite(redis_url: str) -> None:
    """Test that different key_prefix values work correctly with overwrite_index=True.

    This is a regression test for issue #74 where custom key_prefix parameters
    would cause message retrieval conflicts due to shared search index names.
    """
    redis_client = Redis.from_url(redis_url)
    session_id = f"isolation_test_{str(ULID())}"

    # First create a history with default prefix to establish the search index
    history_default = RedisChatMessageHistory(
        session_id=f"{session_id}_default", redis_url=redis_url
    )

    try:
        # Add message to default history first (creates search index with prefix)
        history_default.add_message(HumanMessage(content="Default message"))
        assert len(history_default.messages) == 1

        # Now create history with custom prefix using overwrite_index=True
        # This should work because it overwrites the index with the new prefix
        history_custom = RedisChatMessageHistory(
            session_id=f"{session_id}_custom",
            redis_url=redis_url,
            key_prefix="custom_app:",
            overwrite_index=True,
        )

        # Add messages to custom prefix history
        history_custom.add_message(HumanMessage(content="Custom message 1"))
        history_custom.add_message(AIMessage(content="Custom message 2"))

        # THE CORE TEST: With overwrite_index=True, messages should be retrievable
        custom_messages = history_custom.messages
        assert len(custom_messages) == 2, (
            f"Expected 2 messages with custom prefix and overwrite_index=True, "
            f"got {len(custom_messages)}"
        )

        # Verify correct content retrieval
        assert custom_messages[0].content == "Custom message 1"
        assert custom_messages[1].content == "Custom message 2"

        # Verify that messages are stored in Redis with correct prefix
        custom_keys = list(redis_client.scan_iter(match="custom_app:*"))
        assert len(custom_keys) >= 2, "Messages should be stored with custom prefix"

        # Note: After overwriting the index, the default history won't work
        # because the index now uses "custom_app:" prefix instead of "chat:"
        # This is expected behavior with overwrite_index=True

    finally:
        # Clean up
        for history in [history_default, history_custom]:
            try:
                history.delete()
            except Exception:
                pass  # Ignore cleanup errors


def test_key_prefix_conflict_warning(redis_url: str, caplog: pytest.LogCaptureFixture) -> None:
    """Test that prefix conflicts generate appropriate warnings.

    This test validates that when overwrite_index=False (default) and a prefix
    conflict occurs, a clear warning is logged to help users understand the issue.
    """
    import logging

    session_id = f"warning_test_{str(ULID())}"

    # First create a history with default prefix
    history_default = RedisChatMessageHistory(
        session_id=f"{session_id}_default", redis_url=redis_url
    )

    try:
        history_default.add_message(HumanMessage(content="Default message"))

        # Clear any existing log records
        caplog.clear()

        # Now create history with custom prefix - this should trigger a warning
        with caplog.at_level(logging.WARNING):
            history_custom = RedisChatMessageHistory(
                session_id=f"{session_id}_custom",
                redis_url=redis_url,
                key_prefix="custom_app:",
                overwrite_index=False,  # Explicit default
            )

        # Check that warning was logged
        warning_logs = [
            record for record in caplog.records if record.levelname == "WARNING"
        ]
        assert len(warning_logs) > 0, "Expected a warning about prefix conflict"

        warning_message = warning_logs[0].message
        assert "already exists with different key prefix" in warning_message
        assert "custom_app:" in warning_message
        assert "overwrite_index=True" in warning_message

        # The custom prefix history should still be created but may not work correctly
        # (this demonstrates the problematic behavior that the warning alerts about)
        history_custom.add_message(HumanMessage(content="Custom message"))

        # Due to the prefix conflict, this might return 0 messages
        # The warning helps users understand why
        custom_messages = history_custom.messages
        # Note: We expect 0 messages due to prefix conflict - this demonstrates
        # the problem that the warning alerts users about
        assert len(custom_messages) == 0, "Expected prefix conflict to cause failure"

    finally:
        # Clean up
        cleanup_histories = [history_default]
        try:
            cleanup_histories.append(history_custom)
        except NameError:
            pass  # history_custom wasn't created due to error

        for history in cleanup_histories:
            try:
                history.delete()
            except Exception:
                pass  # Ignore cleanup errors
