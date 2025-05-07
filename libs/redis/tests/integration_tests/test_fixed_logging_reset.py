"""Test for the fix for Issue #62: Logs reset to INFO when importing
RedisChatMessageHistory."""

import io
import logging
import os

from langchain_core.messages import HumanMessage


def test_fixed_logging_reset() -> None:
    """Test that verifies our fix for the Redis push_response logger issue."""
    # Reset logging config
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set up a root logger with debug level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Create a test logger
    test_logger = logging.getLogger("test_app_logger")
    test_logger.setLevel(logging.DEBUG)
    test_logger.propagate = True

    # Create a stream handler to capture output
    console_output = io.StringIO()
    handler = logging.StreamHandler(console_output)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Log a debug message before imports
    test_logger.debug("Debug message before imports")

    # Check the level
    before_level = root_logger.level
    test_logger_before_level = test_logger.level

    # Import RedisChatMessageHistory
    from langchain_redis import RedisChatMessageHistory

    # Check the level after import
    after_import_level = root_logger.level
    test_logger_after_import_level = test_logger.level

    # Log a debug message after import
    test_logger.debug("Debug message after imports")

    # Check if push_response logger exists
    has_push_response_after_import = "push_response" in logging.root.manager.loggerDict

    # Use RedisChatMessageHistory with our fix
    try:
        # Get Redis URL from environment
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")

        # Create a chat history instance
        history = RedisChatMessageHistory(
            session_id="test_session",
            redis_url=redis_url,
            ttl=60,
        )

        # Add a message
        history.add_message(HumanMessage(content="Hello"))
    except Exception:
        pass

    # Check if push_response logger exists after using RedisChatMessageHistory
    has_push_response_after = "push_response" in logging.root.manager.loggerDict

    # Log another debug message
    test_logger.debug("Debug message after using RedisChatMessageHistory")

    # Get the captured output
    output = console_output.getvalue()

    # Verify all our debug messages were captured (meaning debug level wasn't affected)
    assert "Debug message before imports" in output
    assert "Debug message after imports" in output
    assert "Debug message after using RedisChatMessageHistory" in output

    # The core test - verify that the push_response logger was not created
    assert (
        not has_push_response_after_import
    ), "push_response logger was created during import"
    assert (
        not has_push_response_after
    ), "push_response logger was created when using RedisChatMessageHistory"

    # Verify that our root and test loggers' levels were not changed
    assert before_level == after_import_level, "Root logger level changed after import"
    assert (
        test_logger_before_level == test_logger_after_import_level
    ), "Test logger level changed after import"

    # Now verify what happens if we try using Redis PubSub with our fix
    import redis

    from langchain_redis.chat_message_history import _noop_push_handler

    client = redis.Redis.from_url(redis_url)
    client.pubsub(push_handler_func=_noop_push_handler)  # Use our custom handler

    # Check if push_response logger exists after PubSub
    has_push_response_after_pubsub = "push_response" in logging.root.manager.loggerDict
    assert (
        not has_push_response_after_pubsub
    ), "push_response logger was created by PubSub"

    # Directly create a PubSub without our fix
    redis.Redis.from_url(redis_url).pubsub(push_handler_func=None)

    # Now the push_response logger should be created
    has_push_response_after_pubsub_no_handler = (
        "push_response" in logging.root.manager.loggerDict
    )
    assert (
        has_push_response_after_pubsub_no_handler
    ), "push_response logger was not created by PubSub with no handler"

    # Verify that debug messages still work after push_response logger was created
    test_logger.debug("Debug message after push_response logger created")
    output_final = console_output.getvalue()
    assert "Debug message after push_response logger created" in output_final
